#pragma once
#include <omp.h>
struct psc_particle_npt
{
  int kind;    ///< particle kind
  double n;    ///< density
  double p[3]; ///< momentum
  double T[3]; ///< temperature
  psc::particle::Tag tag;
};

// ======================================================================
// SetupParticles

template <typename MP>
struct SetupParticles
{
  using Mparticles = MP;
  using real_t = typename MP::real_t;

  SetupParticles(const Grid_t& grid, int n_populations = 0)
    : kinds_{grid.kinds}, norm_{grid.norm}, n_populations_{n_populations}
  {
    if (n_populations_ == 0) {
      n_populations_ = kinds_.size();
    }
  }

  // ----------------------------------------------------------------------
  // get_n_in_cell
  //
  // helper function for partition / particle setup

  int get_n_in_cell(const psc_particle_npt& npt)
  {
    if (fractional_n_particles_per_cell) {
      int n_prts = npt.n / norm_.cori;
      float rmndr = npt.n / norm_.cori - n_prts;
      float ran = random() / ((float)RAND_MAX + 1);
      if (ran < rmndr) {
        n_prts++;
      }
      return n_prts;
    }
    return npt.n / norm_.cori + .5;
  }

  // ----------------------------------------------------------------------
  // setupParticle

  psc::particle::Inject setupParticle(const psc_particle_npt& npt, Double3 pos,
                                      double wni)
  {
    double beta = norm_.beta;

    assert(npt.kind >= 0 && npt.kind < kinds_.size());
    double m = kinds_[npt.kind].m;

    float ran1, ran2, ran3, ran4, ran5, ran6;
    do {
      ran1 = random() / ((float)RAND_MAX + 1);
      ran2 = random() / ((float)RAND_MAX + 1);
      ran3 = random() / ((float)RAND_MAX + 1);
      ran4 = random() / ((float)RAND_MAX + 1);
      ran5 = random() / ((float)RAND_MAX + 1);
      ran6 = random() / ((float)RAND_MAX + 1);
    } while (ran1 >= 1.f || ran2 >= 1.f || ran3 >= 1.f || ran4 >= 1.f ||
             ran5 >= 1.f || ran6 >= 1.f);

    double pxi =
      npt.p[0] + sqrtf(-2.f * npt.T[0] / m * sqr(beta) * logf(1.0 - ran1)) *
                   cosf(2.f * M_PI * ran2);
    double pyi =
      npt.p[1] + sqrtf(-2.f * npt.T[1] / m * sqr(beta) * logf(1.0 - ran3)) *
                   cosf(2.f * M_PI * ran4);
    double pzi =
      npt.p[2] + sqrtf(-2.f * npt.T[2] / m * sqr(beta) * logf(1.0 - ran5)) *
                   cosf(2.f * M_PI * ran6);

    if (initial_momentum_gamma_correction) {
      double gam;
      if (sqr(pxi) + sqr(pyi) + sqr(pzi) < 1.) {
        gam = 1. / sqrt(1. - sqr(pxi) - sqr(pyi) - sqr(pzi));
        pxi *= gam;
        pyi *= gam;
        pzi *= gam;
      }
    }

    return psc::particle::Inject{pos, {pxi, pyi, pzi}, wni, npt.kind, npt.tag};
  }

  // ----------------------------------------------------------------------
  // setup_particles

  template <typename FUNC>
  void operator()(Mparticles& mprts, FUNC init_npt)
  {
    setupParticles(mprts, [&](int kind, Double3 pos, int p, Int3 idx,
                              psc_particle_npt& npt) { init_npt(kind, pos, npt); });
  }

  // ----------------------------------------------------------------------
  // setupParticles

  template <typename FUNC>
  void setupParticles(Mparticles& mprts, FUNC init_npt)
  {
    static int pr, pr_a, pr_b, pr_c, pr_d, pr_e;
    if(!pr){
      pr = prof_register("setup_injector", 1., 0, 0);
      pr_a = prof_register("setup_setupParticles", 1., 0, 0);
      pr_b = prof_register("setup_init_npt", 1., 0, 0);
      pr_c = prof_register("setup_ldims_inj", 1., 0, 0);
      pr_d = prof_register("setup_mprts_injector", 1., 0, 0);
      pr_e = prof_register("setup_grid", 1., 0, 0);
    }
    //#pragma omp parallel// shared(grid, inj, mprts, init_npt, pr)
    prof_start(pr_e);
    const auto& grid = mprts.grid();
    prof_stop(pr_e);
    // mprts.reserve_all(n_prts_by_patch); FIXME
    prof_start(pr_d);
    auto inj = mprts.injector();
    prof_stop(pr_d);


    //{
    //int thread_id = omp_get_thread_num();
    //std::cout <<"JOHN " << omp_get_num_threads() << std::endl;

    for (int p = 0; p < mprts.n_patches(); ++p) {
      prof_start(pr_c);
      auto ldims = grid.ldims;
      auto injector = inj[p];
      prof_stop(pr_c);
      //#pragma omp for
      for (int jz = 0; jz < ldims[2]; jz++) {
        for (int jy = 0; jy < ldims[1]; jy++) {
          for (int jx = 0; jx < ldims[0]; jx++) {
            Double3 pos = {grid.patches[p].x_cc(jx), grid.patches[p].y_cc(jy),
                           grid.patches[p].z_cc(jz)};
            // FIXME, the issue really is that (2nd order) particle pushers
            // don't handle the invariant dim right
            if (grid.isInvar(0) == 1)
              pos[0] = grid.patches[p].x_nc(jx);
            if (grid.isInvar(1) == 1)
              pos[1] = grid.patches[p].y_nc(jy);
            if (grid.isInvar(2) == 1)
              pos[2] = grid.patches[p].z_nc(jz);

            int n_q_in_cell = 0;
            for (int pop = 0; pop < n_populations_; pop++) {
              psc_particle_npt npt{};
              if (pop < kinds_.size()) {
                npt.kind = pop;
              }
              prof_start(pr_b);
              init_npt(pop, pos, p, {jx, jy, jz}, npt);
              prof_stop(pr_b);
              int n_in_cell;
              if (pop != neutralizing_population) {
                n_in_cell = get_n_in_cell(npt);
                n_q_in_cell += kinds_[npt.kind].q * n_in_cell;
              } else {
                // FIXME, should handle the case where not the last population
                // is neutralizing
                assert(neutralizing_population == n_populations_ - 1);
                n_in_cell = -n_q_in_cell / kinds_[npt.kind].q;
              }
              for (int cnt = 0; cnt < n_in_cell; cnt++) {
                real_t wni;
                if (fractional_n_particles_per_cell) {
                  wni = 1.;
                } else {
                  wni = npt.n / (n_in_cell * norm_.cori);
                }
                prof_start(pr_a);
                auto prt = setupParticle(npt, pos, wni);
                prof_stop(pr_a);
                //if (thread_id == 0) prof_start(pr);
                prof_start(pr);
                injector(prt);
                prof_stop(pr);
                //if (thread_id == 0) prof_stop(pr);
              }
            }
          }
        }
      }
    }
   
   //}//omp parallel
  }

  // ----------------------------------------------------------------------
  // partition

  template <typename FUNC>
  std::vector<uint> partition(const Grid_t& grid, FUNC init_npt)
  {
    std::vector<uint> n_prts_by_patch(grid.n_patches());

    for (int p = 0; p < grid.n_patches(); ++p) {
      auto ilo = Int3{}, ihi = grid.ldims;

      for (int jz = ilo[2]; jz < ihi[2]; jz++) {
        for (int jy = ilo[1]; jy < ihi[1]; jy++) {
          for (int jx = ilo[0]; jx < ihi[0]; jx++) {
            Double3 pos = {grid.patches[p].x_cc(jx), grid.patches[p].y_cc(jy),
                           grid.patches[p].z_cc(jz)};
            // FIXME, the issue really is that (2nd order) particle pushers
            // don't handle the invariant dim right
            if (grid.isInvar(0) == 1)
              pos[0] = grid.patches[p].x_nc(jx);
            if (grid.isInvar(1) == 1)
              pos[1] = grid.patches[p].y_nc(jy);
            if (grid.isInvar(2) == 1)
              pos[2] = grid.patches[p].z_nc(jz);

            int n_q_in_cell = 0;
            for (int pop = 0; pop < n_populations_; pop++) {
              psc_particle_npt npt{};
              if (pop < kinds_.size()) {
                npt.kind = pop;
              };
              init_npt(pop, pos, npt);

              int n_in_cell;
              if (pop != neutralizing_population) {
                n_in_cell = get_n_in_cell(npt);
                n_q_in_cell += kinds_[npt.kind].q * n_in_cell;
              } else {
                // FIXME, should handle the case where not the last population
                // is neutralizing
                assert(neutralizing_population == n_populations_ - 1);
                n_in_cell = -n_q_in_cell / kinds_[npt.kind].q;
              }
              n_prts_by_patch[p] += n_in_cell;
            }
          }
        }
      }
    }

    return n_prts_by_patch;
  }

  // the initial number of particles in a cell for this population will be st so
  // that it achieves neutrality
  int neutralizing_population = {-1};
  bool fractional_n_particles_per_cell = {false};
  bool initial_momentum_gamma_correction = {false};

private:
  const Grid_t::Kinds kinds_;
  const Grid_t::Normalization norm_;
  int n_populations_;
};
