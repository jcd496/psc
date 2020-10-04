
// ======================================================================
// PushParticlesVb

template <typename C>
struct PushParticlesVb
{
  static const int MAX_NR_KINDS = 10;

  using Mparticles = typename C::Mparticles;
  using MfieldsState = typename C::MfieldsState;
  using AdvanceParticle_t = typename C::AdvanceParticle_t;
  using InterpolateEM_t = typename C::InterpolateEM_t;
  using Current = typename C::Current_t;
  using Dim = typename C::Dim;
  using real_t = typename Mparticles::real_t;
  using Real3 = Vec3<real_t>;

  using checks_order = checks_order_1st;

  // ----------------------------------------------------------------------
  // push_mprts

  static void push_mprts(Mparticles& mprts, MfieldsState& mflds)
  {
    const auto& grid = mprts.grid();
    PI<real_t> pi(grid);
    Real3 dxi = Real3{1., 1., 1.} / Real3(grid.domain.dx);
    real_t dq_kind[MAX_NR_KINDS];
    auto& kinds = grid.kinds;
    assert(kinds.size() <= MAX_NR_KINDS);
    for (int k = 0; k < kinds.size(); k++) {
      dq_kind[k] = .5f * grid.norm.eta * grid.dt * kinds[k].q / kinds[k].m;
    }
    InterpolateEM_t ip;
    AdvanceParticle_t advance(grid.dt);
    Current current(grid);

    auto accessor = mprts.accessor_();
    for (int p = 0; p < mflds.n_patches(); p++) {
      auto flds = mflds[p];
      auto prts = accessor[p];
      typename InterpolateEM_t::fields_t EM(flds);
      typename Current::fields_t J(flds);

      flds.zero(JXI, JXI + 3);

      for (auto prt : prts) {
        Real3& x = prt.x();

        real_t xm[3];
        for (int d = 0; d < 3; d++) {
          xm[d] = x[d] * dxi[d];
        }
        ip.set_coeffs(xm);

        // FIELD INTERPOLATION
        Real3 E = {ip.ex(EM), ip.ey(EM), ip.ez(EM)};
        Real3 H = {ip.hx(EM), ip.hy(EM), ip.hz(EM)};

        // x^(n+0.5), p^n -> x^(n+0.5), p^(n+1.0)
        real_t dq = dq_kind[prt.kind()];
        advance.push_p(prt.u(), E, H, dq);

        // x^(n+0.5), p^(n+1.0) -> x^(n+1.5), p^(n+1.0)
        auto v = advance.calc_v(prt.u());
        advance.push_x(x, v);

        int lf[3];
        real_t of[3], xp[3];
        pi.find_idx_off_pos_1st_rel(x, lf, of, xp, real_t(0.));

        // CURRENT DENSITY BETWEEN (n+.5)*dt and (n+1.5)*dt
        int lg[3];
        if (!Dim::InvarX::value) {
          lg[0] = ip.cx.g.l;
        }
        if (!Dim::InvarY::value) {
          lg[1] = ip.cy.g.l;
        }
        if (!Dim::InvarZ::value) {
          lg[2] = ip.cz.g.l;
        }
        current.calc_j(J, xm, xp, lf, lg, prt.qni_wni(), v);
      }
    }
  }

  // ----------------------------------------------------------------------
  // stagger_mprts_patch

  static void stagger_mprts_patch(Mparticles& mprts, MfieldsState& mflds)
  {
    const auto& grid = mprts.grid();
    Real3 dxi = Real3{1., 1., 1.} / Real3(grid.domain.dx);
    real_t dq_kind[MAX_NR_KINDS];
    auto& kinds = grid.kinds;
    assert(kinds.size() <= MAX_NR_KINDS);
    for (int k = 0; k < kinds.size(); k++) {
      dq_kind[k] = .5f * grid.eta * grid.dt * kinds[k].q / kinds[k].m;
    }
    InterpolateEM_t ip;
    AdvanceParticle_t advance(grid.dt);

    auto accessor = mprts.accessor_();
    for (int p = 0; p < mflds.n_patches(); p++) {
      auto flds = mflds[p];
      auto prts = accessor[p];
      typename InterpolateEM_t::fields_t EM(flds);

      for (auto prt : prts) {
        // field interpolation
        real_t* x = prt.x;

        real_t xm[3];
        for (int d = 0; d < 3; d++) {
          xm[d] = x[d] * dxi[d];
        }

        // FIELD INTERPOLATION

        ip.set_coeffs(xm);
        // FIXME, we're not using EM instead flds_em
        real_t E[3] = {ip.ex(EM), ip.ey(EM), ip.ez(EM)};
        real_t H[3] = {ip.hx(EM), ip.hy(EM), ip.hz(EM)};

        // x^(n+1/2), p^{n+1/2} -> x^(n+1/2), p^{n}
        int kind = prt->kind();
        real_t dq = dq_kind[kind];
        advance.push_p(&prt->pxi, E, H, -.5f * dq);
      }
    }
  }
};
