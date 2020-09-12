
#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"

#include "../libpsc/psc_heating/psc_heating_impl.hxx"
#include "heating_spot_foil.hxx"
#include "inject_impl.hxx"

#include "../libpsc/psc_output_fields/fields_item_fields.hxx"

//#define DIM_3D

// ======================================================================
// Particle kinds
//
// Particle kinds can be used to define different species, or different
// populations of the same species
//
// Here, we only enumerate the types, the actual information gets set up later.
// The last kind (MY_ELECTRON) will be used as "neutralizing kind", ie, in the
// initial setup, the code will add as many electrons as there are ions in a
// cell, at the same position, to ensure the initial plasma is neutral
// (so that Gauss's Law is satisfied).
enum
{
  MY_ELECTRON_HE,
  MY_ELECTRON,
  MY_ION,
  N_MY_KINDS,
};

// ======================================================================
// PscFlatfoilParams

struct PscFlatfoilParams
{
  double BB;
  double Zi;
  double mass_ratio;
  double lambda0;
  double target_n;  // target density
  double target_Te; // target electron temperature
  double target_Ti; // target ion_temperatore
  double target_Te_heat;
  double target_Ti_heat;
  double target_Te_HE_heat;

  double background_n;
  double background_Te;
  double background_Ti;
  double electron_HE_ratio;

  int inject_interval;

  int heating_begin;
  int heating_end;
  int heating_interval;

  // The following parameters are calculated from the above / and other
  // information

  double d_i;
};

// ======================================================================
// Global parameters
//
// I'm not a big fan of global parameters, but they're only for
// this particular case and they help make things simpler.

// An "anonymous namespace" makes these variables visible in this source file
// only
namespace
{

// Parameters specific to this case. They don't really need to be collected in a
// struct, but maybe it's nice that they are

PscFlatfoilParams g;

std::string read_checkpoint_filename;

// This is a set of generic PSC params (see include/psc.hxx),
// like number of steps to run, etc, which also should be set by the case
PscParams psc_params;

} // namespace

// ======================================================================
// InjectFoil
//
// This class describes the actual foil, specifically, where the foil
// is located and what its plasma parameters are

struct InjectFoilParams
{
  double xl, xh;
  double yl, yh;
  double zl, zh;
  double n;
  double Te, Ti;
};

class InjectFoil : InjectFoilParams
{
public:
  InjectFoil() = default;

  InjectFoil(const InjectFoilParams& params) : InjectFoilParams(params) {}

  bool is_inside(double crd[3])
  {
    return (crd[0] >= xl && crd[0] <= xh && crd[1] >= yl && crd[1] <= yh &&
            crd[2] >= zl && crd[2] <= zh);
  }

  void init_npt(int pop, double crd[3], psc_particle_npt& npt)
  {
    if (!is_inside(crd)) {
      npt.n = 0;
      return;
    }

    switch (pop) {
      case MY_ION:
        npt.n = n;
        npt.T[0] = Ti;
        npt.T[1] = Ti;
        npt.T[2] = Ti;
        break;
      case MY_ELECTRON_HE:
        npt.n = g.electron_HE_ratio * n;
        npt.T[0] = g.target_Te_HE_heat;
        npt.T[1] = g.target_Te_HE_heat;
        npt.T[2] = g.target_Te_HE_heat;
        break;
      case MY_ELECTRON:
        npt.n = (1 - g.electron_HE_ratio) * n;
        npt.T[0] = Te;
        npt.T[1] = Te;
        npt.T[2] = Te;
        break;
      default: assert(0);
    }
  }
};

// ======================================================================
// PSC configuration
//
// This sets up compile-time configuration for the code, in particular
// what data structures and algorithms to use
//
// EDIT to change order / floating point type / cuda / 2d/3d

#ifdef DIM_3D
using Dim = dim_xyz;
#else
using Dim = dim_yz;
#endif

#if 1
#ifdef USE_CUDA
using PscConfig = PscConfig1vbecCuda<Dim>;
#else
using PscConfig = PscConfig1vbecSingle<Dim>;
#endif

#else

#include "particle_with_id.h"

using PscConfig =
  PscConfig_<Dim, MparticlesSimple<ParticleWithId<float>>, MfieldsStateSingle,
             MfieldsSingle, PscConfigPushParticles1vbec>;

#endif

// ----------------------------------------------------------------------

using MfieldsState = PscConfig::MfieldsState;
using Mparticles = PscConfig::Mparticles;
using Balance = PscConfig::Balance;
using Collision = PscConfig::Collision;
using Checks = PscConfig::Checks;
using Marder = PscConfig::Marder;
using OutputParticles = PscConfig::OutputParticles;
using Inject = typename InjectSelector<Mparticles, InjectFoil, Dim>::Inject;
using Heating = typename HeatingSelector<Mparticles>::Heating;

// ======================================================================
// setupParameters

void setupParameters()
{
  // -- set some generic PSC parameters
  psc_params.nmax = 10000001; // 5001;
  psc_params.cfl = 0.75;
  psc_params.write_checkpoint_every_step = 100;
  psc_params.stats_every = 1;

  // -- start from checkpoint:
  //
  // Uncomment when wanting to start from a checkpoint, ie.,
  // instead of setting up grid, particles and state fields here,
  // they'll be read from a file
  // FIXME: This parameter would be a good candidate to be provided
  // on the command line, rather than requiring recompilation when change.

  // read_checkpoint_filename = "checkpoint_500.bp";

  // -- Set some parameters specific to this case
  g.BB = 0.;
  g.Zi = 1.;
  g.mass_ratio = 64.;
  g.lambda0 = 20.;

  g.target_n = 2.5;
  g.target_Te = 0.001;
  g.target_Ti = 0.001;

  g.electron_HE_ratio = 0.01;

  g.target_Te_heat = 0.04;
  g.target_Ti_heat = 0.0;
  g.target_Te_HE_heat = 0.4;

  g.background_n = .002;
  g.background_Te = .001;
  g.background_Ti = .001;
}

// ======================================================================
// setupGrid
//
// This helper function is responsible for setting up the "Grid",
// which is really more than just the domain and its decomposition, it
// also encompasses PC normalization parameters, information about the
// particle kinds, etc.

Grid_t* setupGrid()
{
  // --- setup domain
#ifdef DIM_3D
  Grid_t::Real3 LL = {80., 80., 3. * 80.}; // domain size (in d_e)
  Int3 gdims = {160, 160, 3 * 160};        // global number of grid points
  Int3 np = {5, 5, 3 * 5};                 // division into patches
#else
  //Grid_t::Real3 LL = {1., 800., 3. * 800.}; // domain size (in d_e)
  //Int3 gdims = {1, 1600, 3 * 1600};         // global number of grid points
  //Int3 np = {1, 50, 3 * 50};                // division into patches
    Grid_t::Real3 LL = {1., 32., 3200.}; // domain size (in d_e)
    Int3 gdims = {1, 32, 2*3200};        // global number of grid points
    Int3 np = {1, 1, 2*100};                 // division into patches
#endif

  Grid_t::Domain domain{gdims, LL, -.5 * LL, np};

  psc::grid::BC bc{{BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};

  // -- setup particle kinds
  // last population ("i") is neutralizing
  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ION] = {g.Zi, g.mass_ratio * g.Zi, "i"};
  kinds[MY_ELECTRON_HE] = {-1., 1., "he_e"};
  kinds[MY_ELECTRON] = {-1., 1., "e"};

  g.d_i = sqrt(kinds[MY_ION].m / kinds[MY_ION].q);

  mpi_printf(MPI_COMM_WORLD, "d_e = %g, d_i = %g\n", 1., g.d_i);
  mpi_printf(MPI_COMM_WORLD, "lambda_De (background) = %g\n",
             sqrt(g.background_Te));

  // -- setup normalization
  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = 100;

  double dt = psc_params.cfl * courant_length(domain);
  Grid_t::Normalization norm{norm_params};

  Int3 ibn = {2, 2, 2};
  if (Dim::InvarX::value) {
    ibn[0] = 0;
  }
  if (Dim::InvarY::value) {
    ibn[1] = 0;
  }
  if (Dim::InvarZ::value) {
    ibn[2] = 0;
  }

  return new Grid_t{domain, bc, kinds, norm, dt, -1, ibn};
}

// ======================================================================
// initializeParticles

void initializeParticles(SetupParticles<Mparticles>& setup_particles,
                         Balance& balance, Grid_t*& grid_ptr, Mparticles& mprts,
                         InjectFoil& inject_target)
{
  // -- set particle initial condition
  partitionAndSetupParticles(setup_particles, balance, grid_ptr, mprts,
                             [&](int kind, Double3 crd, psc_particle_npt& npt) {
                               switch (kind) {
                                 case MY_ION:
                                   npt.n = g.background_n;
                                   npt.T[0] = g.background_Ti;
                                   npt.T[1] = g.background_Ti;
                                   npt.T[2] = g.background_Ti;
                                   break;
                                 case MY_ELECTRON_HE:
                                   npt.n = 0.;
                                   npt.T[0] = g.background_Te;
                                   npt.T[1] = g.background_Te;
                                   npt.T[2] = g.background_Te;
                                   break;
                                 case MY_ELECTRON:
                                   npt.n = g.background_n;
                                   npt.T[0] = g.background_Te;
                                   npt.T[1] = g.background_Te;
                                   npt.T[2] = g.background_Te;
                                   break;
                                 default: assert(0);
                               }

                               if (inject_target.is_inside(crd)) {
                                 // replace values above by target values
                                 inject_target.init_npt(kind, crd, npt);
                               }
                             });
}

// ======================================================================
// initializeFields

void initializeFields(MfieldsState& mflds)
{
  setupFields(mflds, [&](int m, double crd[3]) {
    switch (m) {
      case HY: return g.BB;
      default: return 0.;
    }
  });
}

// ======================================================================
// run
//
// This is basically the main function of this run,
// which sets up everything and then uses PscIntegrator to run the
// simulation

void run()
{
  mpi_printf(MPI_COMM_WORLD, "*** Setting up...\n");

  // ----------------------------------------------------------------------
  // setup various parameters first

  setupParameters();

  // ----------------------------------------------------------------------
  // Set up grid, state fields, particles

  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);
  if (!read_checkpoint_filename.empty()) {
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);
  }

  // ----------------------------------------------------------------------
  // Set up various objects needed to run this case

  // -- Balance
  psc_params.balance_interval = 500;
  Balance balance{psc_params.balance_interval, 3};

  // -- Sort
  psc_params.sort_interval = 10;

  // -- Collision
  int collision_interval = 10;
  double collision_nu =
    3.76 * std::pow(g.target_Te_heat, 2.) / g.Zi / g.lambda0;
  Collision collision{grid, collision_interval, collision_nu};

  // -- Checks
  ChecksParams checks_params{};
  checks_params.continuity_every_step = 0;
  checks_params.continuity_threshold = 1e-4;
  checks_params.continuity_verbose = true;
  checks_params.continuity_dump_always = false;
  checks_params.gauss_every_step = 100;
  checks_params.gauss_threshold = 1e-4;
  checks_params.gauss_verbose = true;
  checks_params.gauss_dump_always = false;
  Checks checks{grid, MPI_COMM_WORLD, checks_params};

  // -- Marder correction
  double marder_diffusion = 0.9;
  int marder_loop = 3;
  bool marder_dump = false;
  psc_params.marder_interval = 100;
  Marder marder(grid, marder_diffusion, marder_loop, marder_dump);

  // ----------------------------------------------------------------------
  // Set up output
  //
  // FIXME, this really is too complicated and not very flexible

  // -- output fields
  OutputFieldsParams outf_params{};
  outf_params.pfield_interval = 500;
  outf_params.tfield_interval = 500;
  outf_params.tfield_average_every = 50;
  outf_params.tfield_moments_average_every = 50;
  OutputFields outf{grid, outf_params};

  // -- output particles
  OutputParticlesParams outp_params{};
  outp_params.every_step = -400;
  outp_params.data_dir = ".";
  outp_params.basename = "prt";
  OutputParticles outp{grid, outp_params};

  int oute_interval = -100;
  DiagEnergies oute{grid.comm(), oute_interval};

  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  // ----------------------------------------------------------------------
  // Set up objects specific to the flatfoil case

  // -- Heating
  HeatingSpotFoilParams heating_foil_params{};
  heating_foil_params.zl = -1. * g.d_i;
  heating_foil_params.zh = 1. * g.d_i;
  heating_foil_params.xc = 0. * g.d_i;
  heating_foil_params.yc = 20. * g.d_i;
  heating_foil_params.rH = 12. * g.d_i;
  heating_foil_params.T[MY_ELECTRON_HE] = g.target_Te_HE_heat;
  heating_foil_params.T[MY_ELECTRON] = g.target_Te_heat;
  heating_foil_params.T[MY_ION] = g.target_Ti_heat;
  heating_foil_params.Mi = grid.kinds[MY_ION].m;
  heating_foil_params.n_kinds = N_MY_KINDS;
  HeatingSpotFoil heating_spot{grid, heating_foil_params};

  g.heating_interval = 20;
  g.heating_begin = 0;
  g.heating_end = 10000000;
  auto& heating = *new Heating{grid, g.heating_interval, heating_spot};

  // -- Particle injection
  InjectFoilParams inject_foil_params;
  inject_foil_params.xl = -100000. * g.d_i;
  inject_foil_params.xh = 100000. * g.d_i;
  inject_foil_params.yl = -100000. * g.d_i;
  inject_foil_params.yh = 100000. * g.d_i;
  double target_zwidth = 1.;
  inject_foil_params.zl = -target_zwidth * g.d_i;
  inject_foil_params.zh = target_zwidth * g.d_i;
  inject_foil_params.n = 2.5;
  inject_foil_params.Te = .001;
  inject_foil_params.Ti = .001;
  InjectFoil inject_target{inject_foil_params};

  g.inject_interval = 20;
  int inject_tau = 40;

  SetupParticles<Mparticles> setup_particles(grid);
  setup_particles.fractional_n_particles_per_cell = true;
  setup_particles.neutralizing_population = MY_ION;

#ifdef USE_CUDA
  using MFields = HMFields;  
#else
  using MFields = MFieldsC;
#endif
  double fac = (g.inject_interval * grid.dt / inject_tau) / (1. + g.inject_interval * grid.dt / inject_tau);
  auto lf_init_npt = [&](int kind, Double3 pos, int p, Int3 idx,
                         psc_particle_npt& npt, MFields mf_n) {
    if (inject_target.is_inside(pos)) {
  
      if(kind == MY_ELECTRON_HE){
        inject_target.init_npt(MY_ELECTRON, pos, npt);
        npt.n -= mf_n[p](MY_ELECTRON, idx[0], idx[1], idx[2]);
        npt.n *= g.electron_HE_ratio;
      }else{
        inject_target.init_npt(kind, pos, npt);
        npt.n -= mf_n[p](kind, idx[0], idx[1], idx[2]);
        if(kind == MY_ELECTRON)
          npt.n *= (1 - g.electron_HE_ratio);
      }
      if (npt.n < 0) {
        npt.n = 0;
      }
      npt.n *= fac;
    }
  };

  Inject inject{grid, g.inject_interval, setup_particles, lf_init_npt};

  auto lf_inject = [&](const Grid_t& grid, Mparticles& mprts) {
    static int pr_inject, pr_heating;
    if (!pr_inject) {
      pr_inject = prof_register("inject", 1., 0, 0);
      pr_heating = prof_register("heating", 1., 0, 0);
    }

    auto comm = grid.comm();
    auto timestep = grid.timestep();

    if (g.inject_interval > 0 && timestep % g.inject_interval == 0) {
      mpi_printf(comm, "***** Performing injection...\n");
      prof_start(pr_inject);
      inject(mprts);
      prof_stop(pr_inject);
    }

    // only heating between heating_tb and heating_te
    if (timestep >= g.heating_begin && timestep < g.heating_end &&
        g.heating_interval > 0 && timestep % g.heating_interval == 0) {
      mpi_printf(comm, "***** Performing heating...\n");
      prof_start(pr_heating);
      heating(mprts);
      prof_stop(pr_heating);
    }
  };

  // ----------------------------------------------------------------------
  // setup initial conditions

  if (read_checkpoint_filename.empty()) {
    initializeParticles(setup_particles, balance, grid_ptr, mprts,
                        inject_target);
    initializeFields(mflds);
  }

  // ----------------------------------------------------------------------
  // hand off to PscIntegrator to run the simulation

  auto psc = makePscIntegrator<PscConfig>(psc_params, *grid_ptr, mflds, mprts,
                                          balance, collision, checks, marder,
                                          diagnostics, lf_inject);

  psc.integrate();
}

// ======================================================================
// main

int main(int argc, char** argv)
{
  psc_init(argc, argv);

  run();

  psc_finalize();
  return 0;
}
