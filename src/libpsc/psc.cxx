
#include "psc.h"
#include "psc_method.h"
#include "psc_diag.h"
#include "psc_output_fields_collection.h"
#include "psc_output_particles.h"
#include "psc_fields_as_c.h"
#include "fields.hxx"
#include "setup_fields.hxx"

#include <mrc_common.h>
#include <mrc_params.h>
#include <mrc_io.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#include <array>

using Mfields_t = MfieldsC;
using Fields = Fields3d<Mfields_t::fields_t>;

struct psc *ppsc;

#define VAR(x) (void *)(offsetof(struct psc, x))
#define VAR_(x, n) (void *)(offsetof(struct psc, x) + n*sizeof(int))

static mrc_param_select bnd_fld_descr[3];
static mrc_param_select bnd_prt_descr[4];

static struct select_init {
  select_init() {
    bnd_fld_descr[0].str = "open";
    bnd_fld_descr[0].val = BND_FLD_OPEN;
    bnd_fld_descr[1].str = "periodic";
    bnd_fld_descr[1].val = BND_FLD_PERIODIC;
    bnd_fld_descr[2].str = "conducting_wall";
    bnd_fld_descr[2].val = BND_FLD_CONDUCTING_WALL;

    bnd_prt_descr[0].str = "reflecting";
    bnd_prt_descr[0].val = BND_PRT_REFLECTING;
    bnd_prt_descr[1].str = "periodic";
    bnd_prt_descr[1].val = BND_PRT_PERIODIC;
    bnd_prt_descr[2].str = "absorbing";
    bnd_prt_descr[2].val = BND_PRT_ABSORBING;
    bnd_prt_descr[3].str = "open";
    bnd_prt_descr[3].val = BND_PRT_OPEN;
  }
} select_initializer;

static struct param psc_descr[] = {
  // psc_params
  { "qq"            , VAR(norm_params.qq)              , PARAM_DOUBLE(1.6021e-19)   },
  { "mm"            , VAR(norm_params.mm)              , PARAM_DOUBLE(9.1091e-31)   },
  { "tt"            , VAR(norm_params.tt)              , PARAM_DOUBLE(1.6021e-16)   },
  { "cc"            , VAR(norm_params.cc)              , PARAM_DOUBLE(3.0e8)        },
  { "eps0"          , VAR(norm_params.eps0)            , PARAM_DOUBLE(8.8542e-12)   },
  { "lw"            , VAR(norm_params.lw)              , PARAM_DOUBLE(3.2e-6)       },
  { "i0"            , VAR(norm_params.i0)              , PARAM_DOUBLE(1e21)         },
  { "n0"            , VAR(norm_params.n0)              , PARAM_DOUBLE(1e26)         },
  { "e0"            , VAR(norm_params.e0)              , PARAM_DOUBLE(0.)           },
  { "nicell"        , VAR(prm.nicell)          , PARAM_INT(200)             },
  
  { "n_state_fields", VAR(n_state_fields)         , MRC_VAR_INT },

  { "method"                  , VAR(method)                  , MRC_VAR_OBJ(psc_method) },
  { "diag"                    , VAR(diag)                    , MRC_VAR_OBJ(psc_diag) },
  { "output_fields_collection", VAR(output_fields_collection), MRC_VAR_OBJ(psc_output_fields_collection) },
  { "output_particles"        , VAR(output_particles)        , MRC_VAR_OBJ(psc_output_particles) },

  {},
};

#undef VAR

// ----------------------------------------------------------------------
// psc_create

static void
_psc_create(struct psc *psc)
{
  assert(!ppsc);
  ppsc = psc;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srandom(rank);
  
  // default 9 state fields (J,E,B)
  psc->n_state_fields = NR_FIELDS;

  psc_output_fields_collection_set_psc(psc->output_fields_collection, psc);
}

// ======================================================================
// psc_setup

// ----------------------------------------------------------------------
// psc_setup_coeff

Grid_t::Normalization psc_setup_coeff(struct psc *psc)
{
  Grid_t::Normalization coeff;

  auto& prm = psc->norm_params;
  assert(psc->prm.nicell > 0);
  double wl = 2. * M_PI * prm.cc / prm.lw;
  double ld = prm.cc / wl;
  assert(ld == 1.); // FIXME, not sure why? (calculation of fnqs?)
  if (prm.e0 == 0.) {
    prm.e0 = sqrt(2.0 * prm.i0 / prm.eps0 / prm.cc) /
      prm.lw / 1.0e6;
  }
  coeff.b0 = prm.e0 / prm.cc;
  coeff.rho0 = prm.eps0 * wl * coeff.b0;
  coeff.phi0 = ld * prm.e0;
  coeff.a0 = prm.e0 / wl;

  double vos = prm.qq * prm.e0 / (prm.mm * wl);
  double vt = sqrt(prm.tt / prm.mm);
  double wp = sqrt(sqr(prm.qq) * prm.n0 / prm.eps0 / prm.mm);

  coeff.cori = 1. / psc->prm.nicell;
  double alpha_ = wp / wl;
  coeff.beta = vt / prm.cc;
  coeff.eta = vos / prm.cc;
  coeff.fnqs = sqr(alpha_) * coeff.cori / coeff.eta;
  return coeff;
}

// ----------------------------------------------------------------------
// psc_setup_mrc_domain

struct mrc_domain *
psc_setup_mrc_domain(const Grid_t::Domain& grid_domain, const GridBc& grid_bc, int nr_patches)
{
  // FIXME, should be split to create, set_from_options, setup time?
  struct mrc_domain *domain = mrc_domain_create(MPI_COMM_WORLD);
  // create a very simple domain decomposition
  int bc[3] = {};
  for (int d = 0; d < 3; d++) {
    if (grid_bc.fld_lo[d] == BND_FLD_PERIODIC && grid_domain.gdims[d] > 1) {
      bc[d] = BC_PERIODIC;
    }
  }

  mrc_domain_set_type(domain, "multi");
  mrc_domain_set_param_int3(domain, "m", grid_domain.gdims);
  mrc_domain_set_param_int(domain, "bcx", bc[0]);
  mrc_domain_set_param_int(domain, "bcy", bc[1]);
  mrc_domain_set_param_int(domain, "bcz", bc[2]);
  mrc_domain_set_param_int(domain, "nr_patches", nr_patches);
  mrc_domain_set_param_int3(domain, "np", grid_domain.np);

  struct mrc_crds *crds = mrc_domain_get_crds(domain);
  mrc_crds_set_type(crds, "uniform");
  mrc_crds_set_param_int(crds, "sw", 2);
  mrc_crds_set_param_double3(crds, "l", grid_domain.corner);
  mrc_crds_set_param_double3(crds, "h", grid_domain.corner + grid_domain.length);

  mrc_domain_set_from_options(domain);
  mrc_domain_setup(domain);

  return domain;
}

// ----------------------------------------------------------------------
// psc_make_grid

Grid_t* psc::make_grid(struct mrc_domain* mrc_domain, const Grid_t::Domain& domain, const GridBc& bc,
		       const Grid_t::Kinds& kinds, Grid_t::Normalization coeff, double dt)
{
  Int3 gdims;
  mrc_domain_get_global_dims(mrc_domain, gdims);
  int n_patches;
  mrc_patch *patches = mrc_domain_get_patches(mrc_domain, &n_patches);
  assert(n_patches > 0);
  Int3 ldims = patches[0].ldims;
  std::vector<Int3> offs;
  for (int p = 0; p < n_patches; p++) {
    assert(ldims == Int3(patches[p].ldims));
    offs.push_back(patches[p].off);
  }

  Grid_t *grid = new Grid_t(domain, offs);

  grid->kinds = kinds;

  grid->bc = bc;
  for (int d = 0; d < 3; d++) {
    if (grid->isInvar(d)) {
      // if invariant in this direction: set bnd to periodic (FIXME?)
      grid->bc.fld_lo[d] = BND_FLD_PERIODIC;
      grid->bc.fld_hi[d] = BND_FLD_PERIODIC;
      grid->bc.prt_lo[d] = BND_PRT_PERIODIC;
      grid->bc.prt_hi[d] = BND_PRT_PERIODIC;
    }
  }
  
  grid->norm = coeff;
  grid->dt = dt;

  return grid;
}

// ----------------------------------------------------------------------
// psc_setup_domain

void psc_setup_domain(struct psc *psc, const Grid_t::Domain& domain, GridBc& bc, const Grid_t::Kinds& kinds,
		      const Grid_t::Normalization& norm, double dt)
{
#if 0
  mpi_printf(MPI_COMM_WORLD, "::: dt      = %g\n", dt);
  mpi_printf(MPI_COMM_WORLD, "::: dx      = %g %g %g\n", domain.dx[0], domain.dx[1], domain.dx[2]);
#endif

  assert(domain.dx[0] > 0.);
  assert(domain.dx[1] > 0.);
  assert(domain.dx[2] > 0.);
  
  for (int d = 0; d < 3; d++) {
    if (psc->ibn[d] != 0) {
      continue;
    }
    // FIXME, old-style particle pushers need 3 ghost points still
    if (domain.gdims[d] == 1) {
      // no ghost points
      psc->ibn[d] = 0;
    } else {
      psc->ibn[d] = 2;
    }
  }

  psc->mrc_domain_ = psc_setup_mrc_domain(domain, bc, -1);
  psc->grid_ = psc->make_grid(psc->mrc_domain_, domain, bc, kinds, norm, dt);

  // make sure that np isn't overridden on the command line
  Int3 np;
  mrc_domain_get_param_int3(psc->mrc_domain_, "np", np);
  assert(np == domain.np);
}

// ----------------------------------------------------------------------
// psc_destroy

static void
_psc_destroy(struct psc *psc)
{
  mrc_domain_destroy(psc->mrc_domain_);

  ppsc = NULL;
}

// ----------------------------------------------------------------------
// _psc_write

static void
_psc_write(struct psc *psc, struct mrc_io *io)
{
  mrc_io_write_int(io, psc, "timestep", psc->timestep);
#if 0
  mrc_io_write_int(io, psc, "nr_kinds", psc->nr_kinds_);

  for (int k = 0; k < psc->nr_kinds_; k++) {
    char s[20];
    sprintf(s, "kind_q%d", k);
    mrc_io_write_double(io, psc, s, psc->kinds_[k].q);
    sprintf(s, "kind_m%d", k);
    mrc_io_write_double(io, psc, s, psc->kinds_[k].m);
    mrc_io_write_string(io, psc, s, psc->kinds_[k].name);
  }
#endif
  mrc_io_write_ref(io, psc, "mrc_domain", psc->mrc_domain_);
  //mrc_io_write_ref(io, psc, "mparticles", psc->particles_);
  //mrc_io_write_ref(io, psc, "mfields", psc->flds);
}

// ----------------------------------------------------------------------
// _psc_read

static void
_psc_read(struct psc *psc, struct mrc_io *io)
{
  assert(!ppsc);
  ppsc = psc;

  psc_setup_coeff(psc);

  mrc_io_read_int(io, psc, "timestep", &psc->timestep);
#if 0
  mrc_io_read_int(io, psc, "nr_kinds", &psc->nr_kinds_);
  psc->kinds_ = new psc_kind[psc->nr_kinds_]();
  for (int k = 0; k < psc->nr_kinds_; k++) {
    char s[20];
    sprintf(s, "kind_q%d", k);
    mrc_io_read_double(io, psc, s, &psc->kinds_[k].q);
    sprintf(s, "kind_m%d", k);
    mrc_io_read_double(io, psc, s, &psc->kinds_[k].m);
    mrc_io_read_string(io, psc, s, &psc->kinds_[k].name);
  }
#endif
  
  psc->mrc_domain_ = mrc_io_read_ref(io, psc, "mrc_domain", mrc_domain);
  //psc_setup_domain(psc, psc->domain_, psc->bc_, psc->kinds_);
#ifdef USE_FORTRAN
  psc_setup_fortran(psc);
#endif

  //psc->particles_ = mrc_io_read_ref(io, psc, "mparticles", psc_mparticles);
  //psc->flds = mrc_io_read_ref(io, psc, "mfields", psc_mfields);

  psc_read_member_objs(psc, io);
}

// ----------------------------------------------------------------------
// _psc_view

static void
_psc_view(struct psc *psc)
{
  const auto& kinds = psc->grid().kinds;
  mrc_domain_view(psc->mrc_domain_);

  MPI_Comm comm = psc_comm(psc);
  mpi_printf(comm, "%20s|\n", "particle kinds");
  for (int k = 0; k < kinds.size(); k++) {
    mpi_printf(comm, "%19s | q = %g m = %g\n",
	       kinds[k].name, kinds[k].q, kinds[k].m);
  }
}

// ======================================================================
// psc class

struct mrc_class_psc_ : mrc_class_psc {
  mrc_class_psc_() {
    name             = "psc";
    size             = sizeof(struct psc);
    param_descr      = psc_descr;
    create           = _psc_create;
    view             = _psc_view;
    destroy          = _psc_destroy;
    write            = _psc_write;
    read             = _psc_read;
  }
} mrc_class_psc;

// ======================================================================
// helpers

// ----------------------------------------------------------------------
// psc_default_dimensionless
//
// sets up parameter defaults for dimensionless units

void
psc_default_dimensionless(struct psc *psc)
{
  psc->norm_params.qq = 1.;
  psc->norm_params.mm = 1.;
  psc->norm_params.tt = 1.;
  psc->norm_params.cc = 1.;
  psc->norm_params.eps0 = 1.;

  psc->norm_params.lw = 2.*M_PI;
  psc->norm_params.i0 = 0.;
  psc->norm_params.n0 = 1.;
  psc->norm_params.e0 = 1.;
}

