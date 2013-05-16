
//#define BOUNDS_CHECK

#include "ggcm_mhd_defs.h"
#include "ggcm_mhd_private.h"
#include "ggcm_mhd_step.h"
#include "ggcm_mhd_crds_private.h"
#include "ggcm_mhd_crds_gen.h"
#include "ggcm_mhd_bnd.h"
#include "ggcm_mhd_diag.h"
#include "ggcm_mhd_ic_private.h"

#include <mrc_ts.h>
#include <mrc_ts_monitor.h>
#include <mrc_fld.h>
#include <mrc_domain.h>
#include <mrc_params.h>
#include <mrc_ddc.h>
#include <mrctest.h>
#include <mrc_io.h> 

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h> 
#include <assert.h>

// ======================================================================
// ggcm_mhd_ic subclass "fadeev"

struct ggcm_mhd_ic_fadeev {
  float mpermi;
  float Bo; 
  float Boz;
  float pert; 
  float eps; 
  float lambda;   
  float dens0; 
};
// ----------------------------------------------------------------------
// ggcm_mhd_ic_fadeev_run

static void
ggcm_mhd_ic_fadeev_run(struct ggcm_mhd_ic *ic)
{
  struct ggcm_mhd_ic_fadeev *sub = mrc_to_subobj(ic, struct ggcm_mhd_ic_fadeev);
  struct ggcm_mhd *gmhd = ic->mhd;
  struct mrc_f3 *f3 = ggcm_mhd_flds_get_mrc_f3(gmhd->flds_base);
  struct mrc_crds *crds = mrc_domain_get_crds(gmhd->domain);  
  struct mrc_f3 *fld_psi = mrc_domain_f3_create(gmhd->domain, SW_2);

  ggcm_mhd_bnd_set_type(gmhd->bnd, "conducting");
  //mrc_domain_set_param_int(gmhd->domain, "bcx", &bc[0]
  mrc_domain_set_param_int(gmhd->domain, "bcy", BC_NONE);	   
  mrc_f3_setup(fld_psi);

  float xl[3], xh[3], L[3], r[3];
  mrc_crds_get_xl_xh(crds, xl, xh);
  for(int i=0; i<3; i++){
    L[i] = xh[i] - xl[i];
  }
  
  float gamma = gmhd->par.gamm;
  float Bo=sub->Bo;
  float pert=sub->pert;
  float Boz=sub->Boz;
  float eps=sub->eps;
  float lam=(sub->lambda)*L[0] ;  // defines island size   
  float kk= (2.*M_PI) / lam ;      

  mrc_f3_foreach(f3, ix, iy, iz, 2, 2) {
    r[0] = .5*(MRC_CRDX(crds, ix) + MRC_CRDX(crds, ix-1));
    r[1] = .5*(MRC_CRDY(crds, iy) + MRC_CRDY(crds, iy-1));
    
    MRC_F3(fld_psi, 0, ix,iy,iz) = -(Bo / kk)*( log(cosh(kk*r[1]) + eps*cos(kk*r[0])));      
  } mrc_f3_foreach_end;

  float *bd2x = ggcm_mhd_crds_get_crd(gmhd->crds, 0, BD2);
  float *bd2y = ggcm_mhd_crds_get_crd(gmhd->crds, 1, BD2);

  mrc_f3_foreach(f3, ix, iy, iz, 1, 1) {
    // FIXME! the staggering for B is okay, but fld_psi and other stuff below needs to be
    // fixed / checked for cell-centered
    r[0] = MRC_CRD(crds, 0, ix);
    r[1] = MRC_CRD(crds, 1, iy);
    r[2] = MRC_CRD(crds, 2, iz);
    
    B1X(f3, ix,iy,iz) =  (MRC_F3(fld_psi, 0, ix,iy+1,iz) - MRC_F3(fld_psi, 0, ix,iy,iz)) / bd2y[iy];
    B1Y(f3, ix,iy,iz) = -(MRC_F3(fld_psi, 0, ix+1,iy,iz) - MRC_F3(fld_psi, 0, ix,iy,iz)) / bd2x[ix];

    MRC_F3(f3, _RR1, ix, iy, iz)  = 0.5*sqr(Bo) * (1.0-sqr(eps)) * 
      exp(2.0*kk* MRC_F3(fld_psi, 0, ix,iy,iz)/(Bo)) + 0.5*sqr(Boz) + sub->dens0;
    MRC_F3(f3, _RV1X, ix,iy,iz) = (pert) * (1.-kk*kk*r[0]*r[0]) *
      MRC_F3(f3, _RR1, ix, iy, iz) * exp(-kk*kk*r[1]*r[1])*sin(kk*r[0]*0.5);	
    MRC_F3(f3, _RV1Y, ix,iy,iz) = -(pert) * ( 0.5*kk*r[1] ) * MRC_F3(f3, _RR1, ix, iy, iz) *
      exp(-kk*kk*r[1]*r[1])*cos(kk*r[0]*0.5);            
    
    MRC_F3(f3, _UU1 , ix, iy, iz) =  MRC_F3(f3, _RR1, ix, iy, iz) / (gamma -1.f) + 	
      .5f * (sqr(MRC_F3(f3, _RV1X, ix, iy, iz)) +
	     sqr(MRC_F3(f3, _RV1Y, ix, iy, iz)) +
	     sqr(MRC_F3(f3, _RV1Z, ix, iy, iz))) / MRC_F3(f3, _RR1, ix, iy, iz) +
      .5f * (sqr(.5*(B1X(f3, ix,iy,iz) + B1X(f3, ix+1,iy,iz))) +
	     sqr(.5*(B1Y(f3, ix,iy,iz) + B1Y(f3, ix,iy+1,iz))) +
	     sqr(.5*(B1Z(f3, ix,iy,iz) + B1Z(f3, ix,iy,iz+1))));
  } mrc_f3_foreach_end;
}

// ----------------------------------------------------------------------
// ggcm_mhd_ic_fadeev_descr

#define VAR(x) (void *)offsetof(struct ggcm_mhd_ic_fadeev, x)
static struct param ggcm_mhd_ic_fadeev_descr[] = {
  {"mpermi", VAR(mpermi), PARAM_FLOAT(1.)},
  {"Bo", VAR(Bo), PARAM_FLOAT(1.0)},
  {"Boz", VAR(Boz), PARAM_FLOAT(10.0)},
  {"pert", VAR(pert), PARAM_FLOAT(0.001)},
  {"eps", VAR(eps), PARAM_FLOAT(0.3)},  
  {"lambda", VAR(lambda), PARAM_FLOAT(0.5)},  
  {"dens0", VAR(dens0), PARAM_FLOAT(5.0)},
  {},
};
#undef VAR

// ----------------------------------------------------------------------
// ggcm_mhd_ic_fadeev_ops

struct ggcm_mhd_ic_ops ggcm_mhd_ic_fadeev_ops = {
  .name        = "fadeev",
  .size        = sizeof(struct ggcm_mhd_ic_fadeev),
  .param_descr = ggcm_mhd_ic_fadeev_descr,
  .run         = ggcm_mhd_ic_fadeev_run,
};


// ======================================================================
// ggcm_mhd class "cweno"

// ----------------------------------------------------------------------
// ggcm_mhd_cweno_create

static void
ggcm_mhd_cweno_create(struct ggcm_mhd *mhd)
{
  mhd->par.rrnorm = 1.f;
  mhd->par.ppnorm = 1.f;
  mhd->par.vvnorm = 1.f;
  mhd->par.bbnorm = 1.f;
  mhd->par.ccnorm = 1.f;
  mhd->par.eenorm = 1.f;
  mhd->par.resnorm = 1.f;
  mhd->par.diffco = 0.f;

  ggcm_mhd_bnd_set_type(mhd->bnd, "none");

  mrc_domain_set_param_int(mhd->domain, "bcx", BC_PERIODIC);
  mrc_domain_set_param_int(mhd->domain, "bcy", BC_PERIODIC);
  mrc_domain_set_param_int(mhd->domain, "bcz", BC_PERIODIC);

  /* set defaults for coord arrays */
  struct mrc_crds *crds = mrc_domain_get_crds(mhd->domain);
  mrc_crds_set_type(crds, "gaussian_2D");
  mrc_crds_set_param_int(crds, "sw", SW_2);   // 'stencil width' 
  mrc_crds_set_param_float3(crds, "l", (float[3]) {  0.0, 0.0, -1.0 });
  mrc_crds_set_param_float3(crds, "h", (float[3]) {  2.*M_PI, 2.*M_PI,  1.0 });

  /* set defaults for the ddc, this does the communication */
  struct mrc_ddc *ddc = mrc_domain_get_ddc(mhd->domain);
  mrc_ddc_set_param_int(ddc, "max_n_fields", 8);
  mrc_ddc_set_param_int3(ddc, "ibn", (int[3]) { SW_2, SW_2, SW_2 });

  // generate MHD solver grid from mrc_crds
  ggcm_mhd_crds_gen_set_type(mhd->crds->crds_gen, "mrc");
  ggcm_mhd_set_param_float(mhd, "isphere", 0.);
  ggcm_mhd_set_param_float(mhd, "diffsphere", 0.);
  ggcm_mhd_set_param_float(mhd, "speedlimit", 1e9);
}

static struct ggcm_mhd_ops ggcm_mhd_cweno_ops = {
  .name             = "cweno",
  .create           = ggcm_mhd_cweno_create,
};

// ======================================================================

extern struct ggcm_mhd_diag_ops ggcm_mhd_diag_c_ops;

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  libmrc_params_init(argc, argv);
  ggcm_mhd_register();

  mrc_class_register_subclass(&mrc_class_ggcm_mhd, &ggcm_mhd_cweno_ops);  
  mrc_class_register_subclass(&mrc_class_ggcm_mhd_diag, &ggcm_mhd_diag_c_ops);

  mrc_class_register_subclass(&mrc_class_ggcm_mhd_ic, &ggcm_mhd_ic_fadeev_ops);  
 
  struct ggcm_mhd *mhd = ggcm_mhd_create(MPI_COMM_WORLD);
  ggcm_mhd_set_type(mhd, "cweno");
  ggcm_mhd_step_set_type(mhd->step, "cweno");
  ggcm_mhd_flds_set_type(mhd->flds_base, "c");
  ggcm_mhd_set_from_options(mhd);
  ggcm_mhd_setup(mhd);
  ggcm_mhd_view(mhd);

  // set up initial condition
  ggcm_mhd_ic_run(mhd->ic);

  // run time integration
  struct mrc_ts *ts = mrc_ts_create(mrc_domain_comm(mhd->domain));
  mrc_ts_set_type(ts, "rk2");
  mrc_ts_set_context(ts, ggcm_mhd_to_mrc_obj(mhd));

  struct mrc_ts_monitor *mon_output =
    mrc_ts_monitor_create(mrc_ts_comm(ts));
  mrc_ts_monitor_set_type(mon_output, "ggcm");
  mrc_ts_monitor_set_name(mon_output, "mrc_ts_output");
  mrc_ts_add_monitor(ts, mon_output);

  mrc_ts_set_dt(ts, 1e-6);
  mrc_ts_set_solution(ts, mrc_f3_to_mrc_obj(ggcm_mhd_flds_get_mrc_f3(mhd->flds_base)));
  mrc_ts_set_rhs_function(ts, ts_ggcm_mhd_step_calc_rhs, mhd);
  mrc_ts_set_from_options(ts);
  mrc_ts_view(ts);
  mrc_ts_setup(ts);
  mrc_ts_solve(ts);
  mrc_ts_view(ts);
  mrc_ts_destroy(ts);  
  ggcm_mhd_destroy(mhd);

  MPI_Finalize();
  return 0;
}

