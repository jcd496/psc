
#pragma once

#include "marder.hxx"
#include "psc_fields_vpic.h"
#include "psc_particles_vpic.h"

// ======================================================================
// psc_marder "vpic"

struct MarderVpic : MarderBase
{
  using real_t = MfieldsVpic::fields_t::real_t;
  
  MarderVpic(MPI_Comm comm, real_t diffusion, int loop, bool dump)
    : comm_{comm}
  {}
  
  // ----------------------------------------------------------------------
  // psc_marder_vpic_clean_div_e

  void psc_marder_vpic_clean_div_e(MfieldsStateVpic& mflds, MparticlesVpic& mprts)
  {
    mpi_printf(comm_, "Divergence cleaning electric field\n");
  
    TIC CleanDivOps::clear_rhof(mflds.vmflds()); TOC(clear_rhof, 1);
    mflds.sim()->accumulate_rho_p(mprts.vmprts_, mflds.vmflds());
    CleanDivOps::synchronize_rho(mflds.vmflds());

    for (int round = 0; round < num_div_e_round_; round++ ) {
      TIC CleanDivOps::compute_div_e_err(mflds.vmflds()); TOC(compute_div_e_err, 1);
      if (round == 0 || round == num_div_e_round_ - 1) {
	double err;
	TIC err = CleanDivOps::compute_rms_div_e_err(mflds.vmflds()); TOC(compute_rms_div_e_err, 1);
	mpi_printf(comm_, "%s rms error = %e (charge/volume)\n",
		   round == 0 ? "Initial" : "Cleaned", err);
      }
      TIC CleanDivOps::clean_div_e(mflds.vmflds()); TOC(clean_div_e, 1);
    }
  }

  // ----------------------------------------------------------------------
  // psc_marder_vpic_clean_div_b

  void psc_marder_vpic_clean_div_b(MfieldsStateVpic& mflds)
  {
    mpi_printf(comm_, "Divergence cleaning magnetic field\n");
  
    for (int round = 0; round < num_div_b_round_; round++) {
      // FIXME, having the TIC/TOC stuff with every use is not pretty
      TIC CleanDivOps::compute_div_b_err(mflds.vmflds()); TOC(compute_div_b_err, 1);
      if (round == 0 || round == num_div_b_round_ - 1) {
	double err;
	TIC err = CleanDivOps::compute_rms_div_b_err(mflds.vmflds()); TOC(compute_rms_div_b_err, 1);
	mpi_printf(comm_, "%s rms error = %e (charge/volume)\n",
		   round == 0 ? "Initial" : "Cleaned", err);
      }
      TIC CleanDivOps::clean_div_b(mflds.vmflds()); TOC(clean_div_b, 1);
    }
  }

  // ----------------------------------------------------------------------
  // run

  void run(MfieldsBase& mflds_base, MparticlesBase& mprts_base) override
  {
    assert(0);
  }
  
  void operator()(MfieldsStateVpic& mflds, MparticlesVpic& mprts)
  {
    struct psc *psc = ppsc; // FIXME

    bool clean_div_e = (clean_div_e_interval_ > 0 && psc->timestep % clean_div_e_interval_ == 0);
    bool clean_div_b = (clean_div_b_interval_ > 0 && psc->timestep % clean_div_b_interval_ == 0);
    bool sync_shared = (sync_shared_interval_ > 0 && psc->timestep % sync_shared_interval_ == 0);

    if (!(clean_div_e || clean_div_b || sync_shared)) {
      return;
    }
  
    // Divergence clean e
    if (clean_div_e) {
      // needs E, rhof, rhob, material
      psc_marder_vpic_clean_div_e(mflds, mprts);
      // upates E, rhof, div_e_err
    }

    // Divergence clean b
    if (clean_div_b) {
      // needs B
      psc_marder_vpic_clean_div_b(mflds);
      // updates B, div_b_err
    }
  
    // Synchronize the shared faces
    if (sync_shared) {
      // needs E, B, TCA
      mpi_printf(comm_, "Synchronizing shared tang e, norm b\n");
      double err;
      TIC err = CleanDivOps::synchronize_tang_e_norm_b(mflds.vmflds()); TOC(synchronize_tang_e_norm_b, 1);
      mpi_printf(comm_, "Domain desynchronization error = %e (arb units)\n", err);
      // updates E, B, TCA
    }
  }

private:
  MPI_Comm comm_;
  int clean_div_e_interval_ = 10; // FIXME, hardcoded...
  int clean_div_b_interval_ = 10;
  int sync_shared_interval_ = 10;
  int num_div_e_round_ = 3;
  int num_div_b_round_ = 3;
};

