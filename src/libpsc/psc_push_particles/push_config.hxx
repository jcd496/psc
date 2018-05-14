
#pragma once

#include "dim.hxx"
#include "fields.hxx"

#include "inc_defs.h"
#include "interpolate.hxx"
#include "inc_curr.c"
#include "pushp_current_esirkepov.hxx"
#include "inc_push.c"
#include "push_particles.hxx"

#define atomicAdd(addr, val) \
  do { *(addr) += (val); } while (0)

template<typename fields_t, typename dim_curr>
struct curr_cache_t : fields_t
{
  using real_t = typename fields_t::real_t;
  
  curr_cache_t(const fields_t& f)
    : fields_t(f.ib(), f.im(), f.n_comps(), f.data_)
  {}
  
  void add(int m, int i, int j, int k, real_t val)
  {
    Fields3d<fields_t, dim_curr> J(*this);
    real_t *addr = &J(JXI+m, i,j,k);
    atomicAdd(addr, val);
  }
};

template<typename MP, typename MF,
	 typename IPEM,
	 typename D, typename O,
	 template<typename, typename> class CURRENT,
	 typename dim_curr = dim_xyz>
struct push_p_config
{
  using Mparticles = MP;
  using Mfields = MF;
  using dim = D;
  using Current_t = CURRENT<curr_cache_t<typename MF::fields_t, dim_curr>, D>;

  using InterpolateEM_t = IPEM;
  using AdvanceParticle_t = AdvanceParticle<typename MP::real_t, D>;
  using CurrentE_t = Current<O, D, InterpolateEM_t, Fields3d<typename MF::fields_t>>;
};

#include "psc_particles_double.h"
#include "psc_particles_single.h"
#include "psc_fields_c.h"
#include "psc_fields_single.h"

template<typename dim>
using Config2ndDouble = push_p_config<MparticlesDouble, MfieldsC,
				      InterpolateEM2nd<Fields3d<MfieldsC::fields_t>, dim>,
				      dim, opt_order_2nd,
				      CurrentNone>;

template<typename dim>
using Config1st = push_p_config<MparticlesDouble, MfieldsC,
				InterpolateEM1st<Fields3d<MfieldsC::fields_t>, dim>,
				dim, opt_order_1st,
				CurrentNone>;

template<typename Mparticles, typename Mfields, typename dim>
using Config1vbec = push_p_config<Mparticles, Mfields,
				  InterpolateEM1vbec<Fields3d<typename Mfields::fields_t>, dim>,
				  dim, opt_order_1st,
				  Current1vbVar1>;

template<typename Mparticles, typename Mfields, typename dim>
using Config1vbecSplit = push_p_config<Mparticles, Mfields,
				       InterpolateEM1vbec<Fields3d<typename Mfields::fields_t>, dim>,
				       dim, opt_order_1st,
				       Current1vbSplit>;

template<typename dim>
using Config1vbecDouble = Config1vbec<MparticlesDouble, MfieldsC, dim>;

template<typename dim>
using Config1vbecSingle = Config1vbec<MparticlesSingle, MfieldsSingle, dim>;

using Config1vbecSingleXZ = push_p_config<MparticlesSingle, MfieldsSingle,
					  InterpolateEM1vbec<Fields3d<MfieldsSingle::fields_t, dim_xz>, dim_xyz>,
					  dim_xyz, opt_order_1st,
					  Current1vbSplit,
					  dim_xz>;
using Config1vbecSingle1 = push_p_config<MparticlesSingle, MfieldsSingle,
					 InterpolateEM1vbec<Fields3d<MfieldsSingle::fields_t, dim_1>, dim_1>,
					 dim_1, opt_order_1st,
					 Current1vbVar1,
					 dim_1>;

