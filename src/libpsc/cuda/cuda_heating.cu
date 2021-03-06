
#include "cuda_iface.h"
#include "cuda_mparticles.cuh"
#include "cuda_mfields.h"
#include "cuda_bits.h"
#include "cuda_base.cuh"
#include "psc_bits.h"
#include "heating_spot_foil.hxx"
#include "heating_cuda_impl.hxx"
#include "balance.hxx"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <curand_kernel.h>

#include <cstdio>

#define THREADS_PER_BLOCK 256

using Float3 = Vec3<float>;

// ----------------------------------------------------------------------
// bm_normal2

static inline float2 bm_normal2(void)
{
  float u1, u2;
  do {
    u1 = random() * (1.f / RAND_MAX);
    u2 = random() * (1.f / RAND_MAX);
  } while (u1 <= 0.f);

  float2 rv;
  rv.x = sqrtf(-2.f * logf(u1)) * cosf(2.f * M_PI * u2);
  rv.y = sqrtf(-2.f * logf(u1)) * sinf(2.f * M_PI * u2);
  return rv;
}

// ----------------------------------------------------------------------
// k_curand_setup

__global__ static void k_curand_setup(curandState* d_curand_states)
{
  int bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  int id = threadIdx.x + bid * THREADS_PER_BLOCK;

  curand_init(1234, id, 0, &d_curand_states[id]);
}

// ----------------------------------------------------------------------
// d_particle_kick

__device__ void d_particle_kick(float4* pxi4, float H, float heating_dt,
                                curandState* state)
{
  float2 r01 = curand_normal2(state);
  float r2 = curand_normal(state);

  float Dp = sqrtf(H * heating_dt);

  pxi4->x += Dp * r01.x;
  pxi4->y += Dp * r01.y;
  pxi4->z += Dp * r2;
}

// ----------------------------------------------------------------------
// k_heating_run_foil

template <typename BS, typename HS>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, 3)
  k_heating_run_foil(HS foil, DMparticlesCuda<BS> dmprts, float heating_dt,
                     Float3* d_xb_by_patch, curandState* d_curand_states)
{
  BlockSimple<BS, typename HS::dim> current_block;
  if (!current_block.init(dmprts)) {
    return;
  }

  Float3 xb; // __shared__
  xb[0] = d_xb_by_patch[current_block.p][0];
  xb[1] = d_xb_by_patch[current_block.p][1];
  xb[2] = d_xb_by_patch[current_block.p][2];

  int bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  int id = threadIdx.x + bid * THREADS_PER_BLOCK;
  /* Copy state to local memory for efficiency */
  curandState local_state = d_curand_states[id];

  int block_begin = dmprts.off_[current_block.bid];
  int block_end = dmprts.off_[current_block.bid + 1];
  for (int n : in_block_loop(block_begin, block_end)) {
    if (n < block_begin) {
      continue;
    }
    float4 xi4 = dmprts.storage.xi4[n];

    int prt_kind = __float_as_int(xi4.w);

    float xx[3] = {
      xi4.x + xb[0],
      xi4.y + xb[1],
      xi4.z + xb[2],
    };
    float H = foil(xx, prt_kind);
    if (H > 0.f) {
      float4 pxi4 = dmprts.storage.pxi4[n];
      d_particle_kick(&pxi4, H, heating_dt, &local_state);
      dmprts.storage.pxi4[n] = pxi4;
    }
  }

  d_curand_states[id] = local_state;
}

// ======================================================================
// cuda_heating_foil

template <typename HS>
struct cuda_heating_foil
{
  cuda_heating_foil(const Grid_t& grid, const HS& heating_spot,
                    double heating_dt)
    : heating_dt_(heating_dt), heating_spot_{heating_spot}, first_time_{true}
  {}

  // no copy constructor / assign, to catch performance issues
  cuda_heating_foil(const cuda_heating_foil&) = delete;
  cuda_heating_foil& operator=(const cuda_heating_foil&) = delete;

  void reset() { first_time_ = true; }

  // ----------------------------------------------------------------------
  // operator()

  template <typename BS>
  void operator()(cuda_mparticles<BS>* cmprts)
  {
    // return cuda_heating_run_foil_gold(cmprts);
    if (cmprts->n_prts == 0) {
      return;
    }

    dim3 dimGrid = BlockSimple<BS, typename HS::dim>::dimGrid(*cmprts);

    if (first_time_) { // FIXME
      d_xb_by_patch_ = cmprts->xb_by_patch;

      d_curand_states_.resize(dimGrid.x * dimGrid.y * dimGrid.z *
                              THREADS_PER_BLOCK);
      k_curand_setup<<<dimGrid, THREADS_PER_BLOCK>>>(
        d_curand_states_.data().get());
      cuda_sync_if_enabled();

      first_time_ = false;
    }

    if (cmprts->need_reorder) {
      cmprts->reorder();
    }

    k_heating_run_foil<BS><<<dimGrid, THREADS_PER_BLOCK>>>(
      heating_spot_, *cmprts, heating_dt_, d_xb_by_patch_.data().get(),
      d_curand_states_.data().get());
    cuda_sync_if_enabled();
  }

  // state (FIXME, shouldn't be part of the interface)
  bool first_time_;
  float heating_dt_;
  HS heating_spot_;

  psc::device_vector<Float3> d_xb_by_patch_;
  psc::device_vector<curandState> d_curand_states_;
};

// ----------------------------------------------------------------------
// particle_kick

__host__ void particle_kick(float4* pxi4, float H, float heating_dt)
{
  float2 r01 = bm_normal2();
  float2 r23 = bm_normal2();

  float Dp = sqrtf(H * heating_dt);

  pxi4->x += Dp * r01.x;
  pxi4->y += Dp * r01.y;
  pxi4->z += Dp * r23.x;
}

// ----------------------------------------------------------------------
// cuda_heating_run_foil_gold

template <typename BS, typename HS>
void cuda_heating_run_foil_gold(HS& foil, float heating_dt,
                                cuda_mparticles<BS>* cmprts)
{
  for (int b = 0; b < cmprts->n_blocks; b++) {
    int p = b / cmprts->n_blocks_per_patch;
    for (int n = cmprts->d_off[b]; n < cmprts->d_off[b + 1]; n++) {
      float4 xi4 = cmprts->d_xi4[n];

      int prt_kind = cuda_float_as_int(xi4.w);

      float* xb = &cmprts->xb_by_patch[p][0];
      float xx[3] = {
        xi4.x + xb[0],
        xi4.y + xb[1],
        xi4.z + xb[2],
      };

      float H = foil(xx, prt_kind);
      // float4 pxi4 = d_pxi4[n];
      // printf("%s xx = %g %g %g H = %g px = %g %g %g\n", (H > 0) ? "H" : " ",
      // 	     xx[0], xx[1], xx[2], H,
      // 	     pxi4.x, pxi4.y, pxi4.z);
      // pxi4.w = H;
      // d_pxi4[n] = pxi4;
      if (H > 0) {
        float4 pxi4 = cmprts->d_pxi4[n];
        particle_kick(&pxi4, H, heating_dt);
        cmprts->d_pxi4[n] = pxi4;
        // printf("H xx = %g %g %g H = %g px = %g %g %g\n", xx[0], xx[1], xx[2],
        // H,
        //        pxi4.x, pxi4.y, pxi4.z);
      }
    }
  }
}

// ======================================================================

template <typename HS, typename MP>
HeatingCuda<HS, MP>::HeatingCuda(const Grid_t& grid, int interval,
                                 HS heating_spot)
  : foil_{new cuda_heating_foil<HS>{grid, heating_spot, interval * grid.dt}},
    balance_generation_cnt_{-1}
{}

template <typename HS, typename MP>
HeatingCuda<HS, MP>::~HeatingCuda()
{
  delete foil_;
}

template <typename HS, typename MP>
void HeatingCuda<HS, MP>::reset(const MP& mprts)
{
  foil_->reset();
}

template <typename HS, typename MP>
void HeatingCuda<HS, MP>::operator()(MP& mprts)
{
  if (psc_balance_generation_cnt > this->balance_generation_cnt_) {
    balance_generation_cnt_ = psc_balance_generation_cnt;
    reset(mprts);
  }

  (*foil_)(mprts.cmprts());
}

// ======================================================================

template struct HeatingCuda<HeatingSpotFoil<dim_yz>, MparticlesCuda<BS144>>;
template struct HeatingCuda<HeatingSpotFoil<dim_xyz>, MparticlesCuda<BS444>>;
