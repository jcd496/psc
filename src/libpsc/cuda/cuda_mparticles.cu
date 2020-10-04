
#include "cuda_mparticles.cuh"
#include "cuda_bits.h"

#include "psc_bits.h"
#include "bs.hxx"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "cuda_base.cuh"

#include <cstdio>
#include <cassert>

// ----------------------------------------------------------------------
// ctor

template <typename BS>
cuda_mparticles<BS>::cuda_mparticles(const Grid_t& grid)
  : cuda_mparticles_base<BS>(grid)
{
  cuda_base_init();

  xb_by_patch.resize(this->n_patches());
  for (int p = 0; p < this->n_patches(); p++) {
    xb_by_patch[p] = Real3(grid.patches[p].xb);
  }
}

// ----------------------------------------------------------------------
// resize
//
// the goal here is to have d_xi4, d_pxi4, d_bidx and d_id always
// have the same size.

template <typename BS>
void cuda_mparticles<BS>::resize(uint n_prts)
{
  cuda_mparticles_base<BS>::resize(n_prts);
  this->by_block_.d_idx.resize(n_prts);
  this->by_block_.d_id.resize(n_prts);
}

// ----------------------------------------------------------------------
// dump_by_patch

template <typename BS>
void cuda_mparticles<BS>::dump_by_patch(uint* n_prts_by_patch)
{
  printf("cuda_mparticles_dump_by_patch: n_prts = %d\n", this->n_prts);
  uint off = 0;
  for (int p = 0; p < this->n_patches(); p++) {
    float* xb = &xb_by_patch[p][0];
    for (int n = 0; n < n_prts_by_patch[p]; n++) {
      auto prt = this->storage.load(n + off);
      uint bidx = this->by_block_.d_idx[n + off],
           id = this->by_block_.d_id[n + off];
      printf("cuda_mparticles_dump_by_patch: [%d/%d] %g %g %g // %d // %g %g "
             "%g // %g b_idx %d id %d\n",
             p, n, prt.x[0] + xb[0], prt.x[1] + xb[1], prt.x[2] + xb[2],
             prt.kind, prt.u[0], prt.u[1], prt.u[2], prt.qni_wni, bidx, id);
    }
    off += n_prts_by_patch[p];
  }
}

// ----------------------------------------------------------------------
// dump

template <typename BS>
void cuda_mparticles<BS>::dump(const std::string& filename) const
{
  FILE* file = fopen(filename.c_str(), "w");
  assert(file);

  fprintf(file, "cuda_mparticles_dump: n_prts = %d\n", this->n_prts);
  uint off = 0;
  auto& d_off = this->by_block_.d_off;
  for (int b = 0; b < this->n_blocks; b++) {
    uint off_b = d_off[b], off_e = d_off[b + 1];
    int p = b / this->n_blocks_per_patch;
    fprintf(file, "cuda_mparticles_dump: block %d: %d -> %d (patch %d)\n", b,
            off_b, off_e, p);
    assert(d_off[b] == off);
    for (int n = d_off[b]; n < d_off[b + 1]; n++) {
      auto prt = this->storage.load(n + off);
      uint bidx = this->by_block_.d_idx[n], id = this->by_block_.d_id[n];
      fprintf(file,
              "mparticles_dump: [%d] %g %g %g // %d // %g %g %g // %g || bidx "
              "%d id %d %s\n",
              n, prt.x[0], prt.x[1], prt.x[2], prt.kind, prt.u[0], prt.u[1],
              prt.u[2], prt.qni_wni, bidx, id,
              b == bidx ? "" : "BIDX MISMATCH!");
    }
    off += off_e - off_b;
  }
  fclose(file);
}

// ----------------------------------------------------------------------
// swap_alt

template <typename BS>
void cuda_mparticles<BS>::swap_alt()
{
  this->storage.xi4.swap(alt_storage.xi4);
  // thrust::swap(this->storage.xi4, alt_storage.xi4);
  this->storage.pxi4.swap(alt_storage.pxi4);
  // thrust::swap(this->storage.pxi4, alt_storage.pxi4);
}

#define THREADS_PER_BLOCK 256

// ----------------------------------------------------------------------
// k_reorder_and_offsets

template <typename BS>
__global__ static void k_reorder_and_offsets(DMparticlesCuda<BS> dmprts,
                                             int nr_prts, const uint* d_bidx,
                                             const uint* d_ids, uint* d_off,
                                             int last_block)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  for (; i <= nr_prts; i += blockDim.x * gridDim.x) {
    int block, prev_block;
    if (i < nr_prts) {
      dmprts.storage.xi4[i] = dmprts.alt_storage.xi4[d_ids[i]];
      dmprts.storage.pxi4[i] = dmprts.alt_storage.pxi4[d_ids[i]];

      block = d_bidx[i];
    } else { // needed if there is no particle in the last block
      block = last_block;
    }

    // OPT: d_bidx[i-1] could use shmem
    // create offsets per block into particle array
    prev_block = -1;
    if (i > 0) {
      prev_block = d_bidx[i - 1];
    }
    for (int b = prev_block + 1; b <= block; b++) {
      d_off[b] = i;
    }
  }
}

// ----------------------------------------------------------------------
// reorder_and_offsets

template <typename BS>
void cuda_mparticles<BS>::reorder_and_offsets(
  const psc::device_vector<uint>& d_idx, const psc::device_vector<uint>& d_id,
  psc::device_vector<uint>& d_off)
{
  if (this->n_patches() == 0) {
    return;
  }

  swap_alt();
  resize(this->n_prts);

  int n_blocks = (this->n_prts + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  if (n_blocks > 32768)
    n_blocks = 32768;
  dim3 dimGrid(n_blocks);
  dim3 dimBlock(THREADS_PER_BLOCK);

  k_reorder_and_offsets<BS><<<dimGrid, dimBlock>>>(
    *this, this->n_prts, d_idx.data().get(), d_id.data().get(),
    d_off.data().get(), this->n_blocks);
  cuda_sync_if_enabled();

  need_reorder = false;
}

// ----------------------------------------------------------------------
// k_reorder

template <typename BS>
__global__ static void k_reorder(DMparticlesCuda<BS> dmprts, int n_prts,
                                 const uint* d_ids)
{
  int i = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;

  if (i < n_prts) {
    int j = d_ids[i];
    dmprts.storage.xi4[i] = dmprts.alt_storage.xi4[j];
    dmprts.storage.pxi4[i] = dmprts.alt_storage.pxi4[j];
  }
}

// ----------------------------------------------------------------------
// reorder

template <typename BS>
void cuda_mparticles<BS>::reorder()
{
  if (!need_reorder) {
    return;
  }

  reorder(this->by_block_.d_id);
  need_reorder = false;
}

// ----------------------------------------------------------------------
// reorder

template <typename BS>
void cuda_mparticles<BS>::reorder(const psc::device_vector<uint>& d_id)
{
  if (this->n_prts == 0) {
    return;
  }

  swap_alt();
  resize(this->n_prts);

  dim3 dimGrid((this->n_prts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

  k_reorder<BS>
    <<<dimGrid, THREADS_PER_BLOCK>>>(*this, this->n_prts, d_id.data().get());
  cuda_sync_if_enabled();
}

// ----------------------------------------------------------------------
// setup_internals

template <typename BS>
void cuda_mparticles<BS>::setup_internals()
{
  // pre-condition: particles sorted by patch, d_off being used to
  // describe patch boundaries

  // assert(check_in_patch_unordered_slow());

  this->by_block_.find_indices_ids(*this);

  // assert(check_bidx_id_unordered_slow());

  this->by_block_.stable_sort();

  this->by_block_.reorder_and_offsets(*this);

  // post-condition:
  // - particles now sorted by block
  // - d_off describes block boundaries
  // - UNUSED: d_bidx has each particle's block index

  // assert(check_ordered());
}

// ----------------------------------------------------------------------
// size

template <typename BS>
uint cuda_mparticles<BS>::size()
{
  return this->n_prts;
}

// ----------------------------------------------------------------------
// inject_initial
//
// adds particles initially, ie., into an empty cmprts
// does not complete setting correct internal state
// (setup_internal() needs to be called next)

template <typename BS>
void cuda_mparticles<BS>::inject_initial(
  const std::vector<Particle>& buf, const std::vector<uint>& n_prts_by_patch)
{
  thrust::host_vector<uint> h_off(this->by_block_.d_off);

  assert(this->storage.xi4.size() == 0);
  assert(this->n_prts == 0);

  uint buf_n = 0;
  for (int p = 0; p < this->n_patches(); p++) {
    assert(h_off[p * this->n_blocks_per_patch] == 0);
    assert(h_off[(p + 1) * this->n_blocks_per_patch] == 0);
    buf_n += n_prts_by_patch[p];
  }

  resize(buf_n);

  HMparticlesCudaStorage h_storage{buf_n};

  auto it = buf.begin();
  uint off = 0;
  for (int p = 0; p < this->n_patches(); p++) {
    auto n_prts = n_prts_by_patch[p];
    h_off[p * this->n_blocks_per_patch] = off;
    h_off[(p + 1) * this->n_blocks_per_patch] = off + n_prts;

    for (int n = 0; n < n_prts; n++) {
      auto prt = *it++;
      this->checkInPatchMod(prt.x);
      h_storage.store(prt, off + n);
    }

    off += n_prts;
  }
  this->n_prts = off;

  thrust::copy(h_storage.xi4.begin(), h_storage.xi4.end(),
               this->storage.xi4.begin());
  thrust::copy(h_storage.pxi4.begin(), h_storage.pxi4.end(),
               this->storage.pxi4.begin());
  thrust::copy(h_off.begin(), h_off.end(), this->by_block_.d_off.begin());
}

// ----------------------------------------------------------------------
// inject

template <typename BS>
void cuda_mparticles<BS>::inject(const std::vector<Particle>& buf,
                                 const std::vector<uint>& buf_n_by_patch)
{
  if (this->n_prts == 0) {
    // if there are no particles yet, we basically just initialize from the
    // buffer
    inject_initial(buf, buf_n_by_patch);
    setup_internals();
    return;
  }

  using Double3 = Vec3<double>;

  uint buf_n = 0;
  for (int p = 0; p < this->n_patches(); p++) {
    buf_n += buf_n_by_patch[p];
    //    printf("p %d buf_n_by_patch %d\n", p, buf_n_by_patch[p]);
  }
  //  printf("buf_n %d\n", buf_n);

  HMparticlesCudaStorage h_storage(buf_n);
  thrust::host_vector<uint> h_bidx(buf_n);
  // thrust::host_vector<uint> h_id(buf_n);

  uint off = 0;
  for (int p = 0; p < this->n_patches(); p++) {
    for (int n = 0; n < buf_n_by_patch[p]; n++) {
      auto prt = buf[off + n];
      h_storage.store(prt, off + n);
      auto bidx = this->blockIndex(prt, p);
      assert(bidx >= 0 && bidx < this->n_blocks);
      h_bidx[off + n] = bidx;
      ;
      // h_id[off + n] = this->n_prts + off + n;
    }
    off += buf_n_by_patch[p];
  }
  assert(off == buf_n);

  if (need_reorder) {
    reorder();
  }

  // assert(check_in_patch_unordered_slow());

  this->by_block_.find_indices_ids(*this);
  // assert(check_bidx_id_unordered_slow());

  resize(this->n_prts + buf_n);

  thrust::copy(h_storage.xi4.begin(), h_storage.xi4.end(),
               this->storage.xi4.begin() + this->n_prts);
  thrust::copy(h_storage.pxi4.begin(), h_storage.pxi4.end(),
               this->storage.pxi4.begin() + this->n_prts);
  thrust::copy(h_bidx.begin(), h_bidx.end(),
               this->by_block_.d_idx.begin() + this->n_prts);
  // thrust::copy(h_id.begin(), h_id.end(), d_id + n_prts);
  // FIXME, looks like ids up until n_prts have already been set above
  thrust::sequence(this->by_block_.d_id.data(),
                   this->by_block_.d_id.data() + this->n_prts + buf_n);

  // for (int i = -5; i <= 5; i++) {
  //   //    float4 xi4 = d_xi4[cmprts->n_prts + i];
  //   uint bidx = d_bidx[cmprts->n_prts + i];
  //   uint id = d_id[cmprts->n_prts + i];
  //   printf("i %d bidx %d %d\n", i, bidx, id);
  // }

  // assert(check_ordered());

  this->n_prts += buf_n;

  this->by_block_.stable_sort();

  this->by_block_.reorder_and_offsets(*this);

  // assert(check_ordered());
}

// ----------------------------------------------------------------------
// get_particles

template <typename BS>
std::vector<typename cuda_mparticles<BS>::Particle>
cuda_mparticles<BS>::get_particles(int beg, int end)
{
  int n_prts = end - beg;
  std::vector<Particle> prts;
  prts.reserve(n_prts);

  reorder(); // FIXME? by means of this, this function disturbs the state...

  thrust::host_vector<float4> xi4(&this->storage.xi4[beg],
                                  &this->storage.xi4[end]);
  thrust::host_vector<float4> pxi4(&this->storage.pxi4[beg],
                                   &this->storage.pxi4[end]);

  for (int n = 0; n < n_prts; n++) {
    int kind = cuda_float_as_int(xi4[n].w);
    prts.emplace_back(Real3{xi4[n].x, xi4[n].y, xi4[n].z},
                      Real3{pxi4[n].x, pxi4[n].y, pxi4[n].z}, pxi4[n].w, kind,
                      psc::particle::Id(), psc::particle::Tag());

#if 0
    uint b = blockIndex(xi4[n], p);
    assert(b < n_blocks);
#endif
  }

  return prts;
}

// ----------------------------------------------------------------------
// get_particles

template <typename BS>
std::vector<uint> cuda_mparticles<BS>::get_offsets() const
{
  thrust::host_vector<uint> h_off(this->by_block_.d_off);
  std::vector<uint> off(this->n_patches() + 1);
  for (int p = 0; p <= this->n_patches(); p++) {
    off[p] = h_off[p * this->n_blocks_per_patch];
  }
  return off;
}

// ----------------------------------------------------------------------
// get_particles

template <typename BS>
std::vector<typename cuda_mparticles<BS>::Particle>
cuda_mparticles<BS>::get_particles()
{
  return get_particles(0, this->n_prts);
}

// ----------------------------------------------------------------------
// get_particles

template <typename BS>
std::vector<typename cuda_mparticles<BS>::Particle>
cuda_mparticles<BS>::get_particles(int p)
{
  // FIXME, doing the copy here all the time would be nice to avoid
  // making sure we actually have a valid d_off would't hurt, either
  thrust::host_vector<uint> h_off(this->by_block_.d_off);

  uint beg = h_off[p * this->n_blocks_per_patch];
  uint end = h_off[(p + 1) * this->n_blocks_per_patch];

  return get_particles(beg, end);
}

// ----------------------------------------------------------------------
// get_particle

template <typename BS>
typename cuda_mparticles<BS>::Particle cuda_mparticles<BS>::get_particle(int p,
                                                                         int n)
{
  auto off = this->by_block_.d_off[p * this->n_blocks_per_patch];
  auto cprts = get_particles(off + n, off + n + 1);
  return cprts[0];
}

#include "cuda_mparticles_gold.cu"
#include "cuda_mparticles_checks.cu"

template struct cuda_mparticles<BS144>;
template struct cuda_mparticles<BS444>;
