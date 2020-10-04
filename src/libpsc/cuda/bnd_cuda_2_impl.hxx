
#pragma once

#include "psc.h"
#include "fields.hxx"
#include "bnd.hxx"

#include <mrc_profile.h>
#include <mrc_ddc.h>

template <typename MF>
struct BndCuda2 : BndBase
{
  using Mfields = MF;
  using real_t = typename Mfields::real_t;

  // ----------------------------------------------------------------------
  // ctor

  BndCuda2(const Grid_t& grid, Int3 ibn)
  {
    static struct mrc_ddc_funcs ddc_funcs = {
      .copy_to_buf = copy_to_buf,
      .copy_from_buf = copy_from_buf,
      .add_from_buf = add_from_buf,
    };

    ddc_ = grid.mrc_domain().create_ddc();
    mrc_ddc_set_funcs(ddc_, &ddc_funcs);
    mrc_ddc_set_param_int3(ddc_, "ibn", ibn);
    mrc_ddc_set_param_int(ddc_, "max_n_fields", 24);
    mrc_ddc_set_param_int(ddc_, "size_of_type", sizeof(real_t));
    mrc_ddc_setup(ddc_);
    balance_generation_cnt_ = psc_balance_generation_cnt;
  }

  // ----------------------------------------------------------------------
  // dtor

  ~BndCuda2() { mrc_ddc_destroy(ddc_); }

  // ----------------------------------------------------------------------
  // reset

  void reset(const Grid_t& grid)
  {
    // FIXME, not really a pretty way of doing this
    this->~BndCuda2();
    new (this) BndCuda2(grid, grid.ibn);
  }

  // ----------------------------------------------------------------------
  // add_ghosts

  void add_ghosts(Mfields& mflds, int mb, int me)
  {
    if (psc_balance_generation_cnt != balance_generation_cnt_) {
      balance_generation_cnt_ = psc_balance_generation_cnt;
      reset(mflds.grid());
    }
    auto h_mflds = hostMirror(mflds);
    copy(mflds, h_mflds);
    mrc_ddc_add_ghosts(ddc_, mb, me, &h_mflds);
    copy(h_mflds, mflds);
  }

  // ----------------------------------------------------------------------
  // fill_ghosts

  void fill_ghosts(Mfields& mflds, int mb, int me)
  {
    if (psc_balance_generation_cnt != balance_generation_cnt_) {
      balance_generation_cnt_ = psc_balance_generation_cnt;
      reset(mflds.grid());
    }
    // FIXME
    // I don't think we need as many points, and only stencil star
    // rather then box
    auto h_mflds = hostMirror(mflds);
    copy(mflds, h_mflds);
    mrc_ddc_fill_ghosts(ddc_, mb, me, &h_mflds);
    copy(h_mflds, mflds);
  }

  // ----------------------------------------------------------------------
  // copy_to_buf

  static void copy_to_buf(int mb, int me, int p, int ilo[3], int ihi[3],
                          void* _buf, void* ctx)
  {
    auto& mf = *static_cast<HMFields*>(ctx);
    auto F = mf[p];
    real_t* buf = static_cast<real_t*>(_buf);

    for (int m = mb; m < me; m++) {
      for (int iz = ilo[2]; iz < ihi[2]; iz++) {
        for (int iy = ilo[1]; iy < ihi[1]; iy++) {
          for (int ix = ilo[0]; ix < ihi[0]; ix++) {
            MRC_DDC_BUF3(buf, m - mb, ix, iy, iz) = F(m, ix, iy, iz);
          }
        }
      }
    }
  }

  static void add_from_buf(int mb, int me, int p, int ilo[3], int ihi[3],
                           void* _buf, void* ctx)
  {
    auto& mf = *static_cast<HMFields*>(ctx);
    auto F = mf[p];
    real_t* buf = static_cast<real_t*>(_buf);

    for (int m = mb; m < me; m++) {
      for (int iz = ilo[2]; iz < ihi[2]; iz++) {
        for (int iy = ilo[1]; iy < ihi[1]; iy++) {
          for (int ix = ilo[0]; ix < ihi[0]; ix++) {
            real_t val =
              F(m, ix, iy, iz) + MRC_DDC_BUF3(buf, m - mb, ix, iy, iz);
            F(m, ix, iy, iz) = val;
          }
        }
      }
    }
  }

  static void copy_from_buf(int mb, int me, int p, int ilo[3], int ihi[3],
                            void* _buf, void* ctx)
  {
    auto& mf = *static_cast<HMFields*>(ctx);
    auto F = mf[p];
    real_t* buf = static_cast<real_t*>(_buf);

    for (int m = mb; m < me; m++) {
      for (int iz = ilo[2]; iz < ihi[2]; iz++) {
        for (int iy = ilo[1]; iy < ihi[1]; iy++) {
          for (int ix = ilo[0]; ix < ihi[0]; ix++) {
            F(m, ix, iy, iz) = MRC_DDC_BUF3(buf, m - mb, ix, iy, iz);
          }
        }
      }
    }
  }

private:
  mrc_ddc* ddc_;
  int balance_generation_cnt_;
};
