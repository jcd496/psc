
#ifndef PSC_FIELDS_CUDA_H
#define PSC_FIELDS_CUDA_H

#include <mpi.h>
#include "fields3d.hxx"
#include "fields_traits.hxx"
#include "cuda_mfields.h"

#include "psc_fields_single.h"

#include "mrc_json.h"

// ======================================================================
// MfieldsCuda

struct MfieldsCuda : MfieldsBase
{
  using real_t = float;
  using Real = real_t;

  MfieldsCuda(const Grid_t& grid, int n_fields, Int3 ibn)
    : MfieldsBase{grid, n_fields, ibn},
      cmflds_{new cuda_mfields(grid, n_fields, ibn)}
  {}

  cuda_mfields* cmflds() { return cmflds_.get(); }
  const cuda_mfields* cmflds() const { return cmflds_.get(); }

  int n_comps() const { return _n_comps(); }
  int n_patches() const { return cmflds_->n_patches(); };
  const Grid_t& grid() const { return _grid(); }

  void reset(const Grid_t& new_grid) override
  {
    *this = MfieldsCuda(new_grid, n_comps(), ibn());
  }

  gt::gtensor_span_device<real_t, 5> gt()
  {
    return gt::adapt_device(cmflds_->storage().data().get(),
                            cmflds_->storage().shape());
  }

  gt::gtensor_span_device<real_t, 5> gt() const
  {
    return gt::adapt_device(cmflds_->storage().data().get(),
                            cmflds_->storage().shape());
  }

  static const Convert convert_to_, convert_from_;
  const Convert& convert_to() override { return convert_to_; }
  const Convert& convert_from() override { return convert_from_; }

  std::unique_ptr<cuda_mfields> cmflds_;
};

inline MfieldsSingle hostMirror(MfieldsCuda& mflds)
{
  return MfieldsSingle{mflds.grid(), mflds.n_comps(), mflds.ibn()};
}

inline MfieldsSingle hostMirror(const MfieldsCuda& mflds)
{
  return MfieldsSingle{mflds.grid(), mflds.n_comps(), mflds.ibn()};
}

inline void copy(const MfieldsCuda& mflds, MfieldsSingle& hmflds)
{
  static int pr;
  if (!pr) {
    pr = prof_register("mflds to host", 1., 0, 0);
  }
  prof_start(pr);
  gt::copy(mflds.gt(), hmflds.storage());
  prof_stop(pr);
}

inline void copy(const MfieldsSingle& hmflds, MfieldsCuda& mflds)
{
  static int pr;
  if (!pr) {
    pr = prof_register("mflds from host", 1., 0, 0);
  }
  prof_start(pr);
  // gt::copy(hmflds.storage(), mflds.gt());
  thrust::copy(hmflds.storage().data(),
               hmflds.storage().data() + hmflds.storage().size(),
               mflds.gt().data());
  prof_stop(pr);
}

// ======================================================================
// MfieldsStateCuda

struct MfieldsStateCuda : MfieldsStateBase
{
  using real_t = MfieldsCuda::real_t;
  using space = gt::space::device;

  MfieldsStateCuda(const Grid_t& grid)
    : MfieldsStateBase{grid, NR_FIELDS, grid.ibn},
      mflds_{grid, NR_FIELDS, grid.ibn}
  {}

  /* void reset(const Grid_t& new_grid) override */
  /* { */
  /*   MfieldsStateBase::reset(new_grid); */
  /*   mflds_.reset(new_grid); */
  /* } */

  cuda_mfields* cmflds() { return mflds_.cmflds(); }

  int n_patches() const { return mflds_.n_patches(); }
  int n_comps() const { return mflds_.n_comps(); }

  const Grid_t& grid() const { return *grid_; }

  static const Convert convert_to_, convert_from_;
  const Convert& convert_to() override { return convert_to_; }
  const Convert& convert_from() override { return convert_from_; }

  MfieldsCuda& mflds() { return mflds_; }
  const MfieldsCuda& mflds() const { return mflds_; }

  auto gt() { return mflds_.gt(); }

private:
  MfieldsCuda mflds_;
};

template <>
struct Mfields_traits<MfieldsCuda>
{
  static constexpr const char* name = "cuda";
};

namespace detail
{
template <typename T>
struct Mfields_from_type_space<T, gt::space::device>
{
  static_assert(std::is_same<T, float>::value, "CUDA only supports float");
  using type = MfieldsCuda;
};
} // namespace detail

inline MfieldsSingle hostMirror(MfieldsStateCuda& mflds)
{
  return hostMirror(mflds.mflds());
}

inline MfieldsSingle hostMirror(const MfieldsStateCuda& mflds)
{
  return hostMirror(mflds.mflds());
}

inline void copy(const MfieldsStateCuda& mflds, MfieldsSingle& hmflds)
{
  copy(mflds.mflds(), hmflds);
}

inline void copy(const MfieldsSingle& hmflds, MfieldsStateCuda& mflds)
{
  copy(hmflds, mflds.mflds());
}

// ----------------------------------------------------------------------
// FIXME hacky workaround for lack of gt::view on device

template <typename E>
GT_INLINE auto view_patch(E&& gt, int p)
{
  return gt::adapt_device<4>(
    (&gt(0, 0, 0, 0, p)).get(),
    {gt.shape(0), gt.shape(1), gt.shape(2), gt.shape(3)});
}

#endif
