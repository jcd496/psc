
#pragma once

#include "particles.hxx"
#include "psc_vpic_bits.h"

// ======================================================================
// InjectorVpic

template <typename Mparticles>
struct InjectorVpic
{
  struct Patch
  {
    Patch(Mparticles& mprts) : mprts_{mprts} {}

    void operator()(const psc::particle::Inject& prt)
    {
      const auto& vgrid = mprts_.vgrid();
      float dVi = 1.f / (vgrid.dx * vgrid.dy * vgrid.dz);
      psc::particle::Inject prt_reweighted = prt;
      prt_reweighted.w *= dVi;
      reweight(prt_reweighted);
    }

    void reweight(const psc::particle::Inject& prt)
    {
      mprts_.inject_particle_reweight(prt);
    }

  private:
    Mparticles& mprts_;
  };

  InjectorVpic(Mparticles& mprts) : mprts_{mprts} {}

  Patch operator[](int p) const { return {mprts_}; }

private:
  Mparticles& mprts_;
};

// ======================================================================
// ConstParticleAccessorVpic

template <typename Mparticles>
struct ConstParticleAccessorVpic
{
  using Particle = typename Mparticles::Particle;
  using real_t = typename Mparticles::real_t;
  using Real3 = Vec3<real_t>;
  using Double3 = Vec3<double>;
  using Species = typename Mparticles::Species;

  ConstParticleAccessorVpic(const Particle& prt, const Species& sp)
    : prt_{prt}, sp_{sp}
  {}

  Real3 u() const { return {prt_.ux, prt_.uy, prt_.uz}; }
  real_t w() const { return prt_.w * sp_.vgrid().dV; }
  real_t qni_wni() const { return w() * sp_.q; }
  int kind() const { return sp_.id; }

  Real3 x() const { return Real3(x_double()); }

  Double3 x_double() const
  {
    const auto& vgrid = sp_.vgrid();
    double x0 = vgrid.x0, y0 = vgrid.y0, z0 = vgrid.z0;
    double x1 = vgrid.x1, y1 = vgrid.y1, z1 = vgrid.z1;
    double nx = vgrid.nx, ny = vgrid.ny, nz = vgrid.nz;

    int i = prt_.i;
    int iz = i / ((nx + 2) * (ny + 2));
    i -= iz * ((nx + 2) * (ny + 2));
    int iy = i / (nx + 2);
    i -= iy * (nx + 2);
    int ix = i;

    // adjust to 0-based (no ghost)
    ix--;
    iy--;
    iz--;

    // back to physical coords
    Double3 x = {ix + .5 * (prt_.dx + 1.), iy + .5 * (prt_.dy + 1.),
                 iz + .5 * (prt_.dz + 1.)};
    x *= (Double3{x1, y1, z1} - Double3{x0, y0, z0}) / Double3{nx, ny, nz};

    return x;
  }

  Double3 position() const
  {
    const auto& vgrid = sp_.vgrid();
    double x0 = vgrid.x0, y0 = vgrid.y0, z0 = vgrid.z0;

    return Double3(x_double()) + Double3{x0, y0, z0};
  }

private:
  const Particle& prt_;
  const Species& sp_;
};

// ======================================================================
// ConstAccessorVpic

template <typename Mparticles>
struct ConstAccessorVpic
{
  using Particle = ConstParticleAccessorVpic<Mparticles>;

  struct Patch
  {
    using accessor = ConstParticleAccessorVpic<Mparticles>;
    using ConstSpeciesIterator = typename Mparticles::ConstSpeciesIterator;
    using Species = typename Mparticles::Species;

    struct const_iterator
      : std::iterator<std::forward_iterator_tag,
                      accessor,        // value type
                      ptrdiff_t,       // difference type
                      const accessor*, // pointer type
                      const accessor&> // reference type

    {
      const_iterator(const Patch& patch, ConstSpeciesIterator sp, uint n)
        : patch_{patch}, sp_{sp}, n_{n}
      {}

      bool operator==(const_iterator other) const
      {
        return sp_ == other.sp_ && n_ == other.n_;
      }
      bool operator!=(const_iterator other) const { return !(*this == other); }

      const_iterator& operator++()
      {
        n_++;
        if (n_ == sp_->np) {
          n_ = 0;
          ++sp_;
        }
        return *this;
      }

      const_iterator operator++(int)
      {
        auto retval = *this;
        ++(*this);
        return retval;
      }
      const accessor operator*() { return {sp_->p[n_], *sp_}; }

    private:
      const Patch patch_;
      ConstSpeciesIterator sp_;
      uint n_;
    };

    Patch(const ConstAccessorVpic& accessor) : accessor_{accessor} {}

    const_iterator begin() const
    {
      return {*this, accessor_.mprts_[0].begin(), 0};
    }
    const_iterator end() const { return {*this, accessor_.mprts_[0].end(), 0}; }
    uint size() const { return accessor_.size(0); }

    const accessor operator[](int n) const
    {
      for (auto& sp : accessor_.mprts_[0]) {
        if (n < sp.np) {
          return {sp.p[n], sp};
        }
        n -= sp.np;
      }
      std::abort();
    }

  private:
    const ConstAccessorVpic& accessor_;
  };

  ConstAccessorVpic(Mparticles& mprts) : mprts_{mprts} {}

  Patch operator[](int p) const { return {*this}; }
  uint size(int p) const { return mprts_.size(); }

private:
  Mparticles& mprts_;
};

// ======================================================================
// MparticlesVpic_

template <typename _Particles>
struct MparticlesVpic_
  : MparticlesBase
  , protected _Particles
{
  using Particles = _Particles;
  using Species = typename Particles::Species;
  using Grid = typename Particles::Grid;
  using Particle = typename Particles::Particle;
  using SpeciesIterator = typename Particles::iterator;
  using ConstSpeciesIterator = typename Particles::const_iterator;
  using ConstAccessor = ConstAccessorVpic<MparticlesVpic_>;
  using real_t = float;
  using Real3 = Vec3<real_t>;

  using Particles::empty;
  using Particles::getNumSpecies;
  using Particles::head;
  using Particles::inject_particle_reweight;
  using typename Particles::ParticleBcList;
  using typename Particles::ParticleMover;

  struct Patch
  {
    Patch(MparticlesVpic_& mprts) : mprts_{mprts} {}

    ConstSpeciesIterator cbegin() const { return mprts_.cbegin(); }
    ConstSpeciesIterator cend() const { return mprts_.cend(); }
    ConstSpeciesIterator begin() const { return mprts_.begin(); }
    ConstSpeciesIterator end() const { return mprts_.end(); }
    SpeciesIterator begin() { return mprts_.begin(); }
    SpeciesIterator end() { return mprts_.end(); }

  private:
    MparticlesVpic_& mprts_;
  };

  struct ConstPatch
  {
    ConstPatch(const MparticlesVpic_& mprts) : mprts_{mprts} {}

    ConstSpeciesIterator cbegin() const { return mprts_.cbegin(); }
    ConstSpeciesIterator cend() const { return mprts_.cend(); }
    ConstSpeciesIterator begin() const { return mprts_.begin(); }
    ConstSpeciesIterator end() const { return mprts_.end(); }

  private:
    const MparticlesVpic_& mprts_;
  };

  // ----------------------------------------------------------------------
  // ctor

  MparticlesVpic_(const Grid_t& grid, Grid* vgrid)
    : MparticlesBase(grid), vgrid_(vgrid)
  {
    assert(grid.n_patches() == 1);
  }

  // ----------------------------------------------------------------------
  // size

  int size() const override
  {
    int n_prts = 0;
    for (auto& sp : *this) {
      n_prts += sp.np;
    }

    return n_prts;
  }

  // ----------------------------------------------------------------------
  // sizeByPatch

  std::vector<uint> sizeByPatch() const override { return {uint(size())}; }

  // ----------------------------------------------------------------------
  // reserve_all
  //
  // This is a bit iffy, since we don't really want to reallocate stuff here,
  // at least for now, and we wouldn't be able to know how to split this into
  // the different species, anyway.

  void reserve_all(const std::vector<uint>& n_prts_by_patch)
  {
    for (int p = 0; p < n_patches(); p++) {
      int n_prts = 0, n_prts_alloced = 0;
      for (auto& sp : *this) {
        n_prts += sp.np;
        n_prts_alloced += sp.max_np;
      }
#if 0
      if (n_prts_by_patch[p] != n_prts) {
	mprintf("vpic_mparticles_reserve_all: %d (currently %d max %d)\n",
		n_prts_by_patch[p], n_prts, n_prts_alloced);
      }
#endif
      assert(n_prts_by_patch[p] <= n_prts_alloced);
    }
  }

  void reset()
  {
    for (auto& sp : *this) {
      sp.np = 0;
    }
  }

  using MparticlesBase::reset;

  void push_back(int kind, const Particle& prt)
  {
    for (auto& sp : *this) {
      if (sp.id == kind) {
        sp.p[sp.np++] = prt;
        return;
      }
    }
    mprintf("prt.kind %d not found in species list!\n", kind);
    assert(0);
  }

  Patch operator[](int p)
  {
    assert(p == 0);
    return {*this};
  }
  ConstPatch operator[](int p) const
  {
    assert(p == 0);
    return {*this};
  }

  InjectorVpic<MparticlesVpic_> injector() { return {*this}; }

  ConstAccessor accessor() { return {*this}; }

  Species* define_species(const char* name, double q, double m,
                          double max_local_np, double max_local_nm,
                          double sort_interval, double sort_out_of_place)
  {
    // Compute a reasonble number of movers if user did not specify
    // Based on the twice the number of particles expected to hit the boundary
    // of a wpdt=0.2 / dx=lambda species in a 3x3x3 domain
    if (max_local_nm < 0) {
      max_local_nm = 2 * max_local_np / 25;
#if 0
      // FIXME, don't know MAX_PIPELINE, and that's mostly gone
      // could move this down into Particles.create()
      if (max_local_nm < 16*(MAX_PIPELINE+1))
	max_local_nm = 16*(MAX_PIPELINE+1);
#endif
    }
    auto sp = this->create(name, q, m, max_local_np, max_local_nm,
                           sort_interval, sort_out_of_place, vgrid_);
    return this->append(sp);
  }

  static const Convert convert_to_, convert_from_;
  const Convert& convert_to() override { return convert_to_; }
  const Convert& convert_from() override { return convert_from_; }

  const Grid& vgrid() { return *vgrid_; }

private:
  Grid* vgrid_;
};
