
#pragma once

#include <mrc_io.h>

class WriterMRC
{
public:
  WriterMRC() : io_(nullptr, &mrc_io_destroy) {}

  explicit operator bool() const { return static_cast<bool>(io_); }

  void open(const std::string& pfx, const std::string& dir = ".")
  {
    assert(!io_);
    io_.reset(mrc_io_create(MPI_COMM_WORLD));
    mrc_io_set_param_string(io_.get(), "basename", pfx.c_str());
    mrc_io_set_param_string(io_.get(), "outdir", dir.c_str());
    mrc_io_set_from_options(io_.get());
    mrc_io_setup(io_.get());
    mrc_io_view(io_.get());
  }

  void close()
  {
    assert(io_);
    io_.reset();
  }

  void begin_step(const Grid_t& grid)
  {
    mrc_io_open(io_.get(), "w", grid.timestep(), grid.timestep() * grid.dt);

    // save some basic info about the run in the output file
    struct mrc_obj* obj = mrc_obj_create(mrc_io_comm(io_.get()));
    mrc_obj_set_name(obj, "psc");
    mrc_obj_dict_add_int(obj, "timestep", grid.timestep());
    mrc_obj_dict_add_float(obj, "time", grid.timestep() * grid.dt);
    mrc_obj_dict_add_float(obj, "cc", grid.norm.cc);
    mrc_obj_dict_add_float(obj, "dt", grid.dt);
    mrc_obj_write(obj, io_.get());
    mrc_obj_destroy(obj);
  }

  void begin_step(int step, double time)
  {
    mrc_io_open(io_.get(), "w", step, time);
  }

  void set_subset(const Grid_t& grid, Int3 rn, Int3 rx)
  {
    if (strcmp(mrc_io_type(io_.get()), "xdmf_collective") == 0) {
      auto gdims = grid.domain.gdims;
      int slab_off[3], slab_dims[3];
      for (int d = 0; d < 3; d++) {
        if (rx[d] > gdims[d])
          rx[d] = gdims[d];

        slab_off[d] = rn[d];
        slab_dims[d] = rx[d] - rn[d];
      }

      mrc_io_set_param_int3(io_.get(), "slab_off", slab_off);
      mrc_io_set_param_int3(io_.get(), "slab_dims", slab_dims);
    }
  }

  void end_step() { mrc_io_close(io_.get()); }

  template <typename Mfields>
  void write(const Mfields& _mflds, const Grid_t& grid, const std::string& name,
             const std::vector<std::string>& comp_names)
  {
    auto&& eval_mflds = evalMfields(_mflds);
    auto& mflds = const_cast<MfieldsC&>(eval_mflds);

    int n_comps = comp_names.size();
    // FIXME, should generally equal the # of component in mflds,
    // but this way allows us to write fewer components, useful to hack around
    // 16-bit vpic material ids, stored together as AOS with floats...

    mrc_fld* fld = grid.mrc_domain().m3_create();
    mrc_fld_set_name(fld, name.c_str());
    mrc_fld_set_param_int(fld, "nr_ghosts", 0);
    mrc_fld_set_param_int(fld, "nr_comps", n_comps);
    mrc_fld_setup(fld);
    for (int m = 0; m < n_comps; m++) {
      mrc_fld_set_comp_name(fld, m, comp_names[m].c_str());
    }

    for (int p = 0; p < mflds.n_patches(); p++) {
      mrc_fld_patch* m3p = mrc_fld_patch_get(fld, p);
      mrc_fld_foreach(fld, i, j, k, 0, 0)
      {
        for (int m = 0; m < n_comps; m++) {
          MRC_M3(m3p, m, i, j, k) = mflds[p](m, i, j, k);
        }
      }
      mrc_fld_foreach_end;
      mrc_fld_patch_put(fld);
    }

    mrc_fld_write(fld, io_.get());
    mrc_fld_destroy(fld);
  }

private:
  std::unique_ptr<struct mrc_io, decltype(&mrc_io_close)> io_;
};
