
#pragma once

#include "fields3d.inl"
#include "grid.hxx"
#include "grid.inl"
#include "particles_simple.inl"
#include <kg/io.h>

// ----------------------------------------------------------------------
// write_checkpoint
//

template <typename Mparticles, typename MfieldsState>
void write_checkpoint(const Grid_t& grid, Mparticles& mprts,
                      MfieldsState& mflds)
{
  mpi_printf(grid.comm(), "**** Writing checkpoint...\n");
#if defined(PSC_HAVE_ADIOS2) && !defined(VPIC)
  MPI_Barrier(grid.comm()); // not really necessary

  std::string filename =
    "checkpoint_" + std::to_string(grid.timestep()) + ".bp";

  auto io = kg::io::IOAdios2{};
  auto writer = io.open(filename, kg::io::Mode::Write);
  writer.put("grid", grid);
  writer.put("mprts", mprts);
  writer.put("mflds", mflds);
  writer.close();
#else
  std::cerr << "write_checkpoint not available without adios2" << std::endl;
  std::abort();
#endif
}

// ----------------------------------------------------------------------
// read_checkpoint
//

template <typename Mparticles, typename MfieldsState>
inline void read_checkpoint(const std::string& filename, Grid_t& grid,
                            Mparticles& mprts, MfieldsState& mflds)
{
  mpi_printf(grid.comm(), "**** Reading checkpoint...\n");
  MPI_Barrier(grid.comm()); // not really necessary

#ifdef PSC_HAVE_ADIOS2
  auto io = kg::io::IOAdios2{};
  auto reader = io.open(filename, kg::io::Mode::Read);
  reader.get("grid", grid);
  mprts.~Mparticles();
  mflds.~MfieldsState();
  new (&mprts) Mparticles(grid);
  new (&mflds) MfieldsState(grid);
  reader.get("mprts", mprts);
  reader.get("mflds", mflds);
  reader.close();

  // FIXME, when we read back a rebalanced grid, other existing fields will
  // still have their own parallel distribution, ie, things will go wrong
#else
  std::cerr << "write_checkpoint not available without adios2" << std::endl;
  std::abort();
#endif
}

// ======================================================================
// Checkpointing
//
// This class is responsible for controling checkpointing -- it is called at
// the beginning of every timestep and decides when / how to write a
// checkpoint.

class Checkpointing
{
public:
  Checkpointing(int interval) : interval_{interval} {}

  // gets called every step, will checkpoint as required
  template <typename Mparticles, typename MfieldsState>
  void operator()(const Grid_t& grid, Mparticles& mprts, MfieldsState& mflds)
  {
    if (interval_ <= 0) {
      return;
    }

    // don't write a checkpoint immediately after start-up (in particular, not
    // immediately after just having restarted from a checkpoint)
    if (first_time_) {
      first_time_ = false;
      return;
    }

    static int pr_ckp;
    if (!pr_ckp) {
      pr_ckp = prof_register("checkpoint_write", 1., 0, 0);
    }

    if (grid.timestep() % interval_ == 0) {
      prof_start(pr_ckp);
      write_checkpoint(grid, mprts, mflds);
      prof_stop(pr_ckp);
    }
  }

  // gets called after the timeloop is done, should checkpoint
  // regardless of timestep (unless checkpointing is disabled)
  template <typename Mparticles, typename MfieldsState>
  void final(const Grid_t& grid, Mparticles& mprts, MfieldsState& mflds)
  {
    if (interval_ <= 0) {
      return;
    }

    write_checkpoint(grid, mprts, mflds);
  }

private:
  int interval_; // write checkpoint every so many steps
  bool first_time_ = true;
};
