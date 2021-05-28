
#include "fields3d.hxx"
#include "psc_fields_cuda.h"

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>

#include <fstream>

std::size_t mem_mfields()
{
  std::size_t mem = 0;
  for (auto mflds : MfieldsBase::instances) {
    auto mflds_cuda = dynamic_cast<MfieldsCuda*>(mflds);
    if (mflds_cuda) {
      auto dims = mflds->_grid().ldims + 2 * mflds->ibn();
      auto n_patches = mflds->_grid().n_patches();
      std::size_t bytes = sizeof(float) * dims[0] * dims[1] * dims[2] *
                          n_patches * mflds->_n_comps();
      // of << "===== MfieldsCuda # of components " << mflds->_n_comps()
      //           << " bytes " << bytes << "\n";
      mem += bytes;
    } else {
      // of << "===== MfieldsBase # of components " << mflds->_n_comps()
      //           << "\n";
    }
  }
  return mem;
}

std::size_t mem_cuda_allocated()
{
  using pool = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  using track = rmm::mr::tracking_resource_adaptor<pool>;
  using log = rmm::mr::logging_resource_adaptor<track>;
  auto mr = rmm::mr::get_current_device_resource();
  auto log_mr = dynamic_cast<log*>(mr);
  assert(log_mr);
  auto track_mr = log_mr->get_upstream();
  return track_mr->get_allocated_bytes();
}

void mem_stats(std::string file, int line, std::ostream& of)
{
  std::size_t mem_fields = mem_mfields();

  std::size_t total = mem_fields + mem_particles + mem_collisions + mem_sort +
                      mem_sort_by_block + mem_bnd + mem_heating + mem_bndp;

  std::size_t allocated = mem_cuda_allocated();

  of << "===== MEM " << file << ":" << line << "\n";
  of << "===== fields     " << mem_fields << " bytes  # "
     << MfieldsBase::instances.size() << "\n";
  of << "===== particles  " << mem_particles << " bytes\n";
  of << "===== collisions " << mem_collisions << " bytes\n";
  of << "===== sort       " << mem_sort << " bytes\n";
  of << "===== sort_block " << mem_sort_by_block << " bytes\n";
  of << "===== bnd        " << mem_bnd << " bytes\n";
  of << "===== heating    " << mem_heating << " bytes\n";
  of << "===== alloced " << allocated << " total " << total << " unaccounted "
     << std::ptrdiff_t(allocated - total) << "\n";
}

void mem_stats_csv(std::ostream& of, int timestep, int n_patches, int n_prts)
{
  std::size_t mem_fields = mem_mfields();

  std::size_t total = mem_fields + mem_particles + mem_collisions + mem_sort +
                      mem_sort_by_block + mem_bnd + mem_heating + mem_bndp;

  std::size_t allocated = mem_cuda_allocated();

  of << timestep << ","
     << n_patches << ","
     << n_prts << ","
     << mem_fields << ","
     << MfieldsBase::instances.size() << ","
     << mem_particles << ","
     << mem_collisions << ","
     << mem_sort << ","
     << mem_sort_by_block << ","
     << mem_bnd << ","
     << mem_heating << ","
     << allocated << ","
     << total << ","
     << std::ptrdiff_t(allocated - total) << ","
     << "\n";
}
