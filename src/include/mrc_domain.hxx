
#pragma once

#include <mrc_domain.h>

struct MrcDomain
{
  MrcDomain(mrc_domain *domain = {}) : domain_{domain} {}

  MPI_Comm comm() const { return mrc_domain_comm(domain_); }
  void view() const { mrc_domain_view(domain_); }

  int nGlobalPatches() const { int n_global_patches; mrc_domain_get_nr_global_patches(domain_, &n_global_patches); return n_global_patches; }
  int nPatches() const { int n_patches; mrc_domain_get_patches(domain_, &n_patches); return n_patches; }
  mrc_patch_info globalPatchInfo(int p) const { mrc_patch_info info; mrc_domain_get_global_patch_info(domain_, p, &info); return info; }
  mrc_patch_info localPatchInfo(int p) const { mrc_patch_info info; mrc_domain_get_local_patch_info(domain_, p, &info); return info; }
  mrc_patch_info levelIdx3PatchInfo(int level, int idx[3]) const { mrc_patch_info info; mrc_domain_get_level_idx3_patch_info(domain_, level, idx, &info); return info; }
  const mrc_patch* getPatches(int* n_patches) const { return mrc_domain_get_patches(domain_, n_patches); }
  void neighborRankPatch(int p, int dir[3], int* nei_rank, int* nei_patch) const { mrc_domain_get_neighbor_rank_patch(domain_, p, dir, nei_rank, nei_patch); }

  mrc_fld* m3_create() const { return mrc_domain_m3_create(domain_); }
  mrc_ddc* create_ddc() const { return mrc_domain_create_ddc(domain_); }
  
  //private:
  mrc_domain* domain_;
};
