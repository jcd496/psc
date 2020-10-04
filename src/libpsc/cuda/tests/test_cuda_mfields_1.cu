
#include "cuda_mfields.h"

#include "fields.hxx"

#include <cmath>

enum
{ // FIXME, duplicated
  JXI,
  JYI,
  JZI,
  EX,
  EY,
  EZ,
  HX,
  HY,
  HZ,
  NR_FIELDS,
};

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <mrc_profile.h>

struct prof_globals prof_globals; // FIXME

int prof_register(const char* name, float simd, int flops, int bytes)
{
  return 0;
}

// ----------------------------------------------------------------------
// init_wave

double init_wave(double x, double y, int m)
{
  double kx = 2. * M_PI / 80., ky = 2. * M_PI / 80.;
  switch (m) {
    case EX: return 1. / sqrtf(2.) * sin(kx * x + ky * y);
    case EY: return -1. / sqrtf(2.) * sin(kx * x + ky * y);
    case HZ: return sin(kx * x + ky * y);
    default: return 0.;
  }
}

// ----------------------------------------------------------------------
// main

int main(void)
{
  using fields_t = fields3d<float>;
  using real_t = fields_t::real_t;

  Grid_t grid{};
  grid.ldims = {1, 8, 8};
  grid.dx = {1., 10., 10.};

  Grid_t::Patch patch{};
  patch.xb = {0., 0., 0.};
  grid.patches.push_back(patch);

  int n_fields = 9;
  Int3 ibn = {0, 2, 2};
  struct cuda_mfields* cmflds = new cuda_mfields(grid, n_fields, ibn);

  int n_patches = cmflds->n_patches;
  const Vec3<double>& dx = grid.dx;

  for (int m = 0; m < n_fields; m++) {
    cmflds->zero_comp_yz(m);
  }

  cmflds->dump("cmflds.json");

  auto mflds = hostMirror(*cmflds);
  auto flds = mflds[0];
  for (int p = 0; p < n_patches; p++) {
    for (int k = flds.ib[2]; k < flds.ib[2] + flds.im[2]; k++) {
      for (int j = flds.ib[1]; j < flds.ib[1] + flds.im[1]; j++) {
        for (int i = flds.ib[0]; i < flds.ib[0] + flds.im[0]; i++) {
          real_t x_cc = (i + .5) * dx[0];
          real_t y_cc = (j + .5) * dx[1];
          real_t x_nc = i * dx[0];
          real_t y_nc = j * dx[1];
          flds(EX, i, j, k) = init_wave(x_cc, y_nc, EX);
          flds(EY, i, j, k) = init_wave(x_nc, y_cc, EY);
          flds(EZ, i, j, k) = init_wave(x_nc, y_nc, EZ);
          flds(HX, i, j, k) = init_wave(x_nc, y_cc, HX);
          flds(HY, i, j, k) = init_wave(x_cc, y_nc, HY);
          flds(HZ, i, j, k) = init_wave(x_cc, y_cc, HZ);
        }
      }
    }
  }
  copy(mflds, *cmflds);

  cmflds->dump("cmflds_wave.json");

  float dt = dx[1];
  cuda_push_fields_E_yz(cmflds, .5 * dt);
  cuda_push_fields_H_yz(cmflds, dt);
  cuda_push_fields_E_yz(cmflds, .5 * dt);

  cmflds->dump("cmflds_wave_1.json");

  delete cmflds;
}
