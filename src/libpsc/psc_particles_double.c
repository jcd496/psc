
#include "psc.h"
#include "psc_particles_double.h"
#include "psc_particles_c.h"

#include <mrc_io.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

static void *
_psc_mparticles_double_alloc_patch(int p, int n_part, unsigned int flags)
{
  particles_double_t *pp = calloc(1, sizeof(*pp));
  pp->n_part = n_part;
  pp->n_alloced = n_part * 1.2;
  pp->particles = calloc(pp->n_alloced, sizeof(*pp->particles));
  return pp;
}

static void
_psc_mparticles_double_free_patch(int p, void *_pp)
{
  particles_double_t *pp = _pp;
  free(pp->particles);
  pp->n_alloced = 0;
  pp->particles = NULL;
  free(pp);
}

void
particles_double_realloc(particles_double_t *pp, int new_n_part)
{
  if (new_n_part <= pp->n_alloced)
    return;

  pp->n_alloced = new_n_part * 1.2;
  pp->particles = realloc(pp->particles, pp->n_alloced * sizeof(*pp->particles));
}

static int
_psc_mparticles_double_nr_particles_by_patch(mparticles_double_t *mparticles, int p)
{
  return psc_mparticles_get_patch_double(mparticles, p)->n_part;
}

static inline void
calc_vxi(particle_double_real_t vxi[3], particle_double_t *part)
{
  particle_double_real_t root =
    1.f / sqrtf(1.f + sqr(part->pxi) + sqr(part->pyi) + sqr(part->pzi));
  vxi[0] = part->pxi * root;
  vxi[1] = part->pyi * root;
  vxi[2] = part->pzi * root;
}

static void
_psc_mparticles_double_copy_to_c(int p, struct psc_mparticles *particles_base,
				 mparticles_c_t *particles, unsigned int flags)
{
  particle_double_real_t dth[3] = { .5 * ppsc->dt, .5 * ppsc->dt, .5 * ppsc->dt };
  // don't shift in invariant directions
  for (int d = 0; d < 3; d++) {
    if (ppsc->domain.gdims[d] == 1) {
      dth[d] = 0.;
    }
  }

  struct psc_patch *patch = ppsc->patch + p;
  particles_double_t *pp_base = psc_mparticles_get_patch_double(particles_base, p);
  particles_c_t *pp = psc_mparticles_get_patch_c(particles, p);
  pp->n_part = pp_base->n_part;
  assert(pp->n_part <= pp->n_alloced);
  for (int n = 0; n < pp_base->n_part; n++) {
    particle_double_t *part_base = particles_double_get_one(pp_base, n);
    particle_c_t *part = particles_c_get_one(pp, n);
    
    particle_c_real_t qni = ppsc->kinds[part_base->kind].q;
    particle_c_real_t mni = ppsc->kinds[part_base->kind].m;
    particle_c_real_t wni = part_base->qni_wni / qni;
    
    particle_double_real_t vxi[3];
    calc_vxi(vxi, part_base);
    part->xi  = part_base->xi - dth[0] * vxi[0] + patch->xb[0];
    part->yi  = part_base->yi - dth[1] * vxi[1] + patch->xb[1];
    part->zi  = part_base->zi - dth[2] * vxi[2] + patch->xb[2];
    part->pxi = part_base->pxi;
    part->pyi = part_base->pyi;
    part->pzi = part_base->pzi;
    part->qni = qni;
    part->mni = mni;
    part->wni = wni;
    part->kind = part->kind;
  }
}

static void
_psc_mparticles_double_copy_from_c(int p, struct psc_mparticles *particles_base,
				   mparticles_c_t *particles, unsigned int flags)
{
  particle_double_real_t dth[3] = { .5 * ppsc->dt, .5 * ppsc->dt, .5 * ppsc->dt };
  // don't shift in invariant directions
  for (int d = 0; d < 3; d++) {
    if (ppsc->domain.gdims[d] == 1) {
      dth[d] = 0.;
    }
  }

  struct psc_patch *patch = ppsc->patch + p;
  particles_double_t *pp_base = psc_mparticles_get_patch_double(particles_base, p);
  particles_c_t *pp = psc_mparticles_get_patch_c(particles, p);
  pp_base->n_part = pp->n_part;
  assert(pp_base->n_part <= pp_base->n_alloced);
  for (int n = 0; n < pp_base->n_part; n++) {
    particle_double_t *part_base = particles_double_get_one(pp_base, n);
    particle_c_t *part = particles_c_get_one(pp, n);
    
    particle_double_real_t qni_wni;
    if (part->qni != 0.) {
      qni_wni = part->qni * part->wni;
    } else {
	qni_wni = part->wni;
    }
    
    part_base->xi          = part->xi - patch->xb[0];
    part_base->yi          = part->yi - patch->xb[1];
    part_base->zi          = part->zi - patch->xb[2];
    part_base->pxi         = part->pxi;
    part_base->pyi         = part->pyi;
    part_base->pzi         = part->pzi;
    part_base->qni_wni     = qni_wni;
    part_base->kind        = part->kind;

    particle_double_real_t vxi[3];
    calc_vxi(vxi, part_base);
    part_base->xi += dth[0] * vxi[0];
    part_base->yi += dth[1] * vxi[1];
    part_base->zi += dth[2] * vxi[2];
  }
}

// ======================================================================
// psc_mparticles: subclass "double"
  
static struct mrc_obj_method _psc_mparticles_double_methods[] = {
  MRC_OBJ_METHOD("copy_to_c",         _psc_mparticles_double_copy_to_c),
  MRC_OBJ_METHOD("copy_from_c",       _psc_mparticles_double_copy_from_c),
  {}
};

struct psc_mparticles_ops psc_mparticles_double_ops = {
  .name                    = "double",
  .methods                 = _psc_mparticles_double_methods,
  .nr_particles_by_patch   = _psc_mparticles_double_nr_particles_by_patch,
  .alloc_patch             = _psc_mparticles_double_alloc_patch,
  .free_patch              = _psc_mparticles_double_free_patch,
  .size_of_particles_t     = sizeof(particles_double_t),
};
