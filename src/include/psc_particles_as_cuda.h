
#ifndef PSC_PARTICLES_AS_CUDA_H
#define PSC_PARTICLES_AS_CUDA_H

#include "psc_particles_cuda.h"

typedef particle_cuda_real_t particle_real_t;
typedef particle_cuda_t particle_t;
using mparticles_t = mparticles_cuda_t;

#define PARTICLE_TYPE               "cuda"

#define PSC_PARTICLES_AS_CUDA 1

#include "particle_iter.h"

#endif

