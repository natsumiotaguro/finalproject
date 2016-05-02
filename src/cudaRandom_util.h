#ifndef CGL_CUDARANDOMUTIL_H
#define CGL_CUDARANDOMUTIL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__device__ extern curandState state_rand;

__device__ void init_rand();

/**
 * Returns a number distributed uniformly over [0, 1].
 */
__device__ double cuda_random_uniform();

/**
 * Returns true with probability p and false with probability 1 - p.
 */
__device__ bool cuda_coin_flip(double p);



#endif  // CGL_CUDARANDOMUTIL_H
