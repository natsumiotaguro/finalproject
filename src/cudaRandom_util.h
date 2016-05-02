#ifndef CGL_CUDARANDOMUTIL_H
#define CGL_CUDARANDOMUTIL_H

#include <cuda.h>
#include <cuda_runtime.h>

__device__ curandState state;

__device__ void init_rand(){
	curand_init((unsigned int) clock(), threadIdx.x, 0, &state);
}

/**
 * Returns a number distributed uniformly over [0, 1].
 */
__device__ double cuda_random_uniform() {
 float a = curand_uniform(&state);
 return a;
}

/**
 * Returns true with probability p and false with probability 1 - p.
 */
__device__ bool cuda_coin_flip(double p) {
	return cuda_random_uniform() < p;
}



#endif  // CGL_CUDARANDOMUTIL_H
