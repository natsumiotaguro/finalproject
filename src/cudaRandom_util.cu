#include "cudaRandom_util.h"

//__device__ curandState state_rand;

__device__ void init_rand(){
	curand_init((unsigned int) clock(), threadIdx.x, 0, &state_rand);
}

/**
 * Returns a number distributed uniformly over [0, 1].
 */
__device__ double cuda_random_uniform() {
 float a = curand_uniform(&state_rand);
 return a;
}

/**
 * Returns true with probability p and false with probability 1 - p.
 */
__device__ bool cuda_coin_flip(double p) {
	return cuda_random_uniform() < p;
}
