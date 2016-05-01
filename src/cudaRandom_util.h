#ifndef CGL_CUDARANDOMUTIL_H
#define CGL_CUDARANDOMUTIL_H

__device__ curandState state;

__device__ void init_rand(){
	curand_init((unsigned int) clock64(), threadIdx.x, 0, &state);
}

/**
 * Returns a number distributed uniformly over [0, 1].
 */
__device__ double random_uniform() {
 float a = curand_uniform(&state);
 return a;
}

/**
 * Returns true with probability p and false with probability 1 - p.
 */
__device__ bool coin_flip(double p) {
	return random_uniform() < p;
}



#endif  // CGL_CUDARANDOMUTIL_H
