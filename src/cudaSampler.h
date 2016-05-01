#ifndef CGL_CUDASAMPLER_H
#define CGL_CUDASAMPLER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cudaVector2D.h"
#include "cudaVector3D.h"
//#include "CGL/misc.h"
#include "cudaRandom_util.h"



#define PI 3.1415926

/**
 * Interface for generating point samples within the unit square
 */
class CudaSampler2D {
 public:

  /**
   * Virtual destructor.
   */
   __device__ virtual ~CudaSampler2D() { }

  /**
   * Take a point sample of the unit square
   */
   __device__ virtual CudaVector2D get_sample() const = 0;

}; // class Sampler2D

/**
 * Interface for generating 3D vector samples
 */
class CudaSampler3D {
 public:

  /**
   * Virtual destructor.
   */
   __device__ virtual ~CudaSampler3D() { }

  /**
   * Take a vector sample of the unit hemisphere
   */
   __device__ virtual CudaVector3D get_sample() const = 0;

}; // class Sampler3D


/**
 * A Sampler2D implementation with uniform distribution on unit square
 */
class CudaUniformGridSampler2D : public CudaSampler2D {
 public:

   __device__ CudaVector2D get_sample() const;

}; // class UniformSampler2D

/**
 * A Sampler3D implementation with uniform distribution on unit hemisphere
 */
class CudaUniformHemisphereSampler3D : public CudaSampler3D {
 public:

   __device__ CudaVector3D get_sample() const;

}; // class UniformHemisphereSampler3D

/**
 * A Sampler3D implementation with cosine-weighted distribution on unit
 * hemisphere.
 */
class CudaCosineWeightedHemisphereSampler3D : public CudaSampler3D {
 public:

   __device__ CudaVector3D get_sample() const;
  // Also returns the pdf at the sample point for use in importance sampling.
  __device__ CudaVector3D get_sample(float* pdf) const;

}; // class UniformHemisphereSampler3D

/**
 * TODO (extra credit) :
 * Jittered sampler implementations
 */



#endif //CGL_CUDASAMPLER_H
