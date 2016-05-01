#include "cudaSampler.h"



// Uniform Sampler2D Implementation //

__device__ CudaVector2D CudaUniformGridSampler2D::get_sample() const {

  return CudaVector2D(random_uniform(), random_uniform());

}

// Uniform Hemisphere Sampler3D Implementation //

__device__ CudaVector3D CudaUniformHemisphereSampler3D::get_sample() const {

  double Xi1 = random_uniform();
  double Xi2 = random_uniform();

  double theta = acos(Xi1);
  double phi = 2.0 * PI * Xi2;

  double xs = sinf(theta) * cosf(phi);
  double ys = sinf(theta) * sinf(phi);
  double zs = cosf(theta);

  return CudaVector3D(xs, ys, zs);

}

__device__ CudaVector3D CudaCosineWeightedHemisphereSampler3D::get_sample() const {
  float f;
  return get_sample(&f);
}

__device__ CudaVector3D CudaCosineWeightedHemisphereSampler3D::get_sample(float *pdf) const {

  double Xi1 = random_uniform();
  double Xi2 = random_uniform();

  double r = sqrt(Xi1);
  double theta = 2. * PI * Xi2;
  *pdf = sqrt(1-Xi1) / PI;
  return CudaVector3D(r*cos(theta), r*sin(theta), sqrt(1-Xi1));
}



