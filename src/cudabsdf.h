#ifndef CGL_STATICSCENE_CUDABSDF_H
#define CGL_STATICSCENE_CUDABSDF_H

#include "cudaSpectrum.h"
#include "cudaVector3D.h"
#include "cudaMatrix3x3.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "helper_math.h"

#include "cudaSampler.h"

#include <algorithm>

// Helper math functions. Assume all vectors are in unit hemisphere //

__device__ inline double cuda_clamp (double n, double lower, double upper) {
  return fmax(lower, fmin(n, upper));
}

__device__ inline double cuda_cos_theta(const CudaVector3D& w) {
  return w.z;
}

__device__ inline double cuda_abs_cos_theta(const CudaVector3D& w) {
  return fabs(w.z);
}

__device__ inline double cuda_sin_theta2(const CudaVector3D& w) {
  return fmax(0.0, 1.0 - cuda_cos_theta(w) * cuda_cos_theta(w));
}

__device__ inline double cuda_sin_theta(const CudaVector3D& w) {
  return sqrt(cuda_sin_theta2(w));
}

__device__ inline double cuda_cos_phi(const CudaVector3D& w) {
  double sinTheta = cuda_sin_theta(w);
  if (sinTheta == 0.0) return 1.0;
  return cuda_clamp(w.x / sinTheta, -1.0, 1.0);
}

__device__ inline double cuda_sin_phi(const CudaVector3D& w) {
  double sinTheta = cuda_sin_theta(w);
  if (sinTheta == 0.0) return 0.0;
  return cuda_clamp(w.y / sinTheta, -1.0, 1.0);
}

__device__ void make_cuda_coord_space(CudaMatrix3x3& o2w, const CudaVector3D& n);

/**
 * Interface for BSDFs.
 */
class CudaBSDF {
 public:

  /**
   * Evaluate BSDF.
   * Given incident light direction wi and outgoing light direction wo. Note
   * that both wi and wo are defined in the local coordinate system at the
   * point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi incident light direction in local space of point of intersection
   * \return reflectance in the given incident/outgoing directions
   */
  __device__ virtual CudaSpectrum f (const CudaVector3D& wo, const CudaVector3D& wi) = 0;

  /**
   * Evaluate BSDF.
   * Given the outgoing light direction wo, compute the incident light
   * direction and store it in wi. Store the pdf of the outgoing light in pdf.
   * Again, note that wo and wi should both be defined in the local coordinate
   * system at the point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi address to store incident light direction
   * \param pdf address to store the pdf of the output incident direction
   * \return reflectance in the output incident and given outgoing directions
   */
  __device__ virtual CudaSpectrum sample_f (const CudaVector3D& wo, CudaVector3D* wi, float* pdf) = 0;

  /**
   * Get the emission value of the surface material. For non-emitting surfaces
   * this would be a zero energy spectrum.
   * \return emission spectrum of the surface material
   */
  __device__ virtual CudaSpectrum get_emission () const = 0;

  /**
   * If the BSDF is a delta distribution. Materials that are perfectly specular,
   * (e.g. water, glass, mirror) only scatter light from a single incident angle
   * to a single outgoing angle. These BSDFs are best described with alpha
   * distributions that are zero except for the single direction where light is
   * scattered.
   */
  __device__ virtual bool is_delta() const = 0;

  /**
   * Reflection helper
   */
  __device__ virtual void reflect(const CudaVector3D& wo, CudaVector3D* wi);

  /**
   * Refraction helper
   */
  __device__ virtual bool refract(const CudaVector3D& wo, CudaVector3D* wi, float ior);

}; // class BSDF

/**
 * Diffuse BSDF.
 */
class CudaDiffuseBSDF : public CudaBSDF {
 public:

  __device__ CudaDiffuseBSDF(const CudaSpectrum& a);

  __device__ CudaSpectrum f(const CudaVector3D& wo, const CudaVector3D& wi);
  __device__ CudaSpectrum sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf);
  __device__ CudaSpectrum get_emission() const { return CudaSpectrum(); }
  __device__ bool is_delta() const { return false; }

private:

  CudaSpectrum albedo;
  CudaCosineWeightedHemisphereSampler3D sampler;

}; // class DiffuseBSDF

/**
 * Mirror BSDF
 */
class CudaMirrorBSDF : public CudaBSDF {
 public:

  __device__ CudaMirrorBSDF(const CudaSpectrum& reflectance);

  __device__ CudaSpectrum f(const CudaVector3D& wo, const CudaVector3D& wi);
  __device__ CudaSpectrum sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf);
  __device__ CudaSpectrum get_emission() const { return CudaSpectrum(); }
  __device__ bool is_delta() const { return true; }

private:

  float roughness;
  CudaSpectrum reflectance;

}; // class MirrorBSDF*/

/**
 * Glossy BSDF.
 */
/*
class GlossyBSDF : public BSDF {
 public:

  GlossyBSDF(const Spectrum& reflectance, float roughness)
    : reflectance(reflectance), roughness(roughness) { }

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf);
  Spectrum get_emission() const { return Spectrum(); }
  bool is_delta() const { return false; }

private:

  float roughness;
  Spectrum reflectance;

}; // class GlossyBSDF*/

/**
 * Refraction BSDF.
 */
class CudaRefractionBSDF : public CudaBSDF {
 public:

  __device__ CudaRefractionBSDF(const CudaSpectrum& transmittance, float roughness, float ior);

  __device__ CudaSpectrum f(const CudaVector3D& wo, const CudaVector3D& wi);
  __device__ CudaSpectrum sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf);
  __device__ CudaSpectrum get_emission() const { return CudaSpectrum(); }
  __device__ bool is_delta() const { return true; }

 private:

  float ior;
  float roughness;
  CudaSpectrum transmittance;

}; // class RefractionBSDF

/**
 * Glass BSDF.
 */
class CudaGlassBSDF : public CudaBSDF {
 public:

  __device__ CudaGlassBSDF(const CudaSpectrum& transmittance, const CudaSpectrum& reflectance,
            float roughness, float ior);

  __device__ CudaSpectrum f(const CudaVector3D& wo, const CudaVector3D& wi);
  __device__ CudaSpectrum sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf);
  __device__ CudaSpectrum get_emission() const { return CudaSpectrum(); }
  __device__ bool is_delta() const { return true; }

 private:

  float ior;
  float roughness;
  CudaSpectrum reflectance;
  CudaSpectrum transmittance;

}; // class GlassBSDF

/**
 * Emission BSDF.
 */
class CudaEmissionBSDF : public CudaBSDF {
 public:

  __device__ CudaEmissionBSDF(const CudaSpectrum& radiance);

  __device__ CudaSpectrum f(const CudaVector3D& wo, const CudaVector3D& wi);
  __device__ CudaSpectrum sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf);
  __device__ CudaSpectrum get_emission() const { return radiance; }
  __device__ bool is_delta() const { return false; }

 private:

  CudaSpectrum radiance;
  CudaCosineWeightedHemisphereSampler3D sampler;

}; // class EmissionBSDF

#endif  // CGL_STATICSCENE_CUDABSDF_H
