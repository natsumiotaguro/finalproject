#ifndef CGL_CUDARAY_H
#define CGL_CUDARAY_H

/*
#include "CGL/CGL.h"
#include "CGL/vector3D.h"
#include "CGL/vector4D.h"
#include "CGL/matrix4x4.h"
#include "CGL/spectrum.h"
*/

#include <limits>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaVector3D.h"
#include <math_constants.h>

#define CUDA_D 0x7f800000

#define PART 5
#define PART_1 (PART >= 1)
#define PART_2 (PART >= 2)
#define PART_3 (PART >= 3)
#define PART_4 (PART >= 4)
#define PART_5 (PART >= 5)

struct CudaRay {
  size_t depth;  ///< depth of the Ray

  CudaVector3D o;  ///< origin
  CudaVector3D d;  ///< direction
  mutable double min_t; ///< treat the ray as a segment (ray "begin" at min_t)
  mutable double max_t; ///< treat the ray as a segment (ray "ends" at max_t)

  CudaVector3D inv_d;  ///< component wise inverse
  int sign[3];     ///< fast ray-bbox intersection

  /**
   * Constructor.
   * Create a ray instance with given origin and direction.
   * \param o origin of the ray
   * \param d direction of the ray
   * \param depth depth of the ray
   */
    __device__ CudaRay(const CudaVector3D& o, const CudaVector3D& d, int depth = 0) {
    this->o = o;
    this->d = d;
    this->depth = depth;
    min_t = CUDA_D;
    inv_d = CudaVector3D(1 / d.x, 1 / d.y, 1 / d.z);
    sign[0] = (inv_d.x < 0);
    sign[1] = (inv_d.y < 0);
    sign[2] = (inv_d.z < 0);
  }

  /**
   * Constructor.
   * Create a ray instance with given origin and direction.
   * \param o origin of the ray
   * \param d direction of the ray
   * \param max_t max t value for the ray (if it's actually a segment)
   * \param depth depth of the ray
   */
    __device__ CudaRay(const CudaVector3D& o, const CudaVector3D& d, double max_t, int depth = 0)
        : o(o), d(d), min_t(0.0), max_t(max_t), depth(depth) {
    inv_d = CudaVector3D(1 / d.x, 1 / d.y, 1 / d.z);
    sign[0] = (inv_d.x < 0);
    sign[1] = (inv_d.y < 0);
    sign[2] = (inv_d.z < 0);
  }


  /**
   * Returns the point t * |d| along the ray.
   */
  __device__ inline CudaVector3D at_time(double t) const { return o + t * d; }

  /**
   * Returns the result of transforming the ray by the given transformation
   * matrix.
   */
  /*
  __device__ CudaRay transform_by(const Matrix4x4& t) const {
    const Vector4D& newO = t * Vector4D(o, 1.0);
    return Ray((newO / newO.w).to3D(), (t * Vector4D(d, 0.0)).to3D());
  }
  */
};

// structure used for logging rays for subsequent visualization
struct CudaLoggedRay {

    __device__ CudaLoggedRay(const CudaRay& r, double hit_t)
        : o(r.o), d(r.d), hit_t(hit_t) {}

    CudaVector3D o;
    CudaVector3D d;
    double hit_t;
};

#endif  // CGL_CUDARAY_H
