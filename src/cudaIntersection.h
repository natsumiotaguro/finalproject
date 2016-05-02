#ifndef CGL_CUDAINTERSECT_H
#define CGL_CUDAINTERSECT_H

#include <vector>

#include "cudaVector3D.h"
#include "cudaSpectrum.h"

#include "cudabsdf.h"

#define CUDA_D 0x7f800000

class CudaPrimitive;

/**
 * A record of an intersection point which includes the time of intersection
 * and other information needed for shading
 */
struct CudaIntersection {

  CudaIntersection() {
    this->t = CUDA_D;
    this->primitive = NULL;
    this->bsdf = NULL;
  }

  double t;    ///< time of intersection

  const CudaPrimitive* primitive;  ///< the primitive intersected

  CudaVector3D n;  ///< normal at point of intersection

  CudaBSDF* bsdf; ///< BSDF of the surface at point of intersection

  // More to follow.
};


#endif // CGL_CUDAINTERSECT_H
