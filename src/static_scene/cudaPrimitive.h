#ifndef CGL_STATICSCENE_CUDAPRIMITIVE_H
#define CGL_STATICSCENE_CUDAPRIMITIVE_H

#include "../cudaIntersection.h"
#include "../cudabbox.h"



/**
 * The abstract base class primitive is the bridge between geometry processing
 * and the shading subsystem. As such, its interface contains methods related
 * to both.
 */
class CudaPrimitive {
 public:

  /**
   * Get the world space bounding box of the primitive.
   * \return world space bounding box of the primitive
   */
  __device__ virtual CudaBBox get_bbox() = 0;

  /**
   * Ray - Primitive intersection.
   * Check if the given ray intersects with the primitive, no intersection
   * information is stored.
   * \param r ray to test intersection with
   * \return true if the given ray intersects with the primitive,
             false otherwise
   */
  __device__ virtual bool intersect(CudaRay& r) = 0;

  /**
   * Ray - Primitive intersection 2.
   * Check if the given ray intersects with the primitive, if so, the input
   * intersection data is updated to contain intersection information for the
   * point of intersection.
   * \param r ray to test intersection with
   * \param i address to store intersection info
   * \return true if the given ray intersects with the primitive,
             false otherwise
   */
  __device__ virtual bool intersect(CudaRay& r, CudaIntersection* i) = 0;

  /**
   * Get BSDF.
   * Return the BSDF of the surface material of the primitive.
   * Note that the BSDFs are not stored in each primitive but in the
   * SceneObject the primitive belongs to.
   */
  __device__ virtual CudaBSDF* get_bsdf() = 0;

  /**
   * Draw with OpenGL (for visualization)
   * \param c desired highlight color
   */
  __device__ virtual void draw(CudaColor& c) = 0;

  /**
   * Draw outline with OpenGL (for visualization)
   * \param c desired highlight color
   */
  __device__ virtual void drawOutline(CudaColor& c) = 0;

};


#endif //CGL_STATICSCENE_CUDAPRIMITIVE_H
