#ifndef CGL_CUDABVH_H
#define CGL_CUDABVH_H

#include "static_scene/scene.h"
#include "static_scene/cudaAggregate.h"
#include "cudaIntersection.h"
#include <vector>
namespace StaticScene {


/**
 * A node in the BVH accelerator aggregate.
 * The accelerator uses a "flat tree" structure where all the primitives are
 * stored in one vector. A node in the data structure stores only the starting
 * index and the number of primitives in the node and uses this information to
 * index into the primitive vector for actual data. In this implementation all
 * primitives (index + range) are stored on leaf nodes. A leaf node has no child
 * node and its range should be no greater than the maximum leaf size used when
 * constructing the BVH.
 */
struct CudaBVHNode {

  __device__ CudaBVHNode(CudaBBox bb) {
    this->bb = bb
    this->l = NULL
    this->r = NULL
    this->prims = NULL
  }

  __device__ ~CudaBVHNode() {
    if (prims) delete prims;
    if (l) delete l;
    if (r) delete r;
  }

  __device__ inline bool isLeaf() const { 
    return l == NULL && r == NULL; 
  }

  CudaBBox bb;        ///< bounding box of the node
  CudaBVHNode* l;     ///< left child node
  CudaBVHNode* r;     ///< right child node
  std::vector<CudaPrimitive *> *prims;

};

/**
 * Bounding Volume Hierarchy for fast Ray - Primitive intersection.
 * Note that the BVHAccel is an Aggregate (A Primitive itself) that contains
 * all the primitives it was built from. Therefore once a BVHAccel Aggregate
 * is created, the original input primitives can be ignored from the scene
 * during ray intersection tests as they are contained in the aggregate.
 */
class CudaBVHAccel : public CudaAggregate {
 public:

  __device__ CudaBVHAccel () { }

  /**
   * Parameterized Constructor.
   * Create BVH from a list of primitives. Note that the BVHAccel Aggregate
   * stores pointers to the primitives and thus the primitives need be kept
   * in memory for the aggregate to function properly.
   * \param primitives primitives to build from
   * \param max_leaf_size maximum number of primitives to be stored in leaves
   */
  __device__ CudaBVHAccel(const std::vector<CudaPrimitive*>& primitives, size_t max_leaf_size = 4);

  /**
   * Destructor.
   * The destructor only destroys the Aggregate itself, the primitives that
   * it contains are left untouched.
   */
  __device__ ~CudaBVHAccel();

  /**
   * Get the world space bounding box of the aggregate.
   * \return world space bounding box of the aggregate
   */
  __device__ CudaBBox get_bbox() const;

  /**
   * Ray - Aggregate intersection.
   * Check if the given ray intersects with the aggregate (any primitive in
   * the aggregate), no intersection information is stored.
   * \param r ray to test intersection with
   * \return true if the given ray intersects with the aggregate,
             false otherwise
   */
  __device__ bool intersect(const CudaRay& r) const {
    ++total_rays;
    return intersect(r, root);
  }

  __device__ bool intersect(const CudaRay& r, CudaBVHNode *node) const;

  /**
   * Ray - Aggregate intersection 2.
   * Check if the given ray intersects with the aggregate (any primitive in
   * the aggregate). If so, the input intersection data is updated to contain
   * intersection information for the point of intersection. Note that the
   * intersected primitive entry in the intersection should be updated to
   * the actual primitive in the aggregate that the ray intersected with and
   * not the aggregate itself.
   * \param r ray to test intersection with
   * \param i address to store intersection info
   * \return true if the given ray intersects with the aggregate,
             false otherwise
   */
  __device__ bool intersect(const CudaRay& r, CudaIntersection* i) const {
    ++total_rays;
    return intersect(r, i, root);
  }

  __device__ bool intersect(const CudaRay& r, CudaIntersection* i, CudaBVHNode *node) const;

  /**
   * Get BSDF of the surface material
   * Note that this does not make sense for the BVHAccel aggregate
   * because it does not have a surface material. Therefore this
   * should always return a null pointer.
   */
  __device__ CudaBSDF* get_bsdf() const { return NULL; }

  /**
   * Get entry point (root) - used in visualizer
   */
  __device__ CudaBVHNode* get_root() const { return root; }

  /**
   * Draw the BVH with OpenGL - used in visualizer
   */
  __device__ void draw(const CudaColor& c) const { }
  __device__ void draw(CudaBVHNode *node, const CudaColor& c) const;

  /**
   * Draw the BVH outline with OpenGL - used in visualizer
   */
  __device__ void drawOutline(const CudaColor& c) const { }
  __device__ void drawOutline(CudaBVHNode *node, const CudaColor& c) const;

  mutable unsigned long long total_rays, total_isects;
 private:
  CudaBVHNode* root; ///< root node of the BVH
  __device__ CudaBVHNode *construct_bvh(const std::vector<CudaPrimitive*>& prims, size_t max_leaf_size);
};

} // namespace StaticScene

#endif // CGL_CUDABVH_H
