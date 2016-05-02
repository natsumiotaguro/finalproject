#ifndef CGL_CUDASTATICSCENE_SCENE_H
#define CGL_CUDASTATICSCENE_SCENE_H

#include "cudaPrimitive.h"
#include <vector>



/**
 * Interface for objects in the scene.
 */
class CudaSceneObject {
 public:

  /**
   * Get all the primitives in the scene object.
   * \return a vector of all the primitives in the scene object
   */
  virtual std::vector<CudaPrimitive*> get_primitives() const = 0;

  /**
   * Get the surface BSDF of the object's surface.
   * \return the BSDF of the objects's surface
   */
  virtual CudaBSDF* get_bsdf() const = 0;

};



/**
 * Interface for lights in the scene.
 */
class CudaSceneLight {
 public:
  __device__ virtual CudaSpectrum sample_L(const CudaVector3D& p, CudaVector3D* wi,
                            float* distToLight, float* pdf) const = 0;
  __device__ virtual bool is_delta_light() const = 0;

};



struct CudaScene {
  CudaSceneScene(const std::vector<CudaSceneObject *>& objects,
        const std::vector<CudaSceneLight *>& lights){
    this->objects = objects;
    this->lights = lights;
  }
  // kept to make sure they don't get deleted, in case the
  //  primitives depend on them (e.g. Mesh Triangles).
  std::vector<SceneObject*> objects;

  // for sake of consistency of the scene object Interface
  std::vector<SceneLight*> lights;

  // TODO (sky) :
  // Adding object with emission BSDFs as mesh lights and sphere lights so 
  // that light sampling configurations also applies to mesh lights. 
//  for (SceneObject *obj : objects) {
//    if (obj->get_material().emit != Spectrum()) {
//      
//      // mesh light
//      if (dynamic_cast<Mesh*>(obj)) {
//        staticLights.push_back()
//      }
//      
//      // sphere light
//      if (dynamic_cast<Sphere*>(obj)) {
//
//      }
//  }

};





#endif //CGL_CUDASTATICSCENE_SCENE_H