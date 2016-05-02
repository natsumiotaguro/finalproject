#include "cudabvh.h"

//#include "CGL/CGL.h"
//#include "static_scene/triangle.h"

#include <iostream>
#include <stack>

using namespace std;


__device__ CudaBVHAccel::CudaBVHAccel(CudaPrimitive** &primitives, size_t prim_len,
                   size_t max_leaf_size) {

  root = construct_bvh(primitives, max_leaf_size, prim_len);
  this->prim_len = prim_len;
}

__device__ CudaBVHAccel::~CudaBVHAccel() {
  if (root) delete root;
}

__device__ CudaBBox CudaBVHAccel::get_bbox() {
  return root->bb;
}

__device__ void CudaBVHAccel::draw(CudaBVHNode *node, CudaColor& c) {
  /*if (node->isLeaf()) {
    for (CudaPrimitive *p : *(node->prims))
      p->draw(c);
  } else {
    draw(node->l, c);
    draw(node->r, c);
  }*/
}

__device__ void CudaBVHAccel::drawOutline(CudaBVHNode *node, CudaColor& c) {
  /*if (node->isLeaf()) {
    for (CudaPrimitive *p : *(node->prims))
      p->drawOutline(c);
  } else {
    drawOutline(node->l, c);
    drawOutline(node->r, c);
  }*/
}

__device__ CudaBVHNode *CudaBVHAccel::construct_bvh(CudaPrimitive**& prims, size_t max_leaf_size, size_t prim_len) {
  
  // TODO Part 2, task 1:
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.


  CudaBBox centroid_box, bbox;

  for (int i = 0; i < prim_len; ++i) {
    CudaBBox bb = prims[i]->get_bbox();
    bbox.expand(bb);
    CudaVector3D c = bb.centroid();
    centroid_box.expand(c);
  }

  // You'll want to adjust this code.
  // Right now we just return a single node containing all primitives.
  CudaBVHNode *node = new CudaBVHNode(bbox);
  node->prims = (CudaPrimitive***)malloc(sizeof(CudaPrimitive**));

  if(prim_len > max_leaf_size){
    //Recurse left and right
    CudaPrimitive **left  = (CudaPrimitive**)malloc(sizeof(CudaPrimitive*) * prim_len);
    CudaPrimitive **right = (CudaPrimitive**)malloc(sizeof(CudaPrimitive*) * prim_len);
    size_t leftIndex = 0;
    size_t rightIndex = 0;
    float divider = 0.5;
    float fail_count = 0; //If axis fails, we try a different axis
    while(fail_count < 3 && (leftIndex == 0 || rightIndex == 0)){
      for (int j = 0; j < prim_len; j++){
          CudaPrimitive* p = *(node->prims)[j];
          CudaVector3D extent = bbox.extent;
          if(extent.x > extent.y && extent.x > extent.z || fail_count == 1){
            //Recurse on x
            float division = (bbox.min.x + bbox.max.x)/2;
            CudaVector3D centroid = p->get_bbox().centroid();
            if(centroid.x < division){
              left[leftIndex] = p;
              leftIndex++;
            }
            else{
              right[rightIndex] = p;
              rightIndex++;
            }
          }
          else if(extent.y > extent.x && extent.y > extent.z || fail_count == 2){
            //Recurse on y
            float division = (bbox.min.y + bbox.max.y)/2;
            CudaVector3D centroid = p->get_bbox().centroid();
            if(centroid.y < division){
              left[leftIndex] = p;
              leftIndex++;
            }
            else{
              right[rightIndex] = p;
              rightIndex++;
            }
          }
          else{
            //Recurse on z
            float division = (bbox.min.z + bbox.max.z)/2;
            CudaVector3D centroid = p->get_bbox().centroid();
            if(centroid.z < division){
              left[leftIndex] = p;
              leftIndex++;
            }
            else{
              right[rightIndex] = p;
              rightIndex++;
            }
          }
      }
    //If left is empty or right is empty, we are doing another iteration
    //Whic means we should empty out the left and right array
      if(leftIndex == 0 || rightIndex == 0){
        //All things were to the right
        leftIndex = 0;
        rightIndex = 0;
      }
      fail_count++;
    }
      //Then, move the division either more to the right 
    if(leftIndex == 0 || rightIndex == 0){
        //All things were to the right
      leftIndex = 0;
      rightIndex = 0;
      return node;
    }
    node->l = construct_bvh(left, max_leaf_size, leftIndex);
    node->r = construct_bvh(right, max_leaf_size, rightIndex);
  }
  return node;
  

}


__device__ bool CudaBVHAccel::intersect(CudaRay& ray, CudaBVHNode *node) {

  // TODO Part 2, task 3:
  // Implement BVH intersection.
  // Currently, we just naively loop over every primitive.
  double t0 = 0;
  double t1 = 0;
  if(node->bb.intersect(ray, t0, t1) == false){
    return false;
  }
  else{
    if(ray.min_t > t1 || ray.max_t < t0){
      return false;
    }

    if(node->isLeaf()){
      //
      for (int j = 0; j < this->prim_len; j++) {
        CudaPrimitive *p = *(node->prims)[j];
        if (p->intersect(ray)){
         return true;
        }
      }
      return false;
    }
    else{
      bool hit1 = intersect(ray, node->l);
      bool hit2 = intersect(ray, node->r);
      return hit1 || hit2;
    }
  }
}



__device__ bool CudaBVHAccel::intersect(CudaRay& ray, CudaIntersection* i, CudaBVHNode *node) {

  // TODO Part 2, task 3:
  // Implement BVH intersection.
  // Currently, we just naively loop over every primitive.
  double t0 = 0;
  double t1 = 0;
  if(node->bb.intersect(ray, t0, t1) == false){
    return false;
  }

  else{
    if(ray.min_t > t1 || ray.max_t < t0){
      return false;
    }
    if(node->isLeaf()){
      //
      bool hit = false;
      for (int j = 0; j < this->prim_len; j++) {
        CudaPrimitive *p = *(node->prims)[j];
        if (p->intersect(ray, i)){
          hit = true;
        }
      }
      return hit;
    }
    else{
      bool hit1 = intersect(ray, i, node->l);
      bool hit2 = intersect(ray, i, node->r);
      return hit1 || hit2;
    }
  }
}
