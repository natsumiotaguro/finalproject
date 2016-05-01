#include "bvh.h"

#include "CGL/CGL.h"
#include "static_scene/triangle.h"

#include <iostream>
#include <stack>

using namespace std;

namespace CGL { namespace StaticScene {

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {

  root = construct_bvh(_primitives, max_leaf_size);

}

BVHAccel::~BVHAccel() {
  if (root) delete root;
}

BBox BVHAccel::get_bbox() const {
  return root->bb;
}

void BVHAccel::draw(BVHNode *node, const Color& c) const {
  if (node->isLeaf()) {
    for (Primitive *p : *(node->prims))
      p->draw(c);
  } else {
    draw(node->l, c);
    draw(node->r, c);
  }
}

void BVHAccel::drawOutline(BVHNode *node, const Color& c) const {
  if (node->isLeaf()) {
    for (Primitive *p : *(node->prims))
      p->drawOutline(c);
  } else {
    drawOutline(node->l, c);
    drawOutline(node->r, c);
  }
}

BVHNode *BVHAccel::construct_bvh(const std::vector<Primitive*>& prims, size_t max_leaf_size) {
  
  // TODO Part 2, task 1:
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.


  BBox centroid_box, bbox;

  for (Primitive *p : prims) {
    BBox bb = p->get_bbox();
    bbox.expand(bb);
    Vector3D c = bb.centroid();
    centroid_box.expand(c);
  }

  // You'll want to adjust this code.
  // Right now we just return a single node containing all primitives.
  BVHNode *node = new BVHNode(bbox);
  node->prims = new vector<Primitive *>(prims);

  if(node->prims->size() > max_leaf_size){
    //Recurse left and right
    vector<Primitive *> left;
    vector<Primitive *> right;
    float divider = 0.5;
    float fail_count = 0; //If axis fails, we try a different axis
    while(fail_count < 3 && (left.empty() == true || right.empty() == true)){

      for (Primitive *p : prims){
          Vector3D extent = bbox.extent;
          if(extent.x > extent.y && extent.x > extent.z || fail_count == 1){
            //Recurse on x
            float division = (bbox.min.x + bbox.max.x)/2;
            Vector3D centroid = p->get_bbox().centroid();
            if(centroid.x < division){
              left.push_back(p);
            }
            else{
              right.push_back(p);
            }
          }
          else if(extent.y > extent.x && extent.y > extent.z || fail_count == 2){
            //Recurse on y
            float division = (bbox.min.y + bbox.max.y)/2;
            Vector3D centroid = p->get_bbox().centroid();
            if(centroid.y < division){
              left.push_back(p);
            }
            else{
              right.push_back(p);
            }
          }
          else{
            //Recurse on z
            float division = (bbox.min.z + bbox.max.z)/2;
            Vector3D centroid = p->get_bbox().centroid();
            if(centroid.z < division){
              left.push_back(p);
            }
            else{
              right.push_back(p);
            }
          }
      }
    //If left is empty or right is empty, we are doing another iteration
    //Whic means we should empty out the left and right array
      if(left.empty() == true || right.empty() == true){
        //All things were to the right
        left.clear();
        right.clear();
      }
      fail_count++;
    }
      //Then, move the division either more to the right 
    if(left.empty() == true || right.empty() == true){
        //All things were to the right
      left.clear();
      right.clear();
      return node;
    }
    node->l = construct_bvh(left, max_leaf_size);
    node->r = construct_bvh(right, max_leaf_size);

  }
  
  return node;
  

}


bool BVHAccel::intersect(const Ray& ray, BVHNode *node) const {

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
      for (Primitive *p : *(node->prims)) {
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



bool BVHAccel::intersect(const Ray& ray, Intersection* i, BVHNode *node) const {

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
      for (Primitive *p : *(node->prims)) {
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
  


}  // namespace StaticScene
}  // namespace CGL
