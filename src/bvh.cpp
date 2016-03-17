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
      //Pick the axis to recurse on (perhaps the largest dimension of the bounding box's extent).
  //Calculate the split point you are using on this axis (perhaps the midpoint of the bounding box).
  //Split all primitives in prims into two new vectors based on whether their bounding box's centroid's coordinate in the chosen axis is less than or greater than the split point. (p->get_bbox().centroid() is a quick way to get a bounding box centroid for Primitive *p.)
  //Recurse, assigning the left and right children of this node to be two new calls to construct_bvh() with the two primitive lists you just generated.
  }
  
  return node;
  

}


bool BVHAccel::intersect(const Ray& ray, BVHNode *node) const {

  // TODO Part 2, task 3:
  // Implement BVH intersection.
  // Currently, we just naively loop over every primitive.

  for (Primitive *p : *(root->prims)) {
    total_isects++;
    if (p->intersect(ray)) 
      return true;
  }
  return false;

}

bool BVHAccel::intersect(const Ray& ray, Intersection* i, BVHNode *node) const {

  // TODO Part 2, task 3:
  // Implement BVH intersection.
  // Currently, we just naively loop over every primitive.

  bool hit = false;
  for (Primitive *p : *(root->prims)) {
    total_isects++;
    if (p->intersect(ray, i)) 
      hit = true;
  }
  return hit;

}

}  // namespace StaticScene
}  // namespace CGL
