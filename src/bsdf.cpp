#include "bsdf.h"

#include <iostream>
#include <algorithm>
#include <utility>

using std::min;
using std::max;
using std::swap;

namespace CGL {

void make_coord_space(Matrix3x3& o2w, const Vector3D& n) {

    Vector3D z = Vector3D(n.x, n.y, n.z);
    Vector3D h = z;
    if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z)) h.x = 1.0;
    else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z)) h.y = 1.0;
    else h.z = 1.0;

    z.normalize();
    Vector3D y = cross(h, z);
    y.normalize();
    Vector3D x = cross(z, y);
    x.normalize();

    o2w[0] = x;
    o2w[1] = y;
    o2w[2] = z;
}

// Diffuse BSDF //

Spectrum DiffuseBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return albedo * (1.0 / PI);
}

Spectrum DiffuseBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *wi = sampler.get_sample(pdf);
  return albedo * (1.0 / PI);
}

// Mirror BSDF //
//local coordinate system at the
//   * point of intersection.
Spectrum MirrorBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  //not used not used NOT USED
  printf("USEING \n");
  return Spectrum();
}

Spectrum MirrorBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {

  // TODO Part 5:
  // Implement MirrorBSDF
  // reflectance in the output incident and given outgoing directions
  *pdf = 1;
  BSDF::reflect(wo, wi); //reflectance divided by the cosine of wi with the normal.
  Spectrum result = reflectance/abs_cos_theta(wo);
  return reflectance/abs_cos_theta(*wi);
}

// Glossy BSDF //

/*
Spectrum GlossyBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum GlossyBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *pdf = 1.0f;
  return reflect(wo, wi, reflectance);
}
*/

// Refraction BSDF //

Spectrum RefractionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
    return Spectrum(); //NOT USED NOT USED NOT USED
  
}

Spectrum RefractionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // TODO Part 5: NOT USED NOT USED NOT USED
  // Implement RefractionBSDF     
    return Spectrum();
}

// Glass BSDF //

Spectrum GlassBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  Vector3D tmp = Vector3D();
  printf("hohoho\n");
  double is_internal = !refract(wo, &tmp, ior);
  if(is_internal){
    return reflectance/abs_cos_theta(wi);
  }
  else{
    double vacuum_check = wo.z;
    double ni = 1;
    double no = ior;
    if(wo.z > 0){
      ni = ior;
      no = 1.0;
    }
    printf("hey hey \n");
    double r0 = pow((ni - no)/(ni + no), 2);
    double cos_wi = dot(wo, Vector3D(0, 0, 1));
    double r = clamp(r0 + (1 - r0)*(pow(1-cos_wi, 5)), 0, 1); //r0 + (1-R0)(1 - cos(theta_0))^5
    
    double is_reflection = coin_flip(r);    //coin flip, decide reflection or refraction
    
    if(is_reflection == true){
      return r*reflectance/fabs(wi.z);
    }
    else{
      return (1 - r)*transmittance*pow((no/ni), 2)/fabs(wi.z); 
      //return refr(di), 1-R*Li(no/ni)^2/cos(thetai), 1-R)
    }
  }
  return Spectrum();
}

Spectrum GlassBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
    Vector3D tmp_wo = -1*wo;
    
    double vacuum_check = wo.z;
    double ni = 1;
    double no = ior;
    if(wo.z < 0){
      ni = ior;
      no = 1.0;
    }

    double r0 = pow((ni - no)/(ni + no), 2);
    double cos_i = -1*dot(tmp_wo, Vector3D(0, 0, 1)); //-1*dot(-1*wo, Vector3D(0, 0, 1));
    double io_ratio = ni/no; 
    
    //printf("N2 n1::%f\n", no/ni);
    float sin2 = io_ratio * io_ratio * (1.0 - cos_i*cos_i);
      
    if(ni > no){
      double critical_angle = asin(no/ni);
      //printf("Critical: %f\n", critical_angle);
      //printf("sinsqure %f\n", sin2);
      if(sin2 > 1){
        reflect(wo, wi);
        *pdf = 1;
        //printf("total internal\n");
        return reflectance/abs_cos_theta(*wi); 
      }
      cos_i = sqrt(1.0 - sin2);
    }
    //printf("sin 2 in the large %f\n", sin2);
    //printf("Cos i %f\n", cos_i);
    double r = clamp(r0 + (1 - r0)*(pow(1-cos_i, 5)), 0, 1); //r0 + (1-R0)(1 - cos(theta_0))^5
    double is_reflection = coin_flip(r);    //coin flip, decide reflection or refraction
   
    //is_reflection = false; //TAKE IT OUT TAKE IT OUT TAKE IT OUT
    //r = 0.1;

    if(is_reflection == true){
      *pdf = r;
      reflect(wo, wi);
      //printf("Coin flip refl: %f\n", r);
      return r*reflectance/abs_cos_theta(*wi);
    }
    else{
      *pdf = 1 - r;
      refract(wo, wi, ior);
   
       //already calculated refraction
      //printf("GOOOD%f , %f,  %f\n", 1-r, abs_cos_theta(*wi)), fabs(wi->z);
      Spectrum tmp_s = (1 - r)*transmittance*pow((no/ni), 2)/abs_cos_theta(*wi); //fabs(cos_i); 
      return transmittance/fabs(wi->z);
      //return refr(di), 1-R*Li(no/ni)^2/cos(thetai), 1-R)
    }
 // }
 // return Spectrum();
}

void BSDF::reflect(const Vector3D& wo, Vector3D* wi) {
  // TODO Part 5:
  // Implement reflection of wo about normal (0,0,1) and store result in wi.
  //*wi = wo; //This makes the ray go the other direction
  *wi = -wo + 2 *dot(Vector3D(0, 0, 1), wo) * Vector3D(0, 0, 1);
  //wi->z = -1*wo.z;
}

bool BSDF::refract(const Vector3D& wo, Vector3D* wi, float ior) {
  // TODO Part 5:
  // Use Snell's Law to refract wo surface and store result ray in wi.
  // Return false if refraction does not occur due to total internal reflection
  // and true otherwise. When dot(wo,n) is positive, then wo corresponds to a
  // ray entering the surface through vacuum.
 Vector3D tmp_wo = -1*wo;
    double vacuum_check = wo.z;
    double ni = 1;
    double no = ior;
    if(wo.z < 0){
      ni = ior;
      no = 1.0;
    }

    double cos_i = -1*dot(tmp_wo, Vector3D(0, 0, 1)); //-1*dot(-1*wo, Vector3D(0, 0, 1));
    double io_ratio = ni/no; 
    //printf("IO Ratio: %f\n", io_ratio);
    float sin2 = io_ratio * io_ratio * (1.0 - cos_i*cos_i);
      
    if(ni > no){
      if(sin2 > 1){
        printf("baaaaadtotal internal\n");
        return false; 
      }
      cos_i = sqrt(1.0 - sin2);
    }
    //printf("sin 2 in the large %f\n", sin2);
  //if(ni*sin2 >= no){
 //   return false;
 // }
  float cos2 = sqrt(1.0 - sin2);
  *wi = (io_ratio * tmp_wo + (io_ratio*cos_i - cos2) * Vector3D(0, 0, 1));
  //wi->x = io_ratio * wo.x;
  //wi->y = io_ratio * wo.y;
  //wi->z = io_ratio * wo.z; 
  //wi->z += (io_ratio*cos_o - cos2); //n * cos I - cos t
  return true;

}

// Emission BSDF //

Spectrum EmissionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum EmissionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *pdf = 1.0 / PI;
  *wi  = sampler.get_sample(pdf);
  return Spectrum();
}

} // namespace CGL
