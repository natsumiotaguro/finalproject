#include "bsdf.h"

#include <iostream>
#include <algorithm>
#include <utility>

using std::min;
using std::max;
using std::swap;

void make_cuda_coord_space(CudaMatrix3x3& o2w, const CudaVector3D& n) {

    CudaVector3D z = CudaVector3D(n.x, n.y, n.z);
    CudaVector3D h = z;
    if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z)) h.x = 1.0;
    else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z)) h.y = 1.0;
    else h.z = 1.0;

    z.normalize();
    CudaVector3D y = cross(h, z);
    y.normalize();
    CudaVector3D x = cross(z, y);
    x.normalize();

    o2w[0] = x;
    o2w[1] = y;
    o2w[2] = z;
}

// Diffuse BSDF //

__device__ CudaDiffuseBSDF::CudaDiffuseBSDF(const CudaSpectrum& a) {
  this->albedo = a;
}

CudaSpectrum CudaDiffuseBSDF::f(const CudaVector3D& wo, const CudaVector3D& wi) {
  return albedo * (1.0 / PI);
}

CudaSpectrum CudaDiffuseBSDF::sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf) {
  *wi = sampler.get_sample(pdf);
  return albedo * (1.0 / PI);
}

// Mirror BSDF //
//local coordinate system at the
//   * point of intersection.
__device__ CudaMirrorBSDF::CudaMirrorBSDF(const CudaSpectrum& reflectance) {
  this->reflectance = reflectance;
}

CudaSpectrum CudaMirrorBSDF::f(const CudaVector3D& wo, const CudaVector3D& wi) {
  //not used not used NOT USED
  printf("USEING \n");
  return CudaSpectrum();
}

CudaSpectrum CudaMirrorBSDF::sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf) {

  // TODO Part 5:
  // Implement MirrorBSDF
  // reflectance in the output incident and given outgoing directions
  *pdf = 1;
  CudaBSDF::reflect(wo, wi); //reflectance divided by the cosine of wi with the normal.
  CudaSpectrum result = reflectance/cuda_abs_cos_theta(wo);
  return reflectance/cuda_abs_cos_theta(*wi);
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

__device__ CudaRefractionBSDF(const CudaSpectrum& transmittance, float roughness, float ior) {
  this->transmittance = transmittance;
  this->roughness = roughness;
  this->ior = ior;
}

CudaSpectrum CudaRefractionBSDF::f(const CudaVector3D& wo, const CudaVector3D& wi) {
    return CudaSpectrum(); //NOT USED NOT USED NOT USED
  
}

CudaSpectrum CudaRefractionBSDF::sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf) {
  // TODO Part 5: NOT USED NOT USED NOT USED
  // Implement RefractionBSDF     
    return CudaSpectrum();
}

// Glass BSDF //

__device__ GlassBSDF::GlassBSDF(const CudaSpectrum& transmittance, const CudaSpectrum& reflectance,
          float roughness, float ior) {
  this->transmittance = transmittance;
  this->reflectance = reflectance;
  this->roughness = roughness;
  this->ior = ior;
}

CudaSpectrum CudaGlassBSDF::f(const CudaVector3D& wo, const CudaVector3D& wi) {
  CudaVector3D tmp = CudaVector3D();
  printf("hohoho\n");
  double is_internal = !refract(wo, &tmp, ior);
  if(is_internal){
    return reflectance/cuda_abs_cos_theta(wi);
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
    double cos_wi = dot(wo, CudaVector3D(0, 0, 1));
    double r = cuda_clamp(r0 + (1 - r0)*(pow(1-cos_wi, 5)), 0, 1); //r0 + (1-R0)(1 - cos(theta_0))^5
    
    double is_reflection = coin_flip(r);    //coin flip, decide reflection or refraction
    
    if(is_reflection == true){
      return r*reflectance/fabs(wi.z);
    }
    else{
      return (1 - r)*transmittance*pow((no/ni), 2)/fabs(wi.z); 
      //return refr(di), 1-R*Li(no/ni)^2/cos(thetai), 1-R)
    }
  }
  return CudaSpectrum();
}

CudaSpectrum CudaGlassBSDF::sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf) {
    CudaVector3D tmp_wo = -1*wo;
    
    double vacuum_check = wo.z;
    double ni = 1;
    double no = ior;
    if(wo.z < 0){
      ni = ior;
      no = 1.0;
    }

    double r0 = pow((ni - no)/(ni + no), 2);
    double cos_i = -1*dot(tmp_wo, CudaVector3D(0, 0, 1)); //-1*dot(-1*wo, Vector3D(0, 0, 1));
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
        return reflectance/cuda_abs_cos_theta(*wi); 
      }
      cos_i = sqrt(1.0 - sin2);
    }
    //printf("sin 2 in the large %f\n", sin2);
    //printf("Cos i %f\n", cos_i);
    double r = cuda_clamp(r0 + (1 - r0)*(pow(1-cos_i, 5)), 0, 1); //r0 + (1-R0)(1 - cos(theta_0))^5
    double is_reflection = coin_flip(r);    //coin flip, decide reflection or refraction
   
    //is_reflection = false; //TAKE IT OUT TAKE IT OUT TAKE IT OUT
    //r = 0.1;

    if(is_reflection == true){
      *pdf = r;
      reflect(wo, wi);
      //printf("Coin flip refl: %f\n", r);
      return r*reflectance/cuda_abs_cos_theta(*wi);
    }
    else{
      *pdf = 1 - r;
      refract(wo, wi, ior);
   
       //already calculated refraction
      //printf("GOOOD%f , %f,  %f\n", 1-r, abs_cos_theta(*wi)), fabs(wi->z);
      CudaSpectrum tmp_s = (1 - r)*transmittance*pow((no/ni), 2)/cuda_abs_cos_theta(*wi); //fabs(cos_i); 
      return transmittance/fabs(wi->z);
      //return refr(di), 1-R*Li(no/ni)^2/cos(thetai), 1-R)
    }
 // }
 // return Spectrum();
}

void CudaBSDF::reflect(const CudaVector3D& wo, CudaVector3D* wi) {
  // TODO Part 5:
  // Implement reflection of wo about normal (0,0,1) and store result in wi.
  //*wi = wo; //This makes the ray go the other direction
  *wi = -wo + 2 *dot(CudaVector3D(0, 0, 1), wo) * CudaVector3D(0, 0, 1);
  //wi->z = -1*wo.z;
}

bool CudaBSDF::refract(const CudaVector3D& wo, CudaVector3D* wi, float ior) {
  // TODO Part 5:
  // Use Snell's Law to refract wo surface and store result ray in wi.
  // Return false if refraction does not occur due to total internal reflection
  // and true otherwise. When dot(wo,n) is positive, then wo corresponds to a
  // ray entering the surface through vacuum.
 CudaVector3D tmp_wo = -1*wo;
    double vacuum_check = wo.z;
    double ni = 1;
    double no = ior;
    if(wo.z < 0){
      ni = ior;
      no = 1.0;
    }

    double cos_i = -1*dot(tmp_wo, CudaVector3D(0, 0, 1)); //-1*dot(-1*wo, Vector3D(0, 0, 1));
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
  *wi = (io_ratio * tmp_wo + (io_ratio*cos_i - cos2) * CudaVector3D(0, 0, 1));
  //wi->x = io_ratio * wo.x;
  //wi->y = io_ratio * wo.y;
  //wi->z = io_ratio * wo.z; 
  //wi->z += (io_ratio*cos_o - cos2); //n * cos I - cos t
  return true;

}

// Emission BSDF //

__device__ CudaEmissionBSDF(const CudaSpectrum& radiance) {
  this->radiance = radiance;
}


CudaSpectrum CudaEmissionBSDF::f(const CudaVector3D& wo, const CudaVector3D& wi) {
  return CudaSpectrum();
}

CudaSpectrum CudaEmissionBSDF::sample_f(const CudaVector3D& wo, CudaVector3D* wi, float* pdf) {
  *pdf = 1.0 / PI;
  *wi  = sampler.get_sample(pdf);
  return CudaSpectrum();
}

