#include "camera.h"

#include "CGL/misc.h"
#include "CGL/vector3D.h"

using std::cout;
using std::endl;
using std::max;
using std::min;

namespace CGL {

using Collada::CameraInfo;

void Camera::configure(const CameraInfo& info, size_t screenW, size_t screenH) {
  this->screenW = screenW;
  this->screenH = screenH;
  nClip = info.nClip;
  fClip = info.fClip;
  hFov = info.hFov;
  vFov = info.vFov;
  /*
  cudaMalloc((void **) &cudac2w, sizeof(CudaMatrix3x3));
  cudaMalloc((void **) &cudaPos, sizeof(CudaVector3D));
  cudaMalloc((void **) &cudaTargetPos, sizeof(CudaVector3D));
  cudaMalloc((void **) &cudahFov, sizeof(double));
  cudaMalloc((void **) &cudavFov, sizeof(double));
  cudaMalloc((void **) &cudaNClip, sizeof(double));
  cudaMalloc((void **) &cudaFClip, sizeof(double));

  cudaMemcpy(cudahFov, &hFov,  sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cudavFov, &vFov,  sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaNClip, &nClip, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaFClip, &fClip, sizeof(double), cudaMemcpyHostToDevice);
*/
  cudaMalloc((void **) &cudahFov, sizeof(double));
  cudaMalloc((void **) &cudavFov, sizeof(double));

    cudaMemcpy(cudahFov, &hFov,  sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cudavFov, &vFov,  sizeof(double), cudaMemcpyHostToDevice);
  
  double ar1 = tan(radians(hFov) / 2) / tan(radians(vFov) / 2);
  ar = static_cast<double>(screenW) / screenH;
  if (ar1 < ar) {
    // hFov is too small
    hFov = 2 * degrees(atan(tan(radians(vFov) / 2) * ar));
  } else if (ar1 > ar) {
    // vFov is too small
    vFov = 2 * degrees(atan(tan(radians(hFov) / 2) / ar));
  }
  screenDist = ((double) screenH) / (2.0 * tan(radians(vFov) / 2));
}

void Camera::place(const Vector3D& targetPos, const double phi,
                   const double theta, const double r, const double minR,
                   const double maxR) {
  double r_ = min(max(r, minR), maxR);
  double phi_ = (sin(phi) == 0) ? (phi + EPS_F) : phi;
  this->targetPos = targetPos;
  this->phi = phi_;
  this->theta = theta;
  this->r = r_;
  this->minR = minR;
  this->maxR = maxR;
  //cudaMemcpy(cudaTargetPos, &targetPos, sizeof(CudaVector3D), cudaMemcpyHostToDevice);

  compute_position();
}

void Camera::copy_placement(const Camera& other) {
  pos = other.pos;
  targetPos = other.targetPos;
  phi = other.phi;
  theta = other.theta;
  minR = other.minR;
  maxR = other.maxR;
  c2w = other.c2w;
 // cudaMemcpy(cudac2w, &c2w, sizeof(CudaMatrix3x3), cudaMemcpyHostToDevice);
 // cudaMemcpy(cudaTargetPos, &targetPos, sizeof(CudaVector3D), cudaMemcpyHostToDevice);
 // cudaMemcpy(cudaPos, &pos, sizeof(CudaVector3D), cudaMemcpyHostToDevice);


}

void Camera::set_screen_size(const size_t screenW, const size_t screenH) {
  this->screenW = screenW;
  this->screenH = screenH;
  ar = 1.0 * screenW / screenH;
  hFov = 2 * degrees(atan(((double) screenW) / (2 * screenDist)));
  vFov = 2 * degrees(atan(((double) screenH) / (2 * screenDist)));
 // cudaMemcpy(cudahFov, &hFov,  sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(cudavFov, &vFov,  sizeof(double), cudaMemcpyHostToDevice);
}

void Camera::move_by(const double dx, const double dy, const double d) {
  const double scaleFactor = d / screenDist;
  const Vector3D& displacement =
    c2w[0] * (dx * scaleFactor) + c2w[1] * (dy * scaleFactor);
  pos += displacement;
  targetPos += displacement;
  //cudaMemcpy(cudaTargetPos, &targetPos, sizeof(CudaVector3D), cudaMemcpyHostToDevice);
  //cudaMemcpy(cudaPos, &pos, sizeof(CudaVector3D), cudaMemcpyHostToDevice);
}

void Camera::move_forward(const double dist) {
  double newR = min(max(r - dist, minR), maxR);
  pos = targetPos + ((pos - targetPos) * (newR / r));
  r = newR;
  //cudaMemcpy(cudaPos, &pos, sizeof(CudaVector3D), cudaMemcpyHostToDevice);
}

void Camera::rotate_by(const double dPhi, const double dTheta) {
  phi = clamp(phi + dPhi, 0.0, (double) PI);
  theta += dTheta;
  compute_position();
}

void Camera::compute_position() {
  double sinPhi = sin(phi);
  if (sinPhi == 0) {
    phi += EPS_F;
    sinPhi = sin(phi);
  }
  const Vector3D dirToCamera(r * sinPhi * sin(theta),
                             r * cos(phi),
                             r * sinPhi * cos(theta));
  pos = targetPos + dirToCamera;
  Vector3D upVec(0, sinPhi > 0 ? 1 : -1, 0);
  Vector3D screenXDir = cross(upVec, dirToCamera);
  screenXDir.normalize();
  Vector3D screenYDir = cross(dirToCamera, screenXDir);
  screenYDir.normalize();

  c2w[0] = screenXDir;
  c2w[1] = screenYDir;
  c2w[2] = dirToCamera.unit();   // camera's view direction is the
                                 // opposite of of dirToCamera, so
                                 // directly using dirToCamera as
                                 // column 2 of the matrix takes [0 0 -1]
                                 // to the world space view direction

  //cudaMemcpy(cudac2w, &c2w, sizeof(CudaMatrix3x3), cudaMemcpyHostToDevice);
  //cudaMemcpy(cudaPos, &pos, sizeof(CudaVector3D), cudaMemcpyHostToDevice);
}

Ray Camera::generate_ray(double x, double y) const {

  // Part 1, Task 2:
  // compute position of the input sensor sample coordinate on the
  // canonical sensor plane one unit away from the pinhole.
  // Note: hFov and vFov are in degrees.
  // 
  Vector3D lower_left  = Vector3D(-tan(radians(hFov)*.5), -tan(radians(vFov)*.5),-1);
  Vector3D upper_right = Vector3D( tan(radians(hFov)*.5),  tan(radians(vFov)*.5),-1);
  Vector3D direction = Vector3D(lower_left.x + x*(upper_right.x - lower_left.x), 
                                 lower_left.y + y*(upper_right.y - lower_left.y),
                                    -1 );
  direction = c2w*direction;
  direction.normalize();
  
  Ray my_ray = Ray(pos, direction);
  my_ray.min_t = nClip;
  my_ray.max_t = fClip;
  return my_ray;

}

__device__ CudaRay Camera::cuda_generate_ray(double x, double y) {

  // Part 1, Task 2:
  // compute position of the input sensor sample coordinate on the
  // canonical sensor plane one unit away from the pinhole.
  // Note: hFov and vFov are in degrees.
  // 
  /*
  double hFovRad = *cudahFov * (PI / 180);
  double vFovRad = *cudavFov * (PI / 180);
  CudaVector3D lower_left  = CudaVector3D(-tan(hFovRad*.5), -tan(vFovRad*.5),-1);
  CudaVector3D upper_right = CudaVector3D( tan(hFovRad*.5),  tan(vFovRad*.5),-1);
  CudaVector3D direction = CudaVector3D(lower_left.x + x*(upper_right.x - lower_left.x), 
                                 lower_left.y + y*(upper_right.y - lower_left.y),
                                    -1 );
  direction = *cudac2w*direction;
  direction.normalize();
  
  CudaRay my_ray = CudaRay(*cudaPos, direction);
  my_ray.min_t = *cudaNClip;
  my_ray.max_t = *cudaFClip;
  */
  return CudaRay(CudaVector3D(0, 0, 0), CudaVector3D(0, 0, 0)); //my_ray;

}




} // namespace CGL
