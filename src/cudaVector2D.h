#ifndef CGL_CUDAVECTOR2D_H
#define CGL_CUDAVECTOR2D_H

#include <ostream>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Defines 2D vectors.
 */
class CudaVector2D {
 public:

  // components
  double x, y;

  /**
   * Constructor.
   * Initializes to vector (0,0).
   */
  __device__ CudaVector2D() : x( 0.0 ), y( 0.0 ) { }

  /**
   * Constructor.
   * Initializes to vector (a,b).
   */
  __device__ CudaVector2D( double x, double y ) : x( x ), y( y ) { }

  /**
   * Constructor.
   * Copy constructor. Creates a copy of the given vector.
   */
  __device__ CudaVector2D( const CudaVector2D& v ) : x( v.x ), y( v.y ) { }

  // additive inverse
  __device__ inline CudaVector2D operator-( void ) const {
    return CudaVector2D( -x, -y );
  }

  // addition
  __device__ inline CudaVector2D operator+( const CudaVector2D& v ) const {
    CudaVector2D u = *this;
    u += v;
    return u;
  }

  // subtraction
  __device__ inline CudaVector2D operator-( const CudaVector2D& v ) const {
    CudaVector2D u = *this;
    u -= v;
    return u;
  }

  // right scalar multiplication
  __device__ inline CudaVector2D operator*( double r ) const {
    CudaVector2D vr = *this;
    vr *= r;
    return vr;
  }

  // scalar division
  __device__ inline CudaVector2D operator/( double r ) const {
    CudaVector2D vr = *this;
    vr /= r;
    return vr;
  }

  // add v
  __device__ inline void operator+=( const CudaVector2D& v ) {
    x += v.x;
    y += v.y;
  }

  // subtract v
  __device__ inline void operator-=( const CudaVector2D& v ) {
    x -= v.x;
    y -= v.y;
  }

  // scalar multiply by r
  __device__ inline void operator*=( double r ) {
    x *= r;
    y *= r;
  }

  // scalar divide by r
  __device__ inline void operator/=( double r ) {
    x /= r;
    y /= r;
  }

  /**
   * Returns norm.
   */
  __device__ inline double norm( void ) const {
    return sqrt( x*x + y*y );
  }

  /**
   * Returns norm squared.
   */
  __device__ inline double norm2( void ) const {
    return x*x + y*y;
  }

  /**
   * Returns unit vector parallel to this one.
   */
  __device__ inline CudaVector2D unit( void ) const {
    return *this / this->norm();
  }


}; // clasd Vector2D

// left scalar multiplication
__device__ inline CudaVector2D operator*( double r, const CudaVector2D& v ) {
   return v*r;
}

// inner product
__device__ inline double dot( const CudaVector2D& v1, const CudaVector2D& v2 ) {
  return v1.x*v2.x + v1.y*v2.y;
}

// cross product
__device__ inline double cross( const CudaVector2D& v1, const CudaVector2D& v2 ) {
  return v1.x*v2.y - v1.y*v2.x;
}

// prints components
//__device__ std::ostream& operator<<( std::ostream& os, const CudaVector2D& v );

#endif // CGL_CUDAVECTOR2D_H
