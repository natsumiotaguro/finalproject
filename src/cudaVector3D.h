#ifndef CGL_CUDAVECTOR3D_H
#define CGL_CUDAVECTOR3D_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <ostream>
#include <cmath>


/**
 * Defines 3D vectors.
 */
class CudaVector3D {
 public:

  // components
  double x, y, z;

  /**
   * Constructor.
   * Initializes tp vector (0,0,0).
   */
  __device__ CudaVector3D() : x( 0.0 ), y( 0.0 ), z( 0.0 ) { }

  /**
   * Constructor.
   * Initializes to vector (x,y,z).
   */
  __device__ CudaVector3D( double x, double y, double z) : x( x ), y( y ), z( z ) { }

  /**
   * Constructor.
   * Initializes to vector (c,c,c)
   */
  __device__ CudaVector3D( double c ) : x( c ), y( c ), z( c ) { }

  /**
   * Constructor.
   * Initializes from existing vector
   */
  __device__ CudaVector3D( const CudaVector3D& v ) : x( v.x ), y( v.y ), z( v.z ) { }

  // returns reference to the specified component (0-based indexing: x, y, z)
  __device__ inline double& operator[] ( const int& index ) {
    return ( &x )[ index ];
  }

  // returns const reference to the specified component (0-based indexing: x, y, z)
  __device__ inline const double& operator[] ( const int& index ) const {
    return ( &x )[ index ];
  }

  __device__ inline bool operator==( const CudaVector3D& v) const {
    return v.x == x && v.y == y && v.z == z;
  }

  // negation
  __device__ inline CudaVector3D operator-( void ) const {
    return CudaVector3D( -x, -y, -z );
  }

  // addition
  __device__ inline CudaVector3D operator+( const CudaVector3D& v ) const {
    return CudaVector3D( x + v.x, y + v.y, z + v.z );
  }

  // subtraction
  __device__ inline CudaVector3D operator-( const CudaVector3D& v ) const {
    return CudaVector3D( x - v.x, y - v.y, z - v.z );
  }

  // right scalar multiplication
  __device__ inline CudaVector3D operator*( const double& c ) const {
    return CudaVector3D( x * c, y * c, z * c );
  }

  // scalar division
  __device__ inline CudaVector3D operator/( const double& c ) const {
    const double rc = 1.0/c;
    return CudaVector3D( rc * x, rc * y, rc * z );
  }

  // addition / assignment
  __device__ inline void operator+=( const CudaVector3D& v ) {
    x += v.x; y += v.y; z += v.z;
  }

  // subtraction / assignment
  __device__ inline void operator-=( const CudaVector3D& v ) {
    x -= v.x; y -= v.y; z -= v.z;
  }

  // scalar multiplication / assignment
  __device__ inline void operator*=( const double& c ) {
    x *= c; y *= c; z *= c;
  }

  // scalar division / assignment
  __device__ inline void operator/=( const double& c ) {
    (*this) *= ( 1./c );
  }

  /**
   * Returns Euclidean length.
   */
  __device__ inline double norm( void ) const {
    return sqrt( x*x + y*y + z*z );
  }

  /**
   * Returns Euclidean length squared.
   */
  __device__ inline double norm2( void ) const {
    return x*x + y*y + z*z;
  }

  /**
   * Returns unit vector.
   */
  __device__ inline CudaVector3D unit( void ) const {
    double rNorm = 1. / sqrt( x*x + y*y + z*z );
    return CudaVector3D( rNorm*x, rNorm*y, rNorm*z );
  }

  /**
   * Divides by Euclidean length.
   */
  __device__ inline void normalize( void ) {
    (*this) /= norm();
  }

}; // class Vector3D

// left scalar multiplication
__device__ inline CudaVector3D operator* ( const double& c, const CudaVector3D& v ) {
  return CudaVector3D( c * v.x, c * v.y, c * v.z );
}

// dot product (a.k.a. inner or scalar product)
__device__ inline double dot( const CudaVector3D& u, const CudaVector3D& v ) {
  return u.x*v.x + u.y*v.y + u.z*v.z ;
}

// cross product
__device__ inline CudaVector3D cross( const CudaVector3D& u, const CudaVector3D& v ) {
  return CudaVector3D( u.y*v.z - u.z*v.y,
                   u.z*v.x - u.x*v.z,
                   u.x*v.y - u.y*v.x );
}

// prints components
//std::ostream& operator<<( std::ostream& os, const Vector3D& v );


#endif // CGL_CUDAVECTOR3D_H
