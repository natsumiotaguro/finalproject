#ifndef CGL_CUDAMATRIX3X3_H
#define CGL_CUDAMATRIX3X3_H

#include "cudaVector3D.h"

#include <iosfwd>


/**
 * Defines a 3x3 matrix.
 * 3x3 matrices are extremely useful in computer graphics.
 */
class CudaMatrix3x3 {

  public:

  // The default constructor.
  __device__ CudaMatrix3x3(void) { }

  // Constructor for row major form data.
  // Transposes to the internal column major form.
  // REQUIRES: data should be of size 9 for a 3 by 3 matrix..
  __device__ CudaMatrix3x3(double * data)
  {
    for( int i = 0; i < 3; i++ ) {
      for( int j = 0; j < 3; j++ ) {
	        // Transpostion happens within the () query.
	        (*this)(i,j) = data[i*3 + j];
      }
    }
  }

  /**
   * Sets all elements to val.
   */
  __device__ void zero(double val = 0.0 );

  /**
   * Returns the determinant of A.
   */
  __device__ double det( void ) const;

  /**
   * Returns the Frobenius norm of A.
   */
  __device__ double norm( void ) const;

  /**
   * Returns the 3x3 identity matrix.
   */
  __device__ static CudaMatrix3x3 identity( void );

  /**
   * Returns a matrix representing the (left) cross product with u.
   */
  __device__ static CudaMatrix3x3 crossProduct( const CudaVector3D& u );

  /**
   * Returns the ith column.
   */
  __device__       CudaVector3D& column( int i );
  __device__ const CudaVector3D& column( int i ) const;

  /**
   * Returns the transpose of A.
   */
  __device__ CudaMatrix3x3 T( void ) const;

  /**
   * Returns the inverse of A.
   */
  __device__ CudaMatrix3x3 inv( void ) const;

  // accesses element (i,j) of A using 0-based indexing
        __device__ double& operator()( int i, int j );
  __device__ const double& operator()( int i, int j ) const;

  // accesses the ith column of A
        __device__ CudaVector3D& operator[]( int i );
  __device__ const CudaVector3D& operator[]( int i ) const;

  // increments by B
  __device__ void operator+=( const CudaMatrix3x3& B );

  // returns -A
  __device__ CudaMatrix3x3 operator-( void ) const;

  // returns A-B
  __device__ CudaMatrix3x3 operator-( const CudaMatrix3x3& B ) const;

  // returns c*A
  __device__ CudaMatrix3x3 operator*( double c ) const;

  // returns A*B
  __device__ CudaMatrix3x3 operator*( const CudaMatrix3x3& B ) const;

  // returns A*x
  __device__ CudaVector3D operator*( const CudaVector3D& x ) const;

  // divides each element by x
  __device__ void operator/=( double x );

  protected:

  // column vectors
  CudaVector3D entries[3];

}; // class Matrix3x3

// returns the outer product of u and v
__device__ CudaMatrix3x3 outer( const CudaMatrix3x3& u, const CudaMatrix3x3& v );

// returns c*A
__device__ CudaMatrix3x3 operator*( double c, const CudaMatrix3x3& A );

// prints entries
//std::ostream& operator<<( std::ostream& os, const CudaMatrix3x3& A );


#endif // CGL_CUDAMATRIX3X3_H
