#ifndef CGL_CUDACOLOR_H
#define CGL_CUDACOLOR_H

//#include "CGL.h"

#include <string>
#include "helper_math.h"


/**
 * Encodes a color via additive red, green, and blue chanel values.
 * Each color chanel value is in the range [0,1]. The alpha value
 * defines the transparency of the color and is also in [0,1].
 */
class CudaColor {
 public:

  // Components.
  float r; /**< value of red chanel   */
  float g; /**< value of green chanel */
  float b; /**< value of blue chanel  */
  float a; /**< value of alpha chanel */

  // constants
  static const CudaColor White;
  static const CudaColor Black;

  /**
   * Constructor.
   * Initialize from component values.
   * By default, alpha is initialized to 1.
   * \param r Value of the red chanel.
   * \param g Value of the green chanel.
   * \param b Value of the blue chanel.
   * \param a Value of the alpha chanel.
   */
  __device__ CudaColor( float r = 0, float g = 0, float b = 0, float a = 1.0 )
      : r( r ), g( g ), b( b ), a( a ) { }

  /**
   * Constructor.
   * Initialize from array of 8-bit component values (RGB only).
   * \param arr Array containing component values.
   */
  __device__ CudaColor( const unsigned char* arr );

  // Addition. Alpha is ignored.
  __device__ inline CudaColor operator+( const CudaColor& rhs ) const {
    return CudaColor( r + rhs.r, g + rhs.g, b + rhs.b, a + rhs.a);
  }

  __device__ inline CudaColor& operator+=( const CudaColor& rhs ) {
    r += rhs.r; g += rhs.g; b += rhs.b; a += rhs.a;
    return *this;
  }

  // Vector multiplication. Alpha is ignored.
  __device__ inline CudaColor operator*( const CudaColor& rhs ) const {
    return CudaColor( r * rhs.r, g * rhs.g, b * rhs.b, a * rhs.a);
  }

  __device__ inline CudaColor& operator*=( const CudaColor& rhs ) {
    r *= rhs.r; g *= rhs.g; b *= rhs.b; a *= rhs.a;
    return *this;
  }

  // Scalar multiplication.
  __device__ inline CudaColor operator*( float s ) const {
    return CudaColor( r * s, g * s, b * s, a * s );
  }

  __device__ inline CudaColor& operator*=( float s ) {
    r *= s; g *= s; b *= s; a *= s;
    return *this;
  }

  // comparison
  __device__ inline bool operator==( const CudaColor& rhs ) const {
    return r == rhs.r && g == rhs.g && b == rhs.b && a == rhs.a;
  }

  __device__ inline bool operator!=( const CudaColor& rhs ) const {
    return !operator==( rhs );
  }

  /**
   * Construct a Color object from a hexadecimal (8-bit per
   * component) ASCII string.
   * Since hexademical strings are typically not used to encode
   * alpha values, the alpha is set to 1 (opaque) by default.
   * This method also accepts the string "none", in which case
   * it returns a color value with alpha zero (transparent).
   * \return Color constructed from the input hex encoding.
   */
  //static CudaColor fromHex( const char* s );

  /**
   * Returns a hexadecimal string #rrggbb encoding this color.
   * \return the hexadecimal encoding of the color.
   */
  //std::string toHex( ) const;


}; // class CudaColor


// Commutable scalar multiplication.
__device__ inline CudaColor operator*( float s, const CudaColor& c ) {
  return c * s;
}

// Prints components.
//std::ostream& operator<<( std::ostream& os, const Color& c );


#endif //CGL_CUDACOLOR_H

