#include "cudaColor.h"

#include <stdio.h>
#include <string.h>
#include <ostream>
#include <sstream>

using namespace std;

// Constants
const CudaColor CudaColor::White  = CudaColor(1,1,1,1);
const CudaColor CudaColor::Black  = CudaColor(0,0,0,1);

__device__ CudaColor::CudaColor( const unsigned char* arr ) {
  float inv = 1.0 / 255.0;
  r = arr[0] * inv;
  g = arr[1] * inv;
  b = arr[2] * inv;
  a = 1.0;
}
/*
CudaColor CudaColor::fromHex( const char* s ) {
  // If the color is "none", return any color
  // with alpha zero (completely transparent).
  if( !strcmp(s, "none") ) {
      return Color(0,0,0,0);
  }

  // Ignore leading hashmark.
  if( s[0] == '#' ) {
      s++;
  }

  // Set stream formatting to hexadecimal.
  stringstream ss;
  ss << hex;

  // Convert to integer.
  unsigned int rgb;
  ss << s;
  ss >> rgb;

  // Extract 8-byte chunks and normalize.
  CudaColor c;
  c.r = (float)( ( rgb & 0xFF0000 ) >> 16 ) / 255.0;
  c.g = (float)( ( rgb & 0x00FF00 ) >>  8 ) / 255.0;
  c.b = (float)( ( rgb & 0x0000FF ) >>  0 ) / 255.0;
  c.a = 1.0; // set alpha to 1 (opaque) by default

  return c;
}
*/

/*
string CudaColor::toHex( void ) const {
  int R = (unsigned char) fmax( 0., min( 255.0, 255.0 * r ));
  int G = (unsigned char) fmax( 0., min( 255.0, 255.0 * g ));
  int B = (unsigned char) fmax( 0., min( 255.0, 255.0 * b ));

  stringstream ss;
  ss << hex;

  ss << R << G << B;
  return ss.str();
}
*/

/*
std::ostream& operator<<( std::ostream& os, const Color& c ) {
  os << "(r=" << c.r;
  os << " g=" << c.g;
  os << " b=" << c.b;
  os << " a=" << c.a;
  os << ")";
  return os;
}
*/


