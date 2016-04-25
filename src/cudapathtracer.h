//#include <vector>
//#include "CGL/vector3D.h"
#include "CGL/spectrum.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace CGL {
Spectrum raytrace_cuda_pixel(size_t x, size_t y);

}