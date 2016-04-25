//#include <vector>
//#include "CGL/vector3D.h"
#ifndef CUDA_PATHTRACER_H
#define CUDA_PATHTRACER_H
#include "CGL/spectrum.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "pathtracer.h"


namespace CGL {
Spectrum raytrace_cuda_pixel(size_t x, size_t y);
void raytrace_cuda_tile(int tile_x, int tile_y,
                                int tile_w, int tile_h, HDRImageBuffer *sampleBuffer,
                                size_t imageTileSize, vector<int> *tile_samples,
                                ImageBuffer *frameBuffer);

}

#endif // CUDA_PATHTRACER_H