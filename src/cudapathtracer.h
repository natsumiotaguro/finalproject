//#include <vector>
//#include "CGL/vector3D.h"
#ifndef CUDA_PATHTRACER_H
#define CUDA_PATHTRACER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "pathtracer.h"

#include "cudaVector2D.h"
#include "cudaVector3D.h"
#include "cudaSpectrum.h"
#include "camera.h"
#include "pathtracer.h"

struct data_necessary{
	//pathtracer.cpp
	size_t* ns_aa;
	HDRImageBuffer *sampleBuffer;
	Camera *camera;
	size_t max_ray_depth;
	Sampler2D *gridSampler;

};


	// //bvh


	// /////////////////////////
	// HDRImageBuffer *sampleBuffer;
	// size_t* imageTileSize;
	// vector<int> *tile_samples;
	// ImageBuffer *frameBuffer



Spectrum raytrace_cuda_pixel(size_t x, size_t y);
void raytrace_cuda_tile(int tile_x, int tile_y,
                                int tile_w, int tile_h, HDRImageBuffer *sampleBuffer,
                                size_t imageTileSize, vector<int> *tile_samples,
                                ImageBuffer *frameBuffer);
void testblahlah();

void cudaMallocNecessary(struct data_necessary* host_data);
__device__ Spectrum trace_cuda_ray(const CudaRay &r, bool includeLe);


#endif // CUDA_PATHTRACER_H