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
#include "cudaSampler.h"

#include "image.h"
#include "camera.h"
#include "pathtracer.h"

namespace CGL {
struct data_necessary{
	//pathtracer.cpp
	size_t* ns_aa;
	HDRImageBuffer *sampleBuffer;
	Camera *camera;
	size_t* max_ray_depth;
	CudaSampler2D *gridSampler;

};

struct no_malloc_necessary{

	size_t imageTileSize;
	vector<int> *tile_samples;
	ImageBuffer *frameBuffer;
};

	// //bvh


	// /////////////////////////
	// HDRImageBuffer *sampleBuffer;
	// size_t* imageTileSize;
	// vector<int> *tile_samples;
	// ImageBuffer *frameBuffer




void raytrace_cuda_tile(int tile_x, int tile_y,
                                int tile_w, int tile_h, HDRImageBuffer *sampleBuffer,
                                size_t imageTileSize, vector<int> *tile_samples,
                                ImageBuffer *frameBuffer);
void testblahlah();

struct data_necessary* cudaMallocNecessary(struct data_necessary* host_data);
__device__ CudaSpectrum trace_cuda_ray(const CudaRay &r, bool includeLe); 
}


#endif // CUDA_PATHTRACER_H