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


namespace CGL {

struct data_necessary{

	//pathtracer.cpp
	size_t* ns_aa;
	HDRImageBuffer *sampleBuffer;
	Camera *camera;
	size_t max_ray_depth;
	Sampler2D *gridSampler;
	

	//camera.cpp
	// double *hFov;
	// double *vFov;
	// Matrix3x3 *c2w;
	// Vector3D *pos;
	// double *nClip;
	// double *fClip;

	// //bvh


	// /////////////////////////
	// HDRImageBuffer *sampleBuffer;
	// size_t* imageTileSize;
	// vector<int> *tile_samples;
	// ImageBuffer *frameBuffer



};

struct data_necessary* cuda_data;

Spectrum raytrace_cuda_pixel(size_t x, size_t y);
void raytrace_cuda_tile(int tile_x, int tile_y,
                                int tile_w, int tile_h, HDRImageBuffer *sampleBuffer,
                                size_t imageTileSize, vector<int> *tile_samples,
                                ImageBuffer *frameBuffer);
void testblahlah();

void cudaMallocNecessary(struct data_necessary* host_data);


}

#endif // CUDA_PATHTRACER_H