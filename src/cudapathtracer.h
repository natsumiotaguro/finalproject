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
#include "cudaIntersection.h"
#include "cudabvh.h"

#include "static_scene/cudaScene.h"
#include "static_scene/scene.h"
//#include "static_scene/environment_light.h"
#include "bvh.h"
#include "image.h"
#include "camera.h"
#include "pathtracer.h"

namespace CGL {
struct data_necessary{
	//pathtracer.cpp
	size_t* ns_aa;
	size_t* ns_area_light;
	HDRImageBuffer *sampleBuffer;
	Camera *camera;
	size_t* max_ray_depth;
	CudaSampler2D *gridSampler;
	CudaScene* scene; 

	CudaBVHAccel* bvh;   
};

struct host_data_necessary{
	//pathtracer.cpp
	size_t* ns_aa;
	size_t* ns_area_light;
	HDRImageBuffer *sampleBuffer;
	Camera *camera;
	size_t* max_ray_depth;
	Sampler2D *gridSampler;
	StaticScene::Scene* scene; 

	StaticScene::BVHAccel* bvh;   
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


CudaScene* *cuda_scene;
CudaBVHAccel *bvh;
void raytrace_cuda_tile(int tile_x, int tile_y,
                                int tile_w, int tile_h, HDRImageBuffer *sampleBuffer,
                                size_t imageTileSize, vector<int> *tile_samples,
                                ImageBuffer *frameBuffer);
void testblahlah();

struct data_necessary* cudaMallocNecessary(struct data_necessary* host_data);
__device__ CudaSpectrum trace_cuda_ray( CudaRay &r, bool includeLe, struct data_necessary* cuda_data); 
__device__ CudaSpectrum estimate_direct_lighting( CudaRay& r,  CudaIntersection& isect, struct data_necessary* cuda_data); 
__device__ CudaSpectrum estimate_indirect_lighting( CudaRay& r,  CudaIntersection& isect, struct data_necessary* cuda_data); 

}


#endif // CUDA_PATHTRACER_H