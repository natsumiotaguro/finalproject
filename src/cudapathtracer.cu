#include "cudapathtracer.h"

namespace CGL {

/*   
*   
*
*/
__global__ void raytrace_cuda_pixel_helper(size_t* x, size_t* y, Spectrum* sp){
	sp->r = 0.30;
	sp->g = 0.40;
	sp->b = 0.050;
}

Spectrum raytrace_cuda_pixel(size_t x, size_t y){
	//PUT CUDA MALLOC STUFF HERE
	//size_t *host_x, *host_y; //Host xy
	size_t *dev_x, *dev_y; //Device x y
	Spectrum *dev_sp;
	int size_tsize = sizeof(size_t);
	//malloc host_x, host_y
	//host_x = malloc(size_tsize);
	// = malloc(size_tsize);

	//cudamalloc x, y, spectrum
	cudaError_t err = cudaMalloc((void **) &dev_x, size_tsize);
	if (err != cudaSuccess){
		printf("%s1\n", cudaGetErrorString(err));
	}
	err = cudaMalloc((void **) &dev_y, size_tsize);
		if (err != cudaSuccess){
		printf("%s2\n", cudaGetErrorString(err));
	}
	err = cudaMalloc((void **) &dev_sp, sizeof(Spectrum));
	if (err != cudaSuccess){
		printf("%s3\n", cudaGetErrorString(err));
	}
	//cudaMemCpy
	err = cudaMemcpy(dev_x, &x, sizeof(size_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		printf("%s4\n", cudaGetErrorString(err));
	}
	err = cudaMemcpy(dev_y, &y, sizeof(size_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		printf("%s5\n", cudaGetErrorString(err));
	}
	//Call helper
	raytrace_cuda_pixel_helper<<<1,1>>>(dev_x, dev_y, dev_sp);

	//Copy Result
	Spectrum *result = (Spectrum *)malloc(sizeof(Spectrum));

	err = cudaMemcpy(result, dev_sp, sizeof(Spectrum), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		printf("%s\n", cudaGetErrorString(err));
	}
	//Cleanup - DON'T FORGET TO UN-MALLOC FREEE IT
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_sp);
	//free(host_x);
	//free(host_y);
	return *result;
}

/*
Spectrum PathTracer::raytrace_pixel(size_t x, size_t y) {

  // Part 1, Task 1:
  // Make a loop that generates num_samples camera rays and traces them 
  // through the scene. Return the average Spectrum. 

  int num_samples = ns_aa; // total samples to evaluate
  Vector2D origin = Vector2D(x,y); // bottom left corner of the pixel
  Spectrum average = Spectrum();
  //Loop, for number of samples, get the color
  Vector2D sampler = Vector2D(0.5, 0.5); //First pixel is always 0.5
  for(int i = 0; i < num_samples; i++){
    Vector2D point = Vector2D(((double)x + sampler.x)/sampleBuffer.w, ((double)y + sampler.y)/sampleBuffer.h);
    Ray r = camera->generate_ray(point.x, point.y);
    r.depth = max_ray_depth;
    average += trace_ray(r, true);

    sampler = gridSampler->get_sample(); //For next iteration
    
  }

}

*/

} //namespace CGL