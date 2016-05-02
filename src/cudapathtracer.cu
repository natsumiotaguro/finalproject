#include "cudapathtracer.h"
namespace CGL {
/*   
*   
*
*/
__global__ void raytrace_cuda_pixel_helper(size_t* x, size_t* y, Spectrum* sp,  struct data_necessary* cuda_data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    sp[i].r = 0.30;
    sp[i].g = 0.40;
    sp[i].b = 0.050;


  // Part 1, Task 1:
  // Make a loop that generates num_samples camera rays and traces them 
  // through the scene. Return the average Spectrum. 

  int num_samples = *cuda_data->ns_aa; // total samples to evaluate
  HDRImageBuffer* sampleBuffer = cuda_data->sampleBuffer;
  Camera* camera = cuda_data->camera;

  CudaVector2D origin = CudaVector2D(*x,*y); // bottom left corner of the pixel
  CudaSpectrum average = CudaSpectrum();
  //Loop, for number of samples, get the color
  CudaVector2D sampler = CudaVector2D(0.5, 0.5); //First pixel is always 0.5
  for(int i = 0; i < num_samples; i++){
    CudaVector2D point = CudaVector2D(((double)*x + sampler.x)/sampleBuffer->w, ((double)*y + sampler.y)/sampleBuffer->h);
    CudaRay r = camera->cuda_generate_ray(point.x, point.y);
    r.depth = *cuda_data->max_ray_depth;
    average += trace_cuda_ray(r, true, cuda_data);

    sampler = cuda_data->gridSampler->get_sample(); //For next iteration
    
  }




}

__global__ void instantiate_Necesary(struct data_necessary* data){
	//BVH Instantiation Here
}

//Returns struct with all CUDA pointers
struct data_necessary* cudaMallocNecessary(struct host_data_necessary* data){
    struct data_necessary* cuda_data;
    cudaMalloc((void **) &cuda_data, sizeof(struct data_necessary));

    size_t* ns_aa;
    cudaMalloc((void **) &ns_aa, sizeof(size_t));
    cudaMemcpy(ns_aa, data->ns_aa, sizeof(size_t), cudaMemcpyHostToDevice);
    cuda_data->ns_aa = ns_aa;

    HDRImageBuffer *sampleBuffer;
    cudaMalloc((void **) &sampleBuffer, sizeof(HDRImageBuffer));
    cudaMemcpy(sampleBuffer, data->sampleBuffer, sizeof(HDRImageBuffer), cudaMemcpyHostToDevice);
    cuda_data->sampleBuffer = sampleBuffer;

    Camera *camera;
    cudaMalloc((void **) &camera, sizeof(Camera));
    cudaMemcpy(camera, data->camera, sizeof(Camera), cudaMemcpyHostToDevice);
    cuda_data->camera = camera;

    size_t* max_ray_depth;
    cudaMalloc((void **) &max_ray_depth, sizeof(size_t));
    cudaMemcpy(max_ray_depth, data->max_ray_depth, sizeof(size_t), cudaMemcpyHostToDevice);
    cuda_data->max_ray_depth = max_ray_depth;

    CudaSampler2D *gridSampler;
    cudaMalloc((void **) &gridSampler, sizeof(CudaSampler2D));
    cudaMemcpy(gridSampler, data->gridSampler, sizeof(CudaSampler2D), cudaMemcpyHostToDevice);
    cuda_data->gridSampler = gridSampler;

    
    return cuda_data;
}


void raytrace_cuda_tile(int tile_x, int tile_y,
                                int tile_w, int tile_h, struct host_data_necessary *data, struct no_malloc_necessary *no_data) {

    struct data_necessary* cuda_data = cudaMallocNecessary(data);

    size_t w = cuda_data->sampleBuffer->w;
    size_t h = cuda_data->sampleBuffer->h;

    size_t num_tiles_w = w / no_data->imageTileSize + 1;

    size_t tile_start_x = tile_x;
    size_t tile_start_y = tile_y;

    size_t tile_end_x = std::min(tile_start_x + tile_w, w);
    size_t tile_end_y = std::min(tile_start_y + tile_h, h);

    size_t tile_idx_x = tile_x / no_data->imageTileSize;
    size_t tile_idx_y = tile_y / no_data->imageTileSize;
    size_t num_samples_tile = (*no_data->tile_samples)[tile_idx_x + tile_idx_y * num_tiles_w];

    size_t *host_x, *host_y;
    size_t *dev_x, *dev_y;
    Spectrum *dev_sp;

    size_t tile_length_x = tile_end_x - tile_start_x;
    size_t tile_length_y = tile_end_y - tile_start_y;

    int x_len = sizeof(size_t) * tile_length_x;
    int y_len = sizeof(size_t) * tile_length_y;

    host_x = (size_t *)malloc(x_len);
    host_y = (size_t *)malloc(y_len);

    for (size_t y = 0; y < tile_length_y; y++) {
        host_y[y] = tile_start_y + y;
    }
    for (size_t x = 0; x < tile_length_x; x++) {
        host_x[x] = tile_start_x + x;
    }

    //cudamalloc x, y, spectrum
    cudaMalloc((void **) &dev_x, x_len);
    cudaMalloc((void **) &dev_y, y_len);
    cudaMalloc((void **) &dev_sp, sizeof(Spectrum) * tile_length_x * tile_length_y);

    //cudaMemCpy
    cudaMemcpy(dev_x, &host_x, x_len, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, &host_y, y_len, cudaMemcpyHostToDevice);
    
    int N = tile_length_x;
    int M = tile_length_y;

    //Call helper
    cudaSetDevice(1);
    raytrace_cuda_pixel_helper<<<N,M>>>(dev_x, dev_y, dev_sp, cuda_data);
    cudaDeviceSynchronize();
    //Copy Result
    Spectrum *result = (Spectrum *)malloc(sizeof(Spectrum) * tile_length_x * tile_length_y);

    cudaMemcpy(result, dev_sp, (sizeof(Spectrum) * tile_length_x * tile_length_y), cudaMemcpyDeviceToHost);
    
    for (size_t x = 0; x < tile_length_x; x++) {
        //if (!continueRaytracing) return;
        for (size_t y = 0; y < tile_length_y; y++) {
            data->sampleBuffer->update_pixel(result[x * tile_length_x + y], tile_start_x + x, tile_start_y + y);
        }
    }
   

    //Cleanup - DON'T FORGET TO UN-MALLOC FREE IT
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_sp);
    free(host_x);
    free(host_y);
    free(result);

    (*no_data->tile_samples)[tile_idx_x + tile_idx_y * num_tiles_w] += 1;
    data->sampleBuffer->toColor(*no_data->frameBuffer, tile_start_x, tile_start_y, tile_end_x, tile_end_y);
    }

      int nDevices;

void testblahlah() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Max Threads Per Block: %d\n",
           prop.maxThreadsPerBlock);
    printf("  Multiprocessor Count: %d\n\n",
           prop.multiProcessorCount);
  }
}

__device__ CudaSpectrum trace_cuda_ray( CudaRay &r, bool includeLe, struct data_necessary* cuda_data) {


  CudaIntersection isect;
  CudaSpectrum L_out;

  // You will extend this in part 2. 
  // If no intersection occurs, we simply return black.
  // This changes if you implement hemispherical lighting for extra credit.
  if (!cuda_data->bvh->intersect(r, &isect)) 
    return L_out;

  // This line returns a color depending only on the normal vector 
  // to the surface at the intersection point.
  // Remove it when you are ready to begin Part 3.
  //return normal_shading(isect.n);

  // We only include the emitted light if the previous BSDF was a delta distribution
  // or if the previous ray came from the camera.
  if (includeLe)
    L_out += isect.bsdf->get_emission();

  // You will implement this in part 3. 
  // Delta BSDFs have no direct lighting since they are zero with probability 1 --
  // their values get accumulated through indirect lighting, where the BSDF 
  // gets to sample itself.
  if (!isect.bsdf->is_delta()) 
    L_out += estimate_direct_lighting(r, isect, cuda_data);
  // You will implement this in part 4.
  // If the ray's depth is zero, then the path must terminate
  // and no further indirect lighting is calculated.
  if (r.depth > 0)
    L_out += estimate_indirect_lighting(r, isect, cuda_data);
  
  return L_out;

}
__device__ CudaSpectrum estimate_direct_lighting( CudaRay& r,  CudaIntersection& isect, struct data_necessary* cuda_data) {

// TODO Part 3

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  CudaMatrix3x3 o2w;
  make_cuda_coord_space(o2w, isect.n);
  CudaMatrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  const CudaVector3D& hit_p = r.o + r.d * isect.t;
  const CudaVector3D& w_out = w2o * (-r.d);

  CudaSpectrum L_out = CudaSpectrum();
  for (int j = 0; j < cuda_data->scene->lights_len; j++){
    CudaSceneLight* light = cuda_data->scene->lights[j];
    int num_samples = 1;
    if(light->is_delta_light() == false){//Check if delta light.
      //If yes, ask for one sample
      num_samples = *cuda_data->ns_area_light;
    }
    CudaSpectrum sample_out = CudaSpectrum();
    for(int i = 0; i < num_samples; i++){
        CudaVector3D wi = CudaVector3D();
        float distToLight = 0;
        float pdf = 0;
        CudaSpectrum rad_in = light->sample_L(hit_p, &wi,
                                            &distToLight,
                                            &pdf); //incoming radiance
        CudaVector3D w_in = w2o*wi; //Object space vector
        if(w_in.z > 0){  
          CudaVector3D direction = wi;
          direction.normalize();
          CudaRay shadow = CudaRay(hit_p + EPS_D*direction, direction);
          shadow.max_t = distToLight;
          if(!cuda_data->bvh->intersect(shadow)){
            CudaSpectrum s = isect.bsdf->f(w_out, w_in); //local space
            sample_out += s * (rad_in * fabs(w_in.z))/ pdf;
          }

        }

    }
    L_out += sample_out/num_samples;
  }
  return L_out;



}


} //namespace
