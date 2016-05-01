#include "cudapathtracer.h"

/*   
*   
*
*/
__global__ void raytrace_cuda_pixel_helper(size_t* x, size_t* y, Spectrum* sp){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    sp[i].r = 0.30;
    sp[i].g = 0.40;
    sp[i].b = 0.050;


  // Part 1, Task 1:
  // Make a loop that generates num_samples camera rays and traces them 
  // through the scene. Return the average Spectrum. 

  int num_samples = *ns_aa; // total samples to evaluate
  CudaVector2D origin = CudaVector2D(*x,*y); // bottom left corner of the pixel
  CudaSpectrum average = CudaSpectrum();
  //Loop, for number of samples, get the color
  CudaVector2D sampler = CudaVector2D(0.5, 0.5); //First pixel is always 0.5
  for(int i = 0; i < num_samples; i++){
    CudaVector2D point = CudaVector2D(((double)x + sampler.x)/sampleBuffer.w, ((double)y + sampler.y)/sampleBuffer.h);
    CudaRay r = camera->cuda_generate_ray(point.x, point.y);
    r.depth = max_ray_depth;
    average += trace_cuda_ray(r, true);

    sampler = gridSampler->get_sample(); //For next iteration
    
  }




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

//Returns struct with all CUDA pointers
struct data_necessary* cudaMallocNecessary(struct data_necessary* data){
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

    size_t max_ray_depth;
    cudaMalloc((void **) &max_ray_depth, sizeof(size_t));
    cudaMemcpy(max_ray_depth, data->max_ray_depth, sizeof(size_t), cudaMemcpyHostToDevice);
    cuda_data->max_ray_depth = max_ray_depth;

    Sampler2D *gridSampler;
    cudaMalloc((void **) &max_ray_depth, sizeof(size_t));
    cudaMemcpy(max_ray_depth, data->max_ray_depth, sizeof(size_t), cudaMemcpyHostToDevice);
    cuda_data->max_ray_depth = max_ray_depth;




    size_t* imageTileSize;
    cudaMalloc((void **) &imageTileSize, sizeof(size_t));
    cudaMemcpy(imageTileSize, data->imageTileSize, sizeof(size_t), cudaMemcpyHostToDevice);
    cuda_data->imageTileSize = imageTileSize;    

    vector<int> *tile_samples;
    cudaMalloc((void **) &tile_samples, sizeof(vector<int>));
    cudaMemcpy(tile_samples, data->tile_samples, x_len, cudaMemcpyHostToDevice);
    cuda_data->imageTileSize = imageTileSize;    


    ImageBuffer *frameBuffer
    cudaMalloc((void **) &dev_x, x_len);
    cudaMemcpy(dev_x, &host_x, x_len, cudaMemcpyHostToDevice);
    
}


void raytrace_cuda_tile(int tile_x, int tile_y,
                                int tile_w, int tile_h, struct data_necessary data) {

    cudaMallocNecessary(data);

    size_t w = sampleBuffer->w;
    size_t h = sampleBuffer->h;

    size_t num_tiles_w = w / imageTileSize + 1;

    size_t tile_start_x = tile_x;
    size_t tile_start_y = tile_y;

    size_t tile_end_x = std::min(tile_start_x + tile_w, w);
    size_t tile_end_y = std::min(tile_start_y + tile_h, h);

    size_t tile_idx_x = tile_x / imageTileSize;
    size_t tile_idx_y = tile_y / imageTileSize;
    size_t num_samples_tile = (*tile_samples)[tile_idx_x + tile_idx_y * num_tiles_w];

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
    raytrace_cuda_pixel_helper<<<N,M>>>(dev_x, dev_y, dev_sp);
    cudaDeviceSynchronize();
    //Copy Result
    Spectrum *result = (Spectrum *)malloc(sizeof(Spectrum) * tile_length_x * tile_length_y);

    cudaMemcpy(result, dev_sp, (sizeof(Spectrum) * tile_length_x * tile_length_y), cudaMemcpyDeviceToHost);
    
    for (size_t x = 0; x < tile_length_x; x++) {
        //if (!continueRaytracing) return;
        for (size_t y = 0; y < tile_length_y; y++) {
            sampleBuffer->update_pixel(result[x * tile_length_x + y], tile_start_x + x, tile_start_y + y);
        }
    }
   

    //Cleanup - DON'T FORGET TO UN-MALLOC FREE IT
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_sp);
    free(host_x);
    free(host_y);
    free(result);

    (*tile_samples)[tile_idx_x + tile_idx_y * num_tiles_w] += 1;
    sampleBuffer->toColor(*frameBuffer, tile_start_x, tile_start_y, tile_end_x, tile_end_y);
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

__device__ Spectrum trace_cuda_ray(const CudaRay &r, bool includeLe) {

  Intersection isect;
  Spectrum L_out;

  // You will extend this in part 2. 
  // If no intersection occurs, we simply return black.
  // This changes if you implement hemispherical lighting for extra credit.
  if (!bvh->intersect(r, &isect)) 
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
  logtimer.startTime(0);
  if (!isect.bsdf->is_delta()) 
    L_out += estimate_direct_lighting(r, isect);
  logtimer.recordTime(0);
  // You will implement this in part 4.
  // If the ray's depth is zero, then the path must terminate
  // and no further indirect lighting is calculated.
  logtimer.startTime(1);
  if (r.depth > 0)
    L_out += estimate_indirect_lighting(r, isect);
  logtimer.recordTime(1);

  return L_out;

}
