#include "cudapathtracer.h"

namespace CGL {

/*   
*   
*
*/
__global__ void raytrace_cuda_pixel_helper(size_t* x, size_t* y, Spectrum* sp){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    sp[i].r = 0.30;
    sp[i].g = 0.40;
    sp[i].b = 0.050;
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


void raytrace_cuda_tile(int tile_x, int tile_y,
                                int tile_w, int tile_h, HDRImageBuffer *sampleBuffer,
                                size_t imageTileSize, vector<int> *tile_samples,
                                ImageBuffer *frameBuffer) {

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

    host_x = (size_t *)malloc(sizeof(size_t) * tile_length_x);
    host_y = (size_t *)malloc(sizeof(size_t) * tile_length_y);

    for (size_t y = 0; y < tile_length_y; y++) {
        host_y[y] = tile_start_y + y;
    }
    for (size_t x = 0; x < tile_length_x; x++) {
        host_x[x] = tile_start_x + x;
    }

    //cudamalloc x, y, spectrum
    cudaMalloc((void **) &dev_x, sizeof(size_t) * tile_length_x);
    cudaMalloc((void **) &dev_y, sizeof(size_t) * tile_length_y);
    cudaMalloc((void **) &dev_sp, sizeof(Spectrum) * tile_length_x * tile_length_y);

    //cudaMemCpy
    cudaMemcpy(dev_x, &host_x, sizeof(size_t) * tile_length_x, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, &host_y, sizeof(size_t) * tile_length_y, cudaMemcpyHostToDevice);
    
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


    (*tile_samples)[tile_idx_x + tile_idx_y * num_tiles_w] += 1;
    sampleBuffer->toColor(*frameBuffer, tile_start_x, tile_start_y, tile_end_x, tile_end_y);
    }


} //namespace CGL