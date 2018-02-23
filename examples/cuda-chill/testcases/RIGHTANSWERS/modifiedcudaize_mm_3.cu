// this source is derived from CHILL AST originally from file 'mm.c' as parsed by frontend compiler rose

__global__ void kernel_gpu(float *c[1024], float *a[1024], float *b[1024]) {
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int j;
  int i;
  for (i = 0; i <= 1023; i += 1) 
    for (j = 0; j <= 1023; j += 1) 
      {
        c[j][i] = c[j][i] + a[bx][i] * b[j][bx];
      }
}
#include "mm.h"

void normalMM(float c[1024][1024], float a[1024][1024], float b[1024][1024]) {
  float * devRO1ptr;
  float * devRO0ptr;
  float * devRW0ptr;
  cudaMalloc((void **)&devRW0ptr, 1048576 * sizeof(float));
  cudaMemcpy(devRW0ptr, c, 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO0ptr, 1048576 * sizeof(float));
  cudaMemcpy(devRO0ptr, a, 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO1ptr, 1048576 * sizeof(float));
  cudaMemcpy(devRO1ptr, b, 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid0 = dim3(1024, 1);
  dim3 dimBlock0 = dim3(1);
  kernel_gpu<<<dimGrid0,dimBlock0>>>((float (*)[1024])devRW0ptr, (float (*)[1024])devRO0ptr, (float (*)[1024])devRO1ptr);
  cudaMemcpy(c, devRW0ptr, 1048576 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(devRW0ptr);
  cudaFree(devRO0ptr);
  cudaFree(devRO1ptr);
}
