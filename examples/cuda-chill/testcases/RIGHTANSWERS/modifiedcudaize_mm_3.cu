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
#define N 1024

void normalMM(float c[1024][1024], float a[1024][1024], float b[1024][1024]) {
  float * devI2Ptr;
  float * devI1Ptr;
  float * devO1Ptr;
  cudaMalloc((void **)devO1Ptr, 1048576 * sizeof(float));
  cudaMalloc((void **)devI1Ptr, 1048576 * sizeof(float));
  cudaMemcpy(devI1Ptr, a, 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI2Ptr, 1048576 * sizeof(float));
  cudaMemcpy(devI2Ptr, b, 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid0 = dim3(1024, 1);
  dim3 dimBlock0 = dim3(1);
  kernel_gpu<<<dimGrid0,dimBlock0>>>((float (*)[1024])float * devO1Ptr, (float (*)[1024])float * devI1Ptr, (float (*)[1024])float * devI2Ptr);
  cudaMemcpy(c, devO1Ptr, 1048576 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(devO1Ptr);
  cudaFree(devI1Ptr);
  cudaFree(devI2Ptr);
}
