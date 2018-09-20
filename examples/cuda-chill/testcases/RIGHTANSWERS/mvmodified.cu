// this source is derived from CHILL AST originally from file 'mv.c' as parsed by frontend compiler rose

__global__ void mv_GPU(float *a, float *c[1024], float *b) {
  float newVariable0;
  int by = blockIdx.y;
  int j;
  int k;
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  {
    {
      newVariable0 = a[32 * bx + tx];
    }
    for (k = 0; k <= 15; k += 1) 
      {
        for (j = 64 * k; j <= 64 * k + 63; j += 1) 
          newVariable0 = newVariable0 + c[j][32 * bx + tx] * b[j];
      }
    {
      a[32 * bx + tx] = newVariable0;
    }
  }
}
#define N 1024

void normalMV(float c[1024][1024], float a[1024], float b[1024]) {
  float * devRO1ptr;
  float * devRO0ptr;
  float * devRW0ptr;
  cudaMalloc((void **)&devRW0ptr, 1024 * sizeof(float));
  cudaMemcpy(devRW0ptr, a, 1024 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO0ptr, 1024 * sizeof(float));
  cudaMemcpy(devRO0ptr, b, 1024 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO1ptr, 1048576 * sizeof(float));
  cudaMemcpy(devRO1ptr, c, 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid0 = dim3(32, 1);
  dim3 dimBlock0 = dim3(32);
  mv_GPU<<<dimGrid0,dimBlock0>>>(devRW0ptr, devRO0ptr, (float (*)[1024])devRO1ptr);
  cudaMemcpy(a, devRW0ptr, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(devRW0ptr);
  cudaFree(devRO0ptr);
  cudaFree(devRO1ptr);
}
