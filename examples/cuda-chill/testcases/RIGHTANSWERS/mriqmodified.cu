// this source is derived from CHILL AST originally from file 'mriq.c' as parsed by frontend compiler rose

__global__ void Kernel_GPU(float *x, float *y, float *z, float *Qi, float *Qr, struct kValues *kVals) {
  float phi;
  float sinArg;
  float cosArg;
  float expArg;
  float newVariable4;
  float newVariable3;
  float newVariable2;
  float newVariable1;
  float newVariable0;
  int by = blockIdx.y;
  int i;
  int ii;
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  {
    {
      newVariable0 = x[tx + 128 * bx];
    }
    {
      newVariable1 = y[128 * bx + tx];
    }
    {
      newVariable2 = z[tx + 128 * bx];
    }
    {
      newVariable3 = Qi[128 * bx + tx];
    }
    {
      newVariable4 = Qr[tx + 128 * bx];
    }
    for (ii = 0; ii <= 23; ii += 1) 
      {
        for (i = 0; i <= 127; i += 1) {
          expArg = 6.28318548f * (kVals[128 * ii + i].Kx * newVariable0 + kVals[128 * ii + i].Ky * newVariable1 + kVals[128 * ii + i].Kz * newVariable2);
          cosArg = cosf(expArg);
          sinArg = sinf(expArg);
          phi = kVals[128 * ii + i].PhiMag;
          newVariable4 += phi * cosArg;
          newVariable3 += phi * sinArg;
        }
      }
    {
      Qr[tx + 128 * bx] = newVariable4;
    }
    {
      Qi[128 * bx + tx] = newVariable3;
    }
  }
}
#include "mriq.h"

void ComputeQCPU(struct kValues kVals[3072], float x[32768], float y[32768], float z[32768], float Qr[32768], float Qi[32768]) {
  float * devRO3ptr;
  float * devRO2ptr;
  float * devRO1ptr;
  struct kValues * devRO0ptr;
  float * devRW1ptr;
  float * devRW0ptr;
  cudaMalloc((void **)&devRW0ptr, 32768 * sizeof(float));
  cudaMemcpy(devRW0ptr, Qi, 32768 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRW1ptr, 32768 * sizeof(float));
  cudaMemcpy(devRW1ptr, Qr, 32768 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO0ptr, 3072 * sizeof(struct kValues));
  cudaMemcpy(devRO0ptr, kVals, 3072 * sizeof(struct kValues), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO1ptr, 32768 * sizeof(float));
  cudaMemcpy(devRO1ptr, x, 32768 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO2ptr, 32768 * sizeof(float));
  cudaMemcpy(devRO2ptr, y, 32768 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO3ptr, 32768 * sizeof(float));
  cudaMemcpy(devRO3ptr, z, 32768 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid0 = dim3(256, 1);
  dim3 dimBlock0 = dim3(128);
  Kernel_GPU<<<dimGrid0,dimBlock0>>>(devRW0ptr, devRW1ptr, devRO0ptr, devRO1ptr, devRO2ptr, devRO3ptr);
  cudaMemcpy(Qi, devRW0ptr, 32768 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(devRW0ptr);
  cudaMemcpy(Qr, devRW1ptr, 32768 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(devRW1ptr);
  cudaFree(devRO0ptr);
  cudaFree(devRO1ptr);
  cudaFree(devRO2ptr);
  cudaFree(devRO3ptr);
}
