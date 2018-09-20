// this source is derived from CHILL AST originally from file 'mm.c' as parsed by frontend compiler rose

__global__ void mm_GPU(float *c[1024], float *a[1024], float *b[1024]) {
  __shared__ float _P2[16][65];
  __shared__ float _P1[128][17];
  float _P3[4][8];
  int k;
  int kk;
  int jjj;
  int iii;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;
  {
    {
      {
        {
          for (iii = 64 * by + ty; iii <= 64 * by + ty + 48; iii += 16) 
            for (jjj = 128 * bx + tx; jjj <= 128 * bx + tx + 112; jjj += 16) 
              _P3[(iii - (64 * by + ty)) / 16][(jjj - (128 * bx + tx)) / 16] = c[iii][jjj];
        }
      }
      for (kk = 0; kk <= 63; kk += 1) {
        {
          {
            _P1[128 * bx + tx - 128 * bx][16 * kk + ty - 16 * kk] = a[16 * kk + ty][128 * bx + tx];
          }
        }
        __syncthreads();
        {
          {
            _P2[16 * kk + tx - 16 * kk][64 * by + ty - 64 * by] = b[64 * by + ty][16 * kk + tx];
          }
        }
        __syncthreads();
        for (k = 0; k <= 15; k += 1) {
          {
            {
              for (iii = 0; iii <= 7; iii += 1) 
                for (jjj = 0; jjj <= 3; jjj += 1) 
                  _P3[(64 * by + 16 * jjj + ty - (64 * by + ty)) / 16][(128 * bx + tx + 16 * iii - (128 * bx + tx)) / 16] = _P3[(64 * by + 16 * jjj + ty - (64 * by + ty)) / 16][(128 * bx + tx + 16 * iii - (128 * bx + tx)) / 16] + _P1[128 * bx + tx + 16 * iii - 128 * bx][16 * kk + k - 16 * kk] * _P2[16 * kk + k - 16 * kk][64 * by + 16 * jjj + ty - 64 * by];
            }
          }
          __syncthreads();
        }
        __syncthreads();
      }
      {
        {
          for (iii = 64 * by + ty; iii <= 64 * by + ty + 48; iii += 16) 
            for (jjj = 128 * bx + tx; jjj <= 128 * bx + tx + 112; jjj += 16) 
              c[iii][jjj] = _P3[(iii - (64 * by + ty)) / 16][(jjj - (128 * bx + tx)) / 16];
        }
      }
    }
  }
}
#include "mm.h"

void normalMM(float c[1024][1024], float a[1024][1024], float b[1024][1024]) {
  float * devRO1ptr;
  float * devRO0ptr;
  float * devRW3ptr;
  cudaMalloc((void **)&devRW3ptr, 1048576 * sizeof(float));
  cudaMemcpy(devRW3ptr, c, 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO0ptr, 1048576 * sizeof(float));
  cudaMemcpy(devRO0ptr, a, 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO1ptr, 1048576 * sizeof(float));
  cudaMemcpy(devRO1ptr, b, 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid0 = dim3(8, 16);
  dim3 dimBlock0 = dim3(16, 16);
  mm_GPU<<<dimGrid0,dimBlock0>>>((float (*)[1024])devRW3ptr, (float (*)[1024])devRO0ptr, (float (*)[1024])devRO1ptr);
  cudaMemcpy(c, devRW3ptr, 1048576 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(devRW3ptr);
  cudaFree(devRO0ptr);
  cudaFree(devRO1ptr);
}
