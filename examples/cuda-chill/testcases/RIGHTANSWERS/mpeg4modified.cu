// this source is derived from CHILL AST originally from file 'mpeg4.c' as parsed by frontend compiler rose

__global__ void kernel_GPU(float *prev[4096 + 16], float *result[4096], float *curr) {
  float _P2[2];
  __shared__ float _P1[47][48];
  int jjj;
  int l;
  int k;
  int iii;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;
  {
    {
      {
        {
          _P1[32 * by + tx - 32 * by][32 * bx + ty - 32 * bx] = prev[32 * bx + ty][32 * by + tx];
        }
      }
      for (iii = 0; iii <= 1; iii += 1) 
        {
          for (k = 0; k <= 15; k += 1) 
            for (l = 32 * by + k; l <= 32 * by + k + 16; l += 16) 
              _P2[(l - (32 * by + k)) / 16] = result[32 * bx + 16 * iii + tx][l];
          for (jjj = 0; jjj <= 1; jjj += 1) 
            {
              for (k = 0; k <= 15; k += 1) 
                for (l = 0; l <= 15; l += 1) 
                  _P2[(32 * by + 16 * jjj + ty - (32 * by + ty)) / 16] += _P1[32 * by + 16 * jjj + ty + l - 32 * by][32 * bx + 16 * iii + tx + k - 32 * bx] * curr[k * (unsigned int)16 + l];
            }
          for (k = 0; k <= 15; k += 1) 
            for (l = 32 * by + k; l <= 32 * by + k + 16; l += 16) 
              result[32 * bx + 16 * iii + tx][l] = _P2[(l - (32 * by + k)) / 16];
        }
      __syncthreads();
    }
  }
}
#define N1 4096

#define N2 4096

#define WINDOW_SIZE 16

void mpeg4_cpu(float result[4096][4096], float prev[4096 + 16][4096 + 16], float curr[16 * 16]) {
  float * devI3Ptr;
  float * devI2Ptr;
  float * devI1Ptr;
  cudaMalloc((void **)devI1Ptr, 16908544 * sizeof(float));
  cudaMemcpy(devI1Ptr, prev, 16908544 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI2Ptr, 16777216 * sizeof(float));
  cudaMemcpy(devI2Ptr, result, 16777216 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI3Ptr, 256 * sizeof(float));
  cudaMemcpy(devI3Ptr, curr, 256 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid0 = dim3(128, 128);
  dim3 dimBlock0 = dim3(16, 16);
  kernel_GPU<<<dimGrid0,dimBlock0>>>((float (*)[4112])float * devI1Ptr, (float (*)[4096])float * devI2Ptr, devI3Ptr);
  cudaFree(devI1Ptr);
  cudaFree(devI2Ptr);
  cudaFree(devI3Ptr);
}
