// this source is derived from CHILL AST originally from file 'cp.c' as parsed by frontend compiler rose

#define N 1

#define VOLSIZEY 512

#define VOLSIZEX 512

#define VOLSIZEZ 1

#define ATOMCOUNT 4000

#define ATOMCOUNTTIMES4 16000

#define GRIDSPACING 0.1

#define zDim 0

float sqrtf(float );
__global__ void kernel_GPU(float *atoms, float *energy) {
  float z;
  float dz;
  float dy;
  float dx;
  float newVariable0;
  __shared__ float _P1[4001];
  int ty = threadIdx.y;
  int n;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;
  {
    {
      {
        _P1[tx - 0] = atoms[tx];
      }
      __syncthreads();
      {
        for (n = 0; n <= 15; n += 1) 
          newVariable0 = energy[n + 16384 * bx + 512 * tx + 16 * by];
      }
      __syncthreads();
      {
        {
          for (n = 0; n <= 3996; n += 4) {
            dx = (float)(0.10000000000000001 * (double)(16 * by + ty) - (double)_P1[n - 0]);
            dy = (float)(0.10000000000000001 * (double)(32 * bx + tx) - (double)_P1[n + 1 - 0]);
            dz = z - _P1[n + 2 - 0];
            newVariable0 += _P1[n + 3 - 0] / sqrtf(dx * dx + dy * dy + dz * dz);
          }
        }
      }
      __syncthreads();
      {
        for (n = 0; n <= 15; n += 1) 
          energy[n + 16384 * bx + 512 * tx + 16 * by] = newVariable0;
      }
      __syncthreads();
    }
  }
}
void cenergy_cpu(float atoms[16000], float *energy, float z) {
  float * devRO0ptr;
  float * devRW1ptr;
  cudaMalloc((void **)&devRW1ptr, 262144 * sizeof(float));
  cudaMemcpy(devRW1ptr, energy, 262144 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&devRO0ptr, 4 * sizeof(float));
  cudaMemcpy(devRO0ptr, atoms, 4 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid0 = dim3(16, 32);
  dim3 dimBlock0 = dim3(32, 16);
  kernel_GPU<<<dimGrid0,dimBlock0>>>(devRW1ptr, devRO0ptr);
  cudaMemcpy(energy, devRW1ptr, 262144 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(devRW1ptr);
  cudaFree(devRO0ptr);
}
