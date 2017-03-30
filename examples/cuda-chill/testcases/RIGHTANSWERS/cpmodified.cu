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
__global__ void kernel_GPU(float *energy, float atoms[16000]) {
  int t2;
  int t4;
  int t6;
  int t10;
  int t14;
  float dx;
  float dy;
  float dz;
  
    // ~cuda~ blockLoop preferredIdx: bx
    for (t2 = 0; t2 <= 15; t2 += 1) 
      // ~cuda~ preferredIdx: by
      for (t4 = 0; t4 <= 31; t4 += 1) 
        // ~cuda~ threadLoop preferredIdx: tx
        for (t6 = 0; t6 <= 31; t6 += 1) 
          // ~cuda~ preferredIdx: ty
          for (t10 = 0; t10 <= 15; t10 += 1) 
            // ~cuda~ preferredIdx: n
            for (t14 = 0; t14 <= 3996; t14 += 4) {
              dx = (float)(0.10000000000000001 * (double)(t10 + 16 * t4) - (double)atoms[t14]);
              dy = (float)(0.10000000000000001 * (double)(32 * t2 + t6) - (double)atoms[t14 + 1]);
              dz = z - atoms[t14 + 2];
              energy[(32 * t2 + t6) * 512 + (t10 + 16 * t4) + 512 * 512 * 0] += atoms[t14 + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
            };
}
void cenergy_cpu(float atoms[16000], float *energy, float z) {
  float * devI1Ptr;
  float * devO1Ptr;
  cudaMalloc((void **)devO1Ptr, 0 * sizeof(float));
  cudaMemcpy(float * devO1Ptr, float *energy, 0 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI1Ptr, 16000 * sizeof(float));
  cudaMemcpy(float * devI1Ptr, float atoms[16000], 16000 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid= dim3(1, 1);
  dim3 dimBlock= dim3(1, 1);
  kernel_GPU<<<dimGrid,dimBlock>>>(devO1Ptr, devI1Ptr);
  cudaMemcpy(float *energy, float * devO1Ptr, 0 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(float * devO1Ptr);
  cudaFree(float * devI1Ptr);
}
