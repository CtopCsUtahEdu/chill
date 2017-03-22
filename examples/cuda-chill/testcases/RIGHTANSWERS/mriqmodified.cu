// this source is derived from CHILL AST originally from file 'mriq.c' as parsed by frontend compiler rose

float sinf(float );
float cosf(float );
void ComputeQCPU(int numK, int numX, struct kValues kVals[3072], float x[32768], float y[32768], float z[32768], float Qr[32768], float Qi[32768]) {
  float * devI4Ptr;
  float * devI3Ptr;
  float * devI2Ptr;
  struct kValues * devI1Ptr;
  float * devO2Ptr;
  float * devO1Ptr;
  cudaMalloc((void **)devO1Ptr, 32768 * sizeof(float));
  cudaMemcpy(float * devO1Ptr, float Qr[32768], 32768 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devO2Ptr, 32768 * sizeof(float));
  cudaMemcpy(float * devO2Ptr, float Qi[32768], 32768 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI1Ptr, 3072 * sizeof(struct kValues));
  cudaMemcpy(struct kValues * devI1Ptr, struct kValues kVals[3072], 3072 * sizeof(struct kValues), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI2Ptr, 32768 * sizeof(float));
  cudaMemcpy(float * devI2Ptr, float x[32768], 32768 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI3Ptr, 32768 * sizeof(float));
  cudaMemcpy(float * devI3Ptr, float y[32768], 32768 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI4Ptr, 32768 * sizeof(float));
  cudaMemcpy(float * devI4Ptr, float z[32768], 32768 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid= dim3 dim3() {
  }(1, 1);
  dim3 dimBlock= dim3 dim3() {
  }(1, 1);
  __global__ void Kernel_GPU(float Qr[32768], float Qi[32768], struct kValues kVals[3072], float x[32768], float y[32768], float z[32768]) {
    int t2;
    int t4;
    int t6;
    int t10;
    float expArg;
    float cosArg;
    float sinArg;
    float phi;
    
      // ~cuda~ preferredIdx: bx
      for (t2 = 0; t2 <= 255; t2 += 1) 
        // ~cuda~ preferredIdx: ii
        for (t4 = 0; t4 <= 23; t4 += 1) 
          // ~cuda~ preferredIdx: tx
          for (t6 = 0; t6 <= 127; t6 += 1) 
            // ~cuda~ preferredIdx: i
            for (t10 = 0; t10 <= 127; t10 += 1) {
              expArg = 6.28318548f * (kVals[128 * t4 + t10].Kx * x[128 * t2 + t6] + kVals[128 * t4 + t10].Ky * y[128 * t2 + t6] + kVals[128 * t4 + t10].Kz * z[128 * t2 + t6]);
              cosArg = cosf(expArg);
              sinArg = sinf(expArg);
              phi = kVals[128 * t4 + t10].PhiMag;
              Qr[128 * t2 + t6] += phi * cosArg;
              Qi[128 * t2 + t6] += phi * sinArg;
            };
  }<<<dimGrid,dimBlock>>>(devO1Ptr, devO2Ptr, devI1Ptr, devI2Ptr, devI3Ptr, devI4Ptr);
  cudaMemcpy(float Qr[32768], float * devO1Ptr, 32768 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(float * devO1Ptr);
  cudaMemcpy(float Qi[32768], float * devO2Ptr, 32768 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(float * devO2Ptr);
  cudaFree(struct kValues * devI1Ptr);
  cudaFree(float * devI2Ptr);
  cudaFree(float * devI3Ptr);
  cudaFree(float * devI4Ptr);
}
__global__ void Kernel_GPU(float Qr[32768], float Qi[32768], struct kValues kVals[3072], float x[32768], float y[32768], float z[32768]) {
  int t2;
  int t4;
  int t6;
  int t10;
  float expArg;
  float cosArg;
  float sinArg;
  float phi;
  
    // ~cuda~ preferredIdx: bx
    for (t2 = 0; t2 <= 255; t2 += 1) 
      // ~cuda~ preferredIdx: ii
      for (t4 = 0; t4 <= 23; t4 += 1) 
        // ~cuda~ preferredIdx: tx
        for (t6 = 0; t6 <= 127; t6 += 1) 
          // ~cuda~ preferredIdx: i
          for (t10 = 0; t10 <= 127; t10 += 1) {
            expArg = 6.28318548f * (kVals[128 * t4 + t10].Kx * x[128 * t2 + t6] + kVals[128 * t4 + t10].Ky * y[128 * t2 + t6] + kVals[128 * t4 + t10].Kz * z[128 * t2 + t6]);
            cosArg = cosf(expArg);
            sinArg = sinf(expArg);
            phi = kVals[128 * t4 + t10].PhiMag;
            Qr[128 * t2 + t6] += phi * cosArg;
            Qi[128 * t2 + t6] += phi * sinArg;
          };
}
