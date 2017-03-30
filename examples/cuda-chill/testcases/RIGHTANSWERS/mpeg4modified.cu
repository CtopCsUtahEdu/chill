// this source is derived from CHILL AST originally from file 'mpeg4.c' as parsed by frontend compiler rose

__global__ void kernel_GPU(float result[4096][4096], float prev[4096 + 16][4096 + 16], float curr[16 * 16]) {
  int t2;
  int t4;
  int t6;
  int t8;
  int t12;
  int t14;
  int t18;
  int t20;
  
    // ~cuda~ blockLoop preferredIdx: bx
    for (t2 = 0; t2 <= 127; t2 += 1) 
      // ~cuda~ preferredIdx: by
      for (t4 = 0; t4 <= 127; t4 += 1) 
        // ~cuda~ preferredIdx: iii
        for (t6 = 0; t6 <= 1; t6 += 1) 
          // ~cuda~ threadLoop preferredIdx: tx
          for (t8 = 0; t8 <= 15; t8 += 1) 
            // ~cuda~ preferredIdx: jjj
            for (t12 = 0; t12 <= 1; t12 += 1) 
              // ~cuda~ preferredIdx: ty
              for (t14 = 0; t14 <= 15; t14 += 1) 
                // ~cuda~ preferredIdx: k
                for (t18 = 0; t18 <= 15; t18 += 1) 
                  // ~cuda~ preferredIdx: l
                  for (t20 = 0; t20 <= 15; t20 += 1) 
                    result[32 * t2 + 16 * t6 + t8][16 * t12 + t14 + 32 * t4] += prev[32 * t2 + 16 * t6 + t8 + t18][16 * t12 + t14 + 32 * t4 + t20] * curr[t18 * (unsigned int)16 + t20];;
}
void mpeg4_cpu(float result[4096][4096], float prev[4096 + 16][4096 + 16], float curr[16 * 16]) {
  float * devI2Ptr;
  float * devI1Ptr;
  float * devO1Ptr;
  cudaMalloc((void **)devO1Ptr, 16777216 * sizeof(float));
  cudaMemcpy(float * devO1Ptr, float result[4096][4096], 16777216 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI1Ptr, 16908544 * sizeof(float));
  cudaMemcpy(float * devI1Ptr, float prev[4096 + 16][4096 + 16], 16908544 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI2Ptr, 256 * sizeof(float));
  cudaMemcpy(float * devI2Ptr, float curr[16 * 16], 256 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid= dim3(1, 1);
  dim3 dimBlock= dim3(1, 1);
  kernel_GPU<<<dimGrid,dimBlock>>>((float (*)[4096])float * devO1Ptr, (float (*)[4096 + 16])float * devI1Ptr, devI2Ptr);
  cudaMemcpy(float result[4096][4096], float * devO1Ptr, 16777216 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(float * devO1Ptr);
  cudaFree(float * devI1Ptr);
  cudaFree(float * devI2Ptr);
}
