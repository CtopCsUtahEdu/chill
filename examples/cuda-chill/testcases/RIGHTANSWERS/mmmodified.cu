// this source is derived from CHILL AST originally from file 'mm.c' as parsed by frontend compiler rose

__global__ void mm_GPU(float c[1024][1024], float a[1024][1024], float b[1024][1024]) {
  int t2;
  int t4;
  int t6;
  int t8;
  int t10;
  int t16;
  int t18;
  int t24;
  
    // ~cuda~ blockLoop preferredIdx: bx
    for (t2 = 0; t2 <= 7; t2 += 1) 
      // ~cuda~ preferredIdx: by
      for (t4 = 0; t4 <= 15; t4 += 1) 
        // ~cuda~ preferredIdx: kk
        for (t6 = 0; t6 <= 63; t6 += 1) 
          // ~cuda~ threadLoop preferredIdx: tx
          for (t8 = 0; t8 <= 15; t8 += 1) 
            // ~cuda~ preferredIdx: iii
            for (t10 = 0; t10 <= 7; t10 += 1) 
              // ~cuda~ preferredIdx: ty
              for (t16 = 0; t16 <= 15; t16 += 1) 
                // ~cuda~ preferredIdx: jjj
                for (t18 = 0; t18 <= 3; t18 += 1) 
                  // ~cuda~ preferredIdx: k
                  for (t24 = 16 * t6; t24 <= 16 * t6 + 15; t24 += 1) 
                    c[64 * t4 + 16 * t18 + t16][128 * t2 + t8 + 16 * t10] = c[64 * t4 + 16 * t18 + t16][128 * t2 + t8 + 16 * t10] + a[t24][128 * t2 + t8 + 16 * t10] * b[64 * t4 + 16 * t18 + t16][t24];;
}
void normalMM(float c[1024][1024], float a[1024][1024], float b[1024][1024]) {
  float * devI2Ptr;
  float * devI1Ptr;
  float * devO1Ptr;
  cudaMalloc((void **)devO1Ptr, 1048576 * sizeof(float));
  cudaMalloc((void **)devI1Ptr, 1048576 * sizeof(float));
  cudaMemcpy(float * devI1Ptr, float a[1024][1024], 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)devI2Ptr, 1048576 * sizeof(float));
  cudaMemcpy(float * devI2Ptr, float b[1024][1024], 1048576 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid= dim3(1, 1);
  dim3 dimBlock= dim3(1, 1);
  mm_GPU<<<dimGrid,dimBlock>>>((float (*)[1024])float * devO1Ptr, (float (*)[1024])float * devI1Ptr, (float (*)[1024])float * devI2Ptr);
  cudaMemcpy(float c[1024][1024], float * devO1Ptr, 1048576 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(float * devO1Ptr);
  cudaFree(float * devI1Ptr);
  cudaFree(float * devI2Ptr);
}
