#include <stdio.h>
#include <stdlib.h>


// this source derived from CHILL AST originally from file 'simple_mm.c' as parsed by frontend compiler rose


void normalMM( int x[1024], int a[1024][1024], int y[1024] )
{
  int * devI2Ptr;
  int * devI1Ptr;
  int * devO1Ptr;
  cudaMalloc(((void **) (&devO1Ptr)),1024 * sizeof(int));
  cudaMalloc(((void **) (&devI1Ptr)),1048576 * sizeof(int));
  cudaMemcpy(devI1Ptr,a,1048576 * sizeof(int),cudaMemcpyHostToDevice);
  cudaMalloc(((void **) (&devI2Ptr)),1024 * sizeof(int));
  cudaMemcpy(devI2Ptr,x,1024 * sizeof(int),cudaMemcpyHostToDevice);
  dim3 dimGrid = dim3(1, 1);
  dim3 dimBlock = dim3(1, 1);
  mm_GPU<<<dimGrid,dimBlock>>>(devO1Ptr, ((int (*)[1024]) devI1Ptr), devI2Ptr);
  cudaMemcpy(y,devO1Ptr,1024 * sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(devO1Ptr);
  cudaFree(devI1Ptr);
  cudaFree(devI2Ptr);

}

__global__ void mm_GPU( int *y, int a[1024][1024], int *x )
{
  int t2;
  int t4;
  int t6;
  // ~cuda~ preferredIdx: ii
  for (t2 = 0; t2 <= 960; t2 += 64) {
    // ~cuda~ preferredIdx: i
    for (t4 = 0; t4 <= 1023; t4 += 1) {
      // ~cuda~ preferredIdx: i
      for (t6 = t2; t6 <= (t2 + 63); t6 += 1) {
        y[t4] += (a[t4][t6] * x[t6]);
      }
    }
  }

}

void main()
{
	int in_x[1024];
	int in_y[1024];
	int in_a[1024][1024];


	normalMM(in_x,in_a,in_y);

}

/*
void normalMM( int x[1024], int a[1024][1024], int y[1024] )
{
  int * devI2Ptr;
  int * devI1Ptr;
  int * devO1Ptr;
  cudaMalloc(((void **) (&devO1Ptr)),1024 * sizeof(int));
  cudaMalloc(((void **) (&devI1Ptr)),1048576 * sizeof(int));
  cudaMemcpy(devI1Ptr,a,1048576 * sizeof(int),cudaMemcpyHostToDevice);
  cudaMalloc(((void **) (&devI2Ptr)),1024 * sizeof(int));
  cudaMemcpy(devI2Ptr,x,1024 * sizeof(int),cudaMemcpyHostToDevice);
  dim3 dimGrid = dim3(1, 1);
  dim3 dimBlock = dim3(1, 1);
  mm_GPU<<<dimGrid,dimBlock>>>(devO1Ptr, ((int (*)[1024]) devI1Ptr), devI2Ptr);
  cudaMemcpy(y,devO1Ptr,1024 * sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(devO1Ptr);
  cudaFree(devI1Ptr);
  cudaFree(devI2Ptr);

}

__global__ void mm_GPU( int *y, int a[1024][1024], int *x )
{
  int t2;
  int t4;
  int t6;
  // ~cuda~ preferredIdx: ii
  for (t2 = 0; t2 <= 960; t2 += 64) {
    // ~cuda~ preferredIdx: i
    for (t4 = 0; t4 <= 1023; t4 += 1) {
      // ~cuda~ preferredIdx: i
      for (t6 = t2; t6 <= (t2 + 63); t6 += 1) {
        y[t4] += (a[t4][t6] * x[t6]);
      }
    }
  }

}
*/
