#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void mm(float **A,float **B,float **C,int ambn,int an,int bm)
{
  int t2;
  int t6;
  int t4;
  int i;
  int j;
  int n;
  for (t2 = 0; t2 <= ambn - 1; t2 += 1) 
    for (t4 = 0; t4 <= an - 1; t4 += 1) 
      if (t2 <= 0) 
        for (t6 = 0; t6 <= bm - 1; t6 += 1) {
          C[t4][t6] = 0.0f;
          C[t4][t6] += (A[t4][t2] * B[t2][t6]);
        }
      else 
        for (t6 = 0; t6 <= bm - 1; t6 += 1) 
          C[t4][t6] += (A[t4][t2] * B[t2][t6]);
}
