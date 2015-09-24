#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void mm(float **A,float **B,float **C,int ambn,int an,int bm)
{
  int t6;
  int t4;
  int t2;
  int i;
  int j;
  int n;
  for (t2 = 0; t2 <= 4 * an - 4; t2 += 1) 
    for (t4 = 0; t4 <= bm - 1; t4 += 1) {
      if (t2 + 1 <= an) {
        C[t2][t4] = 0.0f;
        if (t2 % 4 == 0) 
          C[t2 / 4][t4] += (A[t2 / 4][0] * B[0][t4]);
      }
      else if (t2 % 4 == 0) 
        C[t2 / 4][t4] += (A[t2 / 4][0] * B[0][t4]);
      if (t2 % 4 == 0) 
        for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
          C[t2 / 4][t4] += (A[t2 / 4][t6] * B[t6][t4]);
    }
}
