#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void mm(float **A,float **B,float **C,int ambn,int an,int bm)
{
  int t6;
  int t4;
  int t2;
  int _t12;
  int _t11;
  int _t10;
  int _t9;
  int _t8;
  int _t7;
  int _t6;
  int _t5;
  int _t4;
  int _t3;
  int _t2;
  int _t1;
  int i;
  int j;
  int n;
  for (t2 = 0; t2 <= an - 1; t2 += 1) 
    for (t4 = 0; t4 <= bm - 1; t4 += 1) {
      C[t2][t4] = 0.0f;
      C[t2][t4] += (A[t2][0] * B[0][t4]);
      C[t2][t4] += (A[t2][1] * B[1][t4]);
      C[t2][t4] += (A[t2][2] * B[2][t4]);
      C[t2][t4] += (A[t2][3] * B[3][t4]);
      for (t6 = 4; t6 <= ambn - 1; t6 += 1) 
        C[t2][t4] += (A[t2][t6] * B[t6][t4]);
    }
}
