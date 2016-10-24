#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void d(float **A,float **B,int n,int m,int is,int js)
{
  int t4;
  int t2;
  int chill_t6;
  int chill_t5;
  int chill_t4;
  int chill_t3;
  int over1;
  int chill_t2;
  int chill_t1;
  int i;
  int j;
  over1 = 0;
  for (t2 = 0; t2 <= 2 * n - 1; t2 += 1) {
    over1 = m % 4;
    for (t4 = 0; t4 <= m - over1 - 1; t4 += 4) {
      A[t2 - 8][t4 + 3] = (A[t2][t4] + B[t2][t4]);
      A[t2 - 8][t4 + 1 + 3] = (A[t2][t4 + 1] + B[t2][t4 + 1]);
      A[t2 - 8][t4 + 2 + 3] = (A[t2][t4 + 2] + B[t2][t4 + 2]);
      A[t2 - 8][t4 + 3 + 3] = (A[t2][t4 + 3] + B[t2][t4 + 3]);
    }
    for (t4 = __rose_gt(m - over1,0); t4 <= m - 1; t4 += 1) 
      A[t2 - 8][t4 + 3] = (A[t2][t4] + B[t2][t4]);
  }
}
