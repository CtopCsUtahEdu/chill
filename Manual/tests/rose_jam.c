#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void mm(float **a,float **b,int n,int m)
{
  int t4;
  int t2;
  int i;
  int j;
  if (2 <= m) 
    for (t2 = 0; t2 <= 2 * n - 1; t2 += 1) 
      for (t4 = 1; t4 <= m - 1; t4 += 1) 
        a[t2 + 1][t4 - 1] = (a[t2][t4] + b[t2][t4]);
}
