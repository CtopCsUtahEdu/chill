#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void f(float **a,int n,int m)
{
  int t4;
  int i;
  int j;
  if (2 <= n) 
    for (t4 = 1; t4 <= n + m - 2; t4 += 1) 
      a[i][j] = (a[i - 1][j + 1] + 1);
}
