#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void f(float **a,int n)
{
  int t4;
  int t2;
  int i;
  int j;
  for (t2 = 0; t2 <= n - 2; t2 += 1) 
    for (t4 = 1; t4 <= n - t2 - 1; t4 += 1) 
      a[t4][t2] = (a[t4 - 1][t2 - 1] + a[t4][t2 - 1]);
}
