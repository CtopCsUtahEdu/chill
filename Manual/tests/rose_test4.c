#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void mm(float *a,float *b,float *c,int n)
{
  int t2;
  int i;
  for (t2 = 0; t2 <= n - 1; t2 += 1) {
    a[t2 + 1] = b[t2];
    b[t2 + 1] = a[t2];
    c[t2] = (a[t2] + b[t2]);
  }
}
