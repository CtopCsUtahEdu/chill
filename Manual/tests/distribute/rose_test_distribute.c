#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void f(float *a,float *b,float *c,float t,int n)
{
  int t2;
  int i;
  if (t + 1 <= a) 
    for (t2 = 0; t2 <= n - 1; t2 += 1) {
      if (a[t2] != 0) 
        if ((b[t2] / a[t2]) <= 1.0) 
          a[t2] = b[t2];
      t = ((b[t2] - a[t2]) + t);
    }
  else 
    for (t2 = 0; t2 <= n - 1; t2 += 1) {
      if (a[t2] != 0) 
        if ((b[t2] / a[t2]) <= 1.0) 
          a[t2] = b[t2];
      t = ((t + b[t2]) - a[t2]);
      b[t2] = a[t2];
    }
}
