#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void vm(float **a,float *b,float *s,int n)
{
  int t4;
  int t2;
  int i;
  int j;
  for (t2 = 1; t2 <= n - 2; t2 += 1) 
    if (t2 <= 1) {
      s[2] = 0.0f;
      b[2] = (b[2] - s[2]);
      for (t4 = 3; t4 <= n - 1; t4 += 1) {
        s[t4] = 0.0f;
        s[t4] = (s[t4] + (b[t2] * a[t2][t4]));
      }
    }
    else {
      b[t2 + 1] = (b[t2 + 1] - s[t2 + 1]);
      for (t4 = t2 + 2; t4 <= n - 1; t4 += 1) 
        s[t4] = (s[t4] + (b[t2] * a[t2][t4]));
    }
}
