


// this source derived from CHILL AST originally from file 'unrolltrap.c' as parsed by frontend compiler rose


void foo( int n, float *x, float *y, float *z, float *f3, float *f1, float *w )
{
  int t4;
  int t2;
  int over1;
  over1 = 0;
  if (0 <= n)
    over1 = ((1 + n) % 2);
  for (t2 = 0; t2 <= (n - over1); t2 += 2) {
    if (1 <= n)
      f3[t2] = (f3[t2] + f1[t2] * w[t2 - t2]);
    for (t4 = (t2 + 1); t4 <= (t2 + n - 1); t4 += 1) {
      f3[t2] = (f3[t2] + f1[t4] * w[t4 - t2]);
      f3[t2 + 1] = (f3[t2 + 1] + f1[t4] * w[t4 - (t2 + 1)]);
    }
    f3[t2] = (f3[t2] + f1[t2 + n] * w[t2 + n - t2]);
    f3[t2] = ((float) ((double) f3[t2]) * 3.14);
    if ((t2 + 1) <= (t2 + n))
      f3[t2 + 1] = (f3[t2 + 1] + f1[t2 + n] * w[t2 + n - (t2 + 1)]);
    f3[t2 + 1] = (f3[t2 + 1] + f1[(t2 + n) + 1] * w[(t2 + n) + 1 - (t2 + 1)]);
    f3[t2 + 1] = ((float) ((double) f3[t2 + 1]) * 3.14);
  }
  if (1 <= over1) {
    for (t4 = n; t4 <= (2 * n - 1); t4 += 1) 
      f3[n] = (f3[n] + f1[t4] * w[t4 - n]);
    if (0 <= n) {
      f3[n] = (f3[n] + f1[2 * n] * w[2 * n - n]);
      f3[n] = ((float) ((double) f3[n]) * 3.14);

    }
  }
}

int main(  )
{
  float  x[14];
  float  y[14];
  float  z[14];
  float  f3[14];
  float  f1[14];
  float  w[14];
  foo(14, x, y, z, f3, f1, w);
  return(0);

}
