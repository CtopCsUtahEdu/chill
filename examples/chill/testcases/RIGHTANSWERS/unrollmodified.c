// this source is derived from CHILL AST originally from file 'unroll.c' as parsed by frontend compiler rose

#define N 14

void foo(int n, float *x, float *y, float *z, float *f3, float *f1, float *w) {
  int t4;
  int over1;
  int t2;
  x[1] = 1.0f;
  x[2] = 1.0f;
  x[3] = 1.0f;
  x[4] = 1.0f;
  x[5] = 1.0f;
  x[6] = 1.0f;
  x[7] = 1.0f;
  x[8] = 1.0f;
  x[9] = 1.0f;
  x[10] = 1.0f;
  x[11] = 1.0f;
  x[12] = 1.0f;
  x[13] = 1.0f;
  x[14] = 1.0f;
  for (t2 = 1; t2 <= 7; t2 += 6) {
    y[t2] = 1.0f;
    y[t2 + 3] = 1.0f;
  }
  y[13] = 1.0f;
  z[n + 1] = 1.0f;
  z[n + 1 + 3] = 1.0f;
  z[n + 1 + 6] = 1.0f;
  z[n + 1 + 9] = 1.0f;
  z[n + 1 + 12] = 1.0f;
  z[n + 1 + 15] = 1.0f;
  z[n + 1 + 18] = 1.0f;
  over1 = 0;
  if (0 <= n) 
    over1 = (1 + n) % 2;
  for (t2 = 0; t2 <= n - over1; t2 += 2) {
    for (t4 = t2; t4 <= n + t2 - 1; t4 += 1) 
      f3[t2] = f3[t2] + f1[t4] * w[t4 - t2];
    f3[t2] = f3[t2] + f1[t2 + n] * w[t2 + n - t2];
    f3[t2] = (float)((double)f3[t2] * 3.1400000000000001);
    for (t4 = t2 + 1; t4 <= n + t2; t4 += 1) 
      f3[t2 + 1] = f3[t2 + 1] + f1[t4] * w[t4 - (t2 + 1)];
    f3[t2 + 1] = f3[t2 + 1] + f1[t2 + n + 1] * w[t2 + n + 1 - (t2 + 1)];
    f3[t2 + 1] = (float)((double)f3[t2 + 1] * 3.1400000000000001);
  }
  if (1 <= over1) {
    for (t4 = n; t4 <= 2 * n - 1; t4 += 1) 
      f3[n] = f3[n] + f1[t4] * w[t4 - n];
    if (0 <= n) {
      f3[n] = f3[n] + f1[2 * n] * w[2 * n - n];
      f3[n] = (float)((double)f3[n] * 3.1400000000000001);
    }
  }
  return;
}
int main() {
  float x[14];
  float y[14];
  float z[14];
  float f3[14];
  float f1[14];
  float w[14];
  foo(14, x, y, z, f3, f1, w);
  return 0;
}
