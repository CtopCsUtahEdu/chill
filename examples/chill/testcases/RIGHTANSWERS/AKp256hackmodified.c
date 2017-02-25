// this source is derived from CHILL AST originally from file 'AKp256hack.c' as parsed by frontend compiler rose

#define N 64

void foo(float *a, float *b, float c, float *d, float e) {
  int t4;
  for (t4 = 0; t4 <= 63; t4 += 1) {
    a[t4] = b[t4] + c;
    d[t4] = a[t4] + e;
  }
  return;
}
