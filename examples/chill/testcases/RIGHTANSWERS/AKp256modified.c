// this source is derived from CHILL AST originally from file 'AKp256.c' as parsed by frontend compiler rose

#define N 64

// SIMPLE EXAMPLE FROM Allen/Kennedy page 256    currently FAILS KNOW FAIL 

void foo(float *a, float *b, float c, float *d, float e) {
  int t2;
  for (t2 = 0; t2 <= 63; t2 += 1) {
    a[t2] = b[t2] + c;
    d[t2] = a[t2] + e;
  }
  return;
}
