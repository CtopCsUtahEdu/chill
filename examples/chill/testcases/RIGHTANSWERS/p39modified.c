// this source is derived from CHILL AST originally from file 'p39.c' as parsed by frontend compiler rose

#define N 16

void foo(float *A, float *B) {
  int t2;
  for (t2 = 1; t2 <= 15; t2 += 1) 
    A[t2 + 2] = A[t2] + B[t2];
  return;
}
int main() {
  float x[16];
  float y[16];
  foo(x, y);
  return 0;
}
