// this source is derived from CHILL AST originally from file 'p40.c' as parsed by frontend compiler rose

#define N 16

#define M 20

void foo(float *A, float *B) {
  int t4;
  int t2;
  for (t2 = 1; t2 <= 15; t2 += 1) 
    for (t4 = 0; t4 <= 19; t4 += 1) 
      A[t2 + 2] = A[t2] + B[t4];
  return;
}
int main() {
  float x[16];
  float y[20];
  foo(x, y);
  return 0;
}
