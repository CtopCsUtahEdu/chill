// this source is derived from CHILL AST originally from file 'unroll2.c' as parsed by frontend compiler rose

#define N 45

void foo(float *y) {
  int t2;
  for (t2 = 1; t2 <= 31; t2 += 15) {
    y[t2] = 1.0f;
    y[t2 + 3] = 1.0f;
    y[t2 + 6] = 1.0f;
    y[t2 + 9] = 1.0f;
    y[t2 + 12] = 1.0f;
  }
  return;
}
int main() {
  float y[45];
  foo(y);
  return 0;
}
