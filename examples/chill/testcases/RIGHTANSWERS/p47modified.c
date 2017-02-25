// this source is derived from CHILL AST originally from file 'p47.c' as parsed by frontend compiler rose

#define X 3

#define N 10

#define M 10

void foo() {
  int t4;
  int t2;
  float A[20][20];
  for (t2 = 1; t2 <= 10; t2 += 1) 
    for (t4 = 1; t4 <= 10; t4 += 1) 
      A[t4 + 1][t2] = A[t4][t2] + 3.0f;
  return;
}
int main() {
  foo();
  return 0;
}
