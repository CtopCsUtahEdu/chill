// this source is derived from CHILL AST originally from file 'p49.c' as parsed by frontend compiler rose

#define X 3

#define Y 7

#define N 10

#define M 10

void foo() {
  int t2;
  float A[100];
  float F[100];
  for (t2 = 1; t2 <= 10; t2 += 1) {
    A[t2 + 1] = F[t2];
    F[t2 + 1] = A[t2];
  }
  return;
}
int main() {
  foo();
  return 0;
}
