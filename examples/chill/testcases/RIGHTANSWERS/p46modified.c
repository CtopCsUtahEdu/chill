// this source is derived from CHILL AST originally from file 'p46.c' as parsed by frontend compiler rose

#define N 16

#define M 20

#define L 25

void foo() {
  int t6;
  int t4;
  int t2;
  float A[100][100][100];
  for (t2 = 1; t2 <= 16; t2 += 1) 
    for (t4 = 1; t4 <= 20; t4 += 1) 
      for (t6 = 1; t6 <= 25; t6 += 1) 
        A[t2 + 1][t4][t6 - 1] = A[t2][t4][t6] + 10.0f;
  return;
}
int main() {
  foo();
  return 0;
}
