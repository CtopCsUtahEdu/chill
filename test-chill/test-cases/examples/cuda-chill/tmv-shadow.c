#define N 1024

void normalMV(float c[N][N], float a[N], float b[N]) {
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      a[i] = a[i] + c[i][j] * b[j];
}
