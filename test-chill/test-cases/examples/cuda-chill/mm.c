#define N 1024

void normalMM(float c[N][N], float a[N][N], float b[N][N]) {
  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++)
        c[j][i] = c[j][i] + a[k][i] * b[j][k];
}
