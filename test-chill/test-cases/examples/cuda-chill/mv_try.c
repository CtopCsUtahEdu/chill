#define N 4096

void normalMV(int n, float c[N][N], float a[N], float b[N]) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      a[i] = a[i] + c[i][j] * b[j];
}
