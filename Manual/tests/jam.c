void mm(float **a, float **b, int n, int m) {
    int i, j;

  for(i = 0; i < 2 * n; i++)
    for (j = 1; j < m; j++)
      a[i+1][j-1] = a[i][j] + b[i][j];
}
