void d(float **A, float **B, int n, int m, int is, int js) {
  int i, j;

  for(i = 0; i < n * 2; i++) 
    for(j = 0; j < m; j++)
	A[i -8 ][j +3] = A[i][j] + B[i][j];
}
