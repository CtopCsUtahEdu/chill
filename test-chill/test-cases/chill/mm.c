

void mm(float A[256][256], float B[256][256], float C[256][256], int ambn, int an, int bm) {
  int i, j, n;

  for(i = 0; i < an; i++) {
    for(j = 0; j < bm; j++) {
      C[i][j] = 0.0f;
      for(n = 0; n < ambn; n++) {
        C[i][j] = C[i][j] + A[i][n] * B[n][j];
      }
    }
  }
}

