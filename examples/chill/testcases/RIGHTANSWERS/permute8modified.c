// this source is derived from CHILL AST originally from file 'permute8910.c' as parsed by frontend compiler rose

// a slightly different permute example. Only one statement inside nested loops

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t4;
  int t2;
  int i;
  int j;
  int n;
  for (t2 = 0; t2 <= bm - 1; t2 += 1) 
    for (t4 = 0; t4 <= an - 1; t4 += 1) 
      C[t4][t2] = 0.0f;
  for (i = 0; i < an; i++) 
    for (j = 0; j < bm; j++) 
      for (n = 0; n < ambn; n++) 
        C[i][j] += A[i][n] * B[n][j];
}
