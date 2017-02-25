// this source is derived from CHILL AST originally from file 'permute8910.c' as parsed by frontend compiler rose

// a slightly different permute example. Only one statement inside nested loops

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t6;
  int t4;
  int t2;
  int i;
  int j;
  for (i = 0; i < an; i++) 
    for (j = 0; j < bm; j++) 
      C[i][j] = 0.0f;
  for (t2 = 0; t2 <= ambn - 1; t2 += 1) 
    for (t4 = 0; t4 <= an - 1; t4 += 1) 
      for (t6 = 0; t6 <= bm - 1; t6 += 1) 
        C[t4][t6] += A[t4][t2] * B[t2][t6];
}
