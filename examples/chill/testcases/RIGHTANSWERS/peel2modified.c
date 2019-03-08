// this source is derived from CHILL AST originally from file 'peel1234.c' as parsed by frontend compiler rose

// example from the CHiLL manual page 13

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t6;
  int t4;
  int t2;
  for (t2 = 0; t2 <= an - 1; t2 += 1) 
    for (t4 = 0; t4 <= bm - 1; t4 += 1) {
      C[t2][t4] = 0.0f;
      for (t6 = 0; t6 <= ambn - 5; t6 += 1) 
        C[t2][t4] += A[t2][t6] * B[t6][t4];
      C[t2][t4] += A[t2][ambn - 4] * B[ambn - 4][t4];
      C[t2][t4] += A[t2][ambn - 3] * B[ambn - 3][t4];
      C[t2][t4] += A[t2][ambn - 2] * B[ambn - 2][t4];
      C[t2][t4] += A[t2][ambn - 1] * B[ambn - 1][t4];
    }
}
