// this source is derived from CHILL AST originally from file 'peel5678.c' as parsed by frontend compiler rose

// example from the CHiLL manual page 13  (ALMOST) 

void mm(float **A, float **B, float **C) {
  int t6;
  int t4;
  int t2;
  for (t2 = 0; t2 <= 7; t2 += 1) 
    for (t4 = 0; t4 <= 15; t4 += 1) {
      C[t2][t4] = 0.0f;
      C[t2][t4] += A[t2][0] * B[0][t4];
      C[t2][t4] += A[t2][1] * B[1][t4];
      C[t2][t4] += A[t2][2] * B[2][t4];
      C[t2][t4] += A[t2][3] * B[3][t4];
      for (t6 = 4; t6 <= 31; t6 += 1) 
        C[t2][t4] += A[t2][t6] * B[t6][t4];
    }
}
