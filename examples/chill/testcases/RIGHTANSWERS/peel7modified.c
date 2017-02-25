// this source is derived from CHILL AST originally from file 'peel5678.c' as parsed by frontend compiler rose

// example from the CHiLL manual page 13  (ALMOST) 

void mm(float **A, float **B, float **C) {
  int t4;
  int t6;
  int t2;
  for (t2 = 0; t2 <= 7; t2 += 1) {
    C[t2][0] = 0.0f;
    C[t2][0] += A[t2][0] * B[0][0];
    for (t6 = 1; t6 <= 31; t6 += 1) 
      C[t2][0] += A[t2][t6] * B[t6][0];
    C[t2][1] = 0.0f;
    C[t2][1] += A[t2][0] * B[0][1];
    for (t6 = 1; t6 <= 31; t6 += 1) 
      C[t2][1] += A[t2][t6] * B[t6][1];
    C[t2][2] = 0.0f;
    C[t2][2] += A[t2][0] * B[0][2];
    for (t6 = 1; t6 <= 31; t6 += 1) 
      C[t2][2] += A[t2][t6] * B[t6][2];
    C[t2][3] = 0.0f;
    C[t2][3] += A[t2][0] * B[0][3];
    for (t6 = 1; t6 <= 31; t6 += 1) 
      C[t2][3] += A[t2][t6] * B[t6][3];
    for (t4 = 4; t4 <= 15; t4 += 1) {
      C[t2][t4] = 0.0f;
      C[t2][t4] += A[t2][0] * B[0][t4];
      for (t6 = 1; t6 <= 31; t6 += 1) 
        C[t2][t4] += A[t2][t6] * B[t6][t4];
    }
  }
}
