// this source is derived from CHILL AST originally from file 'peel5678.c' as parsed by frontend compiler rose

// example from the CHiLL manual page 13  (ALMOST) 

void mm(float **A, float **B, float **C) {
  int t6;
  int t4;
  int t2;
  for (t2 = 0; t2 <= 7; t2 += 1) {
    for (t4 = 0; t4 <= 11; t4 += 1) {
      C[t2][t4] = 0.0f;
      C[t2][t4] += A[t2][0] * B[0][t4];
      for (t6 = 1; t6 <= 31; t6 += 1) 
        C[t2][t4] += A[t2][t6] * B[t6][t4];
    }
    C[t2][12] = 0.0f;
    C[t2][12] += A[t2][0] * B[0][12];
    for (t6 = 1; t6 <= 31; t6 += 1) 
      C[t2][12] += A[t2][t6] * B[t6][12];
    C[t2][13] = 0.0f;
    C[t2][13] += A[t2][0] * B[0][13];
    for (t6 = 1; t6 <= 31; t6 += 1) 
      C[t2][13] += A[t2][t6] * B[t6][13];
    C[t2][14] = 0.0f;
    C[t2][14] += A[t2][0] * B[0][14];
    for (t6 = 1; t6 <= 31; t6 += 1) 
      C[t2][14] += A[t2][t6] * B[t6][14];
    C[t2][15] = 0.0f;
    C[t2][15] += A[t2][0] * B[0][15];
    for (t6 = 1; t6 <= 31; t6 += 1) 
      C[t2][15] += A[t2][t6] * B[t6][15];
  }
}
