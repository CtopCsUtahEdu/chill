// this source is derived from CHILL AST originally from file 'permute123456.c' as parsed by frontend compiler rose

// code from CHiLL manual permute example

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t2;
  int t6;
  int t4;
  for (t4 = 0; t4 <= bm - 1; t4 += 1) 
    for (t6 = 0; t6 <= an - 1; t6 += 1) {
      C[t6][t4] = 0.0f;
      C[t6][t4] += A[t6][0] * B[0][t4];
    }
  for (t2 = 1; t2 <= ambn - 1; t2 += 1) 
    for (t4 = 0; t4 <= bm - 1; t4 += 1) 
      for (t6 = 0; t6 <= an - 1; t6 += 1) 
        C[t6][t4] += A[t6][t2] * B[t2][t4];
}
