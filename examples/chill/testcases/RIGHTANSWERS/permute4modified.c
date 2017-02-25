// this source is derived from CHILL AST originally from file 'permute123456.c' as parsed by frontend compiler rose

// code from CHiLL manual permute example

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t6;
  int t4;
  int t2;
  for (t2 = 0; t2 <= bm - 1; t2 += 1) 
    for (t4 = 0; t4 <= an - 1; t4 += 1) {
      C[t4][t2] = 0.0f;
      C[t4][t2] += A[t4][0] * B[0][t2];
      for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
        C[t4][t2] += A[t4][t6] * B[t6][t2];
    }
}
