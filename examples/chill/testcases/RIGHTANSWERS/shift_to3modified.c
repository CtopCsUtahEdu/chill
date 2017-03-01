// this source is derived from CHILL AST originally from file 'shift_to.c' as parsed by frontend compiler rose

// code from CHiLL manual shift example on page 18

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t6;
  int t4;
  int t2;
  for (t2 = 0; t2 <= an - 1; t2 += 1) 
    for (t4 = 0; t4 <= bm - 1; t4 += 1) {
      C[t2][t4] = 0.0f;
      C[t2][t4] += A[t2][0] * B[0][t4];
      for (t6 = -4; t6 <= ambn - 6; t6 += 1) 
        C[t2][t4] += A[t2][t6 + 5] * B[t6 + 5][t4];
    }
}
