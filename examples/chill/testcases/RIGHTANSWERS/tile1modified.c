// this source is derived from CHILL AST originally from file 'tile.c' as parsed by frontend compiler rose

// code from CHiLL manual tile example on page 21

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t8;
  int t6;
  int t4;
  int t2;
  for (t2 = 0; t2 <= bm - 1; t2 += 4) 
    for (t4 = 0; t4 <= an - 1; t4 += 1) 
      for (t6 = t2; t6 <= (t2 + 3 < bm - 1 ? t2 + 3 : bm - 1); t6 += 1) {
        C[t4][t6] = 0.0f;
        C[t4][t6] += A[t4][0] * B[0][t6];
        for (t8 = 1; t8 <= ambn - 1; t8 += 1) 
          C[t4][t6] += A[t4][t8] * B[t8][t6];
      }
}
