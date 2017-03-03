// this source is derived from CHILL AST originally from file 'datacopy34.c' as parsed by frontend compiler rose

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t6;
  int t4;
  int t2;
  float _P1[ambn];
  if (1 <= bm && 1 <= ambn) 
    for (t2 = 0; t2 <= an - 1; t2 += 1) 
      for (t4 = 0; t4 <= bm - 1; t4 += 1) {
        for (t6 = 0; t6 <= ambn - 1; t6 += 1) 
          _P1[t6 - 0] = A[t2][t6];
        for (t6 = 0; t6 <= ambn - 1; t6 += 1) 
          C[t2][t4] = C[t2][t4] + _P1[t6 - 0] * B[t6][t4];
      }
}
