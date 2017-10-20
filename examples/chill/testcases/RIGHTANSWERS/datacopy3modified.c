// this source is derived from CHILL AST originally from file 'datacopy34.c' as parsed by frontend compiler rose

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t8;
  int t6;
  int t4;
  int t2;
  float newVariable0;
  if (1 <= an && 1 <= bm) 
    for (t2 = 0; t2 <= ambn - 1; t2 += 16) 
      for (t4 = 0; t4 <= an - 1; t4 += 1) 
        for (t6 = 0; t6 <= bm - 1; t6 += 1) {
          newVariable0 = C[t4][t6];
          for (t8 = t2; t8 <= (t2 + 15 < ambn - 1 ? t2 + 15 : ambn - 1); t8 += 1) 
            newVariable0 = newVariable0 + A[t4][t8] * B[t8][t6];
          C[t4][t6] = newVariable0;
        }
}
