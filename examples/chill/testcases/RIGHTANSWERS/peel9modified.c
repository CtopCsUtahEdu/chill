// this source is derived from CHILL AST originally from file 'peel9101112.c' as parsed by frontend compiler rose

// example from the CHiLL manual page 13

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t6;
  int t4;
  int t2;
  if (1 <= bm) 
    if (4 <= ambn) 
      for (t2 = 0; t2 <= an - 1; t2 += 1) 
        for (t4 = 0; t4 <= bm - 1; t4 += 1) {
          C[t2][t4] = 0.0f;
          C[t2][t4] += A[t2][0] * B[0][t4];
          C[t2][t4] += A[t2][1] * B[1][t4];
          C[t2][t4] += A[t2][2] * B[2][t4];
          C[t2][t4] += A[t2][3] * B[3][t4];
          for (t6 = 4; t6 <= ambn - 1; t6 += 1) 
            C[t2][t4] += A[t2][t6] * B[t6][t4];
        }
    else 
      if (3 <= ambn) 
        for (t2 = 0; t2 <= an - 1; t2 += 1) 
          for (t4 = 0; t4 <= bm - 1; t4 += 1) {
            C[t2][t4] = 0.0f;
            C[t2][t4] += A[t2][0] * B[0][t4];
            C[t2][t4] += A[t2][1] * B[1][t4];
            C[t2][t4] += A[t2][2] * B[2][t4];
          }
      else 
        if (2 <= ambn) 
          for (t2 = 0; t2 <= an - 1; t2 += 1) 
            for (t4 = 0; t4 <= bm - 1; t4 += 1) {
              C[t2][t4] = 0.0f;
              C[t2][t4] += A[t2][0] * B[0][t4];
              C[t2][t4] += A[t2][1] * B[1][t4];
            }
        else 
          if (1 <= ambn) 
            for (t2 = 0; t2 <= an - 1; t2 += 1) 
              for (t4 = 0; t4 <= bm - 1; t4 += 1) {
                C[t2][t4] = 0.0f;
                C[t2][t4] += A[t2][0] * B[0][t4];
              }
          else 
            for (t2 = 0; t2 <= an - 1; t2 += 1) 
              for (t4 = 0; t4 <= bm - 1; t4 += 1) 
                C[t2][t4] = 0.0f;
}
