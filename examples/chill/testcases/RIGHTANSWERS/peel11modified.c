// this source is derived from CHILL AST originally from file 'peel9101112.c' as parsed by frontend compiler rose

// example from the CHiLL manual page 13

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t4;
  int t6;
  int t2;
  if (1 <= bm) 
    if (1 <= ambn) 
      if (4 <= bm) 
        for (t2 = 0; t2 <= an - 1; t2 += 1) {
          C[t2][0] = 0.0f;
          C[t2][0] += A[t2][0] * B[0][0];
          for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
            C[t2][0] += A[t2][t6] * B[t6][0];
          C[t2][1] = 0.0f;
          C[t2][1] += A[t2][0] * B[0][1];
          for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
            C[t2][1] += A[t2][t6] * B[t6][1];
          C[t2][2] = 0.0f;
          C[t2][2] += A[t2][0] * B[0][2];
          for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
            C[t2][2] += A[t2][t6] * B[t6][2];
          C[t2][3] = 0.0f;
          C[t2][3] += A[t2][0] * B[0][3];
          for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
            C[t2][3] += A[t2][t6] * B[t6][3];
          for (t4 = 4; t4 <= bm - 1; t4 += 1) {
            C[t2][t4] = 0.0f;
            C[t2][t4] += A[t2][0] * B[0][t4];
            for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
              C[t2][t4] += A[t2][t6] * B[t6][t4];
          }
        }
      else 
        if (3 <= bm) 
          for (t2 = 0; t2 <= an - 1; t2 += 1) {
            C[t2][0] = 0.0f;
            C[t2][0] += A[t2][0] * B[0][0];
            for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
              C[t2][0] += A[t2][t6] * B[t6][0];
            C[t2][1] = 0.0f;
            C[t2][1] += A[t2][0] * B[0][1];
            for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
              C[t2][1] += A[t2][t6] * B[t6][1];
            C[t2][2] = 0.0f;
            C[t2][2] += A[t2][0] * B[0][2];
            for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
              C[t2][2] += A[t2][t6] * B[t6][2];
          }
        else 
          if (2 <= bm) 
            for (t2 = 0; t2 <= an - 1; t2 += 1) {
              C[t2][0] = 0.0f;
              C[t2][0] += A[t2][0] * B[0][0];
              for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
                C[t2][0] += A[t2][t6] * B[t6][0];
              C[t2][1] = 0.0f;
              C[t2][1] += A[t2][0] * B[0][1];
              for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
                C[t2][1] += A[t2][t6] * B[t6][1];
            }
          else 
            for (t2 = 0; t2 <= an - 1; t2 += 1) {
              C[t2][0] = 0.0f;
              C[t2][0] += A[t2][0] * B[0][0];
              for (t6 = 1; t6 <= ambn - 1; t6 += 1) 
                C[t2][0] += A[t2][t6] * B[t6][0];
            }
    else 
      if (4 <= bm) 
        for (t2 = 0; t2 <= an - 1; t2 += 1) {
          C[t2][0] = 0.0f;
          C[t2][1] = 0.0f;
          C[t2][2] = 0.0f;
          C[t2][3] = 0.0f;
          for (t4 = 4; t4 <= bm - 1; t4 += 1) 
            C[t2][t4] = 0.0f;
        }
      else 
        if (3 <= bm) 
          for (t2 = 0; t2 <= an - 1; t2 += 1) {
            C[t2][0] = 0.0f;
            C[t2][1] = 0.0f;
            C[t2][2] = 0.0f;
          }
        else 
          if (2 <= bm) 
            for (t2 = 0; t2 <= an - 1; t2 += 1) {
              C[t2][0] = 0.0f;
              C[t2][1] = 0.0f;
            }
          else 
            for (t2 = 0; t2 <= an - 1; t2 += 1) 
              C[t2][0] = 0.0f;
}
