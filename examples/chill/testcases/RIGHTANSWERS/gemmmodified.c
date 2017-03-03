// this source is derived from CHILL AST originally from file 'gemm.c' as parsed by frontend compiler rose

void gemm(int **A, int **B, int **C, int n) {
  int t12;
  int t10;
  int t8;
  int t6;
  int t4;
  int t2;
  int over2;
  int over1;
  int _P2[8][512];
  int _P1[512][128];
  over1 = 0;
  over2 = 0;
  for (t2 = 0; t2 <= n - 1; t2 += 512) 
    for (t4 = 0; t4 <= n - 1; t4 += 128) {
      for (t6 = t2; t6 <= (n - 1 < t2 + 511 ? n - 1 : t2 + 511); t6 += 1) 
        for (t8 = t4; t8 <= (n - 1 < t4 + 127 ? n - 1 : t4 + 127); t8 += 1) 
          _P1[t6 - t2][t8 - t4] = A[t6][t8];
      for (t6 = 0; t6 <= n - 1; t6 += 8) {
        for (t8 = t6; t8 <= (t6 + 7 < n - 1 ? t6 + 7 : n - 1); t8 += 1) 
          for (t10 = t2; t10 <= (t2 + 511 < n - 1 ? t2 + 511 : n - 1); t10 += 1) 
            _P2[t8 - t6][t10 - t2] = B[t8][t10];
        over1 = n % 2;
        if (1 <= over2) 
          if (t6 + 9 <= n) 
            for (t8 = t4; t8 <= (-over1 + n - 1 < t4 + 126 ? -over1 + n - 1 : t4 + 126); t8 += 2) {
              over2 = n % 2;
              for (t10 = t6; t10 <= t6 + 6; t10 += 2) 
                for (t12 = t2; t12 <= (n - 1 < t2 + 511 ? n - 1 : t2 + 511); t12 += 1) {
                  C[t10][t8] = C[t10][t8] + _P1[t12 - t2][t8 - t4] * _P2[t10 - t6][t12 - t2];
                  C[t10][t8 + 1] = C[t10][t8 + 1] + _P1[t12 - t2][t8 + 1 - t4] * _P2[t10 - t6][t12 - t2];
                  C[t10 + 1][t8] = C[t10 + 1][t8] + _P1[t12 - t2][t8 - t4] * _P2[t10 + 1 - t6][t12 - t2];
                  C[t10 + 1][t8 + 1] = C[t10 + 1][t8 + 1] + _P1[t12 - t2][t8 + 1 - t4] * _P2[t10 + 1 - t6][t12 - t2];
                }
            }
          else 
            for (t8 = t4; t8 <= (-over1 + n - 1 < t4 + 126 ? -over1 + n - 1 : t4 + 126); t8 += 2) {
              over2 = n % 2;
              for (t10 = t6; t10 <= n - 2; t10 += 2) 
                for (t12 = t2; t12 <= (n - 1 < t2 + 511 ? n - 1 : t2 + 511); t12 += 1) {
                  C[t10][t8] = C[t10][t8] + _P1[t12 - t2][t8 - t4] * _P2[t10 - t6][t12 - t2];
                  C[t10][t8 + 1] = C[t10][t8 + 1] + _P1[t12 - t2][t8 + 1 - t4] * _P2[t10 - t6][t12 - t2];
                  C[t10 + 1][t8] = C[t10 + 1][t8] + _P1[t12 - t2][t8 - t4] * _P2[t10 + 1 - t6][t12 - t2];
                  C[t10 + 1][t8 + 1] = C[t10 + 1][t8 + 1] + _P1[t12 - t2][t8 + 1 - t4] * _P2[t10 + 1 - t6][t12 - t2];
                }
              for (t12 = t2; t12 <= (n - 1 < t2 + 511 ? n - 1 : t2 + 511); t12 += 1) {
                C[n - 1][t8] = C[n - 1][t8] + _P1[t12 - t2][t8 - t4] * _P2[n - 1 - t6][t12 - t2];
                C[n - 1][t8 + 1] = C[n - 1][t8 + 1] + _P1[t12 - t2][t8 + 1 - t4] * _P2[n - 1 - t6][t12 - t2];
              }
            }
        else 
          for (t8 = t4; t8 <= (t4 + 126 < n - over1 - 1 ? t4 + 126 : n - over1 - 1); t8 += 2) {
            over2 = n % 2;
            for (t10 = t6; t10 <= (n - 1 < t6 + 6 ? n - 1 : t6 + 6); t10 += 2) 
              for (t12 = t2; t12 <= (t2 + 511 < n - 1 ? t2 + 511 : n - 1); t12 += 1) {
                C[t10][t8] = C[t10][t8] + _P1[t12 - t2][t8 - t4] * _P2[t10 - t6][t12 - t2];
                C[t10][t8 + 1] = C[t10][t8 + 1] + _P1[t12 - t2][t8 + 1 - t4] * _P2[t10 - t6][t12 - t2];
                C[t10 + 1][t8] = C[t10 + 1][t8] + _P1[t12 - t2][t8 - t4] * _P2[t10 + 1 - t6][t12 - t2];
                C[t10 + 1][t8 + 1] = C[t10 + 1][t8 + 1] + _P1[t12 - t2][t8 + 1 - t4] * _P2[t10 + 1 - t6][t12 - t2];
              }
          }
        if (n - 128 <= t4 && 1 <= over1) 
          for (t10 = t6; t10 <= (n - 1 < t6 + 7 ? n - 1 : t6 + 7); t10 += 1) 
            for (t12 = t2; t12 <= (n - 1 < t2 + 511 ? n - 1 : t2 + 511); t12 += 1) 
              C[t10][n - 1] = C[t10][n - 1] + _P1[t12 - t2][n - 1 - t4] * _P2[t10 - t6][t12 - t2];
      }
    }
}
