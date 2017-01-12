


// this source derived from CHILL AST originally from file 'peel34.c' as parsed by frontend compiler rose

// example from the CHiLL manual page 13

void mm( float **A, float **B, float **C )
{
  int t6;
  int t4;
  int t2;
  for (t2 = 0; t2 <= 15; t2 += 1) {
     
    for (t4 = 0; t4 <= 27; t4 += 1) {
      C[t2][t4] = 0.0f;
      C[t2][t4] += (A[t2][0] * B[0][t4]);
      for (t6 = 1; t6 <= 31; t6 += 1) {
        C[t2][t4] += (A[t2][t6] * B[t6][t4]);
      }
    }



    
    // 1
    C[t2][28] = 0.0f;
    C[t2][28] += (A[t2][0] * B[0][28]);
    for (t6 = 1; t6 <= 31; t6 += 1) {
      C[t2][28] += (A[t2][t6] * B[t6][28]);
    }

    // 2
    C[t2][29] = 0.0f;
    C[t2][29] += (A[t2][0] * B[0][29]);
    for (t6 = 1; t6 <= 31; t6 += 1) {
      C[t2][29] += (A[t2][t6] * B[t6][29]);
    }

    // 3
    C[t2][30] = 0.0f;
    C[t2][30] += (A[t2][0] * B[0][30]);
    for (t6 = 1; t6 <= 31; t6 += 1) {
      C[t2][30] += (A[t2][t6] * B[t6][30]);
    }

    // 4
    C[t2][31] = 0.0f;
    C[t2][31] += (A[t2][0] * B[0][31]);
    for (t6 = 1; t6 <= 31; t6 += 1) {
      C[t2][31] += (A[t2][t6] * B[t6][31]);
    }
  }

}
