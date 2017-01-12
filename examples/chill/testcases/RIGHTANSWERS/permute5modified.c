


// this source derived from CHILL AST originally from file 'permute123456.c' as parsed by frontend compiler rose

// code from CHiLL manual permute example

void mm( float **A, float **B, float **C, int ambn, int an, int bm )
{
  int t4;
  int t6;
  int t2;
  for (t2 = 0; t2 <= (an - 1); t2 += 1) {
    for (t6 = 0; t6 <= (bm - 1); t6 += 1) {
      C[t2][t6] = 0.0f;
      C[t2][t6] += (A[t2][0] * B[0][t6]);
    }
    for (t4 = 1; t4 <= (ambn - 1); t4 += 1) {
      for (t6 = 0; t6 <= (bm - 1); t6 += 1) {
        C[t2][t6] += (A[t2][t4] * B[t4][t6]);
      }
    }
  }

}
