// this source is derived from CHILL AST originally from file 'bug1.c' as parsed by frontend compiler rose

// example from the CHiLL manual page 13

void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
  int t4;
  int t2;
  int wat;
  int i;
  for (t2 = 0; t2 <= an - 1; t2 += 1) 
    wat = t2;
  for (t2 = 0; t2 <= bm - 1; t2 += 1) {
    C[i][t2] = 0.0f;
    wat = t2;
    C[i][t2] += A[i][0] * B[0][t2];
    for (t4 = 1; t4 <= ambn - 1; t4 += 1) 
      C[i][t2] += A[i][t4] * B[t4][t2];
  }
}
