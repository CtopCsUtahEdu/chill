// this source is derived from CHILL AST originally from file 'fuse_distribute.c' as parsed by frontend compiler rose

void foo(double A[100], double B[100]) {
  int t4;
  int t2;
  for (t2 = 0; t2 <= 99; t2 += 1) {
    for (t4 = 0; t4 <= 99; t4 += 1) 
      A[t4] = 1;
    B[0] = 1;
    for (t4 = 1; t4 <= 99; t4 += 1) {
      B[t4] = 1;
      B[t4 - 1] = B[t4 - 1 + 1] * A[t4 - 1];
    }
  }
}
