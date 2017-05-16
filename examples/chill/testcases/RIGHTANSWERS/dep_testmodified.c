// this source is derived from CHILL AST originally from file 'dep_test.c' as parsed by frontend compiler rose

#define N 2048

void foo1(double X1[2048], double Y1[2048], double X2[2048][2048], double Y2[2048][2048], double X3[2048][2048][2048], double Y3[2048][2048][2048]) {
  int t6;
  int t4;
  int t2;
  for (t2 = 9; t2 <= 2047; t2 += 1) {
    X2[t2 + 3][2 * t2 - 2] = 1;
    Y2[t2][t2] = X2[2 * t2 + 9][300 - t2] + 2;
  }
  for (t2 = 9; t2 <= 200; t2 += 1) 
    for (t4 = 6; t4 <= 176; t4 += 1) {
      X3[t4][5 * t4 - 1][2 * t2 + 3] = 1;
      Y3[t4][t2][t2] = X3[3 * t4 + 2][2 * t2 - 6][t2 - 1] + 2;
    }
  for (t2 = 0; t2 <= 100; t2 += 1) 
    for (t4 = t2; t4 <= t2 + 50; t4 += 1) {
      X1[2 * t2 + 3 * t4 + 12] = 1;
      Y1[t2] = X1[2 * t2 + 3 * t4 - 5];
    }
  for (t2 = 0; t2 <= 35; t2 += 1) 
    for (t4 = 0; t4 <= 35; t4 += 1) {
      X2[2 * t4 - 1][4 * t2 - 3 + 2 * t2 + 2] = 1;
      Y2[t4][t2] = X2[2 * t4 + 9][4 * t2 + 9] + 2;
    }
  for (t2 = 0; t2 <= 2047; t2 += 1) 
    for (t4 = 0; t4 <= 2048; t4 += 1) 
      for (t6 = 0; t6 <= 4096; t6 += 1) {
        X3[2 * t2 - 6 * t4 + 20 * t6][3 * t2 - 3 * t4][t2 - 4 * t6] = 1;
        Y3[t6][t4][t2] = X3[4 * t4 - 10 * t6 + 2][2 * t4 + 1][2 * t6 + 1] + 2;
      }
}
