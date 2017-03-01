// this source is derived from CHILL AST originally from file 'jacobi_simple.c' as parsed by frontend compiler rose

#define N       1024

void jacobi(double A[1024][1024]) {
  int t6;
  int t4;
  int t2;
  for (t2 = 2; t2 <= 1090; t2 += 64) 
    for (t4 = (1 > t2 - 1022 ? 1 : t2 - 1022); t4 <= (100 < t2 + 62 ? 100 : t2 + 62); t4 += 1) 
      for (t6 = (t4 + 1 > t2 ? t4 + 1 : t2); t6 <= (t2 + 63 < t4 + 1022 ? t2 + 63 : t4 + 1022); t6 += 1) 
        A[t6 - t4][t4] = A[t6 - t4 - 1][t4 - 1] + A[t6 - t4][t4 - 1] + A[t6 - t4 + 1][t4 - 1];
}
