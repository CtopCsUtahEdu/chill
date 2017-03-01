// this source is derived from CHILL AST originally from file 'jacobi1.c' as parsed by frontend compiler rose

#define N 512

int main() {
  int t6;
  int t4;
  int t2;
  float a[512][512];
  for (t2 = 4; t2 <= 580; t2 += 64) 
    for (t4 = (2 > t2 - 511 ? 2 : t2 - 511); t4 <= (100 < t2 + 61 ? 100 : t2 + 61); t4 += 1) 
      for (t6 = (t4 + 2 > t2 ? t4 + 2 : t2); t6 <= (t2 + 63 < t4 + 511 ? t2 + 63 : t4 + 511); t6 += 1) 
        a[t4][t6 - t4] = a[t4 - 1][t6 - t4 - 1] + a[t4 - 1][t6 - t4] + a[t4 - 1][t6 - t4 + 1];
  return 0;
}
