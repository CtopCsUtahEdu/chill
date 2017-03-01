// this source is derived from CHILL AST originally from file 'skew.c' as parsed by frontend compiler rose

// example from page 19 of the CHiLL manual

void f(float **a, int n, int m) {
  int t4;
  int t2;
  for (t2 = 1; t2 <= n - 1; t2 += 1) 
    for (t4 = t2; t4 <= t2 + m - 1; t4 += 1) 
      a[t2][-t2 + t4] = a[t2 - 1][-t2 + t4 + 1] + 1.0f;
}
