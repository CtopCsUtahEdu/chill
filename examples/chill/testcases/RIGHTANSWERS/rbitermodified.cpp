// this source is derived from CHILL AST originally from file 'rbiter.cpp' as parsed by frontend compiler rose

int f(int N, int M, int *a) {
  int t4;
  int t2;
  if (N + 1 <= M) 
    for (t2 = 0; t2 <= 63; t2 += 1) 
      for (t4 = N + (-t2 - 1 - N) % 2; t4 <= M - 1; t4 += 2) 
        a[t2] = 0;
}
