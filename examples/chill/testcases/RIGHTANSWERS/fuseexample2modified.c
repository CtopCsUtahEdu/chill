// this source is derived from CHILL AST originally from file 'fuseexample.c' as parsed by frontend compiler rose

// example from Chill manual for fuse command

// as of Aug 2016, this fails

void mm(float **a, float **b, float **c, int ambn, int an, int bm) {
  int t6;
  int t4;
  int t2;
  if (1 <= bm) 
    if (1 <= ambn) 
      for (t2 = 0; t2 <= an - 1; t2 += 1) {
        for (t4 = 0; t4 <= bm - 1; t4 += 1) 
          c[t2][t4] = 0.0f;
        for (t4 = 0; t4 <= bm - 1; t4 += 1) 
          for (t6 = 0; t6 <= ambn - 1; t6 += 1) 
            c[t2][t4] += a[t2][t6] * b[t6][t4];
      }
    else 
      for (t2 = 0; t2 <= an - 1; t2 += 1) 
        for (t4 = 0; t4 <= bm - 1; t4 += 1) 
          c[t2][t4] = 0.0f;
}
