

// example from Chill manual for fuse command
// as of Aug 2016, this fails

void mm( float **a, float **b, float **c, int ambn, int an, int bm) {

  int i,j,n;

  for (i=0; i<an; i += 1) {
    for (j=0; j<bm; j++) {
      c[i][j] = 0.0f;
    }
  }

  for (i=0; i<an; i += 1) {
    for (j=0; j<bm; j++) {
      for (n=0; n<ambn; n++) {
        
        c[i][j] += a[i][n] * b[n][j];
      }
    }
  }

}
