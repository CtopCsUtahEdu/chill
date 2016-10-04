void mm(float **A,float **B,float **C, int ambn, int an, int bm)
{
  int n;
  int j;
  int i;
  for (i = 0; i < an; i += 1) 
    for (j = 0; j < bm; j += 1) 
      C[i][j] = 0.0f;

  for (i = 0; i < an; i += 1) 
    for (j = 0; j < bm; j += 1) 
      for (n = 0; n < ambn; n += 1) 
	  C[i][j] += (A[i][n] * B[n][j]);
}
