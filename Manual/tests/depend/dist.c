void mm(float **A,float **B,float **C, int ambn, int an, int bm)
{
  int n;
  int j;
  int i;
  //  int x;
  //  for (x = 0; x < 1; x++) {
      for (i = 0; i <= an - 1; i += 1) 
	  for (j = 0; j <= bm - 1; j += 1) 
	      C[i][j] = 0.0f;
      for (i = 0; i <= an - 1; i += 1) 
	  for (j = 0; j <= bm - 1; j += 1) 
	      for (n = 0; n <= ambn - 1; n += 1) 
		  C[i][j] += (A[i][n] * B[n][j]);
      //  }
}
