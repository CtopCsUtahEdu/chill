void mm(float **A, float **B, float **C, int ambn, int an, int bm) {
   int i, j, n;
   for(i = 0; i < an; i++) {
      for(j = 0; j < bm; j++) {
         for(n = 0; n < ambn; n++) {
            C[i][j] = C[i][j] + A[i][n] * B[n][j];
         }
      }
   }
}
