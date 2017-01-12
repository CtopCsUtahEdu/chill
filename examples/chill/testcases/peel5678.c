

// example from the CHiLL manual page 13  (ALMOST) 

void mm(float **A, float **B, float **C) {
   
   int i, j, n;
   for(i = 0; i < 8; i++)  // loop 1
      for(j = 0; j < 16; j++) { // loop 2  with KNOWN end condition
         C[i][j] = 0.0f;
         for(n = 0; n < 32; n++) // loop 3  with KNOWN end condition 
            C[i][j] += A[i][n] * B[n][j];
      }
}
