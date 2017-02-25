
// example from page 19 of the CHiLL manual


void f ( float **a, int n, int m ) {

   int i, j;

   for (i=1; i<n; i++) { // starts at 1 
      for (j=0; j<m; j++) { // starts at 0
         a[i][j] = a[i-1][j+1] + 1.0f;
      }
   }
}


