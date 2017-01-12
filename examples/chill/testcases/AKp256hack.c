

#define N 64

void foo(float* a, float *b, float c, float *d, float e) {
   int i,j ;

   for (j=1; j<2; j++) {  // CHILL needs this fake loop
      
      // 2 loops with identical iterations 
      for (i = 0; i < N; i++) 
      {
         a[i] = b[i] + c;
      }
      
      for (i = 0; i < N; i++) 
      {
         d[i] = a[i] + e;
      }
   }

   

  return;
}

