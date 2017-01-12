

#define N 64

// SIMPLE EXAMPLE FROM Allen/Kennedy page 257    SHOULD FAIL
// because fusing will change the meaning of the code
void foo(float* a, float *b, float c, float *d, float e) {
   int i,j ;

   // 2 loops with identical iterations 
   for (i = 0; i < N; i++) 
   {
      a[i] = b[i] + c;
   }
      
   for (i = 0; i < N; i++) 
   {
      d[i] = a[i+1] + e;
   }

  return;
}

