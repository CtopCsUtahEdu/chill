
#define N 24
#define M 32


// Kennedy and Allen page 403
// first unroll and jam example

void figure8point1( )
{

   float A[ 48 ];
   float B[ M ];

   for (int i=0; i< 2*N; i++) {
      for (int j=0; j<M; j++) {
         A[i] = A[i] + B[j]; 
      }
   }


}
