#define N 10
#define J 10
#define NELT 1000
#define M 10
#define I 10
#define L 10
#define K 10

void local_grad_3(double *U,double *ur,double *us,double *Dt,double *ut,double *D)
{

  int l,k,nelt,n,m,dummyLoop;

  for (dummyLoop=0; dummyLoop<1; dummyLoop++){
  for (nelt=0; nelt<NELT; nelt++){
   for (n=0; n<N; n++){
    for (m=0; m<J; m++){
     for (k=0; k<K; k++){
      for (l=0; l<L; l++){
       ur[nelt*N*J*K + n*J*K + m*K + k ] = ur[nelt*N*J*K + n*J*K + m*K + k ] + (D[l*K + k ] * U[nelt*N*M*L + 
n*M*L + m*L + l ]);
      }
     }
    }
   }
  }
  for (nelt=0; nelt<NELT; nelt++){
   for (n=0; n<N; n++){
    for (m=0; m<M; m++){
     for (k=0; k<K; k++){
      for (l=0; l<M; l++){
       us[nelt*N*M*K + n*M*K + m*K + k ] = us[nelt*N*M*K + n*M*K + m*K + k ] + (U[nelt*N*M*L + n*M*L + l*L + 
k ] * Dt[m*N + l ]);
      }
     }
    }
   }
  }
  for (nelt=0; nelt<NELT; nelt++){
   for (n=0; n<I; n++){
    for (m=0; m<J; m++){
     for (k=0; k<K; k++){
      for (l=0; l<N; l++){
       ut[nelt*I*J*K + n*J*K + m*K + k ] = ut[nelt*I*J*K + n*J*K + m*K + k ] + (U[nelt*N*M*L + l*M*L + m*L + 
k ] * Dt[n*N + l ]);
      }
     }
    }
   }
  }
