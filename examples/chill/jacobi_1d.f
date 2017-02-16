      SUBROUTINE jacobi(N,A,B)
      INTEGER N,T,I

      REAL*8 A(N),B(N)

      DO 10, T=1,100
         DO 20, I=2,N-1
            B(I)=0.25D0*(A(I-1)+A(I+1))+0.5D0*A(I)
 20      CONTINUE
         DO 30, I=2,N-1
            A(I) = B(I)
 30      CONTINUE
 10   CONTINUE

      END
      
