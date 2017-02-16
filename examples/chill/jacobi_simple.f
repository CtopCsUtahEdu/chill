      SUBROUTINE jacobi(N,A)
      INTEGER N,T,I

      REAL*8 A(N,N)

      DO 10, T=2,100
         DO 20, I=2,N-1
            A(T,I)=A(T-1,I-1)+A(T-1,I)+A(T-1,I+1)
 20      CONTINUE
 10   CONTINUE

      END
      
