      SUBROUTINE trmm(N,A,B,C)
      INTEGER N
      REAL*8  A(N,N), B(N,N), C(N,N)
      
      INTEGER I,J,K

      DO 10,K=1,N
         DO 20,I=K,N
            DO 30, J=K,N
               C(I,J) = C(I,J)+A(I,K)*B(K,J)
 30         CONTINUE
 20      CONTINUE
 10   CONTINUE

      END
