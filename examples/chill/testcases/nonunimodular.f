      SUBROUTINE foo(N,C)
      INTEGER N
      REAL*8  C(N,N)
      REAL*8  X(-1:4,-1:6), B(-1:4,-1:6)
      
      INTEGER I,J,K

C     Loop #0
      DO 10,I=1,N
         DO 20,J=I,N
            C(I,J) = 1.0D0
 20      CONTINUE
 10   CONTINUE

C     Loop #1
      DO 30,I=0,4
         DO 40,J=0,6
            X(I,J) = 2 * X(I-1,J)
            B(I,J) = B(I,J-1) + X(I,J)
 40      CONTINUE
 30   CONTINUE

C     Loop #2
      DO 50,I=1,N
         DO 60,J=1,N
            DO 70,K=1,N
               C(I,J) = C(I,J)+1.0D0
 70         CONTINUE
 60      CONTINUE
 50   CONTINUE

C     Loop #3
      DO 80,I=1,3
         DO 90,J=1,3
            C(4*J-2*I+3,I+J) = 1.0D0
 90      CONTINUE
 80   CONTINUE
            
      END
