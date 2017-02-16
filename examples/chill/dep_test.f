      SUBROUTINE foo1(N, X1, Y1, X2, Y2, X3, Y3)
      INTEGER N
      REAL*8  X1(N), Y1(N)
      REAL*8  X2(N,N), Y2(N,N)
      REAL*8  X3(N,N,N), Y3(N,N,N)
      
      INTEGER I,J

      DO 10,I=10,N-1
         X2(2*I-2, I+3) = 1.0
         Y2(I,I) = X2(300-I, 2*I+9)+2.0
 10   CONTINUE

      DO 20,I=10,200
         DO 30, J=7,167
            X3(2*I+3, 5*J-1, J) = 1.0
            Y3(I,I,J) = X3(I-1, 2*I-6, 3*J+2)+2.0
 30      CONTINUE
 20   CONTINUE

      DO 40,I=0,100
         DO 50, J = I, I+50
            X1(2*I+3*J+12) = 1.0
            Y1(I) = X1(2*I+3*J-5)+2.0
 50      CONTINUE
 40   CONTINUE

      DO 60,I=0,35
         DO 70, J=0,35
            X2(4*I-3+2*I+2, 2*J-1) = 1.0
            Y2(I,J) = X2(4*I+9, 2*J+9)+2.0
 70      CONTINUE
 60   CONTINUE

      DO 80,I=0,N-1
         DO 90, J=0,N
            DO 100, K=0,N*2
               X3(I-4*K, 3*I-3*J, 2*I-6*J+20*K) = 1.0
               Y3(I,J,K) = X3(2*K+1, 2*J+1, 4*J-10*K+2)+2.0
 100         CONTINUE
 90      CONTINUE
 80   CONTINUE

      END
