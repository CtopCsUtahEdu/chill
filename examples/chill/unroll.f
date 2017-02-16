      SUBROUTINE foo(N,X,Y,Z,F3,F1,W)
      INTEGER N
      REAL*8  X(N),Y(N),Z(N),F3(0:N),F1(0:N),W(0:N)
      REAL*8 DT
      
      INTEGER I,J


       DO 10,I=1,14
         X(I) = 1.0D0
 10   CONTINUE
    
      DO 20,I=1,14,3
         Y(I) = 1.0D0
 20   CONTINUE

       DO 30,I=N+1,N+20,3
         Z(I) = 1.0D0
 30   CONTINUE
    
      DO 40,I=0,N
         DO 50,J=I,I+N
            F3(I) = F3(I) + F1(J) * W(J-I)
 50      CONTINUE
         F3(I) = F3(I) * DT
 40   CONTINUE

      END
