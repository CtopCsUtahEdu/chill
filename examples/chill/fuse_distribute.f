      SUBROUTINE foo()
      REAL*8 A(100), B(100)

      INTEGER I, J

      DO 10, I=1,100
         DO 20, J=1,100
            A(J) = 1.0D0
 20      CONTINUE
         DO 30, J=1,100
            B(J) = 1.0D0
 30      CONTINUE
         DO 40, J=1,99
            B(J) = B(J+1)*A(J)
 40      CONTINUE
 10   CONTINUE

      END
