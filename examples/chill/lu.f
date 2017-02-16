      SUBROUTINE lu(n,A)
      IMPLICIT NONE

      INTEGER n
      REAL*8 A(n,n)

      INTEGER i,j,k

      DO 10 k = 1, n-1
         DO 20 i = k+1, n
            A(i, k) = A(i, k) / A(k, k)
 20      CONTINUE

         DO 30 i = k+1, n
            DO 40 j = k+1, n
               A(i, j) = A(i, j) - A(i, k)*A(k, j)
 40         CONTINUE
 30      CONTINUE
 10   CONTINUE

      END
