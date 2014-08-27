PROGRAM matmul
INTEGER :: N, i, j, k
REAL(kind=8) :: a(10,10), b(10,10), c(10,10), ct(10,10), mysum
DO i = 1, 10, 1
DO j = 1, 10, 1
a(i,j) = i + j
b(i,j) = i - j
c(i,j) = 0.0
ct(i,j) = 0.0
END DO
b(i,i) = 1.0
END DO
DO j = 1, 10, 1
DO k = 1, 10, 1
DO i = 1, 10, 1
c(i,j) = c(i,j) + a(i,k) * b(k,j)
END DO
END DO
END DO
CALL gemm(10,a,b,ct)
mysum = 0.0
DO i = 1, 10, 1
DO j = 1, 10, 1
mysum = c(i,j) - ct(i,j)
END DO
END DO
IF (abs(mysum) >= 0.00001) THEN
WRITE (*, FMT=*) "Something wrong"
ELSE
WRITE (*, FMT=*) "Output matches"
END IF
END PROGRAM matmul

SUBROUTINE gemm(N,A,B,C)
INTEGER :: t12
INTEGER :: t10
INTEGER :: t8
INTEGER :: t6
INTEGER :: t4
INTEGER :: t2
INTEGER :: chill_t64
INTEGER :: chill_t63
INTEGER :: chill_t62
INTEGER :: chill_t61
INTEGER :: chill_t60
INTEGER :: chill_t59
INTEGER :: chill_t58
INTEGER :: chill_t57
INTEGER :: chill_t56
INTEGER :: chill_t55
INTEGER :: chill_t54
INTEGER :: chill_t53
INTEGER :: chill_t52
INTEGER :: chill_t51
INTEGER :: chill_t50
INTEGER :: chill_t49
INTEGER :: chill_t48
INTEGER :: chill_t47
INTEGER :: over2
INTEGER :: chill_t46
INTEGER :: chill_t45
INTEGER :: chill_t44
INTEGER :: chill_t43
INTEGER :: chill_t42
INTEGER :: chill_t41
INTEGER :: chill_t40
INTEGER :: chill_t39
INTEGER :: chill_t38
INTEGER :: chill_t37
INTEGER :: chill_t36
INTEGER :: chill_t35
INTEGER :: chill_t34
INTEGER :: chill_t33
INTEGER :: chill_t32
INTEGER :: chill_t31
INTEGER :: chill_t30
INTEGER :: chill_t29
INTEGER :: chill_t28
INTEGER :: chill_t27
INTEGER :: chill_t26
INTEGER :: chill_t25
INTEGER :: chill_t24
INTEGER :: chill_t23
INTEGER :: over1
INTEGER :: chill_t22
INTEGER :: chill_t21
INTEGER :: chill_t20
INTEGER :: chill_t19
INTEGER :: chill_t18
INTEGER :: chill_t17
INTEGER :: chill_t16
INTEGER :: chill_t15
REAL(kind=8), DIMENSION(8,512) :: f_P2
INTEGER :: chill_t14
INTEGER :: chill_t13
INTEGER :: chill_t12
INTEGER :: chill_t11
INTEGER :: chill_t10
INTEGER :: chill_t9
INTEGER :: chill_t8
INTEGER :: chill_t7
REAL(kind=8), DIMENSION(512,128) :: f_P1
INTEGER :: chill_t1
INTEGER :: chill_t2
INTEGER :: chill_t4
INTEGER :: chill_t6
INTEGER :: chill_t5
INTEGER :: N
REAL(kind=8) :: A(N,N), B(N,N), C(N,N)
INTEGER :: I, J, K
over1 = 0
over2 = 0
DO t2 = 1, N, 512
DO t4 = 1, N, 128
DO t6 = t2, merge(N,t2 + 511,N <= t2 + 511), 1
DO t8 = t4, merge(t4 + 127,N,t4 + 127 <= N), 1
f_P1(t8 - t4 + 1,t6 - t2 + 1) = A(t8,t6)
END DO
END DO
DO t6 = 1, N, 8
DO t8 = t6, merge(N,t6 + 7,N <= t6 + 7), 1
DO t10 = t2, merge(N,t2 + 511,N <= t2 + 511), 1
f_P2(t10 - t2 + 1,t8 - t6 + 1) = B(t10,t8)
END DO
END DO
over1 = MOD(N,2)
DO t8 = t4, merge(-over1 + N,t4 + 126,-over1 + N <= t4 + 126), 2
over2 = MOD(N,2)
DO t10 = t6, merge(t6 + 6,N - over2,t6 + 6 <= N - over2), 2
DO t12 = t2, merge(t2 + 511,N,t2 + 511 <= N), 1
C(t8,t10) = C(t8,t10) + f_P1(t8 - t4 + 1,t12 - t2 + 1) * f_P2(t12 - t2 + 1,t10 - t6 + 1)
C(t8 + 1,t10) = C(t8 + 1,t10) + f_P1(t8 + 1 - t4 + 1,t12 - t2 + 1) * f_P2(t12 - t2 + 1,t10 - t6 + 1)
C(t8,t10 + 1) = C(t8,t10 + 1) + f_P1(t8 - t4 + 1,t12 - t2 + 1) * f_P2(t12 - t2 + 1,t10 + 1 - t6 + 1)
C(t8 + 1,t10 + 1) = C(t8 + 1,t10 + 1) + f_P1(t8 + 1 - t4 + 1,t12 - t2 + 1) * f_P2(t12 - t2 + 1,t10 + 1 - t6 + 1)
END DO
END DO
IF (N - 7 <= t6 .AND. 1 <= over2) THEN
DO t12 = t2, merge(N,t2 + 511,N <= t2 + 511), 1
C(t8,N) = C(t8,N) + f_P1(t8 - t4 + 1,t12 - t2 + 1) * f_P2(t12 - t2 + 1,N - t6 + 1)
C(t8 + 1,N) = C(t8 + 1,N) + f_P1(t8 + 1 - t4 + 1,t12 - t2 + 1) * f_P2(t12 - t2 + 1,N - t6 + 1)
END DO
END IF
END DO
IF (N - 127 <= t4 .AND. 1 <= over1) THEN
DO t10 = t6, merge(t6 + 7,N,t6 + 7 <= N), 1
DO t12 = t2, merge(t2 + 511,N,t2 + 511 <= N), 1
C(N,t10) = C(N,t10) + f_P1(N - t4 + 1,t12 - t2 + 1) * f_P2(t12 - t2 + 1,t10 - t6 + 1)
END DO
END DO
END IF
END DO
END DO
END DO
END SUBROUTINE 

