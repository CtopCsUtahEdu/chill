program matmul

    integer N,i,j,k
    real*8 a(10,10), b(10,10), c(10,10), ct(10,10),mysum

    do i=1,10,1
      do j=1,10,1
        a(i,j) = i+j 
        b(i,j) = i-j
        c(i,j) = 0.0
        ct(i,j) = 0.0
      end do
      b(i,i) = 1.0;
    end do


      DO j=1,10,1
         DO k=1,10,1
            DO i=1,10,1
               c(i,j) = c(i,j)+a(i,k)*b(k,j)
            end do
        end do
      end do



    call gemm(10,a,b,ct)

    mysum = 0.0
    do i=1,10,1
      do j=1,10,1
        mysum = c(i,j) - ct(i,j)
      end do
    end do

   if (abs(mysum) >= 0.00001) then
     write (*,*) "Something wrong"
   else
     write (*,*) "Output matches"
   end if
    
end program matmul

      SUBROUTINE gemm(N,A,B,C)
      INTEGER N
      REAL*8  A(N,N), B(N,N), C(N,N)

      INTEGER I,J,K

      DO J=1,N,1
         DO K=1,N,1
            DO I=1,N,1
               C(I,J) = C(I,J)+A(I,K)*B(K,J)
						end do
				end do
			end do

      END subroutine
