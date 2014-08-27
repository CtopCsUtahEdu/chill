c
c These have been separated out from ccsd_t_singles_l.F and ccsd_t_doubles_l.F
c
      subroutine clean_sd_t_s1_1(h3d,h2d,h1d,p6d,p5d,p4d,
     2                     triplesx,t1sub,v2sub)
      IMPLICIT NONE
      integer h3d,h2d,h1d,p6d,p5d,p4d
      integer h3,h2,h1,p6,p5,p4
      integer N
			double precision triplesx(16,16,16,16,16,16)
      double precision t1sub(16,16)
      double precision v2sub(16,16,16,16)
      
      N = 16       

      do p4=1,10
      do p5=1,10
      do p6=1,10
      do h1=1,10
      do h2=1,10
      do h3=1,10
       triplesx(h3,h2,h1,p6,p5,p4)=triplesx(h3,h2,h1,p6,p5,p4)
     1   + t1sub(p4,h1)*v2sub(h3,h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end

