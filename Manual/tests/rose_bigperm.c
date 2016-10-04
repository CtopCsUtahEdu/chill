#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

void f(float *a1,float **a2,float ***a3,float ****a4,float *****a5,int n1,int n2,int n3,int n4,int n5)
{
  int t10;
  int t8;
  int t6;
  int t4;
  int t2;
  int i1;
  int i2;
  int i3;
  int i4;
  int i5;
  for (t2 = 0; t2 <= n1 - 1; t2 += 1) 
    for (t4 = 0; t4 <= n2 - 1; t4 += 1) 
      for (t6 = 2 * t2 + 4 * t4; t6 <= 2 * t2 + 4 * t4 + 6 * n3 - 6; t6 += 6) 
        for (t8 = 0; t8 <= n4 - 1; t8 += 1) 
          for (t10 = 0; t10 <= n5 - 1; t10 += 1) 
            a5[t2][t4][(-(2 * t2) - 4 * t4 + t6) / 6][t8][t10] = 0.0f;
}
