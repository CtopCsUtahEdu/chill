#define __rose_lt(x,y) ((x)<(y)?(x):(y))
#define __rose_gt(x,y) ((x)>(y)?(x):(y))

int main()
{
  int t8;
  int t6;
  int t4;
  int t2;
  int _t4;
  int _t3;
  int _t2;
  int _t1;
  int a[10UL][10UL][10UL][10UL];
  int i;
  int j;
  int k;
  int l;
  for (t2 = 0; t2 <= 9; t2 += 1) 
    for (t4 = 0; t4 <= 3; t4 += 1) 
      for (t6 = t4 + 6; t6 <= 9; t6 += 1) 
        for (t8 = 0; t8 <= -t4 + t6 - 6; t8 += 1) 
          a[t2][t4][t6 + 1][t8] = a[t2][t4][t6][t8];
  for (t2 = 0; t2 <= 9; t2 += 1) 
    for (t4 = 0; t4 <= 9; t4 += 1) 
      for (t6 = 0; t6 <= 9; t6 += 1) 
        for (t8 = __rose_gt(-t4 + t6 - 5,0); t8 <= 9; t8 += 1) 
          a[t2][t4][t6 + 1][t8] = a[t2][t4][t6][t8];
//    a[i+1][j-1] = a[i][j];
  return 0;
}
