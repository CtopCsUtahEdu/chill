// this source is derived from CHILL AST originally from file 'dep_extraction_ic0.c' as parsed by frontend compiler rose

#define row_______(ip) row[1 * ip + 1]
#define row_____(ip) row[1 * ip]
#define col______(i, m, k, l) col[1 * l]
#define col_______(i, m, k) col[1 * k + 1]
#define col_____(i, m, k) col[1 * k]
#define col____(i, m, k, l) col[1 * l + 1]
#define col__(i, m, k, l) col[1 * l]
#define row_col_____(i, m) row[1 * col_(i, m)]
#define row_col___(i, m) row[1 * col_(i, m) + 1]
#define row_col____(i, m) row[1 * col_(i, m) + 1]
#define row_col__(i, m) row[1 * col_(i, m)]
#define col___(i, m) col[1 * m + 1]
#define col_(i, m) col[1 * m]
#define row____(i) row[1 * i]
#define row__(i) row[1 * i + 1]
#define row___(i) row[1 * i + 1]
#define row_(i) row[1 * i]
//#include <math.h>

double sqrt(double in);
//  return in/10;

//}

void ic0_csr(int n, double *val, int *row, int *col) {
  int i;
  int k;
  int l;
  int m;
  double temp;
  for (i = 0; i < n - 1; i += 1) {
    temp = val[row[i]];
    //S1

    val[row[i]] = val[row[i]] / sqrt(temp);
    for (m = row[i] + 1; m < row[i + 1]; m += 1) 
      //S2

      val[m] = val[m] / val[row[i]];
    for (m = row[i] + 1; m < row[i + 1]; m += 1) 
      for (k = row[col[m]]; k < row[col[m] + 1]; k += 1) 
        for (l = m; l < row[i + 1]; l += 1) 
          if (col[l] == col[k]) 
            if (col[l + 1] <= col[k]) 
              //S3

              val[k] -= val[m] * val[l];
  }
}
