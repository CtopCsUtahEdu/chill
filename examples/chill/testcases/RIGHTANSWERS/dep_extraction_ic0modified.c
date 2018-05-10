// this source is derived from CHILL AST originally from file 'dep_extraction_ic0.c' as parsed by frontend compiler rose

#define row_______(ip) row[1 * ip + 1]
#define row_____(ip) row[1 * ip]
#define row____(i) row[1 * i + 1]
#define row__(i) row[1 * i]
#define row_col__________(i, m) row[1 * col______(i, m)]
#define row_col________(i, m) row[1 * col______(i, m) + 1]
#define row_col_________(i, m) row[1 * col______(i, m) + 1]
#define row_col_______(i, m) row[1 * col______(i, m)]
#define col________(i, m) col[1 * m + 1]
#define col______(i, m) col[1 * m]
#define row___(i) row[1 * i]
#define row_(i) row[1 * i + 1]
#define col_____(i, m, k, l) col[1 * l + 1]
#define col____(i, m, k) col[1 * k + 1]
#define col__(i, m, k) col[1 * k]
#define col___(i, m, k, l) col[1 * l]
#define col_(i, m, k, l) col[1 * l + 1]
//#include <math.h>

double sqrt(double in);
//  return in/10;

//}

void ic0_csr(int n, double *val, int *row, int *col) {
  int i;
  int k;
  int l;
  int m;
  for (i = 0; i < n - 1; i++) {
    //S1

    val[row[i]] = val[row[i]] / sqrt(val[row[i]]);
    for (m = row[i] + 1; m < row[i + 1]; m++) 
      //S2

      val[m] = val[m] / val[row[i]];
    for (m = row[i] + 1; m < row[i + 1]; m++) 
      for (k = row[col[m]]; k < row[col[m] + 1]; k++) 
        for (l = m; l < row[i + 1]; l++) 
          if (col[l] == col[k]) 
            if (col[l + 1] <= col[k]) 
              //S3

              val[k] -= val[m] * val[l];
  }
}
