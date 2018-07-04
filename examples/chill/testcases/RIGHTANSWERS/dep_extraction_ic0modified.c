// this source is derived from CHILL AST originally from file 'dep_extraction_ic0.c' as parsed by frontend compiler rose

#define row_______(chill_idx1p) row[1 * chill_idx1p + 1]
#define row_____(chill_idx1p) row[1 * chill_idx1p]
#define col______(chill_idx1, chill_idx2, chill_idx3, chill_idx4) col[1 * chill_idx4]
#define col_______(chill_idx1, chill_idx2, chill_idx3) col[1 * chill_idx3 + 1]
#define col_____(chill_idx1, chill_idx2, chill_idx3) col[1 * chill_idx3]
#define col____(chill_idx1, chill_idx2, chill_idx3, chill_idx4) col[1 * chill_idx4 + 1]
#define col__(chill_idx1, chill_idx2, chill_idx3, chill_idx4) col[1 * chill_idx4]
#define row_col_____(chill_idx1, chill_idx2) row[1 * col_(chill_idx1, chill_idx2)]
#define row_col___(chill_idx1, chill_idx2) row[1 * col_(chill_idx1, chill_idx2) + 1]
#define row_col____(chill_idx1, chill_idx2) row[1 * col_(chill_idx1, chill_idx2) + 1]
#define row_col__(chill_idx1, chill_idx2) row[1 * col_(chill_idx1, chill_idx2)]
#define col___(chill_idx1, chill_idx2) col[1 * chill_idx2 + 1]
#define col_(chill_idx1, chill_idx2) col[1 * chill_idx2]
#define row____(chill_idx1) row[1 * chill_idx1]
#define row__(chill_idx1) row[1 * chill_idx1 + 1]
#define row___(chill_idx1) row[1 * chill_idx1 + 1]
#define row_(chill_idx1) row[1 * chill_idx1]
//#include <math.h>

double sqrt(double in);
//  return in/10;

//}

void ic0_csr(int n, double *val, int *row, int *col) {
  int chill_idx4;
  int chill_idx3;
  int chill_idx2;
  int chill_idx1;
  double temp;
  for (chill_idx1 = 0; chill_idx1 < n - 1; chill_idx1 += 1) {
    temp = val[row[chill_idx1]];
    //S1

    val[row[chill_idx1]] = val[row[chill_idx1]] / sqrt(temp);
    for (chill_idx2 = row[chill_idx1] + 1; chill_idx2 < row[chill_idx1 + 1]; chill_idx2 += 1) 
      //S2

      val[chill_idx2] = val[chill_idx2] / val[row[chill_idx1]];
    for (chill_idx2 = row[chill_idx1] + 1; chill_idx2 < row[chill_idx1 + 1]; chill_idx2 += 1) 
      for (chill_idx3 = row[col[chill_idx2]]; chill_idx3 < row[col[chill_idx2] + 1]; chill_idx3 += 1) 
        for (chill_idx4 = chill_idx2; chill_idx4 < row[chill_idx1 + 1]; chill_idx4 += 1) 
          if (col[chill_idx4] == col[chill_idx3]) 
            if (col[chill_idx4 + 1] <= col[chill_idx3]) 
              //S3

              val[chill_idx3] -= val[chill_idx2] * val[chill_idx4];
  }
}
