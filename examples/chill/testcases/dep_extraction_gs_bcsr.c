//#define BS 2

void gs_bcsr(double ***values, double **y, const double **b, double ***idiag, int *rowptr, int *colidx, int BS, double *sum)
{
//  double sum[2]={0};
  int n, i, ii, j, jj;

  for (i = 0; i < n; ++i) {
    for (ii = 0; ii < BS; ++ii) {
      sum[ii] = b[i][ii];//S0
    }
    for (j = rowptr[i]; j < rowptr[i+1]; ++j) {
      for (jj = 0; jj < BS; ++jj) {
        for (ii = 0; ii < BS; ++ii) {
          sum[ii] -= values[j][jj][ii] * y[colidx[j]][jj];//S1
        }
      }
    }

    for (ii = 0; ii < BS; ++ii) {
      y[i][ii] = 0;  //S2
    }
    for (jj = 0; jj < BS; ++jj) {
      for ( ii = 0; ii < BS; ++ii) {
        y[i][ii] += idiag[i][jj][ii] * sum[jj];//S3
      }
    }
  } // for each row
}

