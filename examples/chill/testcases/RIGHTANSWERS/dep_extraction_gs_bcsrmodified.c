// this source is derived from CHILL AST originally from file 'dep_extraction_gs_bcsr.c' as parsed by frontend compiler rose

#define colidx____(ip, jp) colidx[1 * jp + 1]
#define colidx__(ip, jp) colidx[1 * jp]
#define colidx___(i, j) colidx[1 * j + 1]
#define colidx_(i, j) colidx[1 * j]
#define rowptr____(i) rowptr[1 * i]
#define rowptr__(i) rowptr[1 * i + 1]
#define rowptr___(i) rowptr[1 * i + 1]
#define rowptr_(i) rowptr[1 * i]
//#define BS 2

void gs_bcsr(double ***values, double **y, const double **b, double ***idiag, int *rowptr, int *colidx, int BS, double *sum) {
  //  double sum[2]={0};

  int n;
  int i;
  int ii;
  int j;
  int jj;
  for (i = 0; i < n; ++i) {
    for (ii = 0; ii < BS; ++ii) 
      //S0

      sum[j] = b[i][j];
    for (j = rowptr[i]; j < rowptr[i + 1]; ++j) 
      for (jj = 0; jj < BS; ++jj) 
        for (ii = 0; ii < BS; ++ii) 
          //S1

          sum[ii] -= values[j][jj][ii] * y[colidx[j]][jj];
    for (ii = 0; ii < BS; ++ii) 
      //S2

      y[i][j] = 0;
    for (jj = 0; jj < BS; ++jj) 
      for (ii = 0; ii < BS; ++ii) 
        //S3

        y[i][jj] += idiag[i][j][jj] * sum[j];
  }
}
