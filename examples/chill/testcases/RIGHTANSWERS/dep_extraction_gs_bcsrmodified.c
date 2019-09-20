// this source is derived from CHILL AST originally from file 'dep_extraction_gs_bcsr.c' as parsed by frontend compiler rose

#define colidx____(ip) colidx[1 * iip + 1]
#define colidx__(ip) colidx[1 * iip]
#define colidx___(i) colidx[1 * ii + 1]
#define colidx_(i) colidx[1 * ii]
#define rowptr____(i) rowptr[1 * i]
#define rowptr__(i) rowptr[1 * i + 1]
#define rowptr___(i) rowptr[1 * i + 1]
#define rowptr_(i) rowptr[1 * i]
//#define BS 2

void gs_bcsr(double ***values, double **y, const double **b, double ***idiag, int *rowptr, int *colidx, int BS, double *sum) {
  int chill_idx4;
  //  double sum[2]={0};

  int n;
  int i;
  int ii;
  int jj;
  for (i = 0; i < n; i += 1) {
    for (ii = 0; ii < BS; ii += 1) 
      //S0

      sum[ii] = b[i][ii];
    for (ii = rowptr[i]; ii < rowptr[i + 1]; ii += 1) 
      for (jj = 0; jj < BS; jj += 1) 
        for (chill_idx4 = 0; chill_idx4 < BS; chill_idx4 += 1) 
          //S1

          sum[chill_idx4] -= values[ii][jj][chill_idx4] * y[colidx[ii]][jj];
    for (ii = 0; ii < BS; ii += 1) 
      //S2

      y[i][ii] = 0;
    for (ii = 0; ii < BS; ii += 1) 
      for (jj = 0; jj < BS; jj += 1) 
        //S3

        y[i][jj] += idiag[i][ii][jj] * sum[ii];
  }
}
