// this source is derived from CHILL AST originally from file 'dep_extraction_gs_bcsr.c' as parsed by frontend compiler rose

#define colidx____(chill_idx1p, chill_idx2p) colidx[1 * chill_idx2p + 1]
#define colidx__(chill_idx1p, chill_idx2p) colidx[1 * chill_idx2p]
#define colidx___(chill_idx1, chill_idx2) colidx[1 * chill_idx2 + 1]
#define colidx_(chill_idx1, chill_idx2) colidx[1 * chill_idx2]
#define rowptr____(chill_idx1) rowptr[1 * chill_idx1]
#define rowptr__(chill_idx1) rowptr[1 * chill_idx1 + 1]
#define rowptr___(chill_idx1) rowptr[1 * chill_idx1 + 1]
#define rowptr_(chill_idx1) rowptr[1 * chill_idx1]
//#define BS 2

void gs_bcsr(double ***values, double **y, const double **b, double ***idiag, int *rowptr, int *colidx, int BS, double *sum) {
  int chill_idx4;
  int chill_idx3;
  int chill_idx2;
  int chill_idx1;
  //  double sum[2]={0};

  int n;
  for (chill_idx1 = 0; chill_idx1 < n; chill_idx1 += 1) {
    for (chill_idx2 = 0; chill_idx2 < BS; chill_idx2 += 1) 
      //S0

      sum[chill_idx2] = b[chill_idx1][chill_idx2];
    for (chill_idx2 = rowptr[chill_idx1]; chill_idx2 < rowptr[chill_idx1 + 1]; chill_idx2 += 1) 
      for (chill_idx3 = 0; chill_idx3 < BS; chill_idx3 += 1) 
        for (chill_idx4 = 0; chill_idx4 < BS; chill_idx4 += 1) 
          //S1

          sum[chill_idx2] -= values[chill_idx2][chill_idx3][chill_idx2] * y[colidx[chill_idx2]][chill_idx3];
    for (chill_idx2 = 0; chill_idx2 < BS; chill_idx2 += 1) 
      //S2

      y[chill_idx1][chill_idx2] = 0;
    for (chill_idx2 = 0; chill_idx2 < BS; chill_idx2 += 1) 
      for (chill_idx3 = 0; chill_idx3 < BS; chill_idx3 += 1) 
        //S3

        y[chill_idx1][chill_idx2] += idiag[chill_idx1][chill_idx3][chill_idx2] * sum[chill_idx3];
  }
}
