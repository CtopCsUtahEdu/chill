// this source is derived from CHILL AST originally from file 'dep_extraction_gs_csr.c' as parsed by frontend compiler rose

#define colidx____(chill_idx1p, chill_idx2p) colidx[1 * chill_idx2p + 1]
#define colidx__(chill_idx1p, chill_idx2p) colidx[1 * chill_idx2p]
#define colidx___(chill_idx1, chill_idx2) colidx[1 * chill_idx2 + 1]
#define colidx_(chill_idx1, chill_idx2) colidx[1 * chill_idx2]
#define rowptr____(chill_idx1) rowptr[1 * chill_idx1]
#define rowptr__(chill_idx1) rowptr[1 * chill_idx1 + 1]
#define rowptr___(chill_idx1) rowptr[1 * chill_idx1 + 1]
#define rowptr_(chill_idx1) rowptr[1 * chill_idx1]
// Gauss-Seidel CSR

void gs_csr(int n, int *rowptr, int *colidx, int *idiag, double *values, double *y, const double *b) {
  int chill_idx2;
  int chill_idx1;
  double sum;
  for (chill_idx1 = 0; chill_idx1 < n; chill_idx1 += 1) {
    sum = b[chill_idx1];
    for (chill_idx2 = rowptr[chill_idx1]; chill_idx2 < rowptr[chill_idx1 + 1]; chill_idx2 += 1) 
      sum -= values[chill_idx2] * y[colidx[chill_idx2]];
    y[chill_idx1] = sum * (double)idiag[chill_idx1];
  }
}
