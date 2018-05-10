// this source is derived from CHILL AST originally from file 'dep_extraction_gs_csr.c' as parsed by frontend compiler rose

#define colidx____(ip, jp) colidx[1 * jp + 1]
#define colidx__(ip, jp) colidx[1 * jp]
#define colidx___(i, j) colidx[1 * j + 1]
#define colidx_(i, j) colidx[1 * j]
#define rowptr____(i) rowptr[1 * i]
#define rowptr__(i) rowptr[1 * i + 1]
#define rowptr___(i) rowptr[1 * i + 1]
#define rowptr_(i) rowptr[1 * i]
// Gauss-Seidel CSR

void gs_csr(int n, int *rowptr, int *colidx, int *idiag, double *values, double *y, const double *b) {
  int i;
  int j;
  double sum;
  for (i = 0; i < n; i++) {
    sum = b[i];
    for (j = rowptr[i]; j < rowptr[i + 1]; j++) 
      sum -= values[j] * y[colidx[j]];
    y[i] = sum * (double)idiag[i];
  }
}
