// this source is derived from CHILL AST originally from file 'dep_extraction_spMVMul_csr.c' as parsed by frontend compiler rose

#define rowptr____(i) rowptr[1 * i]
#define rowptr__(i) rowptr[1 * i + 1]
#define rowptr___(i) rowptr[1 * i + 1]
#define rowptr_(i) rowptr[1 * i]
// sparse matrix vector multiply (SpMV) for CSR format

void spMVMul_csr(int n, int *rowptr, int *col, double *val, double *x, double *y) {
  int i;
  int k;
  int tmp;
  for (i = 0; i < n; i++) {
    tmp = (int)y[i];
    for (k = rowptr[i]; k < rowptr[i + 1]; k++) 
      y[i] = (double)tmp + val[k] * x[col[k]];
  }
}
