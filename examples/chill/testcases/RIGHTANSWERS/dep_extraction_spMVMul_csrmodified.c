// this source is derived from CHILL AST originally from file 'dep_extraction_spMVMul_csr.c' as parsed by frontend compiler rose

#define rowptr____(chill_idx1) rowptr[1 * chill_idx1]
#define rowptr__(chill_idx1) rowptr[1 * chill_idx1 + 1]
#define rowptr___(chill_idx1) rowptr[1 * chill_idx1 + 1]
#define rowptr_(chill_idx1) rowptr[1 * chill_idx1]
// sparse matrix vector multiply (SpMV) for CSR format

void spMVMul_csr(int n, int *rowptr, int *col, double *val, double *x, double *y) {
  int chill_idx2;
  int chill_idx1;
  int tmp;
  for (chill_idx1 = 0; chill_idx1 < n; chill_idx1 += 1) {
    tmp = (int)y[chill_idx1];
    for (chill_idx2 = rowptr[chill_idx1]; chill_idx2 < rowptr[chill_idx1 + 1]; chill_idx2 += 1) 
      y[chill_idx1] = (double)tmp + val[chill_idx2] * x[col[chill_idx2]];
  }
}
