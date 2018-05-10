// this source is derived from CHILL AST originally from file 'dep_extraction_fs_csr.c' as parsed by frontend compiler rose

#define colIdx____(ip, jp) colIdx[1 * jp + 1]
#define colIdx__(ip, jp) colIdx[1 * jp]
#define colIdx___(i, j) colIdx[1 * j + 1]
#define colIdx_(i, j) colIdx[1 * j]
#define rowPtr____(i) rowPtr[1 * i]
#define rowPtr__(i) rowPtr[1 * i + 1]
#define rowPtr___(i) rowPtr[1 * i + 1]
#define rowPtr_(i) rowPtr[1 * i]
void fs_csr(int n, int *rowPtr, int *colIdx, double *val, double *b, double *x) {
  int i;
  int j;
  int tmp;
  for (i = 0; i < n; i++) {
    tmp = (int)b[i];
    for (j = rowPtr[i]; j < rowPtr[i + 1] - 1; j++) 
      tmp -= val[j] * x[colIdx[j]];
    x[i] = (double)tmp / val[rowPtr[i + 1] - 1];
  }
}
