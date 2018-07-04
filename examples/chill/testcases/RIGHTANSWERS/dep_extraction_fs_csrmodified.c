// this source is derived from CHILL AST originally from file 'dep_extraction_fs_csr.c' as parsed by frontend compiler rose

#define colIdx____(chill_idx1p, chill_idx2p) colIdx[1 * chill_idx2p + 1]
#define colIdx__(chill_idx1p, chill_idx2p) colIdx[1 * chill_idx2p]
#define colIdx___(chill_idx1, chill_idx2) colIdx[1 * chill_idx2 + 1]
#define colIdx_(chill_idx1, chill_idx2) colIdx[1 * chill_idx2]
#define rowPtr____(chill_idx1) rowPtr[1 * chill_idx1]
#define rowPtr__(chill_idx1) rowPtr[1 * chill_idx1 + 1]
#define rowPtr___(chill_idx1) rowPtr[1 * chill_idx1 + 1]
#define rowPtr_(chill_idx1) rowPtr[1 * chill_idx1]
void fs_csr(int n, int *rowPtr, int *colIdx, double *val, double *b, double *x) {
  int chill_idx2;
  int chill_idx1;
  int tmp;
  for (chill_idx1 = 0; chill_idx1 < n; chill_idx1 += 1) {
    tmp = (int)b[chill_idx1];
    for (chill_idx2 = rowPtr[chill_idx1]; chill_idx2 < rowPtr[chill_idx1 + 1] - 1; chill_idx2 += 1) 
      tmp -= val[chill_idx2] * x[colIdx[chill_idx2]];
    x[chill_idx1] = (double)tmp / val[rowPtr[chill_idx1 + 1] - 1];
  }
}
