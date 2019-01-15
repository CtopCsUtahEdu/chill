// this source is derived from CHILL AST originally from file 'dep_extraction_fs_csc.c' as parsed by frontend compiler rose

#define Li____(chill_idx1, chill_idx2) Li[1 * chill_idx2 + 1]
#define Li__(chill_idx1, chill_idx2) Li[1 * chill_idx2]
#define Li___(chill_idx1p, chill_idx2p) Li[1 * chill_idx2p + 1]
#define Li_(chill_idx1p, chill_idx2p) Li[1 * chill_idx2p]
#define Lp____(chill_idx1) Lp[1 * chill_idx1]
#define Lp__(chill_idx1) Lp[1 * chill_idx1 + 1]
#define Lp___(chill_idx1) Lp[1 * chill_idx1 + 1]
#define Lp_(chill_idx1) Lp[1 * chill_idx1]
// Forward Solve CSC

void fs_csc(int n, int *Lp, int *Li, double *Lx, double *x) {
  int chill_idx2;
  int chill_idx1;
  for (chill_idx1 = 0; chill_idx1 < n; chill_idx1 += 1) {
    // Diagonal operation

    x[chill_idx1] /= Lx[Lp[chill_idx1]];
    for (chill_idx2 = Lp[chill_idx1] + 1; chill_idx2 < Lp[chill_idx1 + 1]; chill_idx2 += 1) 
      // off-diagonal 

      x[Li[chill_idx2]] -= Lx[chill_idx2] * x[chill_idx1];
  }
}
