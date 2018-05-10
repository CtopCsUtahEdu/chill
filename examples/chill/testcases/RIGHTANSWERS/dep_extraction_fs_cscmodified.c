// this source is derived from CHILL AST originally from file 'dep_extraction_fs_csc.c' as parsed by frontend compiler rose

#define Li____(j, p) Li[1 * p + 1]
#define Li__(j, p) Li[1 * p]
#define Li___(jp, pp) Li[1 * pp + 1]
#define Li_(jp, pp) Li[1 * pp]
#define Lp____(j) Lp[1 * j]
#define Lp__(j) Lp[1 * j + 1]
#define Lp___(j) Lp[1 * j + 1]
#define Lp_(j) Lp[1 * j]
// Forward Solve CSC

void fs_csc(int n, int *Lp, int *Li, double *Lx, double *x) {
  int j;
  int p;
  for (j = 0; j < n; j++) {
    // Diagonal operation

    x[j] /= Lx[Lp[j]];
    for (p = Lp[j] + 1; p < Lp[j + 1]; p++) 
      // off-diagonal 

      x[Li[p]] -= Lx[p] * x[j];
  }
}
