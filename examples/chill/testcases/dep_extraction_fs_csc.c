// Forward Solve CSC


void fs_csc(int n, int* Lp, int* Li, double* Lx, double *x)
{
  int j, p;    
  for (j = 0 ; j < n ; j++)
  {
    x[j] /= Lx[Lp[j]] ; // Diagonal operation
    for (p = Lp[j]+1 ; p < Lp[j+1] ; p++)
    {
      x[Li[p]] -= Lx[p] * x[j] ; // off-diagonal 
    }
  }
}

