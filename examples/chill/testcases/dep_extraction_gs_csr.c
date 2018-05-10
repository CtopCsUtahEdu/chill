// Gauss-Seidel CSR

void gs_csr(int n, int *rowptr, int *colidx,int *idiag, double *values, double *y, const double *b)
{
  int i,j;
  double sum; 

  for (i = 0; i < n; i++) {
    sum = b[i];
    for (j = rowptr[i]; j < rowptr[i + 1]; j++) {
      sum -= values[j]*y[colidx[j]];
    }
    y[i] = sum*idiag[i];
  } // for each row
}
