
// sparse matrix vector multiply (SpMV) for CSR format

void spMVMul_csr(int n, int* rowptr, int* col, double* val, double *x, double *y)
{
  int i,k, tmp;

  for (i=0; i<n; i++) {
    tmp = y[i];
    for(k=rowptr[i]; k<rowptr[i+1]; k++){
      y[i] = tmp + val[k]*x[col[k]];
    }
  }

}

