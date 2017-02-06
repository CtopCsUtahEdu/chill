void gemm(int **A, int **B, int **C, int n) {
    int i,j,k;
    for (j = 0; j<n; ++j)
        for (k = 0; k<n;++k)
            for (i = 0; i<n; ++i)
                C[j][i] = C[j][i] + A[k][i]*B[j][k];
}
