
#define N       1024


void trmm(double A[N][N], double B[N][N], double C[N][N]) {
    int i, j, k;
    for(k = 0; k < N; k++) {
        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++) {
                C[j][i] = C[j][i] + A[k][i] * B[j][k];
            }
        }
    }
}


