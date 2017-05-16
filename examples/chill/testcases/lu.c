
#define N       1024


void lu(double A[N][N]) {
    int i, j, k;
    for(k = 0; k < N-1; k++) {
        for(i = k+1; i < N; i++) {
            A[k][i] = A[k][i] / A[k][k];
        }
        
        for(i = k+1; i < N; i++) {
            for(j = k+1; k < N; j++) {
                A[j][i] = A[j][i] - A[k][i]*A[j][k];
            }
        }
    }
}


