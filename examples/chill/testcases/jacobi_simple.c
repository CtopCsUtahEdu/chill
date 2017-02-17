
#define N       1024


void jacobi(double A[N][N]) {
    int t,i;
    for(t = 1; t <= 100; t++) {
        for(i = 1; i < N-1; i++) {
            A[i][t] = A[i-1][t-1] + A[i][t-1] + A[i+1][t-1];
        }
    }
}


