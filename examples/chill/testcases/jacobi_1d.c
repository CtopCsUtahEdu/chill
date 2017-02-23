
#define N       1024


void jacobi(double A[N], double B[N]) {
    int t,i;
    for(t = 1; t < N; t++) {
        for(i = 1; i < N-1; i++) {
            B[i] = 0.25*(A[i-1] + A[i+1]) + 0.5*A[i];
        }
        
        for(i = 1; i < N-1; i++) {
            A[i] = B[i];
        }
    }
}


