#define N                   64
#define XIDX(i)             i + 1
#define BIDX(i)             i + 1

void foo(double C[N][N]) {
    double X[7][5];
    double B[7][5];
    
    int i, j, k;
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            C[j][i] = 1.0;
        }
    }
    
    for(i = 0; i < 5; i++) {
        for(j = 0; j < 7; j++) {
            X[XIDX(j)][XIDX(i)] = 2*X[XIDX(j)][XIDX(i-1)];
            B[BIDX(j)][BIDX(i)] = B[BIDX(j-1)][BIDX(i)] + X[XIDX(j)][XIDX(i)];
        }
    }
    
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            for(k = 0; k < N; k++) {
                C[j][i] = C[j][i] + 1.0;
            }
        }
    }
    
    for(i = 0; i < 3; i++) {
        for(j = 0; j < 3; j++) {
            C[i+j][4*j-2*i + 3] = 1.0;
        }
    }
}
