#define N 2048

void foo1(double X1[N], double Y1[N], double X2[N][N], double Y2[N][N], double X3[N][N][N], double Y3[N][N][N]) {
    int i, j, k;
    
    for(i = 9; i < N; i++) {
        X2[i+3][2*i - 2] = 1.0;
        Y2[i][i]         = X2[2*i+9][300-i] + 2.0;
    }
    
    for(i = 9; i <= 200; i++) {
        for(j = 6; j <= 176; j++) {
            X3[j][5*j-1][2*i+3] = 1.0;
            Y3[j][i][i]         = X3[3*j+2][2*i-6][i-1] + 2.0;
        }
    }
    
    for(i = 0; i <= 100; i++) {
        for(j = i; j <= i + 50; j++) {
            X1[2*i+3*j+12] = 1.0;
            Y1[i]          = X1[2*i+3*j-5];
        }
    }
    
    for(i = 0; i <= 35; i++) {
        for(j = 0; j <= 35; j++) {
            X2[2*j-1][4*i-3+2*i+2] = 1.0;
            Y2[j][i]               = X2[2*j+9][4*i+9] + 2.0;
        }
    }
    
    for(i = 0; i <= N-1; i++) {
        for(j = 0; j <= N; j++) {
            for(k = 0; k <= 2*N; k++) {
                X3[2*i-6*j+20*k][3*i-3*j][i-4*k] = 1.0;
                Y3[k][j][i]                      = X3[4*j-10*k+2][2*j+1][2*k+1] + 2.0;
            }
        }
    }
}
