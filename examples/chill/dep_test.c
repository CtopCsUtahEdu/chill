
void foo1(double X1[N], double Y1[N], double X2[N][N], double Y2[N][N], double X3[N][N][N], double Y3[N][N][N]) {
    int i, j;
    
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
    
    for(i = 
}
