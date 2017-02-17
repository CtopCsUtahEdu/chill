
void foo(double A[100], double B[100]) {
    int i, j;
    for(i = 0; i < 100; i++) {
        for(j = 0; j < 100; j++) {
            A[j] = 1.0;
        }
        for(j = 0; j < 100; j++) {
            B[j] = 1.0;
        }
        for(j = 0; j < 99; j++) {
            B[j] = B[j+1]*A[j];
        }
    }
}

