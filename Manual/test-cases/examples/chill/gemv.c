#ifndef N
#define N 512
#endif

/*
<test name=gemv define="{'N':512}">
procedure int gemv(
    out float[N]    a = matrix([],  lambda i:   random(2,-2)),
    in  float[N]    b = matrix([],  lambda i:   random(2,-2)),
    in  float[N][N] c = matrix([,], lambda i,j: random(2,-2)))
</test>
*/
int gemv(float a[N], float b[N], float c[N][N]) {
    int i, j;

    for (i = 1; i < N; i++)
        for (j = 1; j < N; j++)
            a[i] = a[i] + c[i][j] * b[j];

    return 0;
}
