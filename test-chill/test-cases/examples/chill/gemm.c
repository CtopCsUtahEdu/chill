
#ifndef N
#define N 512
#endif

/*
<test name=gemm define="{'N':512}">
procedure int gemm(
    in  float[N][N] a = matrix([,], lambda i,j: random(2,-2)),
    in  float[N][N] b = matrix([,], lambda i,j: random(2,-2)),
    out float[N][N] c = matrix([,], lambda i,j: 0))
</test>
*/
int gemm(float a[N][N], float b[N][N], float c[N][N]) {
	int i, j, k;
	int n = N;
	for (j = 0; j < n; j++)
		for (k = 0; k < n; k++)
			for (i = 0; i < n; i++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}

	return 0;
}

