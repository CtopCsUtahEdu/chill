
#ifndef N
#define N 512
#endif

/*
<test name=jacobi define="{'N':512}">
procedure int jacobi(
    in out float[N][N] a = matrix [i,j] random(2,-2))
</test>
*/
int jacobi(float a[N][N]) {
    int t, i;
	for (t = 2; t <= 100; t++)
		for (i = 2; i <= N - 1; i++)
			a[t][i] = a[t - 1][i - 1] + a[t - 1][i] + a[t - 1][i + 1];

	return 0;
}
