void f(float **a, int n, int m) {
    int i, j;
    for (i = 1; i < n; i++)
	for (j = 0; j < m; j++)
	    a[i][j] = a[i-1][j+1] + 1;
}
