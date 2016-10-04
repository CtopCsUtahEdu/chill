void f(float **a, int n) {
    int i, j;
    for (i = 1; i < n; i++)
	for (j = 0; j < n-i; j++)
	    a[i][j] = a[i-1][j-1] + a[i][j-1];
}

