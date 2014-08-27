int main() {

	float a[512][512], b[512][512], c[512][512];

	int i, j, k;
	int n;
	for (j = 0; j < n; j++)
		for (k = 0; k < n; k++)
			for (i = 0; i < n; i++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}

	return 0;
}

