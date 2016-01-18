int main() {

	int i, j, k;
	int a[10][10][10];

	for (i = 0; i < 10; i++)
		for (j = 0; j < 10; j++)
			for (k = 0; k < 10; k++)
				a[i][j + 1][k - 1] = a[i][j][k];

	return 0;
}
