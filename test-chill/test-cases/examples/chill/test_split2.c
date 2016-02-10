int main() {

	int a[10][10][10][10];
	int i, j, k, l;

	for (i = 0; i < 10; i++)
		for (j = 0; j < 10; j++)
			for (k = 0; k < 10; k++)
				for (l = 0; l < 10; l++)
					a[i][j][k + 1][l - 1] = a[i][j][k][l];
	//    a[i+1][j-1] = a[i][j];

	return 0;
}
