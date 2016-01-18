int main() {

	int a[10][10];
	int i, j;
	for (i = 0; i < 10; i++) {
		for (j = 0; j < 10; j++)
			a[i][j] = a[i][j] + 5;
		for (j = 0; j < 10; j++)
			a[i][j + 1] = a[i][j + 1] + 5;

	}

}
