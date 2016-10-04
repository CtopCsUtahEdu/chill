int a[10][10];
int main() {

	int temp;
	int i, j;

	for (i = 0; i < 10; i++) {
		for (j = 0; j < 10; j++) {
			a[i + 1][j - 1] = a[i][j];
		}

	}

	return 0;

}
