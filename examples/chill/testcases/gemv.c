#define N 10

int main() {
	// int n;
	float a[N];
	float b[N];
	float c[N][N];

	int i, j;

	for (i = 1; i < N; i++)
		for (j = 1; j < N; j++)
			a[i] = a[i] + c[i][j] * b[j];

}
