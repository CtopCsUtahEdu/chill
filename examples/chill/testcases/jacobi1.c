#define N 512

int main() {
	int i, t;

	float a[N][N];

	for (t = 2; t <= 100; t++)
		for (i = 2; i <= N - 1; i++)
			a[t][i] = a[t - 1][i - 1] + a[t - 1][i] + a[t - 1][i + 1];

	return 0;
}
