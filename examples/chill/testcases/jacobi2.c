#define N 512

int main() {
	double a[N];
	double b[N];
	int t, i;
	for (t = 1; t <= 100; t++) {
		for (i = 2; i <= N - 1; i++)
			b[i] = (double) 0.25 * (a[i - 1] + a[i + 1]) + (double) 0.5 * a[i];

		for (i = 2; i <= N - 1; i++)
			a[i] = b[i];
	}
	return 0;
}
