#define N 14
void foo(int n, float* x, float* y, float* z, float* f3, float* f1, float* w) {
	int dt;

	int i, j;

	for (i = 1; i <= 14; i++)
		x[i] = 1.0;

	for (i = 1; i <= 14; i += 3)
		y[i] = 1.0;

	for (i = N + 1; i <= N + 20; i += 3)
		z[i] = 1.0;

	for (i = 0; i <= N; i++) {
		for (j = i; j <= i + N; j++)
			f3[i] = f3[i] + f1[j] * w[j - i];
		f3[i] = f3[i] * dt;
	}

	return 0;
}

int main() {
	float x[N], y[N], z[N], f3[N], f1[N], w[N];

	foo(N, x, y, z, f3, f1, w);
	return 0;
}

