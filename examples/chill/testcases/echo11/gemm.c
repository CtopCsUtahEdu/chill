
#define N 512 

int main() {

	float a[N][N], b[N][N], c[N][N];

	int i, j, k;

	for (j = 0; j < N; j++)
		for (k = 0; k < N; k++)
			for (i = 0; i < N; i++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}

	return 0;
}

