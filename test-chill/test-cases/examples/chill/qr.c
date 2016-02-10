#include <math.h>

int main() {

	int M, N;
	float** A;
	float *s;
	float *Rdiag;
	float *nrm;
	int i, j, k;
        float t;
	for (k = 0; k < N; k++) {
		nrm[k] = 0;

		for (i = k; i < M; i++)
			nrm[k] = sqrt(nrm[k] * nrm[k] + A[i][k] * A[i][k]);
                //t = A[k][k];

		//if (t < 0)
		//	nrm[k] = -nrm[k];
		for (i = k; i < M; i++)
			A[i][k] = A[i][k] / nrm[k];

		A[k][k] = A[k][k] + 1;

		for (j = k + 1; j < N; j++) {
			s[j] = 0; //S6

			for (i = k; i < M; i++)
				s[j] = s[j] + A[i][k] * A[i][j]; //S7

			s[j] = -s[j] / A[k][k]; //S8

			for (i = k; i < M; i++)
				A[i][j] = A[i][j] + s[j] * A[i][k]; //S9

		}

		Rdiag[k] = -nrm[k];

	}

	return 0;
}
