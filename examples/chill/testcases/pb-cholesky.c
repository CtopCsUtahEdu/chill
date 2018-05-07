/*
 * PolyBench/C 4.0 - cholesky
 */

#include "pb-cholesky.h"

void cholesky(int n, float A[N][N]) {
    int i, j, k;

#pragma scop
    for(i = 0; i < N; i++) {
        for(j = 0; j < i; j++) {
            for(k = 0; k < j; k++) {
                A[i][j] -= A[i][k] * A[j][k];
            }
            A[i][j] /= A[j][j];
        }
        // i == j case
        for(k = 0; k < i; k++) {
            A[i][i] -= A[i][k] * A[i][k];
        }
        A[i][i] = sqrtf(A[i][i]);
    }
#pragma endscop
}

