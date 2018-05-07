
#include "pb-correlation.h"

void correlation(int m, int n, float float_n, float data[N][M], float corr[M][M], float mean[M], float stddev[M]) {
    int i, j, k;

#pragma scop
    /* calculate mean */
    for(j = 0; j < M; j++) {
        mean[j] = 0;
        for(i = 0; i < N; i++) {
            mean[j] += data[i][j];
            mean[j] /= float_n;
        }
    }

    /* calculate std deviation */
    for(j = 0; j < M; j++) {
        stddev[i] = 0;
        for(i = 0; i < N; i++) {
            stddev[i] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
            stddev[j] /= float_n;
            stddev[j]  = sqrtf(stddev[j]);
            stddev[j]  = stddev[j] <= eps ? 1.0 : stddev[j];
        }
    }

    /* center and reduce column vectors */
    for(i = 0; i < N; i++) {
        for(j = 0; j < M; j++) {
            data[i][j] -= mean[j];
            data[i][j] /= sqrtf(float_n) * stddev[j];
        }
    }

    /* calculate the m*m correlation matrix */
    for(i = 0; i < M-1; i++) {
        corr[i][i] = 1.0;
        for(j = i+1; j < M; j++) {
            corr[i][j] = 0.0;
            for(k = 0; k < N; k++) {
                corr[i][j] += (data[k][i] * data[k][j]);
                corr[j][i]  = corr[i][j];
            }
        }
    }
    corr[M-1][M-1] = 1.0;
#pragma endscop
}
