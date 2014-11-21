#define AN 3
#define BM 2
#define AMBN 5

/*

<test name='mm_small'>

procedure void mm(
    in  float[3][5] A = matrix([*,*],lambda i,j: random(-8,8)),
    in  float[5][2] B = matrix([*,*],lambda i,j: random(-8,8)),
    out float[3][2] C = matrix([*,*],lambda i,j: 0))

</test>

*/

void mm(float A[AN][AMBN], float B[AMBN][BM], float C[AN][BM]) {
    int i;
    int j;
    int k;
    for(i = 0; i < AN; i++) {
        for(j = 0; j < BM; j++) {
            for(k = 0; k < AMBN; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
