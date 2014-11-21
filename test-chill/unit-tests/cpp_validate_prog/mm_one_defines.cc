
/*
<test name='mm_small' define="{'AN':3, 'BM':2, 'AMBN':5}">

procedure void mm(
    in  float[AN][AMBN] A = matrix([*,*],lambda i,j: random(-8,8)),
    in  float[AMBN][BM] B = matrix([*,*],lambda i,j: random(-8,8)),
    out float[AN][BM]   C = matrix([*,*],lambda i,j: 0))

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
