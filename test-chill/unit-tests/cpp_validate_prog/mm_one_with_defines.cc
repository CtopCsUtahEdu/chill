
/*
<test name='mm_small' define="{'AN':3, 'BM':2, 'AMBN':5}">

with {evendist2:lambda i,j: random(-8,8), zero2:lambda i,j: 0}
procedure void mm(
    in  float[AN][AMBN] A = matrix([*,*],evendist2),
    in  float[AMBN][BM] B = matrix([*,*],evendist2),
    out float[AN][BM]   C = matrix([*,*],zero2))

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
