/*
<test name=small define="{'AN':2, 'AMBN':5, 'BM':3}">
    procedure void mm(
        in  float[AN][AMBN] A = matrix([,],lambda i,j: i*AMBN + j),
        in  float[AMBN][BM] B = matrix([,],lambda i,j: i*BM + j),
        out float[AN][BM]   C = matrix([,],lambda i,j: 0))
</test>

<test name=medium define="{'AN':20, 'AMBN':50, 'BM':30}">
    procedure void mm(
        in  float[AN][AMBN] A = matrix([,],lambda i,j: i*AMBN + j),
        in  float[AMBN][BM] B = matrix([,],lambda i,j: i*BM + j),
        out float[AN][BM]   C = matrix([,],lambda i,j: 0))
</test>

<test name=big define="{'AN':200, 'AMBN':500, 'BM':300}">
    procedure void mm(
        in  float[AN][AMBN] A = matrix([,],lambda i,j: i*AMBN + j),
        in  float[AMBN][BM] B = matrix([,],lambda i,j: i*BM + j),
        out float[AN][BM]   C = matrix([,],lambda i,j: 0))
</test>
*/

void mm(float A[AN][AMBN], float B[AMBN][BM], float C[AN][BM]) {
    for(int w = 0; w < 100; w++) {
        for(int i = 0; i < AN; i++) {
            for(int j = 0; j < BM; j++) {
                C[i][j] = 0;
                for(int k = 0; k < AMBN; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
}
