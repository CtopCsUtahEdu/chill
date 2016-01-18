#include <time.h>
#include <fstream>
#include <cstdio>


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

int main(int argc, char** argv) {
    float A[3][5];
    float B[5][2];
    float C[3][2];
    timespec start_time;
    timespec end_time;
    
    std::ifstream datafile_initialize(argv[1]);
    datafile_initialize.read((char*)A, 15*sizeof(float));
    datafile_initialize.read((char*)B, 10*sizeof(float));
    datafile_initialize.read((char*)C, 6*sizeof(float));
    datafile_initialize.close();
    
    clock_gettime(CLOCK_REALTIME, &start_time);
    mm(A,B,C);
    clock_gettime(CLOCK_REALTIME, &end_time);
    
    std::ofstream datafile_out(argv[2]);
    datafile_out.write((char*)C, 6*sizeof(float));
    datafile_out.close();
    
    double time_diff = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0;
    std::printf("(%f,)", time_diff);
    return 0;
}
