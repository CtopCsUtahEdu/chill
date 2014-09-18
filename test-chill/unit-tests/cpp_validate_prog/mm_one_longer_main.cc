#define AN 3
#define BM 2
#define AMBN 5
//#define PRINT

#include <time.h>
#include <fstream>
#include <cstdio>

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
            C[i][j] = 0;
            for(k = 0; k < AMBN; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    float A[3][5] = {{0,1,2,3,4},{5,6,7,8,9},{10,11,12,13,14}};
    float B[5][2] = {{0,1},{2,3},{4,5},{6,7},{8,9}};
    float C[3][2] = {{0,0},{0,0},{0,0}};
    timespec start_time;
    timespec end_time;
    
    if (argc == 3) {
        std::ifstream is(argv[1], std::ifstream::in | std::ifstream::binary);
        is.read((char*)A, 15*sizeof(float));
        is.read((char*)B, 10*sizeof(float));
        is.close();
    }
    
    clock_gettime(CLOCK_REALTIME, &start_time);
    for(int i = 0; i < 10000; i++) {
        mm(A,B,C);
    }
    clock_gettime(CLOCK_REALTIME, &end_time);
    
    if (argc == 3) {
        std::ofstream os(argv[2], std::ofstream::out | std::ofstream::binary);
        os.write((char*)C, 6*sizeof(float));
        os.close();
    }
    
    #ifdef PRINT
    std::printf("A:\n");
    for(int i = 0; i < 3; i++) {
        std::printf("[");
        for(int j = 0; j < 5; j++) {
            std::printf("%f,",A[i][j]);
        }
        std::printf("]\n");
    }
    std::printf("B:\n");
    for(int i = 0; i < 5; i++) {
        std::printf("[");
        for(int j = 0; j < 2; j++) {
            std::printf("%f,",B[i][j]);
        }
        std::printf("]\n");
    }
    std::printf("C:\n");
    for(int i = 0; i < 3; i++) {
        std::printf("[");
        for(int j = 0; j < 2; j++) {
            std::printf("%f,",C[i][j]);
        }
        std::printf("]\n");
    }
    #else
    double time_diff = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0;
    std::printf("(%f,)", time_diff);
    #endif
    return 0;
}
