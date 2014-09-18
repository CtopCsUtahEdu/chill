#include <time.h>
#include <fstream>
#include <cstdio>

//# defines
//# test-proc

int main(int argc, char** argv) {
    //# declarations
    timespec start_time;
    timespec end_time;
    
    std::ifstream datafile_initialize(argv[1]);
    //# read-in
    //# read-out
    datafile_initialize.close();
    
    clock_gettime(CLOCK_REALTIME, &start_time);
    //# run
    clock_gettime(CLOCK_REALTIME, &end_time);
    
    std::ofstream datafile_out(argv[2]);
    //# write-out
    datafile_out.close();
    
    double time_diff = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0;
    std::printf("(%f,)", time_diff);
    return 0;
}
