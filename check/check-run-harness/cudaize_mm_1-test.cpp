
#include <cstdlib>
#include <iostream>

#include "check-run-harness.hpp"

//>procedure_name:          normalMM
//>procedure_compiler:      nvcc
//>procedure_linker:        nvcc
//>original_header:         examples/cuda-chill/testcases/mm.h
//>original_source:         examples/cuda-chill/testcases/mm.c
//>generated_source:        examples/cuda-chill/testcases/modifiedcudaize_mm_1.cu

#include "../../examples/cuda-chill/testcases/mm.h"

extern void normalMM_original(float  c[N][N], float a[N][N], float b[N][N]);
extern void normalMM_generated(float c[N][N], float a[N][N], float b[N][N]);


int main(int argc, const char** argv) {

    float a[N][N];
    float b[N][N];
    float c_original[N][N];
    float c_generated[N][N];

    fill_random(a);
    fill_random(b);

    fill_zero(c_original);
    fill_zero(c_generated);

    normalMM_original(c_original, a, b);
    normalMM_generated(c_generated, a, b);

    float error = compute_error(c_original, c_generated);
    if(error <= min_error<float>()) {
        exit(PASS);
    }
    else {
        std::cout << "failed: error=" << error << "\n";
        exit(FAIL);
    }
}
