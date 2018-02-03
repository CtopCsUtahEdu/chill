/*
 * mriq-test.cpp
 *
 *  Created on: Jan 19, 2018
 *      Author: derick
 */

#include <cstdlib>
#include <iostream>

#include "check-run-harness.hpp"

//>procedure_compiler:      nvcc
//>procedure_linker:        nvcc
//>script:                  examples/cuda-chill/testcases/mriq.py
//>original_header:         examples/cuda-chill/testcases/mriq.h

struct kValues {
    float Kx;
    float Ky;
    float Kz;
    float PhiMag;
};

#define M 32768
#define N 3072

extern void ComputeQCPU_original(struct kValues kVals[M], float x[N], float y[N], float z[N],float Qr[N], float Qi[N]);
extern void ComputeQCPU_generated(struct kValues kVals[M], float x[N], float y[N], float z[N],float Qr[N], float Qi[N]);

template<> kValues rand<kValues>() {
    return {
         rand<float>(),
         rand<float>(),
         rand<float>(),
         rand<float>()
    };
}

int main(int argc, const char** argv) {
    kValues kVals[M];
    float   x[N];
    float   y[N];
    float   z[N];

    float   Qr_original[N];
    float   Qi_original[N];
    float   Qr_generated[N];
    float   Qi_generated[N];

    fill_random(x);
    fill_random(y);
    fill_random(z);
    fill_random(kVals);

    fill_zero(Qr_original);
    fill_zero(Qr_generated);
    fill_zero(Qi_original);
    fill_zero(Qi_generated);

    ComputeQCPU_original (kVals, x, y, z, Qr_original,  Qi_original);
    ComputeQCPU_generated(kVals, x, y, z, Qr_generated, Qi_generated);

    auto error = (compute_error(Qi_original,  Qi_generated) +
                  compute_error(Qr_generated, Qr_generated))/2;
    if(error <= min_error<float>()) {
        exit(PASS);
    }
    else {
        std::cout << "failed: error=" << error << "\n";
        exit(FAIL);
    }
}
