/*
 * cp-test.cpp
 *
 *  Created on: Apr 5, 2018
 *      Author: derick
 */

#include <cstdlib>
#include <iostream>

#include "check-run-harness.hpp"

//>procedure_compiler:      nvcc
//>procedure_linker:        nvcc
//>script:                  examples/cuda-chill/testcases/cp.py
//>original_header:         examples/cuda-chill/testcases/dummy.h

#define N                   1

#define VOLSIZEY            512
#define VOLSIZEX            512
#define VOLSIZEZ            1
#define ATOMCOUNT           4000
#define ATOMCOUNTTIMES4     16000
#define GRIDSPACING         0.1
#define zDim                0

void cenergy_cpu_original(float atoms[ATOMCOUNTTIMES4],float *energy,float z);
void cenergy_cpu_generated(float atoms[ATOMCOUNTTIMES4],float *energy,float z);

int main(int argc, const char** argv) {
    float atoms[ATOMCOUNTTIMES4];

    float energy_original[VOLSIZEX * VOLSIZEY * VOLSIZEZ];
    float energy_generated[VOLSIZEX * VOLSIZEY * VOLSIZEZ];

    fill_random(atoms);
    fill_zero(energy_original);
    fill_zero(energy_generated);

    cenergy_cpu_original(atoms, energy_original, 0);
    cenergy_cpu_generated(atoms, energy_generated, 0);

    float error = compute_error(energy_original, energy_generated);
    if(error <= min_error<float>()) {
        exit(PASS);
    }
    exit(FAIL);
}
