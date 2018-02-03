/*
 * mriq-test.cpp
 *
 *  Created on: Jan 19, 2018
 *      Author: derick
 */

#include <cstdlib>
#include <iostream>

#include "check-run-harness.hpp"

//>procedure_compiler:      gcc -fopenmp
//>procedure_linker:        g++ -fopenmp
//>script:                  examples/chill/testcases/omp_for1.py
//>original_header:         examples/chill/testcases/dummy.h

extern "C" void mm_original(float **A, float **B, float **C, int ambn, int an, int bm);
extern "C" void mm_generated(float **A, float **B, float **C, int ambn, int an, int bm);

const size_t the_ambn   = 8;
const size_t the_an     = 8;
const size_t the_bm     = 8;

int main(int argc, const char** argv) {

    float**     A_in        = nullptr;
    float**     B_in        = nullptr;
    float**     C_original  = nullptr;
    float**     C_generated = nullptr;

    alloc_and_fill_random(A_in, the_an, the_ambn);
    alloc_and_fill_random(B_in, the_ambn, the_bm);
    alloc_and_fill_zero(C_original, the_an, the_bm);
    alloc_and_fill_zero(C_generated, the_an, the_bm);

    mm_original(A_in, B_in, C_original, the_ambn, the_an, the_bm);
    mm_generated(A_in, B_in, C_generated, the_ambn, the_an, the_bm);

//    std::cout << "A:" << std::endl;
//    print_matrix(A_in, the_an, the_ambn, std::cout);
//    std::cout << "B:" << std::endl;
//    print_matrix(B_in, the_ambn, the_bm, std::cout);
//
//    std::cout << "C original" << std::endl;
//    print_matrix(C_original,    the_an, the_bm, std::cout);
//    std::cout << "C generated" << std::endl;
//    print_matrix(C_generated,   the_an, the_bm, std::cout);

    auto error = compute_error(C_original,  C_generated, the_an, the_bm);
    if(error <= min_error<float>()) {
        std::cout << "pass" << std::endl;
        exit(PASS);
    }
    else {
        std::cout << "failed: error=" << error << std::endl;
        exit(FAIL);
    }
}
