/*
 * mriq.h
 *
 *  Created on: Jan 25, 2018
 *      Author: derick
 */

#ifndef EXAMPLES_CUDA_CHILL_TESTCASES_MRIQ_H_
#define EXAMPLES_CUDA_CHILL_TESTCASES_MRIQ_H_

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

extern float sinf(float);
extern float cosf(float);

#define N 32768
#define M 3072

#endif /* EXAMPLES_CUDA_CHILL_TESTCASES_MRIQ_H_ */
