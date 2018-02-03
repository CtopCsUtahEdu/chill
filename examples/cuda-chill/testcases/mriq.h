
#ifndef EXAMPLES_CUDA_CHILL_TESTCASES_MRIQ_H_
#define EXAMPLES_CUDA_CHILL_TESTCASES_MRIQ_H_

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

#ifdef __cplusplus
#define NO_EXCEPT       noexcept
#else
#define NO_EXCEPT
#endif

extern float sinf(float);
extern float cosf(float);

#define N 32768
#define M 3072

#endif /* EXAMPLES_CUDA_CHILL_TESTCASES_MRIQ_H_ */
