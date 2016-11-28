#define N 32768
#define M 3072
struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};
extern float sinf(float);
extern float cosf(float);

void
ComputeQCPU(int numK, int numX,struct kValues kVals[M],float x[N], float y[N], float z[N],float Qr[N], float Qi[N]) {
  float expArg;
  float cosArg;
  float sinArg;
  float phi;
  int i;
  int j;
  numK = M;
  numX = N;
  for ( i = 0; i < M; i++) {
    for ( j = 0; j < N; j++) {
      expArg = 6.2831853071795864769252867665590058f * (kVals[i].Kx * x[j] +kVals[i].Ky * y[j] +kVals[i].Kz * z[j]);
      cosArg = cosf(expArg);
      sinArg = sinf(expArg);
      phi = kVals[i].PhiMag;
      Qr[j] += phi * cosArg;
      Qi[j] += phi * sinArg;
    }
  }
}
  
