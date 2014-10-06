#define X 32768
#define K 256
struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};
extern float sin(float);
extern float cos(float);

void mriFH_cpu(float *rPhi,float *rRho,float *iRho, float *iPhi, float *rD, float *iD, float *kx, float *ky, float *kz, float *dx, float *dy, float *dz, float *rFHref, float *iFHref)
{

    	float rfh;
	float ifh;
	float exp;
	float cArg;
	float sArg;
    	//float rRho[K];
	//float iRho[K];
        unsigned int k;
	unsigned int x;
 
      
    for (x = 0; x < X; ++x) {
        for (k = 0; k < K; ++k) {
            
	       exp = 2 * 3.14159 * (kx[k]* dx[x] + ky[k]* dy[x] + kz[k]* dz[x]);
	       cArg = cos(exp);
	       sArg = sin(exp);
            rFHref[x] += rRho[k]* cArg - iRho[k]* sArg;
            iFHref[x] += iRho[k]*cArg + rRho[k]*sArg;
        }
         
    }
}

