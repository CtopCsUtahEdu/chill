#define N 1

#define VOLSIZEY 512
#define VOLSIZEX 512
#define VOLSIZEZ 1
#define ATOMCOUNT 4000
#define ATOMCOUNTTIMES4 16000
#define GRIDSPACING 0.1
#define zDim 0

extern float sqrtf(float);

void cenergy_cpu(float atoms[ATOMCOUNTTIMES4],float *energy,float z)
{
    int i,j,n;
    float dx,dy,dz; 
   
    for (j=0; j<VOLSIZEY; j++) {
        for (i=0; i<VOLSIZEX; i++) {
            for (n=0;n<ATOMCOUNT;n+=4) {
                    dx = (GRIDSPACING * i) - atoms[n];
                    dy = (GRIDSPACING * j) - atoms[n+1];
                    dz = z - atoms[n+2];
                    energy[(j*VOLSIZEX + i)+VOLSIZEX*VOLSIZEY*zDim] += atoms[n+3]/sqrtf( (dx*dx) + (dy*dy)+ (dz*dz) ) ;
            }
        }
    }
}

