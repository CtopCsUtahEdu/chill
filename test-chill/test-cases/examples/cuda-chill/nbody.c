#define NBODIES 16384
#define SOFTENINGSQUARED 0.01f
#define DELTATIME 0.001f
#define DAMPING 1.0f

#define NBLOCKSY 1
#define NBLOCKSX (NBODIES/NTHREADSX)
#define NTHREADSY 1 
#define NTHREADSX 64

#define BLOCKSIZE 128

#define SHARED 1
#define TIMER 1
#define VERIFY 1

extern float sqrtf(float);

void nbody_cpu(float* oldpos,float* oldpos1, float *newpos, float *oldvel, float *newvel, float *force)
{
    float r0,r1,r2;
    float invDist, invDistCube, mass, invMass;
    unsigned int i,j;
    for(i = 0; i < NBODIES; ++i) {
        //force[i*4  ] = 0;
        //force[i*4+1] = 0;
        //force[i*4+2] = 0;
        //force[i*4+3] = 0;
        for(j = 0; j < NBODIES; ++j) {
	    r0 = oldpos[j*4]-oldpos1[i*4];
	    r1 = oldpos[j*4+1]-oldpos1[i*4+1];
	    r2 = oldpos[j*4+2]-oldpos1[i*4+2];

	    invDist = 1.0/sqrtf(r0 * r0 + r1 * r1 + r2 * r2 + SOFTENINGSQUARED);
	    invDistCube =  invDist * invDist * invDist;
	    mass = oldpos1[i*4+3];

	    force[i*4] = force[i*4] + r0 * mass * invDistCube;
	    force[i*4+1] = force[i*4+1] + r1 * mass * invDistCube;
	    force[i*4+2] = force[i*4+2] + r2 * mass * invDistCube;

        }
    }

/*    for (i = 0; i < NBODIES; ++i) {
        invMass = oldvel[4*i+3];

        oldvel[4*i] += (force[4*i] * invMass) * DELTATIME * DAMPING;
        oldvel[4*i+1] += (force[4*i+1] * invMass) * DELTATIME * DAMPING;
        oldvel[4*i+2] += (force[4*i+2] * invMass) * DELTATIME * DAMPING;

        oldpos[4*i] += oldvel[4*i] * DELTATIME;
        oldpos[4*i+1] += oldvel[4*i+1] * DELTATIME;
        oldpos[4*i+2] += oldvel[4*i+2] * DELTATIME;

        newpos[4*i+0] = oldpos[4*i];
        newpos[4*i+1] = oldpos[4*i+1];
        newpos[4*i+2] = oldpos[4*i+2];
        newpos[4*i+3] = oldpos[4*i+3];

        newvel[4*i+0] = oldvel[4*i];
        newvel[4*i+1] = oldvel[4*i+1];
        newvel[4*i+2] = oldvel[4*i+2];
        newvel[4*i+3] = oldvel[4*i+3];
    }*/
}
