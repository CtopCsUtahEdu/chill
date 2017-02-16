//#include <stdint.h>
#define uint64_t unsigned int

#include "defines.h"
#include "box.h"
#include "mg.h"

#define SIZE 32
#define GHOSTS 1
#define PR_SIZE 32

#define ALPHA (-128.0/30.0)
#define BETA (14.0/30.0)
#define GAMMA (3.0/30.0)
#define DELTA (1.0/30.0)

#define PENCIL (SIZE + 2 * GHOSTS)
#define PLANE  ( PENCIL * PENCIL )


void smooth_box_1_32(domain_type * domain, int level, int box_id, int phi_id, int rhs_id, double x, double y, int sweep)
{
	int i,j,k;
	int ii,jj,kk;
	int t;


	int box,s;
	box = box_id;
	s=sweep;


	int pencil = domain->subdomains[box].levels[level].pencil;
	int  plane = domain->subdomains[box].levels[level].plane;
	int ghosts = domain->subdomains[box].levels[level].ghosts;
	int  dim_k = domain->subdomains[box].levels[level].dim.k;
	int  dim_j = domain->subdomains[box].levels[level].dim.j;
	int  dim_i = domain->subdomains[box].levels[level].dim.i;
	double h2inv = 1.0/(domain->h[level]*domain->h[level]);
 	double TwoThirds = 2.0/3.0;


	/*
	double * __restrict__ rhs    = domain->subdomains[box].levels[level].grids[  rhs_id] + ghosts*(1+pencil+plane);
	double * __restrict__ alpha  = domain->subdomains[box].levels[level].grids[__alpha ] + ghosts*(1+pencil+plane);
	double * __restrict__ lambda = domain->subdomains[box].levels[level].grids[__lambda] + ghosts*(1+pencil+plane);

	double * __restrict__ in;     
	double * __restrict__ out;

	 in     = domain->subdomains[box].levels[level].grids[  phi_id] + ghosts*(1+pencil+plane) ;//in is phi
	 out    = domain->subdomains[box].levels[level].grids[  __temp] + ghosts*(1+pencil+plane) ;//out is phi_new


	double (* __restrict__ _in)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
	double (* __restrict__ _out)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
	double (* __restrict__ _rhs)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
	double (* __restrict__ _lambda)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];

	double _temp[PENCIL][PENCIL][PENCIL];
	double _temp2[PENCIL][PENCIL][PENCIL];

	_in = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(in);
	_out = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(out);
	_lambda = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(lambda);
	_rhs = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(rhs);
	*/

	
	double  _in[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
	double  _out[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
	double  _rhs[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
	double  _lambda[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];


	for (t=0; t<GHOSTS; t++){

		if((t+s) %2 == 0){


			for (k=0; k<SIZE+GHOSTS-1; k++){
				for (j=0; j<SIZE+GHOSTS-1; j++){
					for (i=0; i<SIZE+GHOSTS-1; i++){

						_out[k][j][i] = ALPHA * _in[k][j][i] 

							+  BETA * (  	_in[k-1][j][i] + _in[k][j-1][i] 
									+ _in[k][j+1][i] + _in[k+1][j][i]
									+ _in[k][j][i-1] + _in[k][j][i+1]
								  )

							+ GAMMA * (  	_in[k-1][j][i-1] + _in[k][j-1][i-1] 
									+ _in[k][j+1][i-1] + _in[k+1][j][i-1]
									+ _in[k-1][j-1][i] + _in[k-1][j+1][i]
									+ _in[k+1][j-1][i] + _in[k+1][j+1][i]
									+ _in[k-1][j][i+1] + _in[k][j-1][i+1]
									+ _in[k][j+1][i+1] + _in[k+1][j][i+1]
								  )

							+ DELTA * (  	_in[k-1][j-1][i-1] + _in[k-1][j+1][i-1] 
									+ _in[k+1][j-1][i-1] + _in[k+1][j+1][i-1]
									+ _in[k-1][j-1][i+1] + _in[k-1][j+1][i+1]
									+ _in[k+1][j-1][i+1] + _in[k+1][j+1][i+1]
								  );

					}}}


			for (k=0; k<SIZE+GHOSTS-1; k++){
				for (j=0; j<SIZE+GHOSTS-1; j++){
					for (i=0; i<SIZE+GHOSTS-1; i++){
						_out[k][j][i]= x * _in[k][j][i] - y*h2inv*_out[k][j][i];

					}}}

			for (k=0; k<SIZE+GHOSTS-1; k++){
				for (j=0; j<SIZE+GHOSTS-1; j++){
					for (i=0; i<SIZE+GHOSTS-1; i++){

						_out[k][j][i] = _in[k][j][i] - TwoThirds * _lambda[k][j][i] *(_out[k][j][i]-_rhs[k][j][i]);

					}}}
		}

		if((t+s)%2 ==1)
		{
			for (k=0; k<SIZE+GHOSTS-1; k++){
				for (j=0; j<SIZE+GHOSTS-1; j++){
					for (i=0; i<SIZE+GHOSTS-1; i++){

						_in[k][j][i] = ALPHA * _out[k][j][i] 

							+  BETA * (  	_out[k-1][j][i] + _out[k][j-1][i] 
									+ _out[k][j+1][i] + _out[k+1][j][i]
									+ _out[k][j][i-1] + _out[k][j][i+1]
								  )

							+ GAMMA * (  	_out[k-1][j][i-1] + _out[k][j-1][i-1] 
									+ _out[k][j+1][i-1] + _out[k+1][j][i-1]
									+ _out[k-1][j-1][i] + _out[k-1][j+1][i]
									+ _out[k+1][j-1][i] + _out[k+1][j+1][i]
									+ _out[k-1][j][i+1] + _out[k][j-1][i+1]
									+ _out[k][j+1][i+1] + _out[k+1][j][i+1]
								  )

							+ DELTA * (  	_out[k-1][j-1][i-1] + _out[k-1][j+1][i-1] 
									+ _out[k+1][j-1][i-1] + _out[k+1][j+1][i-1]
									+ _out[k-1][j-1][i+1] + _out[k-1][j+1][i+1]
									+ _out[k+1][j-1][i+1] + _out[k+1][j+1][i+1]
								  );

					}}}


			for (k=0; k<SIZE+GHOSTS-1; k++){
				for (j=0; j<SIZE+GHOSTS-1; j++){
					for (i=0; i<SIZE+GHOSTS-1; i++){
						_in[k][j][i]= x * _out[k][j][i] - y*h2inv*_in[k][j][i];

					}}}

			for (k=0; k<SIZE+GHOSTS-1; k++){
				for (j=0; j<SIZE+GHOSTS-1; j++){
					for (i=0; i<SIZE+GHOSTS-1; i++){

						_in[k][j][i] = _out[k][j][i] - TwoThirds * _lambda[k][j][i] *(_in[k][j][i]-_rhs[k][j][i]);

					}}}
		}

	}

}
