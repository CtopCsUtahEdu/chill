//#include <stdint.h>

#include "defines.h"
#include "box.h"

#define uint64_t unsigned int
#include "mg.h"

#define SIZE 64
#define GHOSTS 4
#define PR_SIZE 64


#define PENCIL (SIZE + 2 * GHOSTS)
#define PLANE  ( PENCIL * PENCIL )


void smooth_box_default(domain_type * domain, int level, int box_id, int phi_id, int rhs_id, double a, double b, int sweep)
{
	int i,j,k;
	int ii,jj,kk;
	int t;


	int box,s;
	box = box_id;
	s=sweep;
	s =0;


	int pencil = domain->subdomains[box].levels[level].pencil;
	int  plane = domain->subdomains[box].levels[level].plane;
	int ghosts = domain->subdomains[box].levels[level].ghosts;
	int  dim_k = domain->subdomains[box].levels[level].dim.k;
	int  dim_j = domain->subdomains[box].levels[level].dim.j;
	int  dim_i = domain->subdomains[box].levels[level].dim.i;
	double h2inv = 1.0/(domain->h[level]*domain->h[level]);
 	double TwoThirds = 0.50;

	double c0= -1.0/7560;
	double c1= 2.0/14175 ;
	double c2 = -11.0/16200;
	double c3= 4.0/2025;
	double c4= -16.0/14175; 
	double c5= -11.0/2100;
	double c6= 64.0/1575;
	double c7= 256.0/2835;
	double c8 = 776.0/1575;
	double c9= -6848.0/1575;


	double * __restrict__ rhs    = domain->subdomains[box].levels[level].grids[  rhs_id] + ghosts*(1+pencil+plane);
	double * __restrict__ alpha  = domain->subdomains[box].levels[level].grids[__alpha ] + ghosts*(1+pencil+plane);
	double * __restrict__ lambda = domain->subdomains[box].levels[level].grids[__lambda] + ghosts*(1+pencil+plane);

	double * __restrict__ in;     
	double * __restrict__ out;


	 in     = domain->subdomains[box].levels[level].grids[  phi_id] + ghosts*(1+pencil+plane) ;//in is phi
	 out    = domain->subdomains[box].levels[level].grids[  __temp] + ghosts*(1+pencil+plane) ;//out is phi_new

	/*
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


	for (t=0; t<GHOSTS/2; t++){

		if((t+s) %2 == 0){


			for (k=-GHOSTS+2+2*t; k<SIZE+GHOSTS-2-2*t; k++){
				for (j=-GHOSTS+2+2*t; j<SIZE+GHOSTS-2+2*t; j++){
					for (i=-GHOSTS+2; i<SIZE+GHOSTS-2; i++){




					_out[k][j][i] = c9 * _in[k][j][i]+

					c8 * (
							_in[k-1][j+0][i+0]+
							_in[k+0][j-1][i+0]+
							_in[k+0][j+0][i-1]+
							_in[k+0][j+0][i+1]+
							_in[k+0][j+1][i+0]+
							_in[k+1][j+0][i+0]
					   )


					+c5 * (

							_in[k-2][j+0][i+0]+
							_in[k+0][j-2][i+0]+
							_in[k+0][j+0][i-2]+
							_in[k+0][j+0][i+2]+
							_in[k+0][j+2][i+0]+
							_in[k+2][j+0][i+0]
					   )

					+c3 * (
							_in[k-2][j-1][i+0]+
							_in[k-2][j+0][i-1]+
							_in[k-2][j+0][i+1]+
							_in[k-2][j+1][i+0]+
							_in[k-1][j-2][i+0]+
							_in[k-1][j+0][i-2]+
							_in[k-1][j+0][i+2]+
							_in[k-1][j+2][i+0]+
							_in[k+0][j-2][i-1]+
							_in[k+0][j-2][i+1]+
							_in[k+0][j-1][i-2]+
							_in[k+0][j-1][i+2]+
							_in[k+0][j+1][i-2]+
							_in[k+0][j+1][i+2]+
							_in[k+0][j+2][i-1]+
							_in[k+0][j+2][i+1]+
							_in[k+1][j-2][i+0]+
							_in[k+1][j+0][i-2]+
							_in[k+1][j+0][i+2]+
							_in[k+1][j+2][i+0]+
							_in[k+2][j-1][i+0]+
							_in[k+2][j+0][i-1]+
							_in[k+2][j+0][i+1]+
							_in[k+2][j+1][i+0]

							)

							+c7 * (
									_in[k-1][j-1][i+0]+
									_in[k-1][j+0][i-1]+
									_in[k-1][j+0][i+1]+
									_in[k-1][j+1][i+0]+
									_in[k+0][j-1][i-1]+
									_in[k+0][j-1][i+1]+
									_in[k+0][j+1][i-1]+
									_in[k+0][j+1][i+1]+
									_in[k+1][j-1][i+0]+
									_in[k+1][j+0][i-1]+
									_in[k+1][j+0][i+1]+
									_in[k+1][j+1][i+0]
							   )

							+c2 * (
									_in[k-2][j-2][i+0]+
									_in[k-2][j+0][i-2]+
									_in[k-2][j+0][i+2]+
									_in[k-2][j+2][i+0]+
									_in[k+0][j-2][i-2]+
									_in[k+0][j-2][i+2]+
									_in[k+0][j+2][i-2]+
									_in[k+0][j+2][i+2]+
									_in[k+2][j-2][i+0]+
									_in[k+2][j+0][i-2]+
									_in[k+2][j+0][i+2]+
									_in[k+2][j+2][i+0]

							   )

							+c6 * (
									_in[k-1][j-1][i-1]+
									_in[k-1][j-1][i+1]+
									_in[k-1][j+1][i-1]+
									_in[k-1][j+1][i+1]+
									_in[k+1][j-1][i-1]+
									_in[k+1][j-1][i+1]+
									_in[k+1][j+1][i-1]+
									_in[k+1][j+1][i+1]

							   )
							+c4 *(
									_in[k-2][j-1][i-1]+
									_in[k-2][j-1][i+1]+
									_in[k-2][j+1][i-1]+
									_in[k-2][j+1][i+1]+
									_in[k-1][j-2][i-1]+
									_in[k-1][j-2][i+1]+
									_in[k-1][j-1][i-2]+
									_in[k-1][j-1][i+2]+
									_in[k-1][j+1][i-2]+
									_in[k-1][j+1][i+2]+
									_in[k-1][j+2][i-1]+
									_in[k-1][j+2][i+1]+
									_in[k+1][j-2][i-1]+
									_in[k+1][j-2][i+1]+
									_in[k+1][j-1][i-2]+
									_in[k+1][j-1][i+2]+
									_in[k+1][j+1][i-2]+
									_in[k+1][j+1][i+2]+
									_in[k+1][j+2][i-1]+
									_in[k+1][j+2][i+1]+
									_in[k+2][j-1][i-1]+
									_in[k+2][j-1][i+1]+
									_in[k+2][j+1][i-1]+
									_in[k+2][j+1][i+1]


									)
									+c1 * (

											_in[k-2][j-2][i-1]+
											_in[k-2][j-2][i+1]+
											_in[k-2][j-1][i-2]+
											_in[k-2][j-1][i+2]+
											_in[k-2][j+1][i-2]+
											_in[k-2][j+1][i+2]+
											_in[k-2][j+2][i-1]+
											_in[k-2][j+2][i+1]+
											_in[k-1][j-2][i-2]+
											_in[k-1][j-2][i+2]+
											_in[k-1][j+2][i-2]+
											_in[k-1][j+2][i+2]+
											_in[k+1][j-2][i-2]+
											_in[k+1][j-2][i+2]+
											_in[k+1][j+2][i-2]+
											_in[k+1][j+2][i+2]+
											_in[k+2][j-2][i-1]+
											_in[k+2][j-2][i+1]+
											_in[k+2][j-1][i-2]+
											_in[k+2][j-1][i+2]+
											_in[k+2][j+1][i-2]+
											_in[k+2][j+1][i+2]+
											_in[k+2][j+2][i-1]+
											_in[k+2][j+2][i+1]
											)	
											+c0 * (
													_in[k-2][j-2][i-2]+
													_in[k-2][j-2][i+2]+
													_in[k-2][j+2][i-2]+
													_in[k-2][j+2][i+2]+
													_in[k+2][j-2][i-2]+
													_in[k+2][j-2][i+2]+
													_in[k+2][j+2][i-2]+
													_in[k+2][j+2][i+2]

											   );
 
					}}}

			
			for (k=-GHOSTS+2+2*t; k<SIZE+GHOSTS-2-2*t; k++){
				for (j=-GHOSTS+2+2*t; j<SIZE+GHOSTS-2+2*t; j++){
					for (i=-GHOSTS+2; i<SIZE+GHOSTS-2; i++){
						_out[k][j][i]= a * _in[k][j][i] - b*h2inv*_out[k][j][i];

					}}}

			for (k=-GHOSTS+2+2*t; k<SIZE+GHOSTS-2-2*t; k++){
				for (j=-GHOSTS+2+2*t; j<SIZE+GHOSTS-2+2*t; j++){
					for (i=-GHOSTS+2; i<SIZE+GHOSTS-2; i++){

						_out[k][j][i] = _in[k][j][i] - TwoThirds * _lambda[k][j][i] *(_out[k][j][i]-_rhs[k][j][i]);

					}}}
		}

		
		if((t+s)%2 ==1)
		{

			for (k=-GHOSTS+2+2*t; k<SIZE+GHOSTS-2-2*t; k++){
				for (j=-GHOSTS+2+2*t; j<SIZE+GHOSTS-2+2*t; j++){
					for (i=-GHOSTS+2; i<SIZE+GHOSTS-2; i++){

						_in[k][j][i] =c9 * _out[k][j][i]+

					c8 * (
							_out[k-1][j+0][i+0]+
							_out[k+0][j-1][i+0]+
							_out[k+0][j+0][i-1]+
							_out[k+0][j+0][i+1]+
							_out[k+0][j+1][i+0]+
							_out[k+1][j+0][i+0]
					   )


					+c5 * (

							_out[k-2][j+0][i+0]+
							_out[k+0][j-2][i+0]+
							_out[k+0][j+0][i-2]+
							_out[k+0][j+0][i+2]+
							_out[k+0][j+2][i+0]+
							_out[k+2][j+0][i+0]
					   )

					+c3 * (
							_out[k-2][j-1][i+0]+
							_out[k-2][j+0][i-1]+
							_out[k-2][j+0][i+1]+
							_out[k-2][j+1][i+0]+
							_out[k-1][j-2][i+0]+
							_out[k-1][j+0][i-2]+
							_out[k-1][j+0][i+2]+
							_out[k-1][j+2][i+0]+
							_out[k+0][j-2][i-1]+
							_out[k+0][j-2][i+1]+
							_out[k+0][j-1][i-2]+
							_out[k+0][j-1][i+2]+
							_out[k+0][j+1][i-2]+
							_out[k+0][j+1][i+2]+
							_out[k+0][j+2][i-1]+
							_out[k+0][j+2][i+1]+
							_out[k+1][j-2][i+0]+
							_out[k+1][j+0][i-2]+
							_out[k+1][j+0][i+2]+
							_out[k+1][j+2][i+0]+
							_out[k+2][j-1][i+0]+
							_out[k+2][j+0][i-1]+
							_out[k+2][j+0][i+1]+
							_out[k+2][j+1][i+0]

							)

							+c7 * (
									_out[k-1][j-1][i+0]+
									_out[k-1][j+0][i-1]+
									_out[k-1][j+0][i+1]+
									_out[k-1][j+1][i+0]+
									_out[k+0][j-1][i-1]+
									_out[k+0][j-1][i+1]+
									_out[k+0][j+1][i-1]+
									_out[k+0][j+1][i+1]+
									_out[k+1][j-1][i+0]+
									_out[k+1][j+0][i-1]+
									_out[k+1][j+0][i+1]+
									_out[k+1][j+1][i+0]
							   )

							+c2 * (
									_out[k-2][j-2][i+0]+
									_out[k-2][j+0][i-2]+
									_out[k-2][j+0][i+2]+
									_out[k-2][j+2][i+0]+
									_out[k+0][j-2][i-2]+
									_out[k+0][j-2][i+2]+
									_out[k+0][j+2][i-2]+
									_out[k+0][j+2][i+2]+
									_out[k+2][j-2][i+0]+
									_out[k+2][j+0][i-2]+
									_out[k+2][j+0][i+2]+
									_out[k+2][j+2][i+0]

							   )

							+c6 * (
									_out[k-1][j-1][i-1]+
									_out[k-1][j-1][i+1]+
									_out[k-1][j+1][i-1]+
									_out[k-1][j+1][i+1]+
									_out[k+1][j-1][i-1]+
									_out[k+1][j-1][i+1]+
									_out[k+1][j+1][i-1]+
									_out[k+1][j+1][i+1]

							   )
							+c4 *(
									_out[k-2][j-1][i-1]+
									_out[k-2][j-1][i+1]+
									_out[k-2][j+1][i-1]+
									_out[k-2][j+1][i+1]+
									_out[k-1][j-2][i-1]+
									_out[k-1][j-2][i+1]+
									_out[k-1][j-1][i-2]+
									_out[k-1][j-1][i+2]+
									_out[k-1][j+1][i-2]+
									_out[k-1][j+1][i+2]+
									_out[k-1][j+2][i-1]+
									_out[k-1][j+2][i+1]+
									_out[k+1][j-2][i-1]+
									_out[k+1][j-2][i+1]+
									_out[k+1][j-1][i-2]+
									_out[k+1][j-1][i+2]+
									_out[k+1][j+1][i-2]+
									_out[k+1][j+1][i+2]+
									_out[k+1][j+2][i-1]+
									_out[k+1][j+2][i+1]+
									_out[k+2][j-1][i-1]+
									_out[k+2][j-1][i+1]+
									_out[k+2][j+1][i-1]+
									_out[k+2][j+1][i+1]


									)
									+c1 * (

											_out[k-2][j-2][i-1]+
											_out[k-2][j-2][i+1]+
											_out[k-2][j-1][i-2]+
											_out[k-2][j-1][i+2]+
											_out[k-2][j+1][i-2]+
											_out[k-2][j+1][i+2]+
											_out[k-2][j+2][i-1]+
											_out[k-2][j+2][i+1]+
											_out[k-1][j-2][i-2]+
											_out[k-1][j-2][i+2]+
											_out[k-1][j+2][i-2]+
											_out[k-1][j+2][i+2]+
											_out[k+1][j-2][i-2]+
											_out[k+1][j-2][i+2]+
											_out[k+1][j+2][i-2]+
											_out[k+1][j+2][i+2]+
											_out[k+2][j-2][i-1]+
											_out[k+2][j-2][i+1]+
											_out[k+2][j-1][i-2]+
											_out[k+2][j-1][i+2]+
											_out[k+2][j+1][i-2]+
											_out[k+2][j+1][i+2]+
											_out[k+2][j+2][i-1]+
											_out[k+2][j+2][i+1]
											)	
											+c0 * (
													_out[k-2][j-2][i-2]+
													_out[k-2][j-2][i+2]+
													_out[k-2][j+2][i-2]+
													_out[k-2][j+2][i+2]+
													_out[k+2][j-2][i-2]+
													_out[k+2][j-2][i+2]+
													_out[k+2][j+2][i-2]+
													_out[k+2][j+2][i+2]

											   );
 
					}}}

				
			for (k=-GHOSTS+2+2*t; k<SIZE+GHOSTS-2-2*t; k++){
				for (j=-GHOSTS+2+2*t; j<SIZE+GHOSTS-2+2*t; j++){
					for (i=-GHOSTS+2; i<SIZE+GHOSTS-2; i++){
						_in[k][j][i]= a * _out[k][j][i] - b*h2inv*_in[k][j][i];

					}}}

			for (k=-GHOSTS+2+2*t; k<SIZE+GHOSTS-2-2*t; k++){
				for (j=-GHOSTS+2+2*t; j<SIZE+GHOSTS-2+2*t; j++){
					for (i=-GHOSTS+2; i<SIZE+GHOSTS-2; i++){

						_in[k][j][i] = _out[k][j][i] - TwoThirds * _lambda[k][j][i] *(_in[k][j][i]-_rhs[k][j][i]);

					}}}
		}

	}

}
