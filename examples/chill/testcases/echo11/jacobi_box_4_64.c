//#include <stdint.h>
#define uint64_t unsigned int

#include "defines.h"
#include "box.h"
#include "mg.h"

#define SIZE 64
#define GHOSTS 4
#define PR_SIZE 64

#define ALPHA (-128.0/30.0)
#define BETA (14.0/30.0)
#define GAMMA (3.0/30.0)
#define DELTA (1.0/30.0)

#define PENCIL (SIZE + 2 * GHOSTS)
#define PLANE  ( PENCIL * PENCIL )

// this comment is before the function
void smooth_box_4_64(domain_type * domain, int level, int box_id, int phi_id, int rhs_id, double x, double y, int sweep)
{
	int i,j,k;
	//int ii,jj,kk;
	int t;


	//int box,s;
	//box = box_id;
        int s;
	s=sweep;

	//double _t;

	//int pencil = domain->subdomains[box].levels[level].pencil;
	//int  plane = domain->subdomains[box].levels[level].plane;
	//int ghosts = domain->subdomains[box].levels[level].ghosts;
	//int  dim_k = domain->subdomains[box].levels[level].dim.k;
	//int  dim_j = domain->subdomains[box].levels[level].dim.j;
	//int  dim_i = domain->subdomains[box].levels[level].dim.i;
	double h2inv = 1.0/(domain->h[level]*domain->h[level]);
 	double TwoThirds = 2.0/3.0;


	double  _in[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
	double  _out[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
	double  _rhs[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
	double  _lambda[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];




	for (t=0; t< GHOSTS; t++){


		if((t+s) %2 == 0){


       for (k=-GHOSTS+1+t; k<SIZE+GHOSTS-1-t; k++){ // this is to the right of k forstmt
          for (j=-GHOSTS+1+t; j<SIZE+GHOSTS-1-t; j++){
             for (i=-GHOSTS+1; i<SIZE+GHOSTS-1; i++){
                
                _out[k][j][i] = ALPHA * _in[k][j][i]   // this is inside an assignment statement
                   
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
                
                
                
             }
          }
       } // this is to the right of closing bracket of k forstmt
       // this is right AFTER k forstmt

       // this is in dead space ebtween 2 for stmts
       

			for (k=-GHOSTS+1+t; k<SIZE+GHOSTS-1-t; k++){
				for (j=-GHOSTS+1+t; j<SIZE+GHOSTS-1-t; j++){
					for (i=-GHOSTS+1; i<SIZE+GHOSTS-1; i++){
						_out[k][j][i]= x * _in[k][j][i] - y*h2inv*_out[k][j][i];

					}}}
			for (k=-GHOSTS+1+t; k<SIZE+GHOSTS-1-t; k++){
				for (j=-GHOSTS+1+t; j<SIZE+GHOSTS-1-t; j++){
					for (i=-GHOSTS+1; i<SIZE+GHOSTS-1; i++){

						_out[k][j][i] = _in[k][j][i] - TwoThirds * _lambda[k][j][i] *(_out[k][j][i]-_rhs[k][j][i]);

					}}}
		}

		if((t+s)%2 ==1)
		{
			for (k=-GHOSTS+1+t; k<SIZE+GHOSTS-1-t; k++){
				for (j=-GHOSTS+1+t; j<SIZE+GHOSTS-1-t; j++){
				for (i=-GHOSTS+1; i<SIZE+GHOSTS-1; i++){
						
						_in[k][j][i+1] = ALPHA * _out[k][j][i] 

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

			

			for (k=-GHOSTS+1+t; k<SIZE+GHOSTS-1-t; k++){
				for (j=-GHOSTS+1+t; j<SIZE+GHOSTS-1-t; j++){
					for (i=-GHOSTS+1; i<SIZE+GHOSTS-1; i++){
						_in[k][j][i]= x * _out[k][j][i] - y*h2inv*_in[k][j][i];

					}}}

			for (k=-GHOSTS+1+t; k<SIZE+GHOSTS-1-t; k++){
				for (j=-GHOSTS+1+t; j<SIZE+GHOSTS-1-t; j++){
					for (i=-GHOSTS+1; i<SIZE+GHOSTS-1; i++){

						_in[k][j][i] = _out[k][j][i] - TwoThirds * _lambda[k][j][i] *(_in[k][j][i]-_rhs[k][j][i]);

					}}}
					
		}

	}

}

// this is after everything 
