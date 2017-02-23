//------------------------------------------------------------------------------------------------------------------------------
#include "defines.h"
#include "box.h"
//------------------------------------------------------------------------------------------------------------------------------

#define PR_SIZE 64
#define GHOSTS 4

//------------------------------------------------------------------------------------------------------------------------------
//Protonu--should rename this function
//1. smooth_multiple_laplaceGSRB4
//2. smooth_multiple_helmholtzGSRB4
//3. smooth_multiple_SOR
void main(box_type *box, int phi_id, int rhs_id, int temp_phi_id, double a, double b, double h, int sweep){
  int i,j,k,s;
  int I,J,K;
  int pencil = box->pencil;
  int plane = box->plane;
  int ghosts = box->ghosts;
  double h2inv = 1.0/(h*h);

  
  double * __restrict__ phi    = box->grids[  phi_id] + ghosts*plane + ghosts*pencil + ghosts; // i.e. [0] = first non ghost zone point
  double * __restrict__ rhs    = box->grids[  rhs_id] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ alpha  = box->grids[__alpha ] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ beta_i = box->grids[__beta_i] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ beta_j = box->grids[__beta_j] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ beta_k = box->grids[__beta_k] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ lambda = box->grids[__lambda] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ temp   = box->grids[ temp_phi_id  ] + ghosts*plane + ghosts*pencil + ghosts;
  

  //Protonu--hacks to get CHiLL's dependence analysis to work

   double (*_phi)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double (*_rhs)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double (*_alpha)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double (*_beta_i)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double (*_beta_j)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double (*_beta_k)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double (*_lambda)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double (*_temp)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];

  /* double _phi[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double _rhs[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double _alpha[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double _beta_i[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double _beta_j[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double _beta_k[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double _lambda[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
   double _temp[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS];
  */
   //Protonu--more hack, this might have to re-implemented later
   //extracring the offsets, with CHiLL we can set bounds to these values
   /*
   int off_i = box->low.i;
   int off_j = box->low.j;
   int off_k = box->low.k;
   */



   _phi = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(phi);
   _rhs = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(rhs);
   _alpha = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(alpha);
   _beta_i = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(beta_i);
   _beta_j = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(beta_j);
   _beta_k = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(beta_k);
   _lambda = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(lambda);
   _temp = (double (*)[PR_SIZE+2*GHOSTS][PR_SIZE+2*GHOSTS])(temp);


   
   K = box->dim.k;
   J = box->dim.j;
   I = box->dim.i;
   


  int color; //  0=red, 1=black
  int ghostsToOperateOn=ghosts-1;

  color = sweep;
 
  #define ghostsToOperateOn (ghosts - s-2 )

  for(s=0; s < ghosts; s++){

	  //For the Red Pass
	  //laplacian where the output is into temp_1 , input in phi
	  //Perform the helmholtz on temp_1
	  //on appropiate red points only, perform the gsrb smooth and output to temp_phi

	  for(k= (s -ghosts+ 1);k<K+(ghosts - s-1);k++){
		  for(j= (s -ghosts+ 1);j<J+(ghosts - s-1);j++){
			  for(i= (s -ghosts+ 1);i<I+(ghosts -s-1);i++){

				if(( i+ j + k + (color+s) ) % 2 == 0 ) {
				  _temp[k][j][i] =  b*h2inv*(
						  _beta_i[k][j][i+1] *( _phi[k][j][i+1]-_phi[k][j][i] )
						  -_beta_i[k][j][i]  *( _phi[k][j][i]-_phi[k][j][i-1] )
						  +_beta_j[k][j+1][i]*( _phi[k][j+1][i]-_phi[k][j][i] )
						  -_beta_j[k][j][i]  *( _phi[k][j][i]-_phi[k][j-1][i] )
						  +_beta_k[k+1][j][i]*( _phi[k+1][j][i]-_phi[k][j][i] )
						  -_beta_k[k][j][i]*( _phi[k][j][i]-_phi[k-1][j][i] )
						  );

				  _temp[k][j][i] =  a*_alpha[k][j][i]*_phi[k][j][i] - _temp[k][j][i]; // helmholtz = a alphi I - helmholtz

				  	  _phi[k][j][i] = _phi[k][j][i] - _lambda[k][j][i]*(_temp[k][j][i]-_rhs[k][j][i]);
				  }

			  }}} 

  }

}



