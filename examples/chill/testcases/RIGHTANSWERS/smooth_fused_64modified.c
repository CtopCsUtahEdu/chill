// this source is derived from CHILL AST originally from file 'smooth_fused_64.c' as parsed by frontend compiler rose

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

void main(box_type *box, int phi_id, int rhs_id, int temp_phi_id, double a, double b, double h, int sweep) {
  int t8;
  int t6;
  int t4;
  int t2;
  int I;
  int J;
  int K;
  double h2inv = 1 / (h * h);
  // i.e. [0] = first non ghost zone point

  double * __restrict__ phi = box->grids[phi_id] + ghosts * plane + ghosts * pencil + ghosts;
  double * __restrict__ rhs = box->grids[rhs_id] + ghosts * plane + ghosts * pencil + ghosts;
  double * __restrict__ alpha = box->grids[2] + ghosts * plane + ghosts * pencil + ghosts;
  double * __restrict__ beta_i = box->grids[3] + ghosts * plane + ghosts * pencil + ghosts;
  double * __restrict__ beta_j = box->grids[4] + ghosts * plane + ghosts * pencil + ghosts;
  double * __restrict__ beta_k = box->grids[5] + ghosts * plane + ghosts * pencil + ghosts;
  double * __restrict__ lambda = box->grids[6] + ghosts * plane + ghosts * pencil + ghosts;
  double * __restrict__ temp = box->grids[temp_phi_id] + ghosts * plane + ghosts * pencil + ghosts;
  //Protonu--hacks to get CHiLL's dependence analysis to work

  double (*_phi)[64 + 8][64 + 8];
  double (*_rhs)[64 + 8][64 + 8];
  double (*_alpha)[64 + 8][64 + 8];
  double (*_beta_i)[64 + 8][64 + 8];
  double (*_beta_j)[64 + 8][64 + 8];
  double (*_beta_k)[64 + 8][64 + 8];
  double (*_lambda)[64 + 8][64 + 8];
  double (*_temp)[64 + 8][64 + 8];
  //Protonu--more hack, this might have to re-implemented later

  //extracring the offsets, with CHiLL we can set bounds to these values

  _phi = (double(*)[64 + 8][64 + 8])phi;
  _rhs = (double(*)[64 + 8][64 + 8])rhs;
  _alpha = (double(*)[64 + 8][64 + 8])alpha;
  _beta_i = (double(*)[64 + 8][64 + 8])beta_i;
  _beta_j = (double(*)[64 + 8][64 + 8])beta_j;
  _beta_k = (double(*)[64 + 8][64 + 8])beta_k;
  _lambda = (double(*)[64 + 8][64 + 8])lambda;
  _temp = (double(*)[64 + 8][64 + 8])temp;
  K = box->dim.k;
  J = box->dim.j;
  I = box->dim.i;
  //  0=red, 1=black

  int color;
  color = sweep;
  for (t2 = 0; t2 <= 3; t2 += 1) 
    for (t4 = t2 - 3; t4 <= -t2 + 66; t4 += 1) 
      for (t6 = t2 - 3; t6 <= -t2 + 66; t6 += 1) 
        for (t8 = t2 - 3 + (-t6 - color - t4 - t2 - (t2 - 3)) % 2; t8 <= -t2 + 66; t8 += 2) {
          _temp[t4][t6][t8] = b * h2inv * (_beta_i[t4][t6][t8 + 1] * (_phi[t4][t6][t8 + 1] - _phi[t4][t6][t8]) - _beta_i[t4][t6][t8] * (_phi[t4][t6][t8] - _phi[t4][t6][t8 - 1]) + _beta_j[t4][t6 + 1][t8] * (_phi[t4][t6 + 1][t8] - _phi[t4][t6][t8]) - _beta_j[t4][t6][t8] * (_phi[t4][t6][t8] - _phi[t4][t6 - 1][t8]) + _beta_k[t4 + 1][t6][t8] * (_phi[t4 + 1][t6][t8] - _phi[t4][t6][t8]) - _beta_k[t4][t6][t8] * (_phi[t4][t6][t8] - _phi[t4 - 1][t6][t8]));
          _temp[t4][t6][t8] = a * _alpha[t4][t6][t8] * _phi[t4][t6][t8] - _temp[t4][t6][t8];
          _phi[t4][t6][t8] = _phi[t4][t6][t8] - _lambda[t4][t6][t8] * (_temp[t4][t6][t8] - _rhs[t4][t6][t8]);
        }
}
