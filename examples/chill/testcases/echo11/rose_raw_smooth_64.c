


// this source derived from CHILL AST originally from file 'raw_smooth_64.c' as parsed by frontend compiler rose

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

void main( box_type *box, int phi_id, int rhs_id, int temp_phi_id, double a, double b, double h, int sweep )
{
  box_type *box;
  int t8;
  int t6;
  int t4;
  int t2;
  int I;
  int J;
  int K;
  double h2inv = 1.0 / (h * h);
  //Protonu--hacks to get CHiLL's dependence analysis to work
  double _phi[64 + 8][64 + 8][64 + 8];
  double _beta_i[64 + 8][64 + 8][64 + 8];
  double _beta_j[64 + 8][64 + 8][64 + 8];
  double _beta_k[64 + 8][64 + 8][64 + 8];
  double _temp[64 + 8][64 + 8][64 + 8];
  K = box->dim.k;
  J = box->dim.j;
  I = box->dim.i;
  //  0=red, 1=black
  int color;
  color = sweep;
  for (t2 = 0; t2 <= 2; t2 += 1) {
    for (t4 = (t2 - 3); t4 <= (-t2 + 2); t4 += 1) {
      for (t6 = (t2 - 3); t6 <= (-t2 + 2); t6 += 1) {
        for (t8 = (-3 + (((((-t2 - color) - t4) - t6) - -3) % 2)); t8 <= 2; t8 += 2) {
          _temp[t4][t6][t8] = ((b * h2inv) * (((_beta_i[t4][t6][t8 + 1] * (_phi[t4][t6][t8 + 1] - _phi[t4][t6][t8]) - (_beta_i[t4][t6][t8] * (_phi[t4][t6][t8] - _phi[t4][t6][t8 - 1]))) + _beta_j[t4][t6 + 1][t8] * (_phi[t4][t6 + 1][t8] - _phi[t4][t6][t8]) - (_beta_j[t4][t6][t8] * (_phi[t4][t6][t8] - _phi[t4][t6 - 1][t8]))) + _beta_k[t4 + 1][t6][t8] * (_phi[t4 + 1][t6][t8] - _phi[t4][t6][t8]) - (_beta_k[t4][t6][t8] * (_phi[t4][t6][t8] - _phi[t4 - 1][t6][t8]))));
        }
      }
    }
  }

}
