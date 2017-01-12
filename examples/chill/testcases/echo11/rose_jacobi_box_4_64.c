


// this source derived from CHILL AST originally from file 'jacobi_box_4_64.c' as parsed by frontend compiler rose

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

void smooth_box_4_64( domain_type *domain, int level, int box_id, int phi_id, int rhs_id, double x, double y, int sweep )
{
  int t8;
  int t6;
  int t4;
  int t2;
  double treg_0;
  double treg_1;
  double treg_2;
  double buffer_0[72];
  double buffer_1[72];
  double buffer_2[72];
  //int box,s;
  //box = box_id;
  int s;
  s = sweep;
  //double _t;
  //int pencil = domain->subdomains[box].levels[level].pencil;
  //int  plane = domain->subdomains[box].levels[level].plane;
  //int ghosts = domain->subdomains[box].levels[level].ghosts;
  //int  dim_k = domain->subdomains[box].levels[level].dim.k;
  //int  dim_j = domain->subdomains[box].levels[level].dim.j;
  //int  dim_i = domain->subdomains[box].levels[level].dim.i;
  double h2inv = 1.0 / (domain->h[level] * domain->h[level]);
  double TwoThirds = 2.0 / 3.0;
  double _in[64 + 8][64 + 8][64 + 8];
  double _out[64 + 8][64 + 8][64 + 8];
  double _rhs[64 + 8][64 + 8][64 + 8];
  double _lambda[64 + 8][64 + 8][64 + 8];
  for (t2 = -6; t2 <= 70; t2 += 1) {
    for (t4 = (((-s % 2) < ((t2 - 66) + ((-s - (t2 - 66)) % 2))?(-s % 2):((t2 - 66) + ((-s - (t2 - 66)) % 2)))); t4 <= (((3) < ((t2 + 3) / 3)?(3):((t2 + 3) / 3))); t4 += 2) {
      for (t6 = (t4 - 3); t6 <= (-t4 + 66); t6 += 1) {
        treg_0 = _in[-(2 * t4) + t2][t6][-4];
        treg_1 = (((_in[(-(2 * t4) + t2) + 1][t6][-4] + _in[(-(2 * t4) + t2) + -1][t6][-4]) + _in[-(2 * t4) + t2][t6 + 1][-4]) + _in[-(2 * t4) + t2][t6 + -1][-4]);
        treg_2 = (((_in[(-(2 * t4) + t2) + 1][t6 + 1][-4] + _in[(-(2 * t4) + t2) + -1][t6 + 1][-4]) + _in[(-(2 * t4) + t2) + -1][t6 + -1][-4]) + _in[(-(2 * t4) + t2) + 1][t6 + -1][-4]);
        buffer_0[0] = ((treg_0 * (14.0 / 30.0) + treg_1 * (3.0 / 30.0)) + treg_2 * (1.0 / 30.0));
        treg_0 = _in[-(2 * t4) + t2][t6][-3];
        treg_1 = (((_in[(-(2 * t4) + t2) + 1][t6][-3] + _in[(-(2 * t4) + t2) + -1][t6][-3]) + _in[-(2 * t4) + t2][t6 + 1][-3]) + _in[-(2 * t4) + t2][t6 + -1][-3]);
        treg_2 = (((_in[(-(2 * t4) + t2) + 1][t6 + 1][-3] + _in[(-(2 * t4) + t2) + -1][t6 + 1][-3]) + _in[(-(2 * t4) + t2) + -1][t6 + -1][-3]) + _in[(-(2 * t4) + t2) + 1][t6 + -1][-3]);
        buffer_1[0] = ((treg_0 * (-128.0 / 30.0) + treg_1 * (14.0 / 30.0)) + treg_2 * (3.0 / 30.0));
        buffer_0[1] = ((treg_0 * (14.0 / 30.0) + treg_1 * (3.0 / 30.0)) + treg_2 * (1.0 / 30.0));
        for (t8 = -3; t8 <= 66; t8 += 1) {
          treg_0 = _in[-(2 * t4) + t2][t6][t8 + 1];
          treg_1 = (((_in[(-(2 * t4) + t2) + 1][t6][t8 + 1] + _in[(-(2 * t4) + t2) + -1][t6][t8 + 1]) + _in[-(2 * t4) + t2][t6 + 1][t8 + 1]) + _in[-(2 * t4) + t2][t6 + -1][t8 + 1]);
          treg_2 = (((_in[(-(2 * t4) + t2) + 1][t6 + 1][t8 + 1] + _in[(-(2 * t4) + t2) + -1][t6 + 1][t8 + 1]) + _in[(-(2 * t4) + t2) + -1][t6 + -1][t8 + 1]) + _in[(-(2 * t4) + t2) + 1][t6 + -1][t8 + 1]);
          buffer_2[(t8 + 0) + 3] = ((treg_0 * (14.0 / 30.0) + treg_1 * (3.0 / 30.0)) + treg_2 * (1.0 / 30.0));
          buffer_1[(t8 + 1) + 3] = ((treg_0 * (-128.0 / 30.0) + treg_1 * (14.0 / 30.0)) + treg_2 * (3.0 / 30.0));
          buffer_0[(t8 + 2) + 3] = ((treg_0 * (14.0 / 30.0) + treg_1 * (3.0 / 30.0)) + treg_2 * (1.0 / 30.0));
        }
        for (t8 = -3; t8 <= 66; t8 += 1) {
          _out[-(2 * t4) + t2][t6][t8] = ((buffer_0[t8 + 3] + buffer_1[t8 + 3]) + buffer_2[t8 + 3]);
        }
      }
    }
    for (t4 = (((-s % 2) < ((t2 - 66) + ((-s - (t2 - 66)) % 2))?(-s % 2):((t2 - 66) + ((-s - (t2 - 66)) % 2)))); t4 <= (((3) < ((t2 + 3) / 3)?(3):((t2 + 3) / 3))); t4 += 2) {
      for (t6 = (t4 - 3); t6 <= (-t4 + 66); t6 += 1) {
        for (t8 = -3; t8 <= 66; t8 += 1) {
          _out[-(2 * t4) + t2][t6][t8] = (x * _in[-(2 * t4) + t2][t6][t8] - ((y * h2inv) * _out[-(2 * t4) + t2][t6][t8]));
        }
      }
    }
    for (t4 = (((-s % 2) < ((t2 - 66) + ((-s - (t2 - 66)) % 2))?(-s % 2):((t2 - 66) + ((-s - (t2 - 66)) % 2)))); t4 <= (((3) < ((t2 + 3) / 3)?(3):((t2 + 3) / 3))); t4 += 2) {
      for (t6 = (t4 - 3); t6 <= (-t4 + 66); t6 += 1) {
        for (t8 = -3; t8 <= 66; t8 += 1) {
          _out[-(2 * t4) + t2][t6][t8] = (_in[-(2 * t4) + t2][t6][t8] - ((TwoThirds * _lambda[-(2 * t4) + t2][t6][t8]) * (_out[-(2 * t4) + t2][t6][t8] - _rhs[-(2 * t4) + t2][t6][t8])));
        }
      }
    }
    for (t4 = ((((-s - 1) % 2) < ((t2 - 66) + (((-s - 1) - (t2 - 66)) % 2))?((-s - 1) % 2):((t2 - 66) + (((-s - 1) - (t2 - 66)) % 2)))); t4 <= (((3) < ((t2 + 3) / 3)?(3):((t2 + 3) / 3))); t4 += 2) {
      for (t6 = (t4 - 3); t6 <= (-t4 + 66); t6 += 1) {
        for (t8 = -3; t8 <= 66; t8 += 1) {
          _in[-(2 * t4) + t2][t6][t8 + 1] = ((((-128.0 / 30.0) * _out[-(2 * t4) + t2][t6][t8] + (14.0 / 30.0) * (((((_out[-(2 * t4) + t2 - 1][t6][t8] + _out[-(2 * t4) + t2][t6 - 1][t8]) + _out[-(2 * t4) + t2][t6 + 1][t8]) + _out[(-(2 * t4) + t2) + 1][t6][t8]) + _out[-(2 * t4) + t2][t6][t8 - 1]) + _out[-(2 * t4) + t2][t6][t8 + 1])) + (3.0 / 30.0) * (((((((((((_out[-(2 * t4) + t2 - 1][t6][t8 - 1] + _out[-(2 * t4) + t2][t6 - 1][t8 - 1]) + _out[-(2 * t4) + t2][t6 + 1][t8 - 1]) + _out[(-(2 * t4) + t2) + 1][t6][t8 - 1]) + _out[-(2 * t4) + t2 - 1][t6 - 1][t8]) + _out[-(2 * t4) + t2 - 1][t6 + 1][t8]) + _out[(-(2 * t4) + t2) + 1][t6 - 1][t8]) + _out[(-(2 * t4) + t2) + 1][t6 + 1][t8]) + _out[-(2 * t4) + t2 - 1][t6][t8 + 1]) + _out[-(2 * t4) + t2][t6 - 1][t8 + 1]) + _out[-(2 * t4) + t2][t6 + 1][t8 + 1]) + _out[(-(2 * t4) + t2) + 1][t6][t8 + 1])) + (1.0 / 30.0) * (((((((_out[-(2 * t4) + t2 - 1][t6 - 1][t8 - 1] + _out[-(2 * t4) + t2 - 1][t6 + 1][t8 - 1]) + _out[(-(2 * t4) + t2) + 1][t6 - 1][t8 - 1]) + _out[(-(2 * t4) + t2) + 1][t6 + 1][t8 - 1]) + _out[-(2 * t4) + t2 - 1][t6 - 1][t8 + 1]) + _out[-(2 * t4) + t2 - 1][t6 + 1][t8 + 1]) + _out[(-(2 * t4) + t2) + 1][t6 - 1][t8 + 1]) + _out[(-(2 * t4) + t2) + 1][t6 + 1][t8 + 1]));
        }
      }
    }
    for (t4 = ((((-s - 1) % 2) < ((t2 - 66) + (((-s - 1) - (t2 - 66)) % 2))?((-s - 1) % 2):((t2 - 66) + (((-s - 1) - (t2 - 66)) % 2)))); t4 <= (((3) < ((t2 + 3) / 3)?(3):((t2 + 3) / 3))); t4 += 2) {
      for (t6 = (t4 - 3); t6 <= (-t4 + 66); t6 += 1) {
        for (t8 = -3; t8 <= 66; t8 += 1) {
          _in[-(2 * t4) + t2][t6][t8] = (x * _out[-(2 * t4) + t2][t6][t8] - ((y * h2inv) * _in[-(2 * t4) + t2][t6][t8]));
        }
      }
    }
    for (t4 = ((((-s - 1) % 2) < ((t2 - 66) + (((-s - 1) - (t2 - 66)) % 2))?((-s - 1) % 2):((t2 - 66) + (((-s - 1) - (t2 - 66)) % 2)))); t4 <= (((3) < ((t2 + 3) / 3)?(3):((t2 + 3) / 3))); t4 += 2) {
      for (t6 = (t4 - 3); t6 <= (-t4 + 66); t6 += 1) {
        for (t8 = -3; t8 <= 66; t8 += 1) {
          _in[-(2 * t4) + t2][t6][t8] = (_out[-(2 * t4) + t2][t6][t8] - ((TwoThirds * _lambda[-(2 * t4) + t2][t6][t8]) * (_in[-(2 * t4) + t2][t6][t8] - _rhs[-(2 * t4) + t2][t6][t8])));
        }
      }
    }
  }

}
// this is after everything 
