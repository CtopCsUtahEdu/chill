// this source is derived from CHILL AST originally from file 'jacobi_box_1_32.c' as parsed by frontend compiler rose

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

void smooth_box_1_32(domain_type *domain, int level, int box_id, int phi_id, int rhs_id, double x, double y, int sweep) {
  int t8;
  int t6;
  int t4;
  double treg_0;
  double treg_1;
  double treg_2;
  double buffer_0[34];
  double buffer_1[34];
  double buffer_2[34];
  int box;
  int s;
  box = box_id;
  s = sweep;
  double h2inv = 1 / (domain->h[level] * domain->h[level]);
  double TwoThirds = 2 / 3;
  double _in[32 + 2][32 + 2][32 + 2];
  double _out[32 + 2][32 + 2][32 + 2];
  double _rhs[32 + 2][32 + 2][32 + 2];
  double _lambda[32 + 2][32 + 2][32 + 2];
  if (s % 2 == 0) 
    for (t4 = 0; t4 <= 31; t4 += 1) 
      for (t6 = 0; t6 <= 31; t6 += 1) 
        for (t8 = 0; t8 <= 31; t8 += 1) {
          _out[t4][t6][t8] = -128 / 30 * _in[t4][t6][t8] + 14 / 30 * (_in[t4 - 1][t6][t8] + _in[t4][t6 - 1][t8] + _in[t4][t6 + 1][t8] + _in[t4 + 1][t6][t8] + _in[t4][t6][t8 - 1] + _in[t4][t6][t8 + 1]) + 3 / 30 * (_in[t4 - 1][t6][t8 - 1] + _in[t4][t6 - 1][t8 - 1] + _in[t4][t6 + 1][t8 - 1] + _in[t4 + 1][t6][t8 - 1] + _in[t4 - 1][t6 - 1][t8] + _in[t4 - 1][t6 + 1][t8] + _in[t4 + 1][t6 - 1][t8] + _in[t4 + 1][t6 + 1][t8] + _in[t4 - 1][t6][t8 + 1] + _in[t4][t6 - 1][t8 + 1] + _in[t4][t6 + 1][t8 + 1] + _in[t4 + 1][t6][t8 + 1]) + 1 / 30 * (_in[t4 - 1][t6 - 1][t8 - 1] + _in[t4 - 1][t6 + 1][t8 - 1] + _in[t4 + 1][t6 - 1][t8 - 1] + _in[t4 + 1][t6 + 1][t8 - 1] + _in[t4 - 1][t6 - 1][t8 + 1] + _in[t4 - 1][t6 + 1][t8 + 1] + _in[t4 + 1][t6 - 1][t8 + 1] + _in[t4 + 1][t6 + 1][t8 + 1]);
          _out[t4][t6][t8] = x * _in[t4][t6][t8] - y * h2inv * _out[t4][t6][t8];
          _out[t4][t6][t8] = _in[t4][t6][t8] - TwoThirds * _lambda[t4][t6][t8] * (_out[t4][t6][t8] - _rhs[t4][t6][t8]);
        }
  else 
    for (t4 = 0; t4 <= 31; t4 += 1) 
      for (t6 = 0; t6 <= 31; t6 += 1) {
        treg_0 = _out[t4][t6][-1];
        treg_1 = _out[t4 + 1][t6][-1] + _out[t4 + -1][t6][-1] + _out[t4][t6 + 1][-1] + _out[t4][t6 + -1][-1];
        treg_2 = _out[t4 + 1][t6 + 1][-1] + _out[t4 + -1][t6 + 1][-1] + _out[t4 + -1][t6 + -1][-1] + _out[t4 + 1][t6 + -1][-1];
        buffer_0[0] = treg_0 * (14 / 30) + treg_1 * (3 / 30) + treg_2 * (1 / 30);
        treg_0 = _out[t4][t6][0];
        treg_1 = _out[t4 + 1][t6][0] + _out[t4 + -1][t6][0] + _out[t4][t6 + 1][0] + _out[t4][t6 + -1][0];
        treg_2 = _out[t4 + 1][t6 + 1][0] + _out[t4 + -1][t6 + 1][0] + _out[t4 + -1][t6 + -1][0] + _out[t4 + 1][t6 + -1][0];
        buffer_1[0] = treg_0 * (-128 / 30) + treg_1 * (14 / 30) + treg_2 * (3 / 30);
        buffer_0[1] = treg_0 * (14 / 30) + treg_1 * (3 / 30) + treg_2 * (1 / 30);
        for (t8 = 0; t8 <= 31; t8 += 1) {
          treg_0 = _out[t4][t6][t8 + 1];
          treg_1 = _out[t4 + 1][t6][t8 + 1] + _out[t4 + -1][t6][t8 + 1] + _out[t4][t6 + 1][t8 + 1] + _out[t4][t6 + -1][t8 + 1];
          treg_2 = _out[t4 + 1][t6 + 1][t8 + 1] + _out[t4 + -1][t6 + 1][t8 + 1] + _out[t4 + -1][t6 + -1][t8 + 1] + _out[t4 + 1][t6 + -1][t8 + 1];
          buffer_2[t8 + 0 + 0] = treg_0 * (14 / 30) + treg_1 * (3 / 30) + treg_2 * (1 / 30);
          buffer_1[t8 + 1 + 0] = treg_0 * (-128 / 30) + treg_1 * (14 / 30) + treg_2 * (3 / 30);
          buffer_0[t8 + 2 + 0] = treg_0 * (14 / 30) + treg_1 * (3 / 30) + treg_2 * (1 / 30);
        }
        for (t8 = 0; t8 <= 31; t8 += 1) 
          _in[t4][t6][t8] = buffer_0[t8 + 0] + buffer_1[t8 + 0] + buffer_2[t8 + 0];
        for (t8 = 0; t8 <= 31; t8 += 1) 
          _in[t4][t6][t8] = x * _out[t4][t6][t8] - y * h2inv * _in[t4][t6][t8];
        for (t8 = 0; t8 <= 31; t8 += 1) 
          _in[t4][t6][t8] = _out[t4][t6][t8] - TwoThirds * _lambda[t4][t6][t8] * (_in[t4][t6][t8] - _rhs[t4][t6][t8]);
      }
}
