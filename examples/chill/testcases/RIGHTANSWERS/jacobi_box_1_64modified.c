// this source is derived from CHILL AST originally from file 'jacobi_box_1_64.c' as parsed by frontend compiler rose

//#include <stdint.h>

#define uint64_t unsigned int

#include "defines.h"

#include "box.h"

#include "mg.h"

#define SIZE 64

#define GHOSTS 1

#define PR_SIZE 64

//#define ALPHA (-128.0/30.0)

//#define BETA (14.0/30.0)

//#define GAMMA (3.0/30.0)

//#define DELTA (1.0/30.0)

#define PENCIL (SIZE + 2 * GHOSTS)

#define PLANE  ( PENCIL * PENCIL )

void smooth_box_1_64(domain_type *domain, int level, int box_id, int phi_id, int rhs_id, double x, double y, int sweep) {
  int t8;
  int t6;
  int t4;
  int i;
  int j;
  int k;
  double ALPHA = -128 / 30;
  double BETA = 14 / 30;
  double GAMMA = 3 / 30;
  double DELTA = 1 / 30;
  int s;
  s = sweep;
  double h2inv = 1 / (domain->h[level] * domain->h[level]);
  double TwoThirds = 2 / 3;
  double _in[64 + 2][64 + 2][64 + 2];
  double _out[64 + 2][64 + 2][64 + 2];
  double _rhs[64 + 2][64 + 2][64 + 2];
  double _lambda[64 + 2][64 + 2][64 + 2];
  i = (j = (k = 0));
  if (s % 2 == 0) 
    for (t4 = 0; t4 <= 63; t4 += 1) 
      for (t6 = 0; t6 <= 63; t6 += 1) 
        for (t8 = 0; t8 <= 63; t8 += 1) {
          _out[t4][t6][t8] = ALPHA * _in[t4][t6][t8] + BETA * (_in[t4 - 1][t6][t8] + _in[t4][t6 - 1][t8] + _in[t4][t6 + 1][t8] + _in[t4 + 1][t6][t8] + _in[t4][t6][t8 - 1] + ALPHA * _in[t4][t6][t8 + 1]) + GAMMA * (_in[t4 - 1][t6][t8 - 1] + _in[t4][t6 - 1][t8 - 1] + _in[t4][t6 + 1][t8 - 1] + _in[t4 + 1][t6][t8 - 1] + _in[t4 - 1][t6 - 1][t8] + _in[t4 - 1][t6 + 1][t8] + _in[t4 + 1][t6 - 1][t8] + _in[t4 + 1][t6 + 1][t8] + _in[t4 - 1][t6][t8 + 1] + _in[t4][t6 - 1][t8 + 1] + _in[t4][t6 + 1][t8 + 1] + _in[t4 + 1][t6][t8 + 1]) + DELTA * (_in[t4 - 1][t6 - 1][t8 - 1] + _in[t4 - 1][t6 + 1][t8 - 1] + _in[t4 + 1][t6 - 1][t8 - 1] + _in[t4 + 1][t6 + 1][t8 - 1] + _in[t4 - 1][t6 - 1][t8 + 1] + _in[t4 - 1][t6 + 1][t8 + 1] + _in[t4 + 1][t6 - 1][t8 + 1] + _in[t4 + 1][t6 + 1][t8 + 1]);
          _out[t4][t6][t8] = x * _in[t4][t6][t8] - y * h2inv * _out[t4][t6][t8];
          _out[t4][t6][t8] = _in[t4][t6][t8] - TwoThirds * _lambda[t4][t6][t8] * (_out[t4][t6][t8] - _rhs[t4][t6][t8]);
        }
  else 
    for (t4 = 0; t4 <= 63; t4 += 1) 
      for (t6 = 0; t6 <= 63; t6 += 1) 
        for (t8 = 0; t8 <= 63; t8 += 1) {
          _in[t4][t6][t8] = ALPHA * _out[t4][t6][t8] + BETA * (_out[t4 - 1][t6][t8] + _out[t4][t6 - 1][t8] + _out[t4][t6 + 1][t8] + _out[t4 + 1][t6][t8] + _out[t4][t6][t8 - 1] + _out[t4][t6][t8 + 1]) + GAMMA * (_out[t4 - 1][t6][t8 - 1] + _out[t4][t6 - 1][t8 - 1] + _out[t4][t6 + 1][t8 - 1] + _out[t4 + 1][t6][t8 - 1] + _out[t4 - 1][t6 - 1][t8] + _out[t4 - 1][t6 + 1][t8] + _out[t4 + 1][t6 - 1][t8] + _out[t4 + 1][t6 + 1][t8] + _out[t4 - 1][t6][t8 + 1] + _out[t4][t6 - 1][t8 + 1] + _out[t4][t6 + 1][t8 + 1] + _out[t4 + 1][t6][t8 + 1]) + DELTA * (_out[t4 - 1][t6 - 1][t8 - 1] + _out[t4 - 1][t6 + 1][t8 - 1] + _out[t4 + 1][t6 - 1][t8 - 1] + _out[t4 + 1][t6 + 1][t8 - 1] + _out[t4 - 1][t6 - 1][t8 + 1] + _out[t4 - 1][t6 + 1][t8 + 1] * DELTA + _out[t4 + 1][t6 - 1][t8 + 1] + _out[t4 + 1][t6 + 1][t8 + 1]);
          _in[t4][t6][t8] = x * _out[t4][t6][t8] - y * h2inv * _in[t4][t6][t8];
          _in[t4][t6][t8] = _out[t4][t6][t8] - TwoThirds * _lambda[t4][t6][t8] * (_in[t4][t6][t8] - _rhs[t4][t6][t8]);
        }
}
