


// this source derived from CHILL AST originally from file 'smooth_fused_64.c' as parsed by frontend compiler rose

typedef struct {
  struct { int i; int j; int k; } low;
  struct { int i; int j; int k; } dim;
  struct { int i; int j; int k; } dim_with_ghosts;
  int ghosts;
  int pencil;
  int plane;
  int volume;
  int bufsizes[27];
  double * __restrict__ surface_bufs[27];
  double * __restrict__ ghost_bufs[27];
  int numGrids;
  double ** __restrict__ grids;
} box_type;

void destroy_box( box_type *box ); // fwd decl

int create_box( box_type *box, int numGrids, int low_i, int low_j, int low_k, int dim_i, int dim_j, int dim_k, int ghosts ); // fwd decl

void main( box_type *box, int phi_id, int rhs_id, int temp_phi_id, double a, double b, double h, int sweep )
{
  int t8;
  int t6;
  int t4;
  int t2;
  int I;
  int J;
  int K;
  double h2inv;
  double * __restrict__ phi;
  double * __restrict__ rhs;
  double * __restrict__ alpha;
  double * __restrict__ beta_i;
  double * __restrict__ beta_j;
  double * __restrict__ beta_k;
  double * __restrict__ lambda;
  double * __restrict__ temp;
  double( *_phi[72][72];
  double( *_rhs[72][72];
  double( *_alpha[72][72];
  double( *_beta_i[72][72];
  double( *_beta_j[72][72];
  double( *_beta_k[72][72];
  double( *_lambda[72][72];
  double( *_temp[72][72];
  _phi = ((double(*)[72UL][72UL]) phi);
  _rhs = ((double(*)[72UL][72UL]) rhs);
  _alpha = ((double(*)[72UL][72UL]) alpha);
  _beta_i = ((double(*)[72UL][72UL]) beta_i);
  _beta_j = ((double(*)[72UL][72UL]) beta_j);
  _beta_k = ((double(*)[72UL][72UL]) beta_k);
  _lambda = ((double(*)[72UL][72UL]) lambda);
  _temp = ((double(*)[72UL][72UL]) temp);
  K = box -> dim.k;
  J = box -> dim.j;
  I = box -> dim.i;
  int color;
  color = sweep;
  for (t2 = 0; t2 <= 3; t2 += 1) {
    for (t4 = (t2 - 3); t4 <= (-t2 + 66); t4 += 1) {
      for (t6 = (t2 - 3); t6 <= (-t2 + 66); t6 += 1) {
        for (t8 = (t2 - 3); t8 <= (-t2 + 66); t8 += 1) 
          if (((((t8 + t6) + t4) + (color + t2)) % 2) == 0) {
            _temp[t4][t6][t8] = ((b * h2inv) * (((_beta_i[t4][t6][t8 + 1] * (_phi[t4][t6][t8 + 1] - _phi[t4][t6][t8]) - (_beta_i[t4][t6][t8] * (_phi[t4][t6][t8] - _phi[t4][t6][t8 - 1]))) + _beta_j[t4][t6 + 1][t8] * (_phi[t4][t6 + 1][t8] - _phi[t4][t6][t8]) - (_beta_j[t4][t6][t8] * (_phi[t4][t6][t8] - _phi[t4][t6 - 1][t8]))) + _beta_k[t4 + 1][t6][t8] * (_phi[t4 + 1][t6][t8] - _phi[t4][t6][t8]) - (_beta_k[t4][t6][t8] * (_phi[t4][t6][t8] - _phi[t4 - 1][t6][t8]))));
            _temp[t4][t6][t8] = ((a * _alpha[t4][t6][t8]) * _phi[t4][t6][t8] - _temp[t4][t6][t8]);
            _phi[t4][t6][t8] = (_phi[t4][t6][t8] - (_lambda[t4][t6][t8] * (_temp[t4][t6][t8] - _rhs[t4][t6][t8])));

          }      }
    }
  }

}
