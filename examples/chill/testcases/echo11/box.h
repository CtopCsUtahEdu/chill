#ifndef _BOX_H
#define _BOX_H
//------------------------------------------------------------------------------------------------------------------------------
typedef struct {
    int i, j, k;
} ijk;

typedef struct {
//  ijk low;               // global coordinates of the first (non-ghost) element of subdomain
//  ijk dim;               // dimensions of this box's core (not counting ghost zone)
//  ijk dim_with_ghosts;   // dimensions of this box's core (not counting ghost zone)
  struct {int i,j,k;} low;
  struct {int i,j,k;} dim;
  struct {int i,j,k;} dim_with_ghosts;
  
  int ghosts;                             // ghost zone depth
  int pencil,plane,volume;                // useful for offsets
  int                       bufsizes[27]; // = sizes of extracted surfaces and ghost zones (pointer to array of 27 elements)
  //Protonu--not sure how Rose/CHiLL will handle __restrict__ type modifier
  //might have to take it out.
  double * __restrict__ surface_bufs[27]; // = extracted surface (rhs on the way down, correction on the way up)
  double * __restrict__   ghost_bufs[27]; // = incoming ghost zone (rhs on the way down, correction on the way up)
  int numGrids;
  double ** __restrict__ grids;
} box_type;
//------------------------------------------------------------------------------------------------------------------------------
void destroy_box(box_type *box);
 int create_box(box_type *box, int numGrids, int low_i, int low_j, int low_k, int dim_i, int dim_j, int dim_k, int ghosts);

#endif
