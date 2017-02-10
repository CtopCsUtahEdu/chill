// Helmholtz ~ Laplacian() = a*alpha*Identity - b*Divergence*beta*Gradient
// GSRB = phi_red = phi_red + lambda(Laplacian(phi_black) - RHS)
#define  __u       0 // = what we're eventually solving for (u), cell centered
#define  __f       1 // = original right-hand side (Au=f), cell centered

#define  __alpha   2 // cell centered constant
#define  __beta_i  3 // face constant (n.b. element 0 is the left face of the ghost zone element)
#define  __beta_j  4 // face constant (n.b. element 0 is the back face of the ghost zone element)
#define  __beta_k  5 // face constant (n.b. element 0 is the bottom face of the ghost zone element)
#define  __lambda  6 // cell centered constant
#define  __ee     7 // = used for correction (ee) in residual correction form, cell centered
#define  __r      8 // = used for initial right-hand side (f-Av) in residual correction form, cell centered
#define  __temp   9 // = used for unrestricted residual (r), cell centered
#define  __temp_phi   10 // = used for unrestricted residual (r), cell centered
#define  __temp_res   11 // = used for unrestricted residual (r), cell centered

//------------------------------------------------------------------------------------------------------------------------------
// box[j].ghost[i] = box[box[j].neighbor[i]].surface[26-i]
//------------------------------------------------------------------------------------------------------------------------------

