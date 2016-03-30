

// so far, just gloabls that USED to be in all the loop_cuda_xxx.cc files

char *k_cuda_texture_memory; //protonu--added to track texture memory type
char *k_ocg_comment;

bool cudaDebug = true;

//class CudaStaticInit {
//public:
//  CudaStaticInit() {
//    cudaDebug = 1; //Change this to 1 for debug
//  }
//};
//static CudaStaticInit junkInitInstance__;

