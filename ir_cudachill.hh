#ifndef IR_CUDA_CHILL_HH
#define IR_CUDA_CHILL_HH

#include "chill_io.hh"
#include "ir_chill.hh"
#include "loop.hh"
#include "loop_cuda_chill.hh"



class IR_cudachillCode : public IR_chillCode{
  
public:
  IR_cudachillCode(const char *filename, const char* proc_name);
  
  std::string cudaFileToWrite;
  
  chillAST_node *globalSymbolTable;  // TODO a vector?
  
  // WHY would this be here ?     TODO  WHY vector of outputRepr and not ints?
  IR_ArraySymbol *CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size,int sharedAnnotation = 1);

  bool commit_loop(Loop *loop, int loop_num);

  ~IR_cudachillCode();
  
};


#endif

