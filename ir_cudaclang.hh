#ifndef IR_CUDA_CLANG_HH
#define IR_CUDA_CLANG_HH

#include "chill_io.hh"
#include "ir_clang.hh"
#include "loop.hh"



class IR_cudaclangCode : public IR_clangCode{
  
public:
  IR_cudaclangCode(const char *paramfilename, const char* proc_name, const char* dest_name = NULL);

  std::string cudaFileToWrite;
  
  chillAST_node *globalSymbolTable;  // TODO a vector?
  
  // now in ir_code ??  chillAST_FunctionDecl * func_defn;     // the function we're modifying  - also IR_clangcode chillfunc;
  
  IR_ArraySymbol *CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size,int sharedAnnotation = 1);

  bool commit_loop(Loop *loop, int loop_num);

  ~IR_cudaclangCode() {};
  
};


#endif

