#ifndef IR_CUDAROSE_HH
#define IR_CUDAROSE_HH

#include "chill_io.hh"
#include "loop.hh"
#include "parser.h"

#include "loop_cuda_chill.hh"

class IR_cudaChillCode : public IR_chillCode{
  
public:
  IR_cudaChillCode(chill::Parser *parser, const char *filename, const char* proc_name, const char* dest_name);
  
  std::string cudaFileToWrite;

  IR_ArraySymbol *CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size,int sharedAnnotation = 1);

  bool commit_loop(Loop *loop, int loop_num);

};


#endif

