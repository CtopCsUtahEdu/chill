#ifndef IR_CUDA_ROSE
#define IR_CUDA_ROSE

#include <code_gen/CG_roseRepr.h>
#include <code_gen/CG_roseBuilder.h>
#include "ir_rose.hh"
#include "loop.hh"
#include "loop_cuda_rose.hh"
#include "ir_rose_utils.hh"



class IR_cudaroseCode : public IR_roseCode{
  
public:
  
  
  IR_cudaroseCode(const char *filename, const char* proc_name);
  
  
  
  SgGlobal *gsym_;
  SgScopeStatement* defn;      
  SgGlobal* first_scope;
  SgSymbolTable* parameter;
  SgSymbolTable* body;
  SgFunctionDefinition* func_defn;    
  std::vector<SgSymbolTable*> write_procs;//procs to write  
  
  
  IR_ArraySymbol *CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size,int sharedAnnotation = 1);
  omega::CG_outputRepr* init_code(){ return init_code_; }
  bool commit_loop(Loop *loop, int loop_num);
  std::vector<SgForStatement *> get_loops()
  { 
    std::vector<SgForStatement *> loops = find_loops(func->get_definition()->get_body()); 
    return loops;
  }
  
  ~IR_cudaroseCode();
  
};


#endif

