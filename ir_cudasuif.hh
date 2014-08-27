#ifndef IR_CUDA_SUIF
#define IR_CUDA_SUIF

#include <code_gen/CG_suifRepr.h>
#include <code_gen/CG_suifBuilder.h>
#include "ir_suif.hh"
#include "loop.hh"
#include "loop_cuda.hh"
#include "ir_suif_utils.hh"



class IR_cudasuifCode : public IR_suifCode{
  
public:
  global_symtab *gsym_;
  std::vector<proc_sym*> write_procs;//procs to write  
  
  
  IR_cudasuifCode(const char *filename, int proc_num);
  IR_ArraySymbol *CreateArraySymbol(const IR_Symbol *sym, 
                                    std::vector<omega::CG_outputRepr *> &size,
                                    int sharedAnnotation = 1);
  omega::CG_outputRepr* init_code(){ return init_code_; }
  bool commit_loop(Loop *loop, int loop_num);
  std::vector<tree_for *> get_loops()
  {
    std::vector<tree_for *> loops = find_loops(psym_->block()->body());
    return loops;
  }
  ~IR_cudasuifCode();
  
};


#endif
