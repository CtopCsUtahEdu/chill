#ifndef LOOP_CUDA_HH
#define LOOP_CUDA_HH

#include "loop.hh"
#include "mem_mapping_utils.hh"
#include <string.h>
#include "rose.h"
//#include <suif1.h>

using namespace omega;
using namespace SageBuilder;
enum MemoryMode { GlobalMem, SharedMem, TexMem };

struct VarDefs {
  std::string name;
  std::string secondName;
  SgExpression* size_expr; //array size as an expression (can be a product of other variables etc)
  SgType* type;
  SgVariableSymbol* in_data; //Variable of array to copy data in from (before kernel call)
  SgVariableSymbol* out_data; //Variable of array to copy data out to (after kernel call)
  std::vector<int> size_multi_dim; //-1 if linearized, the constant size N, of a NxN 2D array otherwise
  bool tex_mapped; //protonu-- true if this variable will be texture mapped, so no need to pass it as a argument
  bool cons_mapped;
  std::string original_name; //this is such a hack, to store the original name, to store a table to textures used
};


class LoopCuda: public Loop{
  
public:
  //std::vector<proc_sym*> new_procs; //Need adding to a fse
  std::vector< std::vector<std::string> > idxNames;
  std::vector< std::pair<int, std::string> > syncs;
  bool useIdxNames;
  std::vector<std::string> index;
  
  SgSymbolTable* symtab;
  SgSymbolTable* parameter_symtab;
  SgSymbolTable* body_symtab;
  SgGlobal* globals;
  SgGlobal* globalScope;  
  SgScopeStatement* func_body;     
  SgFunctionDefinition* func_definition;   
  //protonu--inserting this here, Gabe's implementation had it 
  //the struct statment as nonSplitLevels
  std::vector<std::vector<int> > stmt_nonSplitLevels;
  
  texture_memory_mapping *texture; //protonu
  constant_memory_mapping *constant_mem;
  std::map<std::string, int> array_dims;
  omega::CG_outputRepr *setup_code;
  omega::CG_outputRepr *teardown_code;
  
  unsigned int code_gen_flags;
  enum CodeGenFlags {
    GenInit      = 0x00,
    GenCudaizeV2 = 0x02,
  };
  
  
  //varibles used by cudaize_codegen
  //block x, y sizes, N and num_red
  int cu_bx, cu_by, cu_n, cu_num_reduce;
  //block statement and level
  int cu_block_stmt, cu_block_level;
  //thread x, y, z
  int cu_tx, cu_ty, cu_tz;
  
  //Anand: Adding CG_outputRepr* representations of cu_bx, cu_by, cu_tx, cu_ty
  //and cu_tz for non constant loop bounds
  
  CG_outputRepr *cu_bx_repr, *cu_by_repr, *cu_tx_repr, *cu_ty_repr, *cu_tz_repr;
  
  //tile statements, and loop-levels (cudaize v1)
  std::vector< std::vector<int> > cu_thread_loop;
  std::vector<int> cu_thread_sync;
  MemoryMode cu_mode;
  
  std::string cu_nx_name, cu_ny_name, cu_kernel_name;
  int nonDummyLevel(int stmt, int level);
  bool symbolExists(std::string s);
  void addSync(int stmt, std::string idx);
  void renameIndex(int stmt, std::string idx, std::string newName);
  bool validIndexes(int stmt, const std::vector<std::string>& idxs);
  CG_outputRepr* extractCudaUB(int stmt_num, int level, int &outUpperBound, int &outLowerBound);
  
  void printCode(int effort=1, bool actuallyPrint=true) const; 
  void printRuntimeInfo() const;
  void printIndexes() const;
  SgNode* getCode(int effort = 1) const;
  void printIS();
  
  
  void permute_cuda(int stmt, const std::vector<std::string>& curOrder);
  //protonu-writing a wrapper for the Chun's new permute function
  bool permute(int stmt_num, const std::vector<int> &pi);
  //end--protonu.
  void tile_cuda(int stmt, int level, int outer_level);
  void tile_cuda(int level, int tile_size, int outer_level, std::string idxName, std::string ctrlName, TilingMethodType method=StridedTile);
  void tile_cuda(int stmt, int level, int tile_size, int outer_level, std::string idxName, std::string ctrlName, TilingMethodType method=StridedTile);
  bool datacopy_privatized_cuda(int stmt_num, int level, const std::string &array_name, const std::vector<int> &privatized_levels, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 1, bool cuda_shared=false);
  bool datacopy_cuda(int stmt_num, int level, const std::string &array_name, std::vector<std::string> new_idxs, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 4, bool cuda_shared=false);
  bool unroll_cuda(int stmt_num, int level, int unroll_amount);
  //protonu--using texture memory
  void copy_to_texture(const char *array_name);
  void copy_to_constant(const char *array_name);
  int findCurLevel(int stmt, std::string idx);
  /**
   *
   * @param kernel_name Name of the GPU generated kernel
   * @param nx Iteration space over the x dimention
   * @param ny Iteration space over the y dimention
   * @param tx Tile dimention over x dimention
   * @param ty Tile dimention over the y dimention
   * @param num_reduce The number of dimentions to reduce by mapping to the GPU implicit blocks/threads
   */
  //stmnt_num is referenced from the perspective of being inside the cudize block loops
  bool cudaize_v2(std::string kernel_name, std::map<std::string, int> array_dims,
                  std::vector<std::string> blockIdxs, std::vector<std::string> threadIdxs);
  SgNode* cudaize_codegen_v2();
  SgNode* codegen();
  
  //protonu--have to add the constructors for the new class
  //and maybe destructors (?)
  LoopCuda();
  //LoopCuda(IR_Code *ir, tree_for *tf, global_symtab* gsym);
  LoopCuda(IR_Control *ir_c, int loop_num);//protonu-added so as to not change ir_suif
  ~LoopCuda();
  
};

#endif
