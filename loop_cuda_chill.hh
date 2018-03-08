#ifndef LOOP_CUDA_CHILL_HH
#define LOOP_CUDA_CHILL_HH


// this should be used if Loop has chillAST internally 
// and not an AST from the front end compiler

#include "chill_ast.hh" 
#include "chill_io.hh"

#include "loop.hh"
//#include "mem_mapping_utils.hh"  // rose dependent
#include <string.h>


#include <code_gen/CG_chillRepr.h> 
#include <code_gen/CG_chillBuilder.h> 

#ifndef ENUMMEMORYMODE 
#define ENUMMEMORYMODE 
enum MemoryMode { GlobalMem, SharedMem, TexMem };
#endif 

#ifndef CUDAVARDEFS
#define CUDAVARDEFS
struct CudaIOVardef {
  std::string name;
  std::string secondName;
  char* type;

  chillAST_node *size_expr; //array size as an expression (can be a product of other variables etc)
  //omega::chillRepr *size_expr; // ?? 

  chillAST_VarDecl *vardecl; 
  chillAST_node *in_data;   //Variable of array to copy data in from (before kernel call)
  chillAST_node *out_data;  //Variable of array to copy data out to (after kernel call)
  chillAST_VarDecl *CPUside_param;  // name of CPU side parameter (see: in_data, out_data, when not NULL)

  std::vector<int> size_multi_dim; //-1 if linearized, the constant size N, of a NxN 2D array otherwise
  bool tex_mapped; //protonu-- true if this variable will be texture mapped, so no need to pass it as a argument
  bool cons_mapped;
  std::string original_name; //this is such a hack, to store the original name, to store a table to textures used

  CudaIOVardef() { // constructor
    //debug_fprintf(stderr, "constructing VarDef\n");
    vardecl = NULL;
    in_data = out_data = NULL;
    CPUside_param = 0;
    size_expr = NULL;
    type = NULL;
    tex_mapped = cons_mapped = false;
  }

  void   print() { 
    debug_fprintf(stderr, "Vardefs:\n");  //  0x%x\n", this); 
    debug_fprintf(stderr, "name %s\n", name.c_str()); 
    debug_fprintf(stderr, "second name %s\n", secondName.c_str()); 
    debug_fprintf(stderr, "original name %s\n", original_name.c_str()); 
    debug_fprintf(stderr, "type ");
    if (!type) debug_fprintf(stderr, "NULL)\n");
    else debug_fprintf(stderr, "%s\n", type); 
    debug_fprintf(stderr, "size ");
    size_expr->print(0, stderr);
    if ( vardecl ) debug_fprintf(stderr, "\nvardecl %p\n", vardecl); 
    else debug_fprintf(stderr, "\nvardecl NULL\n"); 

    //for (int i=0; i<size_multi_dim.size(); i++) { 
    //  if (i) debug_fprintf(stderr, "x");
    //  debug_fprintf(stderr, "%d", size_multi_dim[i]);
    //} 
    debug_fprintf(stderr, "\n");
    if (tex_mapped)  debug_fprintf(stderr, "tex  mapped\n");
    if (cons_mapped) debug_fprintf(stderr, "cons mapped\n");
  };
};
#endif


chillAST_VarDecl *addBuiltin( char *nameofbuiltin, char *typeOfBuiltin, chillAST_node *somecode); // fwd decl 

class LoopCuda: public Loop{  // chill version 
  
public:
  //std::vector<proc_sym*> new_procs; //Need adding to a fse
  std::vector< std::vector<std::string> > idxNames;
  std::vector< std::pair<int, std::string> > syncs;
  bool useIdxNames;
  std::vector<std::string> index;
  std::vector<std::set< int> > cudaized;

  //Anand: Adding a placeholder for variables that
  //will be passed as parameters to cuda kernel function
  //could be possibly modified by cudaize
  std::set<std::string> kernel_parameters;

  // typedef std::vector<chillAST_VarDecl *> symbolTable; 
  chillAST_SymbolTable *CPUparamSymtab;  // 
  chillAST_SymbolTable *CPUbodySymtab;  // 

  void printsyms() { 
    fflush(stdout); 

    printf("\nparameter_symtab has %d entries\n", CPUparamSymtab->size());
    printSymbolTable( CPUparamSymtab ); 
    printf("\n"); fflush(stdout); 

    printf("body_symtab has %d entries\n", CPUbodySymtab->size()); 
    printSymbolTable( CPUbodySymtab ); 
    printf("\n"); fflush(stdout); 
  }


  //protonu--inserting this here, Gabe's implementation had it 
  //the struct statment as nonSplitLevels
  std::vector<std::vector<int> > stmt_nonSplitLevels;
  
#ifdef INTERNALS_ROSE 
  texture_memory_mapping *texture; //protonu        depends on rose internals
  constant_memory_mapping *constant_mem;         // depends on rose 
#endif
  std::map<std::string, int> array_sizes;
  std::vector<std::map<std::string, int> >Varray_dims;
  omega::CG_outputRepr *setup_code;
  omega::CG_outputRepr *teardown_code;
  
  unsigned int code_gen_flags;
  enum CodeGenFlags {
    GenInit      = 0x00,
    GenCudaizeV2 = 0x02,
  };
  
  
  //varibles used by cudaize_codegen
  //block x, y sizes, N and num_red
  int cu_bx, cu_by;
  int cu_n, cu_num_reduce;
  //thread x, y, z
  int cu_tx, cu_ty, cu_tz;

  //Anand: Adding CG_outputRepr* representations of cu_bx, cu_by, cu_tx, cu_ty
  //and cu_tz for non constant loop bounds
  omega::CG_chillRepr *cu_bx_repr, *cu_by_repr, *cu_tx_repr, *cu_ty_repr, *cu_tz_repr;

  // currently, using int cu_[bt][xy] for constant, or CG_chillRepr *cu_[bt][xy]_repr for non-constant.
  // this leads to HORRIBLE code. 
  // tru using chillAST_node *[bt][xy]_ast instead
  //chillAST_node *bxAst, *byAst, *txAst, *tyAst;

  // Anand's cudaize needs vectors for cu_bx, cu_by, cu_tx, cu_ty, cu_tz
  std::vector<int> Vcu_bx, Vcu_by;
  std::vector<int> Vcu_tx, Vcu_ty, Vcu_tz;
  std::vector<omega::CG_outputRepr *> Vcu_bx_repr, Vcu_by_repr, Vcu_tx_repr, Vcu_ty_repr, Vcu_tz_repr;
  std::vector<chillAST_node *> VbxAst, VbyAst, VtxAst, VtyAst, VtzAst;



  //block statement and level
  int cu_block_stmt, cu_block_level;
  
  //Anand: adding map of blockids and threadids per statements that are cudaized
  std::map<int, std::vector<int> > block_and_thread_levels;



  
  //tile statements, and loop-levels (cudaize v1)
  std::vector< std::vector<int> > cu_thread_loop;
  std::vector<int> cu_thread_sync;
  MemoryMode cu_mode;
  
  std::string cu_nx_name, cu_ny_name, cu_kernel_name; // TODO remove 
  std::vector<std::string> Vcu_kernel_name; 

  int nonDummyLevel(int stmt, int level);
  bool symbolExists(std::string s);
  void addSync(int stmt, std::string idx);
  void printSyncs();
  void renameIndex(int stmt, std::string idx, std::string newName);
  bool validIndexes(int stmt, const std::vector<std::string>& idxs);
  omega::CG_outputRepr* extractCudaUB(int stmt_num, int level, int &outUpperBound, int &outLowerBound);
  
  void printCode(int stmt_num, int effort=3, bool actuallyPrint=true) const; 
  void printRuntimeInfo() const;
  void printIndexes() const;
  chillAST_node* getCode(int effort=3) const;    
  chillAST_node* getCode(int effort, std::set<int> stmts) const;

  void printIS();
  
  
  void permute_cuda(int stmt, const std::vector<std::string>& curOrder);
  //protonu-writing a wrapper for the Chun's new permute function
  bool permute(int stmt_num, const std::vector<int> &pi);
  //end--protonu.
	void tile_cuda(int stmt, int level, int outer_level, TilingMethodType method = CountedTile);
  //void tile_cuda(int stmt, int level, int outer_level);
  void tile_cuda(int level, int tile_size, int outer_level, std::string idxName, std::string ctrlName, TilingMethodType method=StridedTile);

  void tile_cuda(int stmt, int level, int tile_size, int outer_level, std::string idxName, std::string ctrlName, TilingMethodType method=StridedTile);
  bool datacopy_privatized_cuda(int stmt_num, int level, const std::string &array_name, const std::vector<int> &privatized_levels, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 1, bool cuda_shared=false);
  bool datacopy_cuda(int stmt_num, int level, const std::string &array_name, std::vector<std::string> new_idxs, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 4, bool cuda_shared=false);
  bool unroll_cuda(int stmt_num, int level, int unroll_amount);


  void flatten_cuda(int stmt_num, std::string idxs, std::vector<int> &loop_levels, std::string inspector_name);
  void ELLify_cuda(int stmt_num, std::vector<std::string> arrays_to_pad, int pad_to,bool dense_pad, std::string pos_array_name);

  void distribute_cuda(std::vector<int> &stmt_nums, int loop_level);
  void fuse_cuda(std::vector<int> &stmt_nums, int loop_level);
  void peel_cuda(int stmt_num, int level, int amount);
  void shift_to_cuda(int stmt_num, int level, int absolute_position);
  void scalar_expand_cuda(int stmt_num, std::vector<int> level, std::string arrName, int memory_type =0, int padding =0,int assign_then_accumulate = 1);
  void split_with_alignment_cuda(int stmt_num, int level, int alignment, int direction=0);

  void compact_cuda(int stmt_num, int level, std::string new_array, int zero,
        std::string data_array);
  void make_dense_cuda(int stmt_num, int loop_level, std::string new_loop_index);
  void addKnown_cuda(std::string var, int value);
  void skew_cuda(std::vector<int> stmt_num,int level, std::vector<int> coefs);
  void reduce_cuda(int stmt_num, std::vector<int> level, int param, std::string func_name,  std::vector<int> seq_level, int bound_level=-1);




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

  /**
   * @param stmt_num The statement to cudaize
   * @param kernel_name Name of the generated kernel function
   * @param array_dims
   * @param blockIdxs Block index names
   * @param threadIdxs Thread index names
   * @param kernel_params
   */
  bool cudaize_v3(int stmt_num,    // 5 args, starting with stmt number
                  std::string kernel_name,
                  std::map<std::string, int> array_dims,
                  std::vector<std::string> blockIdxs,
                  std::vector<std::string> threadIdxs,
                  std::vector<std::string> kernel_params);
  chillAST_FunctionDecl *function_that_contains_this_loop; 
  chillAST_node* cudaize_codegen_v2(); 
  chillAST_node* codegen();            


  //protonu--have to add the constructors for the new class
  //and maybe destructors (?)
  LoopCuda();
  //LoopCuda(IR_Code *ir, tree_for *tf, global_symtab* gsym);
  LoopCuda(IR_Control *ir_c, int loop_num); 
  ~LoopCuda();
  
};


#endif
