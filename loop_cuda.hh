#ifndef LOOP_CUDA_HH
#define LOOP_CUDA_HH

#include "loop.hh"
#include <string.h>
#include <suif1.h>


enum MemoryMode { GlobalMem, SharedMem, TexMem };

//protonu --class introduced to hold texture memory information in one single place
//this might help me get over the weird memory issues I am having with the Loop class
//where someone/something corrupts my memory
class texture_memory_mapping{
private:
  bool tex_mem_used;
  std::vector< std::string > tex_mapped_array_name;
public:
  texture_memory_mapping ( bool used, const char * array_name){
    tex_mem_used = used;
    tex_mapped_array_name.push_back(std::string(array_name));
  }
  
  void add(const char * array_name) {
    tex_mapped_array_name.push_back(std::string(array_name));
  }
  
  bool is_tex_mem_used()    {return tex_mem_used;}
  bool is_array_tex_mapped(const char * array_name){
    
    for( int i=0; i<tex_mapped_array_name.size(); i++){
      if(!(strcmp(array_name, tex_mapped_array_name[i].c_str())))
        return true;
    }
    return false;
  }
  texture_memory_mapping()  {tex_mem_used = false;}
};

//protonu --class introduced to hold constant memory information in one single place
//this might help me get over the weird memory issues I am having with the Loop class
//where someone/something corrupts my memory
class constant_memory_mapping{
private:
  bool cons_mem_used;
  std::vector< std::string > cons_mapped_array_name;
public:
  constant_memory_mapping ( bool used, const char * array_name){
    cons_mem_used = used;
    cons_mapped_array_name.push_back(std::string(array_name));
  }
  
  void add(const char * array_name) {
    cons_mapped_array_name.push_back(std::string(array_name));
  }
  
  bool is_cons_mem_used()   {return cons_mem_used;}
  bool is_array_cons_mapped(const char * array_name){
    
    for( int i=0; i<cons_mapped_array_name.size(); i++){
      if(!(strcmp(array_name, cons_mapped_array_name[i].c_str())))
        return true;
    }
    return false;
  }
  constant_memory_mapping() {cons_mem_used = false;}
};


class LoopCuda: public Loop{
  
public:
  std::vector<proc_sym*> new_procs; //Need adding to a fse
  std::vector< std::vector<std::string> > idxNames;
  std::vector< std::pair<int, std::string> > syncs;
  bool useIdxNames;
  std::vector<std::string> index;
  proc_symtab *symtab;
  global_symtab *globals;
  
  //protonu--inserting this here, Gabe's implementation had it 
  //the struct statment as nonSplitLevels
  std::vector<omega::Tuple<int> > stmt_nonSplitLevels;
  
  texture_memory_mapping *texture; //protonu
  constant_memory_mapping *constant_mem; //protonu
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
  void extractCudaUB(int stmt_num, int level, int &outUpperBound, int &outLowerBound);
  
  void printCode(int effort=1, bool actuallyPrint=true) const; 
  void printRuntimeInfo() const;
  void printIndexes() const;
  tree_node_list* getCode(int effort = 1) const;
  
  
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
  //protonu--using constant memory
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
  tree_node_list* cudaize_codegen_v2();
  tree_node_list* codegen();
  
  //protonu--have to add the constructors for the new class
  //and maybe destructors (?)
  LoopCuda();
  //LoopCuda(IR_Code *ir, tree_for *tf, global_symtab* gsym);
  LoopCuda(IR_Control *ir_c, int loop_num);//protonu-added so as to not change ir_suif
  ~LoopCuda();
  
};

#endif
