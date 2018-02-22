#ifndef MEM_MAPPING_UTILS_HH
#define MEM_MAPPING_UTILS_HH

#include <vector>
#include <string.h>
#include <map>

#ifdef FRONTEND_ROSE 
#include "rose.h"
#endif 

using namespace SageInterface;
using namespace SageBuilder;

struct CudaIOVardef;

class memory_mapping {
private:
  bool mem_used;
  std::vector< std::string > mapped_array_name;
  std::map<std::string, SgVariableSymbol*> mapped_symbol;  // tied to Rose
  std::map<std::string, CudaIOVardef*> vardefs;
public:
  memory_mapping();
  memory_mapping(bool used, const char* array_name);
  void add(const char* array_name);
  bool is_mem_used();
  bool is_array_mapped(const char* array_name);
  void set_mapped_symbol(const char* array_name, SgVariableSymbol* sym);  // tied to Rose
  void set_vardef(const char* array_name, CudaIOVardef* vardef);
  SgVarRefExp* get_mapped_symbol_exp(const char* array_name);  // tied to Rose
  CudaIOVardef* get_vardef(const char* vardef_name);
};

//protonu --class introduced to hold texture memory information in one single place
//this might help me get over the weird memory issues I am having with the Loop class
//where someone/something corrupts my memory

class texture_memory_mapping : public memory_mapping {
private:
  std::map<std::string, SgVariableSymbol*> devptr_symbol;  // tied to Rose
  // a hack for multi-dimensional texture mapping
  //std::map<std::string, std::vector<int> > dims;
public:
  texture_memory_mapping ( bool used, const char * array_name);
  //texture_memory_mapping (bool used, const char* array_name, int width, int height);
  // this function is a hack to get arround a bug
  // void add(const char* array_name, int width, int height);
  void set_devptr_symbol(const char * array_name, SgVariableSymbol* sym);  // tied to Rose
  SgVarRefExp* get_devptr_symbol_exp(const char * array_name);  // tied to Rose
  //int get_dim_length(const char* array_name, int dim);
  //int get_dims(const char* array_name);
  texture_memory_mapping();
};

class constant_memory_mapping : public memory_mapping {
public:
  constant_memory_mapping();
  constant_memory_mapping(bool used, const char* array_name);
};

#endif
