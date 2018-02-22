#include <vector>
#include <string.h>
#include <map>

#ifdef FRONTEND_ROSE 
#include "rose.h"

#include "mem_mapping_utils.hh"

using namespace SageBuilder;
using namespace SageInterface;

memory_mapping::memory_mapping (bool used, const char * array_name){
  this->mem_used = used;
  this->add(array_name);
}

texture_memory_mapping::texture_memory_mapping(bool used, const char* array_name) : memory_mapping(used, array_name) { }
constant_memory_mapping::constant_memory_mapping(bool used, const char* array_name) : memory_mapping(used, array_name) { }
//texture_memory_mapping::texture_memory_mapping (bool used, const char* array_name, int width, int height) {
//  tex_mem_used = used;
//  this->add(array_name, width, height);
//}

void memory_mapping::add(const char * array_name) {
  this->mapped_array_name.push_back(std::string(array_name));
  //std::vector<int> ivec = std::vector<int>();
  //dims[std::string(array_name)] = ivec;
}
//void texture_memory_mapping::add(const char* array_name, int width, int height) {
//  tex_mapped_array_name.push_back(std::string(array_name));
//  std::vector<int> ivec = std::vector<int>();
//  ivec.push_back(width);
//  ivec.push_back(height);
//  dims[std::string(array_name)] = ivec;
//}

bool memory_mapping::is_mem_used(){
  return this->mem_used;
}
bool memory_mapping::is_array_mapped(const char * array_name){
  
  for( int i=0; i<mapped_array_name.size(); i++){
    if(!(strcmp(array_name, mapped_array_name[i].c_str())))
      return true;
  }
  return false;
}
void memory_mapping::set_mapped_symbol(const char * array_name, SgVariableSymbol* sym) {
  this->mapped_symbol[std::string(array_name)] = sym;
}
void texture_memory_mapping::set_devptr_symbol(const char * array_name, SgVariableSymbol* sym) {
  devptr_symbol[std::string(array_name)] = sym;
}
void memory_mapping::set_vardef(const char* array_name, CudaIOVardef* vardef) {
  this->vardefs[std::string(array_name)] = vardef;
}
SgVarRefExp* memory_mapping::get_mapped_symbol_exp(const char * array_name) {
  return buildVarRefExp(this->mapped_symbol[std::string(array_name)]);
}
SgVarRefExp* texture_memory_mapping::get_devptr_symbol_exp(const char * array_name) {
  return buildVarRefExp(devptr_symbol[std::string(array_name)]);
}
CudaIOVardef* memory_mapping::get_vardef(const char* vardef_name) {
  return this->vardefs[std::string(vardef_name)];
}
//int texture_memory_mapping::get_dims(const char* array_name) {
//  return (int)(dims[std::string(array_name)].size());
//}
//int texture_memory_mapping::get_dim_length(const char* array_name, int dim) {
//  return dims[std::string(array_name)][dim];
//}
memory_mapping::memory_mapping() {
  mem_used = false;
}
texture_memory_mapping::texture_memory_mapping() : memory_mapping() { }
constant_memory_mapping::constant_memory_mapping() : memory_mapping() { }


#endif  // FRONTEND_ROSE 
