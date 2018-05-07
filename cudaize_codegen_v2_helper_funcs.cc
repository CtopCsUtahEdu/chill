#include "loop_cuda_chill.hh"

//#define TRANSFORMATION_FILE_INFO Sg_File_Info::generateDefaultFileInfoForTransformationNode()
#include <code_gen/CG_stringBuilder.h>

#include <omega/code_gen/include/codegen.h>
#include <code_gen/CG_utils.h>
#include <code_gen/CG_outputRepr.h>
#include "loop.hh"
#include <math.h>

#include "omegatools.hh"

#include "ir_cudachill.hh"  // cudachill?   TODO
//#include "ir_clang.hh"   // includes all the "translate from clang to chill", so needs clang paths. bad.

#include "chill_error.hh"
#include <vector>
#include <strings.h>

enum class io_dir {
  write_only        = 1,
  read_only         = 2,
  read_write        = write_only | read_only,
};

static void get_io_array_refs(
    const std::map<std::string, int>& array_sizes,
    const std::vector<IR_chillArrayRef*>& refs,
    io_dir dir,
    std::vector<CudaIOVardef>& arrayVars) noexcept {
  for(int i = 0; i < refs.size(); i++) {
    auto vref   = refs[i];
    auto vname  = vref->name();

    //TODO: maybe exclude non-parameter array refs
    auto param = vref->chillASE->multibase();
    if(!param->isParmVarDecl()) {
      continue;
    }

    CudaIOVardef v;
    // set parameter name
    v.original_name = vname;
    // set devptr name
    switch(dir) {
    case io_dir::write_only:
      v.name = std::string("dev") + "WO" + std::to_string(i) + "ptr";
      break;
    case io_dir::read_only:
      v.name = std::string("dev") + "RO" + std::to_string(i) + "ptr";
      break;
    case io_dir::read_write:
      v.name = std::string("dev") + "RW" + std::to_string(i) + "ptr";
      break;
    }

    v.tex_mapped  = false;
    v.cons_mapped = false;
    v.type        = strdup(param->underlyingtype);


    // set size expression
    // -------------------
    chillAST_node* so = new chillAST_Sizeof(v.type);
    int numelements = 1;

    // if the array size was set by a call to cudaize
    if(array_sizes.find(vname) != array_sizes.end()) {
      numelements = array_sizes.at(vname);
    }
    else {
      for(int idx = 0; idx < param->numdimensions; idx++) {
        numelements *= param->getArraySizeAsInt(idx);
      }
    }

    chillAST_IntegerLiteral* numofthings = new chillAST_IntegerLiteral(numelements);
    v.size_expr = new chillAST_BinaryOperator(numofthings, "*", so);

    v.CPUside_param = param;
    switch(dir) {
    case io_dir::write_only:
      v.in_data       = nullptr;
      v.out_data      = param;
      break;
    case io_dir::read_only:
      v.in_data       = param;
      v.out_data      = nullptr;
      break;
    case io_dir::read_write:
      v.in_data       = param;
      v.out_data      = param;
      break;
    }

    arrayVars.push_back(v);
  }
}

static void array_refs_byname(
    std::map<std::string, std::vector<chillAST_ArraySubscriptExpr*>>&   m,
    const std::vector<chillAST_ArraySubscriptExpr*>&                    refs) noexcept {
  for(auto ref: refs) {
    //TODO: check that ref comes directly from a variable declaration
    auto vname = std::string(ref->multibase()->varname);
    m[vname].push_back(ref);
  }
}


void get_io_refs(
    const IR_Code*                                    ir,
    const std::map<std::string, int>&                 array_dims,
    const std::vector<chillAST_ArraySubscriptExpr*>&  refs,
    std::vector<CudaIOVardef>&                        arrayVars) noexcept {

  std::map<std::string, std::vector<chillAST_ArraySubscriptExpr*>>  refsbyname;
  array_refs_byname(refsbyname, refs);

  //std::vector<IR_chillArrayRef*> ir_refs;
  std::vector<IR_chillArrayRef*> ro_refs;
  std::vector<IR_chillArrayRef*> wo_refs;
  std::vector<IR_chillArrayRef*> rw_refs;
  for(auto p: refsbyname) {
    auto&               asevec    = p.second;
    auto&               name      = p.first;
    IR_chillArrayRef*   ir_ref    = new IR_chillArrayRef(ir, asevec[0], name.c_str(), false);
    bool                is_read   = false;
    bool                is_write  = false;
    for(auto ase: asevec) {
      if(ase->imwrittento) {
        is_write = true;
        ir_ref->iswrite = true;
        if(ase->imreadfrom) {
          is_read = true;
        }
      }
      else if(ase->imreadfrom) {
        assert(false && "This should not happen");
      }
      else {
        is_read = true;
      }
    }

    if(is_read && is_write) {
      rw_refs.push_back(ir_ref);
    }
    else if(is_read) {
      ro_refs.push_back(ir_ref);
    }
    else {
      wo_refs.push_back(ir_ref);
    }
  }

  get_io_array_refs(array_dims, rw_refs, io_dir::read_write, arrayVars);
  get_io_array_refs(array_dims, ro_refs, io_dir::read_only,  arrayVars);
  get_io_array_refs(array_dims, wo_refs, io_dir::write_only, arrayVars);

}
