#ifndef IR_ROSE_HH
#define IR_ROSE_HH

#include <omega.h>
#include "ir_code.hh"
#include "ir_rose_utils.hh"
#include <AstInterface_ROSE.h>
#include "chill_error.hh"
#include "staticSingleAssignment.h"
#include "VariableRenaming.h"
#include "ssaUnfilteredCfg.h"
#include "virtualCFG.h"
#include <omega.h>

struct IR_roseScalarSymbol: public IR_ScalarSymbol {
  SgVariableSymbol* vs_;
  
  IR_roseScalarSymbol(const IR_Code *ir, SgVariableSymbol *vs) {
    ir_ = ir;
    vs_ = vs;
  }
  
  std::string name() const;
  int size() const;
  bool operator==(const IR_Symbol &that) const;
  IR_Symbol *clone() const;
};

struct IR_roseArraySymbol: public IR_ArraySymbol {
  
  SgVariableSymbol* vs_;
  
  IR_roseArraySymbol(const IR_Code *ir, SgVariableSymbol* vs) {
    ir_ = ir;
    vs_ = vs;
  }
  std::string name() const;
  int elem_size() const;
  int n_dim() const;
  omega::CG_outputRepr *size(int dim) const;
  bool operator==(const IR_Symbol &that) const;
  IR_ARRAY_LAYOUT_TYPE layout_type() const;
  IR_Symbol *clone() const;
  
};

struct IR_roseConstantRef: public IR_ConstantRef {
  union {
    omega::coef_t i_;
    double f_;
  };
  
  IR_roseConstantRef(const IR_Code *ir, omega::coef_t i) {
    ir_ = ir;
    type_ = IR_CONSTANT_INT;
    i_ = i;
  }
  IR_roseConstantRef(const IR_Code *ir, double f) {
    ir_ = ir;
    type_ = IR_CONSTANT_FLOAT;
    f_ = f;
  }
  omega::coef_t integer() const {
    assert(is_integer());
    return i_;
  }
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
  
};

struct IR_roseScalarRef: public IR_ScalarRef {
  SgAssignOp *ins_pos_;
  int op_pos_; // -1 means destination operand, otherwise source operand
  SgVarRefExp *vs_;
  int is_write_;
  IR_roseScalarRef(const IR_Code *ir, SgVarRefExp *sym) {
    ir_ = ir;
    ins_pos_ = isSgAssignOp(sym->get_parent());
    op_pos_ = 0;
    if (ins_pos_ != NULL)
      if (sym == isSgVarRefExp(ins_pos_->get_lhs_operand()))
        op_pos_ = -1;
    
    vs_ = sym;
  }
  IR_roseScalarRef(const IR_Code *ir, SgVarRefExp *ins, int pos) {
    ir_ = ir;
    /*  ins_pos_ = ins;
        op_pos_ = pos;
        SgExpression* op;
        if (pos == -1)
        op = ins->get_lhs_operand();
        else
        op = ins->get_rhs_operand();
        
    */
    
    is_write_ = pos;
    
    /*  if (vs_ == NULL || pos > 0)
        throw ir_error(
        "Src operand not a variable or more than one src operand!!");
    */
    
    vs_ = ins;
    
  }
  bool is_write() const;
  IR_ScalarSymbol *symbol() const;
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
};

struct IR_roseArrayRef: public IR_ArrayRef {
  
  SgPntrArrRefExp *ia_;
  
  int is_write_;
  IR_roseArrayRef(const IR_Code *ir, SgPntrArrRefExp *ia, int write) {
    ir_ = ir;
    ia_ = ia;
    is_write_ = write;
  }
  bool is_write() const;
  omega::CG_outputRepr *index(int dim) const;
  IR_ArraySymbol *symbol() const;
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
};

struct IR_roseLoop: public IR_Loop {
  SgNode *tf_;
  
  IR_roseLoop(const IR_Code *ir, SgNode *tf) {
    ir_ = ir;
    tf_ = tf;
  }
  
  IR_ScalarSymbol *index() const;
  omega::CG_outputRepr *lower_bound() const;
  omega::CG_outputRepr *upper_bound() const;
  IR_CONDITION_TYPE stop_cond() const;
  IR_Block *body() const;
  IR_Block *convert();
  int step_size() const;
  IR_Control *clone() const;
};

struct IR_roseBlock: public IR_Block {
  SgNode* tnl_;
  SgNode *start_, *end_;
  
  IR_roseBlock(const IR_Code *ir, SgNode *tnl, SgNode *start, SgNode *end) {
    ir_ = ir;
    tnl_ = tnl;
    start_ = start;
    end_ = end;
  }
  
  IR_roseBlock(const IR_Code *ir, SgNode *tnl) {
    ir_ = ir;
    tnl_ = tnl;
    start_ = tnl_->get_traversalSuccessorByIndex(0);
    end_ = tnl_->get_traversalSuccessorByIndex(
      (tnl_->get_numberOfTraversalSuccessors()) - 1);
  }
  omega::CG_outputRepr *extract() const;
  omega::CG_outputRepr *original() const;
  IR_Control *clone() const;
};

struct IR_roseIf: public IR_If {
  SgNode *ti_;
  
  IR_roseIf(const IR_Code *ir, SgNode *ti) {
    ir_ = ir;
    ti_ = ti;
  }
  ~IR_roseIf() {
  }
  omega::CG_outputRepr *condition() const;
  IR_Block *then_body() const;
  IR_Block *else_body() const;
  IR_Block *convert();
  IR_Control *clone() const;
};

class IR_roseCode_Global_Init {
private:
  static IR_roseCode_Global_Init *pinstance;
public:
  SgProject* project;
  static IR_roseCode_Global_Init *Instance(char** argv);
};

class IR_roseCode: public IR_Code {
protected:
  SgSourceFile* file;
  SgGlobal *root;
  SgGlobal *firstScope;
  SgSymbolTable* symtab_;
  SgSymbolTable* symtab2_;
  SgSymbolTable* symtab3_;
  SgDeclarationStatementPtrList::iterator p;
  SgFunctionDeclaration *func;
  bool is_fortran_;
  int i_;
  StaticSingleAssignment *ssa_for_scalar;
  ssa_unfiltered_cfg::SSA_UnfilteredCfg *main_ssa;
  VariableRenaming *varRenaming_for_scalar;
public:
  IR_roseCode(const char *filename, const char* proc_name);
  ~IR_roseCode();
  
  IR_ScalarSymbol *CreateScalarSymbol(const IR_Symbol *sym, int memory_type =
                                      0);
  IR_ArraySymbol *CreateArraySymbol(const IR_Symbol *sym,
                                    std::vector<omega::CG_outputRepr *> &size, int memory_type = 0);
  IR_ScalarRef *CreateScalarRef(const IR_ScalarSymbol *sym);
  IR_ArrayRef *CreateArrayRef(const IR_ArraySymbol *sym,
                              std::vector<omega::CG_outputRepr *> &index);
  int ArrayIndexStartAt() {
    if (is_fortran_)
      return 1;
    else
      return 0;
  }
  
  void populateLists(SgNode* tnl_1, SgStatementPtrList* list_1,
                     SgStatementPtrList& output_list_1);
  void populateScalars(const omega::CG_outputRepr *repr1,
                       std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
                       std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
                       std::set<std::string> &indices, std::vector<std::string> &index);
  //        std::set<std::string> &def_vars);
  /*void findDefinitions(SgStatementPtrList &list_1,
    std::set<VirtualCFG::CFGNode> &reaching_defs_1,
    std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
    std::set<std::string> &def_vars);
  */
  /*    void checkDependency(SgStatementPtrList &output_list_1,
        std::vector<DependenceVector> &dvs1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
        std::vector<std::string> &index, int i, int j);
        void checkSelfDependency(SgStatementPtrList &output_list_1,
        std::vector<DependenceVector> &dvs1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
        std::vector<std::string> &index, int i, int j);
        void checkWriteDependency(SgStatementPtrList &output_list_1,
        std::vector<DependenceVector> &dvs1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
        std::vector<std::string> &index, int i, int j);
  */
  std::vector<IR_ArrayRef *> FindArrayRef(
    const omega::CG_outputRepr *repr) const;
  std::vector<IR_ScalarRef *> FindScalarRef(
    const omega::CG_outputRepr *repr) const;
  std::vector<IR_Control *> FindOneLevelControlStructure(
    const IR_Block *block) const;
  IR_Block *MergeNeighboringControlStructures(
    const std::vector<IR_Control *> &controls) const;
  IR_Block *GetCode() const;
  void ReplaceCode(IR_Control *old, omega::CG_outputRepr *repr);
  void ReplaceExpression(IR_Ref *old, omega::CG_outputRepr *repr);
  
  IR_OPERATION_TYPE QueryExpOperation(const omega::CG_outputRepr *repr) const;
  IR_CONDITION_TYPE QueryBooleanExpOperation(
    const omega::CG_outputRepr *repr) const;
  std::vector<omega::CG_outputRepr *> QueryExpOperand(
    const omega::CG_outputRepr *repr) const;
  IR_Ref *Repr2Ref(const omega::CG_outputRepr *) const;
  /*    std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> >
        FindScalarDeps(const omega::CG_outputRepr *repr1,
        const omega::CG_outputRepr *repr2, std::vector<std::string> index,
        int i, int j);
  */
  void finalizeRose();
  friend class IR_roseArraySymbol;
  friend class IR_roseArrayRef;
};

#endif
