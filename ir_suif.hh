#ifndef IR_SUIF_HH
#define IR_SUIF_HH

#include <map>
#include <code_gen/CG_suifRepr.h>
#include <code_gen/CG_suifBuilder.h>
#include "ir_code.hh"

struct IR_suifScalarSymbol: public IR_ScalarSymbol {
  var_sym *vs_;

  IR_suifScalarSymbol(const IR_Code *ir, var_sym *vs) {
    ir_ = ir;
    vs_ = vs;
  }
  std::string name() const;
  int size() const;
  bool operator==(const IR_Symbol &that) const;
  IR_Symbol *clone() const;
};


struct IR_suifArraySymbol: public IR_ArraySymbol {
  var_sym *vs_;
  int indirect_;
  int offset_;

  IR_suifArraySymbol(const IR_Code *ir, var_sym *vs, int indirect = 0, int offset = 0) {
    ir_ = ir;
    vs_ = vs;
    indirect_ = indirect;
    offset_ = offset;
  }
  std::string name() const;
  int elem_size() const;
  int n_dim() const;
  omega::CG_outputRepr *size(int dim) const;
  bool operator==(const IR_Symbol &that) const;
  IR_ARRAY_LAYOUT_TYPE layout_type() const;
  IR_Symbol *clone() const;
};


struct IR_suifConstantRef: public IR_ConstantRef {
  union {
    omega::coef_t i_;
    double f_;
  };

  IR_suifConstantRef(const IR_Code *ir, omega::coef_t i) {
    ir_ = ir;
    type_ = IR_CONSTANT_INT;
    i_ = i;
  }
  IR_suifConstantRef(const IR_Code *ir, double f) {
    ir_ = ir;
    type_ = IR_CONSTANT_FLOAT;
    f_ = f;
  }
  omega::coef_t integer() const {assert(is_integer()); return i_;}
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
};


struct IR_suifScalarRef: public IR_ScalarRef {
  instruction *ins_pos_;  
  int op_pos_; // -1 means destination operand, otherwise source operand
  var_sym *vs_;

  IR_suifScalarRef(const IR_Code *ir, var_sym *sym) {
    ir_ = ir;
    ins_pos_ = NULL;
    vs_ = sym;
  }
  IR_suifScalarRef(const IR_Code *ir, instruction *ins, int pos) {
    ir_ = ir;
    ins_pos_ = ins;
    op_pos_ = pos;
    operand op;
    if (pos == -1)
      op = ins->dst_op();
    else
      op = ins->src_op(pos);
    assert(op.is_symbol());
    vs_ = op.symbol();
  } 
  bool is_write() const;
  IR_ScalarSymbol *symbol() const;
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
};
  

struct IR_suifArrayRef: public IR_ArrayRef {
  in_array *ia_;

  IR_suifArrayRef(const IR_Code *ir, in_array *ia) {
    ir_ = ir;
    ia_ = ia;
  }
  bool is_write() const;
  omega::CG_outputRepr *index(int dim) const;
  IR_ArraySymbol *symbol() const;
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
};


struct IR_suifLoop: public IR_Loop {
  tree_for *tf_;
  
  IR_suifLoop(const IR_Code *ir, tree_for *tf) { ir_ = ir; tf_ = tf; }
  ~IR_suifLoop() {}
  IR_ScalarSymbol *index() const;
  omega::CG_outputRepr *lower_bound() const;
  omega::CG_outputRepr *upper_bound() const;
  IR_CONDITION_TYPE stop_cond() const;
  IR_Block *body() const;
  int step_size() const;
  IR_Block *convert();
  IR_Control *clone() const;
};


struct IR_suifBlock: public IR_Block {
  tree_node_list *tnl_;
  tree_node_list_e *start_, *end_;

  IR_suifBlock(const IR_Code *ir, tree_node_list *tnl, tree_node_list_e *start, tree_node_list_e *end) {
    ir_ = ir; tnl_ = tnl; start_ = start; end_ = end;
  }
  IR_suifBlock(const IR_Code *ir, tree_node_list *tnl) {
    ir_ = ir; tnl_ = tnl; start_ = tnl_->head(); end_ = tnl_->tail();
  }
  ~IR_suifBlock() {}
  omega::CG_outputRepr *extract() const;
  IR_Control *clone() const;
};


struct IR_suifIf: public IR_If {
  tree_if *ti_;

  IR_suifIf(const IR_Code *ir, tree_if *ti) { ir_ = ir; ti_ = ti; }
  ~IR_suifIf() {}
  omega::CG_outputRepr *condition() const;
  IR_Block *then_body() const;
  IR_Block *else_body() const;
  IR_Block *convert();
  IR_Control *clone() const;
};

  
// singleton class for global suif initialization
class IR_suifCode_Global_Init {
private:
  static IR_suifCode_Global_Init *pinstance;  
protected:
  IR_suifCode_Global_Init();
  IR_suifCode_Global_Init(const IR_suifCode_Global_Init &);
  IR_suifCode_Global_Init & operator= (const IR_suifCode_Global_Init &);
public:
  static IR_suifCode_Global_Init *Instance();
  ~IR_suifCode_Global_Init() {}
};

// singleton class for global suif cleanup
class IR_suifCode_Global_Cleanup {
public:
  IR_suifCode_Global_Cleanup() {}
  ~IR_suifCode_Global_Cleanup();
};

class IR_suifCode: public IR_Code{
protected:
  file_set_entry *fse_;
  proc_sym *psym_;
  proc_symtab *symtab_;
  bool is_fortran_;

public:
  IR_suifCode(const char *filename, int proc_num);
  ~IR_suifCode();
  
  IR_ScalarSymbol *CreateScalarSymbol(const IR_Symbol *sym, int memory_type = 0);
  IR_ArraySymbol *CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size, int memory_type = 0);
  IR_ScalarRef *CreateScalarRef(const IR_ScalarSymbol *sym);
  IR_ArrayRef *CreateArrayRef(const IR_ArraySymbol *sym, std::vector<omega::CG_outputRepr *> &index);
  int ArrayIndexStartAt() {if (is_fortran_) return 1; else return 0;}

  std::vector<IR_ArrayRef *> FindArrayRef(const omega::CG_outputRepr *repr) const;
  std::vector<IR_ScalarRef *> FindScalarRef(const omega::CG_outputRepr *repr) const;
  std::vector<IR_Control *> FindOneLevelControlStructure(const IR_Block *block) const;
  IR_Block *MergeNeighboringControlStructures(const std::vector<IR_Control *> &controls) const;
  IR_Block *GetCode() const;
  void ReplaceCode(IR_Control *old, omega::CG_outputRepr *repr);
  void ReplaceExpression(IR_Ref *old, omega::CG_outputRepr *repr);

  IR_OPERATION_TYPE QueryExpOperation(const omega::CG_outputRepr *repr) const;
  IR_CONDITION_TYPE QueryBooleanExpOperation(const omega::CG_outputRepr *repr) const;
  std::vector<omega::CG_outputRepr *> QueryExpOperand(const omega::CG_outputRepr *repr) const;
  IR_Ref *Repr2Ref(const omega::CG_outputRepr *) const;
  
  friend class IR_suifArraySymbol;
  friend class IR_suifArrayRef;
};

#endif
