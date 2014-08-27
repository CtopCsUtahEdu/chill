/*****************************************************************************
 Copyright (C) 2009-2011 University of Utah
 All Rights Reserved.

 Purpose:
   CHiLL's SUIF interface.

 Notes:
   Array supports mixed pointer and array type in a single declaration.

 History:
   02/23/2009 Created by Chun Chen.
*****************************************************************************/

#include <typeinfo>
#include <useful.h>
#include "ir_suif.hh"
#include "ir_suif_utils.hh"
#include "chill_error.hh"

// ----------------------------------------------------------------------------
// Class: IR_suifScalarSymbol
// ----------------------------------------------------------------------------

std::string IR_suifScalarSymbol::name() const {
  return vs_->name();
}


int IR_suifScalarSymbol::size() const {
  return vs_->type()->size();
}


bool IR_suifScalarSymbol::operator==(const IR_Symbol &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_suifScalarSymbol *l_that = static_cast<const IR_suifScalarSymbol *>(&that);
  return this->vs_ == l_that->vs_;
}

IR_Symbol *IR_suifScalarSymbol::clone() const {
  return new IR_suifScalarSymbol(ir_, vs_);
}

// ----------------------------------------------------------------------------
// Class: IR_suifArraySymbol
// ----------------------------------------------------------------------------

std::string IR_suifArraySymbol::name() const {
  return vs_->name();
}


int IR_suifArraySymbol::elem_size() const {
  type_node *tn = vs_->type();
  if (tn->is_modifier())
    tn = static_cast<modifier_type *>(tn)->base();
  
  while (tn->is_array())
    tn = static_cast<array_type *>(tn)->elem_type();
  
  return tn->size();
}


int IR_suifArraySymbol::n_dim() const {
  type_node *tn = vs_->type();
  if (tn->is_modifier())
    tn = static_cast<modifier_type *>(tn)->base();
  
  int n = 0;
  while (true) {
    if (tn->is_array()) {
      n++;
      tn = static_cast<array_type *>(tn)->elem_type();
    }
    else if (tn->is_ptr()) {
      n++;
      tn = static_cast<ptr_type *>(tn)->ref_type();
    }
    else
      break;
  }
  
  return n - indirect_;
}


omega::CG_outputRepr *IR_suifArraySymbol::size(int dim) const {
  type_node *tn = vs_->type();
  if (tn->is_modifier())
    tn = static_cast<modifier_type *>(tn)->base();
  
  for (int i = 0; i < dim; i++) {
    if (tn->is_array())
      tn = static_cast<array_type *>(tn)->elem_type();
    else if (tn->is_ptr())
      tn = static_cast<ptr_type *>(tn)->ref_type();
    else
      throw ir_error("array parsing error");
  }
  if (tn->is_ptr())
    return new omega::CG_suifRepr(operand());
  else if (!tn->is_array())
    throw ir_error("array parsing error");
  
  array_bound ub = static_cast<array_type *>(tn)->upper_bound();
  int c = 1;
  omega::CG_outputRepr *ub_repr = NULL;
  if (ub.is_constant())
    c += ub.constant();
  else if (ub.is_variable()) {
    var_sym *vs = ub.variable();
    
    if (static_cast<const IR_suifCode *>(ir_)->init_code_ != NULL) {
      tree_node_list *tnl = static_cast<omega::CG_suifRepr *>(static_cast<const IR_suifCode *>(ir_)->init_code_)->GetCode();
      tree_node_list_iter iter(tnl);
      while(!iter.is_empty()) {
        tree_node *tn = iter.step();
        if (tn->is_instr()) {
          instruction *ins = static_cast<tree_instr *>(tn)->instr();
          operand dst = ins->dst_op();
          if (dst.is_symbol() && dst.symbol() == vs) {
            operand op;
            if (ins->opcode() == io_cpy)
              op = ins->src_op(0).clone();
            else
              op = operand(ins->clone());
            
            ub_repr = new omega::CG_suifRepr(op);
            break;
          }
        }
      }
    }
    if (ub_repr == NULL)
      ub_repr = new omega::CG_suifRepr(operand(vs));
  }
  else
    throw ir_error("array parsing error");
  
  array_bound lb = static_cast<array_type *>(tn)->lower_bound();  
  omega::CG_outputRepr *lb_repr = NULL;
  if (lb.is_constant())
    c -= lb.constant();
  else if (lb.is_variable()) {
    var_sym *vs = ub.variable();
    
    tree_node_list *tnl = static_cast<omega::CG_suifRepr *>(static_cast<const IR_suifCode *>(ir_)->init_code_)->GetCode();
    tree_node_list_iter iter(tnl);
    while(!iter.is_empty()) {
      tree_node *tn = iter.step();
      if (tn->is_instr()) {
        instruction *ins = static_cast<tree_instr *>(tn)->instr();
        operand dst = ins->dst_op();
        if (dst.is_symbol() && dst.symbol() == vs) {
          operand op;
          if (ins->opcode() == io_cpy)
            op = ins->src_op(0).clone();
          else
            op = operand(ins->clone());
          
          lb_repr = new omega::CG_suifRepr(op);
          break;
        }
      }
    }
    if (lb_repr == NULL)
      lb_repr = new omega::CG_suifRepr(operand(vs));
  }
  else
    throw ir_error("array parsing error");
  
  omega::CG_outputRepr *repr = ir_->builder()->CreateMinus(ub_repr, lb_repr);
  if (c != 0)
    repr = ir_->builder()->CreatePlus(repr, ir_->builder()->CreateInt(c));
  
  return repr;
}


IR_ARRAY_LAYOUT_TYPE IR_suifArraySymbol::layout_type() const {
  if (static_cast<const IR_suifCode *>(ir_)->is_fortran_)
    return IR_ARRAY_LAYOUT_COLUMN_MAJOR;
  else
    return IR_ARRAY_LAYOUT_ROW_MAJOR;
}


bool IR_suifArraySymbol::operator==(const IR_Symbol &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_suifArraySymbol *l_that = static_cast<const IR_suifArraySymbol *>(&that);
  return this->vs_ == l_that->vs_ && this->offset_ == l_that->offset_;
}


IR_Symbol *IR_suifArraySymbol::clone() const {
  return new IR_suifArraySymbol(ir_, vs_, indirect_, offset_);
}

// ----------------------------------------------------------------------------
// Class: IR_suifConstantRef
// ----------------------------------------------------------------------------

bool IR_suifConstantRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_suifConstantRef *l_that = static_cast<const IR_suifConstantRef *>(&that);
  
  if (this->type_ != l_that->type_)
    return false;
  
  if (this->type_ == IR_CONSTANT_INT)
    return this->i_ == l_that->i_;
  else
    return this->f_ == l_that->f_;
}


omega::CG_outputRepr *IR_suifConstantRef::convert() {
  if (type_ == IR_CONSTANT_INT) {
    omega::CG_suifRepr *result = new omega::CG_suifRepr(operand(static_cast<int>(i_), type_s32));
    delete this;
    return result;
  }
  else
    throw ir_error("constant type not supported");
}


IR_Ref *IR_suifConstantRef::clone() const {
  if (type_ == IR_CONSTANT_INT)
    return new IR_suifConstantRef(ir_, i_);
  else if (type_ == IR_CONSTANT_FLOAT)
    return new IR_suifConstantRef(ir_, f_);
  else
    throw ir_error("constant type not supported");
}


// ----------------------------------------------------------------------------
// Class: IR_suifScalarRef
// ----------------------------------------------------------------------------

bool IR_suifScalarRef::is_write() const {
  if (ins_pos_ != NULL && op_pos_ == -1)
    return true;
  else
    return false;
}


IR_ScalarSymbol *IR_suifScalarRef::symbol() const {
  return new IR_suifScalarSymbol(ir_, vs_);
}


bool IR_suifScalarRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_suifScalarRef *l_that = static_cast<const IR_suifScalarRef *>(&that);
  
  if (this->ins_pos_ == NULL)
    return this->vs_ == l_that->vs_;
  else 
    return this->ins_pos_ == l_that->ins_pos_ && this->op_pos_ == l_that->op_pos_;
}


omega::CG_outputRepr *IR_suifScalarRef::convert() {
  omega::CG_suifRepr *result = new omega::CG_suifRepr(operand(vs_));
  delete this;
  return result;
}


IR_Ref * IR_suifScalarRef::clone() const {
  if (ins_pos_ == NULL)
    return new IR_suifScalarRef(ir_, vs_);
  else
    return new IR_suifScalarRef(ir_, ins_pos_, op_pos_);
}


// ----------------------------------------------------------------------------
// Class: IR_suifArrayRef
// ----------------------------------------------------------------------------

bool IR_suifArrayRef::is_write() const {
  return ::is_lhs(const_cast<in_array *>(ia_));
}


omega::CG_outputRepr *IR_suifArrayRef::index(int dim) const {
  operand op = find_array_index(ia_, n_dim(), dim, static_cast<const IR_suifCode *>(ir_)->is_fortran_);
  return new omega::CG_suifRepr(op.clone());
}


IR_ArraySymbol *IR_suifArrayRef::symbol() const {
  in_array *current = ia_;
  
  // find the indirectness of the symbol, i.e., if it is (**A)[i,j]
  int indirect = 0;
  if (!static_cast<const IR_suifCode *>(ir_)->is_fortran_) {
    operand op = ia_->base_op();
    while (op.is_instr()) {
      instruction *ins = op.instr();
      if (ins->opcode() == io_lod) {
        indirect++;
        op = ins->src_op(0);
      }
      else
        break;
    }
    if (op.is_symbol())
      indirect++;
  }
  
  while (true) {
    operand op = current->base_op();
    if (op.is_symbol()) {
      return new IR_suifArraySymbol(ir_, op.symbol(), indirect);
    }
    else if (op.is_instr()) {
      instruction *ins = op.instr();
      if (ins->opcode() == io_ldc) {
        immed value = static_cast<in_ldc *>(ins)->value();
        if (value.is_symbol()) {
          sym_node *the_sym = value.symbol();
          if (the_sym->is_var())
            return new IR_suifArraySymbol(ir_, static_cast<var_sym *>(the_sym), indirect);
          else
            break;
        }
        else
          break;
      }
      else if (ins->opcode() == io_cvt) {
        operand op = static_cast<in_rrr *>(ins)->src_op();
        if (op.is_symbol()) {
          return new IR_suifArraySymbol(ir_, op.symbol(), indirect);
        }
        else if (op.is_instr()) {
          instruction *ins = op.instr();
          if (ins->opcode() == io_lod) {
            operand op = static_cast<in_rrr *>(ins)->src_op();
            if (op.is_symbol()) {
              return new IR_suifArraySymbol(ir_, op.symbol(), indirect);
            }
            else if (op.is_instr()) {
              instruction *ins = op.instr();
              if (ins->opcode() == io_array) {
                current = static_cast<in_array *>(ins);
                continue;
              }
              else if (ins->opcode() == io_add) {
                operand op1 = ins->src_op(0);
                operand op2 = ins->src_op(1);
                if (!op1.is_symbol() || !op2.is_immed())
                  throw ir_error("can't recognize array reference format");
                immed im = op2.immediate();
                if (!im.is_integer())
                  throw ir_error("can't recognize array reference format");                  
                return new IR_suifArraySymbol(ir_, op1.symbol(), indirect, im.integer());
              }
              else
                break;
            }
            else
              break;
          }
          else
            break;
        }
        else
          break;
      }
      else {
        while (ins->opcode() == io_lod) {
          operand op = ins->src_op(0);
          if (op.is_instr())
            ins = op.instr();
          else if (op.is_symbol())
            return new IR_suifArraySymbol(ir_, op.symbol(), indirect);
          else
            break;
        }
        break;
      }
    }
    else
      break;
  }
  
  fprintf(stderr, "Warning: null array symbol found, dependence graph bloated!\n");
  
  return new IR_suifArraySymbol(ir_, NULL);
}


bool IR_suifArrayRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_suifArrayRef *l_that = static_cast<const IR_suifArrayRef *>(&that);
  
  return this->ia_ == l_that ->ia_;
}


omega::CG_outputRepr *IR_suifArrayRef::convert() {
  omega::CG_suifRepr *result = new omega::CG_suifRepr(operand(this->ia_->clone()));
  delete this;
  return result;
}


IR_Ref *IR_suifArrayRef::clone() const {
  return new IR_suifArrayRef(ir_, ia_);
}



// ----------------------------------------------------------------------------
// Class: IR_suifLoop
// ----------------------------------------------------------------------------

IR_ScalarSymbol *IR_suifLoop::index() const {
  var_sym *vs = tf_->index();
  return new IR_suifScalarSymbol(ir_, vs);
}

omega::CG_outputRepr *IR_suifLoop::lower_bound() const {
  tree_node_list *tnl = tf_->lb_list();
  tree_node_list_iter iter(tnl);
  if (iter.is_empty())
    return new omega::CG_suifRepr(operand());
  tree_node *tn = iter.step();
  if (!iter.is_empty())
    throw ir_error("cannot handle lower bound");
  if (tn->kind() != TREE_INSTR)
    throw ir_error("cannot handle lower bound");
  instruction *ins = static_cast<tree_instr *>(tn)->instr();
  return new omega::CG_suifRepr(operand(ins));
}

omega::CG_outputRepr *IR_suifLoop::upper_bound() const {
  tree_node_list *tnl = tf_->ub_list();
  tree_node_list_iter iter(tnl);
  if (iter.is_empty())
    return new omega::CG_suifRepr(operand());
  tree_node *tn = iter.step();
  if (!iter.is_empty())
    throw ir_error("cannot handle lower bound");
  if (tn->kind() != TREE_INSTR)
    throw ir_error("cannot handle lower bound");
  instruction *ins = static_cast<tree_instr *>(tn)->instr();
  return new omega::CG_suifRepr(operand(ins));
}

IR_CONDITION_TYPE IR_suifLoop::stop_cond() const {
  if (tf_->test() == FOR_SLT || tf_->test() == FOR_ULT)
    return IR_COND_LT;
  else if (tf_->test() == FOR_SLTE || tf_->test() == FOR_ULTE)
    return IR_COND_LE;
  else if (tf_->test() == FOR_SGT || tf_->test() == FOR_UGT)
    return IR_COND_GT;
  else if (tf_->test() == FOR_SGTE || tf_->test() == FOR_UGTE)
    return IR_COND_GE;
  else
    throw ir_error("loop stop condition unsupported");
}

IR_Block *IR_suifLoop::body() const {
  tree_node_list *tnl = tf_->body();
  return new IR_suifBlock(ir_, tnl);
}

int IR_suifLoop::step_size() const {
  operand op = tf_->step_op();
  if (!op.is_null()) {
    if (op.is_immed()) {
      immed im = op.immediate();
      if (im.is_integer())
        return im.integer();
      else
        throw ir_error("cannot handle non-integer stride");
    }
    else
      throw ir_error("cannot handle non-constant stride");
  }
  else
    return 1;
}


IR_Block *IR_suifLoop::convert() {
  const IR_Code *ir = ir_;
  tree_node_list *tnl = tf_->parent();
  tree_node_list_e *start, *end;
  start = end = tf_->list_e();
  delete this;
  return new IR_suifBlock(ir, tnl, start, end);
}


IR_Control *IR_suifLoop::clone() const {
  return new IR_suifLoop(ir_, tf_);
}

// ----------------------------------------------------------------------------
// Class: IR_suifBlock
// ----------------------------------------------------------------------------

omega::CG_outputRepr *IR_suifBlock::extract() const {
  tree_node_list *tnl = new tree_node_list;
  tree_node_list_iter iter(tnl_);
  while (!iter.is_empty()) {
    tree_node *tn = iter.peek();
    if (tn->list_e() == start_)
      break;
    tn = iter.step();
  }
  
  while (!iter.is_empty()) {
    tree_node *tn = iter.step();
    tnl->append(tn->clone());
    if (tn->list_e() == end_)
      break;
  }
  
  return new omega::CG_suifRepr(tnl);
}

IR_Control *IR_suifBlock::clone() const {
  return new IR_suifBlock(ir_, tnl_, start_, end_);
}


// ----------------------------------------------------------------------------
// Class: IR_suifIf
// ----------------------------------------------------------------------------
omega::CG_outputRepr *IR_suifIf::condition() const {
  tree_node_list *tnl = ti_->header();
  tree_node_list_iter iter(tnl);
  if (iter.is_empty())
    throw ir_error("unrecognized if structure");
  tree_node *tn = iter.step();
  if (!iter.is_empty())
    throw ir_error("unrecognized if structure");
  if (!tn->is_instr())
    throw ir_error("unrecognized if structure");
  instruction *ins = static_cast<tree_instr *>(tn)->instr();
  if (!ins->opcode() == io_bfalse)
    throw ir_error("unrecognized if structure");
  operand op = ins->src_op(0);
  return new omega::CG_suifRepr(op);
}

IR_Block *IR_suifIf::then_body() const {
  tree_node_list *tnl = ti_->then_part();
  if (tnl == NULL)
    return NULL;
  tree_node_list_iter iter(tnl);
  if (iter.is_empty())
    return NULL;
  
  return new IR_suifBlock(ir_, tnl);
}

IR_Block *IR_suifIf::else_body() const {
  tree_node_list *tnl = ti_->else_part();
  if (tnl == NULL)
    return NULL;
  tree_node_list_iter iter(tnl);
  if (iter.is_empty())
    return NULL;
  
  return new IR_suifBlock(ir_, tnl);
}


IR_Block *IR_suifIf::convert() {
  const IR_Code *ir = ir_;
  tree_node_list *tnl = ti_->parent();
  tree_node_list_e *start, *end;
  start = end = ti_->list_e();
  delete this;
  return new IR_suifBlock(ir, tnl, start, end);
}


IR_Control *IR_suifIf::clone() const {
  return new IR_suifIf(ir_, ti_);
}


// ----------------------------------------------------------------------------
// Class: IR_suifCode_Global_Init
// ----------------------------------------------------------------------------

IR_suifCode_Global_Init *IR_suifCode_Global_Init::pinstance = NULL;


IR_suifCode_Global_Init *IR_suifCode_Global_Init::Instance () {
  if (pinstance == NULL)
    pinstance = new IR_suifCode_Global_Init;
  return pinstance;
}


IR_suifCode_Global_Init::IR_suifCode_Global_Init() {
  LIBRARY(useful, init_useful, exit_useful);
  LIBRARY(annotes, init_annotes, exit_annotes);
  
  int argc = 1;
  char *argv[1];
  argv[0] = "chill";
  init_suif(argc, argv);
}


// ----------------------------------------------------------------------------
// Class: IR_suifCode_Global_Cleanup
// ----------------------------------------------------------------------------

IR_suifCode_Global_Cleanup::~IR_suifCode_Global_Cleanup() {
  delete IR_suifCode_Global_Init::Instance();
  exit_suif1();
}


namespace {
  IR_suifCode_Global_Cleanup suifcode_global_cleanup_instance;
}

// ----------------------------------------------------------------------------
// Class: IR_suifCode
// ----------------------------------------------------------------------------

IR_suifCode::IR_suifCode(const char *filename, int proc_num): IR_Code() {
  IR_suifCode_Global_Init::Instance();
  
  std::string new_filename(filename);
  int pos = new_filename.find_last_of('.');
  new_filename = new_filename.substr(0, pos) + ".lxf";
  fileset->add_file(const_cast<char *>(filename), const_cast<char *>(new_filename.c_str()));  
  fileset->reset_iter();
  fse_ = fileset->next_file();
  fse_->reset_proc_iter();
  
  int cur_proc = 0;
  while ((psym_ = fse_->next_proc()) && cur_proc < proc_num)
    ++cur_proc;
  if (cur_proc != proc_num) {
    throw ir_error("procedure number cannot be found");
  }
  
  if (psym_->src_lang() == src_fortran)
    is_fortran_ = true;
  else
    is_fortran_ = false;
  
  if (!psym_->is_in_memory())
    psym_->read_proc(TRUE, is_fortran_);
  push_clue(psym_->block());
  
  symtab_ = psym_->block()->proc_syms();
  ocg_ = new omega::CG_suifBuilder(symtab_);
}


IR_suifCode::~IR_suifCode() {
  tree_node_list *tnl = psym_->block()->body();
  
  if (init_code_ != NULL)
    tnl->insert_before(static_cast<omega::CG_suifRepr *>(init_code_)->GetCode(), tnl->head());
  if (cleanup_code_ != NULL)
    tnl->insert_after(static_cast<omega::CG_suifRepr *>(cleanup_code_)->GetCode(), tnl->tail());
  
  pop_clue(psym_->block());
  if (!psym_->is_written())
    psym_->write_proc(fse_);
  psym_->flush_proc();
}


IR_ScalarSymbol *IR_suifCode::CreateScalarSymbol(const IR_Symbol *sym, int) {
  if (typeid(*sym) == typeid(IR_suifScalarSymbol)) {
    type_node *tn = static_cast<const IR_suifScalarSymbol *>(sym)->vs_->type();
    while (tn->is_modifier())
      tn = static_cast<modifier_type *>(tn)->base();
    var_sym *vs = symtab_->new_unique_var(tn);
    return new IR_suifScalarSymbol(this, vs);
  }
  else if (typeid(*sym) == typeid(IR_suifArraySymbol)) {
    type_node *tn = static_cast<const IR_suifArraySymbol *>(sym)->vs_->type();
    while (tn->is_modifier())
      tn = static_cast<modifier_type *>(tn)->base();
    while (tn->is_array() || tn->is_ptr()) {
      if (tn->is_array())
        tn = static_cast<array_type *>(tn)->elem_type();
      else if (tn->is_ptr())
        tn = static_cast<ptr_type *>(tn)->ref_type();
    }
    while (tn->is_modifier())
      tn = static_cast<modifier_type *>(tn)->base();
    var_sym *vs = symtab_->new_unique_var(tn);
    return new IR_suifScalarSymbol(this, vs);
  }
  else
    throw std::bad_typeid();
}


IR_ArraySymbol *IR_suifCode::CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size, int) {
  type_node *tn;
  
  if (typeid(*sym) == typeid(IR_suifScalarSymbol)) {
    tn = static_cast<const IR_suifScalarSymbol *>(sym)->vs_->type();
  }
  else if (typeid(*sym) == typeid(IR_suifArraySymbol)) {
    tn = static_cast<const IR_suifArraySymbol *>(sym)->vs_->type();
    if (tn->is_modifier())
      tn = static_cast<modifier_type *>(tn)->base();
    while (tn->is_array() || tn->is_ptr()) {
      if (tn->is_array())
        tn = static_cast<array_type *>(tn)->elem_type();
      else if (tn->is_ptr())
        tn = static_cast<ptr_type *>(tn)->ref_type();
    } 
  }
  else
    throw std::bad_typeid();
  
  if (is_fortran_)
    for (int i = 0; i < size.size(); i++) {
      var_sym *temporary = symtab_->new_unique_var(type_s32);
      init_code_ = ocg_->StmtListAppend(init_code_, ocg_->CreateAssignment(0, new omega::CG_suifRepr(operand(temporary)), size[i]));
      
      tn = new array_type(tn, array_bound(1), array_bound(temporary));
      symtab_->add_type(tn);
    }
  else     
    for (int i = size.size()-1; i >= 0; i--) {
      var_sym *temporary = symtab_->new_unique_var(type_s32);
      init_code_ = ocg_->StmtListAppend(init_code_, ocg_->CreateAssignment(0, new omega::CG_suifRepr(operand(temporary)), size[i]));
      
      tn = new array_type(tn, array_bound(1), array_bound(temporary));
      symtab_->add_type(tn);
    }
  
  static int suif_array_counter = 1;
  std::string s = std::string("_P") + omega::to_string(suif_array_counter++);
  var_sym *vs = new var_sym(tn, const_cast<char *>(s.c_str()));
  vs->add_to_table(symtab_);
  
  return new IR_suifArraySymbol(this, vs);
}


IR_ScalarRef *IR_suifCode::CreateScalarRef(const IR_ScalarSymbol *sym) {
  return new IR_suifScalarRef(this, static_cast<const IR_suifScalarSymbol *>(sym)->vs_);
}


IR_ArrayRef *IR_suifCode::CreateArrayRef(const IR_ArraySymbol *sym, std::vector<omega::CG_outputRepr *> &index) {
  if (sym->n_dim() != index.size())
    throw std::invalid_argument("incorrect array symbol dimensionality");
  
  const IR_suifArraySymbol *l_sym = static_cast<const IR_suifArraySymbol *>(sym);
  
  var_sym *vs = l_sym->vs_;
  type_node *tn1 = vs->type();
  if (tn1->is_modifier())
    tn1 = static_cast<modifier_type *>(tn1)->base();
  
  type_node *tn2 = tn1;
  while (tn2->is_array() || tn2->is_ptr()) {
    if (tn2->is_array())
      tn2 = static_cast<array_type *>(tn2)->elem_type();
    else if (tn2->is_ptr())
      tn2 = static_cast<ptr_type *>(tn2)->ref_type();
  }
  
  instruction *base_ins;
  if (tn1->is_ptr()) {
    base_symtab *cur_symtab;
    
    cur_symtab = symtab_;
    type_node *found_array_tn = NULL;
    while (cur_symtab != NULL) {
      type_node_list_iter iter(cur_symtab->types());
      while (!iter.is_empty()) {
        type_node *tn = iter.step();
        if (!tn->is_array())
          continue;
        if (static_cast<array_type *>(tn)->elem_type() == static_cast<ptr_type *>(tn1)->ref_type()) {
          array_bound b = static_cast<array_type *>(tn)->upper_bound();
          if (b.is_unknown()) {
            found_array_tn = tn;
            break;
          }
        }
      }
      if (found_array_tn == NULL)
        cur_symtab = cur_symtab->parent();
      else
        break;
    }
    
    cur_symtab = symtab_;
    type_node *found_ptr_array_tn = NULL;
    while (cur_symtab != NULL) {
      type_node_list_iter iter(cur_symtab->types());
      while (!iter.is_empty()) {
        type_node *tn = iter.step();
        if (!tn->is_ptr())
          continue;
        if (static_cast<ptr_type *>(tn)->ref_type() == found_array_tn) {
          found_ptr_array_tn = tn;
          break;
        }
      }
      if (found_ptr_array_tn == NULL)
        cur_symtab = cur_symtab->parent();
      else
        break;
    }
    
    if (found_ptr_array_tn == NULL)
      throw ir_error("can't find the type for the to-be-created array");
    base_ins = new in_rrr(io_cvt, found_ptr_array_tn, operand(), operand(vs));
  }
  else {
    base_ins = new in_ldc(tn1->ptr_to(), operand(), immed(vs));
  }
  
  in_array *ia = new in_array(tn2->ptr_to(), operand(), operand(base_ins), tn2->size(), l_sym->n_dim());
  
  for (int i = 0; i < index.size(); i++) {
    int t;
    if (is_fortran_)
      t = index.size() - i - 1;
    else
      t = i;
    
    omega::CG_suifRepr *bound = static_cast<omega::CG_suifRepr *>(l_sym->size(t));
    ia->set_bound(t, bound->GetExpression());
    delete bound;
    omega::CG_suifRepr *idx = static_cast<omega::CG_suifRepr *>(index[i]);
    ia->set_index(t, idx->GetExpression());
    delete idx;
  }
  
  return new IR_suifArrayRef(this, ia);
}


std::vector<IR_ArrayRef *> IR_suifCode::FindArrayRef(const omega::CG_outputRepr *repr) const {
  std::vector<IR_ArrayRef *> arrays;
  
  tree_node_list *tnl = static_cast<const omega::CG_suifRepr *>(repr)->GetCode();
  if (tnl != NULL) {
    tree_node_list_iter iter(tnl);
    while (!iter.is_empty()) {
      tree_node *tn = iter.step();
      switch (tn->kind()) {
      case TREE_FOR: {
        tree_for *tnf = static_cast<tree_for *>(tn);
        omega::CG_suifRepr *r = new omega::CG_suifRepr(tnf->body());
        std::vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
        break;
      }
      case TREE_IF: {
        tree_if *tni = static_cast<tree_if *>(tn);
        omega::CG_suifRepr *r = new omega::CG_suifRepr(tni->header());
        std::vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
        r = new omega::CG_suifRepr(tni->then_part());
        a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
        r = new omega::CG_suifRepr(tni->else_part());
        a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
        break;
      }
      case TREE_BLOCK: {
        omega::CG_suifRepr *r = new omega::CG_suifRepr(static_cast<tree_block *>(tn)->body());
        std::vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
        break;
      }
      case TREE_INSTR: {
        omega::CG_suifRepr *r = new omega::CG_suifRepr(operand(static_cast<tree_instr *>(tn)->instr()));
        std::vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(arrays));
        break;
      }
      default:
        throw ir_error("control structure not supported");
      }
    }
  }
  else {
    operand op = static_cast<const omega::CG_suifRepr *>(repr)->GetExpression();
    if (op.is_instr()) {
      instruction *ins = op.instr();
      switch (ins->opcode()) {
      case io_array: {
        IR_suifArrayRef *ref = new IR_suifArrayRef(this, static_cast<in_array *>(ins));
        for (int i = 0; i < ref->n_dim(); i++) {
          omega::CG_suifRepr *r = new omega::CG_suifRepr(find_array_index(ref->ia_, ref->n_dim(), i, is_fortran_));
          std::vector<IR_ArrayRef *> a = FindArrayRef(r);
          delete r;
          std::copy(a.begin(), a.end(), back_inserter(arrays));
        }
        arrays.push_back(ref);
        break;
      }
      case io_str:
      case io_memcpy: {
        omega::CG_suifRepr *r1 = new omega::CG_suifRepr(ins->src_op(1));
        std::vector<IR_ArrayRef *> a1 = FindArrayRef(r1);
        delete r1;
        std::copy(a1.begin(), a1.end(), back_inserter(arrays));
        omega::CG_suifRepr *r2 = new omega::CG_suifRepr(ins->src_op(0));
        std::vector<IR_ArrayRef *> a2 = FindArrayRef(r2);
        delete r2;
        std::copy(a2.begin(), a2.end(), back_inserter(arrays));
        break;
      }
      default:
        for (int i = 0; i < ins->num_srcs(); i++) {
          omega::CG_suifRepr *r = new omega::CG_suifRepr(ins->src_op(i));
          std::vector<IR_ArrayRef *> a = FindArrayRef(r);
          delete r;
          std::copy(a.begin(), a.end(), back_inserter(arrays));
        }
      }
    }
  }
  
  return arrays;
}


std::vector<IR_ScalarRef *> IR_suifCode::FindScalarRef(const omega::CG_outputRepr *repr) const {
  std::vector<IR_ScalarRef *> scalars;
  
  tree_node_list *tnl = static_cast<const omega::CG_suifRepr *>(repr)->GetCode();
  if (tnl != NULL) {
    tree_node_list_iter iter(tnl);
    while (!iter.is_empty()) {
      tree_node *tn = iter.step();
      switch (tn->kind()) {
      case TREE_FOR: {
        tree_for *tnf = static_cast<tree_for *>(tn);
        omega::CG_suifRepr *r = new omega::CG_suifRepr(tnf->body());
        std::vector<IR_ScalarRef *> a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
        break;
      }
      case TREE_IF: {
        tree_if *tni = static_cast<tree_if *>(tn);
        omega::CG_suifRepr *r = new omega::CG_suifRepr(tni->header());
        std::vector<IR_ScalarRef *> a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
        r = new omega::CG_suifRepr(tni->then_part());
        a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
        r = new omega::CG_suifRepr(tni->else_part());
        a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
        break;
      }
      case TREE_BLOCK: {
        omega::CG_suifRepr *r = new omega::CG_suifRepr(static_cast<tree_block *>(tn)->body());
        std::vector<IR_ScalarRef *> a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
        break;
      }
      case TREE_INSTR: {
        omega::CG_suifRepr *r = new omega::CG_suifRepr(operand(static_cast<tree_instr *>(tn)->instr()));
        std::vector<IR_ScalarRef *> a = FindScalarRef(r);
        delete r;
        std::copy(a.begin(), a.end(), back_inserter(scalars));
        break;
      }
      default:
        throw ir_error("control structure not supported");
      }
    }
  }
  else {
    operand op = static_cast<const omega::CG_suifRepr *>(repr)->GetExpression();
    if (op.is_instr()) {
      instruction *ins = op.instr();
      for (int i = 0; i < ins->num_srcs(); i++) {
        operand op = ins->src_op(i);
        if (op.is_symbol())
          scalars.push_back(new IR_suifScalarRef(this, ins, i));
        else if (op.is_instr()) {
          omega::CG_suifRepr *r = new omega::CG_suifRepr(op);
          std::vector<IR_ScalarRef *> a = FindScalarRef(r);
          delete r;
          std::copy(a.begin(), a.end(), back_inserter(scalars));
        }
      }
      
      operand op = ins->dst_op();
      if (op.is_symbol())
        scalars.push_back(new IR_suifScalarRef(this, ins, -1));
    }
    else if (op.is_symbol())
      scalars.push_back(new IR_suifScalarRef(this, op.symbol()));
  }
  
  return scalars;
}


std::vector<IR_Control *> IR_suifCode::FindOneLevelControlStructure(const IR_Block *block) const {
  std::vector<IR_Control *> controls;
  
  IR_suifBlock *l_block = static_cast<IR_suifBlock *>(const_cast<IR_Block *>(block));
  tree_node_list_iter iter(l_block->tnl_);
  while(!iter.is_empty()) {
    tree_node *tn = iter.peek();
    if (tn->list_e() == l_block->start_)
      break;
    iter.step();
  }
  tree_node_list_e *start = NULL;
  tree_node_list_e *prev = NULL;
  while (!iter.is_empty()) {
    tree_node *tn = iter.step();
    if (tn->kind() == TREE_FOR) {
      if (start != NULL) {
        controls.push_back(new IR_suifBlock(this, l_block->tnl_, start, prev));
        start = NULL;
      }
      controls.push_back(new IR_suifLoop(this, static_cast<tree_for *>(tn)));
    }
    else if (tn->kind() == TREE_IF) {
      if (start != NULL) {
        controls.push_back(new IR_suifBlock(this, l_block->tnl_, start, prev));
        start = NULL;
      }
      controls.push_back(new IR_suifIf(this, static_cast<tree_if *>(tn)));
    }
    else if (start == NULL && !is_null_statement(tn)) {
      start = tn->list_e();
    }
    prev = tn->list_e();
    if (prev == l_block->end_)
      break;
  }
  
  if (start != NULL && start != l_block->start_)
    controls.push_back(new IR_suifBlock(this, l_block->tnl_, start, prev));
  
  return controls;
}


IR_Block *IR_suifCode::MergeNeighboringControlStructures(const std::vector<IR_Control *> &controls) const {
  if (controls.size() == 0)
    return NULL;
  
  tree_node_list *tnl = NULL;
  tree_node_list_e *start, *end;
  for (int i = 0; i < controls.size(); i++) {
    switch (controls[i]->type()) {
    case IR_CONTROL_LOOP: {
      tree_for *tf = static_cast<IR_suifLoop *>(controls[i])->tf_;
      if (tnl == NULL) {
        tnl = tf->parent();
        start = end = tf->list_e();
      }
      else {
        if (tnl != tf->parent())
          throw ir_error("controls to merge not at the same level");
        end = tf->list_e();
      }
      break;
    }
    case IR_CONTROL_BLOCK: {
      if (tnl == NULL) {
        tnl = static_cast<IR_suifBlock *>(controls[0])->tnl_;
        start = static_cast<IR_suifBlock *>(controls[0])->start_;
        end = static_cast<IR_suifBlock *>(controls[0])->end_;
      }
      else {
        if (tnl != static_cast<IR_suifBlock *>(controls[0])->tnl_)
          throw ir_error("controls to merge not at the same level");
        end = static_cast<IR_suifBlock *>(controls[0])->end_;
      }
      break;
    }
    default:
      throw ir_error("unrecognized control to merge");
    }
  }
  
  return new IR_suifBlock(controls[0]->ir_, tnl, start, end);
}


IR_Block *IR_suifCode::GetCode() const {
  return new IR_suifBlock(this, psym_->block()->body());
}


void IR_suifCode::ReplaceCode(IR_Control *old, omega::CG_outputRepr *repr) {
  tree_node_list *tnl = static_cast<omega::CG_suifRepr *>(repr)->GetCode();
  
  switch (old->type()) {
  case IR_CONTROL_LOOP: {
    tree_for *tf_old = static_cast<IR_suifLoop *>(old)->tf_;
    tree_node_list *tnl_old = tf_old->parent();
    
    tnl_old->insert_before(tnl, tf_old->list_e());
    tnl_old->remove(tf_old->list_e());
    delete tf_old;
    
    break;
  }
  case IR_CONTROL_BLOCK: {
    IR_suifBlock *sb = static_cast<IR_suifBlock *>(old);
    tree_node_list_iter iter(sb->tnl_);
    bool need_deleting = false;
    while (!iter.is_empty()) {
      tree_node *tn = iter.step();
      tree_node_list_e *pos = tn->list_e();
      if (pos == sb->start_) {
        sb->tnl_->insert_before(tnl, pos);
        need_deleting = true;
      }
      if (need_deleting) {
        sb->tnl_->remove(pos);
        delete tn;
      }
      if (pos == sb->end_)
        break;
    }
    
    break;
  }
  default:
    throw ir_error("control structure to be replaced not supported");
  }
  
  delete old;
  delete repr;
}


void IR_suifCode::ReplaceExpression(IR_Ref *old, omega::CG_outputRepr *repr) {
  operand op = static_cast<omega::CG_suifRepr *>(repr)->GetExpression();
  
  if (typeid(*old) == typeid(IR_suifArrayRef)) {
    in_array *ia_orig = static_cast<IR_suifArrayRef *>(old)->ia_;
    
    if (op.is_instr()) {
      instruction *ia_repl = op.instr();
      if (ia_repl->opcode() == io_array) {
        if (ia_orig->elem_type()->is_struct()) {
          static_cast<in_array *>(ia_repl)->set_offset(ia_orig->offset());
          struct_type *tn = static_cast<struct_type *>(ia_orig->elem_type());
          int left;
          type_node *field_tn = tn->field_type(tn->find_field_by_offset(ia_orig->offset(), left));
          static_cast<in_array *>(ia_repl)->set_result_type(field_tn->ptr_to());
        }
        replace_instruction(ia_orig, ia_repl);
        delete ia_orig;
      }
      else {
        instruction *parent_instr = ia_orig->dst_op().instr();
        if (parent_instr->opcode() == io_str) {
          throw ir_error("replace left hand arrary reference not supported yet");
        }
        else if (parent_instr->opcode() == io_lod) {
          instruction *instr = parent_instr->dst_op().instr();
          if (instr->dst_op() == operand(parent_instr)) {
            parent_instr->remove();
            instr->set_dst(op);
          }
          else {
            for (int i = 0; i < instr->num_srcs(); i++)
              if (instr->src_op(i) == operand(parent_instr)) {
                parent_instr->remove();
                instr->set_src_op(i, op);
                break;
              }
          }
          
          delete parent_instr;
        }
        else
          throw ir_error("array reference to be replaced does not appear in any instruction");
      }
    }
    else if (op.is_symbol()) {
      var_sym *vs = op.symbol();
      instruction *parent_instr = ia_orig->dst_op().instr();
      if (parent_instr->opcode() == io_str) {
        tree_node *tn = parent_instr->parent();
        operand op = parent_instr->src_op(1).clone();
        instruction *new_instr = new in_rrr(io_cpy, vs->type(), operand(vs), op);
        tree_node_list *tnl = tn->parent();
        tnl->insert_before(new tree_instr(new_instr), tn->list_e());
        tnl->remove(tn->list_e());
        
        delete tn;
      }
      else if (parent_instr->opcode() == io_lod) {
        instruction *instr = parent_instr->dst_op().instr();
        if (instr->dst_op() == operand(parent_instr)) {
          parent_instr->remove();
          instr->set_dst(operand(vs));
        }
        else {
          for (int i = 0; i < instr->num_srcs(); i++)
            if (instr->src_op(i) == operand(parent_instr)) {
              parent_instr->remove();
              instr->set_src_op(i, operand(vs));
              break;
            }
        }
        
        delete parent_instr;
      }
      else
        throw ir_error("array reference to be replaced does not appear in any instruction");
    }
    else
      throw ir_error("can't handle replacement expression");
  }
  else
    throw ir_error("replacing a scalar variable not implemented");
  
  delete old;
  delete repr;
}



IR_OPERATION_TYPE IR_suifCode::QueryExpOperation(const omega::CG_outputRepr *repr) const {
  operand op = static_cast<const omega::CG_suifRepr *>(repr)->GetExpression();
  
  if (op.is_immed())
    return IR_OP_CONSTANT;
  else if (op.is_symbol())
    return IR_OP_VARIABLE;
  else if (op.is_instr()) {
    instruction *ins = op.instr();
    switch (ins->opcode()) {
    case io_cpy:
      return IR_OP_ASSIGNMENT;
    case io_add:
      return IR_OP_PLUS;
    case io_sub:
      return IR_OP_MINUS;
    case io_mul:
      return IR_OP_MULTIPLY;
    case io_div:
      return IR_OP_DIVIDE;
    case io_neg:
      return IR_OP_NEGATIVE;
    case io_min:
      return IR_OP_MIN;
    case io_max:
      return IR_OP_MAX;
    case io_cvt:
      return IR_OP_POSITIVE;
    default:
      return IR_OP_UNKNOWN;
    }
  }
  else if (op.is_null())
    return IR_OP_NULL;
  else
    return IR_OP_UNKNOWN;
}


IR_CONDITION_TYPE IR_suifCode::QueryBooleanExpOperation(const omega::CG_outputRepr *repr) const {
  operand op = static_cast<const omega::CG_suifRepr *>(repr)->GetExpression();
  if (op.is_instr()) {
    instruction *ins = op.instr();
    switch (ins->opcode()) {
    case io_seq:
      return IR_COND_EQ;
    case io_sne:
      return IR_COND_NE;
    case io_sl:
      return IR_COND_LT;
    case io_sle:
      return IR_COND_LE;
    default:
      return IR_COND_UNKNOWN;
    }
  }
  else
    return IR_COND_UNKNOWN;
}


std::vector<omega::CG_outputRepr *> IR_suifCode::QueryExpOperand(const omega::CG_outputRepr *repr) const {
  std::vector<omega::CG_outputRepr *> v;
  
  operand op = static_cast<const omega::CG_suifRepr *>(repr)->GetExpression();
  if (op.is_immed() || op.is_symbol()) {
    omega::CG_suifRepr *repr = new omega::CG_suifRepr(op);
    v.push_back(repr);
  }
  else if (op.is_instr()) {
    instruction *ins = op.instr();
    omega::CG_suifRepr *repr;
    operand op1, op2;
    switch (ins->opcode()) {
    case io_cpy:
    case io_neg:
    case io_cvt:
      op1 = ins->src_op(0);
      repr = new omega::CG_suifRepr(op1);
      v.push_back(repr);
      break;
    case io_add:
    case io_sub:
    case io_mul:
    case io_div:
    case io_min:
    case io_max:
      op1 = ins->src_op(0);
      repr = new omega::CG_suifRepr(op1);
      v.push_back(repr);
      op2 = ins->src_op(1);
      repr = new omega::CG_suifRepr(op2);
      v.push_back(repr);
      break;
    case io_seq:
    case io_sne:
    case io_sl:
    case io_sle:
      op1 = ins->src_op(0);
      repr = new omega::CG_suifRepr(op1);
      v.push_back(repr);
      op2 = ins->src_op(1);
      repr = new omega::CG_suifRepr(op2);
      v.push_back(repr);
      break;
    default:
      throw ir_error("operation not supported");
    }
  }
  else
    throw ir_error("operand type not supported");
  
  return v;
}


// IR_Constant *IR_suifCode::QueryExpConstant(const CG_outputRepr *repr) const {
//   CG_suifRepr *l_repr = static_cast<CG_suifRepr *>(const_cast<CG_outputRepr *>(repr));

//   operand op = l_repr->GetExpression();
//   if (op.is_immed()) {
//     immed im = op.immediate();

//     switch (im.kind()) {
//     case im_int:
//       return new IR_suifConstant(this, static_cast<coef_t>(im.integer()));
//     case im_extended_int:
//       return new IR_suifConstant(this, static_cast<coef_t>(im.long_int()));
//     case im_float:
//       return new IR_suifConstant(this, im.flt());
//     default:
//       assert(-1);
//     }
//   }
//   else
//     assert(-1);
// }


// IR_ScalarRef *IR_suifCode::QueryExpVariable(const CG_outputRepr *repr) const {
//   CG_suifRepr *l_repr = static_cast<CG_suifRepr *>(const_cast<CG_outputRepr *>(repr));

//   operand op = l_repr->GetExpression();
//   if (op.is_symbol())
//     return new IR_suifScalarRef(this, op.symbol());
//   else
//     assert(-1);
// }


IR_Ref *IR_suifCode::Repr2Ref(const omega::CG_outputRepr *repr) const {
  operand op = static_cast<const omega::CG_suifRepr *>(repr)->GetExpression();
  if (op.is_immed()) {
    immed im = op.immediate();
    
    switch (im.kind()) {
    case im_int:
      return new IR_suifConstantRef(this, static_cast<omega::coef_t>(im.integer()));
    case im_extended_int:
      return new IR_suifConstantRef(this, static_cast<omega::coef_t>(im.long_int()));
    case im_float:
      return new IR_suifConstantRef(this, im.flt());
    default:
      throw ir_error("immediate value not integer or floatint point");
    }
  }
  else if (op.is_symbol())
    return new IR_suifScalarRef(this, op.symbol());
  else
    throw ir_error("unrecognized reference type");
}
