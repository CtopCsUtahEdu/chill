/*****************************************************************************
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
   CHiLL's SUIF interface.

 Notes:
   Array supports mixed pointer and array type in a single declaration.

 History:
   2/2/2011 Created by Protonu Basu. 
*****************************************************************************/

#include <typeinfo>
#include "ir_cudasuif.hh"
#include "loop.hh"
#include "loop_cuda.hh"
#include "ir_suif_utils.hh"


IR_cudasuifCode::IR_cudasuifCode(const char *filename, int proc_num)
  :IR_suifCode(filename, proc_num)
{
  //setting up gsym_ here
  fileset->reset_iter();
  gsym_ = fileset->globals();
  
}



IR_ArraySymbol *IR_cudasuifCode::CreateArraySymbol(const IR_Symbol *sym,
                                                   std::vector<omega::CG_outputRepr *> &size,
                                                   int sharedAnnotation)
{
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
      init_code_ = ocg_->StmtListAppend(init_code_, ocg_->StmtListAppend(ocg_->CreateAssignment(0, new omega::CG_suifRepr(operand(temporary)), size[i]),NULL));
      
      tn = new array_type(tn, array_bound(1), array_bound(temporary));
      symtab_->add_type(tn);
    }
  else     
    for (int i = size.size()-1; i >= 0; i--) {
      var_sym *temporary = symtab_->new_unique_var(type_s32);
      //init_code_ = ocg_->StmtListAppend(init_code_, ocg_->CreateStmtList(ocg_->CreateAssignment(0, new omega::CG_suifRepr(operand(temporary)), size[i])));
      init_code_ = ocg_->StmtListAppend(init_code_, ocg_->StmtListAppend(ocg_->CreateAssignment(0, new omega::CG_suifRepr(operand(temporary)), size[i]), NULL));
      
      tn = new array_type(tn, array_bound(1), array_bound(temporary));
      symtab_->add_type(tn);
      if(i == 0 && sharedAnnotation == 1){
        tn = static_cast<omega::CG_suifBuilder*>(ocg_)->ModifyType(tn, "__shared__");
        symtab_->add_type(tn);
      }
    }
  
  static int suif_array_counter = 1;
  std::string s = std::string("_P") + omega::to_string(suif_array_counter++);
  var_sym *vs = new var_sym(tn, const_cast<char *>(s.c_str()));
  vs->add_to_table(symtab_);
  
  return new IR_suifArraySymbol(this, vs);
}


bool IR_cudasuifCode::commit_loop(Loop *loop, int loop_num) {  
  if (loop == NULL)
    return true;
  
  //Call code-gen part of any scripting routines that were run.
  // internally call GetCode
  // Add stuff before and after (setup, teardown
  // return a tnl
  LoopCuda *cu_loop = (LoopCuda *)loop;
  tree_node_list *tnl = cu_loop->codegen();
  if(!tnl)
    return false;
  
  //set up our new procs
  for(int i=0; i<cu_loop->new_procs.size(); i++)
  {
    printf("setting proc fse\n");
    cu_loop->new_procs[i]->set_fse(fse_);
    write_procs.push_back(cu_loop->new_procs[i]);
  }
  
  //Only thing that should be left will be the inserting of the tnl* into the loop
  
  omega::CG_outputRepr *repr = new omega::CG_suifRepr(tnl);
  if (cu_loop->init_code != NULL)
    repr = ocg_->StmtListAppend(cu_loop->init_code->clone(), repr);
  
  std::vector<tree_for *> loops = find_loops(psym_->block()->body());
  tnl = loops[loop_num]->parent();
  
  if (cu_loop->setup_code != NULL) {
    tree_node_list *setup_tnl = static_cast<omega::CG_suifRepr *>(cu_loop->setup_code->clone())->GetCode();
    //TODO: I think this is a hack we can undo if we have loop->codegen()
    //loo->getCode(), maybe also get rid of setup and teardown...
    //fix_unfinished_comment(setup_tnl, indexes_string);
    tnl->insert_before(setup_tnl, loops[loop_num]->list_e());
  }
  tnl->insert_before(static_cast<omega::CG_suifRepr *>(repr)->GetCode(), loops[loop_num]->list_e());
  if (cu_loop->teardown_code != NULL) {
    tree_node_list *setup_tnl = static_cast<omega::CG_suifRepr *>(cu_loop->teardown_code->clone())->GetCode();
    tnl->insert_before(setup_tnl, loops[loop_num]->list_e());
  }
  
  tnl->remove(loops[loop_num]->list_e());
  
  delete repr;
  return true;
}

IR_cudasuifCode::~IR_cudasuifCode()
{
  for(int i=0; i<write_procs.size(); i++)
  {
    if (!write_procs[i]->is_written())
      write_procs[i]->write_proc(fse_);
    write_procs[i]->flush_proc();
  }
}
