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
#include "ir_cudarose.hh"
#include "loop.hh"
#include "loop_cuda_rose.hh"
//#include "ir_suif_utils.hh"

using namespace SageBuilder;
using namespace SageInterface;

IR_cudaroseCode::IR_cudaroseCode(const char *filename, const char* proc_name) :
  IR_roseCode(filename, proc_name) {
  
  //std::string file_suffix = StringUtility::fileNameSuffix(filename);
  
  //if (CommandlineProcessing::isCFileNameSuffix(file_suffix))
  //{
  std::string orig_name = StringUtility::stripPathFromFileName(filename);
  std::string naked_name = StringUtility::stripFileSuffixFromFileName(
    orig_name);
  file->set_unparse_output_filename("rose_" + naked_name + ".cu");
  
  //}
  
  gsym_ = root;
  first_scope = firstScope;
  parameter = symtab2_;
  body = symtab3_;
  defn = func->get_definition()->get_body();
  func_defn = func->get_definition();
}



IR_ArraySymbol *IR_cudaroseCode::CreateArraySymbol(const IR_Symbol *sym,
                                                   std::vector<omega::CG_outputRepr *> &size, int sharedAnnotation) {
  SgType *tn;
  SgVariableSymbol* vs;
  if (typeid(*sym) == typeid(IR_roseScalarSymbol)) {
    tn = static_cast<const IR_roseScalarSymbol *>(sym)->vs_->get_type();
  } else if (typeid(*sym) == typeid(IR_roseArraySymbol)) {
    tn = static_cast<const IR_roseArraySymbol *>(sym)->vs_->get_type();
    while (isSgArrayType(tn) || isSgPointerType(tn)) {
      if (isSgArrayType(tn))
        tn = isSgArrayType(tn)->get_base_type();
      else if (isSgPointerType(tn))
        tn = isSgPointerType(tn)->get_base_type();
      else
        throw ir_error(
          "in CreateScalarSymbol: symbol not an array nor a pointer!");
    }
  } else
    throw std::bad_typeid();
  
  for (int i = size.size() - 1; i >= 0; i--)
    tn = buildArrayType(tn,
                        static_cast<omega::CG_roseRepr *>(size[i])->GetExpression());
  
  static int rose_array_counter = 1;
  std::string s = std::string("_P") + omega::to_string(rose_array_counter++);
  SgVariableDeclaration* defn2 = buildVariableDeclaration(
    const_cast<char *>(s.c_str()), tn);
  SgInitializedNamePtrList& variables2 = defn2->get_variables();
  
  SgInitializedNamePtrList::const_iterator i2 = variables2.begin();
  SgInitializedName* initializedName2 = *i2;
  vs = new SgVariableSymbol(initializedName2);
  
  prependStatement(defn2,
                   isSgScopeStatement(func->get_definition()->get_body()));
  
  vs->set_parent(symtab_);
  symtab_->insert(SgName(s.c_str()), vs);
  
  SgStatementPtrList* tnl5 = new SgStatementPtrList;
  
  (*tnl5).push_back(isSgStatement(defn2));
  
  omega::CG_roseRepr* stmt = new omega::CG_roseRepr(tnl5);
  
  init_code_ = ocg_->StmtListAppend(init_code_,
                                    static_cast<omega::CG_outputRepr *>(stmt));
  
  if (sharedAnnotation == 1)
    isSgNode(defn2)->setAttribute("__shared__",
                                  new AstTextAttribute("__shared__"));
  
  return new IR_roseArraySymbol(this, vs);
}

bool IR_cudaroseCode::commit_loop(Loop *loop, int loop_num) {
  if (loop == NULL)
    return true;
  
  LoopCuda *cu_loop = (LoopCuda *) loop;
  SgNode *tnl = cu_loop->codegen();
  if (!tnl)
    return false;
  
  SgStatementPtrList* new_list = NULL;
  if (isSgBasicBlock(tnl)) {
    new_list = new SgStatementPtrList;
    for (SgStatementPtrList::iterator it =
           isSgBasicBlock(tnl)->get_statements().begin();
         it != isSgBasicBlock(tnl)->get_statements().end(); it++)
      (*new_list).push_back(*it);
  }
  
  //Only thing that should be left will be the inserting of the tnl* into the loop
  omega::CG_outputRepr *repr;
  if (new_list == NULL)
    repr = new omega::CG_roseRepr(tnl);
  else
    repr = new omega::CG_roseRepr(new_list);
  if (cu_loop->init_code != NULL)
    repr = ocg_->StmtListAppend(cu_loop->init_code->clone(), repr);
  
  std::vector<SgForStatement *> loops = find_loops(
    func->get_definition()->get_body());
  tnl = isSgNode(loops[loop_num])->get_parent();
  
  if (cu_loop->setup_code != NULL) {
    SgStatementPtrList* setup_tnl =
      static_cast<omega::CG_roseRepr *>(cu_loop->setup_code)->GetList();
    
    SgStatement* target = isSgStatement(loops[loop_num]);
    
    for (SgStatementPtrList::iterator it = (*setup_tnl).begin();
         it != (*setup_tnl).end(); it++) {
      
      isSgStatement(tnl)->insert_statement(target, *it, false);
      isSgNode(*it)->set_parent(tnl);
      target = *it;
    }
    
    //SgStatementPtrList
    // for SgStatementPtrList::it
    //TODO: I think this is a hack we can undo if we have loop->codegen()
    //loo->getCode(), maybe also get rid of setup and teardown...
    //fix_unfinished_comment(setup_tnl, indexes_string);
    //isSgStatement(tnl)->replace_statement(isSgStatement(loops[loop_num]), *setup_tnl);
    isSgStatement(tnl)->remove_statement(isSgStatement(loops[loop_num]));
  }
  
  delete repr;
  
  return true;
}

IR_cudaroseCode::~IR_cudaroseCode() {
}

