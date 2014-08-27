/*****************************************************************************
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
 Cudaize methods

 Notes:

 History:
 1/7/10 Created by Gabe Rudy by migrating code from loop.cc
 31/1/11 Modified by Protonu Basu
*****************************************************************************/
#define TRANSFORMATION_FILE_INFO Sg_File_Info::generateDefaultFileInfoForTransformationNode()
#include <code_gen/CG_stringBuilder.h>
#include <codegen.h>
#include <code_gen/CG_utils.h>
#include <code_gen/CG_outputRepr.h>
#include "loop_cuda_rose.hh"
#include "loop.hh"
#include <math.h>
//#include <useful.h>
#include "omegatools.hh"
#include "ir_cudarose.hh"
#include "ir_rose.hh"
#include "ir_rose_utils.hh"
#include "chill_error.hh"
#include <vector>
#include "Outliner.hh"
//#define DEBUG
using namespace omega;
using namespace SageBuilder;
using namespace SageInterface;
//using namespace Outliner;
//using namespace ASTtools;
char *k_cuda_texture_memory; //protonu--added to track texture memory type
//extern char *omega::k_cuda_texture_memory; //protonu--added to track texture memory type
extern char *omega::k_ocg_comment;

static int cudaDebug;
class CudaStaticInit {
public:
  CudaStaticInit() {
    cudaDebug = 0; //Change this to 1 for debug
  }
};
static CudaStaticInit junkInitInstance__;

std::string& upcase(std::string& s) {
  for (int i = 0; i < s.size(); i++)
    s[i] = toupper(s[i]);
  return s;
}

void printVs(const std::vector<std::string>& curOrder) {
  if (!cudaDebug) return;
  for (int i = 0; i < curOrder.size(); i++) {
    if (i > 0)
      printf(",");
    printf("%s", curOrder[i].c_str());
  }
  printf("\n");
}

void printVS(const std::vector<std::string>& curOrder) {
  if(!cudaDebug) return;
  for (int i = 0; i < curOrder.size(); i++) {
    if (i > 0)
      printf(",");
    printf("%s", curOrder[i].c_str());
  }
  printf("\n");
}

LoopCuda::~LoopCuda() {
  const int m = stmt.size();
  for (int i = 0; i < m; i++)
    stmt[i].code->clear();
}

bool LoopCuda::symbolExists(std::string s) {
  
  if (body_symtab->find_variable(SgName(s.c_str()))
      || parameter_symtab->find_variable(SgName(s.c_str())))
    return true;
  if (globals->lookup_variable_symbol(SgName(s.c_str())))
    return true;
  for (int i = 0; i < idxNames.size(); i++)
    for (int j = 0; j < idxNames[i].size(); j++)
      if (strcmp(idxNames[i][j].c_str(), s.c_str()) == 0)
        return true;
  return false;
}

void LoopCuda::addSync(int stmt_num, std::string idxName) {
  //we store these and code-gen inserts sync to omega comments where stmt
  //in loop that has idxName being generated
  syncs.push_back(make_pair(stmt_num, idxName));
}

void LoopCuda::renameIndex(int stmt_num, std::string idx, std::string newName) {
  int level = findCurLevel(stmt_num, idx);
  if (idxNames.size() <= stmt_num || idxNames[stmt_num].size() < level)
    throw std::runtime_error("Invalid statment number of index");
  idxNames[stmt_num][level - 1] = newName.c_str();
}

enum Type {
  Int
};

SgNode* wrapInIfFromMinBound(SgNode* then_part, SgForStatement* loop,
                             SgScopeStatement* symtab, SgVariableSymbol* bound_sym) {
  // CG_roseBuilder *ocg = new CG_roseBuilder(
  
  SgBinaryOp* test_expr = isSgBinaryOp(loop->get_test_expr());
  SgExpression* upperBound;
  SgExpression* conditional;
  upperBound = test_expr->get_rhs_operand();
  CG_outputRepr *ifstmt;
  
  SgCallExpression *call;
  if (call = isSgCallExpression(upperBound))
    if (isSgVarRefExp(call->get_function())->get_symbol()->get_name().getString()
        == "__rose_lt") {
      SgExprListExp* arg_list = call->get_args();
      SgExpression *if_bound = *(arg_list->get_expressions().begin());
      /*This relies on the minimum expression being the rhs operand of
       * the min instruction.
       */
      SgIfStmt *ifstmt = buildIfStmt(
        buildLessOrEqualOp(buildVarRefExp(bound_sym), if_bound),
        isSgStatement(then_part), NULL);
      return isSgNode(ifstmt);
      
    }
  
/*  if (isSgConditionalExp(upperBound)) {
    conditional = isSgConditionalExp(upperBound)->get_conditional_exp();
    
    if (isSgBinaryOp(conditional)) {
    SgBinaryOp* binop = isSgBinaryOp(conditional);
    
    if (isSgLessThanOp(binop) || isSgLessOrEqualOp(binop)) {
    SgIfStmt *ifstmt = buildIfStmt(
    buildLessOrEqualOp(buildVarRefExp(bound_sym),
    test_expr), isSgStatement(then_part), NULL);
    return isSgNode(ifstmt);
    }
    
    }
    
    }
*/
  return then_part;
}

/**
 * This would be better if it was done by a CHiLL xformation instead of at codegen
 *
 * state:
 * for(...)
 *   for(...)
 *     cur_body
 *   stmt1
 *
 * stm1 is in-between two loops that are going to be reduced. The
 * solution is to put stmt1 at the end of cur_body but conditionally run
 * in on the last step of the for loop.
 *
 * A CHiLL command that would work better:
 *
 * for(...)
 *   stmt0
 *   for(for i=0; i<n; i++)
 *     cur_body
 *   stmt1
 * =>
 * for(...)
 *   for(for i=0; i<n; i++)
 *     if(i==0) stmt0
 *     cur_body
 *     if(i==n-1) stmt1
 */

std::vector<SgForStatement*> findCommentedFors(const char* index, SgNode* tnl) {
  std::vector<SgForStatement *> result;
  bool next_loop_ok = false;
  
  if (isSgBasicBlock(tnl)) {
    
    SgStatementPtrList& list = isSgBasicBlock(tnl)->get_statements();
    
    for (SgStatementPtrList::iterator it = list.begin(); it != list.end();
         it++) {
      std::vector<SgForStatement*> t = findCommentedFors(index,
                                                         isSgNode(*it));
      std::copy(t.begin(), t.end(), back_inserter(result));
    }
  } else if (isSgForStatement(tnl)) {
    
    AstTextAttribute* att =
      (AstTextAttribute*) (isSgNode(tnl)->getAttribute(
                             "omega_comment"));
    std::string comment = att->toString();
    
    if (comment.find("~cuda~") != std::string::npos
        && comment.find("preferredIdx: ") != std::string::npos) {
      std::string idx = comment.substr(
        comment.find("preferredIdx: ") + 14, std::string::npos);
      if (idx.find(" ") != std::string::npos)
        idx = idx.substr(0, idx.find(" "));
      if (strcmp(idx.c_str(), index) == 0)
        next_loop_ok = true;
    }
    
    if (next_loop_ok) {
      //printf("found loop %s\n", static_cast<tree_for *>(tn)->index()->name());
      result.push_back(isSgForStatement(tnl));
    } else {
      //printf("looking down for loop %s\n", static_cast<tree_for *>(tn)->index()->name());
      std::vector<SgForStatement*> t = findCommentedFors(index,
                                                         isSgForStatement(tnl)->get_loop_body());
      std::copy(t.begin(), t.end(), back_inserter(result));
    }
    next_loop_ok = false;
  } else if (isSgIfStmt(tnl)) {
    //printf("looking down if\n");
    SgIfStmt *tni = isSgIfStmt(tnl);
    std::vector<SgForStatement*> t = findCommentedFors(index,
                                                       tni->get_true_body());
    std::copy(t.begin(), t.end(), back_inserter(result));
  }
  
  return result;
}

SgNode* forReduce(SgForStatement* loop, SgVariableSymbol* reduceIndex,
                  SgScopeStatement* body_syms) {
  //We did the replacements all at once with recursiveFindPreferedIdxs
  //replacements r;
  //r.oldsyms.append(loop->index());
  //r.newsyms.append(reduceIndex);
  //tree_for* new_loop = (tree_for*)loop->clone_helper(&r, true);
  SgForStatement* new_loop = loop;
  
  //return body one loops in
  SgNode* tnl = loop_body_at_level(new_loop, 1);
  //wrap in conditional if necessary
  tnl = wrapInIfFromMinBound(tnl, new_loop, body_syms, reduceIndex);
  return tnl;
}

void recursiveFindRefs(SgNode* code, std::set<const SgVariableSymbol *>& syms,
                       SgFunctionDefinition* def) {
  
  SgStatement* s = isSgStatement(code);
  // L = {symbols defined within 's'}, local variables declared within 's'
  ASTtools::VarSymSet_t L;
  ASTtools::collectDefdVarSyms(s, L);
  //dump (L, "L = ");
  
  // U = {symbols used within 's'}
  ASTtools::VarSymSet_t U;
  ASTtools::collectRefdVarSyms(s, U);
  //dump (U, "U = ");
  
  // U - L = {symbols used within 's' but not defined in 's'}
  // variable references to non-local-declared variables
  ASTtools::VarSymSet_t diff_U_L;
  set_difference(U.begin(), U.end(), L.begin(), L.end(),
                 inserter(diff_U_L, diff_U_L.begin()));
  //dump (diff_U_L, "U - L = ");
  
  // Q = {symbols defined within the function surrounding 's' that are
  // visible at 's'}, including function parameters
  ASTtools::VarSymSet_t Q;
  ASTtools::collectLocalVisibleVarSyms(def->get_declaration(), s, Q);
//    dump (Q, "Q = ");
  
  // (U - L) \cap Q = {variables that need to be passed as parameters
  // to the outlined function}
  // a sub set of variables that are not globally visible (no need to pass at all)
  // It excludes the variables with a scope between global and the enclosing function
  set_intersection(diff_U_L.begin(), diff_U_L.end(), Q.begin(), Q.end(),
                   inserter(syms, syms.begin()));
  
  /* std::vector<SgVariableSymbol *> scalars;
  //SgNode  *tnl = static_cast<const omega::CG_roseRepr *>(repr)->GetCode();
  SgStatement* stmt;
  SgExpression* exp;
  if (tnl != NULL) {
  if(stmt = isSgStatement(tnl)){
  if(isSgBasicBlock(stmt)){
  SgStatementPtrList& stmts = isSgBasicBlock(stmt)->get_statements();
  for(int i =0; i < stmts.size(); i++){
  //omega::CG_roseRepr *r = new omega::CG_roseRepr(isSgNode(stmts[i]));
  std::vector<SgVariableSymbol *> a = recursiveFindRefs(isSgNode(stmts[i]));
  //delete r;
  std::copy(a.begin(), a.end(), back_inserter(scalars));
  }
  
  }
  else if(isSgForStatement(stmt)){
  
  SgForStatement *tnf =  isSgForStatement(stmt);
  //omega::CG_roseRepr *r = new omega::CG_roseRepr(isSgStatement(tnf->get_loop_body()));
  std::vector<SgVariableSymbol *> a = recursiveFindRefs(isSgNode(tnf->get_loop_body()));
  //delete r;
  std::copy(a.begin(), a.end(), back_inserter(scalars));
  }
  else if(isSgFortranDo(stmt)){
  SgFortranDo *tfortran =  isSgFortranDo(stmt);
  omega::CG_roseRepr *r = new omega::CG_roseRepr(isSgStatement(tfortran->get_body()));
  std::vector<SgVariableSymbol *> a = recursiveFindRefs(r);
  delete r;
  std::copy(a.begin(), a.end(), back_inserter(scalars));
  }
  
  else if(isSgIfStmt(stmt) ){
  SgIfStmt* tni = isSgIfStmt(stmt);
  //omega::CG_roseRepr *r = new omega::CG_roseRepr(isSgNode(tni->get_conditional()));
  std::vector<SgVariableSymbol *> a = recursiveFindRefs(isSgNode(tni->get_conditional()));
  //delete r;
  std::copy(a.begin(), a.end(), back_inserter(scalars));
  //r = new omega::CG_roseRepr(isSgNode(tni->get_true_body()));
  a = recursiveFindRefs(isSgNode(tni->get_true_body()));
  //delete r;
  std::copy(a.begin(), a.end(), back_inserter(scalars));
  //r = new omega::CG_roseRepr(isSgNode(tni->get_false_body()));
  a = recursiveFindRefs(isSgNode(tni->get_false_body()));
  //delete r;
  std::copy(a.begin(), a.end(), back_inserter(scalars));
  }
  else if(isSgExprStatement(stmt)) {
  //omega::CG_roseRepr *r = new omega::CG_roseRepr(isSgExpression(isSgExprStatement(stmt)->get_expression()));
  std::vector<SgVariableSymbol *> a = recursiveFindRefs(isSgNode(isSgExprStatement(stmt)->get_expression()));
  //delete r;
  std::copy(a.begin(), a.end(), back_inserter(scalars));
  
  }
  }
  }
  else{
  SgExpression* op = isSgExpression(tnl);
  if(isSgVarRefExp(op)){
  
  scalars.push_back(isSgVarRefExp(op)->get_symbol());
  
  }
  else if( isSgAssignOp(op)){
  //omega::CG_roseRepr *r1 = new omega::CG_roseRepr(isSgAssignOp(op)->get_lhs_operand());
  std::vector<SgVariableSymbol *> a1 = recursiveFindRefs(isSgNode(isSgAssignOp(op)->get_lhs_operand()));
  //delete r1;
  std::copy(a1.begin(), a1.end(), back_inserter(scalars));
  //omega::CG_roseRepr *r2 = new omega::CG_roseRepr(isSgAssignOp(op)->get_rhs_operand());
  std::vector<SgVariableSymbol *> a2 = recursiveFindRefs(isSgNode(isSgAssignOp(op)->get_rhs_operand()));
  //delete r2;
  std::copy(a2.begin(), a2.end(), back_inserter(scalars));
  
  }
  else if(isSgBinaryOp(op)){
  // omega::CG_roseRepr *r1 = new omega::CG_roseRepr(isSgBinaryOp(op)->get_lhs_operand());
  std::vector<SgVariableSymbol *> a1 = recursiveFindRefs(isSgNode(isSgBinaryOp(op)->get_lhs_operand()));
  //delete r1;
  std::copy(a1.begin(), a1.end(), back_inserter(scalars));
  //omega::CG_roseRepr *r2 = new omega::CG_roseRepr(isSgBinaryOp(op)->get_rhs_operand());
  std::vector<SgVariableSymbol *> a2 = recursiveFindRefs((isSgBinaryOp(op)->get_rhs_operand()));
  //delete r2;
  std::copy(a2.begin(), a2.end(), back_inserter(scalars));
  }
  else if(isSgUnaryOp(op)){
  //omega::CG_roseRepr *r1 = new omega::CG_roseRepr(isSgUnaryOp(op)->get_operand());
  std::vector<SgVariableSymbol *> a1 = recursiveFindRefs(isSgNode(isSgUnaryOp(op)->get_operand()));
  //delete r1;
  std::copy(a1.begin(), a1.end(), back_inserter(scalars));
  }
  
  }
  return scalars;
  
  
  */
  
}

SgNode* recursiveFindReplacePreferedIdxs(SgNode* code, SgSymbolTable* body_syms,
                                         SgSymbolTable* param_syms, SgScopeStatement* body,
                                         std::map<std::string, SgVariableSymbol*>& loop_idxs,
                                         SgGlobal* globalscope, bool sync = false) {
  //tree_node_list* tnl = new tree_node_list;
  //tree_node_list_iter tnli(code);
  SgVariableSymbol* idxSym = 0;
  std::vector<SgStatement*> r1;
  std::vector<SgNode*> r2;
  SgNode* tnli;
  SgNode* tnli1;
  SgNode* tnli2;
  SgBasicBlock * clone;
  
  if (isSgForStatement(code)) {
    AstTextAttribute* att =
      (AstTextAttribute*) (isSgNode(code)->getAttribute(
                             "omega_comment"));
    
    std::string comment;
    if (att != NULL)
      comment = att->toString();
    
    if (comment.find("~cuda~") != std::string::npos
        && comment.find("preferredIdx: ") != std::string::npos) {
      std::string idx = comment.substr(
        comment.find("preferredIdx: ") + 14, std::string::npos);
      if (idx.find(" ") != std::string::npos)
        idx = idx.substr(0, idx.find(" "));
      if (loop_idxs.find(idx) != loop_idxs.end())
        idxSym = loop_idxs.find(idx)->second;
      //Get the proc variable sybol for this preferred index
      if (idxSym == 0) {
        idxSym = body_syms->find_variable(idx.c_str());
        if (!idxSym)
          idxSym = param_syms->find_variable(idx.c_str());
        //printf("idx not found: lookup %p\n", idxSym);
        if (!idxSym) {
          SgVariableDeclaration* defn = buildVariableDeclaration(
            SgName((char*) idx.c_str()), buildIntType());
          //idxSym = new var_sym(type_s32, (char*)idx.c_str());
          SgInitializedNamePtrList& variables = defn->get_variables();
          SgInitializedNamePtrList::const_iterator i =
            variables.begin();
          SgInitializedName* initializedName = *i;
          SgVariableSymbol* vs = new SgVariableSymbol(
            initializedName);
          prependStatement(defn, body);
          vs->set_parent(body_syms);
          body_syms->insert(SgName((char*) idx.c_str()), vs);
          idxSym = vs;
          //printf("idx created and inserted\n");
        }
        //Now insert into our map for future
	if (cudaDebug)
	    std::cout << idx << "\n\n";
        loop_idxs.insert(make_pair(idx, idxSym));
      }
      //See if we have a sync as well
      if (comment.find("sync") != std::string::npos) {
        //printf("Inserting sync after current block\n");
        sync = true;
      }
      
    }
    if (idxSym) {
      SgForInitStatement* list =
        isSgForStatement(code)->get_for_init_stmt();
      SgStatementPtrList& initStatements = list->get_init_stmt();
      SgStatementPtrList::const_iterator j = initStatements.begin();
      const SgVariableSymbol* index;
      
      if (SgExprStatement *expr = isSgExprStatement(*j))
        if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
          if (SgVarRefExp* var_ref = isSgVarRefExp(
                op->get_lhs_operand()))
            index = var_ref->get_symbol();
      
      std::vector<SgVarRefExp *> array = substitute(code, index, NULL,
                                                    isSgNode(body_syms));
      
      for (int j = 0; j < array.size(); j++)
        array[j]->set_symbol(idxSym);
    }
    
    SgStatement* body_ = isSgStatement(
      recursiveFindReplacePreferedIdxs(
        isSgNode((isSgForStatement(code)->get_loop_body())),
        body_syms, param_syms, body, loop_idxs, globalscope));
    
    omega::CG_roseRepr * tnl = new omega::CG_roseRepr(code);
    omega::CG_outputRepr* block = tnl->clone();
    tnli = static_cast<const omega::CG_roseRepr *>(block)->GetCode();
    
    isSgForStatement(tnli)->set_loop_body(body_);
    body_->set_parent(tnli);
    
    if (idxSym) {
      SgForInitStatement* list =
        isSgForStatement(tnli)->get_for_init_stmt();
      SgStatementPtrList& initStatements = list->get_init_stmt();
      SgStatementPtrList::const_iterator j = initStatements.begin();
      const SgVariableSymbol* index;
      
      if (SgExprStatement *expr = isSgExprStatement(*j))
        if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
          if (SgVarRefExp* var_ref = isSgVarRefExp(
                op->get_lhs_operand()))
            index = var_ref->get_symbol();
      
      std::vector<SgVarRefExp *> array = substitute(tnli, index, NULL,
                                                    isSgNode(body_syms));
      
      for (int j = 0; j < array.size(); j++)
        array[j]->set_symbol(idxSym);
    }
    //  std::cout << isSgNode(body_)->unparseToString() << "\n\n";
    if (att != NULL)
      tnli->setAttribute("omega_comment", att);
    
    if (sync) {
      SgName name_syncthreads("__syncthreads");
      SgFunctionSymbol * syncthreads_symbol =
        globalscope->lookup_function_symbol(name_syncthreads);
      
      // Create a call to __syncthreads():
      SgFunctionCallExp * syncthreads_call = buildFunctionCallExp(
        syncthreads_symbol, buildExprListExp());
      
      SgExprStatement* stmt = buildExprStatement(syncthreads_call);
      
      /*    if (SgBasicBlock* bb = isSgBasicBlock(
            isSgForStatement(code)->get_loop_body()))
            appendStatement(isSgStatement(stmt), bb);
            
            else if (SgStatement* ss = isSgStatement(
            isSgForStatement(code)->get_loop_body())) {
            SgBasicBlock* bb2 = buildBasicBlock();
            
            isSgNode(ss)->set_parent(bb2);
            appendStatement(ss, bb2);
            
            appendStatement(isSgStatement(stmt), bb2);
            isSgNode(stmt)->set_parent(bb2);
            isSgForStatement(code)->set_loop_body(bb2);
            isSgNode(bb2)->set_parent(code);
            }
      */
      
      SgBasicBlock* bb2 = buildBasicBlock();
      
      bb2->append_statement(isSgStatement(tnli));
      bb2->append_statement(stmt);
      /* SgNode* parent = code->get_parent();
         if(!isSgStatement(parent))
         throw loop_error("Parent not a statement");
         
         if(isSgForStatement(parent)){
         if(SgStatement *ss = isSgForStatement(isSgForStatement(parent)->get_loop_body())){
         omega::CG_roseRepr * tnl = new omega::CG_roseRepr(ss);
         omega::CG_outputRepr* block= tnl->clone();
         
         SgNode *new_ss = static_cast<const omega::CG_roseRepr *>(block)->GetCode();
         SgBasicBlock* bb2 = buildBasicBlock();
         
         isSgNode(new_ss)->set_parent(bb2);
         appendStatement(isSgStatement(new_ss), bb2);
         appendStatement(isSgStatement(stmt), bb2);
         isSgNode(stmt)->set_parent(bb2);
         
         isSgStatement(parent)->replace_statement_from_basicBlock(ss, isSgStatement(bb2));
         
         }else if(isSgBasicBlock(isSgForStatement(parent)->get_loop_body()))
         isSgStatement(isSgForStatement(parent)->get_loop_body())->insert_statement(isSgStatement(code), stmt, false);
         else
         throw loop_error("parent statement type undefined!!");
         
         }
         else if(isSgBasicBlock(parent))
         isSgStatement(parent)->insert_statement(isSgStatement(code), stmt, false);
         else
         throw loop_error("parent statement type undefined!!");
         
         //tnl->print();
         *
         *
         */
      sync = true;
      return isSgNode(bb2);
      
    } else
      return tnli;
  } else if (isSgIfStmt(code)) {
    SgStatement* body_ = isSgStatement(
      recursiveFindReplacePreferedIdxs(
        isSgNode((isSgIfStmt(code)->get_true_body())),
        body_syms, param_syms, body, loop_idxs, globalscope));
    
    omega::CG_roseRepr * tnl = new omega::CG_roseRepr(code);
    omega::CG_outputRepr* block = tnl->clone();
    tnli = static_cast<const omega::CG_roseRepr *>(block)->GetCode();
    
    isSgIfStmt(tnli)->set_true_body(body_);
    
    if ((isSgIfStmt(code)->get_false_body()))
      isSgIfStmt(tnli)->set_false_body(
        isSgStatement(
          recursiveFindReplacePreferedIdxs(
            isSgNode(
              (isSgIfStmt(code)->get_false_body())),
            body_syms, param_syms, body, loop_idxs,
            globalscope)));
    
    return tnli;
  } else if (isSgStatement(code) && !isSgBasicBlock(code)) {
    omega::CG_roseRepr * tnl = new omega::CG_roseRepr(code);
    omega::CG_outputRepr* block = tnl->clone();
    tnli = static_cast<const omega::CG_roseRepr *>(block)->GetCode();
    
    return tnli;
    
  } else if (isSgBasicBlock(code)) {
    SgStatementPtrList& tnl = isSgBasicBlock(code)->get_statements();
    
    SgStatementPtrList::iterator temp;
    clone = buildBasicBlock();
    bool sync_found = false;
    for (SgStatementPtrList::const_iterator it = tnl.begin();
         it != tnl.end(); it++) {
      
      if (isSgForStatement(*it)) {
        AstTextAttribute* att =
          (AstTextAttribute*) (isSgNode(*it)->getAttribute(
                                 "omega_comment"));
        
        std::string comment;
        if (att != NULL)
          comment = att->toString();
        
        if (comment.find("~cuda~") != std::string::npos
            && comment.find("preferredIdx: ")
            != std::string::npos) {
          std::string idx = comment.substr(
            comment.find("preferredIdx: ") + 14,
            std::string::npos);
          if (idx.find(" ") != std::string::npos)
            idx = idx.substr(0, idx.find(" "));
          //printf("sym_tab preferred index: %s\n", idx.c_str());
          if (loop_idxs.find(idx) != loop_idxs.end())
            idxSym = loop_idxs.find(idx)->second;
          //Get the proc variable sybol for this preferred index
          if (idxSym == 0) {
            idxSym = body_syms->find_variable(idx.c_str());
            if (!idxSym)
              idxSym = param_syms->find_variable(idx.c_str());
            //printf("idx not found: lookup %p\n", idxSym);
            if (!idxSym) {
              SgVariableDeclaration* defn =
                buildVariableDeclaration(
                  SgName((char*) idx.c_str()),
                  buildIntType());
              //idxSym = new var_sym(type_s32, (char*)idx.c_str());
              SgInitializedNamePtrList& variables =
                defn->get_variables();
              SgInitializedNamePtrList::const_iterator i =
                variables.begin();
              SgInitializedName* initializedName = *i;
              SgVariableSymbol* vs = new SgVariableSymbol(
                initializedName);
              prependStatement(defn, body);
              vs->set_parent(body_syms);
              body_syms->insert(SgName((char*) idx.c_str()), vs);
              //printf("idx created and inserted\n");
              idxSym = vs;
            }
            //Now insert into our map for future
	    if (cudaDebug)
		std::cout << idx << "\n\n";
            loop_idxs.insert(make_pair(idx, idxSym));
            
          }
          //See if we have a sync as well
          if (comment.find("sync") != std::string::npos) {
            //printf("Inserting sync after current block\n");
            sync = true;
          }
          
        }
        if (idxSym) {
          SgForInitStatement* list =
            isSgForStatement(*it)->get_for_init_stmt();
          SgStatementPtrList& initStatements = list->get_init_stmt();
          SgStatementPtrList::const_iterator j =
            initStatements.begin();
          const SgVariableSymbol* index;
          
          if (SgExprStatement *expr = isSgExprStatement(*j))
            if (SgAssignOp* op = isSgAssignOp(
                  expr->get_expression()))
              if (SgVarRefExp* var_ref = isSgVarRefExp(
                    op->get_lhs_operand()))
                index = var_ref->get_symbol();
          
          std::vector<SgVarRefExp *> array = substitute(*it, index,
                                                        NULL, isSgNode(body_syms));
          
          for (int j = 0; j < array.size(); j++)
            array[j]->set_symbol(idxSym);
          
        }
        
        SgStatement* body_ =
          isSgStatement(
            recursiveFindReplacePreferedIdxs(
              isSgNode(
                (isSgForStatement(*it)->get_loop_body())),
              body_syms, param_syms, body, loop_idxs,
              globalscope));
        
        omega::CG_roseRepr * tnl = new omega::CG_roseRepr(*it);
        omega::CG_outputRepr* block = tnl->clone();
        tnli =
          static_cast<const omega::CG_roseRepr *>(block)->GetCode();
        
        isSgForStatement(tnli)->set_loop_body(body_);
        body_->set_parent(tnli);
        if (idxSym) {
          SgForInitStatement* list =
            isSgForStatement(tnli)->get_for_init_stmt();
          SgStatementPtrList& initStatements = list->get_init_stmt();
          SgStatementPtrList::const_iterator j =
            initStatements.begin();
          const SgVariableSymbol* index;
          
          if (SgExprStatement *expr = isSgExprStatement(*j))
            if (SgAssignOp* op = isSgAssignOp(
                  expr->get_expression()))
              if (SgVarRefExp* var_ref = isSgVarRefExp(
                    op->get_lhs_operand()))
                index = var_ref->get_symbol();
          
          std::vector<SgVarRefExp *> array = substitute(tnli, index,
                                                        NULL, isSgNode(body_syms));
          
          for (int j = 0; j < array.size(); j++)
            array[j]->set_symbol(idxSym);
        }
        idxSym = 0;
        //  std::cout << isSgNode(body_)->unparseToString() << "\n\n";
        if (att != NULL)
          tnli->setAttribute("omega_comment", att);
        clone->append_statement(isSgStatement(tnli));
        if (sync) {
          SgName name_syncthreads("__syncthreads");
          SgFunctionSymbol * syncthreads_symbol =
            globalscope->lookup_function_symbol(
              name_syncthreads);
          
          // Create a call to __syncthreads():
          SgFunctionCallExp * syncthreads_call = buildFunctionCallExp(
            syncthreads_symbol, buildExprListExp());
          
          SgExprStatement* stmt = buildExprStatement(
            syncthreads_call);
          
          /*    if (SgBasicBlock* bb = isSgBasicBlock(
                isSgForStatement(code)->get_loop_body()))
                appendStatement(isSgStatement(stmt), bb);
                
                else if (SgStatement* ss = isSgStatement(
                isSgForStatement(code)->get_loop_body())) {
                SgBasicBlock* bb2 = buildBasicBlock();
                
                isSgNode(ss)->set_parent(bb2);
                appendStatement(ss, bb2);
                
                appendStatement(isSgStatement(stmt), bb2);
                isSgNode(stmt)->set_parent(bb2);
                isSgForStatement(code)->set_loop_body(bb2);
                isSgNode(bb2)->set_parent(code);
                }
          */
          
          //SgBasicBlock* bb2 = buildBasicBlock();
          clone->append_statement(stmt);
          /* SgNode* parent = code->get_parent();
             if(!isSgStatement(parent))
             throw loop_error("Parent not a statement");
             
             if(isSgForStatement(parent)){
             if(SgStatement *ss = isSgForStatement(isSgForStatement(parent)->get_loop_body())){
             omega::CG_roseRepr * tnl = new omega::CG_roseRepr(ss);
             omega::CG_outputRepr* block= tnl->clone();
             
             SgNode *new_ss = static_cast<const omega::CG_roseRepr *>(block)->GetCode();
             SgBasicBlock* bb2 = buildBasicBlock();
             
             isSgNode(new_ss)->set_parent(bb2);
             appendStatement(isSgStatement(new_ss), bb2);
             appendStatement(isSgStatement(stmt), bb2);
             isSgNode(stmt)->set_parent(bb2);
             
             isSgStatement(parent)->replace_statement_from_basicBlock(ss, isSgStatement(bb2));
             
             }else if(isSgBasicBlock(isSgForStatement(parent)->get_loop_body()))
             isSgStatement(isSgForStatement(parent)->get_loop_body())->insert_statement(isSgStatement(code), stmt, false);
             else
             throw loop_error("parent statement type undefined!!");
             
             }
             else if(isSgBasicBlock(parent))
             isSgStatement(parent)->insert_statement(isSgStatement(code), stmt, false);
             else
             throw loop_error("parent statement type undefined!!");
             
             //tnl->print();
             *
             *
             */
          sync = true;
          //    return isSgNode(bb2);
          
        }
        
        //  return tnli;
      } else if (isSgIfStmt(*it)) {
        SgStatement* body_ = isSgStatement(
          recursiveFindReplacePreferedIdxs(
            isSgNode((isSgIfStmt(*it)->get_true_body())),
            body_syms, param_syms, body, loop_idxs,
            globalscope));
        
        omega::CG_roseRepr * tnl = new omega::CG_roseRepr(*it);
        omega::CG_outputRepr* block = tnl->clone();
        tnli1 =
          static_cast<const omega::CG_roseRepr *>(block)->GetCode();
        
        isSgIfStmt(tnli1)->set_true_body(body_);
        
        if ((isSgIfStmt(*it)->get_false_body()))
          isSgIfStmt(tnli1)->set_false_body(
            isSgStatement(
              recursiveFindReplacePreferedIdxs(
                isSgNode(
                  (isSgIfStmt(*it)->get_false_body())),
                body_syms, param_syms, body,
                loop_idxs, globalscope)));
        
        clone->append_statement(isSgStatement(tnli1));
        //return tnli;
      } else if (isSgStatement(*it)) {
        omega::CG_roseRepr * tnl = new omega::CG_roseRepr(*it);
        omega::CG_outputRepr* block = tnl->clone();
        tnli2 =
          static_cast<const omega::CG_roseRepr *>(block)->GetCode();
        
        clone->append_statement(isSgStatement(tnli2));
        //return tnli;
        
      }
    }
    
    return isSgNode(clone);
    
  }
  
  /*    if (!isSgBasicBlock(
        recursiveFindReplacePreferedIdxs(isSgNode(*it), body_syms,
        param_syms, body, loop_idxs, globalscope))) {
        SgStatement *to_push = isSgStatement(
        recursiveFindReplacePreferedIdxs(isSgNode(*it),
        body_syms, param_syms, body, loop_idxs,
        globalscope, sync));
        clone->append_statement(to_push);
        
        if ((sync_found) && isSgForStatement(to_push)) {
        SgName name_syncthreads("__syncthreads");
        SgFunctionSymbol * syncthreads_symbol =
        globalscope->lookup_function_symbol(
        name_syncthreads);
        
        // Create a call to __syncthreads():
        SgFunctionCallExp * syncthreads_call = buildFunctionCallExp(
        syncthreads_symbol, buildExprListExp());
        
        SgExprStatement* stmt = buildExprStatement(
        syncthreads_call);
        
        clone->append_statement(isSgStatement(stmt));
        }
        //  std::cout<<isSgNode(*it)->unparseToString()<<"\n\n";
        } else {
        
        SgStatementPtrList& tnl2 = isSgBasicBlock(
        recursiveFindReplacePreferedIdxs(isSgNode(*it),
        body_syms, param_syms, body, loop_idxs,
        globalscope))->get_statements();
        for (SgStatementPtrList::const_iterator it2 = tnl2.begin();
        it2 != tnl2.end(); it2++) {
        clone->append_statement(*it2);
        
        sync_found = true;
        //  std::cout<<isSgNode(*it2)->unparseToString()<<"\n\n";
        }
        }
        
        }
        return isSgNode(clone);
        }
  */
//  return tnl;
}

// loop_vars -> array references
// loop_idxs -> <idx_name,idx_sym> map for when we encounter a loop with a different preferredIndex
// dim_vars -> out param, fills with <old,new> var_sym pair for 2D array dimentions (messy stuff)
SgNode* swapVarReferences(SgNode* code,
                          std::set<const SgVariableSymbol *>& syms, SgSymbolTable* param,
                          SgSymbolTable* body, SgScopeStatement* body_stmt) {
  //Iterate over every expression, looking up each variable and type
  //reference used and possibly replacing it or adding it to our symbol
  //table
  //
  //We use the built-in cloning helper methods to seriously help us with this!
  
  //Need to do a recursive mark
  
  std::set<const SgVariableSymbol *>::iterator myIterator;
  for (myIterator = syms.begin(); myIterator != syms.end(); myIterator++) {
    SgName var_name = (*myIterator)->get_name();
    std::string x = var_name.getString();
    
    if ((param->find_variable(var_name) == NULL)
        && (body->find_variable(var_name) == NULL)) {
      SgInitializedName* decl = (*myIterator)->get_declaration();
      
      SgVariableSymbol* dvs = new SgVariableSymbol(decl);
      SgVariableDeclaration* var_decl = buildVariableDeclaration(
        dvs->get_name(), dvs->get_type());
      
      AstTextAttribute* att = (AstTextAttribute*) (isSgNode(
                                                     decl->get_declaration())->getAttribute("__shared__"));
      if (isSgNode(decl->get_declaration())->attributeExists(
            "__shared__"))
        var_decl->get_declarationModifier().get_storageModifier().setCudaShared();
      
      appendStatement(var_decl, body_stmt);
      
      dvs->set_parent(body);
      body->insert(var_name, dvs);
    }
    
    std::vector<SgVarRefExp *> array = substitute(code, *myIterator, NULL,
                                                  isSgNode(body));
    
    SgVariableSymbol* var = (SgVariableSymbol*) (*myIterator);
    for (int j = 0; j < array.size(); j++)
      array[j]->set_symbol(var);
  }
  
  return code;
}

bool LoopCuda::validIndexes(int stmt, const std::vector<std::string>& idxs) {
  for (int i = 0; i < idxs.size(); i++) {
    bool found = false;
    for (int j = 0; j < idxNames[stmt].size(); j++) {
      if (strcmp(idxNames[stmt][j].c_str(), idxs[i].c_str()) == 0) {
        found = true;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

bool LoopCuda::cudaize_v2(std::string kernel_name,
                          std::map<std::string, int> array_dims,
                          std::vector<std::string> blockIdxs,
                          std::vector<std::string> threadIdxs) {
  CG_outputBuilder *ocg = ir->builder();
  int stmt_num = 0;
  if (cudaDebug) {
    printf("cudaize_v2(%s, {", kernel_name.c_str());
    //for(
    printf("}, blocks={");
    printVs(blockIdxs);
    printf("}, thread={");
    printVs(threadIdxs);
    printf("})\n");
  }
  
  this->array_dims = array_dims;
  if (!validIndexes(stmt_num, blockIdxs)) {
    throw std::runtime_error("One of the indexes in the block list was not "
                             "found in the current set of indexes.");
  }
  if (!validIndexes(stmt_num, threadIdxs)) {
    throw std::runtime_error(
      "One of the indexes in the thread list was not "
      "found in the current set of indexes.");
  }
  if (blockIdxs.size() == 0)
    throw std::runtime_error("Cudaize: Need at least one block dimention");
  int block_level = 0;
  //Now, we will determine the actual size (if possible, otherwise
  //complain) for the block dimentions and thread dimentions based on our
  //indexes and the relations for our stmt;
  for (int i = 0; i < blockIdxs.size(); i++) {
    int level = findCurLevel(stmt_num, blockIdxs[i]);
    int ub, lb;
    CG_outputRepr* ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    if (lb != 0) {
      //attempt to "normalize" the loop with an in-place tile and then re-check our bounds
      if (cudaDebug)
        printf(
          "Cudaize: doing tile at level %d to try and normalize lower bounds\n",
          level);
      tile(stmt_num, level, 1, level, CountedTile);
      idxNames[stmt_num].insert(idxNames[stmt_num].begin() + (level), ""); //TODO: possibly handle this for all sibling stmts
      ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    }
    if (lb != 0) {
      char buf[1024];
      sprintf(buf,
              "Cudaize: Loop at level %d does not have 0 as it's lower bound",
              level);
      throw std::runtime_error(buf);
    }
    if (ub < 0) {
      char buf[1024];
      sprintf(buf,
              "Cudaize: Loop at level %d does not have a hard upper bound",
              level);
      //Anand: Commenting out error indication for lack of constant upper bound
      //throw std::runtime_error(buf);
    }
    if (cudaDebug)
      printf("block idx %s level %d lb: %d ub %d\n", blockIdxs[i].c_str(),
             level, lb, ub);
    if (i == 0) {
      block_level = level;
      if (ubrepr == NULL) {
        cu_bx = ub + 1;
        cu_bx_repr = NULL;
      } else {
        cu_bx = 0;
        cu_bx_repr = ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
      }
      idxNames[stmt_num][level - 1] = "bx";
    } else if (i == 1) {
      if (ubrepr == NULL) {
        cu_by = ub + 1;
        cu_by_repr = NULL;
      } else {
        cu_by = 0;
        cu_by_repr = ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
      }
      idxNames[stmt_num][level - 1] = "by";
    }
  }
  if (!cu_by && !cu_by_repr)
    block_level = 0;
  int thread_level1 = 0;
  int thread_level2 = 0;
  for (int i = 0; i < threadIdxs.size(); i++) {
    int level = findCurLevel(stmt_num, threadIdxs[i]);
    int ub, lb;
    CG_outputRepr* ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    if (lb != 0) {
      //attempt to "normalize" the loop with an in-place tile and then re-check our bounds
      if (cudaDebug)
        printf(
          "Cudaize: doing tile at level %d to try and normalize lower bounds\n",
          level);
      tile(stmt_num, level, 1, level, CountedTile);
      idxNames[stmt_num].insert(idxNames[stmt_num].begin() + (level), "");
      ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    }
    if (lb != 0) {
      char buf[1024];
      sprintf(buf,
              "Cudaize: Loop at level %d does not have 0 as it's lower bound",
              level);
      throw std::runtime_error(buf);
    }
    if (ub < 0) {
      char buf[1024];
      sprintf(buf,
              "Cudaize: Loop at level %d does not have a hard upper bound",
              level);
      //Anand: Commenting out error indication for lack of constant upper bound
      //throw std::runtime_error(buf);
    }
    
    if (cudaDebug)
      printf("thread idx %s level %d lb: %d ub %d\n",
             threadIdxs[i].c_str(), level, lb, ub);
    if (i == 0) {
      thread_level1 = level;
      if (ubrepr == NULL) {
        cu_tx = ub + 1;
        cu_tx_repr = NULL;
      } else {
        cu_tx = 0;
        cu_tx_repr = ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
      }
      idxNames[stmt_num][level - 1] = "tx";
    } else if (i == 1) {
      thread_level2 = level;
      if (ubrepr == NULL) {
        cu_ty = ub + 1;
        cu_ty_repr = NULL;
      } else {
        cu_ty = 0;
        cu_ty_repr = ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
      }
      idxNames[stmt_num][level - 1] = "ty";
    } else if (i == 2) {
      if (ubrepr == NULL) {
        cu_tz = ub + 1;
        cu_tz_repr = NULL;
      } else {
        cu_tz = 0;
        cu_tz_repr = ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
      }
      idxNames[stmt_num][level - 1] = "tz";
    }
  }
  if (!cu_ty && !cu_ty_repr)
    thread_level1 = 0;
  if (!cu_tz && !cu_tz_repr)
    thread_level2 = 0;
  
  //Make changes to nonsplitlevels
  const int m = stmt.size();
  for (int i = 0; i < m; i++) {
    if (block_level) {
      //stmt[i].nonSplitLevels.append((block_level)*2);
      stmt_nonSplitLevels[i].push_back((block_level) * 2);
    }
    if (thread_level1) {
      //stmt[i].nonSplitLevels.append((thread_level1)*2);
      stmt_nonSplitLevels[i].push_back((thread_level1) * 2);
    }
    if (thread_level2) {
      //stmt[i].nonSplitLevels.append((thread_level1)*2);
      stmt_nonSplitLevels[i].push_back((thread_level1) * 2);
    }
  }
  
  if (cudaDebug) {
    printf("Codegen: current names: ");
    printVS(idxNames[stmt_num]);
  }
  //Set codegen flag
  code_gen_flags |= GenCudaizeV2;
  
  //Save array dimention sizes
  this->array_dims = array_dims;
  cu_kernel_name = kernel_name.c_str();
  
}

/*
 * setupConstantVar
 * handles constant variable declaration
 * and adds a global constant variable
 * parameters:
 *   constant - the constant_memory_mapping object for this loop
 *   arr_def  - the VarDefs object for the mapped variable
 *   globals  - Rose Global variables
 *   i        - an index to keep new variable names unique
 *   symtab   - global symbol table
 */
static void setupConstantVar(constant_memory_mapping* constant, VarDefs* arr_def, SgGlobal* globals, int i, SgSymbolTable* symtab) {
  char* buf1 = new char[32];
  snprintf(buf1, 32, "cs%dRef", i+1);
  arr_def->secondName = buf1;
  
  char buf2[64];
  snprintf(buf2, 64, "__device__ __constant__ float");
  
  SgVariableDeclaration* consvar_decl = buildVariableDeclaration(
                     SgName(std::string(buf1)), buildArrayType(
                                 buildOpaqueType(SgName(buf2),globals),
                                 arr_def->size_expr));
  SgInitializedNamePtrList& variables = consvar_decl->get_variables();
  SgInitializedNamePtrList::const_iterator j = variables.begin();
  SgInitializedName* initializedName = *j;
  SgVariableSymbol* consvar_sym = new SgVariableSymbol(initializedName);
  prependStatement(consvar_decl, globals);

  consvar_sym->set_parent(symtab);
  symtab->insert(SgName(std::string(buf1)), consvar_sym);
  
  constant->set_mapped_symbol(arr_def->original_name.c_str(), consvar_sym);
  constant->set_vardef(arr_def->original_name.c_str(), arr_def);
}

/*
 * cudaBindConstantVar
 * allocs a variable to constant memory
 *   constant  - the constant mapping object
 *   arr_def   - the VarDefs abject
 *   globals   - global symbol table
 *   stmt_list - the GPU functions' statement list
 */
static void cudaBindConstantVar(constant_memory_mapping* constant, VarDefs* arr_def, SgGlobal* globals, SgStatementPtrList* stmt_list) {
  SgName cudaMemcpyToSymbol_name("cudaMemcpyToSymbol");
  SgFunctionDeclaration* cudaMemcpyToSymbol_decl = buildNondefiningFunctionDeclaration(
    cudaMemcpyToSymbol_name, buildVoidType(), buildFunctionParameterList(), globals);
  SgExprListExp* args = buildExprListExp();
  args->append_expression(buildCastExp(constant->get_mapped_symbol_exp(arr_def->original_name.c_str()),
                                       buildPointerType(buildVoidType())));
  args->append_expression(buildVarRefExp(arr_def->in_data));
  args->append_expression(arr_def->size_expr);
  stmt_list->push_back(buildExprStatement(
    buildFunctionCallExp(buildFunctionRefExp(cudaMemcpyToSymbol_decl), args)));
}

static void consmapArrayRefs(constant_memory_mapping* constant, std::vector<IR_ArrayRef*>* refs, SgGlobal* globals, IR_Code* ir, CG_roseBuilder* ocg) {
  // if constant mapping is not being used, ignore this function
  if(constant == NULL) return;
  for(int i = 0; i < refs->size(); i++) {
    IR_ArrayRef* aref = (*refs)[i];
    if(constant->is_array_mapped(aref->name().c_str())) {
      // get array reference dimensions
      int dims = aref->symbol()->n_dim();
      if(dims > 2) {
        printf(" \n CHiLL does not handle constant memory mapping for more than 2D arrays.\n");
        return;
      }
      
      SgExpression* varexp = constant->get_mapped_symbol_exp(aref->name().c_str());
      SgExpression* index_exp;
      // build index expression
      if(dims == 1) {
        index_exp = static_cast<omega::CG_roseRepr*>(aref->index(0)->clone())->GetExpression();
      }
      if(dims == 2) {
        VarDefs* arr_def = constant->get_vardef(aref->name().c_str());
        CG_outputRepr* i0 = aref->index(0)->clone();
        CG_outputRepr* i1 = aref->index(1)->clone();
        CG_outputRepr* sz = new CG_roseRepr(buildIntVal(arr_def->size_multi_dim[0]));
        CG_outputRepr* exp = ocg->CreatePlus(ocg->CreateTimes(sz->clone(), i0), i1);
        index_exp = static_cast<omega::CG_roseRepr*>(exp->clone())->GetExpression();
      }
      ir->ReplaceExpression(aref, new CG_roseRepr(buildPntrArrRefExp(varexp, index_exp)));
    }
  }
}

/*
 * setupTexmappingVar
 * handles texture variable declaration
 * and adds a global texture object 
 * parameters:
 *   texture    - the texture_memory_mapping object
 *   arr_def    - the VarDefs object for the mapped variable
 *   globals    - Rose Global variables
 *   i          - an index to keep the new variable names unique
 *   devptr_sym - the devptr that the original variable is associated with
 *   symtab     - GPU function symbol table
 */
static void setupTexmappingVar(texture_memory_mapping* texture, VarDefs* arr_def, SgGlobal* globals, int i, SgVariableSymbol* devptr_sym, SgSymbolTable* symtab) {  
  char* buf1 = new char[32];
  snprintf(buf1, 32, "tex%dRef", i+1);
  arr_def->secondName = buf1;
  
  char buf2[64];
  // single-dimensional 
  snprintf(buf2, 64, "texture<float, %d, cudaReadModeElementType>", 1);
  // multi-dimensional
  // snprintf(buf2, 64, "texture<float, %d, cudaReadModeElemetType>", (int)(arr_def->size_multi_dim.size())); //*/
  
  SgVariableDeclaration* texvar_decl = buildVariableDeclaration(SgName(std::string(buf1)), buildOpaqueType(buf2, globals));
  
  SgInitializedNamePtrList& variables = texvar_decl->get_variables();
  SgInitializedNamePtrList::const_iterator j = variables.begin();
  SgInitializedName* initializedName = *j;
  SgVariableSymbol* texvar_sym = new SgVariableSymbol(initializedName);
  prependStatement(texvar_decl, globals);
  
  texvar_sym->set_parent(symtab);
  symtab->insert(SgName(buf1), texvar_sym);
  
  texture->set_mapped_symbol(arr_def->original_name.c_str(), texvar_sym);
  texture->set_devptr_symbol(arr_def->original_name.c_str(), devptr_sym);
  texture->set_vardef(arr_def->original_name.c_str(), arr_def);
}


/*
 * One dimensional version of cudaBindTexture
 * see cudaBindTexture for details
 */
static SgFunctionCallExp* cudaBindTexture1D(texture_memory_mapping* texture, VarDefs* arr_def, SgGlobal* globals) {
  SgName cudaBindTexture_name("cudaBindTexture");
  SgFunctionDeclaration* cudaBindTexture_decl = buildNondefiningFunctionDeclaration(
      cudaBindTexture_name, buildVoidType(), buildFunctionParameterList(), globals);
  
  SgExprListExp* args = buildExprListExp();
  args->append_expression(buildIntVal(0));
  args->append_expression(texture->get_mapped_symbol_exp(arr_def->original_name.c_str()));
  args->append_expression(texture->get_devptr_symbol_exp(arr_def->original_name.c_str()));
  args->append_expression(arr_def->size_expr);
  return buildFunctionCallExp(buildFunctionRefExp(cudaBindTexture_decl), args);
}

/*
 * Two dimensional version of cudaBindTexture
 * see cudaBindTexture for details
 */
//static SgFunctionCallExp* cudaBindTexture2D(texture_memory_mapping* texture, VarDefs* arr_def, SgGlobal* globals) {
//  SgName cudaBindTexture_name("cudaBindTexture2D");
//  SgFunctionDeclaration* cudaBindTexture_decl = buildNondefiningFunctionDeclaration(
//      cudaBindTexture_name, buildVoidType(), buildFunctionParameterList(), globals);
//  
//  SgExprListExp* args = buildExprListExp();
//  args->append_expression(buildIntVal(0));
//  args->append_expression(texture->get_tex_mapped_symbol_exp(arr_def->original_name.c_str()));
//  args->append_expression(texture->get_devptr_symbol_exp(arr_def->original_name.c_str()));
//  args->append_expression(buildIntVal(texture->get_dim_length(arr_def->original_name.c_str(), 0)));
//  args->append_expression(buildIntVal(texture->get_dim_length(arr_def->original_name.c_str(), 1)));
//  args->append_expression(arr_def->size_expr);
//  return buildFunctionCallExp(buildFunctionRefExp(cudaBindTexture_decl), args);
//}

/*
 * cudaBindTexture
 * binds a variable to a texture
 * parameters:
 *    texture   - the texture mapping object
 *    arr_def   - the VarDefs object
 *    globals   - global symbol table
 *    stmt_list - the GPU functions' statement list
 * notes:
 *    only supports binding 1D textures, may need to consider cudaBindTexture2D for 2D textures
 */
static void cudaBindTexture(texture_memory_mapping* texture, VarDefs* arr_def, SgGlobal* globals, SgStatementPtrList* stmt_list) {
  //int dims = (int)(arr_def->size_multi_dim.size());
  //int dims = texture->get_dims(arr_def->original_name.c_str());
  //if(dims == 1)
    stmt_list->push_back(
      buildExprStatement(cudaBindTexture1D(texture, arr_def, globals)));
  //if(dims == 2)
  //  stmt_list->push_back(
  //    buildExprStatement(cudaBindTexture2D(texture, arr_def, globals)));
}

/*
 * texmapArrayRefs
 * maps array reference expresions of texture mapped variables to the tex1D function
 * parameters:
 *    texture - the texture mapping object
 *    refs    - a list of all array read operations
 *    globals - global symbol table
 *    ir      - handles IR_Code operations
 *    ocg     - handles CG_roseBuilder operations
**/
static void texmapArrayRefs(texture_memory_mapping* texture, std::vector<IR_ArrayRef*>* refs, SgGlobal* globals, IR_Code* ir, CG_roseBuilder *ocg) {
  // if texture mapping is not being used, ignore this function
  if(texture == NULL) return;
  for(int i = 0; i < refs->size(); i++) {
    IR_ArrayRef* aref = (*refs)[i];
    if(texture->is_array_mapped(aref->name().c_str())) {
      
      // get array dimensions
      VarDefs* arr_def = texture->get_vardef(aref->name().c_str());
      int dims = aref->symbol()->n_dim();
      if(dims > 2) {
        printf(" \n CHiLL does not handle texture mapping for more than 2D arrays.\n");
        // TODO throw some sort of error. or handle in texture_copy function
        return;
      }
      
      // build texture lookup function declaration
      char texNDfetch_strName[16];
      sprintf(texNDfetch_strName, "tex%dDfetch", 1); // for now, only support tex1Dfetch
      //sprintf(texNDfetch_strName, "tex%dDfetch", dims);
      SgFunctionDeclaration* fetch_decl = buildNondefiningFunctionDeclaration(
          SgName(texNDfetch_strName), buildFloatType(), buildFunctionParameterList(), globals);
      
      // build args
      SgExprListExp* args = buildExprListExp();
      args->append_expression(texture->get_mapped_symbol_exp(aref->name().c_str()));
      
      // set indexing args
      //for(int i = 0; i < dims; i++) {
      //  args->append_expression((static_cast<omega::CG_roseRepr*>(aref->index(i)->clone()))->GetExpression());
      //}
      if(dims == 1) {
        args->append_expression(static_cast<omega::CG_roseRepr*>(aref->index(0)->clone())->GetExpression());
      }
      else if(dims == 2) {
        CG_outputRepr* i0 = aref->index(0)->clone();
        CG_outputRepr* i1 = aref->index(1)->clone();
        CG_outputRepr* sz = new CG_roseRepr(buildIntVal(arr_def->size_multi_dim[0]));
        CG_outputRepr* expr = ocg->CreatePlus(ocg->CreateTimes(sz->clone(), i0), i1);
        args->append_expression(static_cast<omega::CG_roseRepr*>(expr->clone())->GetExpression());
      }
      
      // build function call and replace original array ref
      SgFunctionCallExp* fetch_call = buildFunctionCallExp(buildFunctionRefExp(fetch_decl), args);
      ir->ReplaceExpression(aref, new CG_roseRepr(fetch_call));
    }
  }
}

SgNode* LoopCuda::cudaize_codegen_v2() {
    if(cudaDebug)
	printf("cudaize codegen V2\n");
  CG_roseBuilder *ocg = dynamic_cast<CG_roseBuilder*>(ir->builder());
  if (!ocg)
    return false;

 //protonu--adding an annote to track texture memory type
  //ANNOTE(k_cuda_texture_memory, "cuda texture memory", TRUE);
  //ANNOTE(k_cuda_constant_memory, "cuda constant memory", TRUE);
  int tex_mem_on = 0;
  int cons_mem_on = 0;


  
  CG_outputRepr* repr;
  std::vector<VarDefs> arrayVars;
  std::vector<VarDefs> localScopedVars;
  
  std::vector<IR_ArrayRef *> ro_refs;
  std::vector<IR_ArrayRef *> wo_refs;
  std::set<std::string> uniqueRefs;
  std::set<std::string> uniqueWoRefs;
  std::set<const SgVariableSymbol *> syms;
  std::set<const SgVariableSymbol *> psyms;
  std::set<const SgVariableSymbol *> pdSyms;
  SgStatementPtrList* replacement_list = new SgStatementPtrList;
  
  for (int j = 0; j < stmt.size(); j++) {
    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[j].code);
    for (int i = 0; i < refs.size(); i++) {
      //printf("ref %s wo %d\n", static_cast<const char*>(refs[i]->name()), refs[i]->is_write());
      SgVariableSymbol* var = body_symtab->find_variable(
        SgName((char*) refs[i]->name().c_str()));
      SgVariableSymbol* var2 = parameter_symtab->find_variable(
        SgName((char*) refs[i]->name().c_str()));
      
      //If the array is not a parameter, then it's a local array and we
      //want to recreate it as a stack variable in the kernel as opposed to
      //passing it in.
      if (var != NULL) {
        //anand-- needs modification, if variable is parameter it wont be part of the
        // block's symbol table but the functiond definition's symbol table
        
        continue;
      }
      if (uniqueRefs.find(refs[i]->name()) == uniqueRefs.end()) {
        
        uniqueRefs.insert(refs[i]->name());
        if (refs[i]->is_write()) {
          uniqueWoRefs.insert(refs[i]->name());
          wo_refs.push_back(refs[i]);
        } else
          ro_refs.push_back(refs[i]);
      }
      if (refs[i]->is_write()
          && uniqueWoRefs.find(refs[i]->name())
          == uniqueWoRefs.end()) {
        uniqueWoRefs.insert(refs[i]->name());
        wo_refs.push_back(refs[i]);
        //printf("adding %s to wo\n", static_cast<const char*>(refs[i]->name()));
      }
      pdSyms.insert((const SgVariableSymbol*) var2);
    }
  }
  
  if (cudaDebug) {
      printf("reading from array ");
      for (int i = 0; i < ro_refs.size(); i++)
	  printf("'%s' ", ro_refs[i]->name().c_str());
      printf("and writing to array ");
      for (int i = 0; i < wo_refs.size(); i++)
	  printf("'%s' ", wo_refs[i]->name().c_str());
      printf("\n");
  }  
  const char* gridName = "dimGrid";
  const char* blockName = "dimBlock";
  
  //TODO: Could allow for array_dims_vars to be a mapping from array
  //references to to variable names that define their length.
  SgVariableSymbol* dim1 = 0;
  SgVariableSymbol* dim2 = 0;
  
  for (int i = 0; i < wo_refs.size(); i++) {
    //TODO: Currently assume all arrays are floats of one or two dimentions
    SgVariableSymbol* outArray = 0;
    std::string name = wo_refs[i]->name();
    outArray = body_symtab->find_variable(SgName((char*) name.c_str()));
    int size_n_d;
    if (outArray == NULL)
      outArray = parameter_symtab->find_variable(
        SgName((char*) name.c_str()));
    
    VarDefs v;
    v.size_multi_dim = std::vector<int>();
    char buf[32];
    snprintf(buf, 32, "devO%dPtr", i + 1);
    v.name = buf;
    if (isSgPointerType(outArray->get_type())) {
      if (isSgArrayType(
            isSgNode(
              isSgPointerType(outArray->get_type())->get_base_type()))) {
        //  v.type = ((array_type *)(((ptr_type *)(outArray->type()))->ref_type()))->elem_type();
        SgType* t =
          isSgPointerType(outArray->get_type())->get_base_type();
        /*   SgExprListExp* dimList = t->get_dim_info();
             SgExpressionPtrList::iterator j= dimList->get_expressions().begin();
             SgExpression* expr=NULL;
             for (; j != dimList->get_expressions().end(); j++)
             expr = *j;
        */
        while (isSgArrayType(t))
          t = isSgArrayType(t)->get_base_type();
        
        if (!isSgType(t)) {
          char buf[1024];
          sprintf(buf, "CudaizeCodeGen: Array type undetected!");
          throw std::runtime_error(buf);
          
        }
        
        v.type = t;
      } else
        v.type = isSgPointerType(outArray->get_type())->get_base_type();
    } else if (isSgArrayType(outArray->get_type())) {
      if (isSgArrayType(
            isSgNode(
              isSgArrayType(outArray->get_type())->get_base_type()))) {
        //  v.type = ((array_type *)(((ptr_type *)(outArray->type()))->ref_type()))->elem_type();
        SgType* t =
          isSgArrayType(outArray->get_type())->get_base_type();
        /*   SgExprListExp* dimList = t->get_dim_info();
             SgExpressionPtrList::iterator j= dimList->get_expressions().begin();
             SgExpression* expr=NULL;
             for (; j != dimList->get_expressions().end(); j++)
             expr = *j;
        */
        while (isSgArrayType(t))
          t = isSgArrayType(t)->get_base_type();
        
        if (!isSgType(t)) {
          char buf[1024];
          sprintf(buf, "CudaizeCodeGen: Array type undetected!");
          throw std::runtime_error(buf);
          
        }
        
        v.type = t;
      } else
        v.type = isSgArrayType(outArray->get_type())->get_base_type();
    } else
      v.type = buildFloatType();
    v.tex_mapped = false;
    v.cons_mapped = false;
    v.original_name = wo_refs[i]->name();
    //Size of the array = dim1 * dim2 * num bytes of our array type
    
    //If our input array is 2D (non-linearized), we want the actual
    //dimentions of the array
    CG_outputRepr* size;
    //Lookup in array_dims
    std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
    if (isSgPointerType(outArray->get_type())
        && isSgArrayType(
          isSgNode(
            isSgPointerType(outArray->get_type())->get_base_type()))) {
      SgType* t = isSgPointerType(outArray->get_type())->get_base_type();
      /*   SgExprListExp* dimList = t->get_dim_info();
           SgExpressionPtrList::iterator j= dimList->get_expressions().begin();
           SgExpression* expr=NULL;
           for (; j != dimList->get_expressions().end(); j++)
           expr = *j;
      */
      if (isSgIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedLongVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedLongVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgLongIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgLongLongIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgLongIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedLongLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedLongLongIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgAddOp(isSgArrayType(t)->get_index())) {
        SgAddOp *op_add = isSgAddOp(isSgArrayType(t)->get_index());
        
        SgExpression *lhs = op_add->get_lhs_operand();
        SgExpression *rhs = op_add->get_rhs_operand();
        
        if (isSgIntVal(lhs))
          size_n_d = (int) isSgIntVal(lhs)->get_value() + (int) (isSgIntVal(rhs)->get_value());
        else if (isSgUnsignedIntVal(lhs))
          size_n_d = (int) isSgUnsignedIntVal(lhs)->get_value()
            + (int) isSgUnsignedIntVal(rhs)->get_value();
        else if (isSgUnsignedLongVal(lhs))
          size_n_d = (int) (isSgUnsignedLongVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongIntVal(lhs))
          size_n_d = (int) (isSgUnsignedLongVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongLongIntVal(lhs))
          size_n_d = (int) (isSgLongLongIntVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongIntVal(lhs))
          size_n_d = (int) (isSgLongIntVal(lhs)->get_value()
                            + isSgLongIntVal(rhs)->get_value());
        else if (isSgUnsignedLongLongIntVal(lhs))
          size_n_d =
            (int) (isSgUnsignedLongLongIntVal(lhs)->get_value()
                   + isSgUnsignedLongLongIntVal(rhs)->get_value());
        
      }
      t = isSgArrayType(t)->get_base_type();
      while (isSgArrayType(t)) {
        int dim;
        if (isSgIntVal(isSgArrayType(t)->get_index()))
          dim =
            (int) (isSgIntVal(isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedLongVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedLongVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedLongLongIntVal(
                   isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedLongLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgAddOp(isSgArrayType(t)->get_index())) {
          SgAddOp *op_add = isSgAddOp(isSgArrayType(t)->get_index());
          
          SgExpression *lhs = op_add->get_lhs_operand();
          SgExpression *rhs = op_add->get_rhs_operand();
          
          if (isSgIntVal(lhs))
            dim = (int) isSgIntVal(lhs)->get_value()
              + (int) (isSgIntVal(rhs)->get_value());
          else if (isSgUnsignedIntVal(lhs))
            dim = (int) isSgUnsignedIntVal(lhs)->get_value()
              + (int) isSgUnsignedIntVal(rhs)->get_value();
          else if (isSgUnsignedLongVal(lhs))
            dim = (int) (isSgUnsignedLongVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongIntVal(lhs))
            dim = (int) (isSgUnsignedLongVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongLongIntVal(lhs))
            dim = (int) (isSgLongLongIntVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongIntVal(lhs))
            dim = (int) (isSgLongIntVal(lhs)->get_value()
                         + isSgLongIntVal(rhs)->get_value());
          else if (isSgUnsignedLongLongIntVal(lhs))
            dim =
              (int) (isSgUnsignedLongLongIntVal(lhs)->get_value()
                     + isSgUnsignedLongLongIntVal(rhs)->get_value());
          
        }
        size_n_d *= dim;
        v.size_multi_dim.push_back(dim);
        t = isSgArrayType(t)->get_base_type();
      }
      //v.size_2d = (int) (isSgIntVal(t->get_index())->get_value());
      
      if (cudaDebug)
	  printf("Detected Multi-dimensional array sized of %d for %s\n",
		 size_n_d, (char*) wo_refs[i]->name().c_str());
      size = ocg->CreateInt(size_n_d);
    } else if (isSgArrayType(outArray->get_type())
               && isSgArrayType(
                 isSgNode(
                   isSgArrayType(outArray->get_type())->get_base_type()))) {
      SgType* t = outArray->get_type();
      /*   SgExprListExp* dimList = t->get_dim_info();
           SgExpressionPtrList::iterator j= dimList->get_expressions().begin();
           SgExpression* expr=NULL;
           for (; j != dimList->get_expressions().end(); j++)
           expr = *j;
      */
      
      if (isSgIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedLongVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedLongVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgLongIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgLongLongIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgLongIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedLongLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedLongLongIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgAddOp(isSgArrayType(t)->get_index())) {
        SgAddOp *op_add = isSgAddOp(isSgArrayType(t)->get_index());
        
        SgExpression *lhs = op_add->get_lhs_operand();
        SgExpression *rhs = op_add->get_rhs_operand();
        
        if (isSgIntVal(lhs))
          size_n_d = (int) isSgIntVal(lhs)->get_value() + (int) (isSgIntVal(rhs)->get_value());
        else if (isSgUnsignedIntVal(lhs))
          size_n_d = (int) isSgUnsignedIntVal(lhs)->get_value()
            + (int) isSgUnsignedIntVal(rhs)->get_value();
        else if (isSgUnsignedLongVal(lhs))
          size_n_d = (int) (isSgUnsignedLongVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongIntVal(lhs))
          size_n_d = (int) (isSgUnsignedLongVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongLongIntVal(lhs))
          size_n_d = (int) (isSgLongLongIntVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongIntVal(lhs))
          size_n_d = (int) (isSgLongIntVal(lhs)->get_value()
                            + isSgLongIntVal(rhs)->get_value());
        else if (isSgUnsignedLongLongIntVal(lhs))
          size_n_d =
            (int) (isSgUnsignedLongLongIntVal(lhs)->get_value()
                   + isSgUnsignedLongLongIntVal(rhs)->get_value());
        
      }
      t = isSgArrayType(t)->get_base_type();
      while (isSgArrayType(t)) {
        int dim;
        if (isSgIntVal(isSgArrayType(t)->get_index()))
          dim =
            (int) (isSgIntVal(isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedLongVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedLongVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedLongLongIntVal(
                   isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedLongLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgAddOp(isSgArrayType(t)->get_index())) {
          SgAddOp *op_add = isSgAddOp(isSgArrayType(t)->get_index());
          
          SgExpression *lhs = op_add->get_lhs_operand();
          SgExpression *rhs = op_add->get_rhs_operand();
          
          if (isSgIntVal(lhs))
            dim = (int) isSgIntVal(lhs)->get_value()
              + (int) (isSgIntVal(rhs)->get_value());
          else if (isSgUnsignedIntVal(lhs))
            dim = (int) isSgUnsignedIntVal(lhs)->get_value()
              + (int) isSgUnsignedIntVal(rhs)->get_value();
          else if (isSgUnsignedLongVal(lhs))
            dim = (int) (isSgUnsignedLongVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongIntVal(lhs))
            dim = (int) (isSgUnsignedLongVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongLongIntVal(lhs))
            dim = (int) (isSgLongLongIntVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongIntVal(lhs))
            dim = (int) (isSgLongIntVal(lhs)->get_value()
                         + isSgLongIntVal(rhs)->get_value());
          else if (isSgUnsignedLongLongIntVal(lhs))
            dim =
              (int) (isSgUnsignedLongLongIntVal(lhs)->get_value()
                     + isSgUnsignedLongLongIntVal(rhs)->get_value());
          
        }
        size_n_d *= dim;
        v.size_multi_dim.push_back(dim);
        t = isSgArrayType(t)->get_base_type();
      }
      
      //v.size_2d = (int) (isSgIntVal(t->get_index())->get_value());
      
      if (cudaDebug)
	  printf("Detected Multi-Dimensional array sized of %d for %s\n",
		 size_n_d, (char*) wo_refs[i]->name().c_str());
      size = ocg->CreateInt(size_n_d);
    } else if (it != array_dims.end()) {
      int ref_size = it->second;
      //size =
      //        ocg->CreateInt(
      //                isSgIntVal(
      //                        isSgArrayType(outArray->get_type())->get_index())->get_value());
      //v.size_2d = isSgArrayType(outArray->get_type())->get_rank();
      //v.var_ref_size = ref_size;
      size = ocg->CreateInt(ref_size);
      
    } else {
      if (dim1) {
        size = ocg->CreateTimes(
          new CG_roseRepr(isSgExpression(buildVarRefExp(dim1))),
          new CG_roseRepr(isSgExpression(buildVarRefExp(dim2))));
      } else {
        char buf[1024];
        sprintf(buf,
                "CudaizeCodeGen: Array reference %s does not have a "
                "detectable size or specififed dimentions",
                name.c_str());
        throw std::runtime_error(buf);
      }
    }
    
    v.size_expr =
      static_cast<CG_roseRepr*>(ocg->CreateTimes(size,
                                                 new omega::CG_roseRepr(
                                                   isSgExpression(buildSizeOfOp(v.type)))))->GetExpression();
    
    v.in_data = 0;
    v.out_data = outArray;
    //Check for in ro_refs and remove it at this point
    std::vector<IR_ArrayRef *>::iterator it_;
    for (it_ = ro_refs.begin(); it_ != ro_refs.end(); it_++) {
      if ((*it_)->name() == wo_refs[i]->name()) {
        break;
      }
    }
    if (it_ != ro_refs.end()) {
      v.in_data = outArray;
      ro_refs.erase(it_);
    }
    
    arrayVars.push_back(v);
    
  }
  
  //protonu-- assuming that all texture mapped memories were originally read only mems
  //there should be safety checks for that, will implement those later
  
  for (int i = 0; i < ro_refs.size(); i++) {
    SgVariableSymbol* inArray = 0;
    std::string name = ro_refs[i]->name();
    inArray = body_symtab->find_variable(SgName((char*) name.c_str()));
    if (inArray == NULL)
      inArray = parameter_symtab->find_variable(
        SgName((char*) name.c_str()));
    
    VarDefs v;
    v.size_multi_dim = std::vector<int>();
    char buf[32];
    snprintf(buf, 32, "devI%dPtr", i + 1);
    v.name = buf;
    int size_n_d;
    if (isSgPointerType(inArray->get_type())) {
      if (isSgArrayType(
            isSgNode(
              isSgPointerType(inArray->get_type())->get_base_type()))) {
        
        SgType* t =
          isSgPointerType(inArray->get_type())->get_base_type();
        
        while (isSgArrayType(t))
          t = isSgArrayType(t)->get_base_type();
        
        if (!isSgType(t)) {
          char buf[1024];
          sprintf(buf, "CudaizeCodeGen: Array type undetected!");
          throw std::runtime_error(buf);
          
        }
        v.type = t;
      } else
        v.type = isSgPointerType(inArray->get_type())->get_base_type();
    } else if (isSgArrayType(inArray->get_type())) {
      if (isSgArrayType(
            isSgNode(
              isSgArrayType(inArray->get_type())->get_base_type()))) {
        
        SgType* t = inArray->get_type();
        while (isSgArrayType(t))
          t = isSgArrayType(t)->get_base_type();
        
        if (!isSgType(t)) {
          char buf[1024];
          sprintf(buf, "CudaizeCodeGen: Array type undetected!");
          throw std::runtime_error(buf);
          
        }
        v.type = t;
      } else
        v.type = isSgArrayType(inArray->get_type())->get_base_type();
    }
    
    else
      v.type = buildFloatType();
    
    v.tex_mapped = false;
    v.cons_mapped = false;
    v.original_name = ro_refs[i]->name();
    
    //derick -- adding texture and constant mapping
    if ( texture != NULL)
      v.tex_mapped = (texture->is_array_mapped(name.c_str()))? true:false; //protonu-track tex mapped vars
    if (v.tex_mapped){
      printf("this variable  %s is mapped to texture memory", name.c_str());
    }
    //derick -- this is commented out until constant memory is implemeted
    if ( constant_mem != NULL)
      v.cons_mapped = (constant_mem->is_array_mapped(name.c_str()))? true:false; //protonu-track tex mapped vars
    if (v.cons_mapped){
      printf("this variable  %s is mapped to constant memory", name.c_str());
    }
    
    //Size of the array = dim1 * dim2 * num bytes of our array type
    //If our input array is 2D (non-linearized), we want the actual
    //dimentions of the array (as it might be less than cu_n
    CG_outputRepr* size;
    //Lookup in array_dims
    std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
    if (isSgPointerType(inArray->get_type())
        && isSgArrayType(
          isSgPointerType(inArray->get_type())->get_base_type())) {
      //SgArrayType* t = isSgArrayType(isSgArrayType(inArray->get_type())->get_base_type());
      //v.size_2d = t->get_rank();
      SgType* t = isSgPointerType(inArray->get_type())->get_base_type();
      /*   SgExprListExp* dimList = t->get_dim_info();
           SgExpressionPtrList::iterator j= dimList->get_expressions().begin();
           SgExpression* expr=NULL;
           for (; j != dimList->get_expressions().end(); j++)
           expr = *j;
      */
      //v.size_2d = 1;
      if (isSgIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedLongVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedLongVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgLongIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgLongLongIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgLongIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedLongLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedLongLongIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgAddOp(isSgArrayType(t)->get_index())) {
        SgAddOp *op_add = isSgAddOp(isSgArrayType(t)->get_index());
        
        SgExpression *lhs = op_add->get_lhs_operand();
        SgExpression *rhs = op_add->get_rhs_operand();
        
        if (isSgIntVal(lhs))
          size_n_d = (int) isSgIntVal(lhs)->get_value() + (int) (isSgIntVal(rhs)->get_value());
        else if (isSgUnsignedIntVal(lhs))
          size_n_d = (int) isSgUnsignedIntVal(lhs)->get_value()
            + (int) isSgUnsignedIntVal(rhs)->get_value();
        else if (isSgUnsignedLongVal(lhs))
          size_n_d = (int) (isSgUnsignedLongVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongIntVal(lhs))
          size_n_d = (int) (isSgUnsignedLongVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongLongIntVal(lhs))
          size_n_d = (int) (isSgLongLongIntVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongIntVal(lhs))
          size_n_d = (int) (isSgLongIntVal(lhs)->get_value()
                            + isSgLongIntVal(rhs)->get_value());
        else if (isSgUnsignedLongLongIntVal(lhs))
          size_n_d =
            (int) (isSgUnsignedLongLongIntVal(lhs)->get_value()
                   + isSgUnsignedLongLongIntVal(rhs)->get_value());
        
      }
      t = isSgArrayType(t)->get_base_type();
      while (isSgArrayType(t)) {
        int dim;
        if (isSgIntVal(isSgArrayType(t)->get_index()))
          dim =
            (int) (isSgIntVal(isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedLongVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedLongVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedLongLongIntVal(
                   isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedLongLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgAddOp(isSgArrayType(t)->get_index())) {
          SgAddOp *op_add = isSgAddOp(isSgArrayType(t)->get_index());
          
          SgExpression *lhs = op_add->get_lhs_operand();
          SgExpression *rhs = op_add->get_rhs_operand();
          
          if (isSgIntVal(lhs))
            dim = (int) isSgIntVal(lhs)->get_value()
              + (int) (isSgIntVal(rhs)->get_value());
          else if (isSgUnsignedIntVal(lhs))
            dim = (int) isSgUnsignedIntVal(lhs)->get_value()
              + (int) isSgUnsignedIntVal(rhs)->get_value();
          else if (isSgUnsignedLongVal(lhs))
            dim = (int) (isSgUnsignedLongVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongIntVal(lhs))
            dim = (int) (isSgUnsignedLongVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongLongIntVal(lhs))
            dim = (int) (isSgLongLongIntVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongIntVal(lhs))
            dim = (int) (isSgLongIntVal(lhs)->get_value()
                         + isSgLongIntVal(rhs)->get_value());
          else if (isSgUnsignedLongLongIntVal(lhs))
            dim =
              (int) (isSgUnsignedLongLongIntVal(lhs)->get_value()
                     + isSgUnsignedLongLongIntVal(rhs)->get_value());
          
        }
        size_n_d *= dim;
        v.size_multi_dim.push_back(dim);
        t = isSgArrayType(t)->get_base_type();
      }
      if (cudaDebug)
	  printf("Detected Multi-dimensional array sized of %d for %s\n",
		 size_n_d, (char*) ro_refs[i]->name().c_str());
      size = ocg->CreateInt(size_n_d);
    } else if (isSgArrayType(inArray->get_type())
               && isSgArrayType(
                 isSgArrayType(inArray->get_type())->get_base_type())) {
      //SgArrayType* t = isSgArrayType(isSgArrayType(inArray->get_type())->get_base_type());
      //v.size_2d = t->get_rank();
      SgType* t = inArray->get_type();
      /*   SgExprListExp* dimList = t->get_dim_info();
           SgExpressionPtrList::iterator j= dimList->get_expressions().begin();
           SgExpression* expr=NULL;
           for (; j != dimList->get_expressions().end(); j++)
           expr = *j;
      */
      
      if (isSgIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedLongVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedLongVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgLongIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgLongLongIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d =
          (int) (isSgLongIntVal(isSgArrayType(t)->get_index())->get_value());
      else if (isSgUnsignedLongLongIntVal(isSgArrayType(t)->get_index()))
        size_n_d = (int) (isSgUnsignedLongLongIntVal(
                            isSgArrayType(t)->get_index())->get_value());
      else if (isSgAddOp(isSgArrayType(t)->get_index())) {
        SgAddOp *op_add = isSgAddOp(isSgArrayType(t)->get_index());
        
        SgExpression *lhs = op_add->get_lhs_operand();
        SgExpression *rhs = op_add->get_rhs_operand();
        
        if (isSgIntVal(lhs))
          size_n_d = (int) isSgIntVal(lhs)->get_value() + (int) (isSgIntVal(rhs)->get_value());
        else if (isSgUnsignedIntVal(lhs))
          size_n_d = (int) isSgUnsignedIntVal(lhs)->get_value()
            + (int) isSgUnsignedIntVal(rhs)->get_value();
        else if (isSgUnsignedLongVal(lhs))
          size_n_d = (int) (isSgUnsignedLongVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongIntVal(lhs))
          size_n_d = (int) (isSgUnsignedLongVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongLongIntVal(lhs))
          size_n_d = (int) (isSgLongLongIntVal(lhs)->get_value()
                            + isSgUnsignedLongVal(rhs)->get_value());
        else if (isSgLongIntVal(lhs))
          size_n_d = (int) (isSgLongIntVal(lhs)->get_value()
                            + isSgLongIntVal(rhs)->get_value());
        else if (isSgUnsignedLongLongIntVal(lhs))
          size_n_d =
            (int) (isSgUnsignedLongLongIntVal(lhs)->get_value()
                   + isSgUnsignedLongLongIntVal(rhs)->get_value());
        
      }
      t = isSgArrayType(t)->get_base_type();
      while (isSgArrayType(t)) {
        int dim;
        if (isSgIntVal(isSgArrayType(t)->get_index()))
          dim =
            (int) (isSgIntVal(isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedLongVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedLongVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgLongIntVal(isSgArrayType(t)->get_index()))
          dim = (int) (isSgLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgUnsignedLongLongIntVal(
                   isSgArrayType(t)->get_index()))
          dim = (int) (isSgUnsignedLongLongIntVal(
                         isSgArrayType(t)->get_index())->get_value());
        else if (isSgAddOp(isSgArrayType(t)->get_index())) {
          SgAddOp *op_add = isSgAddOp(isSgArrayType(t)->get_index());
          
          SgExpression *lhs = op_add->get_lhs_operand();
          SgExpression *rhs = op_add->get_rhs_operand();
          
          if (isSgIntVal(lhs))
            dim = (int) isSgIntVal(lhs)->get_value()
              + (int) (isSgIntVal(rhs)->get_value());
          else if (isSgUnsignedIntVal(lhs))
            dim = (int) isSgUnsignedIntVal(lhs)->get_value()
              + (int) isSgUnsignedIntVal(rhs)->get_value();
          else if (isSgUnsignedLongVal(lhs))
            dim = (int) (isSgUnsignedLongVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongIntVal(lhs))
            dim = (int) (isSgUnsignedLongVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongLongIntVal(lhs))
            dim = (int) (isSgLongLongIntVal(lhs)->get_value()
                         + isSgUnsignedLongVal(rhs)->get_value());
          else if (isSgLongIntVal(lhs))
            dim = (int) (isSgLongIntVal(lhs)->get_value()
                         + isSgLongIntVal(rhs)->get_value());
          else if (isSgUnsignedLongLongIntVal(lhs))
            dim =
              (int) (isSgUnsignedLongLongIntVal(lhs)->get_value()
                     + isSgUnsignedLongLongIntVal(rhs)->get_value());
          
        }
        size_n_d *= dim;
        v.size_multi_dim.push_back(dim);
        t = isSgArrayType(t)->get_base_type();
      }
      if (cudaDebug)
	  printf("Detected Multi-Dimensional array sized of %d for %s\n",
		 size_n_d, (char*) ro_refs[i]->name().c_str());
      size = ocg->CreateInt(size_n_d);
    }
    
    else if (it != array_dims.end()) {
      int ref_size = it->second;
      //                v.var_ref_size = ref_size;
      size = ocg->CreateInt(ref_size);
    } else {
      if (dim1) {
        size = ocg->CreateTimes(
          new CG_roseRepr(isSgExpression(buildVarRefExp(dim1))),
          new CG_roseRepr(isSgExpression(buildVarRefExp(dim2))));
      } else {
        char buf[1024];
        sprintf(buf,
                "CudaizeCodeGen: Array reference %s does not have a "
                "detectable size or specififed dimentions",
                name.c_str());
        throw std::runtime_error(buf);
      }
    }
    v.size_expr =
      static_cast<CG_roseRepr*>(ocg->CreateTimes(size,
                                                 new omega::CG_roseRepr(
                                                   isSgExpression(buildSizeOfOp(v.type)))))->GetExpression();
    
    v.in_data = inArray;
    v.out_data = 0;
    arrayVars.push_back(v);
  }
  
  if (arrayVars.size() < 2) {
    fprintf(stderr,
            "cudaize error: Did not find two arrays being accessed\n");
    return false;
  }

  //protonu--debugging tool--the printf statement
  //tex_mem_on signals use of tex mem
  /* derick -- texmapping near malloc mcopy
  for(int i=0; i<arrayVars.size(); i++)
  {
	  //printf("var name %s, tex_mem used %s\n", arrayVars[i].name.c_str(), (arrayVars[i].tex_mapped)?"true":"false");
	  if (arrayVars[i].tex_mapped  ) tex_mem_on ++;
	  //if (arrayVars[i].cons_mapped  ) cons_mem_on ++;
  }
  */
  
  //Add our mallocs (and input array memcpys)
  for (int i = 0; i < arrayVars.size(); i++) {
    if(arrayVars[i].cons_mapped) {
      setupConstantVar(constant_mem, &arrayVars[i], globals, i, symtab);
      SgStatementPtrList *tnl = new SgStatementPtrList;
      cudaBindConstantVar(constant_mem, &arrayVars[i], globals, tnl);
      setup_code = ocg->StmtListAppend(setup_code, new CG_roseRepr(tnl));
    } 
    else {
      SgVariableDeclaration* defn = buildVariableDeclaration(
        SgName(arrayVars[i].name.c_str()),
        buildPointerType(arrayVars[i].type));
      SgInitializedNamePtrList& variables = defn->get_variables();
      SgInitializedNamePtrList::const_iterator j = variables.begin();
      SgInitializedName* initializedName = *j;
      SgVariableSymbol* dvs = new SgVariableSymbol(initializedName);
      prependStatement(defn, func_body);
      
      dvs->set_parent(body_symtab);
      body_symtab->insert(SgName(arrayVars[i].name.c_str()), dvs);
    
//    SgVariableSymbol* dvs = body_symtab->find_variable(SgName(arrayVars[i].name.c_str()));
    
    //  if(dvs == NULL)
    //      dvs =  parameter_symtab->find_variable(SgName(arrayVars[i].name.c_str()));
    
    //cudaMalloc args
    // SgBasicBlock* block = buildBasicBlock();
      SgName name_cuda_malloc("cudaMalloc");
      SgFunctionDeclaration * decl_cuda_malloc =
        buildNondefiningFunctionDeclaration(name_cuda_malloc,
                                            buildVoidType(), buildFunctionParameterList(), globals);
      
      SgName name_cuda_copy("cudaMemcpy");
      SgFunctionDeclaration * decl_cuda_copy =
        buildNondefiningFunctionDeclaration(name_cuda_copy,
                                            buildVoidType(), buildFunctionParameterList(), globals);
      
      SgExprListExp* args = buildExprListExp();
      args->append_expression(
        buildCastExp(buildAddressOfOp(buildVarRefExp(dvs)),
                     buildPointerType(buildPointerType(buildVoidType()))));
      args->append_expression(arrayVars[i].size_expr);
    
//    decl_cuda_malloc->get_parameterList()->append_arg
      SgFunctionCallExp *the_call = buildFunctionCallExp(
        buildFunctionRefExp(decl_cuda_malloc), args);
      
      SgExprStatement* stmt = buildExprStatement(the_call);
    
    //  (*replacement_list).push_back (stmt);
    
      SgStatementPtrList* tnl = new SgStatementPtrList;
      (*tnl).push_back(stmt);
      setup_code = ocg->StmtListAppend(setup_code, new CG_roseRepr(tnl));
      if (arrayVars[i].in_data) {
      
        SgExprListExp * cuda_copy_in_args = buildExprListExp();
        cuda_copy_in_args->append_expression(
          isSgExpression(buildVarRefExp(dvs)));
        cuda_copy_in_args->append_expression(
          isSgExpression(buildVarRefExp(arrayVars[i].in_data)));
        CG_roseRepr* size_exp = new CG_roseRepr(arrayVars[i].size_expr);
        cuda_copy_in_args->append_expression(
          static_cast<CG_roseRepr*>(size_exp->clone())->GetExpression());
        cuda_copy_in_args->append_expression(
          buildOpaqueVarRefExp("cudaMemcpyHostToDevice", globals));
      
//                                      cuda_copy_in_args->append_expression(
//                                              new SgVarRefExp(sourceLocation, )
//                                      );
        SgFunctionCallExp * cuda_copy_in_func_call = buildFunctionCallExp(
          buildFunctionRefExp(decl_cuda_copy), cuda_copy_in_args);
      
        SgExprStatement* stmt = buildExprStatement(cuda_copy_in_func_call);
      
        SgStatementPtrList *tnl = new SgStatementPtrList;
        (*tnl).push_back(stmt);
        setup_code = ocg->StmtListAppend(setup_code, new CG_roseRepr(tnl));
      
        if(arrayVars[i].tex_mapped) {
          setupTexmappingVar(texture, &arrayVars[i], globals, i, dvs, symtab);
          SgStatementPtrList *tnl = new SgStatementPtrList;
          cudaBindTexture(texture, &arrayVars[i], globals, tnl);
          setup_code = ocg->StmtListAppend(setup_code, new CG_roseRepr(tnl));
        }
      }
    }
  }
  
  //Build dimGrid dim3 variables based on loop dimentions and ti/tj
  char blockD1[120];
  char blockD2[120];
  if (dim1) {
    snprintf(blockD1, 120, "%s/%d",
             dim1->get_declaration()->get_name().getString().c_str(), cu_tx);
    snprintf(blockD2, 120, "%s/%d",
             dim2->get_declaration()->get_name().getString().c_str(), cu_ty);
  } else {
    snprintf(blockD1, 120, "%d", cu_bx);
    snprintf(blockD2, 120, "%d", cu_by);
    //snprintf(blockD1, 120, "%d/%d", cu_nx, cu_tx);
    //snprintf(blockD2, 120, "%d/%d", cu_ny, cu_ty);
  }
  
  SgInitializedName* arg1 = buildInitializedName("i", buildIntType());
  SgInitializedName* arg2 = buildInitializedName("j", buildIntType());
  SgInitializedName* arg3 = buildInitializedName("k", buildIntType());
  SgName type_name("dim3");
  //SgClassSymbol * type_symbol = globalScope->lookup_class_symbol(type_name);
  
  //ROSE_ASSERT(type_symbol != NULL);
  
  //SgClassDeclaration * dim3classdecl = isSgClassDeclaration(
  //        type_symbol->get_declaration());
  
  SgFunctionDeclaration * funcdecl = buildNondefiningFunctionDeclaration(
    SgName("dim3"), buildOpaqueType("dim3", globalScope),
    //isSgType(dim3classdecl->get_type()),
    buildFunctionParameterList(arg1, arg2, arg3), globalScope);
  
  if (cu_bx && cu_by)
    repr = ocg->CreateDim3((const char*) gridName, ocg->CreateInt(cu_bx),
                           ocg->CreateInt(cu_by));
  else if (cu_bx_repr && cu_by_repr)
    repr = ocg->CreateDim3((const char*) gridName, cu_bx_repr, cu_by_repr);
  else if (cu_bx_repr)
    repr = ocg->CreateDim3((const char*) gridName, cu_bx_repr,
                           ocg->CreateInt(1));
  setup_code = ocg->StmtListAppend(setup_code, repr);
  //SgStatementPtrList* dimList = static_cast<CG_roseRepr *>(repr)->GetList();
  
  //for(SgStatementPtrList::iterator it = (*dimList).begin(); it != (*dimList).end(); it++)
  //    (*replacement_list).push_back (*it);
  
  //  repr = ocg->CreateDim3((const char*)blockName, cu_tx,cu_ty);
  
  if (cu_tz > 1 || cu_tz_repr) {
    
    if (cu_tx && cu_ty && cu_tz)
      repr = ocg->CreateDim3((char*) blockName, ocg->CreateInt(cu_tx),
                             ocg->CreateInt(cu_ty), ocg->CreateInt(cu_tz));
    else if (cu_tx_repr && cu_ty_repr && cu_tz_repr)
      repr = ocg->CreateDim3((char*) blockName, cu_tx_repr, cu_ty_repr,
                             cu_tz_repr);
    // SgStatementPtrList* dimList = static_cast<CG_roseRepr *>(repr)->GetList();
    
    // for(SgStatementPtrList::iterator it = (*dimList).begin(); it != (*dimList).end(); it++)
    //    (*replacement_list).push_back (*it);
    
  } else {
    if (cu_tx && cu_ty)
      repr = ocg->CreateDim3((char*) blockName, ocg->CreateInt(cu_tx),
                             ocg->CreateInt(cu_ty));
    else if (cu_tx_repr && cu_ty_repr)
      repr = ocg->CreateDim3((char*) blockName, cu_tx_repr, cu_ty_repr);
    //SgStatementPtrList* dimList = static_cast<CG_roseRepr *>(repr)->GetList();
    
    //for(SgStatementPtrList::iterator it = (*dimList).begin(); it != (*dimList).end(); it++)
    //   (*replacement_list).push_back (*it);
    
  }
  
  setup_code = ocg->StmtListAppend(setup_code, repr);
  
  SgCudaKernelExecConfig* config = new SgCudaKernelExecConfig(
    buildVarRefExp(gridName), buildVarRefExp(blockName), NULL, NULL);
  //SgCudaKernelExecConfig* config = new SgCudaKernelExecConfig(buildIntVal(cu_bx), , NULL, NULL);
  SgExprListExp* iml = new SgExprListExp();
  SgCastExp* dim_s;
  
  //Creating Kernel function
  SgBasicBlock* bb = new SgBasicBlock(TRANSFORMATION_FILE_INFO);
  SgFunctionDefinition* kernel_defn = new SgFunctionDefinition(
    TRANSFORMATION_FILE_INFO, bb);
  SgFunctionDeclaration* kernel_decl_ = new SgFunctionDeclaration(
    TRANSFORMATION_FILE_INFO, SgName((char*)cu_kernel_name.c_str()),buildFunctionType(buildVoidType(), buildFunctionParameterList()), kernel_defn);
  SgFunctionDeclaration* kernel_decl = new SgFunctionDeclaration(
    TRANSFORMATION_FILE_INFO, SgName((char*)cu_kernel_name.c_str()),buildFunctionType(buildVoidType(), buildFunctionParameterList()), kernel_defn);
  
  //((kernel_decl->get_declarationModifier()).get_storageModifier()).setStatic();
  
  kernel_decl->set_definingDeclaration(kernel_decl);
  kernel_defn->set_parent(kernel_decl);
  bb->set_parent(kernel_defn);
  bb->set_endOfConstruct(TRANSFORMATION_FILE_INFO);
  bb->get_endOfConstruct()->set_parent(bb);
  
  //SgFunctionSymbol* functionSymbol = new SgFunctionSymbol(kernel_decl_);
  //globals->insert_symbol(SgName((char*) cu_kernel_name.c_str()),
  //            functionSymbol);
  SgFunctionSymbol* functionSymbol2 = new SgFunctionSymbol(kernel_decl);
  
  globals->insert_symbol(SgName((char*) cu_kernel_name.c_str()),
                         functionSymbol2);
  
  kernel_decl_->set_parent(globals);
  
  kernel_decl_->set_scope(globals);
  
  kernel_decl_->setForward();
  
  globals->prepend_declaration(kernel_decl_);
  
  kernel_decl->set_endOfConstruct(TRANSFORMATION_FILE_INFO);
  kernel_decl->get_endOfConstruct()->set_parent(kernel_decl);
  
  kernel_decl->set_parent(globals);
  kernel_decl->set_scope(globals);
  
  kernel_decl->get_definition()->set_endOfConstruct(TRANSFORMATION_FILE_INFO);
  kernel_decl->get_definition()->get_endOfConstruct()->set_parent(
    kernel_decl->get_definition());
  
  globals->append_statement(kernel_decl);
  
  //printf("%s %s\n", static_cast<const char*>(cu_kernel_name), dims);
  //--derick - kernel function parameters  
  for (int i = 0; i < arrayVars.size(); i++)
    //Throw in a type cast if our kernel takes 2D array notation
    //like (float(*) [1024])
  {
    //protonu--throwing in another hack to stop the caller from passing tex mapped
    //vars to the kernel.
    if (arrayVars[i].tex_mapped == true || arrayVars[i].cons_mapped)
      continue;
    if (!(arrayVars[i].size_multi_dim.empty())) {
      //snprintf(dims,120,"(float(*) [%d])%s", arrayVars[i].size_2d,
      //         const_cast<char*>(arrayVars[i].name.c_str()));
      
      SgType* t = arrayVars[i].type;
      for (int k = arrayVars[i].size_multi_dim.size() - 1; k >= 0; k--) {
        t = buildArrayType(t,
                           buildIntVal(arrayVars[i].size_multi_dim[k]));
      }
      SgVariableSymbol* temp = body_symtab->find_variable(
        SgName((char*) arrayVars[i].name.c_str()));
      if (temp == NULL)
        temp = parameter_symtab->find_variable(
          SgName((char*) arrayVars[i].name.c_str()));
      
      dim_s = buildCastExp(buildVarRefExp(temp), buildPointerType(t),
                           SgCastExp::e_C_style_cast);
      
      //printf("%d %s\n", i, dims);
      iml->append_expression(dim_s);
      
      SgInitializedName* id = buildInitializedName(
        (char*) arrayVars[i].original_name.c_str(),
        buildPointerType(t));
      kernel_decl->get_parameterList()->append_arg(id);
      kernel_decl_->get_parameterList()->append_arg(id);
      id->set_file_info(TRANSFORMATION_FILE_INFO);
      
      // DQ (9/8/2007): We now test this, so it has to be set explicitly.
      id->set_scope(kernel_decl->get_definition());
      
      // DQ (9/8/2007): Need to add variable symbol to global scope!
      //printf ("Fixing up the symbol table in scope = %p = %s for SgInitializedName = %p = %s \n",globalScope,globalScope->class_name().c_str(),var1_init_name,var1_init_name->get_name().str());
      SgVariableSymbol *var_symbol = new SgVariableSymbol(id);
      kernel_decl->get_definition()->insert_symbol(id->get_name(),
                                                   var_symbol);
      
      // if(kernel_decl->get_definition()->get_symbol_table()->find((const) id) == NULL)
      
    } else {
      //printf("%d %s\n", i, static_cast<const char*>(arrayVars[i].name));
      SgVariableSymbol* temp = body_symtab->find_variable(
        SgName((char*) arrayVars[i].name.c_str()));
      if (temp == NULL)
        temp = parameter_symtab->find_variable(
          SgName((char*) arrayVars[i].name.c_str()));
      iml->append_expression(buildVarRefExp(temp));
      SgInitializedName* id = buildInitializedName(
        (char*) arrayVars[i].original_name.c_str(),
        buildPointerType(arrayVars[i].type));
      kernel_decl->get_parameterList()->append_arg(id);
      kernel_decl_->get_parameterList()->append_arg(id);
      id->set_file_info(TRANSFORMATION_FILE_INFO);
      
      // DQ (9/8/2007): We now test this, so it has to be set explicitly.
      id->set_scope(kernel_decl->get_definition());
      
      // DQ (9/8/2007): Need to add variable symbol to global scope!
      //printf ("Fixing up the symbol table in scope = %p = %s for SgInitializedName = %p = %s \n"$
      SgVariableSymbol *var_symbol = new SgVariableSymbol(id);
      kernel_decl->get_definition()->insert_symbol(id->get_name(),
                                                   var_symbol);
      
    }
    
  }
  if (dim1) {
    iml->append_expression(buildVarRefExp(dim1));
    SgInitializedName* id = buildInitializedName(
      dim1->get_name().getString().c_str(), dim1->get_type());
    kernel_decl->get_parameterList()->append_arg(id);
    
    iml->append_expression(buildVarRefExp(dim2));
    SgInitializedName* id2 = buildInitializedName(
      dim2->get_name().getString().c_str(), dim2->get_type());
    
    kernel_decl->get_parameterList()->append_arg(id);
    kernel_decl_->get_parameterList()->append_arg(id);
  }
  
  kernel_decl->get_functionModifier().setCudaKernel();
  kernel_decl_->get_functionModifier().setCudaKernel();
  SgCudaKernelCallExp * cuda_call_site = new SgCudaKernelCallExp(
    TRANSFORMATION_FILE_INFO, buildFunctionRefExp(kernel_decl), iml,config);
  
  //  SgStatementPtrList *tnl2 = new SgStatementPtrList;
  
  (*replacement_list).push_back(buildExprStatement(cuda_call_site));
  
  setup_code = ocg->StmtListAppend(setup_code,
                                   new CG_roseRepr(replacement_list));
  
  //cuda free variables
  for (int i = 0; i < arrayVars.size(); i++) {
    if (arrayVars[i].out_data) {
      
      SgName name_cuda_copy("cudaMemcpy");
      SgFunctionDeclaration * decl_cuda_copyout =
        buildNondefiningFunctionDeclaration(name_cuda_copy,
                                            buildVoidType(), buildFunctionParameterList(),
                                            globals);
      
      SgExprListExp* args = buildExprListExp();
      SgExprListExp * cuda_copy_out_args = buildExprListExp();
      cuda_copy_out_args->append_expression(
        isSgExpression(buildVarRefExp(arrayVars[i].out_data)));
      cuda_copy_out_args->append_expression(
        isSgExpression(buildVarRefExp(arrayVars[i].name)));
      CG_roseRepr* size_exp = new CG_roseRepr(arrayVars[i].size_expr);
      cuda_copy_out_args->append_expression(
        static_cast<CG_roseRepr*>(size_exp->clone())->GetExpression());
      cuda_copy_out_args->append_expression(
        buildOpaqueVarRefExp("cudaMemcpyDeviceToHost", globals));
      
//                                      cuda_copy_in_args->append_expression(
//                                              new SgVarRefExp(sourceLocation, )
//                                      );
      SgFunctionCallExp * cuda_copy_out_func_call = buildFunctionCallExp(
        buildFunctionRefExp(decl_cuda_copyout), cuda_copy_out_args);
      
      SgFunctionCallExp *the_call = buildFunctionCallExp(
        buildFunctionRefExp(decl_cuda_copyout), cuda_copy_out_args);
      
      SgExprStatement* stmt = buildExprStatement(the_call);
      
      SgStatementPtrList* tnl3 = new SgStatementPtrList;
      
      (*tnl3).push_back(stmt);
      
      //   tree_node_list* tnl = new tree_node_list;
      //   tnl->append(new tree_instr(the_call));
      setup_code = ocg->StmtListAppend(setup_code, new CG_roseRepr(tnl3));
      
    }
    if(!arrayVars[i].cons_mapped) {
      SgName name_cuda_free("cudaFree");
      SgFunctionDeclaration * decl_cuda_free =
        buildNondefiningFunctionDeclaration(name_cuda_free,
                                            buildVoidType(), buildFunctionParameterList(), globals);
      
      SgExprListExp* args3 = buildExprListExp();
      
      SgVariableSymbol* tmp = body_symtab->find_variable(
        SgName(arrayVars[i].name.c_str()));
      if (tmp == NULL)
        tmp = parameter_symtab->find_variable(
          SgName(arrayVars[i].name.c_str()));
      
      args3->append_expression(buildVarRefExp(tmp));
      
      SgFunctionCallExp *the_call2 = buildFunctionCallExp(
        buildFunctionRefExp(decl_cuda_free), args3);
      
      SgExprStatement* stmt2 = buildExprStatement(the_call2);
      
      SgStatementPtrList* tnl4 = new SgStatementPtrList;
      
      (*tnl4).push_back(stmt2);
      //(*replacement_list).push_back (stmt2);
    
      setup_code = ocg->StmtListAppend(setup_code, new CG_roseRepr(tnl4));
    }
  }
  
  // ---------------
  // BUILD THE KERNEL
  // ---------------
  
  //Extract out kernel body
  SgNode* code = getCode();
  //Create kernel function body
  //Add Params
  std::map<std::string, SgVariableSymbol*> loop_vars;
  //In-Out arrays
  for (int i = 0; i < arrayVars.size(); i++) {
    /*   if(arrayVars[i].in_data)
         fptr = arrayVars[i].in_data->type()->clone();
         else
         fptr = arrayVars[i].out_data->type()->clone();
    */
    
    // fptr = new_proc_syms->install_type(fptr);
    std::string name =
      arrayVars[i].in_data ?
      arrayVars[i].in_data->get_declaration()->get_name().getString() :
      arrayVars[i].out_data->get_declaration()->get_name().getString();
    //SgVariableSymbol* sym = new var_sym(fptr, arrayVars[i].in_data ? arrayVars[i].in_data->name() : arrayVars[i].out_data->name());
    
    SgVariableSymbol* sym =
      kernel_decl->get_definition()->get_symbol_table()->find_variable(
        (const char*) name.c_str());
    /* SgVariableDeclaration*  defn = buildVariableDeclaration(SgName(name.c_str()), buildFloatType());
       SgInitializedNamePtrList& variables = defn->get_variables();
       SgInitializedNamePtrList::const_iterator i = variables.begin();
       SgInitializedName* initializedName = *i;
       SgVariableSymbol* sym = new SgVariableSymbol(initializedName);
       prependStatement(defn, isSgScopeStatement(root_));
       
       vs->set_parent(symtab2_);
       symtab2_->insert(SgName(_s.c_str()), vs);
    */
    
    if (sym != NULL)
      loop_vars.insert(
        std::pair<std::string, SgVariableSymbol*>(std::string(name),
                                                  sym));
  }
  
  //Figure out which loop variables will be our thread and block dimention variables
  std::vector<SgVariableSymbol *> loop_syms;
  //Get our indexes
  std::vector<const char*> indexes; // = get_loop_indexes(code,cu_num_reduce);
  int threadsPos = 0;
  
  CG_outputRepr *body = NULL;
  SgFunctionDefinition* func_d = func_definition;
  //std::vector<SgVariableSymbol *> symbols =  recursiveFindRefs(code);
  
  SgName name_sync("__syncthreads");
  SgFunctionDeclaration * decl_sync = buildNondefiningFunctionDeclaration(
    name_sync, buildVoidType(), buildFunctionParameterList(),
    globalScope);
  
  recursiveFindRefs(code, syms, func_d);
  
  //SgFunctionDeclaration* func = Outliner::generateFunction (code, (char*)cu_kernel_name.c_str(), syms, pdSyms, psyms, NULL, globalScope);
  
  if (cu_bx > 1 || cu_bx_repr) {
    indexes.push_back("bx");
    SgName type_name("blockIdx.x");
    SgClassSymbol * type_symbol = globalScope->lookup_class_symbol(
      type_name);
    SgVariableDeclaration * var_decl = buildVariableDeclaration("bx",
                                                                buildIntType(), NULL,
                                                                isSgScopeStatement(kernel_decl->get_definition()->get_body()));
    SgStatementPtrList *tnl = new SgStatementPtrList;
    // (*tnl).push_back(isSgStatement(var_decl));
    appendStatement(var_decl, kernel_decl->get_definition()->get_body());
    
    SgVariableSymbol* bx =
      kernel_decl->get_definition()->get_body()->lookup_variable_symbol(
        SgName("bx"));
    SgStatement* assign = isSgStatement(
      buildAssignStatement(buildVarRefExp(bx),
                           buildOpaqueVarRefExp("blockIdx.x",
                                                kernel_decl->get_definition()->get_body())));
    (*tnl).push_back(assign);
    // body = ocg->StmtListAppend(body,
    //                                  new CG_roseRepr(tnl));
    appendStatement(assign, kernel_decl->get_definition()->get_body());
    
  }
  if (cu_by > 1 || cu_by_repr) {
    indexes.push_back("by");
    SgName type_name("blockIdx.y");
    SgClassSymbol * type_symbol = globalScope->lookup_class_symbol(
      type_name);
    SgVariableDeclaration * var_decl = buildVariableDeclaration("by",
                                                                buildIntType(), NULL,
                                                                isSgScopeStatement(kernel_decl->get_definition()->get_body()));
    // SgStatementPtrList *tnl = new SgStatementPtrList;
    // (*tnl).push_back(isSgStatement(var_decl));
    appendStatement(var_decl, kernel_decl->get_definition()->get_body());
    
    SgVariableSymbol* by =
      kernel_decl->get_definition()->get_body()->lookup_variable_symbol(
        SgName("by"));
    SgStatement* assign = isSgStatement(
      buildAssignStatement(buildVarRefExp(by),
                           buildOpaqueVarRefExp("blockIdx.y",
                                                kernel_decl->get_definition()->get_body())));
    //(*tnl).push_back(assign);
    // body = ocg->StmtListAppend(body,
    //                                  new CG_roseRepr(tnl));
    appendStatement(assign, kernel_decl->get_definition()->get_body());
    
  }
  if (cu_tx_repr || cu_tx > 1) {
    threadsPos = indexes.size();
    indexes.push_back("tx");
    SgName type_name("threadIdx.x");
    SgClassSymbol * type_symbol = globalScope->lookup_class_symbol(
      type_name);
    SgVariableDeclaration * var_decl = buildVariableDeclaration("tx",
                                                                buildIntType(), NULL,
                                                                isSgScopeStatement(kernel_decl->get_definition()->get_body()));
    //  SgStatementPtrList *tnl = new SgStatementPtrList;
    //  (*tnl).push_back(isSgStatement(var_decl));
    appendStatement(var_decl, kernel_decl->get_definition()->get_body());
    
    SgVariableSymbol* tx =
      kernel_decl->get_definition()->get_body()->lookup_variable_symbol(
        SgName("tx"));
    SgStatement* assign = isSgStatement(
      buildAssignStatement(buildVarRefExp(tx),
                           buildOpaqueVarRefExp("threadIdx.x",
                                                kernel_decl->get_definition()->get_body())));
    //(*tnl).push_back(assign);
    // body = ocg->StmtListAppend(body,
    //                                  new CG_roseRepr(tnl));
    appendStatement(assign, kernel_decl->get_definition()->get_body());
    
  }
  if (cu_ty_repr || cu_ty > 1) {
    indexes.push_back("ty");
    SgName type_name("threadIdx.y");
    SgClassSymbol * type_symbol = globalScope->lookup_class_symbol(
      type_name);
    SgVariableDeclaration * var_decl = buildVariableDeclaration("ty",
                                                                buildIntType(), NULL,
                                                                isSgScopeStatement(kernel_decl->get_definition()->get_body()));
    appendStatement(var_decl, kernel_decl->get_definition()->get_body());
    
    // SgStatementPtrList *tnl = new SgStatementPtrList;
    // (*tnl).push_back(isSgStatement(var_decl));
    SgVariableSymbol* ty =
      kernel_decl->get_definition()->get_body()->lookup_variable_symbol(
        SgName("ty"));
    SgStatement* assign = isSgStatement(
      buildAssignStatement(buildVarRefExp(ty),
                           buildOpaqueVarRefExp("threadIdx.y",
                                                kernel_decl->get_definition()->get_body())));
    // (*tnl).push_back(assign);
    //  body = ocg->StmtListAppend(body,
    //                                   new CG_roseRepr(tnl));
    appendStatement(assign, kernel_decl->get_definition()->get_body());
    
  }
  if (cu_tz_repr || cu_tz > 1) {
    indexes.push_back("tz");
    SgName type_name("threadIdx.z");
    SgClassSymbol * type_symbol = globalScope->lookup_class_symbol(
      type_name);
    SgVariableDeclaration * var_decl = buildVariableDeclaration("tz",
                                                                buildIntType(), NULL,
                                                                isSgScopeStatement(kernel_decl->get_definition()->get_body()));
    //   SgStatementPtrList *tnl = new SgStatementPtrList;
    //   (*tnl).push_back(isSgStatement(var_decl));
    appendStatement(var_decl, kernel_decl->get_definition()->get_body());
    
    SgVariableSymbol* tz =
      kernel_decl->get_definition()->get_body()->lookup_variable_symbol(
        "tz");
    SgStatement* assign = isSgStatement(
      buildAssignStatement(buildVarRefExp(tz),
                           buildOpaqueVarRefExp("threadIdx.z",
                                                kernel_decl->get_definition()->get_body())));
    //    (*tnl).push_back(assign);
    //     body = ocg->StmtListAppend(body,
    //                                      new CG_roseRepr(tnl));
    appendStatement(assign, kernel_decl->get_definition()->get_body());
    
  }
  
  std::map<std::string, SgVariableSymbol*> loop_idxs; //map from idx names to their new syms
  
  SgNode* swapped_ = swapVarReferences(code, syms,
                                       kernel_decl->get_definition()->get_symbol_table(),
                                       kernel_decl->get_definition()->get_body()->get_symbol_table(),
                                       kernel_decl->get_definition()->get_body());
  
  //std::cout << swapped_->unparseToString() << std::endl << std::endl;
  
  SgNode *swapped = recursiveFindReplacePreferedIdxs(swapped_,
                                                     kernel_decl->get_definition()->get_body()->get_symbol_table(),
                                                     kernel_decl->get_definition()->get_symbol_table(),
                                                     kernel_decl->get_definition()->get_body(), loop_idxs, globalScope); //in-place swapping
  //swapped->print();
  
  if (!isSgBasicBlock(swapped)) {
    appendStatement(isSgStatement(swapped),
                    kernel_decl->get_definition()->get_body());
    swapped->set_parent(
      isSgNode(kernel_decl->get_definition()->get_body()));
  } else {
    
    for (SgStatementPtrList::iterator it =
           isSgBasicBlock(swapped)->get_statements().begin();
         it != isSgBasicBlock(swapped)->get_statements().end(); it++) {
      appendStatement(*it, kernel_decl->get_definition()->get_body());
      (*it)->set_parent(
        isSgNode(kernel_decl->get_definition()->get_body()));
      
    }
    
  }
  
  for (int i = 0; i < indexes.size(); i++) {
    std::vector<SgForStatement*> tfs = findCommentedFors(indexes[i],
                                                         swapped);
    for (int k = 0; k < tfs.size(); k++) {
      //printf("replacing %p tfs for index %s\n", tfs[k], indexes[i]);
      SgNode* newBlock = forReduce(tfs[k], loop_idxs[indexes[i]],
                                   kernel_decl->get_definition());
      //newBlock->print();
      swap_node_for_node_list(tfs[k], newBlock);
      //printf("AFTER SWAP\n");        newBlock->print();
    }
  }

  //--derick replace array refs of texture mapped vars here
  body = new CG_roseRepr(kernel_decl->get_definition()->get_body());
  std::vector<IR_ArrayRef*> refs = ir->FindArrayRef(body);
  texmapArrayRefs(texture, &refs, globals, ir, ocg);
  // do the same for constant mapped vars
  consmapArrayRefs(constant_mem, &refs, globals, ir, ocg);
  
  return swapped;
}

//Order taking out dummy variables
std::vector<std::string> cleanOrder(std::vector<std::string> idxNames) {
  std::vector<std::string> results;
  for (int j = 0; j < idxNames.size(); j++) {
    if (idxNames[j].length() != 0)
      results.push_back(idxNames[j]);
  }
  return results;
}

//First non-dummy level in ascending order
int LoopCuda::nonDummyLevel(int stmt, int level) {
  //level comes in 1-basd and should leave 1-based
  for (int j = level - 1; j < idxNames[stmt].size(); j++) {
    if (idxNames[stmt][j].length() != 0) {
      //printf("found non dummy level of %d with idx: %s when searching for %d\n", j+1, (const char*) idxNames[stmt][j], level);
      return j + 1;
    }
  }
  char buf[128];
  sprintf(buf, "%d", level);
  throw std::runtime_error(
    std::string("Unable to find a non-dummy level starting from ")
    + std::string(buf));
}

int LoopCuda::findCurLevel(int stmt, std::string idx) {
  for (int j = 0; j < idxNames[stmt].size(); j++) {
    if (strcmp(idxNames[stmt][j].c_str(), idx.c_str()) == 0)
      return j + 1;
  }
  throw std::runtime_error(
    std::string("Unable to find index ") + idx
    + std::string(" in current list of indexes"));
}

void LoopCuda::permute_cuda(int stmt,
                            const std::vector<std::string>& curOrder) {
  //printf("curOrder: ");
  //printVs(curOrder);
  //printf("idxNames: ");
  //printVS(idxNames[stmt]);
  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt]);
  bool same = true;
  std::vector<int> pi;
  for (int i = 0; i < curOrder.size(); i++) {
    bool found = false;
    for (int j = 0; j < cIdxNames.size(); j++) {
      if (strcmp(cIdxNames[j].c_str(), curOrder[i].c_str()) == 0) {
        pi.push_back(j + 1);
        found = true;
        if (j != i)
          same = false;
      }
    }
    if (!found) {
      throw std::runtime_error(
        "One of the indexes in the permute order were not "
        "found in the current set of indexes.");
    }
  }
  for (int i = curOrder.size(); i < cIdxNames.size(); i++) {
    pi.push_back(i);
  }
  if (same)
    return;
  permute(stmt, pi);
  //Set old indexe names as new
  for (int i = 0; i < curOrder.size(); i++) {
    idxNames[stmt][i] = curOrder[i].c_str(); //what about sibling stmts?
  }
}

bool LoopCuda::permute(int stmt_num, const std::vector<int> &pi) {
// check for sanity of parameters
  if (stmt_num >= stmt.size() || stmt_num < 0)
    throw std::invalid_argument("invalid statement " + to_string(stmt_num));
  const int n = stmt[stmt_num].xform.n_out();
  if (pi.size() > (n - 1) / 2)
    throw std::invalid_argument(
      "iteration space dimensionality does not match permute dimensionality");
  int first_level = 0;
  int last_level = 0;
  for (int i = 0; i < pi.size(); i++) {
    if (pi[i] > (n - 1) / 2 || pi[i] <= 0)
      throw std::invalid_argument(
        "invalid loop level " + to_string(pi[i])
        + " in permuation");
    
    if (pi[i] != i + 1) {
      if (first_level == 0)
        first_level = i + 1;
      last_level = i + 1;
    }
  }
  if (first_level == 0)
    return true;
  
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> active = getStatements(lex, 2 * first_level - 2);
  Loop::permute(active, pi);
}

void LoopCuda::tile_cuda(int stmt, int level, int outer_level) {
  tile_cuda(stmt, level, 1, outer_level, "", "", CountedTile);
}
void LoopCuda::tile_cuda(int level, int tile_size, int outer_level,
                         std::string idxName, std::string ctrlName, TilingMethodType method) {
  tile_cuda(0, level, tile_size, outer_level, idxName, ctrlName, method);
}

void LoopCuda::tile_cuda(int stmt, int level, int tile_size, int outer_level,
                         std::string idxName, std::string ctrlName, TilingMethodType method) {
  //Do regular tile but then update the index and control loop variable
  //names as well as the idxName to reflect the current state of things.
  //printf("tile(%d,%d,%d,%d)\n", stmt, level, tile_size, outer_level);
  //printf("idxNames before: ");
  //printVS(idxNames[stmt]);
  
  tile(stmt, level, tile_size, outer_level, method);
  
  if (idxName.size())
    idxNames[stmt][level - 1] = idxName.c_str();
  if (tile_size == 1) {
    //potentially rearrange loops
    if (outer_level < level) {
      std::string tmp = idxNames[stmt][level - 1];
      for (int i = level - 1; i > outer_level - 1; i--) {
        if (i - 1 >= 0)
          idxNames[stmt][i] = idxNames[stmt][i - 1];
      }
      idxNames[stmt][outer_level - 1] = tmp;
    }
    //TODO: even with a tile size of one, you need a insert (of a dummy loop)
    idxNames[stmt].insert(idxNames[stmt].begin() + (level), "");
  } else {
    if (!ctrlName.size())
      throw std::runtime_error("No ctrl loop name for tile");
    //insert
    idxNames[stmt].insert(idxNames[stmt].begin() + (outer_level - 1),
                          ctrlName.c_str());
  }
  
  //printf("idxNames after: ");
  //printVS(idxNames[stmt]);
}

bool LoopCuda::datacopy_privatized_cuda(int stmt_num, int level,
                                        const std::string &array_name,
                                        const std::vector<int> &privatized_levels, bool allow_extra_read,
                                        int fastest_changing_dimension, int padding_stride,
                                        int padding_alignment, bool cuda_shared) {
  int old_stmts = stmt.size();
  //  printf("before datacopy_privatized:\n");
  printIS();
  //datacopy_privatized(stmt_num, level, array_name, privatized_levels, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, cuda_shared);
  if (cuda_shared)
    datacopy_privatized(stmt_num, level, array_name, privatized_levels,
                        allow_extra_read, fastest_changing_dimension, padding_stride,
                        padding_alignment, 1);
  else
    datacopy_privatized(stmt_num, level, array_name, privatized_levels,
                        allow_extra_read, fastest_changing_dimension, padding_stride,
                        padding_alignment, 0);
  //  printf("after datacopy_privatized:\n");
  printIS();
  
  //Adjust idxNames to reflect updated state
  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt_num]);
  int new_stmts = stmt.size();
  for (int i = old_stmts; i < new_stmts; i++) {
    //printf("fixing up statement %d\n", i);
    std::vector<std::string> idxs;
    
    //protonu-making sure the vector of nonSplitLevels grows along with
    //the statement structure
    stmt_nonSplitLevels.push_back(std::vector<int>());
    
    //Indexes up to level will be the same
    for (int j = 0; j < level - 1; j++)
      idxs.push_back(cIdxNames[j]);
    
    //Expect privatized_levels to match
    for (int j = 0; j < privatized_levels.size(); j++)
      idxs.push_back(cIdxNames[privatized_levels[j] - 1]);//level is one-based
    
    //all further levels should match order they are in originally
    if (privatized_levels.size()) {
      int last_privatized = privatized_levels.back();
      int top_level = last_privatized
        + (stmt[i].IS.n_set() - idxs.size());
      //printf("last privatized_levels: %d top_level: %d\n", last_privatized, top_level);
      for (int j = last_privatized; j < top_level; j++) {
        idxs.push_back(cIdxNames[j]);
        //printf("pushing back: %s\n", (const char*)cIdxNames[j]);
      }
    }
    idxNames.push_back(idxs);
  }
}

bool LoopCuda::datacopy_cuda(int stmt_num, int level,
                             const std::string &array_name, 
                             const std::vector<std::string> new_idxs,
                             bool allow_extra_read, int fastest_changing_dimension,
                             int padding_stride, int padding_alignment, bool cuda_shared) {
  
  int old_stmts = stmt.size();
  //datacopy(stmt_num,level,array_name,allow_extra_read,fastest_changing_dimension,padding_stride,padding_alignment,cuda_shared);
  //  printf("before datacopy:\n");
  //  printIS();
  if (cuda_shared)
    datacopy(stmt_num, level, array_name, allow_extra_read,
             fastest_changing_dimension, padding_stride, padding_alignment,
             1);
  else
    datacopy(stmt_num, level, array_name, allow_extra_read,
             fastest_changing_dimension, padding_stride, padding_alignment,
             0);
  //  printf("after datacopy:\n");
  printIS();

  //Adjust idxNames to reflect updated state
  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt_num]);
  int new_stmts = stmt.size();
  for (int i = old_stmts; i < new_stmts; i++) {
    //printf("fixing up statement %d\n", i);
    std::vector<std::string> idxs;
    
    //protonu-making sure the vector of nonSplitLevels grows along with
    //the statement structure
    stmt_nonSplitLevels.push_back(std::vector<int>());
    
    //Indexes up to level will be the same
    for (int j = 0; j < level - 1; j++)
      idxs.push_back(cIdxNames[j]);
    
    //all further levels should get names from new_idxs
    int top_level = stmt[i].IS.n_set();
    //printf("top_level: %d level: %d\n", top_level, level);
    if (new_idxs.size() < top_level - level + 1)
      throw std::runtime_error(
        "Need more new index names for new datacopy loop levels");
    
    for (int j = level - 1; j < top_level; j++) {
      idxs.push_back(new_idxs[j - level + 1].c_str());
      //printf("pushing back: %s\n", new_idxs[j-level+1].c_str());
    }
    idxNames.push_back(idxs);
  }
}

bool LoopCuda::unroll_cuda(int stmt_num, int level, int unroll_amount) {
  int old_stmts = stmt.size();
  //bool b= unroll(stmt_num, , unroll_amount);
  
  int dim = 2 * level - 1;
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_loop = getStatements(lex, dim - 1);
  
  level = nonDummyLevel(stmt_num, level);
  //printf("unrolling %d at level %d\n", stmt_num,level);
  
  //protonu--using the new version of unroll, which returns
  //a set of ints instead of a bool. To keep Gabe's logic
  //I'll check the size of the set, if it's 0 return true
  //bool b= unroll(stmt_num, level, unroll_amount);
  std::set<int> b_set = unroll(stmt_num, level, unroll_amount, idxNames);
  bool b = false;
  if (b_set.size() == 0)
    b = true;
  //end--protonu
  
  //Adjust idxNames to reflect updated state
  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt_num]);
  std::vector<std::string> origSource = idxNames[stmt_num];
  ;
  //Drop index names at level
  if (unroll_amount == 0) {
    //For all statements that were in this unroll together, drop index name for unrolled level
    idxNames[stmt_num][level - 1] = "";
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++) {
      //printf("in same loop as %d is %d\n", stmt_num, (*i));
      //idxNames[(*i)][level-1] = "";
      idxNames[(*i)] = idxNames[stmt_num];
    }
  }
  
  lex = getLexicalOrder(stmt_num);
  same_loop = getStatements(lex, dim - 1);
  
  bool same_as_source = false;
  int new_stmts = stmt.size();
  for (int i = old_stmts; i < new_stmts; i++) {
    //Check whether we had a sync for the statement we are unrolling, if
    //so, propogate that to newly created statements so that if they are
    //in a different loop structure, they will also get a syncthreads
    int size = syncs.size();
    for (int j = 0; j < size; j++) {
      if (syncs[j].first == stmt_num)
        syncs.push_back(make_pair(i, syncs[j].second));
    }
    
    //protonu-making sure the vector of nonSplitLevels grows along with
    //the statement structure
    stmt_nonSplitLevels.push_back(std::vector<int>());
    
    //We expect that new statements have a constant for the variable in
    //stmt[i].IS at level (as seen with print_with_subs), otherwise there
    //will be a for loop at level and idxNames should match stmt's
    //idxNames pre-unrolled
    Relation IS = stmt[i].IS;
    //Ok, if you know how the hell to get anything out of a Relation, you
    //should probably be able to do this more elegantly. But for now, I'm
    //hacking it.
    std::string s = IS.print_with_subs_to_string();
    //s looks looks like
    //{[_t49,8,_t51,_t52,128]: 0 <= _t52 <= 3 && 0 <= _t51 <= 15 && 0 <= _t49 && 64_t49+16_t52+_t51 <= 128}
    //where level == 5, you see a integer in the input set
    
    //If that's not an integer and this is the first new statement, then
    //we think codegen will have a loop at that level. It's not perfect,
    //not sure if it can be determined without round-tripping to codegen.
    int sIdx = 0;
    int eIdx = 0;
    for (int j = 0; j < level - 1; j++) {
      sIdx = s.find(",", sIdx + 1);
      if (sIdx < 0)
        break;
    }
    if (sIdx > 0) {
      eIdx = s.find("]");
      int tmp = s.find(",", sIdx + 1);
      if (tmp > 0 && tmp < eIdx)
        eIdx = tmp; //", before ]"
      if (eIdx > 0) {
        sIdx++;
        std::string var = s.substr(sIdx, eIdx - sIdx);
        //printf("%s\n", s.c_str());
        //printf("set var for stmt %d at level %d is %s\n", i, level, var.c_str());
        if (atoi(var.c_str()) == 0 && i == old_stmts) {
          //TODO:Maybe do see if this new statement would be in the same
          //group as the original and if it would, don't say
          //same_as_source
          if (same_loop.find(i) == same_loop.end()) {
            printf(
              "stmt %d level %d, newly created unroll statement should have same level indexes as source\n",
              i, level);
            same_as_source = true;
          }
        }
      }
    }
    
    //printf("fixing up statement %d n_set %d with %d levels\n", i, stmt[i].IS.n_set(), level-1);
    if (same_as_source)
      idxNames.push_back(origSource);
    else
      idxNames.push_back(idxNames[stmt_num]);
  }
  
  return b;
}

void LoopCuda::copy_to_texture(const char *array_name) {
  //protonu--placeholder for now
  //set the bool for using cuda memory as true
  //in a vector of strings, put the names of arrays to tex mapped
  if (!texture)
    texture = new texture_memory_mapping(true, array_name);
  else
    texture->add(array_name);
  
}

//void LoopCuda::copy_to_texture_2d(const char *array_name, int width, int height) {
//  if (!texture)
//    texture = new texture_memory_mapping(true, array_name, width, height);
//  else
//    texture->add(array_name, width, height);
//}

void LoopCuda::copy_to_constant(const char *array_name) {
  if(!constant_mem)
    constant_mem = new constant_memory_mapping(true, array_name);
  else
    constant_mem->add(array_name);
}

//protonu--moving this from Loop
SgNode* LoopCuda::codegen() {
  if (code_gen_flags & GenCudaizeV2)
    return cudaize_codegen_v2();
  //Do other flagged codegen methods, return plain vanilla generated code
  return getCode();
}

//These three are in Omega code_gen.cc and are used as a massive hack to
//get out some info from MMGenerateCode. Yea for nasty side-effects.
namespace omega {
  extern int checkLoopLevel;
  extern int stmtForLoopCheck;
  extern int upperBoundForLevel;
  extern int lowerBoundForLevel;
}

CG_outputRepr* LoopCuda::extractCudaUB(int stmt_num, int level,
                                       int &outUpperBound, int &outLowerBound) {
  // check for sanity of parameters
  const int m = stmt.size();
  if (stmt_num >= m || stmt_num < 0)
    throw std::invalid_argument("invalid statement " + to_string(stmt_num));
  const int n = stmt[stmt_num].xform.n_out();
  if (level > (n - 1) / 2 || level <= 0)
    throw std::invalid_argument("invalid loop level " + to_string(level));
  
  int dim = 2 * level - 1;
  
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_loop = getStatements(lex, dim - 1);
  
  // extract the intersection of the iteration space to be considered
  Relation hull;
  {
    hull = Relation::True(n);
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++) {
      Relation r = getNewIS(*i);
      for (int j = dim + 2; j <= r.n_set(); j++)
        r = Project(r, r.set_var(j));
      hull = Intersection(hull, r);
      hull.simplify(2, 4);
    }
    
    for (int i = 2; i <= dim + 1; i += 2) {
      //std::string name = std::string("_t") + to_string(t_counter++);
      std::string name = std::string("_t")
        + to_string(tmp_loop_var_name_counter++);
      hull.name_set_var(i, name);
    }
    hull.setup_names();
  }
  
  // extract the exact loop bound of the dimension to be unrolled
  if (is_single_iteration(hull, dim)) {
    throw std::runtime_error(
      "No loop availabe at level to extract upper bound.");
  }
  Relation bound = get_loop_bound(hull, dim);
  if (!bound.has_single_conjunct() || !bound.is_satisfiable()
      || bound.is_tautology())
    throw loop_error(
      "loop error: unable to extract loop bound for cudaize");
  
  // extract the loop stride
  EQ_Handle stride_eq;
  /*int stride = 1;
    {
    bool simple_stride = true;
    int strides = countStrides(bound.query_DNF()->single_conjunct(),
    bound.set_var(dim + 1), stride_eq, simple_stride);
    if (strides > 1)
    throw loop_error("loop error: too many strides");
    else if (strides == 1) {
    int sign = stride_eq.get_coef(bound.set_var(dim + 1));
    //      assert(sign == 1 || sign == -1);
    Constr_Vars_Iter it(stride_eq, true);
    stride = abs((*it).coef / sign);
    }
    }
  */
  int stride = 1;
  {
    
    coef_t stride;
    std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(bound,
                                                                    bound.set_var(dim + 1));
    if (result.second == NULL)
      stride = 1;
    else
      stride = abs(result.first.get_coef(result.second))
        / gcd(abs(result.first.get_coef(result.second)),
              abs(result.first.get_coef(bound.set_var(dim + 1))));
    
    if (stride > 1)
      throw loop_error("loop error: too many strides");
    /*else if (stride == 1) {
      int sign = result.first.get_coef(bound.set_var(dim+1));
      assert(sign == 1 || sign == -1);
      } */
  }
  
  if (stride != 1) {
    char buf[1024];
    sprintf(buf, "Cudaize: Loop at level %d has non-one stride of %d",
            level, stride);
    throw std::runtime_error(buf);
  }
  
  //Use code generation system to build tell us our bound information. We
  //need a hard upper bound a 0 lower bound.
  
  checkLoopLevel = level * 2;
  stmtForLoopCheck = stmt_num;
  upperBoundForLevel = -1;
  lowerBoundForLevel = -1;
  printCode(1, false);
  checkLoopLevel = 0;
  
  outUpperBound = upperBoundForLevel;
  outLowerBound = lowerBoundForLevel;
  
  if (outUpperBound == -1) {
    
    CG_result* temp = last_compute_cgr_;
    
    while (temp) {
      CG_loop * loop;
      if (loop = dynamic_cast<CG_loop*>(temp)) {
        if (loop->level_ == 2 * level) {
          Relation bound = copy(loop->bounds_);
          Variable_ID v = bound.set_var(2 * level);
          for (GEQ_Iterator e(
                 const_cast<Relation &>(bound).single_conjunct()->GEQs());
               e; e++) {
            if ((*e).get_coef(v) < 0
                && (*e).is_const_except_for_global(v))
              return output_upper_bound_repr(ir->builder(), *e, v,
                                             bound,
                                             std::vector<std::pair<CG_outputRepr *, int> >(
                                               bound.n_set(),
                                               std::make_pair(
                                                 static_cast<CG_outputRepr *>(NULL),
                                                 0)));
          }
        }
        if (loop->level_ > 2 * level)
          break;
        else
          temp = loop->body_;
      } else
        break;
    }
  }
  
  return NULL;
}

void LoopCuda::printCode(int effort, bool actuallyPrint) const {
  const int m = stmt.size();
  if (m == 0)
    return;
  const int n = stmt[0].xform.n_out();
  
  /*or (int i = 0; i < m; i++) {
    IS[i + 1] = stmt[i].IS;
    xform[i + 1] = stmt[i].xform;
    
    //nonSplitLevels[i+1] = stmt[i].nonSplitLevels;
    }
  */
  
  // invalidate saved codegen computation
  if (last_compute_cgr_ != NULL) {
    delete last_compute_cgr_;
    last_compute_cgr_ = NULL;
  }
  
  if (last_compute_cg_ != NULL) {
    delete last_compute_cg_;
    last_compute_cg_ = NULL;
  }
  
  //Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
  /*CG_stringBuilder *ocg = new CG_stringBuilder();
    Tuple<CG_outputRepr *> nameInfo;
    for (int i = 1; i <= m; i++)
    nameInfo.append(new CG_stringRepr("s" + to_string(i)));
  */
  
  // -- replacing MMGenerateCode
  // -- formally CG_outputRepr* repr = MMGenerateCode(ocg, xform, IS, nameInfo, known, nonSplitLevels, syncs, idxTupleNames, effort);
  // -- in the future, these if statements need to be cleaned up.
  // -- something like check_lastComputeCG might be a decent protected member function
  // -- and/or something that returns a std::vector<CG_outputRepr*> that also checks last_compute_cg_
  //if (last_compute_cg_ == NULL) {
  std::vector<Relation> IS(m);
  std::vector<Relation> xforms(m);
  std::vector<std::vector<int> > nonSplitLevels(m);
  
  /*    std::vector < std::vector <std::string> > idxTupleNames;
        if (useIdxNames) {
        for (int i = 0; i < idxNames.size(); i++) {
        Tuple<std::string> idxs;
        for (int j = 0; j < idxNames[i].size(); j++)
        idxs.append(idxNames[i][j]);
        idxTupleNames.append(idxs);
        }
        }
  */
  for (int i = 0; i < m; i++) {
    IS[i] = stmt[i].IS;
    xforms[i] = stmt[i].xform;
    nonSplitLevels[i] = stmt_nonSplitLevels[i];
  }
  Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
  
  last_compute_cg_ = new CodeGen(xforms, IS, known, nonSplitLevels, idxNames,
                                 syncs);

  delete last_compute_cgr_;  // this was just done  above? 
  last_compute_cgr_ = NULL;
  //}
  
  if (last_compute_cgr_ == NULL || last_compute_effort_ != effort) {
    delete last_compute_cgr_;
    last_compute_cgr_ = last_compute_cg_->buildAST(effort);
    last_compute_effort_ = effort;
  }
  
  //std::vector<CG_outputRepr *> stmts(m);
  //for (int i = 0; i < m; i++)
  //    stmts[i] = stmt[i].code;
  //CG_outputRepr* repr = last_compute_cgr_->printRepr(ocg, stmts);
  // -- end replacing MMGenerateCode
  std::string repr = last_compute_cgr_->printString();
  
  if (actuallyPrint)
    std::cout << repr << std::endl;
  //std::cout << static_cast<CG_stringRepr*>(repr)->GetString();
  /*
    for (int i = 1; i <= m; i++)
    delete nameInfo[i];
  */
  
  //delete ocg;
}

void LoopCuda::printRuntimeInfo() const {
  for (int i = 0; i < stmt.size(); i++) {
    Relation IS = stmt[i].IS;
    Relation xform = stmt[i].xform;
    printf("stmt[%d]\n", i);
    printf("IS\n");
    IS.print_with_subs();
    
    printf("xform[%d]\n", i);
    xform.print_with_subs();
    
  }
}

void LoopCuda::printIndexes() const {
  for (int i = 0; i < stmt.size(); i++) {
    printf("stmt %d nset %d ", i, stmt[i].IS.n_set());
    
    for (int j = 0; j < idxNames[i].size(); j++) {
      if (j > 0)
        printf(",");
      printf("%s", idxNames[i][j].c_str());
    }
    printf("\n");
  }
}

SgNode* LoopCuda::getCode(int effort) const {
  const int m = stmt.size();
  if (m == 0)
    return new SgNode;
  const int n = stmt[0].xform.n_out();
  /*
    Tuple<CG_outputRepr *> ni(m);
    Tuple < Relation > IS(m);
    Tuple < Relation > xform(m);
    vector < vector <int> > nonSplitLevels(m);
    for (int i = 0; i < m; i++) {
    ni[i + 1] = stmt[i].code;
    IS[i + 1] = stmt[i].IS;
    xform[i + 1] = stmt[i].xform;
    nonSplitLevels[i + 1] = stmt_nonSplitLevels[i];
    
    //nonSplitLevels[i+1] = stmt[i].nonSplitLevels;
    }
  */
  //Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
//#ifdef DEBUG
//#endif
  //std::cout << GetString(MMGenerateCode(new CG_stringBuilder(), xform, IS, ni, known,
  //                nonSplitLevels, syncs, idxTupleNames, effort));
  if (last_compute_cgr_ != NULL) {
    delete last_compute_cgr_;
    last_compute_cgr_ = NULL;
  }
  
  if (last_compute_cg_ != NULL) {
    delete last_compute_cg_;
    last_compute_cg_ = NULL;
  }
  
  CG_outputBuilder *ocg = ir->builder();
  // -- replacing MMGenerateCode
  // -- formally CG_outputRepr* repr = MMGenerateCode(ocg, xform, IS, nameInfo, known, nonSplitLevels, syncs, idxTupleNames, effort);
  // -- in the future, these if statements need to be cleaned up.
  // -- something like check_lastComputeCG might be a decent protected member function
  // -- and/or something that returns a std::vector<CG_outputRepr*> that also checks last_compute_cg_
  //if (last_compute_cg_ == NULL) {
  std::vector<Relation> IS(m);
  std::vector<Relation> xforms(m);
  std::vector<std::vector<int> > nonSplitLevels(m);
  for (int i = 0; i < m; i++) {
    IS[i] = stmt[i].IS;
    xforms[i] = stmt[i].xform;
    nonSplitLevels[i] = stmt_nonSplitLevels[i];
  }
  
  /*std::vector < std::vector<std::string> > idxTupleNames;
    if (useIdxNames) {
    for (int i = 0; i < idxNames.size(); i++) {
    std::vector<std::string> idxs;
    for (int j = 0; j < idxNames[i].size(); j++)
    idxs.push_back(idxNames[i][j]);
    idxTupleNames.push_back(idxs);
    }
    }
  */
  Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
  
  last_compute_cg_ = new CodeGen(xforms, IS, known, nonSplitLevels, idxNames,
                                 syncs);
  delete last_compute_cgr_;
  last_compute_cgr_ = NULL;
  //}
  
  if (last_compute_cgr_ == NULL || last_compute_effort_ != effort) {
    delete last_compute_cgr_;
    last_compute_cgr_ = last_compute_cg_->buildAST(effort);
    last_compute_effort_ = effort;
  }
  
  std::vector<CG_outputRepr *> stmts(m);
  for (int i = 0; i < m; i++)
    stmts[i] = stmt[i].code;
  CG_outputRepr* repr = last_compute_cgr_->printRepr(ocg, stmts);
  // -- end replacing MMGenerateCode
  
  //CG_outputRepr *overflow_initialization = ocg->CreateStmtList();
  CG_outputRepr *overflow_initialization = ocg->StmtListAppend(NULL, NULL);
  for (std::map<int, std::vector<Free_Var_Decl *> >::const_iterator i =
         overflow.begin(); i != overflow.end(); i++)
    for (std::vector<Free_Var_Decl *>::const_iterator j = i->second.begin();
         j != i->second.end(); j++)
      //overflow_initialization = ocg->StmtListAppend(overflow_initialization, ocg->CreateStmtList(ocg->CreateAssignment(0, ocg->CreateIdent((*j)->base_name()), ocg->CreateInt(0))));
      overflow_initialization = ocg->StmtListAppend(
        overflow_initialization,
        ocg->StmtListAppend(
          ocg->CreateAssignment(0,
                                ocg->CreateIdent((*j)->base_name()),
                                ocg->CreateInt(0)), NULL));
  
  repr = ocg->StmtListAppend(overflow_initialization, repr);
  SgNode *tnl = static_cast<CG_roseRepr *>(repr)->GetCode();
  SgStatementPtrList *list = static_cast<CG_roseRepr *>(repr)->GetList();
  
  if (tnl != NULL)
    return tnl;
  else if (tnl == NULL && list != NULL) {
    SgBasicBlock* bb2 = buildBasicBlock();
    
    for (SgStatementPtrList::iterator it = (*list).begin();
         it != (*list).end(); it++)
      bb2->append_statement(*it);
    
    tnl = isSgNode(bb2);
  } else
    throw loop_error("codegen failed");
  
  delete repr;
  /*
    for (int i = 1; i <= m; i++)
    delete ni[i];
  */
  return tnl;
  
}

//protonu--adding constructors for the new derived class
LoopCuda::LoopCuda() :
  Loop(), code_gen_flags(GenInit) {
}

LoopCuda::LoopCuda(IR_Control *irc, int loop_num) :
  Loop(irc) {
  setup_code = NULL;
  teardown_code = NULL;
  code_gen_flags = 0;
  cu_bx = cu_by = cu_tx = cu_ty = cu_tz = 1;
  cu_bx_repr = NULL;
  cu_tx_repr = NULL;
  cu_by_repr = NULL;
  cu_ty_repr = NULL;
  cu_tz_repr = NULL;
  
  cu_num_reduce = 0;
  cu_mode = GlobalMem;
  texture = NULL;
  constant_mem = NULL;
  
  int m = stmt.size();
  //printf("\n the size of stmt(initially) is: %d\n", stmt.size());
  for (int i = 0; i < m; i++)
    stmt_nonSplitLevels.push_back(std::vector<int>());
  
  globals = ((IR_cudaroseCode *) ir)->gsym_;
  globalScope = ((IR_cudaroseCode *) ir)->first_scope;
  parameter_symtab = ((IR_cudaroseCode *) ir)->parameter;
  body_symtab = ((IR_cudaroseCode *) ir)->body;
  func_body = ((IR_cudaroseCode *) ir)->defn;
  func_definition = ((IR_cudaroseCode *) ir)->func_defn;
  std::vector<SgForStatement *> tf = ((IR_cudaroseCode *) ir)->get_loops();
  
  symtab = tf[loop_num]->get_symbol_table();
  
  std::vector<SgForStatement *> deepest = find_deepest_loops(
    isSgNode(tf[loop_num]));
  
  for (int i = 0; i < deepest.size(); i++) {
    SgVariableSymbol* vs;
    SgForInitStatement* list = deepest[i]->get_for_init_stmt();
    SgStatementPtrList& initStatements = list->get_init_stmt();
    SgStatementPtrList::const_iterator j = initStatements.begin();
    if (SgExprStatement *expr = isSgExprStatement(*j))
      if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
        if (SgVarRefExp* var_ref = isSgVarRefExp(op->get_lhs_operand()))
          vs = var_ref->get_symbol();
    
    index.push_back(vs->get_name().getString().c_str()); //reflects original code index names
  }
  
  for (int i = 0; i < stmt.size(); i++)
    idxNames.push_back(index); //refects prefered index names (used as handles in cudaize v2)
  useIdxNames = false;
  
}

void LoopCuda::printIS() {
 if (!cudaDebug) return;
  int k = stmt.size();
  for (int i = 0; i < k; i++) {
    printf(" printing statement:%d\n", i);
    stmt[i].IS.print();
  }
}

