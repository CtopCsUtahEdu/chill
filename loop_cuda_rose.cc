
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

#define FRONTEND_ROSE  // TODO  wrap this file in #ifdef 

#include <malloc.h>

//#define TRANSFORMATION_FILE_INFO Sg_File_Info::generateDefaultFileInfoForTransformationNode()

// these are very similar (the same?) 
#include "loop_cuda_chill.hh"


#include <code_gen/CG_stringBuilder.h>
#include <codegen.h>
#include <code_gen/CG_utils.h>
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



extern char *omega::k_cuda_texture_memory; //protonu--added to track texture memory type
extern char *k_ocg_comment;

extern bool cudaDebug;


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
  fflush(stdout); 
}

void printVS(const std::vector<std::string>& curOrder) {
  if(!cudaDebug) return;
  for (int i = 0; i < curOrder.size(); i++) {
    if (i > 0)
      printf(",");
    printf("%s", curOrder[i].c_str());
  }
  printf("\n");
  fflush(stdout); 
}

LoopCuda::~LoopCuda() {
  const int m = stmt.size();
  for (int i = 0; i < m; i++)
    stmt[i].code->clear();
}

bool LoopCuda::symbolExists(std::string s) {
  debug_fprintf(stderr, "LoopCuda::symbolExists( %s )  TODO loop_cuda_rose.cc L89\nDIE\n", s.c_str()); 
  exit(-1);   // DFL 
  /*  
  if (body_symtab->find_variable(SgName(s.c_str()))  commented OUT
      || parameter_symtab->find_variable(SgName(s.c_str())))
    return true;
  if (globals->lookup_variable_symbol(SgName(s.c_str())))
    return true;
  for (int i = 0; i < idxNames.size(); i++)
    for (int j = 0; j < idxNames[i].size(); j++)
      if (strcmp(idxNames[i][j].c_str(), s.c_str()) == 0)
        return true;
  */
  return false;
}

void LoopCuda::addSync(int stmt_num, std::string idxName) {
  debug_fprintf(stderr, "addsync\n"); 
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

int wrapInIfFromMinBound(chillAST_node* then_part, 
                         chillAST_ForStmt* loop,
                         chillAST_node *symtab,    
                         chillAST_node* bound_sym) {
  debug_fprintf(stderr, "wrapInIfFromMinBound()\n"); exit(-1); // DFL 

  return 0; 
  /*
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
      //This relies on the minimum expression being the rhs operand of
      // the min instruction.
      //
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
  //return then_part;
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
void findReplacePreferedIdxs(chillAST_node *newkernelcode,
                             chillAST_FunctionDecl *kernel ) { 
  debug_fprintf(stderr, "\nrecursiveFindReplacePreferedIdxs( sync 0 )    perhaps adding syncthreads\n"); 

  const std::vector<chillAST_VarDecl *> symtab = kernel->getSymbolTable();

  newkernelcode->findLoopIndexesToReplace( symtab, false ); 

  
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
        if (cudaDebug) std::cout << idx << "\n\n";
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
            if (cudaDebug) std::cout << idx << "\n\n";
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
        body_->set_parent(tnli1);
        
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
  
}

// loop_vars -> array references
// loop_idxs -> <idx_name,idx_sym> map for when we encounter a loop with a different preferredIndex
// dim_vars -> out param, fills with <old,new> var_sym pair for 2D array dimensions (messy stuff)
void swapVarReferences( chillAST_node *newkernelcode,
                        chillAST_FunctionDecl *kernel ) { 

  // find what variables are referenced in the new kernel code
  vector<chillAST_VarDecl*> newdecls;
  newkernelcode->gatherVarDecls( newdecls );
  
  debug_fprintf(stderr, "%d variables in kernel\n", newdecls.size()); 
  for (int i=0; i<newdecls.size(); i++) { 
    debug_fprintf(stderr, "variable name %s  ", newdecls[i]->varname); 
    
    chillAST_VarDecl *isParam    = kernel->hasParameterNamed( newdecls[i]->varname ); 
    chillAST_VarDecl *isLocalVar = kernel->hasVariableNamed(  newdecls[i]->varname ); 

    if (isParam)    debug_fprintf(stderr, "is a parameter\n");
    if (isLocalVar) debug_fprintf(stderr, "is already defined in the kernel\n");

    if (!isParam && (!isLocalVar)) { 
      debug_fprintf(stderr, "needed to be added to kernel symbol table\n");
      kernel->addDecl(  newdecls[i] ); 
      kernel->addChild( newdecls[i] );  // adds to body! 
    }
  }
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
  debug_fprintf(stderr, " LoopCuda::cudaize_v2() ROSE\n");
  exit(-1); 
  for(std::map<std::string, int>::iterator it = array_dims->begin(); it != array_dims->end(); it++)  {
    debug_fprintf(stderr, "array_dims  '%s'  %d\n", it->first.c_str(), it->second)
  }

  this->array_dims = array_dims;

  debug_fprintf(stderr, "\nBEFORE ir->builder()\n"); 
  CG_outputBuilder *ocg = ir->builder();
  debug_fprintf(stderr, "AFTER  ir->builder()\n"); 
  int stmt_num = 0;
  if (cudaDebug) {
    printf("cudaize_v2(%s, {", kernel_name.c_str());
    //for(
    printf("}, blocks={");
    printVs(blockIdxs);
    printf("}, thread={");
    printVs(threadIdxs);
    printf("})\n");
    fflush(stdout); 
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
    throw std::runtime_error("Cudaize: Need at least one block dimension");
  int block_level = 0;
  //Now, we will determine the actual size (if possible, otherwise
  //complain) for the block dimensions and thread dimensions based on our
  //indexes and the relations for our stmt;
  for (int i = 0; i < blockIdxs.size(); i++) {
    int level = findCurLevel(stmt_num, blockIdxs[i]);
    int ub, lb;
    CG_outputRepr* ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    fflush(stdout); 
    debug_fprintf(stderr, "lb %ldL\n", lb); 

    if (lb != 0) {
      debug_fprintf(stderr, "lb != 0?\n");
      //attempt to "normalize" the loop with an in-place tile and then re-check our bounds
      if (cudaDebug) {
        printf(
          "Cudaize1: doing tile at level %d to try and normalize lower bounds\n",
          level);
        fflush(stdout); 
      }
      debug_fprintf(stderr, "calling tile()\n"); 
      tile(stmt_num, level, 1, level, CountedTile);
      idxNames[stmt_num].insert(idxNames[stmt_num].begin() + (level), ""); //TODO: possibly handle this for all sibling stmts
      ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    }
    else debug_fprintf(stderr, "lb == 0?\n");

    fflush(stdout); 
    debug_fprintf(stderr, "lb2 %ldL\n", lb); 
    if (lb != 0) {
      debug_fprintf(stderr, "lb2 != 0?\n");
      char buf[1024];
      sprintf(buf,
              "Cudaize: Loop at level %d does not have 0 as its lower bound",
              level);
      throw std::runtime_error(buf);
    }
    else debug_fprintf(stderr, "lb2 == 0?\n");

    if (ub < 0) {
      char buf[1024];
      sprintf(buf,
              "Cudaize: Loop at level %d does not have a hard upper bound",
              level);
      //Anand: Commenting out error indication for lack of constant upper bound
      //throw std::runtime_error(buf);
    }
    if (cudaDebug) {
      printf("block idx %s level %d lb: %d ub %d\n", blockIdxs[i].c_str(),
             level, lb, ub);
      fflush(stdout); 
    }
    else  debug_fprintf(stderr, "NO CUDADEBUG\n");

    if (i == 0) {
      block_level = level;
      if (ubrepr == NULL) {
        cu_bx = ub + 1;
        cu_bx_repr = NULL;
      } else {
        cu_bx = 0;
        cu_bx_repr = (omega::CG_chillRepr*)ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
      }
      idxNames[stmt_num][level - 1] = "bx";
    } else if (i == 1) {
      if (ubrepr == NULL) {
        cu_by = ub + 1; debug_fprintf(stderr, "cu_by = %d\n", cu_by); 
        cu_by_repr = NULL;
      } else {
        cu_by = 0;          debug_fprintf(stderr, "cu_by = %d\n", cu_by); 
        cu_by_repr = (omega::CG_chillRepr*)ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
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
    fflush(stdout); 
    debug_fprintf(stderr, "lb3 %ldL\n", lb); 
    if (lb != 0) {
      debug_fprintf(stderr, "lb3 != 0?\n");
      //attempt to "normalize" the loop with an in-place tile and then re-check our bounds
      if (cudaDebug) {
        printf(
          "Cudaize2: doing tile at level %d to try and normalize lower bounds\n",
          level);
        fflush(stdout); 
      }
      tile(stmt_num, level, 1, level, CountedTile);
      idxNames[stmt_num].insert(idxNames[stmt_num].begin() + (level), "");
      ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    }
    else debug_fprintf(stderr, "lb3 == 0?\n");

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
        cu_tx_repr = (omega::CG_chillRepr*)ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
      }
      idxNames[stmt_num][level - 1] = "tx";
    } else if (i == 1) {
      thread_level2 = level;
      if (ubrepr == NULL) {
        cu_ty = ub + 1;
        cu_ty_repr = NULL;
      } else {
        cu_ty = 0;
        cu_ty_repr = (omega::CG_chillRepr*)ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
      }
      idxNames[stmt_num][level - 1] = "ty";
    } else if (i == 2) {
      if (ubrepr == NULL) {
        cu_tz = ub + 1;
        cu_tz_repr = NULL;
      } else {
        cu_tz = 0;
        cu_tz_repr = (omega::CG_chillRepr*)ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
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
    printf("cudaize_vman svn 
2: current names: ");
    printVS(idxNames[stmt_num]);
  }
  //Set codegen flag
  code_gen_flags |= GenCudaizeV2;
  
  //Save array dimension sizes
  this->array_dims = array_dims;
  cu_kernel_name = kernel_name.c_str();
  
}

chillAST_VarDecl *addBuiltin( char *nameofbuiltin, char *typeOfBuiltin, chillAST_node *somecode) { 
  // create a builtin, like "blockIdx.x", so we can use it in assignment statements
  chillAST_VarDecl *bi = new chillAST_VarDecl( typeOfBuiltin, nameofbuiltin, "", NULL); // somecode ); // ->getSourceFile() );
  bi->isABuiltin = true;
  return bi;
}


/*
 * setupConstantVar  comment 
 * handles CUDA constant variable declaration
 * and adds a global CUDA constant variable
 * parameters:
 *   constant - the constant_memory_mapping object for this loop
 *   arr_def  - the VarDefs object for the mapped variable
 *   globals  - Rose Global variables
 *   i        - an index to keep new variable names unique
 *   symtab   - global symbol table
 */
static void setupConstantVar(constant_memory_mapping* constant, VarDefs* arr_def, SgGlobal* globals, int i, SgSymbolTable* symtab) { // definition 
  /* 
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
  */ 

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
  debug_fprintf(stderr, "cudaBindConstantVar()  die\n");  int *i=0; int j=i[0]; 
  /* 
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

//static void consmapArrayRefs(constant_memory_mapping* constant, std::vector<IR_ArrayRef*>* refs, SgGlobal* globals, IR_Code* ir, CG_roseBuilder* ocg) {
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
  */ 
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
  debug_fprintf(stderr, "cudaBindTexture1D() die\n"); int *i=0; int j = i[0];
  return NULL; 

  /* 
  SgName cudaBindTexture_name("cudaBindTexture");
  SgFunctionDeclaration* cudaBindTexture_decl = buildNondefiningFunctionDeclaration(
    cudaBindTexture_name, buildVoidType(), buildFunctionParameterList(), globals);
  
  SgExprListExp* args = buildExprListExp();
  args->append_expression(buildIntVal(0));
  args->append_expression(texture->get_mapped_symbol_exp(arr_def->original_name.c_str()));
  args->append_expression(texture->get_devptr_symbol_exp(arr_def->original_name.c_str()));
  args->append_expression(arr_def->size_expr);
  return buildFunctionCallExp(buildFunctionRefExp(cudaBindTexture_decl), args);
  */ 
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
static void texmapArrayRefs(texture_memory_mapping* texture, std::vector<IR_ArrayRef*>* refs, SgGlobal* globals, IR_Code* ir, CG_outputBuilder *ocg) {
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



chillAST_node* LoopCuda::cudaize_codegen_v2() {
  debug_fprintf(stderr, "cudaize codegen V2 (ROSE)\n");
  for(std::map<std::string, int>::iterator it = array_dims->begin(); it != array_dims->end(); it++)  {
    debug_fprintf(stderr, "array_dims  '%s'  %d\n", it->first.c_str(), it->second)
  }

  //protonu--adding an annote to track texture memory type
  //ANNOTE(k_cuda_texture_memory, "cuda texture memory", TRUE);
  //ANNOTE(k_cuda_constant_memory, "cuda constant memory", TRUE);

  int tex_mem_on = 0;
  int cons_mem_on = 0;
  
  debug_fprintf(stderr, "here goes nothing\n"); 
  
  
  CG_outputRepr* repr;


  std::vector<VarDefs> arrayVars;
  std::vector<VarDefs> localScopedVars;
  
  std::set<std::string> uniqueRefs;
  std::set<std::string> uniqueWoRefs;
  std::vector<IR_ArrayRef *> ro_refs;
  std::vector<IR_ArrayRef *> wo_refs;

  std::vector<chillAST_VarDecl *> parameterSymbols; 
  
  // the C code function will become the GPUside function

  // this is dumb. The only thing we've got is the position of the function in the file.
  // we remove the function body abd build a new one.
  // the parameters are not in the right order (probably)
  
  chillAST_FunctionDecl *origfunction =  function_that_contains_this_loop; 
  const char *fname = origfunction->functionName;
  int numparams = origfunction->parameters.size();
  //debug_fprintf(stderr, "func 0x%x has name 0x%x  %d parameters\n", func, fname, numparams); 




  // make a new function that will be the CPU side cuda code
  // it will take the name and parameters from the original C code 
  chillAST_node *p = origfunction->getParent();
  debug_fprintf(stderr, "parent of func is a %s with %d children\n", 
          p->getTypeString(), p->getNumChildren()); 
  chillAST_SourceFile *srcfile = origfunction->getSourceFile();
  debug_fprintf(stderr, "srcfile of func is %s\n", srcfile->SourceFileName );



  chillAST_FunctionDecl *CPUsidefunc = new chillAST_FunctionDecl(origfunction->returnType, fname,p);
  for (int i=0; i<numparams; i++) { 
    CPUsidefunc->addParameter( origfunction->parameters[i] ) ; 
  }
  chillAST_CompoundStmt *CPUfuncbody =  new chillAST_CompoundStmt; // so we can easily access
  CPUsidefunc->setBody( CPUfuncbody ); // but empty 
  //CPUsidefunc->setParent( origfunction->getParent() ); // unneeded


  debug_fprintf(stderr, "kernel name should be %s (?)\n", cu_kernel_name.c_str()); 
  chillAST_FunctionDecl *GPUKernel =  new  chillAST_FunctionDecl( origfunction->returnType /* ?? */,
                                                                  cu_kernel_name.c_str(), // fname, 
                                                                  p); 
  chillAST_CompoundStmt *GPUkernelbody =  new chillAST_CompoundStmt; // so we can easily access
  GPUKernel->setBody( GPUkernelbody ); // but empty 

  
  // change name of GPU side function 
  int which = p->findChild( origfunction ); 
  debug_fprintf(stderr, "func is child %d of srcfile\n", which);
  p->insertChild( which,  GPUKernel );
  p->insertChild( which,  CPUsidefunc );


  //which = p->findChild( CPUsidefunc ); 
  //debug_fprintf(stderr, "\nCPUsidefunc is now child %d of srcfile\n", which);
  //which = p->findChild( GPUKernel ); 
  //debug_fprintf(stderr, "GPUKernel is now child %d of srcfile\n", which);
  which = p->findChild( origfunction ); 
  //debug_fprintf(stderr, "original function  is now child %d of srcfile\n", which);
  //p->removeChild ( ) or similar to remove original 
  p->removeChild( which ); 


  char *part = strdup( srcfile->SourceFileName );
  char *dot = rindex( part, '.' );
  if  (dot) { 
    *dot = '\0';
  }

  // name is passed to cudaize, and stored in LoopCuda
  //char newname[800];
  //sprintf(newname, "%s_GPU\0", part);
  //debug_fprintf(stderr, "GPU side function will be %s\n", newname ); 
  //GPUKernel->setName( newname ); 

  GPUKernel->setFunctionGPU();   
  chillAST_CompoundStmt *kernelbody = new chillAST_CompoundStmt;
  GPUKernel->setBody( kernelbody );
  

  CPUsidefunc->print(); printf("\n\n"); fflush(stdout); 
  GPUKernel->print();   printf("\n\n"); fflush(stdout); 


  
  
  debug_fprintf(stderr, "this loop is in function %s\n", fname); 
  debug_fprintf(stderr, "function %s has %d parameters:\n",  fname, numparams ); 
  for (int i=0; i< function_that_contains_this_loop->parameters.size(); i++) { 
    debug_fprintf(stderr, "%d/%d  %s\n", i, numparams,  function_that_contains_this_loop->parameters[i]->varname); 
  }
  
  
  
  
  debug_fprintf(stderr, "%d statements\n", stmt.size()); 
  for (int j = 0; j < stmt.size(); j++) {
    debug_fprintf(stderr, "\nstmt j %d\n", j); 
    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[j].code);
    debug_fprintf(stderr, "%d array references in stmt j %d\n", refs.size(), j); 
    
    debug_fprintf(stderr, "\nabout to dump statement j %d\n", j); 
    CG_chillRepr * repr = (CG_chillRepr *) stmt[j].code;
    repr->dump(); 
    fflush(stdout); debug_fprintf(stderr, "\n\n\n\n");
    
    for (int i = 0; i < refs.size(); i++) {
      //const char *vname = static_cast<const char*>(refs[i]->name().c_str());
      char *vname = strdup( refs[i]->name().c_str() ) ; // just for printing 
      printf("ref i %d, ref var %s, write? %d\n", i, vname, refs[i]->is_write());
      fflush(stdout); 
      
      // at this point, GPUkernel has no parameters, we're creating them now.
      // look to see if the original function had these parameters
      chillAST_VarDecl *param = origfunction->hasParameterNamed( refs[i]->name().c_str() ); 
      if (!param) { 
        //debug_fprintf(stderr, "variable %s is NOT a parameter, it must be defined in the function body\n",vname); 
        continue;
      }
      
      //debug_fprintf(stderr, "%s is a parameter of original function\n", vname);
      debug_fprintf(stderr, "%s is a parameter\n", vname);
      
      // see if this ref is in uniqueRefs
      if (uniqueRefs.find(refs[i]->name()) == uniqueRefs.end()) {
        
        debug_fprintf(stderr, "adding variable %s to uniqueRefs\n", vname); 
        // if not, add it
        uniqueRefs.insert(refs[i]->name()); 
        
        // and if it's a write, add it to Unique Write (Only?) Refs as well
        if (refs[i]->is_write()) {
          debug_fprintf(stderr, "adding variable %s to unique WRITE Refs\n", vname); 
          uniqueWoRefs.insert(refs[i]->name()); // a set
          wo_refs.push_back(refs[i]);           // a vector of the same info? 
        } else { 
          ro_refs.push_back(refs[i]);
        }
      }
      
      
      if (refs[i]->is_write()
          && (uniqueWoRefs.find(refs[i]->name()) == uniqueWoRefs.end())) { // wasn't there before
        debug_fprintf(stderr, "adding variable %s to unique WRITE Refs even though we'd seen it as a read before\n", vname); 
        uniqueWoRefs.insert(refs[i]->name());
        wo_refs.push_back(refs[i]);
        //printf("adding %s to wo\n", static_cast<const char*>(refs[i]->name()));
      }
      
      // do a std::set manually
      bool inthere = false;
      for (int k=0; k<parameterSymbols.size(); k++) { 
        if (!strcmp( param->varname, parameterSymbols[k]->varname)) inthere = true;
      }
      if (!inthere) parameterSymbols.push_back( param ) ; 
      debug_fprintf(stderr, "parameterSymbols now has %d elements\n", parameterSymbols.size()); 

      //pdSyms.insert((const chillAST_VarDecl*) param);
      //debug_fprintf(stderr, "pdsyms now has %d elements\n", pdSyms.size()); 
    }
  } // for each stmt 
  
  debug_fprintf(stderr, "we read from %d parameter arrays, and write to %d parameter arrays\n", ro_refs.size(), wo_refs.size()); 
  printf("reading from array parameters ");
  for (int i = 0; i < ro_refs.size(); i++)
    printf("'%s' ", ro_refs[i]->name().c_str());
  printf("and writing to array parameters ");
  for (int i = 0; i < wo_refs.size(); i++)
    printf("'%s' ", wo_refs[i]->name().c_str());
  printf("\n"); fflush(stdout); 
  



  const char* gridName  = "dimGrid";       // hardcoded 
  const char* blockName = "dimBlock";
  
  //TODO: Could allow for array_dims_vars to be a mapping from array
  //references to to variable names that define their length.
  
  for (int i = 0; i < wo_refs.size(); i++) {
    std::string name = wo_refs[i]->name();
    
    debug_fprintf(stderr, "\nwritten parameter %d %s\n", i, name.c_str()); 
    
    char *tmpname = strdup( name.c_str() ); 
    // find the variable declaration in original 
    chillAST_VarDecl *param = origfunction->findParameterNamed( tmpname ); 
    if (!param) { 
      debug_fprintf(stderr, "loop_cuda_ROSE.cc can't find wo parameter named %s in function %s\n",tmpname,fname);
      exit(-1); 
    }
    //param->print(); printf("\n"); fflush(stdout); 
    
    VarDefs v; // scoping seems wrong/odd
    v.size_multi_dim = std::vector<int>();
    char buf[32];
    snprintf(buf, 32, "devO%dPtr", i + 1);
    v.name = buf;
    v.original_name = name; 
    
    v.tex_mapped  = false;
    v.cons_mapped = false;
    
    // find the underlying type of the array
    debug_fprintf(stderr, "finding underlying type of %s to make variable %s match\n",name.c_str(),buf);
    v.type = strdup(param->underlyingtype); // memory leak 
    //debug_fprintf(stderr, "v.type is %s\n", param->underlyingtype); 
    
    chillAST_node *so = new chillAST_Sizeof( v.type ); 
    //CG_chillRepr *thingsize = new omega::CG_chillRepr(  so );
    
    debug_fprintf(stderr, "\nloop_cuda_xxxx.cc  calculating size of output %s\n", buf ); 

    int numitems = 1;
    if (param->numdimensions < 1 || 
        param->arraysizes == NULL) { 
      //Lookup in array_dims (the cudaize call has this info for some variables?) 
      std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
      if (it == array_dims.end()) { debug_fprintf(stderr, "Can't find %s in array_dims\n", name.c_str()); 
        numitems = 123456;
      }
      else { 
        debug_fprintf(stderr, "it %s %d\n", (*it).first.c_str(), (*it).second);  
        numitems = (*it).second; 
      }
    }
    else { 
      debug_fprintf(stderr, "numdimensions = %d\n", param->numdimensions);
      for (int i=0; i<param->numdimensions; i++) { 
        numitems *= param->arraysizes[i]; 
      }
    } 


    chillAST_IntegerLiteral *numthings = new chillAST_IntegerLiteral( numitems ); 
    
    debug_fprintf(stderr, "creating int mult size expression numitems %d x sizeof( %s )\n", numitems, v.type ); 
    
    // create a mult  
    v.size_expr = new chillAST_BinaryOperator( numthings, "*", so, NULL); 
    
    v.CPUside_param = param;
    v.in_data = 0;
    v.out_data = param;
    
    //Check for this variable in ro_refs and remove it at this point if it is both read and write
    std::vector<IR_ArrayRef *>::iterator it_;
    for (it_ = ro_refs.begin(); it_ != ro_refs.end(); it_++) {
      if ((*it_)->name() == wo_refs[i]->name()) {
        debug_fprintf(stderr, "found array ref for %s in ro, removing it from writes\n", (*it_)->name().c_str()); 
        break;
      }
    }
    if (it_ != ro_refs.end()) {
      v.in_data = param;           // ?? 
      ro_refs.erase(it_);
    }
    
    debug_fprintf(stderr, "loop_cuda_rose.ccc L1537 adding written v to arrayVars\n\n"); 
    v.print(); 
    arrayVars.push_back(v);
  } //  wo_refs 
  
  
  
  
  
  
  
  
  for (int i = 0; i < ro_refs.size(); i++) {
    std::string name = ro_refs[i]->name();
    char *tmpname = strdup( name.c_str() ); 
    
    debug_fprintf(stderr, "\nread parameter %d %s \n", i, name.c_str()); 
    
    // find the variable declaration 
    chillAST_VarDecl *param = origfunction->findParameterNamed( tmpname ); 
    if (!param) { 
      debug_fprintf(stderr, "loop_cuda_ROSE2.cc can't find ro parameter named %s in function %s\n",tmpname,fname);
      exit(-1);
    }
    
    VarDefs v; // scoping seems wrong/odd
    v.size_multi_dim = std::vector<int>();
    char buf[32];
    snprintf(buf, 32, "devI%dPtr", i + 1);
    v.name = buf;
    v.original_name = name; 
    v.tex_mapped = false;
    v.cons_mapped = false;


    // find the underlying type of the array
    debug_fprintf(stderr, "finding underlying type of %s to make variable %s match\n",name.c_str(),buf);
    v.type = strdup(param->underlyingtype); // memory leak 
    //debug_fprintf(stderr, "v.type is %s\n", param->underlyingtype); 
    chillAST_node *so = new chillAST_Sizeof( v.type ); 
    
#ifdef NOTYET
    //derick -- adding texture and constant mapping
    if ( texture != NULL) { 
      v.tex_mapped = (texture->is_array_mapped(name.c_str()))? true:false; //protonu-track tex mapped vars
    }
    if (v.tex_mapped){
      printf("this variable  %s is mapped to texture memory", name.c_str());
    }
    //derick -- this is commented out until constant memory is implemeted
    if ( constant_mem != NULL) { 
      v.cons_mapped = (constant_mem->is_array_mapped(name.c_str()))? true:false; //protonu-track tex mapped vars
    }
    if (v.cons_mapped){
      printf("this variable  %s is mapped to constant memory", name.c_str());
    }
#endif  // NOTYET
    
    //debug_fprintf(stderr, "\ncalculating size of input %s\n", buf );    
    //Size of the array = dim1 * dim2 * num bytes of our array type
    //If our input array is 2D (non-linearized), we want the actual
    //dimensions of the array (as it might be less than cu_n
    //CG_outputRepr* size;
    
    int numitems = 1;
    param->print(0, stderr); debug_fprintf(stderr, "\n");


    if (param->numdimensions < 1 || 
        param->arraysizes == NULL) { 
      //Lookup in array_dims (the cudaize call has this info for some variables?) 
      std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
      //debug_fprintf(stderr, "it %s %d\n", (*it).first.c_str(), (*it).second);  
      debug_fprintf(stderr, "LUA command says this variable %s should be size %d\n",  (*it).first.c_str(), (*it).second); 
      numitems = (*it).second; 

    }
    else { 
      debug_fprintf(stderr, "numdimensions = %d\n", param->numdimensions);
      for (int i=0; i<param->numdimensions; i++) { 
        numitems *= param->arraysizes[i]; 
      }
    } 




    chillAST_IntegerLiteral *numthings = new chillAST_IntegerLiteral( numitems ); 
    
    debug_fprintf(stderr, "creating int mult size expression numitems %d x sizeof( %s )\n", numitems, v.type ); 
    
    // create a mult  
    v.size_expr = new chillAST_BinaryOperator( numthings, "*", so, NULL); // 1024 * sizeof(float)  etc
    
    v.CPUside_param = param;
    v.in_data = param;
    v.out_data = 0;
    
    
    debug_fprintf(stderr, "loop_cuda_rose.ccc L1636 adding input v to arrayVars\n\n"); 
    v.print(); 
    arrayVars.push_back(v);   
  } // end of READ refs
  
  
  debug_fprintf(stderr, "\n\nAdd our mallocs (and input array memcpys) %d arrayVars\n", arrayVars.size());
  //Add our mallocs (and input array memcpys)
  for (int i = 0; i < arrayVars.size(); i++) {

    //debug_fprintf(stderr, "0x%x\n", arrayVars[i].vardecl); 
    debug_fprintf(stderr, "arrayVar %d\nC side: %s    Kernel side %s\n", i, arrayVars[i].name.c_str(), arrayVars[i].original_name.c_str() ); 


    const char *kernelparamname = arrayVars[i].original_name.c_str(); 
    int pdsymoffset = -1;
    for (int j=0;j<parameterSymbols.size(); j++) { 
      if (!(strcmp( kernelparamname, parameterSymbols[j]->varname))) pdsymoffset = j;
    }
    if ( pdsymoffset == -1 ) { 
      // complain 
    }
    else { 
      // we will not know all array sizes for the kernel definition(??)
      chillAST_VarDecl *param =  (chillAST_VarDecl *)parameterSymbols[pdsymoffset]->clone(); 
      param->knownArraySizes = false; // ?? 
      
      //debug_fprintf(stderr, "adding %s to GPUKernel parameters\n", kernelparamname);
      GPUKernel->addParameter( param );
      
    }

    if(arrayVars[i].cons_mapped) {
      debug_fprintf(stderr, "arrayVar %d is cons mapped  (TODO) \n", i); 
      exit(-1); 
    }
    else { 
      debug_fprintf(stderr, "buildVariableDeclaration %s\n", arrayVars[i].name.c_str()); 
      // create a CHILL variable declaration and put it in the CPU side function

      char typ[128];
      sprintf(typ, "%s *", arrayVars[i].type); 

      chillAST_VarDecl *var = new chillAST_VarDecl( typ,
                                                    arrayVars[i].name.c_str(),
                                                    "", // TODO
                                                    NULL);
      // set the array info to match
      // store variable decl where we can get it easilly later
      arrayVars[i].vardecl = var;

      CPUfuncbody->insertChild(0, var );  // add the CPUside variable declaration 

      // do the CPU side cudaMalloc 
      chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( var, CPUfuncbody ); 
      chillAST_CStyleAddressOf *AO = new chillAST_CStyleAddressOf( DRE );
      chillAST_CStyleCastExpr *casttovoidptrptr = new chillAST_CStyleCastExpr( "void **", AO, NULL ); 
      chillAST_CudaMalloc *cmalloc = new chillAST_CudaMalloc( casttovoidptrptr, arrayVars[i].size_expr, NULL); 
      CPUfuncbody->addChild( cmalloc );

      debug_fprintf(stderr, "\ncudamalloc is:\n"); 
      cmalloc->print(); printf("\n"); fflush(stdout); 
      
      debug_fprintf(stderr, "\nnow the memcpy (for input variables only????)\n"); 
      if (arrayVars[i].in_data) {  // if it's input to the calculation, and we need to copy the data to the GPU
        //debug_fprintf(stderr, "it's an input to the calculation, so we need to copy the data to the GPU\n"); 
        
        // do the CPU side cudaMemcpy, CPU to GPU("device")
        //DRE = new chillAST_DeclRefExpr( var, CPUfuncbody ); 
        chillAST_CudaMemcpy *cmemcpy = new chillAST_CudaMemcpy( var, 
                                                                (chillAST_VarDecl*)(arrayVars[i].in_data), 
                                                                arrayVars[i].size_expr, "cudaMemcpyHostToDevice"); 
        CPUfuncbody->addChild( cmemcpy );
        
        //printf("\n"); cmemcpy->print(); printf("\n");fflush(stdout); 
      } // for input variables only (must be copied to GPU before the kernel call) 
      //else { 
      //  debug_fprintf(stderr, "it's not an input to the calculation, so no memcpy over to the GPU\n"); 
      //} 
    }  // not cons mapped 
  }   // for all arrayvars 


  debug_fprintf(stderr, "\nBuild dimGrid dim3 variables based on loop dimensions and ti/tj\n"); 
  //Build dimGrid dim3 variables based on loop dimensions and ti/tj
  char blockD1[120];
  char blockD2[120];
  int dim1 = 0;            // TODO 
  if (dim1) {
     debug_fprintf(stderr,"cu_tx, cu_ty    CASE NOT HANDLED\n"); 
     exit(-1); 
#ifdef NOTYET
   snprintf(blockD1, 120, "%s/%d",
             dim1->get_declaration()->get_name().getString().c_str(), cu_tx);
    snprintf(blockD2, 120, "%s/%d",
             dim2->get_declaration()->get_name().getString().c_str(), cu_ty);
#endif
  } else {
    debug_fprintf(stderr,"cu_bx, cu_by\n"); 
    snprintf(blockD1, 120, "%d", cu_bx);
    snprintf(blockD2, 120, "%d", cu_by);
    //snprintf(blockD1, 120, "%d/%d", cu_nx, cu_tx);
    //snprintf(blockD2, 120, "%d/%d", cu_ny, cu_ty);
  }
  debug_fprintf(stderr, "blockD1 '%s'\n", blockD1); 
  debug_fprintf(stderr, "blockD2 '%s'\n", blockD2); 
  
  chillAST_FunctionDecl *dimbuiltin = new chillAST_FunctionDecl( "dim3", "dim3" );
  dimbuiltin->setBuiltin();

  chillAST_CallExpr *CE1 = new chillAST_CallExpr( dimbuiltin, NULL );

  // create ARGS ro dim3. 
  debug_fprintf(stderr, "create ARGS to dim3\n"); 
  if (cu_bx && cu_by) {                                      // 2 constants
    debug_fprintf(stderr, "dim3 dimGrid %d %d\n", cu_bx, cu_by); 
    CE1->addArg( new chillAST_IntegerLiteral( cu_bx ));
    CE1->addArg( new chillAST_IntegerLiteral( cu_by ));
  }
  else if (cu_bx_repr && cu_by_repr) {                        // 2 expressions? 
    debug_fprintf(stderr, "dim3 dimGrid cu_bx_repr  cu_by_repr\n" ); 
    chillAST_node *code1  = cu_bx_repr-> GetCode();
    chillAST_node *code2  = cu_bx_repr-> GetCode();
    CE1->addArg( code1 ); 
    CE1->addArg( code2 ); 
  }
  else if (cu_bx_repr) {
    debug_fprintf(stderr, "dim3 dimGrid  cu_bx_repr 1\n");            // one expression, and a constant?
    cu_bx_repr->dump(); fflush(stdout); 
    chillAST_node *code  = cu_bx_repr-> GetCode();
    
    CE1->addArg( code ); 
    CE1->addArg( new chillAST_IntegerLiteral( cu_by ));

  }

  chillAST_VarDecl *dimgriddecl = new chillAST_VarDecl( "dim3", "dimGrid", "", NULL );
  dimgriddecl->setInit(CE1);
  CPUfuncbody->addChild( dimgriddecl ); 
  debug_fprintf(stderr, "appending DIMGRID repr to setup code\n\n");


  debug_fprintf(stderr, "\nDIMBLOCK\n"); 
  // DIMBLOCK 
  int bs1 = 32;
  int bs2 = 1;
  if (cu_tz > 1) { //  || cu_tz_repr) {
    debug_fprintf(stderr, "cu_tz\n"); 
    exit(-1); 
    
  }
  else { 
    debug_fprintf(stderr, "NOT cu_tz\n"); 
    if (cu_tx && cu_ty) { 
      debug_fprintf(stderr, "cu_tx && cu_ty\n"); 
      bs1 = cu_tx;
      bs2 = cu_ty; 
    }
    else if (cu_tx_repr && cu_ty_repr) { 
      debug_fprintf(stderr, "cu_tx && cu_ty REPR\n"); 
      exit(-1); 
    }
    
  }
  


  chillAST_CallExpr *CE2 = new chillAST_CallExpr( dimbuiltin, NULL );
  CE2->addArg( new chillAST_IntegerLiteral( bs1 ));
  CE2->addArg( new chillAST_IntegerLiteral( bs2 ));
  chillAST_VarDecl *dimblockdecl = new chillAST_VarDecl( "dim3", "dimBlock", "", NULL );
  dimblockdecl->setInit(CE2);
  
  CPUfuncbody->addChild( dimblockdecl ); 


  // kernel call 
  debug_fprintf(stderr, "KERNEL CALL\n"); 
  chillAST_CallExpr *kcall = new chillAST_CallExpr( GPUKernel,  CPUfuncbody);
  kcall->grid = dimgriddecl; 
    kcall->block =  dimblockdecl; 
  debug_fprintf(stderr, "kernel function parameters\n"); 
  for (int i = 0; i < arrayVars.size(); i++) { 
    //Throw in a type cast if our kernel takes 2D array notation
    //like (float(*) [1024])
    
    if (arrayVars[i].tex_mapped || arrayVars[i].cons_mapped) continue;

    chillAST_VarDecl *v = arrayVars[i].vardecl;
    chillAST_VarDecl *param = arrayVars[i].CPUside_param;

    //debug_fprintf(stderr, "numdimensions %d\n", param->numdimensions); 
    
    if (param->numdimensions > 1) { 
      debug_fprintf(stderr, "array Var %d %s is multidimensional\n",i, v->varname);
      //debug_fprintf(stderr, "underlying type %s\narraypart %s\n", v->underlyingtype, v->arraypart); 
      char line[128];
      sprintf(line, "%s (*)", v->underlyingtype ); 
      //debug_fprintf(stderr, "line '%s'\n", line);
      // we'll pass in a cast of the variable instead of just the variable.
      for (int i=1; i<param->numdimensions; i++) { 
        int l = strlen(line);
        //debug_fprintf(stderr, "l %d\n", l); 
        char *ptr = &line[l];
        //debug_fprintf(stderr, "[%d]", param->arraysizes[i]); 
        sprintf(ptr, "[%d]", param->arraysizes[i]); 
        //debug_fprintf(stderr, "i %d line '%s'\n", i, line);
        chillAST_CStyleCastExpr *CE = new chillAST_CStyleCastExpr( line, v, NULL );
        kcall->addArg( CE );
      }
      //int l = strlen(line);
      //char *ptr = line + l;
      //sprintf(ptr, ")");
      //debug_fprintf(stderr, "line '%s'\n", line); 
      
    }
    else { 
      debug_fprintf(stderr, "array Var %d %s is NOT multidimensional\n",i, v->varname);

      // we just need a decl ref expr inserted as the parameter/argument
      // when it prints, it will print just the array name
      chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( v, NULL);
      kcall->addArg( DRE );
    }
  }


   CPUfuncbody->addChild( kcall );           
  //kcall->addArg( 



  debug_fprintf(stderr, "\nfreeing Cuda variables\n"); 
  //cuda free variables
  for (int i = 0; i < arrayVars.size(); i++) {
    debug_fprintf(stderr, "arrayVar %d\n", i); 

    // Memcopy back if we have an output 
    if (arrayVars[i].out_data) {

      chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( arrayVars[i].vardecl, CPUfuncbody ); 
      chillAST_CudaMemcpy *cmemcpy = new chillAST_CudaMemcpy( (chillAST_VarDecl*)arrayVars[i].out_data, // wrong info
                                                              arrayVars[i].vardecl, 
                                                              arrayVars[i].size_expr, "cudaMemcpyDeviceToHost"); 
      CPUfuncbody->addChild( cmemcpy );
    }

    // CudaFree the variable
    chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( arrayVars[i].vardecl, CPUfuncbody ); 
    chillAST_CudaFree *cfree = new chillAST_CudaFree( arrayVars[i].vardecl, CPUfuncbody ); 
    CPUfuncbody->addChild( cfree );
    
  }
  //CPUsidefunc->print(); fflush(stdout); 
  //GPUKernel->print();   fflush(stdout); 




  debug_fprintf(stderr, "BUILD THE KERNEL\n"); 

  //Extract out kernel loop  (somewhat misnamed. This is NOT the body of the GPUKernel YET) 
  chillAST_node *kernelloop = getCode(  ); 
  debug_fprintf(stderr, "loop_cuda_rose.cc L1669 returned from getCode()\n");

  //debug_fprintf(stderr, "loop_cuda_chill.cc L1685  kernelloop =\n");
  //GPUKernel->getBody()->print(); fflush(stdout);
  //debug_fprintf(stderr, "\n\n"); 
         
  debug_fprintf(stderr, "loop_cuda_rose.cc L1685   kernelloop = \n");
  kernelloop->print(); 
  debug_fprintf(stderr, "\n\n"); 

  debug_fprintf(stderr, "%d arrayvars\n", arrayVars.size());  
  
  // this should just be sitting in a member of arrayVars
  std::map<std::string, chillAST_VarDecl*> loop_vars;
  for (int i = 0; i < arrayVars.size(); i++) {
    debug_fprintf(stderr, "arrayVars[%d]  name %s\n", i, arrayVars[i].original_name.c_str()); 
    //if (arrayVars[i].in_data)  debug_fprintf(stderr, "input ");
    //if (arrayVars[i].out_data)  debug_fprintf(stderr, "output ");
    //debug_fprintf(stderr, "\n");

    chillAST_VarDecl *d = GPUKernel->hasParameterNamed( arrayVars[i].original_name.c_str() ); 
    if (d) { 
      debug_fprintf(stderr, "inserting %s into loop_vars\n", arrayVars[i].original_name.c_str()); 
      loop_vars.insert(std::pair<std::string, chillAST_VarDecl*>(std::string(arrayVars[i].original_name), d));
    }
  }
  
  debug_fprintf(stderr, "\nfind variables used in the kernel (?)\n"); 

  // find all variables used in the function
  vector<chillAST_VarDecl*> decls;
  kernelloop->gatherVarDecls( decls );
  debug_fprintf(stderr, "%d variables in kernel\n", decls.size()); 
  for (int i=0; i<decls.size(); i++) { 
    debug_fprintf(stderr, "%s\n", decls[i]->varname); 
  }

  int nump = GPUKernel->parameters.size();
  debug_fprintf(stderr, "\n%d parameters to GPUKernel\n", nump); 
  for (int i=0; i<nump; i++) debug_fprintf(stderr, "parameter %s\n",  GPUKernel->parameters[i]->varname );
  debug_fprintf(stderr, "\n"); 

  

  //Figure out which loop variables will be our thread and block dimension variables
  debug_fprintf(stderr, "Figure out which loop variables will be our thread and block dimension variables\n"); 

  //Get our indexes  (threadIdx and blockIdx will replace some loops) 
  std::vector<const char*> indexes;

  if (cu_bx > 1 || cu_bx_repr) {
    indexes.push_back("bx");
    chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.x", "int", GPUKernel );
    chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *bxdecl = new chillAST_VarDecl( "int", "bx", "", GPUKernel );
    GPUKernel->addDecl( bxdecl );
    chillAST_DeclRefExpr *bx = new chillAST_DeclRefExpr( bxdecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( bx, "=", bid ); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(bxdecl); 
    kernelbody->addChild(assign); 
  }

  if (cu_by > 1 || cu_by_repr) {
    indexes.push_back("by");
    chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.y", "int", GPUKernel );
    chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *bydecl = new chillAST_VarDecl( "int", "by", "", GPUKernel );
    GPUKernel->addDecl( bydecl );
    chillAST_DeclRefExpr *by = new chillAST_DeclRefExpr( bydecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( by, "=", bid ); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(bydecl); 
    kernelbody->addChild(assign); 
  }  
  if (cu_tx_repr || cu_tx > 1) {
    //threadsPos = indexes.size();
    indexes.push_back("tx");
    chillAST_VarDecl *tiddecl = addBuiltin( "threadIdx.x", "int",     GPUKernel);
    chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( tiddecl ); 
    chillAST_VarDecl *txdecl = new chillAST_VarDecl( "int", "tx", "", GPUKernel);
    GPUKernel->addDecl( txdecl );
    chillAST_DeclRefExpr *tx = new chillAST_DeclRefExpr( txdecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( tx, "=", tid ); 
    assign->print(0, stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(txdecl); 
    kernelbody->addChild(assign); 
  }
  if (cu_ty_repr || cu_ty > 1) {
    indexes.push_back("ty");
    chillAST_VarDecl *biddecl = addBuiltin( "threadIdx.y", "int", GPUKernel );
    chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *tydecl = new chillAST_VarDecl( "int", "ty", "", GPUKernel );
    GPUKernel->addDecl( tydecl );
    chillAST_DeclRefExpr *ty = new chillAST_DeclRefExpr( tydecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( ty, "=", tid ); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(tydecl); 
    kernelbody->addChild(assign); 
  }
  if (cu_tz_repr || cu_tz > 1) {
    indexes.push_back("tz");
    chillAST_VarDecl *biddecl = addBuiltin( "threadIdx.z", "int", GPUKernel );
    chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *tzdecl = new chillAST_VarDecl( "int", "tz", "", GPUKernel );
    GPUKernel->addDecl( tzdecl );
    chillAST_DeclRefExpr *tz = new chillAST_DeclRefExpr( tzdecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( tz, "=", tid ); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(tzdecl); 
    kernelbody->addChild(assign); 
  }


  debug_fprintf(stderr, "\n"); 
  for (int i = 0; i < indexes.size(); i++) {
    debug_fprintf(stderr, "indexes[%i] = '%s'\n", i, indexes[i] ); 
  }

  debug_fprintf(stderr, "\nbefore swapVarReferences(), code is\n{\n"); 
  kernelbody->print();

  debug_fprintf(stderr, "}\n\nswapVarReferences()\n"); 
  //swapVarReferences( kernelloop, GPUKernel );

  debug_fprintf(stderr, "\nafter swapVarReferences(), code is\n"); 
  kernelbody->print();
  debug_fprintf(stderr, "\n\n");
  
  
  debug_fprintf(stderr, "now replace indexes ... (and add syncs)\n"); 
  findReplacePreferedIdxs( kernelloop, GPUKernel );
  debug_fprintf(stderr, "DONE WITH replace indexes ... (and add syncs)\n"); 

  debug_fprintf(stderr, "\nswapped 2\nshould have syncs\nshould have indexes replaced by bx, tx, etc \n\n"); 
  kernelloop->print();

  // now remove loops that will be done by spreaking the loop count across cores
  // these are loops that have out indeces gathered above aas loop variables
  debug_fprintf(stderr, "removing loops for variables that will be determined by core index\n"); 
  chillAST_CompoundStmt *CS = new chillAST_CompoundStmt();
  
  CS->addChild( kernelloop ); // in case top level loop will go away
  //debug_fprintf(stderr, "arbitrary compoundstmt 0x%x to hold child kernelloop  0x%x\n", CS, kernelloop); 
  for (int i = 0; i < indexes.size(); i++) {
    debug_fprintf(stderr, "\nindexes[%i] = '%s'\n", i, indexes[i] ); 
    debug_fprintf(stderr, "forReduce()\n");
    
    kernelloop->loseLoopWithLoopVar( strdup(indexes[i]) ); 
  }


  


  debug_fprintf(stderr, "END cudaize codegen V2 (ROSE)\n\n\n");
  debug_fprintf(stderr, "\nat end of cudaize_codegen_v2(), returning\n");
  CS->print(); 


  // variables in CS have not been added to GPUKernel.   fix that
  // should probably do this earlier/elsewhere
  vector<chillAST_VarDecl*> kerneldecls;
  vector<chillAST_VarDecl*> kerneldeclsused;
  GPUKernel->gatherVarDecls( kerneldecls );
  CS->gatherVarUsage( kerneldeclsused );




  debug_fprintf(stderr, "kernel defines %d variables\n", kerneldecls.size()); 
  for (int i=0; i<kerneldecls.size(); i++) { 
    chillAST_VarDecl *vd = kerneldecls[i]; 
    if (vd->isParmVarDecl()) { 
      vd->print(); 
      printf("  (parameter)");
      printf("\n"); fflush(stdout); 
    }
  }
  for (int i=0; i<kerneldecls.size(); i++) { 
    chillAST_VarDecl *vd = kerneldecls[i]; 
    if (vd->isBuiltin()) { 
      vd->print(); 
      printf("  (builtin)");
      printf("\n"); fflush(stdout); 
    }
  }
  for (int i=0; i<kerneldecls.size(); i++) { 
    chillAST_VarDecl *vd = kerneldecls[i]; 
    if ( (!vd->isParmVarDecl()) && (!vd->isBuiltin()) ) { 
      vd->print(); 
      printf("\n"); fflush(stdout); 
    }
  }
         printf("\n"); fflush(stdout); 


         debug_fprintf(stderr, "kernel uses    %d variables\n", kerneldeclsused.size()); 
       for (int i=0; i<kerneldeclsused.size(); i++) { 
         chillAST_VarDecl *vd = kerneldeclsused[i];
         debug_fprintf(stderr, "%2d %s\n", i, vd->varname); 
       }
         debug_fprintf(stderr, "\n\n");  



  int numdeclared = kerneldecls.size(); 
  for (int i=0; i<kerneldeclsused.size(); i++) { 
    chillAST_VarDecl *vd = kerneldeclsused[i];
    bool isdeclared = false;
    debug_fprintf(stderr, "%2d %s ", i, vd->varname); 
    if (vd->isBuiltin())     isdeclared = true;
    if (isdeclared) debug_fprintf(stderr, " (builtin)");
    else { 
      if (vd->isParmVarDecl()) isdeclared = true;
      if (isdeclared) debug_fprintf(stderr, " (param)");
    }
    for (int j=0; j<numdeclared; j++) { 
      if (kerneldeclsused[i] == kerneldecls[j] ) {
        isdeclared = true; 
        debug_fprintf(stderr, " (used %d is decl %d)", i, j); 
        break;
      }
    }
    debug_fprintf(stderr, "\n"); 

    if (!isdeclared) { 
      debug_fprintf(stderr, "declaration for %s needs to be added\n", vd->varname);
      GPUKernel->addChild( vd ); 
    }
  }  



  // take contents of CS and stuff it into GPUKernel, at the end after the declarations we might have just added 
  GPUKernel->addChild( CS ) ; // ?? could do each statement

  //debug_fprintf(stderr, "\nGPU side func is \n");
  //GPUKernel->print();

  return CS; 
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

  if (pi.size() > (n - 1) / 2) { 
    debug_fprintf(stderr, "loop_cuda_rose.cc L 2213, pi.size() %d    n %d   (n+1)/2 %d\n", pi.size(), n, (n+1)/2);
    for (int i=0; i<pi.size(); i++) debug_fprintf(stderr, "pi[%d] = %d\n", i, pi[i]);
    
    throw std::invalid_argument(
      "iteration space dimensionality does not match permute dimensionality");
  }
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
  debug_fprintf(stderr, "LoopCuda::datacopy_privatized_cuda()\n"); 
  int old_stmts = stmt.size();
  printf("before datacopy_privatized:\n");
  printIS(); fflush(stdout); 
  //datacopy_privatized(stmt_num, level, array_name, privatized_levels, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, cuda_shared);
  if (cuda_shared)
    datacopy_privatized(stmt_num, level, array_name, privatized_levels,
                        allow_extra_read, fastest_changing_dimension, padding_stride,
                        padding_alignment, 1);
  else
    datacopy_privatized(stmt_num, level, array_name, privatized_levels,
                        allow_extra_read, fastest_changing_dimension, padding_stride,
                        padding_alignment, 0);
  printf("after datacopy_privatized:\n");
  printIS(); fflush(stdout); 
  
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
  printf("before datacopy:\n"); fflush(stdout); 
  printIS(); fflush(stdout);  
  if (cuda_shared)
    datacopy(stmt_num, level, array_name, allow_extra_read,
             fastest_changing_dimension, padding_stride, padding_alignment,
             1);
  else
    datacopy(stmt_num, level, array_name, allow_extra_read,
             fastest_changing_dimension, padding_stride, padding_alignment,
             0);
  printf("after datacopy:\n"); fflush(stdout); 
  printIS(); fflush(stdout); 
  
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
  fflush(stdout); 
  debug_fprintf(stderr, "\nLoopCuda::unroll_cuda( stmt_num %d,  level %d,  unroll_amount %d )\n",stmt_num, level, unroll_amount); 

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
chillAST_node* LoopCuda::codegen() {
  debug_fprintf(stderr, "LoopCuda::codegen()  (ROSE)\n"); 
  if (code_gen_flags & GenCudaizeV2) { 
    debug_fprintf(stderr, "LoopCuda::codegen() calling cudaize_codegen_v2()\n"); 
    chillAST_node* n = cudaize_codegen_v2(); // this is chill
    //debug_fprintf(stderr, "back from cudaize_codegen_v2()\n"); 
    //n->print(); 
    return n;
  }
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
      std::string name = std::string("chill_t")
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
  debug_fprintf(stderr, "loop_cuda_ROSE.cc printCode(3, false)\n"); 
  printCode(3, false); 
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
                && (*e).is_const_except_for_global(v)
                )
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
  debug_fprintf(stderr, "LoopCuda::printCode( effort %d ) ROSE\n", effort); 

  const int m = stmt.size();
  if (m == 0)
    return;
  const int n = stmt[0].xform.n_out();
  
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
  cout.flush(); 
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



chillAST_node* LoopCuda::getCode(int effort) const {
  debug_fprintf(stderr, "loop_cuda_rose.cc LoopCuda::getCode( effort %d )\n", effort);  
  const int m = stmt.size();
  debug_fprintf(stderr, "%d statements\n", m);
  if (m == 0)
    return NULL;

  const int n = stmt[0].xform.n_out();
  debug_fprintf(stderr, "n %d\n", n); 

  //debug_fprintf(stderr, "stmt[0] "); 
  //stmt[0].code->dump(); 
  //debug_fprintf(stderr, "\n\n"); 
  
  if (last_compute_cgr_ != NULL) {
    delete last_compute_cgr_;
    last_compute_cgr_ = NULL;
  }
  
  if (last_compute_cg_ != NULL) {
    delete last_compute_cg_;
    last_compute_cg_ = NULL;
  }
  
  CG_outputBuilder *ocg = ir->builder();
  debug_fprintf(stderr, "replacing MMGenerateCode\n"); 
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
  
  debug_fprintf(stderr, "known\n"); 
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
  debug_fprintf(stderr, "AST built?\n"); 

  std::vector<CG_outputRepr *> stmts(m);
  for (int i = 0; i < m; i++)
    stmts[i] = stmt[i].code;

  debug_fprintf(stderr, "before printRepr()\n"); 
  CG_outputRepr* repr = last_compute_cgr_->printRepr(ocg, stmts);
  // -- end replacing MMGenerateCode
  debug_fprintf(stderr, "end replacing MMGenerateCode\n"); 


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



  // return the code block we just created
  //((CG_chillRepr *)repr)->printChillNodes();
  vector<chillAST_node*> cnodes = ((CG_chillRepr *)repr)->chillnodes;  // is this doing somethine incredibly expensive?
  int numnodes = cnodes.size();
  debug_fprintf(stderr, "%d chillAST nodes in the vector\n", numnodes);
 
  if (numnodes == 0) return NULL; // ??
  if (numnodes == 1) { 
    // this seems to be the exit path that is actually used
    //debug_fprintf(stderr, "the one node is of type %s\nexiting LoopCuda::getCode()\n", cnodes[0]->getTypeString()); 

    return cnodes[0];
  }

  debug_fprintf(stderr, "more than one chillAST_node.  I'll put them all in a compound statement  UNTESTED\n" ); 
  chillAST_CompoundStmt *CS = new chillAST_CompoundStmt( );
  for (int i=0; i<numnodes; i++) { 
    CS->addChild( cnodes[i] );
    cnodes[i]->setParent( CS );  // perhaps not needed
  }
  return CS;
}



//protonu--adding constructors for the new derived class
LoopCuda::LoopCuda() :
  Loop(), code_gen_flags(GenInit) {
  debug_fprintf(stderr, "making LoopCuda ROSE variety\n"); 

}

LoopCuda::LoopCuda(IR_Control *irc, int loop_num) :
  Loop(irc) {
  debug_fprintf(stderr, "making LoopCuda ROSE variety\n"); 
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
  debug_fprintf(stderr, "\n the size of stmt(initially) is: %d\n", stmt.size());
  for (int i = 0; i < m; i++)
    stmt_nonSplitLevels.push_back(std::vector<int>());
  
  chillAST_FunctionDecl *FD = ((IR_cudaroseCode *) ir)->func_defn;
  function_that_contains_this_loop = FD;  // keep around for later 
  
  chillAST_node *  func_body = FD->getBody(); 
  //debug_fprintf(stderr, "got body\n"); 
  
  std::vector<chillAST_ForStmt *> loops;
  func_body->get_top_level_loops( loops); 
  debug_fprintf(stderr, "%d loops    loop_num %d\n", loops.size(), loop_num); 
  
  std::vector<chillAST_ForStmt *> deeploops;
  loops[loop_num]->get_deep_loops( deeploops); 
  debug_fprintf(stderr, "%d deepest\n", deeploops.size()); 
  
  std::vector<std::string> loopvars;
  for (int i=0; i<deeploops.size(); i++) { 
    deeploops[i]->gatherLoopVars( loopvars );
  }
  //debug_fprintf(stderr, "%d loop variables\n", loopvars.size());
  for (int i=0; i<loopvars.size(); i++) { 
    debug_fprintf(stderr, "index[%d] = '%s'\n", i, loopvars[i].c_str());
  }
  
  debug_fprintf(stderr, "adding IDXNAMES\n"); 
  for (int i = 0; i < stmt.size(); i++){
    idxNames.push_back(loopvars); //refects prefered index names (used as handles in cudaize v2)
    //pushes the entire array of loop vars for each stmt? 
  }
  useIdxNames = false;
}

void LoopCuda::printIS() {
  if (!cudaDebug) return;
  int k = stmt.size();
  for (int i = 0; i < k; i++) {
    printf(" printing statement:%d\n", i);
    stmt[i].IS.print();
  }
  fflush(stdout); 
}

