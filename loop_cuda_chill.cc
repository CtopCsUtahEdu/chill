
/*****************************************************************************
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
 Cudaize methods

 Notes:

 History:
 1/7/10 Created by Gabe Rudy by migrating code from loop.cc
 31/1/11 Modified by Protonu Basu


  this is class LoopCuda, for creating CUDA output.
  traditionally, it contained AST info from the front end compiler.

  loop_cuda_rose.cc   
  loop_cuda_clang.cc  etc

  THIS version contains chillAST  internally, so should not have to change
  if you want to add a new front end parser. 


*****************************************************************************/


#include <malloc.h>

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

//#define DEBUG
using namespace omega;


struct ConstCharStarComparator
{
  bool operator()(const char *s1, const char *s2) const
  {
    return (strcmp(s1, s2) < 0);
  }
};
typedef std::set<const char *, ConstCharStarComparator> what_t;

int charstarvectorindex( const char *str, std::vector< char * > vec ) { 
  for (int i=0; i<vec.size(); i++) if (!strcmp(str, vec[i])) return i;
  return -1; // not found 
}

extern char *k_cuda_texture_memory; //protonu--added to track texture memory type
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
  debug_fprintf(stderr, "LoopCuda::symbolExists( %s )  TODO loop_cuda_chill.cc L89\nDIE\n", s.c_str()); 
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

void LoopCuda::printSyncs() {
  int numsyncs = syncs.size();
  for (int i=0; i<numsyncs; i++) { 
    debug_fprintf(stderr, "%d %d %s\n", i, syncs[i].first, syncs[i].second.c_str());
  }
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
  debug_fprintf(stderr, "renaming inxname[%d][%d] to %s\n", stmt_num, level - 1,  newName.c_str());
  idxNames[stmt_num][level - 1] = newName.c_str();
  
}

enum Type {
  Int
};


// Helper function for wrapInIfFromMinBound
static chillAST_node* getInitBound(chillAST_BinaryOperator* assignOp) {
    assert(assignOp->isAssignmentOp());

    chillAST_node* condition = nullptr;

    if(assignOp->lhs->isBinaryOperator() && assignOp->lhs->isRemOp()) {
        auto rhs   = assignOp->rhs->as<chillAST_node>();
        auto remOp = assignOp->lhs->as<chillAST_BinaryOperator>();

        // lower bound condition for x = a % b:

        auto a         = remOp->lhs->as<chillAST_node>();
        auto b         = remOp->rhs->as<chillAST_node>();
        auto zero      = new chillAST_IntegerLiteral(0);
        condition = new chillAST_BinaryOperator(
                new chillAST_BinaryOperator(
                        new chillAST_BinaryOperator(
                                assignOp->lhs,
                                "-",
                                a),
                        "%",
                        b),
                "==",
                zero);

    }

    assignOp->rhs = new chillAST_IntegerLiteral(0);
    return condition;
}

chillAST_node* wrapInIfFromMinBound(
                         chillAST_node* then_part,
                         chillAST_ForStmt* loop,
                         chillAST_node *symtab,    
                         chillAST_node* bound_sym) {
  debug_fprintf(stderr, "wrapInIfFromMinBound()\n");
  /*
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
      // This relies on the minimum expression being the rhs operand of
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

  // Handle loop init stmts
  chillAST_node* condition = nullptr;
  auto for_init_stmt = loop->getInit();
  if(for_init_stmt->isCompoundStmt()) {
      //TODO: do we even support this ???
      for(auto stmt: for_init_stmt->as<chillAST_CompoundStmt>()->getChildren()) {
          condition = getInitBound(stmt->as<chillAST_BinaryOperator>());
      }
  }
  else {
      condition = getInitBound(for_init_stmt->as<chillAST_BinaryOperator>());
  }

  // Handle loop increment stmts

  auto for_incr_expr = loop->getInc();
  int stepsize = 1;
  if(for_incr_expr->isBinaryOperator()) {
      if(!strcmp(for_incr_expr->as<chillAST_BinaryOperator>()->op, "+=")) {
          stepsize = for_incr_expr->as<chillAST_BinaryOperator>()->rhs->as<chillAST_IntegerLiteral>()->value;

          if(stepsize > 1) {
              auto ti = new chillAST_IfStmt(condition, loop->body, nullptr);
              loop->body = ti;
              then_part = ti;
          }
      }
  }

  // Handle loop test case stmts
  auto for_test_expr = loop->getCond()->as<chillAST_BinaryOperator>();
  auto upper_bound = for_test_expr->lhs->as<chillAST_node>();

  if(upper_bound->isCallExpr()) {
      auto upper_bound_call = upper_bound->as<chillAST_CallExpr>();

      //TODO: if is call to __rose_lt...
  }


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






void findReplacePreferedIdxs(chillAST_node *newkernelcode,
                             chillAST_FunctionDecl *kernel ) { 
  debug_fprintf(stderr, "\nrecursiveFindReplacePreferedIdxs( sync 0 )    perhaps adding syncthreads\n"); 

  chillAST_SymbolTable *symtab = kernel->getSymbolTable();
  debug_fprintf(stderr, "this is the symbol table (things defined in the kernel)\n"); 
  printSymbolTableMoreInfo( symtab); 

  //newkernelcode->findLoopIndexesToReplace( symtab, false ); 
  kernel->getBody()->findLoopIndexesToReplace( symtab, false ); 
}




bool LoopCuda::validIndexes(int stmt_num, const std::vector<std::string>& idxs) {
  for (int i = 0; i < idxs.size(); i++) {
    bool found = false;
    for (int j = 0; j < idxNames[stmt_num].size(); j++) {
      if (strcmp(idxNames[stmt_num][j].c_str(), idxs[i].c_str()) == 0) {
        found = true;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

#define NOTRUNONGPU (ordered_cudaized_stmts[iter].second == -1)
#define RUNONGPU (ordered_cudaized_stmts[iter].second != -1)

// Anand's April 2015 version, with statement numbers and 1 more arg
bool LoopCuda::cudaize_v3(int stmt_num, 
                          std::string kernel_name,
                          std::map<std::string, int> array_sizes,
                          std::vector<std::string> blockIdxs,
                          std::vector<std::string> threadIdxs,
                          std::vector<std::string> kernel_params) {

  // set array sizes
  for(auto dim_pair: array_sizes) {
    this->array_sizes[dim_pair.first] = dim_pair.second;
  }

  cudaDebug = true; 

  debug_fprintf(stderr, "\n(chill) LoopCuda::cudaize_v3( stmt_num %d, kernel_name %s )    ANAND'S\n", stmt_num, kernel_name.c_str());
  
  debug_fprintf(stderr, "blocks= ");
  printVs(blockIdxs);
  printf("thread= ");
  printVs(threadIdxs);
  printf("\n");
  fflush(stdout); 
  
  
  debug_fprintf(stderr, "%d kernel_params:\n", kernel_params.size() ); 
  for (int i = 0; i < kernel_params.size(); i++) { 
    debug_fprintf(stderr, "kernel_parameter: %s\n", kernel_params[i].c_str()); // input to this routine
    // loop_cuda member, a set   std::set<std::string> kernel_parameters;
    kernel_parameters.insert(kernel_params[i]);  
  }
  debug_fprintf(stderr, "\n"); 


  CG_outputBuilder *ocg = ir->builder();



  //this->array_dims.push_back(array_dims);
  if (!validIndexes(stmt_num, blockIdxs)) {
    throw std::runtime_error("One of the indexes in the block list was not "
                             "found in the current set of indexes.");
  }
  if (!validIndexes(stmt_num, threadIdxs)) {
    throw std::runtime_error(
                             "One of the indexes in the thread list was not "
                             "found in the current set of indexes.");
  }

  if (blockIdxs.size() == 0) {
    //  throw std::runtime_error("Cudaize: Need at least one block dimension");
    Vcu_bx.push_back(1); 
    Vcu_bx_repr.push_back(NULL);
    VbxAst.push_back( new chillAST_IntegerLiteral( 1 )); 
  }
  
  if (threadIdxs.size() == 0) {
    //  throw std::runtime_error("Cudaize: Need at least one block dimension");
    Vcu_tx.push_back(1);
    Vcu_tx_repr.push_back(NULL);
    VtxAst.push_back( new chillAST_IntegerLiteral( 1 )); 
  }


  int block_level = 0;
  std::vector<int> thread_and_block_levels;
  //Now, we will determine the actual size (if possible, otherwise
  //complain) for the block dimensions and thread dimensions based on our
  //indexes and the relations for our stmt;
  debug_fprintf(stderr, "loopCuda::cudaize_vX() blockIdxs.size() %d\n", blockIdxs.size()); 
  for (int i = 0; i < blockIdxs.size(); i++) {
    debug_fprintf(stderr, "blockIdxs i %d\n", i); 
    int level = findCurLevel(stmt_num, blockIdxs[i]);

    // integer (constant) and non-constant versions of upper bound
    int ub, lb;
    CG_outputRepr* ubrepr = NULL; 
    chillAST_node *ubcode = NULL; // chillAST version of non-constant upper bound 

    ub = -1;
    lb = 9999;
    debug_fprintf(stderr, "loopCuda::cudaize_vX() extractCudaUB( stmt %d, level %d, ub %d, lb %d)\n", stmt_num, level, ub, lb);
    ubrepr = extractCudaUB(stmt_num, level, ub, lb);

    if (lb != 0) {
      //attempt to "normalize" the loop with an in-place tile and then re-check our bounds
      if (cudaDebug)
        printf(
               "Cudaize: doing tile at level %d to try and normalize lower bounds\n",
               level);
      fflush(stdout);

      tile(stmt_num, level, 1, level, CountedTile);
      idxNames[stmt_num].insert(idxNames[stmt_num].begin() + (level), ""); //TODO: possibly handle this for all sibling stmts
      ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    }

    if (ubrepr) {
      ubcode = ((CG_chillRepr *)ubrepr)->GetCode(); 
      debug_fprintf(stderr, "loop_cuda_chill.cc non-constant upper bound:\n");
      ubcode->print(0, stderr); debug_fprintf(stderr, "\n"); 
    }
    
    if (lb != 0) {
      char buf[1024];
      sprintf(buf,
              "Cudaize: Loop at level %d does not have 0 as its lower bound",
              level);
      throw std::runtime_error(buf);
    }

    // this now does nothing 
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
    fflush(stdout);

    // bx
    if (i == 0) {                                          // bx
      debug_fprintf(stderr, "blockIdxs i == 0, therefore bx?\n"); 
      block_level = level;
      if (ubrepr == NULL) {
        Vcu_bx.push_back(ub + 1); // a constant int ub + 1
        Vcu_bx_repr.push_back(NULL);
        VbxAst.push_back( new chillAST_IntegerLiteral( ub + 1 ));
        cu_bx   = ub + 1;
      } else {
        Vcu_bx.push_back(0);  
        Vcu_bx_repr.push_back( // NON constant ub + 1 
                              (omega::CG_chillRepr*)ocg->CreatePlus(ubrepr, ocg->CreateInt(1)));

        debug_fprintf(stderr, "loop_cuda_chill.cc non-constant upper bound bx :\n");
        chillAST_BinaryOperator *UBP1; // ub + 1
        UBP1 = new chillAST_BinaryOperator( ubcode, "+", new chillAST_IntegerLiteral(1) ); 
        UBP1->print(0, stderr); debug_fprintf(stderr, "\n"); 
        VbxAst.push_back( UBP1 ); // 
      }
      debug_fprintf(stderr, "setting idxNames[%d][%d] to bx\n", stmt_num, level-1); 
      idxNames[stmt_num][level - 1] = "bx";
    }
    // by
    else if (i == 1) {                                // by
      debug_fprintf(stderr, "blockIdxs i == 1, therefore by?\n"); 
      if (ubrepr == NULL) {
        Vcu_by.push_back(ub + 1); // constant 
        Vcu_by_repr.push_back(NULL);
        VbyAst.push_back( new chillAST_IntegerLiteral( ub + 1 ));
        cu_by   = ub + 1;
      } else {
        Vcu_by.push_back(0);
        Vcu_by_repr.push_back( // NON constant ub + 1
                              (omega::CG_chillRepr*)ocg->CreatePlus(ubrepr, ocg->CreateInt(1)));

        debug_fprintf(stderr, "loop_cuda_chill.cc non-constant upper bound bx :\n");
        chillAST_BinaryOperator *UBP1; // ub + 1
        UBP1 = new chillAST_BinaryOperator( ubcode, "+", new chillAST_IntegerLiteral(1) ); 
        UBP1->print(0, stderr); debug_fprintf(stderr, "\n"); 
        VbxAst.push_back( UBP1 ); // 
      }
      debug_fprintf(stderr, "setting idxNames[%d][%d] to by\n", stmt_num, level-1); 
      idxNames[stmt_num][level - 1] = "by";
    }
    
    thread_and_block_levels.push_back(level);
    
  }
  
  if (VbyAst.size() == 0) {  // TODO probably wrong 
    block_level = 0; // ?? no by ?? 
    debug_fprintf(stderr, "THERE IS NO BY (??) \n"); 
    VbyAst.push_back( new chillAST_IntegerLiteral( 1 ));  
  }
  
  // what is this ??? 
  if (Vcu_by.size() == 0 && Vcu_by_repr.size() == 0) block_level = 0; // there are none ??  OR
  else if (Vcu_by.size() > 0 &&  // there are some constant 
           !Vcu_by[Vcu_by.size() - 1] // the last one doesn't exist
           && Vcu_by_repr.size() > 0 // there are non-constant 
           && !Vcu_by_repr[Vcu_by_repr.size() - 1]) // the last one doesn't exist 
    {
       debug_fprintf(stderr, "I think this is impossible\n"); 
       block_level = 0;
    }
  
  // ??? 
  if (blockIdxs.size() < 2) {
    Vcu_by_repr.push_back(NULL);
    Vcu_by.push_back(0);
  }
  // must have at least 2 ??? 




  int thread_level1 = 0;
  int thread_level2 = 0;
  for (int i = 0; i < threadIdxs.size(); i++) {
    int level = findCurLevel(stmt_num, threadIdxs[i]);
    int ub, lb;
    CG_outputRepr* ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    chillAST_node *ubcode = NULL; 

    if (lb != 0) {
      //attempt to "normalize" the loop with an in-place tile and then re-check our bounds
      if (cudaDebug)
        printf(
               "Cudaize: doing tile at level %d to try and normalize lower bounds\n",
               level);
      fflush(stdout);
      tile(stmt_num, level, 1, level, CountedTile);
      idxNames[stmt_num].insert(idxNames[stmt_num].begin() + (level), "");
      ubrepr = extractCudaUB(stmt_num, level, ub, lb);
    }

    if (ubrepr) {
      ubcode = ((CG_chillRepr *)ubrepr)->GetCode(); 
      debug_fprintf(stderr, "loop_cuda_chill.cc non-constant upper bound:\n");
      ubcode->print(0, stderr); debug_fprintf(stderr, "\n"); 
    }
    
    
    if (lb != 0) {
      char buf[1024];
      sprintf(buf,
              "Cudaize: Loop at level %d does not have 0 as its lower bound",
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
    
    if (cudaDebug) { 
      printf("thread idx %s level %d lb: %d ub %d\n",
             threadIdxs[i].c_str(), level, lb, ub);
      fflush(stdout);
    }
    
    if (i == 0) {                                               // tx
      thread_level1 = level;
      if (ubrepr == NULL) {
        Vcu_tx.push_back(ub + 1); // a constant int ub + 1
        Vcu_tx_repr.push_back(NULL);
        VtxAst.push_back( new chillAST_IntegerLiteral( ub + 1 ));
        cu_tx   = ub + 1;
      } else {
        Vcu_tx.push_back(0);
        Vcu_tx_repr.push_back(// NON constant ub + 1 
                             ocg->CreatePlus(ubrepr, ocg->CreateInt(1)));

        debug_fprintf(stderr, "loop_cuda_chill.cc non-constant upper bound tx :\n");
        chillAST_BinaryOperator *UBP1; // ub + 1
        UBP1 = new chillAST_BinaryOperator( ubcode, "+", new chillAST_IntegerLiteral(1) ); 
        UBP1->print(0, stderr); debug_fprintf(stderr, "\n"); 
        VtxAst.push_back( UBP1 ); // 
      }
      debug_fprintf(stderr, "setting idxNames[%d][%d] to tx\n", stmt_num, level-1); 
      idxNames[stmt_num][level - 1] = "tx";
      
    }
    else if (i == 1) {                                               // ty
      thread_level2 = level;
      if (ubrepr == NULL) {
        Vcu_ty.push_back(ub + 1); // a constant int ub + 1
        Vcu_ty_repr.push_back(NULL);
        VtyAst.push_back( new chillAST_IntegerLiteral( ub + 1 ));
        cu_ty   = ub + 1;
      } else {                      // NON constant ub + 1 
        Vcu_ty.push_back(0);
        Vcu_ty_repr.push_back(
                             ocg->CreatePlus(ubrepr, ocg->CreateInt(1)));

        debug_fprintf(stderr, "loop_cuda_chill.cc non-constant upper bound ty :\n");
        chillAST_BinaryOperator *UBP1; // ub + 1
        UBP1 = new chillAST_BinaryOperator( ubcode, "+", new chillAST_IntegerLiteral(1) ); 
        UBP1->print(0, stderr); debug_fprintf(stderr, "\n"); 
        VtyAst.push_back( UBP1 ); // 
      }
      debug_fprintf(stderr, "setting idxNames[%d][%d] to ty\n", stmt_num, level-1); 
      idxNames[stmt_num][level - 1] = "ty";
      
    }
    else if (i == 2) {                                               // tz
      if (ubrepr == NULL) {
        Vcu_tz.push_back(ub + 1); // constant ub + 1
        Vcu_tz_repr.push_back(NULL);
        VtzAst.push_back( new chillAST_IntegerLiteral( ub + 1 ));
        cu_tz   = ub + 1;
      } else {
        Vcu_tz.push_back(0);
        Vcu_tz_repr.push_back( // NON constant ub + 1
                             ocg->CreatePlus(ubrepr, ocg->CreateInt(1)));

        debug_fprintf(stderr, "loop_cuda_chill.cc non-constant upper bound tz :\n");
        chillAST_BinaryOperator *UBP1; // ub + 1
        UBP1 = new chillAST_BinaryOperator( ubcode, "+", new chillAST_IntegerLiteral(1) ); 
        UBP1->print(0, stderr); debug_fprintf(stderr, "\n"); 
        VtzAst.push_back( UBP1 ); // 

      }
      debug_fprintf(stderr, "setting idxNames[%d][%d] to tz\n", stmt_num, level-1); 
      idxNames[stmt_num][level - 1] = "tz";
      
    }
    thread_and_block_levels.push_back(level);
  }

  
  if (Vcu_ty.size() == 0 && Vcu_ty_repr.size() == 0)
    thread_level1 = 0;
  
  if (Vcu_tz.size() == 0 && Vcu_tz_repr.size() == 0)
    thread_level2 = 0;
  
  if (Vcu_ty.size() > 0 && !Vcu_ty[Vcu_ty.size() - 1] && Vcu_ty_repr.size() > 0
      && !Vcu_ty_repr[Vcu_ty_repr.size() - 1]) { 
    debug_fprintf(stderr, "I think this is impossible ty\n");
    thread_level1 = 0;
  }

  if (Vcu_tz.size() > 0 && !Vcu_tz[Vcu_tz.size() - 1] && Vcu_tz_repr.size() > 0
      && !Vcu_tz_repr[Vcu_tz_repr.size() - 1]) {
    debug_fprintf(stderr, "I think this is impossible tz\n");
    thread_level2 = 0;
  }
  

  // ??? 
  if (threadIdxs.size() < 2) {
    Vcu_ty.push_back(0);
    Vcu_ty_repr.push_back(NULL);
    
  }
  if (threadIdxs.size() < 3) {
    Vcu_tz.push_back(0);
    Vcu_tz_repr.push_back(NULL);
  }
  
  //Make changes to nonsplitlevels
  
  std::vector<int> lex = getLexicalOrder(stmt_num);
  
  //cudaized = getStatements(lex, 0);
  
  debug_fprintf(stderr, "ADDING TO CUDAIZED\n"); 
  cudaized.push_back(getStatements(lex, 0));
  
  for (std::set<int>::iterator i = cudaized[cudaized.size() - 1].begin();
       i != cudaized[cudaized.size() - 1].end(); i++) {
    if (block_level) {
      //stmt[i].nonSplitLevels.append((block_level)*2);
      stmt_nonSplitLevels[*i].push_back((block_level) * 2);
    }
    if (thread_level1) {
      //stmt[i].nonSplitLevels.append((thread_level1)*2);
      stmt_nonSplitLevels[*i].push_back((thread_level1) * 2);
    }
    if (thread_level2) {
      //stmt[i].nonSplitLevels.append((thread_level1)*2);
      stmt_nonSplitLevels[*i].push_back((thread_level2) * 2);
    }
    idxNames[*i] = idxNames[stmt_num];
  }
  
  if (cudaDebug) {
    printf("Codegen: current names: ");
    printVS(idxNames[stmt_num]);
    fflush(stdout);
  }
  //Set codegen flag
  code_gen_flags |= GenCudaizeV2;
  debug_fprintf(stderr, "\n(chill) ANAND'S LoopCuda::cudaize_v3() Set codegen flag\n"); 

  //Save array dimension sizes
  this->Varray_dims.push_back(array_sizes);
  Vcu_kernel_name.push_back(kernel_name.c_str()); debug_fprintf(stderr, "Vcu_kernel_name.push_back(%s)\n", kernel_name.c_str()); 

  block_and_thread_levels.insert(
                                 std::pair<int, std::vector<int> >(stmt_num,
                                                                   thread_and_block_levels));

  cu_kernel_name = kernel_name.c_str();
  debug_fprintf(stderr, "cu_kernel_name WILL BE %s\n", cu_kernel_name.c_str()); 

  debug_fprintf(stderr, "\n(chill) ANAND'S LoopCuda::cudaize_v3( stmt_num %d ) DONE\n", stmt_num);

}








#include "cudaize_codegen_v2.cc"




// END "cudaize_codegen_v2.cc"







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
  debug_fprintf(stderr, "LoopCuda::permute_cuda()\n"); 
  printf("curOrder: ");
  printVs(curOrder);
  printf("idxNames: ");
  printVS(idxNames[stmt]);

  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt]);
  bool same = true;
  std::vector<int> pi;
  for (int i = 0; i < curOrder.size(); i++) {
    bool found = false;
    for (int j = 0; j < cIdxNames.size(); j++) {
      if (strcmp(cIdxNames[j].c_str(), curOrder[i].c_str()) == 0) {
        debug_fprintf(stderr, "pushing pi for j+1=%d\n", j+1); 

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
    debug_fprintf(stderr, "pushing pi for i=%d\n", i); 
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
 debug_fprintf(stderr, "\n\nloop_cuda_CHILL.cc L 761, pi.size() %d  > ((n=%d)-1)/2 =  %d\n", pi.size(), n, (n-1)/2);
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




//void LoopCuda::tile_cuda(int stmt, int level, int outer_level) {  // 3 params
//  debug_fprintf(stderr, "LoopCuda::tile_cuda(stmt %d, level %d, outer_level %d)\n",  
//          stmt, level, outer_level);
//  tile_cuda(stmt, level, 1, outer_level, "", "", CountedTile);
//} 


void LoopCuda::tile_cuda(int stmt_num, int level, int outer_level,// 3 numbers, and a method 
                         TilingMethodType method  /* defaults to CountedTile */ ) {
  debug_fprintf(stderr, "LoopCuda::tile_cuda(stmt_num %d, level %d, outer_level %d, TilingMethodType method)\n", stmt_num, level, outer_level); 
  printsyms(); 

  debug_fprintf(stderr, "before tile()\n");
  tile_cuda(stmt_num, level, 1, outer_level, "", "", method);  // calls  version with 4 numbers, 2 strings, and a method 

  debug_fprintf(stderr, "after tile, before getLexicalOrder(stmt_num %d)\n", stmt_num); 
  printsyms(); 
  debug_fprintf(stderr, "LoopCuda::tile_cuda() DONE\n"); 
}


void LoopCuda::tile_cuda(int level, int tile_size, int outer_level,
                         std::string idxName, std::string ctrlName, TilingMethodType method) {
  debug_fprintf(stderr, "LoopCuda::tile_cuda( level %d, tile_size %d, outer_level %d, idxName %s, ctrlName %s, method)\n",
          level, tile_size, outer_level, idxName.c_str(), ctrlName.c_str() ); 
  printsyms(); 
  tile_cuda(0, level, tile_size, outer_level, idxName, ctrlName, method);
}




void LoopCuda::tile_cuda(int stmt_num, int level, int tile_size, int outer_level,
                         std::string idxName, std::string ctrlName, TilingMethodType method) {
  fflush(stdout); 
  debug_fprintf(stderr, "\nLoopCuda::tile_cuda 1234 name name method\n"); 
  printsyms(); 
  //debug_fprintf(stderr, "after printsyms\n"); 
  //debug_fprintf(stderr, "\n%d statements\n", stmt.size());
  //for (int i=0; i<stmt.size(); i++) { 
  //  debug_fprintf(stderr, "%2d   ", i); 
  //  ((CG_chillRepr *)stmt[i].code)->Dump();
  //} 
  //debug_fprintf(stderr, "\n"); 

  //Do regular tile but then update the index and control loop variable
  //names as well as the idxName to reflect the current state of things.
  //printf("tile(%d,%d,%d,%d)\n", stmt_num, level, tile_size, outer_level);
  //printf("idxNames before: ");
  //printVS(idxNames[stmt_num]);
  
  fflush(stdout); 
  debug_fprintf(stderr, "loop_cuda_chill.cc, tile_cuda(), before tile()\n"); 
  tile(stmt_num, level, tile_size, outer_level, method);
  debug_fprintf(stderr, "loop_cuda_chill.cc, tile_cuda(), after tile, before getLexicalOrder(stmt_num %d)\n", stmt_num); 
  
  debug_fprintf(stderr, "after tile, before getLexicalOrder(stmt_num %d)\n", stmt_num); 
  std::vector<int> lex = getLexicalOrder(stmt_num);
  int dim = 2 * level - 1;
  std::set<int> same_loop = getStatements(lex, dim - 1);
  
  for (std::set<int>::iterator j = same_loop.begin(); j != same_loop.end();
       j++) {
    debug_fprintf(stderr, "*j = %d\n", *j); 
    if (idxName.size())
      idxNames[*j][level - 1] = idxName.c_str();
    
    if (tile_size == 1) {
      //potentially rearrange loops
      if (outer_level < level) {
        std::string tmp = idxNames[*j][level - 1];
        for (int i = level - 1; i > outer_level - 1; i--) {
          if (i - 1 >= 0)
            idxNames[*j][i] = idxNames[*j][i - 1];
        }
        idxNames[*j][outer_level - 1] = tmp;
      }
      //TODO: even with a tile size of one, you need a insert (of a dummy loop)
      idxNames[*j].insert(idxNames[*j].begin() + (level), "");
    } else {
      if (!ctrlName.size())
        throw std::runtime_error("No ctrl loop name for tile");
      //insert
      idxNames[*j].insert(idxNames[*j].begin() + (outer_level - 1),
                          ctrlName.c_str());
    }
  }
  
  //printf("idxNames after: ");
  //printVS(idxNames[stmt_num]);
  printsyms(); 
  debug_fprintf(stderr, "LoopCuda::tile_cuda 1234 name name method DONE\n\n\n"); 
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



void LoopCuda::ELLify_cuda(int stmt_num, std::vector<std::string> arrays_to_pad,
                           int pad_to, bool dense_pad, std::string pos_array_name) {
  
  int old_stmts = stmt.size();
  
  ELLify(stmt_num, arrays_to_pad, pad_to, dense_pad, pos_array_name);
  
  int new_stmts = stmt.size();
  for (int i = 0; i < new_stmts - old_stmts; i++) {
    idxNames.push_back(idxNames[stmt_num]);
    stmt_nonSplitLevels.push_back(std::vector<int>());
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
  
  struct mallinfo info;
  
  bool same_as_source = false;
  int new_stmts = stmt.size();
  
  int size = syncs.size();
  //debug_fprintf(stderr, "BEFORE LOOP syncs size %d\n", size); 
  //printSyncs(); 
  
  for (int i = old_stmts; i < new_stmts; i++) {
    info = mallinfo();
    debug_fprintf(stderr, "i=%d/%d\n", i, new_stmts-1); 
    debug_fprintf(stderr, "total allocated space:  %llu bytes\n", info.uordblks);
    //Check whether we had a sync for the statement we are unrolling, if
    //so, propogate that to newly created statements so that if they are
    //in a different loop structure, they will also get a syncthreads
    size = syncs.size();
    //debug_fprintf(stderr, "syncs size %d\n", size); 
    
    for (int j = 0; j < size; j++) {
      if (syncs[j].first == stmt_num) { 
        //debug_fprintf(stderr, "ADDING A SYNCTHREADS, for stmt unrolling  j %d/%d\n", j, size-1); 
        //syncs.push_back(make_pair(i, syncs[j].second));
        addSync( i, syncs[j].second ); 
      }
    }
    //printSyncs(); 
    
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
#ifdef INTERNALS_ROSE 
  // so far, memory mapping to a type is rose-dependent 
  if (!texture)
    texture = new texture_memory_mapping(true, array_name);
  else
    texture->add(array_name);
#endif   
}



void LoopCuda::copy_to_constant(const char *array_name) {
  
#ifdef INTERNALS_ROSE 
  // so far, dependent on rose 
  if(!constant_mem)
    constant_mem = new constant_memory_mapping(true, array_name);
  else
    constant_mem->add(array_name);
#endif
  
}



//protonu--moving this from Loop
chillAST_node* LoopCuda::codegen() {
  debug_fprintf(stderr, "LoopCuda::codegen()\n"); 
  if (code_gen_flags & GenCudaizeV2) { 
    debug_fprintf(stderr, "LoopCuda::codegen() calling cudaize_codegen_v2()\n"); 
    chillAST_node* n = cudaize_codegen_v2();
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



CG_outputRepr* LoopCuda::extractCudaUB(int stmt_num, 
                                       int level,
                                       int &outUpperBound, 
                                       int &outLowerBound) {
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
  debug_fprintf(stderr, "Use code generation system to build tell us our bound information. (umwut)\n"); 
  checkLoopLevel = level * 2;
  stmtForLoopCheck = 0; 
  upperBoundForLevel = -1;
  lowerBoundForLevel = -1;
  debug_fprintf(stderr, "before printcode(stmt_num, 1,false)\n"); 
  printCode(stmt_num, 3, false);
  debug_fprintf(stderr, "after printcode(stmt_num, 1,false)\n"); 
  checkLoopLevel = 0;
  
  outUpperBound = upperBoundForLevel;
  outLowerBound = lowerBoundForLevel;
  debug_fprintf(stderr, "UL bounds %d %d\n", outUpperBound,  outLowerBound); 
  
  if (outUpperBound == -1) {
    CG_result* temp = last_compute_cgr_;
    CG_outputRepr *UPPERBOUND = NULL;
    CG_outputRepr *LOWERBOUND = NULL;
    while (temp) {
      CG_loop * loop;
      if (loop = dynamic_cast<CG_loop*>(temp)) {
        if (loop->level_ == 2 * level) {
          Relation bound = copy(loop->bounds_);
          Variable_ID v = bound.set_var(2 * level);
          for (GEQ_Iterator e(const_cast<Relation &>(bound).single_conjunct()->GEQs());
               e; e++) {
            if ((*e).get_coef(v) < 0
                //  && (*e).is_const_except_for_global(v)
                )
              UPPERBOUND =
                output_upper_bound_repr(ir->builder(), *e,
                                        v, bound,
                                        std::vector<
                                          std::pair<CG_outputRepr *,
                                                    int> >(
                                                           bound.n_set(),
                                                           std::make_pair(
                                                                          static_cast<CG_outputRepr *>(NULL),
                                                                          0)),
                                        uninterpreted_symbols[stmt_num]);
          }
          if (UPPERBOUND) {
            for (GEQ_Iterator e(
                                
                                const_cast<Relation &>(bound).single_conjunct()->GEQs());
                 e; e++) {
              if ((*e).get_coef(v) > 0)
                //  && (*e).is_const_except_for_global(v))
                LOWERBOUND =
                  output_inequality_repr(ir->builder(),
                                         *e, v, bound,
                                         std::vector<
                                           std::pair<
                                             CG_outputRepr *,
                                             int> >(
                                                    bound.n_set(),
                                                    std::make_pair(
                                                                   static_cast<CG_outputRepr *>(NULL),
                                                                   0)),
                                         uninterpreted_symbols[stmt_num]);
            }
            
            if (LOWERBOUND)
              return ir->builder()->CreateMinus(UPPERBOUND,
                                                LOWERBOUND);
            else
              return UPPERBOUND;
            
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

/*   if (outUpperBound == -1) {
     
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
     && (*e).is_const_except_for_global(v)) {
     debug_fprintf(stderr, "LoopCuda::extractCudaUB returning complicated\n"); 
     return output_upper_bound_repr(ir->builder(), *e, v,
     bound,
     std::vector<std::pair<CG_outputRepr *, int> >(
     bound.n_set(),
     std::make_pair(
     static_cast<CG_outputRepr *>(NULL),
     0)),
     uninterpreted_symbols[stmt_num]);
     }
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
*/ 




void LoopCuda::printCode(int stmt_num, int effort, bool actuallyPrint) const {
  debug_fprintf(stderr, "\n\nLoopCuda::printCode( stmt_num %d, effort %d ) CHILL\n",stmt_num, effort ); 
  
  std::set<int> stmts = getStatements(getLexicalOrder(stmt_num), 0);
  for (std::set<int>::iterator it = stmts.begin(); it != stmts.end(); it++) {
    debug_fprintf(stderr, "stmt %d\n", *it); 
  }
  
  const int m = stmts.size(); // was stmt.size()
  debug_fprintf(stderr, "m %d statements\n", m); 
  if (m == 0)
    return;
  const int n = stmt[stmt_num].xform.n_out();
  debug_fprintf(stderr, "xforms? in stmt %d   n %d\n", stmt_num, n); 
  
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
  std::vector<std::vector<std::string> > loopIdxNames_;
  std::vector<std::pair<int, std::string> > syncs_;
  
  std::vector<Relation> IS;
  std::vector<Relation> xforms;
  std::vector<std::vector<int> > nonSplitLevels;
  int count = 0;
  
  debug_fprintf(stderr, "\nBUILDING loopIdxNames_\n"); 
  for (std::set<int>::iterator i = stmts.begin(); i != stmts.end(); i++) {
    IS.push_back(stmt[*i].IS);
    xforms.push_back(stmt[*i].xform);
    debug_fprintf(stderr, "xforms.push( stmt[%d].xform);\n", *i ); 
    if (stmt_nonSplitLevels.size() > *i)
      nonSplitLevels.push_back(stmt_nonSplitLevels[*i]);

    debug_fprintf(stderr, "adding loopIdxNames_[%d]\n", loopIdxNames_.size()); 
    printVS(idxNames[*i]);
    loopIdxNames_.push_back(idxNames[*i]);
    for (int j = 0; j < syncs.size(); j++) {
      if (syncs[j].first == *i) {
        std::pair<int, std::string> temp;
        temp.first = count;
        temp.second = syncs[j].second;
        syncs_.push_back(temp);
      }
      
    }
    
    count++;
  }

  debug_fprintf(stderr, "\nloopIdxNames.size() %d\n", loopIdxNames_.size());
  for (int i=0; i<loopIdxNames_.size(); i++) { 
    for (int j=0; j<loopIdxNames_[i].size(); j++) { 
      debug_fprintf(stderr, "loopIdxNames_[i %d][j %d]  %s\n", i, j,loopIdxNames_[i][j].c_str() ); 
    }
    debug_fprintf(stderr, "\n"); 
  }
  //debug_fprintf(stderr, "\n");
  
  // anand removed 
  //for (int i = 0; i < m; i++) {
  //  IS[i] = stmt[i].IS;
  //  xforms[i] = stmt[i].xform;
  //  nonSplitLevels[i] = stmt_nonSplitLevels[i];
  //} 
  
  
  Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
  
  debug_fprintf(stderr, "in LoopCuda::printCode, last_compute_cg_ = new CodeGen()\n");
  debug_fprintf(stderr, "IS %d  xforms %d\n", IS.size(), xforms.size()); 
  last_compute_cg_ = new CodeGen(xforms, IS, known, nonSplitLevels, 
                                 loopIdxNames_, syncs);
  debug_fprintf(stderr, "in LoopCuda::printCode, back from CodeGen()\n\n"); 
  
  delete last_compute_cgr_;  // this was just done  above? 
  last_compute_cgr_ = NULL;
  //}
  
  if (last_compute_cgr_ == NULL || last_compute_effort_ != effort) { // always
    delete last_compute_cgr_;
    last_compute_cgr_ = last_compute_cg_->buildAST(effort); // ? 
    last_compute_effort_ = effort;
  }
  
  //std::vector<CG_outputRepr *> stmts(m);
  //for (int i = 0; i < m; i++)
  //    stmts[i] = stmt[i].code;
  //CG_outputRepr* repr = last_compute_cgr_->printRepr(ocg, stmts);
  // -- end replacing MMGenerateCode
  fflush(stdout); 
  
  debug_fprintf(stderr, "loop_cuda_chill.c line 2086, calling CG_result printString()\n");
  std::string repr = last_compute_cgr_->printString( uninterpreted_symbols_stringrepr );
  fflush(stdout); 
  debug_fprintf(stderr, "loop_cuda_chill.c line 2088, BACK FROM calling CG_result printString()\n");
  
  //if (1 || actuallyPrint) { 
  //debug_fprintf(stderr, "actuallyPrint\n"); 
  std::cout << repr << std::endl;
  std::cout.flush();
  //}
  //else debug_fprintf(stderr, "NOT actuallyPrint\n"); 
  fflush(stdout); 
  //std::cout << static_cast<CG_stringRepr*>(repr)->GetString();
  /*
    for (int i = 1; i <= m; i++)
    delete nameInfo[i];
  */
  
  debug_fprintf(stderr, "LoopCuda::printCode() CHILL DONE\n"); 
  
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


chillAST_node* LoopCuda::getCode(int effort, std::set<int> stmts) const {
  debug_fprintf(stderr, "loop_cuda_chill.cc LoopCuda::getCode( effort %d, set of ints )\n", effort ); 
  
  const int m = stmts.size();
  debug_fprintf(stderr, "%d stmts\n", m);
  if (m == 0) return NULL;
  
  int ref_stmt = *(stmts.begin());
  
  const int n = stmt[ref_stmt].xform.n_out();
  debug_fprintf(stderr, "n %d\n", n); 
  
  if (last_compute_cgr_ != NULL) {
    delete last_compute_cgr_;
    last_compute_cgr_ = NULL;
  }
  
  if (last_compute_cg_ != NULL) {
    delete last_compute_cg_;
    last_compute_cg_ = NULL;
  }
  
  CG_outputBuilder *ocg = ir->builder();
  debug_fprintf(stderr, "in LoopCuda::getCode(), replacing MMGenerateCode (only not really)\n"); 
  std::vector<std::vector<std::string> > loopIdxNames_;
  std::vector<std::pair<int, std::string> > syncs_;
  
  std::vector<Relation> IS;
  std::vector<Relation> xforms;
  std::vector<std::vector<int> > nonSplitLevels;
  int count_ = 0;
  for (std::set<int>::iterator i = stmts.begin(); i != stmts.end(); i++) {
    IS.push_back(stmt[*i].IS);
    xforms.push_back(stmt[*i].xform);
    if (stmt_nonSplitLevels.size() > *i)
      nonSplitLevels.push_back(stmt_nonSplitLevels[*i]);
    loopIdxNames_.push_back(idxNames[*i]);
    for (int j = 0; j < syncs.size(); j++) {
      if (syncs[j].first == *i) {
        std::pair<int, std::string> temp;
        temp.first = count_;
        temp.second = syncs[j].second;
        syncs_.push_back(temp);
      }
      
    }
    
    count_++;
  }
  
  
  debug_fprintf(stderr, "known\n"); 
  Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
  
  last_compute_cg_ = new CodeGen(xforms, IS, known, nonSplitLevels,
                                 loopIdxNames_, syncs_);
  delete last_compute_cgr_;
  last_compute_cgr_ = NULL;
  
  if (last_compute_cgr_ == NULL || last_compute_effort_ != effort) {
    delete last_compute_cgr_;
    last_compute_cgr_ = last_compute_cg_->buildAST(effort);
    last_compute_effort_ = effort;
  }
  debug_fprintf(stderr, "AST built?\n"); 
  
  std::vector<CG_outputRepr *> stmts_(m);
  int count = 0;
  for (std::set<int>::iterator i = stmts.begin(); i != stmts.end(); i++)
    stmts_[count++] = stmt[*i].code;
  CG_outputRepr* repr = last_compute_cgr_->printRepr(ocg, stmts_,
                                                     uninterpreted_symbols);
  // -- end replacing MMGenerateCode
  debug_fprintf(stderr, "end replacing MMGenerateCode\n"); 
  
  
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
  
  debug_fprintf(stderr, "OK, now what?\n"); 
  
  // return the code block we just created
  //((CG_chillRepr *)repr)->printChillNodes();
  vector<chillAST_node*> cnodes = ((CG_chillRepr *)repr)->chillnodes;  // is this doing somethine incredibly expensive?
  int numnodes = cnodes.size();
  //debug_fprintf(stderr, "%d chillAST nodes in the vector\n", numnodes);
  
  if (numnodes == 0) return NULL; // ??
  if (numnodes == 1) { 
    // this seems to be the exit path that is actually used
    debug_fprintf(stderr, "the one node is of type %s\nexiting LoopCuda::getCode()\n", cnodes[0]->getTypeString()); 
    
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



chillAST_node* LoopCuda::getCode(int effort) const {
  debug_fprintf(stderr, "loop_cuda_chill.cc L3527 LoopCuda::getCode( effort %d )\n", effort);
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
  debug_fprintf(stderr, "replacing MMGenerateCode (only not really)\n"); 
  
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
  
  if (last_compute_cgr_ == NULL || last_compute_effort_ != effort) {
    delete last_compute_cgr_;
    last_compute_cgr_ = last_compute_cg_->buildAST(effort);
    last_compute_effort_ = effort;
  }
  debug_fprintf(stderr, "AST built?\n"); 
  
  std::vector<CG_outputRepr *> stmts(m);
  for (int i = 0; i < m; i++)
    stmts[i] = stmt[i].code;
  
  CG_outputRepr* repr = last_compute_cgr_->printRepr(ocg, stmts,uninterpreted_symbols);
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
  
  
  debug_fprintf(stderr, "OK, now what?\n"); 
  
  // return the code block we just created
  //((CG_chillRepr *)repr)->printChillNodes();
  vector<chillAST_node*> cnodes = ((CG_chillRepr *)repr)->chillnodes;  // is this doing somethine incredibly expensive?
  int numnodes = cnodes.size();
  //debug_fprintf(stderr, "%d chillAST nodes in the vector\n", numnodes);
  
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
  debug_fprintf(stderr, "making EMPTY LoopCuda CHILL variety\n"); 
}




LoopCuda::LoopCuda(IR_Control *irc, int loop_num) :
  Loop(irc) { // <-- this does a LOT
  debug_fprintf(stderr, "making LoopCuda CHILL variety\n"); 
  debug_fprintf(stderr, "loop_cuda_chill.cc   LoopCuda::LoopCuda()\n"); 
  
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
#ifdef INTERNALS_ROSE 
  texture = NULL;  // depends on ROSE 
#endif 
  int m = stmt.size();
  debug_fprintf(stderr, "\nLoopCuda the size of stmt(initially) is: %d\n", stmt.size());
  for (int i = 0; i < m; i++)
    stmt_nonSplitLevels.push_back(std::vector<int>());
  
  chillAST_FunctionDecl *FD = ir->GetChillFuncDefinition(); // TODO if this was just ir_code no clang needed
  function_that_contains_this_loop = FD;  // keep around for later 
  
  chillAST_node *  func_body = FD->getBody(); 
  // debug_fprintf(stderr, "got body\n"); 
  
  CPUparamSymtab = &( FD->parameters ); 
  
  //chillAST_CompoundStmt *CS = (chillAST_CompoundStmt *)FD->getBody();
  CPUbodySymtab = FD->getSymbolTable(); // gets it from body 
  
  printsyms(); 
  
  //debug_fprintf(stderr, "param symbol table has %d entries\n", CPUparamSymtab->size()); 
  //printSymbolTable( CPUparamSymtab ); 
  //debug_fprintf(stderr, "body symbol table has %d entries\n", CPUbodySymtab->size()); 
  //printSymbolTable( CPUbodySymta ); 
  
  
  
  std::vector<chillAST_ForStmt *> loops;
  func_body->get_top_level_loops( loops); 
  debug_fprintf(stderr, "%d loops    loop_num %d\n", loops.size(), loop_num); 
  
  std::vector<chillAST_ForStmt *> deeploops;
  //loops[loop_num]->get_deep_loops( deeploops); 
  loops[loop_num]->find_deepest_loops( deeploops);  // loops[loop_num]  is chillAST_ForStmt *

  debug_fprintf(stderr, "%d deepest\n", deeploops.size()); 
  
  std::vector<std::string> loopvars;
  for (int i=0; i<deeploops.size(); i++) { 
    deeploops[i]->gatherLoopVars( loopvars );
  }
  
  debug_fprintf(stderr, "\nloopCuda::loopCuda() %d loop variables\n", loopvars.size());
  for (int i=0; i<loopvars.size(); i++) { 
    debug_fprintf(stderr, "index[%d] = '%s'\n", i, loopvars[i].c_str());
  }
  
  debug_fprintf(stderr, "\nin LoopCuda::LoopCuda   adding IDXNAMES  %d stmts\n", stmt.size()); 
  for (int i = 0; i < stmt.size(); i++){
    idxNames.push_back(loopvars); //refects prefered index names (used as handles in cudaize v2)
    //pushes the entire array of loop vars for each stmt? 
  }

  useIdxNames = false;
}





void LoopCuda::printIS() {
  int k = stmt.size();
  for (int i = 0; i < k; i++) {
    printf(" printing statement:%d\n", i);
    stmt[i].IS.print();
  }
  fflush(stdout);
}


void LoopCuda::flatten_cuda(int stmt_num, std::string idxs,
                            std::vector<int> &loop_levels, std::string inspector_name) {
  debug_fprintf(stderr, "LoopCuda::flatten_cuda()\n"); 
  printsyms(); 

  debug_fprintf(stderr, "\n%d statements\n", stmt.size());
  for (int i=0; i<stmt.size(); i++) { 
    debug_fprintf(stderr, "%2d   ", i); 
    ((CG_chillRepr *)stmt[i].code)->Dump();
  }
  debug_fprintf(stderr, "\n"); 



  flatten(stmt_num, idxs, loop_levels, inspector_name);
  debug_fprintf(stderr, "LoopCuda::flatten_cuda() back from flatten()\n"); 
  printsyms(); 
  
  for (std::vector<int>::iterator i = loop_levels.begin();
       i != loop_levels.end(); i++)
    
    kernel_parameters.insert(
                             inspector_name + "." + stmt[stmt_num].IS.set_var(*i)->name());
  
  kernel_parameters.insert(inspector_name + "." "count");
  bool initial_val = false;
  
  idxNames.push_back(idxNames[stmt_num]);
  idxNames[stmt_num + 1].push_back(idxs);
  stmt_nonSplitLevels.push_back(std::vector<int>());
  //syncs.push_back()
  
  printsyms(); 
  debug_fprintf(stderr, "LoopCuda::flatten_cuda() DONE\n");   
}

void LoopCuda::distribute_cuda(std::vector<int> &stmt_nums, int loop_level) {
  
  std::set<int> stmts;
  
  for (int i = 0; i < stmt_nums.size(); i++)
    stmts.insert(stmt_nums[i]);
  
  distribute(stmts, loop_level);
  
  //syncs.push_back()
  
}

void LoopCuda::fuse_cuda(std::vector<int> &stmt_nums, int loop_level) {
  
  std::set<int> stmts;
  
  for (int i = 0; i < stmt_nums.size(); i++)
    stmts.insert(stmt_nums[i]);
  
  fuse(stmts, loop_level);
  
  //syncs.push_back()
  
}

void LoopCuda::shift_to_cuda(int stmt_num, int level, int absolute_position) {
  
  shift_to(stmt_num, level, absolute_position);
  
}



void LoopCuda::scalar_expand_cuda(int stmt_num, std::vector<int> level,
                                  std::string arrName, int memory_type, int padding,
                                  int assign_then_accumulate) {

  debug_fprintf(stderr, "\nLoopCuda::scalar_expand_cuda( )\n"); 

  int oldsize = stmt.size();
  debug_fprintf(stderr, "\nSEC %d statements\n", oldsize);
  for (int i=0; i<stmt.size(); i++) { 
    debug_fprintf(stderr, "%2d   ", i); 
    ((CG_chillRepr *)stmt[i].code)->Dump();
  } 
  debug_fprintf(stderr, "\n"); 

  int old_num_stmts = num_statement();
  scalar_expand(stmt_num, level, arrName, memory_type, padding,  // Loop::scalar_expand - generic. no chill
                assign_then_accumulate);
  int new_num_stmts = num_statement();
  
  std::vector<std::string> namez = idxNames[stmt_num];
  debug_fprintf(stderr, "pushing ALL %d of THESE repeatedly (%d times)???\n", namez.size(), new_num_stmts - old_num_stmts); 
  for (int i = 0; i < namez.size(); i++) debug_fprintf(stderr, "%d %s\n", i, namez[i].c_str()); 
  debug_fprintf(stderr, "\n"); 

  for (int i = 0; i < new_num_stmts - old_num_stmts; i++) {  // ??? 
    idxNames.push_back(idxNames[stmt_num]);
    stmt_nonSplitLevels.push_back(std::vector<int>());
  }

  // these new statements may have array variables that are not delcared anywhere
  // find them? 


  debug_fprintf(stderr, "\nSEC %d (was %d) statements\n", stmt.size(), oldsize);
  for (int i=0; i<stmt.size(); i++) { 
    debug_fprintf(stderr, "%2d   ", i); 
    CG_chillRepr *CR = ((CG_chillRepr *)stmt[i].code);
    CR->Dump();
    chillAST_node *n = CR->GetCode();
    std::vector<chillAST_ArraySubscriptExpr*> arefs;
    n->gatherArrayRefs(arefs, false); 
    debug_fprintf(stderr, "%d array refs\n", arefs.size()); 
    for (int j=0; j<arefs.size(); j++){ 
        arefs[j]->print(0,stderr); debug_fprintf(stderr, "\n"); 
        debug_fprintf(stderr, "base is ");
        chillAST_VarDecl *b = arefs[j]->multibase();
        b->print(0,stderr);
        
        // make sure it's in the symbol table(s) AND in the func ??

    }
  } 
  
  // print func ???
  printsyms(); 
  debug_fprintf(stderr, "\n"); 

  // 
  debug_fprintf(stderr, "func\n"); 
 chillAST_FunctionDecl *fd = ir->GetChillFuncDefinition();
  fd->print(0,stderr); 

}



void LoopCuda::split_with_alignment_cuda(int stmt_num, int level, int alignment,
                                         int direction) {
  
  split_with_alignment(stmt_num, level, alignment, direction);
  idxNames.push_back(idxNames[stmt_num]);
  stmt_nonSplitLevels.push_back(std::vector<int>());
}



void LoopCuda::compact_cuda(int stmt_num, int level, std::string new_array,
                            int zero, std::string data_array) {
  int old_num_stmts = num_statement();
  compact(stmt_num, level, new_array, zero, data_array);
  int new_num_stmts = num_statement();
  int i;
  for (i = 0; i < new_num_stmts - old_num_stmts - 1; i++) {
    idxNames.push_back(idxNames[stmt_num]);
    stmt_nonSplitLevels.push_back(std::vector<int>());
  }
  std::vector<std::string> last_index;
  for (int j = 0; j < idxNames[stmt_num].size() - 1; j++)
    last_index.push_back(idxNames[stmt_num][j]);
  
  idxNames.push_back(last_index);
  stmt_nonSplitLevels.push_back(std::vector<int>());
  
}



void LoopCuda::make_dense_cuda(int stmt_num, int loop_level,
                               std::string new_loop_index) {
  
  make_dense(stmt_num, loop_level, new_loop_index);
  std::vector<std::string> new_idx;
  for (int i = 0; i < loop_level - 1; i++)
    new_idx.push_back(idxNames[stmt_num][i]);
  
  new_idx.push_back(new_loop_index);
  
  for (int i = loop_level - 1; i < idxNames[stmt_num].size(); i++)
    new_idx.push_back(idxNames[stmt_num][i]);
  
  idxNames[stmt_num] = new_idx;
  
}



void LoopCuda::addKnown_cuda(std::string var, int value) {
  
  int num_dim = known.n_set();
  Relation rel(num_dim);
  F_And *f_root = rel.add_and();
  EQ_Handle h = f_root->add_EQ();
  
  Free_Var_Decl *g = NULL;
  for (unsigned i = 0; i < freevar.size(); i++) {
    std::string name = freevar[i]->base_name();
    if (name == var) {
      g = freevar[i];
      break;
    }
  }
  if (g == NULL)
    throw std::invalid_argument("symbolic variable " + var + " not found");
  else {
    h.update_coef(rel.get_local(g), 1);
    h.update_const(-value);
    
  }
  
  addKnown(rel);
  
}



void LoopCuda::skew_cuda(std::vector<int> stmt_num, int level,
                         std::vector<int> coefs) {
  
  std::set<int> stmts;
  for (int i = 0; i < stmt_num.size(); i++)
    stmts.insert(stmt_num[i]);
  
  skew(stmts, level, coefs);
  
}



void LoopCuda::reduce_cuda(int stmt_num, std::vector<int> level, int param,
                           std::string func_name, std::vector<int> seq_level, int bound_level) {
  
  debug_fprintf(stderr, "LoopCuda::reduce_cuda()\n"); 
  debug_fprintf(stderr, "LoopCuda::reduce_cuda( stmt_num %d, level (",stmt_num ); 
  for (int i=0; i<level.size(); i++) {
    debug_fprintf(stderr, "%d, ", level[i]);
  }
  debug_fprintf(stderr, "), param %d, ", param ); 
  debug_fprintf(stderr, "func_name %s, seq_level (", func_name.c_str()); 
  for (int i=0; i<seq_level.size(); i++) {
    debug_fprintf(stderr, "%d, ", seq_level[i]);
  }
  debug_fprintf(stderr, "), bound_level %d)\n", bound_level);

  std::vector<int> cudaized_levels =
    block_and_thread_levels.find(stmt_num)->second;
  debug_fprintf(stderr, "%d cudaized_levels\n", cudaized_levels.size()); 

  reduce(stmt_num, level, param, func_name, seq_level, cudaized_levels,
         bound_level);
  
}



void LoopCuda::peel_cuda(int stmt_num, int level, int amount) {
  debug_fprintf(stderr, "LoopCuda::peel_cuda( stmt_num %d, level %d, amount %d)\n", stmt_num, level, amount); 
  debug_fprintf(stderr, "\n%d statements\n", stmt.size());
  for (int i=0; i<stmt.size(); i++) { 
    debug_fprintf(stderr, "%2d   ", i); 
    ((CG_chillRepr *)stmt[i].code)->Dump();
  }
  debug_fprintf(stderr, "\n"); 



  int old_stmt_num = stmt.size();
  peel(stmt_num, level, amount);
  int new_stmt_num = stmt.size();
  //For all statements that were in this unroll together, drop index name for unrolled level
  for (int i = old_stmt_num; i < new_stmt_num; i++) {
    idxNames.push_back(idxNames[stmt_num]);
    stmt_nonSplitLevels.push_back(std::vector<int>());
  }
  //syncs.push_back()
  
}



bool LoopCuda::cudaize_v2(std::string kernel_name,
                          std::map<std::string, int> array_sizes,
                          std::vector<std::string> blockIdxs,
                          std::vector<std::string> threadIdxs) {
  debug_fprintf(stderr, "\n(chill) LoopCuda::cudaize_v2( NO stmt_num  )    WHY IS THiS GETTING CALLED?\n");
  
  /* 
  int *i=0; 
  int j = i[0]; 
  
  debug_fprintf(stderr, "BEFORE ir->builder()\n"); 
  CG_outputBuilder *ocg = ir->builder();
  
  debug_fprintf(stderr, "AFTER  ir->builder()\n"); 
  int stmt_num = 0;
  //if (cudaDebug) {
  printf("cudaize_v2(%s, {", kernel_name.c_str());
  //for(
  printf("}, blocks={");
  printVs(blockIdxs);
  printf("}, thread={");
  printVs(threadIdxs);
  printf("})\n");
  fflush(stdout); 
  //}
  
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
               "Cudaize1: doing tile at level %d to try and normalize lower bounds lb %d\n",
               level, lb);
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
        cu_bx = ub + 1;  //debug_fprintf(stderr, "cu_bx = %d\n", cu_bx); 
        cu_bx_repr = NULL;
      } else {
        cu_bx = 0;  //debug_fprintf(stderr, "cu_bx = %d\n", cu_bx); 
        cu_bx_repr = (omega::CG_chillRepr*)ocg->CreatePlus(ubrepr, ocg->CreateInt(1));
      }
      idxNames[stmt_num][level - 1] = "bx";
    } else if (i == 1) {
      if (ubrepr == NULL) {
        cu_by = ub + 1;  debug_fprintf(stderr, "cu_by = %d\n", cu_by); 
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
      //if (cudaDebug)
      printf(
             "Cudaize2: doing tile at level %d to try and normalize lower bounds\n",
             level);
      fflush(stdout); 
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
    printf("cudaize_v2: current names: ");
    printVS(idxNames[stmt_num]);
  }
  // Set codegen flag
  code_gen_flags |= GenCudaizeV2;
  
  //Save array dimension sizes
  this->array_dims = array_dims;
  
  cu_kernel_name = kernel_name.c_str();
  debug_fprintf(stderr, "cu_kernel_name WILL BE %s   cudaize_v2 NOT VECTOR\n", cu_kernel_name); 
  
  */
}






chillAST_VarDecl *addBuiltin( char *nameofbuiltin, char *typeOfBuiltin, chillAST_node *somecode) { 
  // create a builtin, like "blockIdx.x", so we can use it in assignment statements
  chillAST_VarDecl *bi = new chillAST_VarDecl( typeOfBuiltin, "", nameofbuiltin); // somecode ); // ->getSourceFile() );
  bi->isABuiltin = true;
  return bi;
}


void swapVarReferences( chillAST_node *newkernelcode,
                        chillAST_FunctionDecl *kernel ) { 
  
  // find what variables are referenced in the new kernel code
  vector<chillAST_VarDecl*> newdecls;
  newkernelcode->gatherVarDecls( newdecls );
  
  debug_fprintf(stderr, "%d variables in kernel\n", newdecls.size()); 
  for (int i=0; i<newdecls.size(); i++) { 
    debug_fprintf(stderr, "variable name %s  ", newdecls[i]->varname); 
    
    chillAST_VarDecl *isParam    = kernel->hasParameterNamed( newdecls[i]->varname ); 
    chillAST_VarDecl *isLocalVar = kernel->funcHasVariableNamed(  newdecls[i]->varname ); 
    
    if (isParam)    debug_fprintf(stderr, "is a parameter\n");
    if (isLocalVar) debug_fprintf(stderr, "is already defined in the kernel\n");
    
    if (!isParam && (!isLocalVar)) { 
      debug_fprintf(stderr, "needed to be added to kernel symbol table\n");
      kernel->addDecl(  newdecls[i] );  // adds to symbol table
      kernel->getBody()->addChild( newdecls[i] );  // adds to body!
    }
  }
}


