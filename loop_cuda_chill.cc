
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

#include <codegen.h>
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


int wrapInIfFromMinBound(chillAST_node* then_part, 
                         chillAST_ForStmt* loop,
                         chillAST_node *symtab,    
                         chillAST_node* bound_sym) {
  debug_fprintf(stderr, "wrapInIfFromMinBound()\n"); exit(-1);    // DFL 

  return 0; 
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
                          std::map<std::string, int> array_dims,
                          std::vector<std::string> blockIdxs, std::vector<std::string> threadIdxs,
                          std::vector<std::string> kernel_params) {

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

    if (i == 0) {                                          // bx
      debug_fprintf(stderr, "blockIdxs i == 0, therefore bx?\n"); 
      block_level = level;
      if (ubrepr == NULL) {
        Vcu_bx.push_back(ub + 1); // a constant int ub + 1
        Vcu_bx_repr.push_back(NULL);
        VbxAst.push_back( new chillAST_IntegerLiteral( ub + 1 )); 
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
    } else if (i == 1) {                                // by
      debug_fprintf(stderr, "blockIdxs i == 1, therefore by?\n"); 
      if (ubrepr == NULL) {
        Vcu_by.push_back(ub + 1); // constant 
        Vcu_by_repr.push_back(NULL);
        VbyAst.push_back( new chillAST_IntegerLiteral( ub + 1 )); 
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
      
    } else if (i == 1) {                                               // ty
      thread_level2 = level;
      if (ubrepr == NULL) {
        Vcu_ty.push_back(ub + 1); // a constant int ub + 1
        Vcu_ty_repr.push_back(NULL);
        VtyAst.push_back( new chillAST_IntegerLiteral( ub + 1 )); 
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
      
    } else if (i == 2) {                                               // tz
      if (ubrepr == NULL) {
        Vcu_tz.push_back(ub + 1); // constant ub + 1
        Vcu_tz_repr.push_back(NULL);
        VtzAst.push_back( new chillAST_IntegerLiteral( ub + 1 )); 
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
  this->Varray_dims.push_back(array_dims);
  Vcu_kernel_name.push_back(kernel_name.c_str()); debug_fprintf(stderr, "Vcu_kernel_name.push_back(%s)\n", kernel_name.c_str()); 

  block_and_thread_levels.insert(
                                 std::pair<int, std::vector<int> >(stmt_num,
                                                                   thread_and_block_levels));

  cu_kernel_name = kernel_name.c_str();
  debug_fprintf(stderr, "cu_kernel_name WILL BE %s\n", cu_kernel_name.c_str()); 

  debug_fprintf(stderr, "\n(chill) ANAND'S LoopCuda::cudaize_v3( stmt_num %d ) DONE\n", stmt_num);

}








//#include "cudaize_codegen_v2.cc"

chillAST_node *LoopCuda::cudaize_codegen_v2() {  // NOT WORKING ON THIS ONE NOW    ANAND'S  ??  
  debug_fprintf(stderr, "\nLoopCuda::cudaize codegen V2 (CHILL)\n");
  for(std::map<std::string, int>::iterator it = array_dims.begin(); it != array_dims.end(); it++)  {
    debug_fprintf(stderr, "array_dims  '%s'  %d\n", it->first.c_str(), it->second);
  }
  debug_fprintf(stderr, "that is the list\n"); 

  CG_chillBuilder *ocg = dynamic_cast<CG_chillBuilder*>(ir->builder());
  if (!ocg) { 
    debug_fprintf(stderr, "no ocg?\n"); 
    return NULL;
  }
  
  
  
  // this routine takes a CPU only function and creates funcs that run the 
  // 'same' code on a GPU.  The result will have one CPU side function, 
  // and one or more GPU kernels.
  chillAST_FunctionDecl *origfunction =  function_that_contains_this_loop; 
  int numparams = origfunction->parameters.size();
  const char *fname = origfunction->functionName;
  chillAST_node *p = origfunction->getParent();
  chillAST_SourceFile *srcfile = origfunction->getSourceFile();
  
  // make a new function that will be the CPU side cuda code
  // it will take the name and parameters from the original C code 
  debug_fprintf(stderr, "creating CPUsidefunc named %s\n", fname); 
  chillAST_FunctionDecl *CPUsidefunc = new chillAST_FunctionDecl(origfunction->returnType, fname, NULL); 

  for (int i=0; i<numparams; i++) { 
    CPUsidefunc->addParameter( origfunction->parameters[i] ) ; 
  }
  chillAST_CompoundStmt *CPUfuncbody =  CPUsidefunc->getBody(); 

  // copy the preprocessing statements from the original (??)
  int numpreproc =  origfunction->preprocessinginfo.size();
  debug_fprintf(stderr, "preprocessinginfo %d\n", numpreproc); 
  if (numpreproc != 0) { 
    debug_fprintf(stderr, "copying %d preprocessing statements from original %s to the new one that calls the GPU\n", numpreproc, fname); 
    for (int i=0; i<numpreproc; i++) { 
      CPUsidefunc->preprocessinginfo.push_back( origfunction->preprocessinginfo[i] );
    }
  }


  
  CPUbodySymtab = origfunction->getSymbolTable();   // local name 
  CPUsidefunc->setSymbolTable(  CPUbodySymtab );
  CPUparamSymtab = CPUsidefunc->getParameterSymbolTable(); // local name 
  
  
  // put this new cpu side function where the old one was in the sourcefile 
  int which = p->findChild( origfunction ); 
  int originalfunctionlocation = which; 
  p->insertChild( which,  CPUsidefunc );
  
  // remove original function  from the source   (bad idea ??) 
  which = p->findChild( origfunction ); 
  debug_fprintf(stderr, "original function  is now child %d of srcfile\n", which);
  p->removeChild( which );  // TODO do this last, because we look for it later
  
  origfunction->print(0,stderr); debug_fprintf(stderr, "\n\n"); 

  debug_fprintf(stderr, "OK, made new (empty) CPU side function %s, removed original function\n",origfunction->functionName );
  p->print(); printf("\n\n\n"); fflush(stdout);
  
  debug_fprintf(stderr, "\nCPUSIDE func:\n"); 
  CPUsidefunc->print(); printf("\n\n"); fflush(stdout); 
  
  
  debug_fprintf(stderr, "this loop is in function %s\n", fname); 
  debug_fprintf(stderr, "function %s has %d parameters:\n",  fname, numparams ); 
  for (int i=0; i< function_that_contains_this_loop->parameters.size(); i++) { 
    debug_fprintf(stderr, "%d/%d  %s\n", i, numparams,  function_that_contains_this_loop->parameters[i]->varname); 
  }
  
  // create a builtin dim3() function    TODO do we need to check if it's already been created?
  chillAST_FunctionDecl *dimbuiltin = new chillAST_FunctionDecl( "dim3", "dim3" );
  dimbuiltin->setBuiltin();
  
  
  //sort cudaized according to lexical order into ordered cudaized statements
  debug_fprintf(stderr, "sort cudaized according to lexical order into ordered cudaized statements\n"); 
  
  debug_fprintf(stderr, "\n%d cudaized statements to be run on the GPU\n",cudaized.size()); 
  std::vector<int> sort_aid;
  std::set<int> all_cudaized_statements;
  
  std::vector<std::pair<std::set<int>, int> > ordered_cudaized_stmts;
  
  for (int i = 0; i < cudaized.size(); i++) {
    int what = get_const(stmt[*(cudaized[i].begin())].xform, 0, Output_Var); // 0 == outermost
    sort_aid.push_back( what ); 
    all_cudaized_statements.insert(cudaized[i].begin(), cudaized[i].end());
  }
  debug_fprintf(stderr, "%d in all_cudaized_statements\n\n", all_cudaized_statements.size()); 
  
  for (int i = 0; i < stmt.size(); i++) {
    
    if (all_cudaized_statements.find(i) == all_cudaized_statements.end()) {
      int j;
      for (j = 0; j < cudaized.size(); j++)
        if (get_const(stmt[i].xform, 0, Output_Var)
            == get_const(stmt[*(cudaized[j].begin())].xform, 0,
                         Output_Var)) {
          cudaized[j].insert(i);
          break;
        }
      if (j == cudaized.size()
          && all_cudaized_statements.find(i)
          == all_cudaized_statements.end())
        sort_aid.push_back(get_const(stmt[i].xform, 0, Output_Var));
      
    }
  }
  
  debug_fprintf(stderr, "sorting ...\n"); 
  std::sort(sort_aid.begin(), sort_aid.end());
  
  for (int i = 0; i < sort_aid.size(); i++) {
    
    int start = i;
    
    while (i + 1 < sort_aid.size() && sort_aid[i + 1] == sort_aid[i])
      i++;
    
    int j;
    for (j = 0; j < cudaized.size(); j++)
      if (get_const(stmt[*(cudaized[j].begin())].xform, 0, Output_Var)
          == sort_aid[start]) {
        ordered_cudaized_stmts.push_back(
                                         std::pair<std::set<int>, int>(cudaized[j], j));
        break;
      }
    if (j == cudaized.size()) {
      std::set<int> temp;
      for (int j = 0; j < stmt.size(); j++) {
        
        if (all_cudaized_statements.find(j)
            == all_cudaized_statements.end())
          if (sort_aid[start]
              == get_const(stmt[j].xform, 0, Output_Var)) {
            
            temp.insert(j);
          }
      }
      ordered_cudaized_stmts.push_back(
                                       std::pair<std::set<int>, int>(temp, -1));
      
    }
  }
  
  debug_fprintf(stderr, "code after\n"); 
  
  
  ////////////////////////////////////////////////////////////////////////////
  
  
  debug_fprintf(stderr, "%d ordered_cudaized_stmts\n", ordered_cudaized_stmts.size() );
  for (int i=0; i<ordered_cudaized_stmts.size(); i++) { 
    debug_fprintf(stderr, "( ");
    for (std::set<int>::iterator j = ordered_cudaized_stmts[i].first.begin(); 
         j != ordered_cudaized_stmts[i].first.end(); j++) {
      debug_fprintf(stderr, "%d ", (*j) );
    } 
    debug_fprintf(stderr, ")     %d\n\n", ordered_cudaized_stmts[i].second); 
  }

  // find pointer-to-int, pointer-to-float? 
  debug_fprintf(stderr, "looking for pointer-to-int, pointer-to-float\n"); 
  std::set<int> ptrs;
  for (int iter = 0; iter < ordered_cudaized_stmts.size(); iter++) {
    debug_fprintf(stderr, "iter %d\n", iter); 
    for (std::set<int>::iterator it =
           ordered_cudaized_stmts[iter].first.begin();
         it != ordered_cudaized_stmts[iter].first.end(); it++) {
      
      std::vector<IR_PointerArrayRef *> ptrRefs = ir->FindPointerArrayRef(
                                                                         stmt[*it].code);
      
      debug_fprintf(stderr, "loop_cuda_XXXX.cc   *it %d,  %d pointer arrayrefs\n", *it, ptrRefs.size()); 
      for (int j = 0; j < ptrRefs.size(); j++)
        for (int k = 0; k < ptr_variables.size(); k++)
          if (ptrRefs[j]->name() == ptr_variables[k]->name())
            ptrs.insert(k);
    }
  }
  debug_fprintf(stderr, "found %d pointers to do mallocs for\n\n", ptrs.size()); 
  
  
  // for each pointer, build malloc( sizeof( int/float ) ) ??
  for (std::set<int>::iterator it = ptrs.begin(); it != ptrs.end(); it++) {
    if (ptr_variables[*it]->elem_type() == IR_CONSTANT_FLOAT) {
      debug_fprintf(stderr, "pointer to float\n");
    } 
    else if (ptr_variables[*it]->elem_type() == IR_CONSTANT_INT) {
      debug_fprintf(stderr, "pointer to INT\n");
    }
    else {
      throw loop_error("Pointer type unidentified in cudaize_codegen_v2!");
    }
    debug_fprintf(stderr, "TODO - DIDN'T ACTUALLY DO THE MALLOC\n");
  } 
  debug_fprintf(stderr, "done making mallocs?\n");
  
  
  
  
  //protonu--adding an annote to track texture memory type
  /*  ANNOTE(k_cuda_texture_memory, "cuda texture memory", TRUE);
      int tex_mem_on = 0;
  */
  
  int tex_mem_on  = 0;
  int cons_mem_on = 0;
  
  debug_fprintf(stderr, "here goes nothing\n"); 
  debug_fprintf(stderr, "%d ordered cudaized stmts\n", ordered_cudaized_stmts.size()); 
  
  std::vector< chillAST_VarDecl * > inspectorargs; // call from main to cpuside inspector has these arguments
  
  for (int iter = 0; iter < ordered_cudaized_stmts.size(); iter++) {
    debug_fprintf(stderr, "HGN iter %d\n", iter);
    
    CG_outputRepr* repr;
    std::vector<VarDefs> arrayVars;
    std::vector<VarDefs> localScopedVars;
    
    std::vector<IR_ArrayRef *> ro_refs; // this one is not used ??? 
    std::vector<IR_ArrayRef *> wo_refs;
    //std::set<std::string> uniqueRefs; // unused ?? 
    //std::set<std::string> uniqueWoRefs;
    
    std::set<const chillAST_VarDecl *> pdSyms;// PD? Parameter definition(?) ....
    std::vector<chillAST_VarDecl *> parameterSymbols; 
    
    
    chillAST_node *code_temp = getCode( 1234, ordered_cudaized_stmts[iter].first);
    
    

    if (code_temp != NULL) {
      
      debug_fprintf(stderr, "\nHGN %d code_temp was NOT NULL\n", iter); 
      printf("\nloop_cuda_chill.cc L903 code_temp:\n"); code_temp->print(); printf("\n\n"); fflush(stdout);
      
      
      debug_fprintf(stderr, "set %d has size %d\n", iter,  ordered_cudaized_stmts[iter].first.size()); 
      
      // find first statement that HAS an inspector?
      std::set<int>::iterator i =
        ordered_cudaized_stmts[iter].first.begin();
      
      debug_fprintf(stderr, "first stmt i is %d\n", *i);
      
      for (; i != ordered_cudaized_stmts[iter].first.end(); i++) {
        if (stmt[*i].has_inspector) { 
          fprintf (stderr, "stmt %d HAS an inspector\n", *i); 
          break;
        }
        else fprintf (stderr, "stmt %d DOES NOT HAVE an inspector\n", *i); 
      }
      
      debug_fprintf(stderr, "i is %d   last is %d\n", *i, *ordered_cudaized_stmts[iter].first.end());
      
      if (i == ordered_cudaized_stmts[iter].first.end()) {  // there was no statement with an inspector
        debug_fprintf(stderr, "no statement had an inspector\n"); 
        if ( NOTRUNONGPU ) { 
          setup_code = ocg->StmtListAppend(setup_code,  // becomes part of setup (after init)
                                           new CG_chillRepr(code_temp));
        }
      } 
      else { // there was an inspector 
        debug_fprintf(stderr, "there was an inspector\n"); 
        
        // create a function that becomes spmv_inspector(  )
        std::set<IR_ArrayRef *> outer_refs;
        for (std::set<int>::iterator j =
               ordered_cudaized_stmts[iter].first.begin();
             j != ordered_cudaized_stmts[iter].first.end(); j++) {
          
          if (stmt[*j].ir_stmt_node != NULL) {
            
            
            // find all loop/if init/cond/iteration in loops this is contained in
            debug_fprintf(stderr, "find all loop/if init/cond/iteration in loops this is contained in\n");
            std::vector<CG_outputRepr *> loop_refs =
              collect_loop_inductive_and_conditionals(stmt[*j].ir_stmt_node);
            debug_fprintf(stderr, "%d loop_refs\n\n", loop_refs.size()); 
            
            // find the array references in  the loop/if 
            for (int i = 0; i < loop_refs.size(); i++) {
              std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(loop_refs[i]);
              
              // make unique list of references (no repeated names) in outer_refs
              for (int l = 0; l < refs.size(); l++) {
                std::set<IR_ArrayRef *>::iterator k =
                  outer_refs.begin();
                for (; k != outer_refs.end(); k++)
                  if ((*k)->name() == refs[l]->name())
                    break;
                if (k == outer_refs.end()) { 
                  debug_fprintf(stderr, "a new array ref\n");
                  outer_refs.insert(refs[l]);
                }
              }
              
            }
            debug_fprintf(stderr, "\n%d non-repeated array name refs\nthey are:\n", outer_refs.size()); 
            for (std::set<IR_ArrayRef *>::iterator k =outer_refs.begin(); k != outer_refs.end(); k++) {
              debug_fprintf(stderr, "%s\n", (*k)->name().c_str());               
            }
            
          }
        }
        //--end
        
        
        char fname[100];
        sprintf(fname, "%s_inspector", (dynamic_cast<IR_chillCode*>(ir))->procedurename); 
        //std::string functionname = string(dynamic_cast<IR_chillCode*>(ir)->procedurename) + "_inspector";
        
        // all these are declared above as well??? 
        debug_fprintf(stderr, "\n\n*** building %s() AST   iter %d\n", fname, iter);
        //chillAST_FunctionDecl *origfunction = function_that_contains_this_loop;
        //chillAST_SourceFile   *srcfile = origfunction->getSourceFile();
        chillAST_FunctionDecl *inspectorFunc = new chillAST_FunctionDecl(strdup("void"), fname,  srcfile ); // this adds inspectorfunc to children of srcfile
 
        debug_fprintf(stderr, "original function was:\n");
        origfunction->print(); printf("\n\n"); fflush(stdout); 
        
        // create a compound statement function body, so we can add vardecls as needed

        debug_fprintf(stderr, "loop_cuda_chill.cc L991  code_temp %s\n", code_temp->getTypeString()); 
        //int *k = 0; int die = k[0];
        

        chillAST_CompoundStmt *inspectorbody;
        
        if (code_temp->isCompoundStmt()) {
          // can we just assign to CPUfunctionBody?  no
          inspectorbody =  (chillAST_CompoundStmt *)code_temp; 
        }
        else {  // old code was just one statement 
          inspectorbody =  new chillAST_CompoundStmt;
          inspectorbody->addChild( code_temp );
        }
        inspectorFunc->setBody( inspectorbody );  // this wlil ruin symbol tables 
        int numdefs = 0; // definitions inside inspectorbody
        
        
        // find which NONarray parameters are used in the code (?)
        code_temp->print(); printf("\n"); fflush(stdout);
        
        vector<chillAST_VarDecl*>  decls;
        code_temp->gatherVarUsage( decls );
        debug_fprintf(stderr, "%d vars used in code_temp\n\n", decls.size());
        for (int i=0; i<decls.size(); i++) { 
          printf("sym "); decls[i]->print(); printf(" \n"); fflush(stdout); 
        }
        printf("\n"); fflush(stdout); 
        
        // any of those that were parameters in the original code should 
        // be added as parameters in the new function
        // we can only check parameter NAMES? TODO  
        // actually, that is what we need!
        // the var in code temp is a clone, but we want to know 
        // about parameters with that NAME in the original (??? !!)
        for (int i=0; i<decls.size(); i++) { 
          //printf("%p  ",decls[i]);decls[i]->dump();printf("\n");fflush(stdout);
          // if (decls[i]->isAParameter) {  // this doesn't work, because we 
          // are dealing with a copy that has substitutions, and doesn't 
          // refer to the original vardecl (should it?)
          
          char *name = decls[i]->varname;
          debug_fprintf(stderr, "\nchecking for a parameter %d named %s in origfunction\n", i, name); 
          
          
          if (origfunction->hasParameterNamed( name )) { 
            debug_fprintf(stderr, "%s was a parameter in the original. adding it to inspectorFunc parameters\n", name);
            
            // this decl should have no parent ??
            // see chill_ast.cc addParameter. 
            //if (decls[i]->parent) { 
            //  debug_fprintf(stderr, "UHOH, this vardecl for %s I'm about to make a parameter of %s already had a parent??\n", name, fname); 
            //  exit(-1); 
            //} 
            
            // have to clone the vardecl and replace it in the code?
            chillAST_VarDecl *param = (chillAST_VarDecl *)decls[i]->clone();
            inspectorargs.push_back( decls[i] ); 
            inspectorFunc->addParameter( param ); 
            code_temp->replaceVarDecls( decls[i], param ); 
          }
          else { 
            debug_fprintf(stderr, "there was no parameter named %s in origfunction\n", name);  
            decls[i]->dump(); fflush(stdout);
            decls[i]->print(); printf("\n"); fflush(stdout);
            if (decls[i]->isAStruct()) { 
              debug_fprintf(stderr, "%s it's a struct\n", name); 
              
              // somehow, this is enough to mean that we need to pass it 
              //   in as a parameter?  TODO 
              // make a NEW vardecl, for the parameter
              chillAST_VarDecl *param = (chillAST_VarDecl *)decls[i]->clone();
              param->setStruct( true ); // clone should handle this ???

              inspectorargs.push_back( decls[i] ); 
              inspectorFunc->addParameter( param );
              param->setByReference( true );
              param->print(); printf("\n"); fflush(stdout);
              
              // swap out the vardecl in the declrefexp in the code?
              code_temp->replaceVarDecls( decls[i], param ); 
            }
            else { 
              // this will not be a parameter, we need to make the vardecl 
              // be inside the body of the function
              debug_fprintf(stderr, "adding VarDecl for %s inside inspectorbody\n", decls[i]->varname); 
              chillAST_VarDecl *vd = (chillAST_VarDecl *)decls[i]->clone();
              inspectorbody->insertChild( numdefs++, vd );
            }
          }
        }
        
        
        
        
        
        
        
        debug_fprintf(stderr, "\n\n*** processing %d outer_refs into parameters\n",outer_refs.size()); 
        for (std::set<IR_ArrayRef *>::iterator l = outer_refs.begin();
             l != outer_refs.end(); l++) {
          
          chillAST_ArraySubscriptExpr *ASE =((IR_chillArrayRef *)(*l))->chillASE;
          chillAST_VarDecl *vd = ASE->multibase();
          char *vname = vd->varname; 
          debug_fprintf(stderr, "vname %s\n", vname); 
          
          if ( chillAST_VarDecl *p = origfunction->hasParameterNamed( vname )) { 
            debug_fprintf(stderr, "%s was a parameter in the original. adding it\n", vname); 
            // we need to make a new version of this parameter. The old 
            // one had a size. this will just be a pointer. (??) TODO
            // for example, int index[494]  in the original becomes 
            // int *index in this function 
            chillAST_VarDecl *newparam = new chillAST_VarDecl( vd->vartype, vd->varname, "*", NULL);
            newparam->print(); printf("\n"); fflush(stdout);
            
            inspectorargs.push_back( p ); 
            inspectorFunc->addParameter( newparam );  
            newparam->print(); printf("\n"); fflush(stdout);
            newparam->dump(); printf("\n"); fflush(stdout);
            
            // substitute this parameter vardecl for the old one in the code
            code_temp->replaceVarDecls( vd, newparam ); 
            
          }
          
        }
        
        
        
        
        
        
        
        // TODO make sure vardecls are included ?
        debug_fprintf(stderr, "\nTHIS IS inspectorFunc\n");
        if (inspectorFunc->parent) {
          debug_fprintf(stderr, "IT HAS A PARENT of type %s\n", inspectorFunc->parent->getTypeString());
        }
        inspectorFunc->print(); printf("\n\n"); fflush(stdout); 
        
        debug_fprintf(stderr, "building call to %s\n", inspectorFunc->functionName); 
        chillAST_CallExpr *CE = new chillAST_CallExpr( inspectorFunc, NULL );
        
        debug_fprintf(stderr, "parameters will be\n");
        for (int i=0; i<inspectorargs.size(); i++) { 
          inspectorargs[i]->print(); printf("\n"); fflush(stdout); 
          CE->addArg( new chillAST_DeclRefExpr( inspectorargs[i], NULL ));
        }
        printf("\n"); fflush(stdout);
        
        
        debug_fprintf(stderr, "call to %s is:\n",inspectorFunc->functionName ); 
        CE->print(); printf(""); fflush(stdout);
        
        debug_fprintf(stderr, "adding inspectorfunc call to setup_code\n"); 
        setup_code = ocg->StmtListAppend(setup_code,
                                         new CG_chillRepr(CE));
        
        
      } // there was an inspector 
      
    } // code_temp not NULL
    else { 
      debug_fprintf(stderr, "HGN %d code_temp WAS NULL\n", iter); 
      exit(-1);
    }
    

    
    // still in int iter loop 
    debug_fprintf(stderr, "\n\n\n\n\n*** dimgrid dimblock\n");
    char gridName[20];
    char blockName[20];
    sprintf(gridName,  "dimGrid%i\0",  iter);
    sprintf(blockName, "dimBlock%i\0", iter); 
    

    // still in int iter loop 2
    chillAST_FunctionDecl *GPUKernel = NULL;  // kernel for replacing ONE STATEMENT 

    if ( RUNONGPU ) { 
      debug_fprintf(stderr, "\ncreating the kernel function\n"); 
      
      chillAST_FunctionDecl *origfunction =  function_that_contains_this_loop; 
      const char *fname = origfunction->functionName;
      int numparams = origfunction->parameters.size();
      debug_fprintf(stderr, "\n\noriginal func has name %s  %d parameters\n", fname, numparams); 
      
      chillAST_node *p = origfunction->getParent();
      //debug_fprintf(stderr, "parent of func is a %s with %d children\n", p->getTypeString(), p->getNumChildren()); 
      // defined above (twice!) chillAST_SourceFile *srcfile = origfunction->getSourceFile();
      //debug_fprintf(stderr, "srcfile of func is %s\n", srcfile->SourceFileName );
      
      debug_fprintf(stderr, "\n\nkernel named %s\n", 
              //cu_kernel_name.c_str()
              Vcu_kernel_name[iter].c_str());

      GPUKernel =  new  chillAST_FunctionDecl( origfunction->returnType /* ?? */,
                                               //cu_kernel_name.c_str(), // fname, 
                                               Vcu_kernel_name[iter].c_str(),
                                               p);  // 
      GPUKernel->setFunctionGPU();   // this is something that runs on the GPU 
      

      std::vector< chillAST_VarDecl*>  GPUKernelParams; // 

      debug_fprintf(stderr, "ordered_cudaized_stmts[iter %d] WILL become a cuda kernel\n", iter); 
      
      // find the array refs in code_temp 
      //std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(code_temp); 
      //debug_fprintf(stderr, "%d array refs in CPUfunc\n", refs.size()); 
      //chillAST_node *code_temp
      std::vector<chillAST_ArraySubscriptExpr*> refs;  
      code_temp->gatherArrayRefs( refs, 0 );
      
      debug_fprintf(stderr, "%d array refs in CPUfunc\n", refs.size());
      //for (int i=0; i<refs.size(); i++) { 
      //  refs[i]->print(); printf("\n"); fflush(stdout); 
      //} 
      debug_fprintf(stderr, "code is:\n");
      //if (NULL == code_temp->parent) debug_fprintf(stderr, "code temp HAS NO PARENT\n");
      //code_temp->print(); printf("\n"); fflush(stdout);
      
      
      debug_fprintf(stderr, "\nchecking if var refs are a parameter\n"); 
      debug_fprintf(stderr, "\nthere are %d refs\n", refs.size()); 
      for (int i = 0; i < refs.size(); i++) {
        refs[i]->print(); printf("\n"); fflush(stdout); 
        debug_fprintf(stderr, "\nvar %d  ref\n", i);
        debug_fprintf(stderr, "ref %d   %s wo UNK\n", i, refs[i]->basedecl->varname); // , refs[i]->is_write());
      }
      
      fflush(stdout); 
      //If the array is not a parameter, then it's a local array and we
      //want to recreate it as a stack variable in the kernel as opposed to
      //passing it in.
      
      //  the thing that is being checked is (was?)  wrong
      
      std::vector<IR_chillArrayRef *> ro_refs; // this one is used 
      std::vector<IR_chillArrayRef *> wo_refs;
      
      //std::vector<chillAST_ArraySubscriptExpr *> ro_refs;// try chillAST versions
      //std::vector<chillAST_ArraySubscriptExpr *> wo_refs;
      
      // this is stupid. Just make an array of character strings
      //std::set<std::string> uniqueRefs;
      //std::set<std::string> uniqueWoRefs;
      //what_t uniqueRefs;
      //what_t uniqueWoRefs;
      std::vector< char * >  uniqueRefs;
      
      for (int i = 0; i < refs.size(); i++) {
        debug_fprintf(stderr, "\nvar %d  ref  in iter %d\n", i, iter); 
        
        //chillAST_VarDecl *vd = refs[i]->multibase(); // this returns a reference to i for c.i n
        chillAST_node *node = refs[i]->multibase2(); // this returns a reference to c.i 
        //debug_fprintf(stderr, "\nnode is a %s\n", node->getTypeString()); 
        
        string stringvar;
        if (node->isVarDecl()) { 
          chillAST_VarDecl *vd = (chillAST_VarDecl *) node ; // this returns a reference to c.i 
          stringvar = vd->varname; 
        }
        else if (node->isMemberExpr()) { 
          chillAST_MemberExpr *me = (chillAST_MemberExpr *) node;
          stringvar = me->stringRep();
        }
        
        const char *lookingfor = strdup( stringvar.c_str() ); 
        printf("%s wo \n", lookingfor); fflush(stdout); 
        
        //debug_fprintf(stderr, "\nL1205, CPUparamSymtab now has %d entries\nCPUparamSymtab\n", CPUparamSymtab->size());
        //printSymbolTable(CPUparamSymtab);
        //debug_fprintf(stderr, "L 1207, CPUbodySymtab now has %d entries\nCPUbodySymtab\n", CPUbodySymtab->size());
        //printSymbolTable(CPUbodySymtab);
        
        //for (int i = 0; i < kernel_parameters.size(); i++) { 
        //  debug_fprintf(stderr, "kernel parameter: %s\n", kernel_parameters[i].c_str()); 
        //} 
        
        
        chillAST_VarDecl *vd = symbolTableFindVariableNamed( CPUbodySymtab,  lookingfor );
        chillAST_VarDecl *p  = symbolTableFindVariableNamed( CPUparamSymtab, lookingfor );
        
        if (vd != NULL) { 
          debug_fprintf(stderr, "varname %s IS in CPUbodySymtab\n", lookingfor); 
          printSymbolTable(CPUbodySymtab);
          printSymbolTable(CPUparamSymtab);
          

          debug_fprintf(stderr, "ONLY check kernel_parameters for '%s' if varname is in CPUbodySymtab\n", lookingfor); 
          
          std::set<std::string>::iterator it;
          it = kernel_parameters.find(stringvar); 
          if ( it == kernel_parameters.end())  { 
            debug_fprintf(stderr, "varname %s is NOT in kernel_parameters. skipping\n", lookingfor); 
            
            continue;
          }
          else { 
            
            debug_fprintf(stderr, "umwut found '%s', which IS a kernel parameter\n", (*it).c_str()); 
            
            int numkp = kernel_parameters.size(); // WARNING, also kernel_params exists. different name!
            debug_fprintf(stderr, "%d in SET kernel_parameters\n", numkp); 
            for (std::set<std::string>::iterator k = kernel_parameters.begin(); k != kernel_parameters.end(); k++) {
              debug_fprintf(stderr, "'%s'\n", (*k).c_str());               
            }
            
          }
        }
        else {
          debug_fprintf(stderr, "\nvarname %s is NOT in CPUbodySymtab\n", lookingfor); 
          debug_fprintf(stderr, "\nbody Symtab is :\n"); printSymbolTable(CPUbodySymtab);
          debug_fprintf(stderr, "\nparam Symtab is :\n"); printSymbolTable(CPUparamSymtab);
          debug_fprintf(stderr, "\nCPUSide func is:\n"); CPUsidefunc->print(0,stderr); 
          debug_fprintf(stderr, "\nCPUfuncbody is:\n"); CPUfuncbody->print(0,stderr); 

          debug_fprintf(stderr, "\nDAMMIT\n\n"); 

        }
        
        
        debug_fprintf(stderr, "\nvar %d   %s\nwill be a GPU kernel parameter??\n", i, lookingfor);
        if (!vd) vd = p;

        if (!vd) { 
          debug_fprintf(stderr, "... but I can't find the decl to copy???\nloop_cuda_chill.cc  L1250\n");
          int *i=0;  int j = i[0]; 
          exit(-1); 
        }
        debug_fprintf(stderr, "need to copy/mimic "); vd->print(0, stderr); debug_fprintf(stderr, "\n");
        chillAST_VarDecl *newparam = (chillAST_VarDecl *) vd->clone() ; // copy 

        vd->print(0,stderr);

        debug_fprintf(stderr,"after cloning\n"); 
          printSymbolTable(CPUbodySymtab);


        //debug_fprintf(stderr, "newparam numdimensions %d\n", newparam->numdimensions );
        // TODO need to remove topmost size? 
        if (newparam->arraysizes) { 
          newparam->arraysizes[0] = 0;  // first size now unknown 
          newparam->knownArraySizes = false; 
          debug_fprintf(stderr, "[]");
          for (int i=1; i<newparam->numdimensions; i++) { 
            debug_fprintf(stderr, "[%d]", newparam->arraysizes[i]); 
          }
          debug_fprintf(stderr, "\n"); 
        }
        newparam->print(0,stderr); debug_fprintf(stderr, "\n"); 

        printSymbolTable(CPUbodySymtab);
        printSymbolTable(CPUparamSymtab);
        
        debug_fprintf(stderr, "GPUKernel addparameter( %s )\n", newparam->varname); 
        GPUKernel->addParameter( newparam );  // TODO this parameter LOOKS like but it not the vardecl that the declrefexpr points to 

        printSymbolTable(CPUbodySymtab);
        printSymbolTable(CPUparamSymtab);

        // so ... add it now ??
        

        //vd->print(); printf("\n"); fflush(stdout); 
        //vd->dump(); printf("\n"); fflush(stdout); 
        //if (vd->isParmVarDecl()) { 
        //  debug_fprintf(stderr, "IS a parameter?\n");
        //  vd->dump(); printf("\n\n"); fflush(stdout); 
        //} 
        //if (!vd->isParmVarDecl()) { 
        //  debug_fprintf(stderr, "local array - kernel stack variable\n"); 
        
        
        //debug_fprintf(stderr, "looking for %s in %d uniqueRefs\n",  stringvar.c_str(), uniqueRefs.size()); 
        int offset = charstarvectorindex(  stringvar.c_str(), uniqueRefs );
        if ( offset == -1 )  { // != -1uniqueRefs.find( stringvar.c_str() )  == uniqueRefs.end()) {
          // wasn't there 
          debug_fprintf(stderr, "adding variable %s to uniqueRefs\n",  stringvar.c_str()); 
          // if not, add it
          //uniqueRefs.insert(  stringvar.c_str() );
          uniqueRefs.push_back(  strdup( stringvar.c_str()) );  // leak 
          
          // remember, refs WAS  std::vector<chillAST_ArraySubscriptExpr*> refs;   WAS 
          //  std::vector<IR_chillArrayRef *> wo_refs;
          if (refs[i]-> imwrittento) { 
            debug_fprintf(stderr, "adding variable %s to unique Write Only Refs\n", stringvar.c_str() );
            wo_refs.push_back( new IR_chillArrayRef( ir, refs[i], refs[i]-> imwrittento /* true */ ) ); 
            
            // later logic in this routine removes reads if it's also a write ... 
            // so let's just not add it.  ??  
            //if (refs[i]->imreadfrom) { // warning ONLY valid if also written to! TODO bad logic
            //  ro_refs.push_back( new IR_chillArrayRef( ir, refs[i], false ) ); 
            //  debug_fprintf(stderr, "adding variable %s to unique Read Only Refs TOO\n", vd->varname);
            //} 
            
          }
          else { // JUST read from 
            debug_fprintf(stderr, "adding variable %s to unique Read Only Refs\n", stringvar.c_str() ); // this is c.i
            debug_fprintf(stderr, "actually, adding "); refs[i]->print(); printf("\n"); fflush(stdout); 
            ro_refs.push_back( new IR_chillArrayRef( ir, refs[i], stringvar.c_str(), false )); // this is i
          }
        } // this is a new  reference 
        else debug_fprintf(stderr, "%s was already there?\n", stringvar.c_str()); 
        
        // NOT a parameter 
      } // for each ref  i 
      
      
      
      printf("\n\nreading from array ");
      for (int i = 0; i < ro_refs.size(); i++) { 
        //printf("("); ro_refs[i]->chillASE->base->print(); printf(")"); 
        printf("'%s' ", ro_refs[i]->chillASE->multibase()->varname); 
      }
      printf("and writing to array ");
      for (int i = 0; i < wo_refs.size(); i++)
        printf("'%s' ", wo_refs[i]->chillASE->multibase()->varname); 
      printf("\n");
      fflush(stdout); 
      
      
      debug_fprintf(stderr, "NOW WE MAKE THE GPU SIDE CODE\n\n"); 
      // the original function has been removed, so the following fails 
      // int which = p->findChild( origfunction ); 
      // debug_fprintf(stderr, "func is child %d of srcfile\n", which);

      // GPUKernel was created with parent p, so it is already there
      //p->insertChild(originalfunctionlocation,  GPUKernel );
      debug_fprintf(stderr, "\n\nkernel named %s\n", GPUKernel->functionName); 
      
      
      
      
      
      // COMMENT NEEDED 
      debug_fprintf(stderr, "loop_cuda_chill.cc COMMENT NEEDED printing kernel parameters\n"); 
      for (std::set<std::string>::iterator i = kernel_parameters.begin();   
           i != kernel_parameters.end(); i++) {
        debug_fprintf(stderr, "kernel parameter %s\n", (*i).c_str()); 
      }

      // PRINT RO REFS 
      for (int j = 0; j < ro_refs.size(); j++)
        debug_fprintf(stderr, "ro_ref %d %s\n", j, ro_refs[j]->name().c_str());  // ro_refs is  std::vector<IR_chillArrayRef *>
      debug_fprintf(stderr, "\n\n"); 
      
      
      // COMMENT NEEDED 
      debug_fprintf(stderr, "COMMENT NEEDED FOR EACH KERNEL PARAMETER\n"); 
      for (std::set<std::string>::iterator i = kernel_parameters.begin();
           i != kernel_parameters.end(); i++) {
        
        std::string kp(*i); // TODO name below is exactly this
        debug_fprintf(stderr, "walking through kernel_parameters %s\n", kp.c_str()); 
        
        int j;
        for (j = 0; j < ro_refs.size(); j++)
          if (ro_refs[j]->name() == *i)  { 
            
            break;
          }
        if (j < ro_refs.size())
          continue;
        
        std::string name = *i;
        debug_fprintf(stderr, "perhaps adding parameter %s to GPU kernel parameters??? \n", name.c_str()); 
        
        // HERE  stmt_code code_temp
        debug_fprintf(stderr, "\n***find all scalar refs in code_temp\n"); 
        std::vector<chillAST_DeclRefExpr*> scalar_refs;
        code_temp->gatherScalarRefs( scalar_refs, 0 );
        
        debug_fprintf(stderr, "SCALAR REFS (not printed)\n"); 
        //for (int k = 0; k < scalar_refs.size(); k++) { 
        //  debug_fprintf(stderr, "scalar ref %d ", k);
        //  scalar_refs[k]->print(); printf("\n"); fflush(stdout);
        //} 

        bool found = false;
        chillAST_node *ref = NULL; 

        debug_fprintf(stderr, "looking for %s in scalar refs\n", name.c_str()); 
        chillAST_DeclRefExpr *dre = NULL;
        for (int k = 0; k < scalar_refs.size() && !found; k++) { 
          if ( name == scalar_refs[k]->declarationName ) { 
            ref = scalar_refs[k];
            
            found = true;
            break;
          }
        }      

        if (!found)  { 
          debug_fprintf(stderr, "we did NOT find the parameter %s in the scalar refs.  look for it in macros?\n", name.c_str()); 
          
          //file we're working on holds the macro definitions 
          int numMacros = srcfile->macrodefinitions.size();
          debug_fprintf(stderr, "there are %d macros\n", numMacros); 

          for (int i=0; i<numMacros && !found; i++) { 
            chillAST_MacroDefinition *macro =  srcfile->macrodefinitions[i];
            debug_fprintf(stderr, "macro %d, name '%s'   ", i, macro->macroName);
            macro->print();  printf("\n"); fflush(stdout); 

            char *blurb = macro->getRhsString();
            if (blurb == NULL) { debug_fprintf(stderr, "macro rhs NULL\n"); }
            else 
            {
              //debug_fprintf(stderr, "macro rhs "); macro->getBody()->print(); debug_fprintf(stderr, "\n");
              //debug_fprintf(stderr, "%p\n", blurb); 
              //debug_fprintf(stderr, "%s\n", blurb); 
              // TODO this will not work in most cases. 
              // It is comparing the RESULT of the macro with the parameter name
              // that will only work if the result does not depend on any macro parameter. 
              if (!strcmp(blurb, name.c_str())) { 
                found = true;
                debug_fprintf(stderr, "macro RHS matches????\n"); 
                debug_fprintf(stderr, "rhs is of type %s\n", macro->getBody()->getTypeString()); 
                // get decl ref expression?  (why?)
                ref = macro->getBody(); // rhs  TODO function name 
              }
            }
            
          } 
          debug_fprintf(stderr, "\n\n"); 

        }
        
        if (!found) { debug_fprintf(stderr, "var_sym == NULL\n"); }
        else { debug_fprintf(stderr, "var_sym NOT == NULL\n"); }
      

        // UMWUT 
        if (found) {
          debug_fprintf(stderr, "checking name '%s' to see if it contains a dot  (TODO) \n", name.c_str()); 
          

        debug_fprintf(stderr, "make sure a symbol with this name is in the symbol table\n"); 


        debug_fprintf(stderr, "creating parameter that is address of value we want to pass to the GPU\n");
        debug_fprintf(stderr, "eg c_count = &c.count;\n");
        exit(-1); 
        } // end of UMWUT 
        else { 
          debug_fprintf(stderr, "var_sym was NULL, no clue what this is doing  %s\n", name.c_str()); 



        } // end of no clue 


      } // for each kernel parameter ???
      
      
      // unclear where this came from. older version, I assume 
      debug_fprintf(stderr, "OK, first the %d OUTPUTS of the GPU code\n",  wo_refs.size() ); 
      for (int i = 0; i < wo_refs.size(); i++) {
        std::string name = wo_refs[i]->name(); // creates a symbol. ...
        debug_fprintf(stderr, "write only ref (output?) %s\n", name.c_str()); 
      }
      debug_fprintf(stderr, "\n");
      debug_fprintf(stderr, "original function  parameters are:\n");
      printSymbolTable( &(origfunction->parameters )); 
      
      
      
      for (int i = 0; i < wo_refs.size(); i++) {
        std::string name = wo_refs[i]->name();
        debug_fprintf(stderr, "output %s\n", name.c_str()); 
        
        char *tmpname = strdup( name.c_str() ); 
        // find the variable declaration in original
        // this seems to have no analog in Anand's code.  ????"
        chillAST_VarDecl *param = origfunction->findParameterNamed( tmpname ); 
        if (!param) { 
          debug_fprintf(stderr, "cudaize_codegen_v2.cc can't find wo parameter named %s in function %s\n",tmpname,fname);
          
          debug_fprintf(stderr, "the parameters are:\n");
          printSymbolTable( &(origfunction->parameters )); 

          origfunction->print(0,stderr); debug_fprintf(stderr, "\n\n"); 
          exit(-1); 
        }
        //param->print(); printf("\n"); fflush(stdout); 
        
        VarDefs v; // scoping seems wrong/odd
        v.size_multi_dim = std::vector<int>();
        char buf[32];
        snprintf(buf, 32, "devO%dPtr", i + 1);
        v.name = buf;
        v.original_name = name; 
        debug_fprintf(stderr, "we'll have an output variable called %s\n", buf);
        
        v.tex_mapped  = false;
        v.cons_mapped = false;
        
        // find the underlying type of the array
        //debug_fprintf(stderr, "finding underlying type of %s to make variable %s match\n",name.c_str(),buf);
        v.type = strdup(param->underlyingtype); // memory leak 
        //debug_fprintf(stderr, "v.type is %s\n", param->underlyingtype); 
        
        chillAST_node *so = new chillAST_Sizeof( v.type ); 
        //CG_chillRepr *thingsize = new omega::CG_chillRepr(  so );
        
        debug_fprintf(stderr, "\nloop_cuda_xxxx.cc  calculating size of output %s\n", buf ); 
        
        int numitems = 1;
        if (param->numdimensions < 1 || 
            param->arraysizes == NULL) {
          debug_fprintf(stderr, "looking in array_dims numdimensions = %d\n", param->numdimensions);

          //Lookup in array_dims (the cudaize call has this info for some variables?) 
          std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
          debug_fprintf(stderr, "it %s %d\n", (*it).first.c_str(), (*it).second);  
          numitems = (*it).second; 
        }
        else { 
          debug_fprintf(stderr, "numdimensions = %d\n", param->numdimensions);
          for (int i=0; i<param->numdimensions; i++) { 
            numitems *= param->arraysizes[i]; 
          }
          debug_fprintf(stderr, "Detected Multi-dimensional array size of %d for %s \n", numitems, tmpname); 
        } 
        
        
        chillAST_IntegerLiteral *numthings = new chillAST_IntegerLiteral( numitems ); 
        
        debug_fprintf(stderr, "creating int mult size expression numitems %d x sizeof( %s )\n", numitems, v.type ); 
        
        debug_fprintf(stderr, "OK, finally figured out the size expression for the output array ...\n"); 
        
        // create a mult  
        v.size_expr = new chillAST_BinaryOperator( numthings, "*", so, NULL); 
        
        v.CPUside_param = param;
        v.in_data = 0;
        v.out_data = param;
        
        //Check for this variable in ro_refs and remove it at this point if it is both read and write
        // why did we bother before ??? 
        std::vector<IR_chillArrayRef *>::iterator it_;
        for (it_ = ro_refs.begin(); it_ != ro_refs.end(); it_++) {
          if ((*it_)->chillASE->multibase()->varname == wo_refs[i]->name()) {
            debug_fprintf(stderr, "found array ref for %s in ro, removing it from writes\n", (*it_)->name().c_str()); 
            break;
          }
        }
        if (it_ != ro_refs.end()) {
          v.in_data = param;           // ?? 
          ro_refs.erase(it_);
        }
        
        debug_fprintf(stderr, "adding written v to arrayVars\n\n"); 
        v.print(); 
        arrayVars.push_back(v);
      } //  wo_refs 
      
      
      // then READ ONLY refs 
      debug_fprintf(stderr, "\n*** OK, then  the %d INPUTS of the GPU code\n",  ro_refs.size() ); 
      for (int i = 0; i < ro_refs.size(); i++) {
        debug_fprintf(stderr, "read only parameter %d %s \n", i,  ro_refs[i]->name().c_str()); 
      }
      debug_fprintf(stderr, "\n");
      
      for (std::set<std::string>::iterator i = kernel_parameters.begin();
           i != kernel_parameters.end(); i++) {
        std::string kp(*i);
        debug_fprintf(stderr, "kernel_parameter %s\n", kp.c_str()); 
      }
      debug_fprintf(stderr, "\n");
      
      
      
      for (int i = 0; i < ro_refs.size(); i++) {
        std::string name = ro_refs[i]->name();
        char *tmpname = strdup( name.c_str() ); 
        debug_fprintf(stderr, "\nread only parameter %d %s \n", i, name.c_str()); 
        
        //chillAST_VarDecl *param = origfunction->findParameterNamed( tmpname ); 
        //if (!param) { 
        //  debug_fprintf(stderr, "loop_cuda_chill.cc can't find ro parameter named %s in function %s\n",tmpname,fname);
        //  debug_fprintf(stderr, "the parameters are:\n");
        //  printSymbolTable( &(origfunction->parameters )); 
        //  exit(-1);
        //} 
        
        VarDefs v; // scoping seems wrong/odd
        v.size_multi_dim = std::vector<int>();
        char buf[32];
        snprintf(buf, 32, "devI%dPtr", i + 1);
        v.name = buf;
        v.original_name = name; 
        v.tex_mapped = false;
        v.cons_mapped = false;
        
        bool isNOTAParameter = (kernel_parameters.find(name) == kernel_parameters.end());
        if (isNOTAParameter) { debug_fprintf(stderr, "%s is NOT a kernel parameter\n", name.c_str() ); }
        else { debug_fprintf(stderr, "%s IS a parameter\n", name.c_str()); } 
        
        
        // find the underlying type of the array
        debug_fprintf(stderr, "finding underlying type of %s to make variable %s match\n",name.c_str(),buf);
        // find the variable declaration
        chillAST_ArraySubscriptExpr* ASE = ro_refs[i]->chillASE;
        chillAST_VarDecl *base = ASE->basedecl; 
        v.type = strdup(base->underlyingtype); // memory leak 
        //debug_fprintf(stderr, "v.type is %s\n", base->underlyingtype); 
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
#endif        
        
        //debug_fprintf(stderr, "\ncalculating size of input %s\n", buf );    
        //Size of the array = dim1 * dim2 * num bytes of our array type
        //If our input array is 2D (non-linearized), we want the actual
        //dimensions of the array (as it might be less than cu_n
        //CG_outputRepr* size;
        
        int numitems = 1;
        base->print(0, stderr); debug_fprintf(stderr, "\n");
        
        
        if (base->numdimensions < 1 || 
            base->arraysizes == NULL) { 
          //Lookup in array_dims (the cudaize call has this info for some variables?) 
          std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
          if (it == array_dims.end()) { 
            debug_fprintf(stderr, "Can't find %s in array_dims\n", name.c_str()); 
            numitems = 123456;
          }
          else { 
            debug_fprintf(stderr, "it %s %d\n", (*it).first.c_str(), (*it).second);  
            numitems = (*it).second; 
          }
          //debug_fprintf(stderr, "it %s %d\n", (*it).first.c_str(), (*it).second);  
          debug_fprintf(stderr, "LUA command says this variable %s should be size %d\n",  (*it).first.c_str(), (*it).second); 
          numitems = (*it).second; 
          
        }
        else { 
          debug_fprintf(stderr, "numdimensions = %d\n", base->numdimensions);
          for (int i=0; i<base->numdimensions; i++) { 
            numitems *= base->arraysizes[i]; 
          }
        } 
        
        
        
        
        chillAST_IntegerLiteral *numthings = new chillAST_IntegerLiteral( numitems ); 
        
        //debug_fprintf(stderr, "creating int mult size expression numitems %d x sizeof( %s )\n", numitems, v.type ); 
        debug_fprintf(stderr, "OK, finally figured out the size expression for the input array ...\n"); 
        
        
        // create a mult  
        v.size_expr = new chillAST_BinaryOperator( numthings, "*", so, NULL); // 1024 * sizeof(float)  etc
        
        v.CPUside_param = base;
        v.in_data = base;
        v.out_data = 0;
        
        
        debug_fprintf(stderr, "adding input v to arrayVars\n\n"); 
        v.print(); 
        arrayVars.push_back(v);   
      } // end of READ refs
      debug_fprintf(stderr, "done with READ ONLY\n\n"); 
      
      
      
      debug_fprintf(stderr, "done with ORDERED STATEMENTS\n"); 
      debug_fprintf(stderr, "WHAT IS THIS TEST\n"); 
      
      debug_fprintf(stderr, "body_symtab now has %d entries\n\nbody_symtab\n", CPUbodySymtab->size());
      printSymbolTable(CPUbodySymtab);
      
      debug_fprintf(stderr, "adding mallocs and array memcpys\n"); 
      debug_fprintf(stderr, "%d arrayVars\n", arrayVars.size()); 
      
      for (int i = 0; i < arrayVars.size(); i++) {
        chillAST_VarDecl *var = arrayVars[i].vardecl;
        char *aname = strdup(arrayVars[i].name.c_str() ); 
        debug_fprintf(stderr, "\narrayVar %d   %s\n", i, aname); 
        arrayVars[i].print();
        
        // there is no way these created arrays (pointers) could be in the CPU function, but check anyway?
        //chillAST_VarDecl *vd = symbolTableFindVariableNamed( CPUBOD->, aname );
        var = CPUsidefunc->findVariableNamed( aname );
                  
        if ( !var )  { // didn't have an actual vardecl. probably always the case
          // make one 
          //debug_fprintf(stderr, "buildVariableDeclaration %s\n", arrayVars[i].name.c_str()); 
          // create a CHILL variable declaration and put it in the CPU side function
          
          char typ[128];
          sprintf(typ, "%s *", arrayVars[i].type); 
          
          var = new chillAST_VarDecl( typ,
                                      arrayVars[i].name.c_str(),
                                      "", // TODO
                                      NULL);

        debug_fprintf(stderr, "adding decl for %s to CPUsidefunc %s\n", var->varname, CPUsidefunc->functionName); 
          CPUsidefunc->insertChild(0, var ); // adds the decl to body code
          CPUsidefunc->addDecl( var ); // also adds to and CHANGES symbol table 
        }
      
        
        // store variable decl where we can get it easily later
        debug_fprintf(stderr, "body_symtab ADDING %s L2952\n", var->varname); 
        arrayVars[i].vardecl = var;
        
        //debug_fprintf(stderr, "body_symtab had %d entries\n", CPUbodySymtab->size()); 
        //debug_fprintf(stderr, "func        had %d entries\n", CPUsidefunc->getSymbolTable()->size()); 
        

        //debug_fprintf(stderr, "func        had %d entries after addDecl()\n", CPUsidefunc->getSymbolTable()->size()); 
        
        
        CPUbodySymtab = CPUsidefunc->getSymbolTable(); // needed ot bodysym is not up to date TODO
        
        //debug_fprintf(stderr, "body_symtab now has %d entries\n", CPUbodySymtab->size()); 
        debug_fprintf(stderr, "body_symtab now has %d entries\n", CPUsidefunc->getSymbolTable()->size()); 
        
        
        
        // do the CPU side cudaMalloc 
        debug_fprintf(stderr, "building CUDAmalloc using %s\n", aname); 
        arrayVars[i].size_expr->print(0,stderr); debug_fprintf(stderr, "\n"); 

        // wait, malloc?
        chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( var, CPUfuncbody ); 
        chillAST_CStyleAddressOf *AO = new chillAST_CStyleAddressOf( DRE );
        chillAST_CStyleCastExpr *casttovoidptrptr = new chillAST_CStyleCastExpr( "void **", AO, NULL ); 
        chillAST_CudaMalloc *cmalloc = new chillAST_CudaMalloc( casttovoidptrptr, arrayVars[i].size_expr, NULL); 
        
        debug_fprintf(stderr, "adding cudamalloc to 'setup code' for the loop\n"); 
        CPUfuncbody->addChild( cmalloc ); // TODO setup_code ?? 
        setup_code = ocg->StmtListAppend(setup_code,
                                         new CG_chillRepr(cmalloc));
        
        //debug_fprintf(stderr, "\ncudamalloc is:\n");
        //cmalloc->print(); printf("\n"); fflush(stdout); 
        
        if (arrayVars[i].in_data) {  
          //debug_fprintf(stderr, "\nnow the memcpy (for input variables only????)\n"); 
          // if it's input to the calculation, and we need to copy the data to the GPU
          debug_fprintf(stderr, "it's an input to the calculation, so we need to copy the data to the GPU\n"); 
          
          // do the CPU side cudaMemcpy, CPU to GPU("device")
          //DRE = new chillAST_DeclRefExpr( var, CPUfuncbody ); 
          chillAST_CudaMemcpy *cmemcpy = new chillAST_CudaMemcpy( var, 
                                                                  (chillAST_VarDecl*)(arrayVars[i].in_data), 
                                                                  arrayVars[i].size_expr, "cudaMemcpyHostToDevice"); 
          
          debug_fprintf(stderr, "cudamemcpy is:\n");
          cmemcpy->print(0, stderr); debug_fprintf(stderr, "\n");
          
          debug_fprintf(stderr, "adding cudamemcpy to 'setup code' for the loop\n"); 
          CPUfuncbody->addChild( cmemcpy ); // TODO setup_code ?? 
          setup_code = ocg->StmtListAppend(setup_code,
                                           new CG_chillRepr(cmemcpy));
          //printf("\n"); cmemcpy->print(); printf("\n");fflush(stdout); 
        }
        
      }  // for each arrayVar
      
      //debug_fprintf(stderr, "perhaps passing scalars to the kernel function\n"); 
      // seemingly not ?
      
      debug_fprintf(stderr, "\nBuild dimGrid dim3 variables based on loop dimensions and ti/tj\n"); 
      //Build dimGrid dim3 variables based on loop dimensions and ti/tj
      debug_fprintf(stderr, "dim3 variables will be dimGrid%d and dimBlock%d ??\n", iter, iter); 
      
      
      debug_fprintf(stderr, "create ARGS for dim3 %s\n", gridName);     
      
      int what = ordered_cudaized_stmts[iter].second;
      
      if (VbxAst.size() == 0) { // fake a constant 1
        
      }

      if ( what >=  VbxAst.size() || what >=  VbyAst.size()) { 
        debug_fprintf(stderr, "what %d\n", what); 
        debug_fprintf(stderr, "Vbx size %d   Vby size %d\n", VbxAst.size(), VbyAst.size()); 
        
        debug_fprintf(stderr, "time to die\n"); 
        exit(-1); 
      }
      debug_fprintf(stderr, "creating dim3 decl of %s( ", gridName );
      VbxAst[what]->print(0,stderr); debug_fprintf(stderr, ", "); 
      
      VbyAst[what]->print(0,stderr); debug_fprintf(stderr, " )\n"); 
      
      chillAST_CallExpr *CE1 = new chillAST_CallExpr( dimbuiltin, NULL );
      CE1->addArg(VbxAst[what]);
      CE1->addArg(VbyAst[what]);
      chillAST_VarDecl *dimgriddecl = new chillAST_VarDecl( "dim3", gridName, "", NULL );
      dimgriddecl->setInit(CE1);
      
      CPUfuncbody->addChild( dimgriddecl );  // TODO remove ?  setup_code ?? 
      debug_fprintf(stderr, "adding dim3 dimGrid to setup code for the statement\n\n");
      setup_code = ocg->StmtListAppend(setup_code,  new CG_chillRepr( dimgriddecl ) ); 
      
      
      
      debug_fprintf(stderr, "\nOK, now %s\n", blockName); 
      chillAST_VarDecl *dimblockdecl = NULL; 
      if (VtzAst.size() > what &&  // there is one
          VtzAst[what]) {          // it is not null 
        
        // there is a 3rd tz to be used
        debug_fprintf(stderr, "tz exists\n"); 
        chillAST_CallExpr *CE2 = new chillAST_CallExpr( dimbuiltin, NULL );
        CE2->addArg(VtxAst[what]);
        CE2->addArg(VtyAst[what]);
        CE2->addArg(VtzAst[what]);
        dimblockdecl = new chillAST_VarDecl( "dim3", blockName, "", NULL );
        dimblockdecl->setInit(CE2);
        
      }
      else { // no tz 
        debug_fprintf(stderr, "no tz\n"); 
        chillAST_CallExpr *CE2 = new chillAST_CallExpr( dimbuiltin, NULL );
        CE2->addArg(VtxAst[what]);
        CE2->addArg(VtyAst[what]);
        dimblockdecl = new chillAST_VarDecl( "dim3", blockName, "", NULL );
        dimblockdecl->setInit(CE2);
        dimblockdecl->print(0,stderr); debug_fprintf(stderr, "\n"); 
      }
      
      // Anand code has test for NULL dimblockdecl ... 
      CPUfuncbody->addChild( dimblockdecl );  // TODO remove ?  setup_code ?? 
      debug_fprintf(stderr, "adding dim3 %s to setup code for the statement\n\n", blockName);
      setup_code = ocg->StmtListAppend(setup_code,  new CG_chillRepr( dimblockdecl ) ); 
      
      
      debug_fprintf(stderr, "\nconfig?  ( the kernel call?? )\n"); 
      
      //debug_fprintf(stderr, "\nkernel named\n");GPUKernel->print(0,stderr); debug_fprintf(stderr, "\n"); 
      
      chillAST_CallExpr *kcall = new chillAST_CallExpr( GPUKernel,  CPUfuncbody);
      kcall->grid = dimgriddecl; 
      kcall->block =  dimblockdecl; 
      debug_fprintf(stderr, "kernel function parameters\n"); 
      for (int i = 0; i < arrayVars.size(); i++) { 
        //Throw in a type cast if our kernel takes 2D array notation
        //like (float(*) [1024])
        
        if (arrayVars[i].tex_mapped || arrayVars[i].cons_mapped) { 
          if (arrayVars[i].tex_mapped) { debug_fprintf(stderr, "arrayVars[i].tex_mapped\n"); }
          if (arrayVars[i].cons_mapped) { debug_fprintf(stderr, "arrayVars[i].cons_mapped\n"); } 
          continue;
        }
        
        chillAST_VarDecl *v = arrayVars[i].vardecl;
        chillAST_VarDecl *param = arrayVars[i].CPUside_param;
        
        debug_fprintf(stderr, "param i %d,  numdimensions %d\n", i, param->numdimensions); 
        
        if (param->numdimensions > 1) { 
          debug_fprintf(stderr, "array Var %d %s is multidimensional\n",i, v->varname);
          debug_fprintf(stderr, "underlying type %s   arraypart '%s'\n", v->underlyingtype, v->arraypart);
          //v->print(0, stderr); debug_fprintf(stderr, "\n"); 
          param->print(0, stderr); debug_fprintf(stderr, "\n\n"); 

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
      }  // for each arrayVar
 

     
      debug_fprintf(stderr, "kernel call is "); kcall->print(0,stderr); debug_fprintf(stderr, "\n"); 
      CPUfuncbody->addChild( kcall );           
      
      debug_fprintf(stderr, "\nfreeing %d Cuda variables\n", arrayVars.size()); 
      //cuda free variables
      for (int i = 0; i < arrayVars.size(); i++) {
        debug_fprintf(stderr, "arrayVar %d\n", i); 
        
        // Memcopy back if we have an output 
        if (arrayVars[i].out_data) {
          debug_fprintf(stderr, "Memcopy back if we have an output\n"); 
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
      debug_fprintf(stderr, "\nDONE freeing %d Cuda variables\n", arrayVars.size()); 
      
      
      
      debug_fprintf(stderr, "loop_cuda_chill.cc BUILD THE KERNEL\n"); 
      chillAST_node *kernelbody = code_temp;  // wrong 
      GPUKernel->setBody( kernelbody ); 
      
      
      //Extract out the CPU loop.  (part of) this will become the GPU side code
      chillAST_node *CPUsideloop = getCode(  );  // CG_chillRepr getCode
      debug_fprintf(stderr, "loop_cuda_chill.cc L1669 returned from getCode()\n");
      
      debug_fprintf(stderr, "\n\n\nloop_cuda_chill.cc L1685   CPUsideloop = \n");
      CPUsideloop->print(); 
      debug_fprintf(stderr, "\n\n"); 
      
      debug_fprintf(stderr, "\nGPUKernel:\n"); 
      GPUKernel->print(0, stderr); debug_fprintf(stderr, "\n\n"); 
      
      debug_fprintf(stderr, "\ncode_temp:\n"); 
      code_temp->print(0, stderr); debug_fprintf(stderr, "\n\n"); 
      
      
      // At this point, we have the CPU-side code, in CPUsideloop
      // we have code_temp, which is what will become the body of the GPU kernel
      
      
      
      // we need to figure out which variables need to be added to the kernel
      
// first, remove / replace loop variables 
      chillAST_SymbolTable *st =  GPUKernel->getSymbolTable();
      if (!st) st = new chillAST_SymbolTable;
      GPUKernel->findLoopIndexesToReplace( st, false );

      debug_fprintf(stderr, "\nfind variables used in the kernel (?)\n"); 
      // find all variables used in the function
      vector<chillAST_VarDecl*> decls;
      GPUKernel->gatherVarUsage( decls );  // finds decls even if the actual decl is not in the ast but the reference to it is
      
      // attempt to get xxdiff to sync
      debug_fprintf(stderr, "\n                                                      ****** recursiveFindRefs()\n");
      
      debug_fprintf(stderr, "%d vars in syms USED in code_temp that need to be added to func_d \n\n", (int)decls.size()); 
      for (int i=0; i<decls.size(); i++) { 
        debug_fprintf(stderr, "%d   %s \n", i, decls[i]->varname); 
        decls[i]->print(0,stderr); debug_fprintf(stderr, "\n"); 
      }
      debug_fprintf(stderr, "\n\n"); 
      
      //int nump = GPUKernel->parameters.size();
      //debug_fprintf(stderr, "\n%d parameters to GPUKernel\n", nump); 
      //for (int i=0; i<nump; i++) debug_fprintf(stderr, "parameter %s\n",  GPUKernel->parameters[i]->varname );
      //debug_fprintf(stderr, "\n"); 
      
      
      
      //Figure out which loop variables will be our thread and block dimension variables
      debug_fprintf(stderr, "\nFigure out which loop variables will be our thread and block dimension variables\n"); 
      
      int beforecode = 0; // offset for var decls
      
      //Get our indexes  (threadIdx and blockIdx will replace some loops) 
      std::vector<const char*> indexes;
      
      //here, as we figure out that bx, by tx, etc will be loop variables, 
      //  we need to note that the thing they replace will NOT be needed. 
        
      if (VbxAst[what])  { // Vcu_bx[what] > 1 || cu_bx_repr[what]) {
        debug_fprintf(stderr, "adding bx to indexes\n"); 
        indexes.push_back("bx");
        
        // add definition of bx, and blockIdx.x to it in GPUKernel
        // int bx;
        chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.x", "int", GPUKernel );
        chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 
        chillAST_VarDecl *bxdecl;
        // see if bx is already defined in the Kernel
        bxdecl = GPUKernel->funcHasVariableNamed( "bx" );
        if (!bxdecl) { 
          debug_fprintf(stderr, "bx was NOT defined in GPUKernel before\n"); 
          GPUKernel->print(0,stderr); debug_fprintf(stderr, "\n\n"); 
          bxdecl= new chillAST_VarDecl( "int", "bx", "",GPUKernel);
          GPUKernel->addDecl( bxdecl ); // to symbol table
          // if it was there, we shouldn't do this? 
          GPUKernel->insertChild(beforecode++, bxdecl); 
        }
        else debug_fprintf(stderr, "bx WAS defined in GPUKernel before\n"); 
        bxdecl->setInit( bid );  // add init

        //GPUKernel->addVariableToSymbolTable( bxdecl ); 


        // separate assign statement (if not using the vardecl init )
        //chillAST_DeclRefExpr *bx = new chillAST_DeclRefExpr( bxdecl ); 
        //chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( bx, "=",bid); 
        //assign->print(0,stderr); debug_fprintf(stderr, "\n"); 
        //GPUKernel->addChild(assign); 

        // remove the 
      }
      
      
      if (VbyAst.size() > 0 && VbyAst[what]) { 
        debug_fprintf(stderr, "adding by to indexes\n");  // TODO wrong test 
        indexes.push_back("by");
        
        // add definition of by, and blockIdx.y to it in GPUKernel
        chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.y", "int", GPUKernel);
        chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 

        chillAST_VarDecl *bydecl;
        // see if by is already defined in the Kernel
        bydecl = GPUKernel->funcHasVariableNamed( "by" ); 
        if (!bydecl) { 
          debug_fprintf(stderr, "by was NOT defined in GPUKernel before\n"); 
          bydecl= new chillAST_VarDecl( "int", "by", "", GPUKernel);
          GPUKernel->addDecl( bydecl ); // to symbol table
          GPUKernel->insertChild(beforecode++, bydecl);
        }
        else debug_fprintf(stderr, "by WAS defined in GPUKernel before\n"); 
        bydecl->setInit( bid ); // add init
        
        // separate assign statement (if not using the vardecl init )
        //chillAST_DeclRefExpr *by = new chillAST_DeclRefExpr( bydecl ); 
        //chillAST_BinaryOperator *assign = new chillAST_BinaryOperator(by,"=",bid); 
        //assign->print(0,stderr); debug_fprintf(stderr, "\n"); 
        //GPUKernel->addChild(assign); 
      }
      
      if (VtxAst.size() > 0 && VtxAst[what]) { 
        debug_fprintf(stderr, "adding tx to indexes\n"); 
        indexes.push_back("tx");
        chillAST_VarDecl *tiddecl = addBuiltin( "threadIdx.x", "int", GPUKernel);
        chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( tiddecl ); 

        chillAST_VarDecl *txdecl;
        // see if tx is already defined in the Kernel
        txdecl = GPUKernel->funcHasVariableNamed( "tx" ); 
        if (!txdecl) { 
          GPUKernel->print(0,stderr); 

          debug_fprintf(stderr, "tx was NOT defined in GPUKernel before\n"); 
          txdecl= new chillAST_VarDecl( "int", "tx", "", GPUKernel);
          GPUKernel->addDecl( txdecl ); // to symbol table
          GPUKernel->insertChild(beforecode++, txdecl);
        }
        else debug_fprintf(stderr, "tx WAS defined in GPUKernel before\n"); 
        txdecl->setInit( tid ); // add init

      }
      
      if (VtyAst.size() > 0 && VtyAst[what]) { 
        debug_fprintf(stderr, "adding ty to indexes\n"); 
        indexes.push_back("ty");
        chillAST_VarDecl *tiddecl = addBuiltin( "threadIdx.y", "int", GPUKernel);
        chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( tiddecl ); 

        chillAST_VarDecl *tydecl;
        // see if ty is already defined in the Kernel
        tydecl = GPUKernel->funcHasVariableNamed( "ty" ); 
        if (!tydecl) { 
          debug_fprintf(stderr, "ty was NOT defined in GPUKernel before\n"); 
          GPUKernel->print(0, stderr); 

          tydecl= new chillAST_VarDecl( "int", "ty", "", GPUKernel);
          GPUKernel->addDecl( tydecl ); // to symbol table
          GPUKernel->insertChild(beforecode++, tydecl);
        }
        else debug_fprintf(stderr, "ty WAS defined in GPUKernel before\n"); 
        tydecl->setInit( tid ); // add init
      }
      
      if (VtzAst.size() > 0 && VtzAst[what]) { 
        debug_fprintf(stderr, "adding tz to indexes\n"); 
        indexes.push_back("tz");
        chillAST_VarDecl *tiddecl = addBuiltin( "threadIdx.z", "int", GPUKernel);
        chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( tiddecl ); 

        chillAST_VarDecl *tzdecl;
        // see if tz is already defined in the Kernel
        tzdecl = GPUKernel->funcHasVariableNamed( "tz" ); 
        if (!tzdecl) { 
          debug_fprintf(stderr, "tz was NOT defined in GPUKernel before\n"); 
          tzdecl= new chillAST_VarDecl( "int", "tz", "", GPUKernel);
          GPUKernel->addDecl( tzdecl ); // to symbol table
          GPUKernel->insertChild(beforecode++, tzdecl);
        }
        else debug_fprintf(stderr, "tz WAS defined in GPUKernel before\n"); 
        tzdecl->setInit( tid ); // add init

      }
      
      
      
      debug_fprintf(stderr, "\n%d indexes\n", (int) indexes.size()); 
      for (int i = 0; i < indexes.size(); i++) {
        debug_fprintf(stderr, "indexes[%i] = '%s'\n", i, indexes[i] ); 
      }
      debug_fprintf(stderr, "\n"); 
      
      
      std::vector<chillAST_ArraySubscriptExpr*> kernelArrayRefs;
      code_temp->gatherArrayRefs( kernelArrayRefs, false );
      //debug_fprintf(stderr, "%d array refs in kernel\n",  kernelArrayRefs.size()); 
      

      // Look for arrayrefs used in kernel, to see if they are parameters to the GPU kernel (?)
      // and if it IS, check to see if it's a member of a struct accessed by a dot, and if it IS,
      // change it to a non-member expression  ??? 
      //
      debug_fprintf(stderr, "\nloop_cuda_chill.cc L2072  looking for %d arrayRefs\n", (int) kernelArrayRefs.size()); 
      for (int i = 0; i < kernelArrayRefs.size(); i++) { 
        //chillAST_node *vd = kernelArrayRefs[i]->multibase2(); 
        
        debug_fprintf(stderr, "ref %d = '", i);
        kernelArrayRefs[i]->basedecl->print(0,stderr); debug_fprintf(stderr, "'\n");
        //kernelArrayRefs[i]->print(0,stderr); debug_fprintf(stderr, "'\n");
        //kernelArrayRefs[i]->base->print(0,stderr); debug_fprintf(stderr, "'\n");
      }
      debug_fprintf(stderr, "in %d kernel_parameters\n", (int)kernel_parameters.size()); 
      for (std::set<std::string>::iterator i = kernel_parameters.begin();
           i != kernel_parameters.end(); i++) {
        debug_fprintf(stderr, "kernel parameter '%s'\n", (*i).c_str()); 
      }
      debug_fprintf(stderr, "\n"); 
      
      // 
      
      // TODO ... stuff comment above
      
      
      //debug_fprintf(stderr, "\nbefore swapVarReferences(), code is\n{\n"); 
      //kernelbody->print();
      //debug_fprintf(stderr, "}\n\nswapVarReferences()\n"); 
      //swapVarReferences( CPUsideloop, GPUKernel );
      //debug_fprintf(stderr, "\nafter swapVarReferences(), ");
      
      debug_fprintf(stderr, "kernel code is\n"); 
      GPUKernel->print();
      debug_fprintf(stderr, "\n\n");
      
      debug_fprintf(stderr, "now replace indexes ... (and add syncs)\n");
      findReplacePreferedIdxs( CPUsideloop, GPUKernel );
      debug_fprintf(stderr, "DONE WITH replace indexes ... (and add syncs)\n"); 
      
      debug_fprintf(stderr, "\nswapped 2\nshould have syncs\nshould have indexes replaced by bx, tx, etc \n"); 
      GPUKernel->print();
      
      
      
      
      //CPUsidefunc->print(0, stderr); debug_fprintf(stderr, "\n\n\n"); 
      
      GPUKernel->print(0, stderr); debug_fprintf(stderr, "\n\n\n");
      
      // now remove loops that will be done by spreading the loop count across cores
      // these are loops that have our indeces gathered above as loop variables
      debug_fprintf(stderr, "removing loops for variables that will be determined by core index\n"); 
      for (int i = 0; i < indexes.size(); i++) {
        debug_fprintf(stderr, "\nindexes[%i] = '%s'\n", i, indexes[i] ); 
        debug_fprintf(stderr, "forReduce()\n");
        
        GPUKernel->getBody()->loseLoopWithLoopVar( strdup(indexes[i]) ); 
      }
      
      
      GPUKernel->print(0, stderr); debug_fprintf(stderr, "\n\n\n");
    
    } // if THIS statement will become a kernel
    else 
    {
      debug_fprintf(stderr, "statement %d will NOT run on GPU\n", iter); 
    }


  } // for int iter  (each statement that COULD  become a kernel )
  
  //debug_fprintf(stderr, "loop_cuda_chill.cc L 1546, exiting\n"); 
  

  
  
  
  
  
  
  
  char *part = strdup( srcfile->SourceFileName );
  char *dot = rindex( part, '.' );
  if  (dot) { 
    *dot = '\0';
  }
  debug_fprintf(stderr, "part '%s'\n", part);
  
  
  
  
  
  
  debug_fprintf(stderr, "%d statements\n", stmt.size()); 
  for (int j = 0; j < stmt.size(); j++) {
    debug_fprintf(stderr, "\nstmt j %d\n", j); 
    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[j].code);
    debug_fprintf(stderr, "%d array references in stmt j %d\n", refs.size(), j); 
    
    debug_fprintf(stderr, "\nabout to dump statement j %d\n", j); 
    CG_chillRepr * repr = (CG_chillRepr *) stmt[j].code;
    repr->dump(); 
    fflush(stdout); debug_fprintf(stderr, "\n\n\n\n");
    
  } // for each stmt 
  
  
#ifdef VERYWORDY
  debug_fprintf(stderr, "we read from %d parameter arrays, and write to %d parameter arrays\n", ro_refs.size(), wo_refs.size()); 
  printf("reading from array parameters ");
  for (int i = 0; i < ro_refs.size(); i++)
    printf("'%s' ", ro_refs[i]->name().c_str());
  printf("and writing to array parameters ");
  for (int i = 0; i < wo_refs.size(); i++)
    printf("'%s' ", wo_refs[i]->name().c_str());
  printf("\n"); fflush(stdout); 
#endif  
  
  
  
  
  
  
  debug_fprintf(stderr, "POWERTHRU  bailing\n"); 
  //exit(0); 
  
#ifdef POWERTHRU  
  // and now READ ONLY   
  
  
  
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
  
  
  debug_fprintf(stderr, "\nBuild dimGrid and dimBlock dim3 variables based on loop dimensions and ti/tj\n"); 
  //Build dimGrid dim3 variables based on loop dimensions and ti/tj
  
  
  chillAST_CallExpr *CE1 = new chillAST_CallExpr( dimbuiltin, NULL );
  
  // create ARGS to dim3. 
  debug_fprintf(stderr, "create ARGS for dim3 dimGrid\n"); 
  debug_fprintf(stderr, "ordered_cudaized_stmts[iter].second %d\n", ordered_cudaized_stmts[iter].second);
  
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
      debug_fprintf(stderr, "array Var %d %s is multidimensional  SECOND\n",i, v->varname);
      debug_fprintf(stderr, "underlying type %s\narraypart %s\n", v->underlyingtype, v->arraypart); 
      char line[128];
      sprintf(line, "%s (*)", v->underlyingtype ); 
      debug_fprintf(stderr, "line '%s'\n", line);
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
        fprintf("CastExpr is "); CE->print(0,stderr); debug_fprintf(stderr, "\n\n"); 
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
  chillAST_node *CPUsideloop = getCode(  );  // chill based getCode
  
  debug_fprintf(stderr, "loop_cuda_chill.cc L1669 returned from getCode()\n");
  
  //debug_fprintf(stderr, "loop_cuda_chill.cc L1685  CPUsideloop =\n");
  //GPUKernel->getBody()->print(); fflush(stdout);
  //debug_fprintf(stderr, "\n\n"); 
  
  debug_fprintf(stderr, "loop_cuda_chill.cc L1685   CPUsideloop = \n");
  CPUsideloop->print(); 
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
  CPUsideloop->gatherVarDecls( decls );
  debug_fprintf(stderr, "%d variables in kernel\n", decls.size()); 
  for (int i=0; i<decls.size(); i++) { 
    debug_fprintf(stderr, "%s\n", decls[i]->varname); 
  }
  
  int nump = GPUKernel->parameters.size();
  debug_fprintf(stderr, "\n%d parameters to GPUKernel\n", nump); 
  for (int i=0; i<nump; i++) { debug_fprintf(stderr, "parameter %s\n",  GPUKernel->parameters[i]->varname ); }
  debug_fprintf(stderr, "\n"); 
  
  
  
  //Figure out which loop variables will be our thread and block dimension variables
  debug_fprintf(stderr, "Figure out which loop variables will be our thread and block dimension variables\n"); 
  
  //Get our indexes  (threadIdx and blockIdx will replace some loops) 
  std::vector<const char*> indexes;
  
  if (cu_bx > 1 || cu_bx_repr) { // this was for hardcoded stmt 0
    indexes.push_back("bx");
    chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.x", "int", GPUKernel );
    chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *bxdecl = new chillAST_VarDecl( "int", "bx", "",GPUKernel);
    GPUKernel->addDecl( bxdecl ); // to symbol table
    GPUKernel->addChild(bxdecl); 
    
    chillAST_DeclRefExpr *bx = new chillAST_DeclRefExpr( bxdecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( bx, "=",bid); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 
    
    GPUKernel->addChild(assign); 
  }
  
  if (cu_by > 1 || cu_by_repr) {
    indexes.push_back("by");
    chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.y", "int", GPUKernel);
    chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *bydecl = new chillAST_VarDecl( "int", "by", "", GPUKernel);
    GPUKernel->addDecl( bydecl );
    chillAST_DeclRefExpr *by = new chillAST_DeclRefExpr( bydecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( by, "=",bid); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 
    
    GPUKernel->addChild(bydecl); 
    GPUKernel->addChild(assign); 
  }  
  if (cu_tx_repr || cu_tx > 1) {
    //threadsPos = indexes.size();
    indexes.push_back("tx");
    chillAST_VarDecl *tiddecl = addBuiltin( "threadIdx.x", "int",     GPUKernel);
    chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( tiddecl ); 
    chillAST_VarDecl *txdecl = new chillAST_VarDecl( "int", "tx", "", GPUKernel);
    GPUKernel->addDecl( txdecl );
    chillAST_DeclRefExpr *tx = new chillAST_DeclRefExpr( txdecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( tx, "=",tid);
    assign->print(0, stderr); debug_fprintf(stderr, "\n"); 
    
    GPUKernel->addChild(txdecl); 
    GPUKernel->addChild(assign); 
  }
  if (cu_ty_repr || cu_ty > 1) {
    indexes.push_back("ty");
    chillAST_VarDecl *biddecl = addBuiltin( "threadIdx.y", "int", GPUKernel );
    chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *tydecl = new chillAST_VarDecl( "int", "ty", "", GPUKernel);
    debug_fprintf(stderr, "loop_cuda_chill.cc L 2598, adding ty to kernelbody??   ty was \n"); 

    GPUKernel->addDecl( tydecl );
    chillAST_DeclRefExpr *ty = new chillAST_DeclRefExpr( tydecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( ty, "=",tid); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 
    
    kernelbody->addChild(tydecl); 
    kernelbody->addChild(assign); 
  }
  if (cu_tz_repr || cu_tz > 1) {
    indexes.push_back("tz");
    chillAST_VarDecl *biddecl = addBuiltin( "threadIdx.z", "int", GPUKernel );
    chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *tzdecl = new chillAST_VarDecl( "int", "tz", "",GPUKernel);
    GPUKernel->addDecl( tzdecl );
    chillAST_DeclRefExpr *tz = new chillAST_DeclRefExpr( tzdecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( tz, "=",tid);
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 
    
    kernelbody->addChild(tzdecl); 
    kernelbody->addChild(assign); 
  }
  
  // multistatement version is up to here 
  
  debug_fprintf(stderr, "\n"); 
  for (int i = 0; i < indexes.size(); i++) {
    debug_fprintf(stderr, "indexes[%i] = '%s'\n", i, indexes[i] ); 
  }
  
  debug_fprintf(stderr, "\nbefore swapVarReferences(), code is\n{\n"); 
  kernelbody->print();
  
  debug_fprintf(stderr, "}\n\nswapVarReferences()\n"); 
  //swapVarReferences( CPUsideloop, GPUKernel );
  
  debug_fprintf(stderr, "\nafter swapVarReferences(), code is\n"); 
  kernelbody->print();
  debug_fprintf(stderr, "\n\n");
  
  
  debug_fprintf(stderr, "now replace indexes ... (and add syncs)\n"); 
  findReplacePreferedIdxs( CPUsideloop, GPUKernel );
  debug_fprintf(stderr, "DONE WITH replace indexes ... (and add syncs)\n"); 
  
  debug_fprintf(stderr, "\nswapped 2\nshould have syncs\nshould have indexes replaced by bx, tx, etc \n\n"); 
  CPUsideloop->print();
  
  // now remove loops that will be done by spreading the loop count across cores
  // these are loops that have our indeces gathered above as loop variables
  debug_fprintf(stderr, "removing loops for variables that will be determined by core index\n"); 
  chillAST_CompoundStmt *CS = new chillAST_CompoundStmt();
  
  CS->addChild( CPUsideloop ); // in case top level loop will go away
  //debug_fprintf(stderr, "arbitrary compoundstmt 0x%x to hold child CPUsideloop  0x%x\n", CS, CPUsideloop); 
  for (int i = 0; i < indexes.size(); i++) {
    debug_fprintf(stderr, "\nindexes[%i] = '%s'\n", i, indexes[i] ); 
    debug_fprintf(stderr, "forReduce()\n");
    
    CPUsideloop->loseLoopWithLoopVar( strdup(indexes[i]) ); 
  }
  
  
  
  
  
  debug_fprintf(stderr, "END cudaize codegen V2 (CHILL)\n\n\n");
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
    if (isdeclared) { debug_fprintf(stderr, " (builtin)"); }
    else { 
      if (vd->isParmVarDecl()) isdeclared = true;
      if (isdeclared) { debug_fprintf(stderr, " (param)"); }
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
  
  
  ////////////////////////////////////////////////////////////////////////
  return CS; 
  
#endif  // POWERTHRU 
  
  debug_fprintf(stderr, "returning from cudaize_codegen_v2()\n"); 
  return NULL; 
}


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
    for (int i=0; i<pi.size(); i++) { debug_fprintf(stderr, "pi[%d] = %d\n", i, pi[i]); }
    
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

#if 0
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
#endif

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
  for (int i = 0; i < namez.size(); i++) { debug_fprintf(stderr, "%d %s\n", i, namez[i].c_str()); }
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
                          std::map<std::string, int> array_dims,
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
  chillAST_VarDecl *bi = new chillAST_VarDecl( typeOfBuiltin, nameofbuiltin, "", NULL); // somecode ); // ->getSourceFile() );
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
    
    if (isParam)    { debug_fprintf(stderr, "is a parameter\n"); }
    if (isLocalVar) { debug_fprintf(stderr, "is already defined in the kernel\n"); }
    
    if (!isParam && (!isLocalVar)) { 
      debug_fprintf(stderr, "needed to be added to kernel symbol table\n");
      kernel->addDecl(  newdecls[i] );  // adds to symbol table
      kernel->addChild( newdecls[i] );  // adds to body! 
    }
  }
}


