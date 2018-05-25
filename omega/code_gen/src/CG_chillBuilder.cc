
/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   generate chill AST code for omega

 Notes:
 
 History:
   02/01/06 created by Chun Chen

*****************************************************************************/

#include <iostream>
#include <stack>
#include <cstring>
#include <code_gen/CG_chillBuilder.h>
#include "scanner.h"

namespace omega {
  
  // substitute at chill AST level
  // forward declarations
  class SubstituteOldVar : public chill::Scanner<const char *, chillAST_node*, CG_chillRepr *, chillAST_node *&> {
  public:

      virtual ~SubstituteOldVar() = default;

  protected:

    virtual void errorRun(chillAST_node *n, const char *oldvar, chillAST_node* parent, CG_chillRepr *newvar, chillAST_node *&newnode) {
      chillAST_node* replaced;
      // This is generic
      for (int i = 0; i < n->getNumChildren(); ++i) {
        replaced = NULL;
        run(n->getChild(i), oldvar, n, newvar, replaced);
        if (replaced)
//          n->replaceChild(n->getChild(i), replaced);
// Mahdi: Change to correct embedded iteration space: from Tuowen's topdown branch
          n->setChild(i, replaced);
      }
    }

    virtual void runS(chillAST_DeclRefExpr *n, const char *oldvar, chillAST_node* parent, CG_chillRepr *newvar, chillAST_node *&newnode) {
      if (streq( oldvar,  n->declarationName)) {
        chillAST_node *firstn = newvar->chillnodes[0];
// Mahdi: Commented to correct embedded iteration space: from Tuowen's topdown branch
//        firstn->parent = parent;
        newnode = firstn;
      }
    }
  };

  chillAST_node *substituteChill( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    if (n == NULL)
      throw std::runtime_error("substituteChill called on null pointer");
    chillAST_node *r = n;
    SubstituteOldVar so;
    so.run(n, oldvar, parent, newvar, r);
    return r;
  }

  CG_chillBuilder::CG_chillBuilder() { 
    toplevel = NULL;
    currentfunction = NULL; // not very useful
    symtab_ = symtab2_ = NULL; 
  }
  
  CG_chillBuilder::CG_chillBuilder(chillAST_SourceFile *top, chillAST_FunctionDecl *func) { 
    //debug_fprintf(stderr, "\nCG_chillBuilder::CG_chillBuilder()\n"); 
    toplevel = top;
    currentfunction = func;
    
    //debug_fprintf(stderr, "\nfunction is:\n"); currentfunction->print(); printf("\n\n"); fflush(stdout); 

    symtab_  = &(currentfunction->parameters);
    symtab2_ = currentfunction->getSymbolTable();
    symtab2_->clear();
    currentfunction->getBody()->gatherVarDecls(*symtab2_);
    
    //printf("\nsymtab_:\n"); fflush(stdout); 
    //printSymbolTable( symtab_ ); 
    //printf("\n\nsymtab2_:\n"); fflush(stdout); 
    //printSymbolTable( symtab2_ ); 
  }
  
  CG_chillBuilder::~CG_chillBuilder() {
    
  }
  
  //-----------------------------------------------------------------------------
  // place holder generation  NOT IN CG_outputBuilder
  //-----------------------------------------------------------------------------
  //
  // FIXME: Function isn't working fully yet
  //
  CG_outputRepr* CG_chillBuilder::CreatePlaceHolder (int indent, 
                                                     CG_outputRepr *stmt,
                                                     Tuple<CG_outputRepr*> &funcList, 
                                                     Tuple<std::string> &loop_vars) const {
    debug_fprintf(stderr, "CG_chillBuilder::CreatePlaceHolder()  TODO \n");
    exit(-1);                   // DFL 
    return NULL; 
    /* 
       if(Expr *expr = static_cast<CG_chillRepr *>(stmt)->GetExpression()) {
       for(int i=1; i<= funcList.size(); ++i) {
       if (funcList[i] == NULL)
       continue;
       
       CG_chillRepr *repr = static_cast<CG_chillRepr*>(funcList[i]);
       Expr* op = repr->GetExpression();
       delete repr;
       
       Expr *exp = expr;
       
       if(isa<BinaryOperator>(exp)) {
       //substitute(static_cast<BinaryOperator*>(exp)->getLHS(), loop_vars[i], op, exp);
       //substitute(static_cast<BinaryOperator*>(exp)->getRHS(), loop_vars[i], op, exp);
       }
       else if(isa<UnaryOperator>(exp))
       //substitute(static_cast<UnaryOperator*>(exp)->getSubExpr(), loop_vars[i], op, exp);
       
       }
       return new CG_chillRepr(expr);
       } else {
       StmtList *tnl = static_cast<CG_chillRepr *>(stmt)->GetCode();
       for(int i=1; i<= funcList.size(); ++i) {
       if (funcList[i] == NULL)
       continue;
       
       CG_chillRepr *repr = static_cast<CG_chillRepr*>(funcList[i]);
       Expr* op = repr->GetExpression();
       delete repr;
       
       for(unsigned j=0; j<tnl->size(); ++j) {
       Expr *exp = static_cast<Expr *>((*tnl)[j]);
       if(isa<BinaryOperator>(exp)) {
       //substitute(static_cast<BinaryOperator*>(exp)->getLHS(), loop_vars[i], op, exp);
       //substitute(static_cast<BinaryOperator*>(exp)->getRHS(), loop_vars[i], op, exp);
       }
       else if(isa<UnaryOperator>(exp))
       //substitute(static_cast<UnaryOperator*>(exp)->getSubExpr(), loop_vars[i], op, exp);
       
       }
       }
       return new CG_chillRepr(*tnl);
       }
    */ 
    
    
  }
  
  
  
  
  //----------------------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateSubstitutedStmt(int indent, 
                                                        CG_outputRepr *stmt,
                                                        const std::vector<std::string> &vars, 
                                                        std::vector<CG_outputRepr*> &subs,
                                                        bool actuallyPrint) const {
    
    int numvars = vars.size(); 
    int numsubs = subs.size(); 
    fflush(stdout); 
    debug_fprintf(stderr, "\n\nin CG_xxxxBuilder.cc (OMEGA)  CG_xxxxBuilder::CreateSubstitutedStmt()\n");
    debug_fprintf(stderr, "%d vars and %d substitutions\n", numvars, (int)subs.size());
    
    
    if (numvars != numsubs) {
      //debug_fprintf(stderr, "umwut?\n"); exit(-1); 
    }
    
    
    //{
    //  vector<chillAST_node*> nodes = ((CG_chillRepr *) stmt)->getChillCode(); 
    //  // 
    //  debug_fprintf(stderr, "%d nodes in old code. was:\n", nodes.size()); 
    //  for(int i=0; i<nodes.size(); i++) 
    //    { 
    //      printf("stmt(%d) = ",i); 
    //      nodes[i]->print(); 
    //      printf("\n"); 
    //    }
    //  //printf("\nreally subbing in %d top level statements\n", nodes.size()); 
    //  fflush(stdout);
    //}     
    
    //for (int i=0; i< numsubs; i++)        {
    //  debug_fprintf(stderr, "sub %d  ", i); 
    //  if (subs[i]) {   ((CG_chillRepr *)subs[i])->Dump(); fflush( stdout );  }
    //  else  { 
    //    //int *crash = NULL;  *crash = 1; 
    //    debug_fprintf(stderr, "(NULL  error!)"); 
    //  }
    //  //debug_fprintf(stderr, "\n"); 
    //} 
    //debug_fprintf(stderr, "\n"); 
    
    
    if (numsubs == 0) {
      
      vector<chillAST_node*> nodes = ((CG_chillRepr *) stmt)->getChillCode(); 
      
      // 
      //debug_fprintf(stderr, "nosubs old code was:\n"); 
      //for(int i=0; i<nodes.size(); i++) 
      //  { 
      //    printf("stmt = "); 
      //    nodes[i]->print(); 
      //    printf("\n"); 
      //  }
      //printf("\nreally subbing in %d top level statements\n", nodes.size()); 
      //fflush(stdout);
      
      
      
      // no cloning !!
      return new CG_chillRepr( nodes );
      
      //debug_fprintf(stderr, "since nothing actually being substituted, this is just a clone\n"); 
      //debug_fprintf(stderr, "old code was AND new code is:\n");
      //for (int i=0; i<nodes.size(); i++) { 
      //  debug_fprintf(stderr, "stmt = ");
      //  nodes[i]->print();  fflush(stdout); 
      //  debug_fprintf(stderr, "\n"); 
      //} 
      //debug_fprintf(stderr, "cloning()\n"); 
      //return stmt->clone(); 
      
    }



    //debug_fprintf(stderr, "numsubs %d\n", numsubs);
    
    // debugging: print the substitutions we'll do 
    
    //if (numsubs > 0) { 
    //  for (int i=0; i< numsubs; i++)        {
    //    debug_fprintf(stderr, "subbing "); 
    //    if (subs[i]) {   
    //      ((CG_chillRepr *)subs[i])->Dump(); fflush( stdout ); 
    //      debug_fprintf(stderr, "for  %s\n", vars[i].c_str() );
    //}        else  { 
    //      //int *crash = NULL;  *crash = 1; 
    //      debug_fprintf(stderr, "(NULL  error!)"); 
    //    }
    //    //debug_fprintf(stderr, "\n"); 
    //  }
    //  debug_fprintf(stderr, "\n"); 
    //} 
    
    
    
    //debug_fprintf(stderr, "OK, now to really substitute ...\n");  
    //CG_outputRepr *newstmt = stmt->clone();
    //CG_chillRepr *n = (CG_chillRepr *) newstmt; 
    //vector<chillAST_node*> newnodes =  n->getChillCode();  
    
    CG_chillRepr *old = (CG_chillRepr *) stmt; 
    vector<chillAST_node*> oldnodes = old->getChillCode();
    
    
    for (int j=0; j<numsubs; j++) { 
      if (subs[j] != NULL) {
        
        //debug_fprintf(stderr, "substitution %d    %s -> ", j,vars[j].c_str()); 
        //if (subs[j]) {  ((CG_chillRepr *)subs[j])->Dump(); fflush( stdout );  }
        
        
        // find the type of thing we'll be using to replace the old variable
        CG_chillRepr *CRSub = (CG_chillRepr *)(subs[j]); 
        vector<chillAST_node*> nodes = CRSub->chillnodes;
        if (1 != nodes.size() )  { // always just one? 
          debug_fprintf(stderr, "CG_chillBuilder::CreateSubstitutedStmt(), replacement is not one statement??\n");
          exit(-1);
        }
        chillAST_node *node = nodes[0]; // always just one? 
        
        for (int i=0; i<oldnodes.size(); i++) {
          oldnodes[i] = substituteChill( vars[j].c_str(), CRSub, oldnodes[i]);
        }
      }
    }
    
    //debug_fprintf(stderr, "\ncode after substituting variables:\n");
    //for(int i=0; i<oldnodes.size(); ++i){ printf("stmt = ");oldnodes[i]->print();printf("\n");}
    //fflush(stdout); 
    
    return new CG_chillRepr( oldnodes );
  }
  
  
  
  //-----------------------------------------------------------------------------
  // assignment generation
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateAssignment(int indent, 
                                                   CG_outputRepr *lhs,
                                                   CG_outputRepr *rhs) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateAssignment()\n"); 
    if(lhs == NULL || rhs == NULL) {
      debug_fprintf(stderr, "Code generation: Missing lhs or rhs\n");
      return NULL;
    }
    
    CG_chillRepr *clhs = (CG_chillRepr *) lhs;
    CG_chillRepr *crhs = (CG_chillRepr *) rhs;
    chillAST_node *lAST = clhs->chillnodes[0]; // always just one?
    chillAST_node *rAST = crhs->chillnodes[0]; // always just one?
    
    chillAST_BinaryOperator *bop = new chillAST_BinaryOperator(lAST->clone(), "=", rAST->clone() ); // clone??
    
    delete lhs; delete rhs;
    return new CG_chillRepr(bop);
  }
  
  
  
  
  CG_outputRepr* CG_chillBuilder::CreatePlusAssignment(int indent,               // += 
                                                       CG_outputRepr *lhs,
                                                       CG_outputRepr *rhs) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreatePlusAssignment()\n"); 
    if(lhs == NULL || rhs == NULL) {
      debug_fprintf(stderr, "Code generation: Missing lhs or rhs\n");
      return NULL;
    }
    
    CG_chillRepr *clhs = (CG_chillRepr *) lhs;
    CG_chillRepr *crhs = (CG_chillRepr *) rhs;
    chillAST_node *lAST = clhs->chillnodes[0]; // always just one?
    chillAST_node *rAST = crhs->chillnodes[0]; // always just one?
    
    chillAST_BinaryOperator *bop = new chillAST_BinaryOperator(lAST->clone(), "+=", rAST->clone() ); // clone??
    
    delete lhs; delete rhs;
    return new CG_chillRepr(bop);
  }
  
  
  
  
  //-----------------------------------------------------------------------------
  // function invocation generation
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateInvoke(const std::string &fname,
                                               std::vector<CG_outputRepr*> &list,
                                               bool is_array) const { // WHAT is an array?
    debug_fprintf(stderr, "CG_roseBuilder::CreateInvoke( fname %s, ...)\n", fname.c_str()); 
    //debug_fprintf(stderr, "%d things in list\n", list.size()); 
    
    // debugging output.  print the "call"
    //debug_fprintf(stderr, "%s", fname.c_str());
    //if (is_array) debug_fprintf(stderr, "["); else debug_fprintf(stderr, "("); 
    //int numparams = list.size(); 
    //for (int i=0; i<numparams; i++) { 
    //  CG_chillRepr *CR = (CG_chillRepr *) list[i];
    //  if (i) printf(","); 
    //  printf(" "); 
    //  CR->GetCode()->print(); 
    //  fflush(stdout); 
    //} 
    //if (numparams) printf(" "); 
    //if (is_array) printf("]\n"); else printf(")\n"); 
    //fflush(stdout); 
    


    if (is_array) { 
      //debug_fprintf(stderr, "CG_chillBuilder::CreateInvoke() %s is_array\n", fname.c_str());
      const char *arrayname = fname.c_str(); 
      
      CG_chillRepr *CR = (CG_chillRepr *) list[0];
      chillAST_node *cast = CR->GetCode();

      //debug_fprintf(stderr, "%s[",  arrayname);
      //cast->print(); printf("] ???\n"); fflush(stdout);
      
      // find the array variable  (scope ??  TODO) 
      chillAST_VarDecl *array = currentfunction->findArrayDecl( arrayname ); 
      if (!array) { 
        debug_fprintf(stderr, "CG_chillBuilder::CreateInvoke(), can't find array %s\n", fname.c_str()); 
      }
      
      // make a declrefexpr that refers to vardecl of array ? 
      chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( array );
      chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( DRE, cast, NULL, NULL); 
      return  new CG_chillRepr( ASE ); 
    }
    
    
    if (fname == std::string("max") || fname == std::string("min")) {
      if (list.size() == 0) { return NULL; }
      else if (list.size() == 1) { return list[1]; }
      else {
        const char *op;
        
        if (fname == std::string("max"))  op = ">";
        else op = "<";
        
        chillAST_node *ternary = minmaxTernary( op,  ((CG_chillRepr*) list[0])->chillnodes[0],
                                                 ((CG_chillRepr*) list[1])->chillnodes[0]);  
        CG_chillRepr *repr = new CG_chillRepr( ternary );
        return repr;
      }
    }
    //else { // special case for reduce? 
    //} 
    else {
      //do a simple function call 
      debug_fprintf(stderr, "building a function call expression\n"); 

      // try to find the function name, for a function in this file
      const char *name = fname.c_str(); 
      //debug_fprintf(stderr, "fname '%s'\n", name);
      chillAST_SourceFile *src = toplevel; // todo don't be dumb
      
      chillAST_node *def = src->findCall(name);
      if (def) {
        chillAST_CallExpr *CE = new chillAST_CallExpr( new chillAST_DeclRefExpr(def) );
        int numparams = list.size();
        for (int i=0; i<numparams; i++) {
          CG_chillRepr *CR = (CG_chillRepr *) list[i];
          CE->addArg( CR->GetCode() );
        }
        return  new CG_chillRepr( CE );
      }

      chillAST_CallExpr *CE = new chillAST_CallExpr( new chillAST_DeclRefExpr(name) );
      int numparams = list.size();
      for (int i=0; i<numparams; i++) {
        CG_chillRepr *CR = (CG_chillRepr *) list[i];
        CE->addArg( CR->GetCode() );
      }
      return  new CG_chillRepr( CE );
    }
  }
  
  
  //-----------------------------------------------------------------------------
  // comment generation - NOTE: Not handled
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateComment(int indent, const std::string &commentText) const {
    return NULL;
  }
  
  
  
  CG_outputRepr* CG_chillBuilder::CreateNullStatement() const {
    return new CG_chillRepr(  new chillAST_NoOp() );
  }
  
  
  
  //---------------------------------------------------------------------------
  // Attribute generation
  //---------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateAttribute(CG_outputRepr *control,
                                                  const std::string &commentText) const {
    
    //debug_fprintf(stderr, "in CG_chillBuilder.cc (OMEGA)   CG_chillBuilder::CreateAttribute()\n");
    //debug_fprintf(stderr, "comment = '%s'\n", commentText.c_str()); 
    
    CG_chillRepr *CR = (CG_chillRepr *) control;
    int numnodes = CR->chillnodes.size(); 
    //debug_fprintf(stderr, "%d chill nodes\n", numnodes); 
    if (numnodes > 0) { 
      //debug_fprintf(stderr, "adding a comment to a %s\n", CR->chillnodes[0]->getTypeString()); 
      CR->chillnodes[0]->metacomment = strdup( commentText.c_str()); 
    }
    else { 
      debug_fprintf(stderr, "CG_chillBuilder::CreateAttribute no chillnodes to attach comment to???\n");
    }
    return  static_cast<CG_chillRepr*>(control);
  };
  
  
  
  
  
  //-----------------------------------------------------------------------------
  // if stmt gen operations
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateIf(int indent, 
                                           CG_outputRepr *guardList,
                                           CG_outputRepr *true_stmtList, 
                                           CG_outputRepr *false_stmtList) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateIf()\n"); 
    
    if (true_stmtList == NULL && false_stmtList == NULL) {
      delete guardList;
      return NULL;
    }
    else if (guardList == NULL) {  // this seems odd 
      return StmtListAppend(true_stmtList, false_stmtList);
    }
    
    vector<chillAST_node*> vectorcode =  static_cast<CG_chillRepr*>(guardList)->getChillCode();
    if (vectorcode.size() != 1 ) {
      debug_fprintf(stderr, "CG_chillBuilder.cc IfStmt conditional is multiple statements?\n");
      exit(-1);
    }
    chillAST_node *conditional = vectorcode[0]; 
    chillAST_CompoundStmt *then_part = NULL;
    chillAST_CompoundStmt *else_part = NULL;
    
    
    if (true_stmtList != NULL) { 
      then_part = new chillAST_CompoundStmt(  );
      vectorcode =  static_cast<CG_chillRepr*>(true_stmtList)->getChillCode(); 
      for (int i=0; i<vectorcode.size(); i++) then_part->addChild( vectorcode[i] ); 
    }
    
    if (false_stmtList != NULL) {
      else_part = new chillAST_CompoundStmt(  );
      vectorcode =  static_cast<CG_chillRepr*>(false_stmtList)->getChillCode(); 
      for (int i=0; i<vectorcode.size(); i++) else_part->addChild( vectorcode[i] ); 
    }
    
    
    chillAST_IfStmt *if_stmt = new chillAST_IfStmt( conditional, then_part, else_part );
    
    delete guardList;  
    delete true_stmtList;
    delete false_stmtList;
    
    return new CG_chillRepr( if_stmt );
  }
  
  
  //-----------------------------------------------------------------------------
  // inductive variable generation, to be used in CreateLoop as control
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateInductive(CG_outputRepr *index,
                                                  CG_outputRepr *lower,
                                                  CG_outputRepr *upper,
                                                  CG_outputRepr *step) const {
    debug_fprintf(stderr, "\nCG_chillBuilder::CreateInductive()\n");
    if (index == NULL || lower == NULL || upper == NULL) {
      debug_fprintf(stderr, "Code generation: invalid arguments to CreateInductive\n");
      return NULL;
    }
    
    
    if (step == NULL) {
      //IntegerLiteral *ilit = new (astContext_)IntegerLiteral(*astContext_, llvm::APInt(32, 1), bint->desugar(), SourceLocation());
      //step = new CG_chillRepr(ilit);
      
      chillAST_IntegerLiteral *intlit = new chillAST_IntegerLiteral(1);
      step = new CG_chillRepr(intlit);
    }
    
    //static_cast<CG_chillRepr*>(index)->printChillNodes(); 
    //static_cast<CG_chillRepr*>(lower)->printChillNodes(); 
    //static_cast<CG_chillRepr*>(upper)->printChillNodes(); 
    //static_cast<CG_chillRepr*>(step )->printChillNodes(); 
    
    // index should be a DeclRefExpr
    vector<chillAST_node*> nodes = static_cast<CG_chillRepr*>(index)->getChillCode();
    //debug_fprintf(stderr, "%d index nodes\n", nodes.size());
    chillAST_node *indexnode = nodes[0];
    if (!streq("DeclRefExpr", indexnode->getTypeString())) {
      debug_fprintf(stderr, "CG_chillBuilder::CreateInductive index is not a DeclRefExpr\n"); 
      if (indexnode->isIntegerLiteral()) debug_fprintf(stderr, "isIntegerLiteral()\n"); 

      debug_fprintf(stderr, "index is %s\n", indexnode->getTypeString());
      indexnode->print(); printf("\n");   fflush(stdout);
      indexnode->dump();  printf("\n\n"); fflush(stdout);
      int *i = 0; int j = i[0];
      exit(-1); 
    }
    
    nodes = static_cast<CG_chillRepr*>(lower)->getChillCode();
    //debug_fprintf(stderr, "%d lower nodes\n", nodes.size());
    chillAST_node *lowernode = nodes[0];
    //debug_fprintf(stderr, "lower node is %s\n", lowernode->getTypeString()); 
    
    nodes = static_cast<CG_chillRepr*>(upper)->getChillCode();
    //debug_fprintf(stderr, "%d upper nodes\n", nodes.size());
    chillAST_node *uppernode = nodes[0];
    //debug_fprintf(stderr, "upper node is %s\n", uppernode->getTypeString()); 
    
    nodes = static_cast<CG_chillRepr*>(step)->getChillCode();
    //debug_fprintf(stderr, "%d step nodes\n", nodes.size());
    chillAST_node *stepnode = nodes[0];
    //debug_fprintf(stderr, "step  node is %s\n",  stepnode->getTypeString()); 
    
    // unclear is this will always be the same 
    // TODO error checking  && incr vs decr
    chillAST_BinaryOperator *init = new  chillAST_BinaryOperator( indexnode, "=", lowernode );
    chillAST_BinaryOperator *cond = new  chillAST_BinaryOperator( indexnode, "<=", uppernode );
    
    //chillAST_BinaryOperator *inc  = new  chillAST_BinaryOperator( indexnode, "+", stepnode, NULL); 
    chillAST_BinaryOperator *incr = new  chillAST_BinaryOperator( indexnode, "+=", stepnode );
    
    chillAST_ForStmt *loop = new chillAST_ForStmt( init, cond, incr, NULL /* NULL BODY DANGER! */);
    
    return new CG_chillRepr(loop); 
    
    /*    
    //vector<chillAST_node*> indexnodes = static_cast<CG_chillRepr*>(index)->getChillCode(); 
    chillAST_DeclRefExpr *index_decl
    Expr *lower_bound; //                 = static_cast<CG_chillRepr*>(lower)->getChillCode();
    Expr *upper_bound; //               = static_cast<CG_chillRepr*>(upper)->getChillCode();
    Expr *step_size  ; //                = static_cast<CG_chillRepr*>(step)->getChillCode();
    
    debug_fprintf(stderr, "gonna die in CG_chillBuilder ~line 459\n");
    
    chillAST_BinaryOperator *for_init_stmt =  NULL; // new (astContext_)BinaryOperator(index_decl, lower_bound, BO_Assign, index_decl->getType(), SourceLocation());
    chillAST_BinaryOperator *test = NULL; // new (astContext_)BinaryOperator(index_decl, upper_bound, BO_LT, index_decl->getType(), SourceLocation());
    chillAST_BinaryOperator *increment = NULL; // new (astContext_)BinaryOperator(index_decl, step_size, BO_AddAssign, index_decl->getType(), SourceLocation());
    
    // For Body is null.. Take care of unnecessary parens!
    ForStmt *for_stmt = NULL; // new (astContext_)ForStmt(*astContext_, for_init_stmt, test, static_cast<VarDecl*>(index_decl->getDecl()), increment, NULL, SourceLocation(), SourceLocation(), SourceLocation());
    
    delete index;    
    delete lower;         
    delete upper;
    delete step;
    
    StmtList sl;
    sl.push_back(for_stmt);
    return new CG_chillRepr(sl);
    */     
  }
  
  
  
  //-----------------------------------------------------------------------------
  // Pragma Attribute
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreatePragmaAttribute(CG_outputRepr *stmt, int looplevel, const std::string &pragmaText) const {
    debug_fprintf(stderr, "CG_chillBuilder::CreatePragmaAttribute()   TODO\n");
    //exit(-1);
    // TODO    effectively a comment? 
    /* 
       SgNode *tnl = static_cast<CG_chillRepr*>(stmt)->tnl_;
       CodeInsertionAttribute* attr = NULL;
       if (!tnl->attributeExists("code_insertion")) {
       attr = new CodeInsertionAttribute();
       tnl->setAttribute("code_insertion", attr);
       }
       else {
       attr = static_cast<CodeInsertionAttribute*>(tnl->getAttribute("code_insertion"));
       }
       attr->add(new PragmaInsertion(looplevel, pragmaText));
    */

    auto chill_stmt = dynamic_cast<CG_chillRepr*>(stmt);
    for(auto node: chill_stmt->chillnodes) {
      switch(node->getType()) {
      case CHILLAST_NODETYPE_LOOP:
          auto loop_node = node->as<chillAST_ForStmt>();
          // TODO: maybe call something like chillAST_ForStmt::setPragma() ?
          if(loop_node->pragma != NULL) {
              free(loop_node->pragma);
          }
          loop_node->pragma = (char*) malloc(pragmaText.size() + 1);
          strcpy(loop_node->pragma, pragmaText.c_str());
          break;
      }
    }

    return stmt;
  }
  
  //-----------------------------------------------------------------------------
  // Prefetch Attribute
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreatePrefetchAttribute(CG_outputRepr* stmt, int looplevel, const std::string &arrName, int hint) const {
    debug_fprintf(stderr, "CG_chillBuilder::CreatePrefetchAttribute()   TODO\n");
    exit(-1); 
    // TODO 
    /* 
       SgNode *tnl = static_cast<CG_chillRepr*>(stmt)->tnl_;
       CodeInsertionAttribute *attr = getOrCreateCodeInsertionAttribute(tnl);
       attr->add(new MMPrefetchInsertion(looplevel, arrName, hint));
    */
    return stmt;
  }
  
  
  
  
  
  
  
  //-----------------------------------------------------------------------------
  // loop stmt generation
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateLoop(int indent, 
                                             CG_outputRepr *control,
                                             CG_outputRepr *stmtList) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateLoop( indent %d)\n", indent); 
    
    if (stmtList == NULL) {
      delete control;
      return NULL;
    }
    else if (control == NULL) {
      debug_fprintf(stderr, "Code generation: no inductive for this loop\n");
      return stmtList;    
    }
    
    // We assume the for statement is already created (using CreateInductive)
    vector<chillAST_node*> code = static_cast<CG_chillRepr*>(control)->getChillCode();
    chillAST_ForStmt *forstmt =  (chillAST_ForStmt *)(code[0]);
    
    vector<chillAST_node*> statements = static_cast<CG_chillRepr*>(stmtList)->getChillCode(); 
    //static_cast<CG_chillRepr*>(stmtList)->printChillNodes(); printf("\n"); fflush(stdout);
    
    chillAST_CompoundStmt *cs = new chillAST_CompoundStmt();
    for (int i=0; i<statements.size(); i++) { 
      cs->addChild( statements[i] );
    }
    
    forstmt->setBody(cs);
    
    delete stmtList;
    return control;
  }
  
  //---------------------------------------------------------------------------
  // copy operation, NULL parameter allowed. this function makes pointer
  // handling uniform regardless NULL status
  //---------------------------------------------------------------------------
  /*
    virtual CG_outputRepr* CG_chillBuilder::CreateCopy(CG_outputRepr *original) const {
    if (original == NULL)
    return NULL;
    else
    return original->clone();
    }
  */
  
  //-----------------------------------------------------------------------------
  // basic int, identifier gen operations
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateInt(int i) const {
    debug_fprintf(stderr, "CG_chillBuilder::CreateInt( %d )\n",i); 
    chillAST_IntegerLiteral *il = new chillAST_IntegerLiteral(i); // parent not available
    return new CG_chillRepr(il);
  }
  CG_outputRepr* CG_chillBuilder::CreateFloat(float f) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateFloat( %f )\n", f); 
    chillAST_FloatingLiteral *fl = new chillAST_FloatingLiteral(f); // parent not available
    return new CG_chillRepr(fl);
  }
  CG_outputRepr* CG_chillBuilder::CreateDouble(double d) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateInt( %f )\n",d); 
    chillAST_FloatingLiteral *dl = new chillAST_FloatingLiteral(d); // parent not available
    return new CG_chillRepr(dl);
  }
  
  
  //----------------------------------------------------------------------------------------
  bool CG_chillBuilder::isInteger(CG_outputRepr *op) const{
    CG_chillRepr *cr = (CG_chillRepr *)op;
    return cr->chillnodes[0]->isIntegerLiteral(); 
  }
  
  
  //----------------------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateIdent(const std::string &_s) const {
    debug_fprintf(stderr, "CG_chillBuilder::CreateIdent( %s )\n", _s.c_str()); 
    
    auto already_parameter = symbolTableFindVariableNamed(symtab_,  _s.c_str());
    auto already_internal  = symbolTableFindVariableNamed(symtab2_, _s.c_str());

    if ( already_parameter )
      return new CG_chillRepr( new chillAST_DeclRefExpr(already_parameter) );
    if ( already_internal )
      return new CG_chillRepr( new chillAST_DeclRefExpr(already_internal) );

    debug_fprintf(stderr, "CG_roseBuilder.cc L919 adding symbol %s to symtab2_ because it was not already there\n", _s.c_str());
    // this is copying roseBuilder, but is probably wrong. it is assuming
    // that the ident is a direct child of the current function

    chillAST_VarDecl *vd = new chillAST_VarDecl( "int", "", _s.c_str()); // parent not available  TODO
    currentfunction->addVariableToSymbolTable( vd ); // use symtab2_  ??

    chillAST_DeclRefExpr *dre = new chillAST_DeclRefExpr( "int", _s.c_str(), (chillAST_node*)vd ); // parent not available
    return new CG_chillRepr( dre );
  }
  



  
  //-----------------------------------------------------------------------------
  // binary arithmetic operations
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreatePlus(CG_outputRepr *lop,
                                             CG_outputRepr *rop) const {
    debug_fprintf(stderr, "CG_chillBuilder::CreatePlus()\n"); 
    
    
    if(rop == NULL) return lop;     // ?? 
    else if(lop == NULL) return rop;
    
    chillAST_node *left  = ((CG_chillRepr*)lop)->chillnodes[0]; 
    chillAST_node *right = ((CG_chillRepr*)rop)->chillnodes[0]; 
    chillAST_BinaryOperator *bop = new chillAST_BinaryOperator( left, "+", right ); // parent not available
    return new CG_chillRepr( bop );
    /*
      Expr *lhs = static_cast<CG_chillRepr*>(lop)->GetExpression();
      Expr *rhs = static_cast<CG_chillRepr*>(rop)->GetExpression();
      
      // Not sure about type!!
      debug_fprintf(stderr, "about to die in CG_chillBuilder ~line 628    CREATE PLUS\n"); 
      BinaryOperator *ins = new (astContext_)BinaryOperator(lhs,
      rhs, 
      BO_Add, 
      lhs->getType(), // qualifyier type 
      VK_LValue, //Expression Value Kind, following the C++11 scheme
      OK_Ordinary, // expression object kind, A further classification of the kind of object referenced by an l-value or x-value. 
      SourceLocation(),
      false );  // fpContractable  ?? 
      
      delete lop; delete rop;
      
      //debug_fprintf(stderr, "                                                                               NEW binary operator 0x%x\n", ins);
      debug_fprintf(stderr, "CG_chillBuilder::CreatePlus  ins 0x%x\n", ins); 
      return new CG_chillRepr(ins);
    */
  }
  
  //-----------------------------------------------------------------------------  
  CG_outputRepr* CG_chillBuilder::CreateMinus(CG_outputRepr *lop,
                                              CG_outputRepr *rop) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateMinus( lop %p   rop %p)\n", lop, rop); 
    debug_fprintf(stderr, "CG_chillBuilder::CreateMinus()\n");
    
    if(rop == NULL) {
      debug_fprintf(stderr, "CG_chillBuilder::CreateMinus(), right side is NULL\n"); 
      return lop; // from protonu's version. 
    }
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    if(clop == NULL) {  // this is really a unary operator ??? 
      //debug_fprintf(stderr, "CG_chillBuilder::CreateMinus()  unary\n");
      chillAST_node *rAST = crop->chillnodes[0]; // always just one?
      chillAST_UnaryOperator *ins = new chillAST_UnaryOperator("-", true, rAST->clone()); // clone?
// Mahdi: Comment to correct embedded iteration space: from Tuowen's topdown branch
//      delete crop;  // ?? note: the chillRepr, not the chillAST_node 
      return new CG_chillRepr(ins);
    } else {
      //debug_fprintf(stderr, "binary\n");
      chillAST_node *lAST = clop->chillnodes[0]; // always just one?
      chillAST_node *rAST = crop->chillnodes[0]; // always just one?
      //lAST->print(); printf(" - ");
      //rAST->print(); printf("\n"); fflush(stdout); 
      
      chillAST_BinaryOperator *bop = new chillAST_BinaryOperator(lAST->clone(), "-", rAST->clone()); // clone??
      
// Mahdi: Comment to correct embedded iteration space: from Tuowen's topdown branch
//      delete clop; delete crop; // ?? note: the chillReprs, not the chillAST_nodes
      return new CG_chillRepr(bop);
    }
  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateTimes(CG_outputRepr *lop,
                                              CG_outputRepr *rop) const {
    debug_fprintf(stderr, "CG_chillBuilder::CreateTimes()\n"); 
    if (rop == NULL || lop == NULL) {
      if (rop != NULL) {
        rop->clear();
        delete rop;
      }
      if (lop != NULL) {
        lop->clear();
        delete lop;
      }                 
      return NULL;
    }             
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "*", rAST );
// Mahdi: Comment to correct embedded iteration space: from Tuowen's topdown branch
//    delete lop; delete rop; // ?? 
    //debug_fprintf(stderr, "CG_chillBuilder::CreateTimes() returning a CG_chillRepr with a binop inside\n");
    return new CG_chillRepr( binop );
  }
  
  
  
  //-----------------------------------------------------------------------------
  //  CG_outputRepr *CG_chillBuilder::CreateDivide(CG_outputRepr *lop, CG_outputRepr *rop) const {
  //    return CreateIntegerFloor(lop, rop);
  //  }
  
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateIntegerDivide(CG_outputRepr *lop,
                                                      CG_outputRepr *rop) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreatIntegerDivide()\n"); 
    if (rop == NULL) {
      debug_fprintf(stderr, "Code generation: divide by NULL\n");
      return NULL;
    }
    else if ( lop == NULL ) {
      delete rop;
      return NULL;
    }
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //debug_fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //debug_fprintf(stderr, " / ");
    //rAST->print(0, stderr);
    //debug_fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "/", rAST );
    delete lop; delete rop; // ?? 
    return new CG_chillRepr( binop );
  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateIntegerFloor(CG_outputRepr* lop, CG_outputRepr* rop) const { 
    //debug_fprintf(stderr, "CG_chillBuilder::CreateIntegerFloor()\n");
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //debug_fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //debug_fprintf(stderr, " / ");
    //rAST->print(0, stderr);
    //debug_fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "/", rAST );
    return new CG_chillRepr( binop );
  }
  
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateIntegerMod(CG_outputRepr *lop,
                                                   CG_outputRepr *rop) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateIntegerMod()   NEEDS WORK\n"); 
    //debug_fprintf(stderr, "LHS "); lop->dump(); 
    //debug_fprintf(stderr, "RHS "); rop->dump(); 
    
    CG_chillRepr *l = (CG_chillRepr *) lop; 
    CG_chillRepr *r = (CG_chillRepr *) rop; 
    
    chillAST_node *lhs = l->GetCode();
    chillAST_node *rhs = r->GetCode();
    
    chillAST_BinaryOperator *BO = new  chillAST_BinaryOperator(lhs, "%", rhs );
    return new CG_chillRepr(BO);
    
    /* 
       if (rop == NULL || lop == NULL) {
       return NULL;
       }
       
       Expr *op1 = static_cast<CG_chillRepr*>(lop)->GetExpression();
       Expr *op2 = static_cast<CG_chillRepr*>(rop)->GetExpression();
       
       // Not sure about type!!
       debug_fprintf(stderr, "gonna die in CG_chillBuilder.cc ~line 394\n"); 
       BinaryOperator *ins = NULL; // new (astContext_)BinaryOperator(op1, op2, BO_Rem, op1->getType(), SourceLocation());
       
       delete lop; delete rop;
       return new CG_chillRepr(ins);
    */
  }
  
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr *CG_chillBuilder::CreateIntegerCeil(CG_outputRepr *lop, CG_outputRepr *rop) const {
    return CreateMinus(NULL, CreateIntegerFloor(CreateMinus(NULL, lop), rop));
  }
  
  
  
  //-----------------------------------------------------------------------------
  // binary logical operations
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateAnd(CG_outputRepr *lop,
                                            CG_outputRepr *rop) const {
    debug_fprintf(stderr, "CG_chillBuilder::CreateAnd()\n");  
    if (rop == NULL)
      return lop;
    else if (lop == NULL)
      return rop;
    
    /* if (rop == NULL || lop == NULL ) {
       debug_fprintf(stderr, "returning NULL!\n"); 
       return NULL;
       }*/
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //debug_fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //debug_fprintf(stderr, " && ");
    //rAST->print(0, stderr);
    //debug_fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "&&", rAST );
    return new CG_chillRepr( binop );
  }
  
  
  //-----------------------------------------------------------------------------
  // binary relational operations
  //-----------------------------------------------------------------------------
  //  CG_outputRepr* CG_chillBuilder::CreateGE(CG_outputRepr *lop,   // use the outputBuilder version
  //                                           CG_outputRepr *rop) const {
  //    
  //    Expr *op1 = static_cast<CG_chillRepr*>(lop)->GetExpression();
  //    Expr *op2 = static_cast<CG_chillRepr*>(rop)->GetExpression();
  
  // Not sure about type!!
  //    debug_fprintf(stderr, "about to die in CG_chillBuilder ~line 480\n"); 
  
  //    BinaryOperator *ins = NULL; // new (astContext_)BinaryOperator(op1, op2, BO_GE, op1->getType(), SourceLocation());
  
  //    delete lop; delete rop;
  //    return new CG_chillRepr(ins);
  //  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateLE(CG_outputRepr *lop,
                                           CG_outputRepr *rop) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateLE()\n");  
    if (rop == NULL || lop == NULL) {
      return NULL;           
    }            
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //debug_fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //debug_fprintf(stderr, " <= ");
    //rAST->print(0, stderr);
    //debug_fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "<=", rAST );
    delete lop; delete rop; // ?? 
    return new CG_chillRepr( binop );
  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateEQ(CG_outputRepr *lop,
                                           CG_outputRepr *rop) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateEQ()\n");  
    if (rop == NULL || lop == NULL) {
      return NULL;           
    }            
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //debug_fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //debug_fprintf(stderr, " = ");
    //rAST->print(0, stderr);
    //debug_fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "==", rAST );
    delete lop; delete rop; // ?? 
    return new CG_chillRepr( binop );
  }
  
  
  
  
  CG_outputRepr* CG_chillBuilder::CreateNEQ(CG_outputRepr *lop,
                                            CG_outputRepr *rop) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateNEQ()\n");  
    if (rop == NULL || lop == NULL) {
      return NULL;           
    }            
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //debug_fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //debug_fprintf(stderr, " != ");
    //rAST->print(0, stderr);
    //debug_fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "!=", rAST );
    delete lop; delete rop; // ?? 
    return new CG_chillRepr( binop );
  }
  
  
  CG_outputRepr* CG_chillBuilder::CreateDotExpression(CG_outputRepr *lop,
                                                      CG_outputRepr *rop) const {
    //debug_fprintf(stderr, "\nCG_chillBuilder::CreateDotExpression()\n");  
    if (rop == NULL || lop == NULL) {
      return NULL;           
    }            
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    //debug_fprintf(stderr, "left is %s,  right is %s\n", lAST->getTypeString(), rAST->getTypeString()); 
    
    if ( !rAST->isVarDecl()) { 
      debug_fprintf(stderr, "CG_chillBuilder::CreateDotExpression() right is a %s, not a vardecl\n",
              rAST->getTypeString());
      exit(-1); 
    }
    chillAST_VarDecl *rvd = (chillAST_VarDecl *)rAST;
    //debug_fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //debug_fprintf(stderr, ".");
    //rAST->print(0, stderr);
    //debug_fprintf(stderr, "  ??\n"); 
    
    //chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, ".", rAST, NULL);
    
    
    // MemberExpr should be a DeclRefExpr on the left?
    chillAST_DeclRefExpr *DRE = NULL;
    if (lAST->isDeclRefExpr()) DRE = (chillAST_DeclRefExpr *)lAST; 
    if (lAST->isVarDecl()) { 
      // make a DeclRefExpr ?  probably an error upstream of here in this case
      DRE = new chillAST_DeclRefExpr( (chillAST_VarDecl *)lAST ); 
    }
    if (!DRE) { 
      debug_fprintf(stderr, "CG_chillBuilder::CreateDotExpression(), can't create base\n");
      exit(-1); 
    }
    chillAST_MemberExpr *memexpr = new chillAST_MemberExpr( DRE, rvd->varname, NULL, CHILL_MEMBER_EXP_DOT );
    
    
    //delete lop; delete rop; // ??  
    return new CG_chillRepr( memexpr );
  }
  
  
  //-----------------------------------------------------------------------------
  // stmt list gen operations
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateStmtList(CG_outputRepr *singleton) const {
    //debug_fprintf(stderr, "CG_chillBuilder::CreateStmtList()\n");  
    if(singleton == NULL) return NULL;
    
    exit(-1);                  // DFL 
    return( NULL ); 
    /* 
       StmtList *tnl = static_cast<CG_chillRepr *>(singleton)->GetCode();
       
       if(tnl->empty()) {
       StmtList foo;
       debug_fprintf(stderr, "gonna die soon  CG_chillBuilder::CreateStmtList()\n");  exit(-1); 
       //foo.push_back(static_cast<CG_chillRepr*>(singleton)->op_);
       return new CG_chillRepr(foo);
       }
       delete singleton;
       return new CG_chillRepr(*tnl);
    */
  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::StmtListInsertLast(CG_outputRepr *list, 
                                                     CG_outputRepr *node) const {
    return StmtListAppend(list, node);
  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::StmtListAppend(CG_outputRepr *list1, 
                                                 CG_outputRepr *list2) const {
    
    //debug_fprintf(stderr, "CG_chillBuilder::StmtListAppend()\n"); 
    
    if(list1 == NULL) return list2;
    else if(list2 == NULL) return list1;
    
    CG_chillRepr *cr1 = (CG_chillRepr *)list1;
    CG_chillRepr *cr2 = (CG_chillRepr *)list2;
    
    int numtoadd = cr2->chillnodes.size();
    //debug_fprintf(stderr, "before: %d nodes and %d nodes\n", cr1->chillnodes.size(), numtoadd ); 
    for (int i=0; i<numtoadd; i++){
      (cr1->chillnodes).push_back(cr2->chillnodes[i] );
    }
    //debug_fprintf(stderr, "after %d nodes\n", cr1->chillnodes.size() ); 
    
    delete list2;
    return list1;
    
  }
  
  
  bool CG_chillBuilder::QueryInspectorType(const std::string &varName) const {
    debug_fprintf(stderr, "CG_chillBuilder::QueryInspectorType( %s )\n", varName.c_str()); 
    int *i=0; int j= i[0]; 
    return false;
  }
  
  
  CG_outputRepr* CG_chillBuilder::CreateArrayRefExpression(const std::string &_s,
                                                           CG_outputRepr *rop) const {
    throw std::runtime_error("create");
    chillAST_node *l = new chillAST_DeclRefExpr(_s.c_str());
    chillAST_node *r = ((CG_chillRepr *)rop)->GetCode();

    chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( l, r, NULL, 0); // unique TODO
    return new CG_chillRepr( ASE );
  }
  
  
  CG_outputRepr* CG_chillBuilder::CreateArrayRefExpression(CG_outputRepr*left, 
                                                           CG_outputRepr*right) const{
    
    chillAST_node *l = ((CG_chillRepr *)left)->GetCode();
    chillAST_node *r = ((CG_chillRepr *)right)->GetCode();

    chillAST_node *base = NULL; 
    
    if (l->isDeclRefExpr()) base = l;
    if (l->isMemberExpr()) base = l;
    if (l->isVarDecl()) { // ?? 
      // make a declRefExpr that uses VarDecl l
      base = (chillAST_node *) new chillAST_DeclRefExpr( (chillAST_VarDecl *)l );
    }
    if (l->isArraySubscriptExpr())
      base = new chillAST_DeclRefExpr(l->multibase());
    
    if (!base)  {
      debug_fprintf(stderr, "CG_chillBuilder::CreateArrayRefExpression(), left is %s\n", l->getTypeString()); 
      
      exit(-1);
    }
    
    
    
    chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( base, r, NULL, 0); // unique TODO
    return new CG_chillRepr( ASE );
  }
  
  
  CG_outputRepr* CG_chillBuilder::ObtainInspectorData(const std::string &_s, const std::string &member_name) const{
    debug_fprintf(stderr, "CG_chillBuilder::ObtainInspectorData( %s, %s)\n", 
            _s.c_str(), member_name.c_str());
    
    //WTF 

    return ObtainInspectorRange( _s, member_name ); 
  }
  
  
  CG_outputRepr *CG_chillBuilder::CreateAddressOf(CG_outputRepr* op) const {
    debug_fprintf(stderr, "CG_chillBuilder::CreateAddressOf()\n");
    exit(-1);
  }
  
  CG_outputRepr* CG_chillBuilder::CreateBreakStatement() const { 
    debug_fprintf(stderr, "CG_chillBuilder::CreateBreakStatement()\n");
    exit(-1);
  }
  
  
  CG_outputRepr *CG_chillBuilder::CreateStatementFromExpression(CG_outputRepr *exp) const { 
    debug_fprintf(stderr, "CG_chillBuilder::CreateStatementFromExpression()\n");
    exit(-1);
  }
  
  


  CG_outputRepr *CG_chillBuilder::CreateStruct(const std::string struct_name,
                                               std::vector<std::string> data_members,
                                               std::vector<CG_outputRepr *> data_types)
  { 
    
    debug_fprintf(stderr, "\nCG_chillBuilder::CreateStruct( %s )\n", struct_name.c_str()); 
    
/* WRONG - a typedef 
    // NEED TO ADD TYPEDEF TO ... SOMETHING 
    
    chillAST_TypedefDecl *tdd = new chillAST_TypedefDecl( ) ;
    
    tdd->setStructName(struct_name.c_str()); 
    tdd->setStruct( true ); 
    int n_memb = data_members.size();
    int n_data_types = data_types.size();
    for (int i=0; i<n_memb; i++) { 
      chillAST_VarDecl *vd;
      debug_fprintf(stderr, "member %s type ", data_members[i].c_str()); 
      if (i <n_data_types) {
        vd = (chillAST_VarDecl *) ((CG_chillRepr *)data_types[i])->GetCode(); 
        vd->varname = strdup(  data_members[i].c_str() ); 
        bool simplepointer = (vd->numdimensions == 1 && !vd->knownArraySizes);
        if (simplepointer) debug_fprintf(stderr, "pointer to "); 
        debug_fprintf(stderr, "%s\n", vd->vartype );
        if (vd->numdimensions > 0 && vd->knownArraySizes) {
          for (int k=0; k<vd->numdimensions; k++) debug_fprintf(stderr, "[%d]", vd->arraysizes[k]);
        }
      }
      else { 
        debug_fprintf(stderr, "type int BY DEFAULT (bad idea)\n");
        vd = new chillAST_VarDecl( "int", data_members[i].c_str(), "", NULL);
      }
      // add vd to suparts of the struct typedef 
      tdd->subparts.push_back( vd ); 
      
      debug_fprintf(stderr, "\n"); 
    }
    
    // put the typedef in the top level ... for now   TODO 
    toplevel->insertChild( 0, tdd); 
    return new CG_chillRepr( tdd ); 
*/


    chillAST_RecordDecl *rd = new chillAST_RecordDecl(struct_name.c_str());
    rd->setStruct( true ); 
    // SO FAR, struct has no members! 

    toplevel->insertChild(0, rd);  // inserts at beginning of file, (after defines?)
    // note: parent at top level so far   TODO 
    //toplevel->print(); printf("\n\n");  fflush(stdout); 

    int n_memb       = data_members.size();
    int n_data_types = data_types.size();
    // add struct members
    for (int i=0; i<n_memb; i++) { 
      chillAST_VarDecl *vd = NULL;
      //debug_fprintf(stderr, "%d member %s type ", i, data_members[i].c_str()); 
      if (i < n_data_types) { 
        // this should always happen, formerly, if no data type was 
        // specified, it was an int. bad idea
        vd = (chillAST_VarDecl *) ((CG_chillRepr *)data_types[i])->GetCode(); 

        // vd did not have a name before 
        vd->varname = strdup(  data_members[i].c_str() ); 

        vd->parent = rd;  // ??

        bool simplepointer = (vd->numdimensions == 1 && vd->isPointer());
        if (simplepointer) {  
          debug_fprintf(stderr, "struct member %s is pointer to %s\n", vd->varname, vd->vartype);
          vd->arraypointerpart = strdup("*"); // ?? 
        }
        else { 
          //debug_fprintf(stderr, "struct member %s is not a pointer TODO!\n", vd->varname); 
          debug_fprintf(stderr, "struct member %s is %s\n", vd->varname, vd->vartype); 
          
          // it should be good to go ??? 
        }
        //vd->print(); printf("\n"); fflush(stdout); 
        //debug_fprintf(stderr, "%s\n", vd->vartype );
        //if (vd->numdimensions > 0 && vd->knownArraySizes) {
        //  for (int k=0; k<vd->numdimensions; k++) debug_fprintf(stderr, "[%d]", vd->arraysizes[k]);
        //} 
      }
      else { 
        debug_fprintf(stderr, "int BY DEFAULT (bad idea) FIXME\n"); // TODO 
        vd = new chillAST_VarDecl( "int", "", data_members[i].c_str());
      }
      rd->addSubpart( vd );
      //debug_fprintf(stderr, "\n"); 
    }
    debug_fprintf(stderr, "\n"); 
    return new CG_chillRepr( rd ); 
  }
  
  
  
  CG_outputRepr *CG_chillBuilder::CreateClassInstance(std::string name ,  // TODO can't make array
                                                      CG_outputRepr *class_def){
    debug_fprintf(stderr, "CG_chillBuilder::CreateClassInstance( %s )\n", name.c_str()); 
    
    CG_chillRepr *CD = (CG_chillRepr *)class_def; 
    chillAST_node *n = CD->GetCode();
    //debug_fprintf(stderr, "class def is of type %s\n", n->getTypeString());
    //n->print(); printf("\n"); fflush(stdout); 

    if (n->isTypeDefDecl()) { 
      chillAST_TypedefDecl *tdd = (chillAST_TypedefDecl *)n;
      //tdd->print(); printf("\n"); fflush(stdout);
      
      chillAST_VarDecl *vd = new chillAST_VarDecl( tdd, "", name.c_str());
      
      // we need to add this to function ??  TODO 
      //debug_fprintf(stderr, "adding typedef instance to symbolTable\n");
      chillAST_SymbolTable *st =  currentfunction->getBody()->getSymbolTable();
      //printSymbolTable(st); 

      currentfunction->getBody()->addVariableToSymbolTable( vd ); // TODO 
      currentfunction->getBody()->insertChild(0, vd);  // TODO 
      //printSymbolTable(st); 
      
      return new CG_chillRepr( vd ); 
    }
    if  (n->isRecordDecl()) { 
      debug_fprintf(stderr, "a RecordDecl\n"); 

      chillAST_RecordDecl *rd = (chillAST_RecordDecl *) n;
      rd->print(); printf("\n"); fflush(stdout);
      rd->dump(); printf("\n");  fflush(stdout);
      
      chillAST_VarDecl *vd = new chillAST_VarDecl( rd, "", name.c_str());

      //debug_fprintf(stderr, "CG_chillBuilder.cc, adding struct instance to body of function's symbolTable\n");


      // we need to add this to function ??  TODO 
      currentfunction->getBody()->addVariableToSymbolTable( vd ); // TODO 
      currentfunction->getBody()->insertChild(0, vd);  // TODO 
      //printf("\nafter adding vardecl, source is:\n");
      currentfunction->getBody()->print(); fflush(stdout);

      //printf("\nafter adding vardecl, symbol table is:\n"); 
      chillAST_SymbolTable *st =  currentfunction->getBody()->getSymbolTable();
      //printSymbolTable(st); fflush(stdout); 
      
      return new CG_chillRepr( vd ); 
    }

    debug_fprintf(stderr, "ERROR: CG_chillBuilder::CreateClassInstance() not sent a class or struct\n"); 
    int *i=0; int j = i[0]; 
    return NULL; 
  }
  
  
  
  CG_outputRepr *CG_chillBuilder::lookup_member_data(CG_outputRepr* classtype, 
                                                     std::string varName, 
                                                     CG_outputRepr *instance) {
    
    
    //debug_fprintf(stderr, "CG_chillBuilder::lookup_member_data( %s )\n", varName.c_str()); 
    
    chillAST_VarDecl* sub = NULL;

    CG_chillRepr *CR = (CG_chillRepr *)classtype;
    chillAST_node *classnode = CR->GetCode();
    //debug_fprintf(stderr, "classnode is %s\n", classnode->getTypeString()); classnode->print(); printf("\n"); fflush(stdout); 
    if (! ( classnode->isTypeDefDecl() || 
            classnode->isRecordDecl() )) { 
      debug_fprintf(stderr, "ERROR: CG_chillBuilder::lookup_member_data(), classnode is not a TypeDefDecl or a RecordDecl\n"); 
      exit(-1); 
    }


    CG_chillRepr *CI = (CG_chillRepr *)instance; 

    chillAST_node *in = CI->GetCode();
    //debug_fprintf(stderr, "instance is %s\n", in->getTypeString()); 
    //in->print(); printf("\n"); fflush(stdout); 
    
    if ( !in->isVarDecl() ) { // error, instance needs to be a vardecl
      debug_fprintf(stderr, "ERROR: CG_chillBuilder::lookup_member_data() instance needs to be a VarDecl, not a %s", in->getTypeString());
      exit(-1);
    }
    chillAST_VarDecl *vd = (chillAST_VarDecl *)in;
    if (vd->typedefinition != classnode && 
      vd->vardef != classnode) { 
      debug_fprintf(stderr, "vd: typedef %p  vardev %p    classnode %p\n", vd->typedefinition, vd->vardef, classnode); 
      debug_fprintf(stderr, "CG_chillBuilder::lookup_member_data(), instance is not of correct class \n");
      
      exit(-1);
    }
    
    

    if (classnode->isTypeDefDecl()){ 
      chillAST_TypedefDecl *tdd = (chillAST_TypedefDecl *)classnode;
      if ( !tdd->isAStruct() ) { 
        debug_fprintf(stderr, "ERROR: CG_chillBuilder::lookup_member_data() instance must be a struct or class\n");
        exit(-1);
      }
      
      sub = tdd->findSubpart( varName.c_str() ); 
    }

    if (classnode->isRecordDecl()){ 
      chillAST_RecordDecl *rd = (chillAST_RecordDecl *)classnode;
      if ( !rd->isAStruct() ) { 
        debug_fprintf(stderr, "ERROR: CG_chillBuilder::lookup_member_data() instance must be a struct or class\n");
        exit(-1);
      }
      
      //debug_fprintf(stderr, "looking for member (subpart) %s in RecordDecl\n",  varName.c_str()); 
      sub = rd->findSubpart( varName.c_str() ); 
    }   

    if (!sub) {
      debug_fprintf(stderr, "CG_chillBuilder::lookup_member_data(), variable %s is not submember of class/struct\n"); 
      exit(-1);
    }
    
    //debug_fprintf(stderr, "subpart (member) %s is\n", varName.c_str()); sub->print(); printf("\n"); fflush(stdout);

    return( new CG_chillRepr( sub ) ); // the vardecl inside the struct typedef 
  }
  
  
  CG_outputRepr* CG_chillBuilder::CreatePointer(std::string  &name) const { 
    //debug_fprintf(stderr, "CG_chillBuilder::CreatePointer( %s )\n", name.c_str()); 
    
    chillAST_VarDecl *vd = new chillAST_VarDecl( "int", "", name.c_str());
    //vd->print(); printf("\n"); fflush(stdout); 
    //vd->dump(); printf("\n"); fflush(stdout); 
    
    //printSymbolTable( currentfunction->getBody()->getSymbolTable() ); 

    chillAST_DeclRefExpr *dre = new chillAST_DeclRefExpr( vd ); // ?? 
    return new CG_chillRepr( dre );  // need a declrefexpr? 
  }
  

  CG_outputRepr* CG_chillBuilder::ObtainInspectorRange(const std::string &structname, const std::string &member) const {
    //debug_fprintf(stderr, "CG_chillBuilder::ObtainInspectorRange(%s,  %s )\n", structname.c_str(), member.c_str()); 
    
    // find a struct/class with name structname and member member
    // return a Member access (or binary dot op )
    // seems like you need to know where (scoping) to look for the struct definition
    
    vector<chillAST_VarDecl*> decls;
    currentfunction->gatherVarDecls( decls );
    //debug_fprintf(stderr, "\nfunc has %d vardecls  (looking for %s)\n", decls.size(), structname.c_str()); 
    
    chillAST_VarDecl *thestructvd = NULL;
    for (int i=0; i<decls.size(); i++) { 
      chillAST_VarDecl *vd = decls[i];
      //vd->print(); printf("\n"); fflush(stdout); 
      
      if (structname == vd->varname) { 
        //debug_fprintf(stderr, "found it!\n"); 
        thestructvd = vd;
        break;
      }
    }
    
    if (!thestructvd) { 
      debug_fprintf(stderr, "CG_chillBuilder::ObtainInspectorRange could not find variable named %s in current function\n", structname.c_str()); 
      exit(-1); 
    }
    
    // make sure the variable is a struct with a member with the correct name
    chillAST_RecordDecl *rd = thestructvd->getStructDef(); 
    if ( !rd ) { 
      debug_fprintf(stderr, "CG_chillBuilder::ObtainInspectorRange(), variable %s is not a struct/class\n",  structname.c_str()); 
      exit(-1);
    }
    
    
    chillAST_VarDecl *sub = rd->findSubpart( member.c_str() ); 
    if (!sub) { 
      debug_fprintf(stderr, "CG_chillBuilder::ObtainInspectorRange(), struct/class %s has no member named %s\n",  structname.c_str(), member.c_str()); 
      exit(-1); 
    }
    
    
    // build up a member expression  (or a binop with dot operation?? )
    // make a declrefexpr that refers to this variable definition
    chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( thestructvd ); 
    chillAST_MemberExpr *ME = new chillAST_MemberExpr( DRE, member.c_str(), NULL ); // uniq TODO
    
    return new CG_chillRepr( ME ); 
  }
  
} // namespace


