
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
#include <code_gen/CG_chillBuilder.h>

namespace omega {
  
  // substitute at chill AST level
  // forward declarations
  chillAST_node *substituteChill(       const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubABinaryOperator(    const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubUnaryOperator(      const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubDeclRefExpr(        const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubArraySubscriptExpr( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubImplicitCastExpr(   const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubCStyleCastExpr(     const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubParenExpr(          const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubCallExpr(           const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubReturnStmt(         const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubIfStmt(             const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent );
  chillAST_node *SubCompoundStmt(       const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent ); 
  chillAST_node *SubMemberExpr(         const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent ); 
  
  
  
  //  chillAST_node *Sub( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent ); // fwd decl
  
  
  
  
  chillAST_node *substituteChill( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    if (n == NULL) {
      fprintf(stderr, "substituteChill() pointer n == NULL\n"); // DIE 
      int *crash = 0;
      crash[0] = 1; 
      exit(-1);
    }
    
    //fprintf(stderr, "substituteChill()    subbing statement of type ");
    //fprintf(stderr, "%s\n", n->getTypeString());
    //if (n->isImplicitCastExpr()) { 
    //  chillAST_ImplicitCastExpr *ICE = (chillAST_ImplicitCastExpr *) n;
    //  fprintf(stderr, "ICE subexpr type %s\n", ICE->subexpr->getTypeString());
    //} 
    //fprintf(stderr, "subbing '%s' in statement ", oldvar); n->print(0, stderr); fprintf(stderr, "\n"); 
    
    chillAST_node *r = n;
    if        (n->isBinaryOperator())     {r=   SubABinaryOperator(oldvar, newvar, n, parent ); 
    } else if (n->isUnaryOperator())      {r=     SubUnaryOperator(oldvar, newvar, n, parent ); 
    } else if (n->isDeclRefExpr())        {r=       SubDeclRefExpr(oldvar, newvar, n, parent ); 
    } else if (n->isArraySubscriptExpr()) {r=SubArraySubscriptExpr(oldvar, newvar, n, parent ); 
    } else if (n->isImplicitCastExpr())   {r=  SubImplicitCastExpr(oldvar, newvar, n, parent ); 
    } else if (n->isParenExpr())          {r=         SubParenExpr(oldvar, newvar, n, parent ); 
    } else if (n->isCStyleCastExpr())     {r=    SubCStyleCastExpr(oldvar, newvar, n, parent ); 
    } else if (n->isReturnStmt())         {r=        SubReturnStmt(oldvar, newvar, n, parent ); 
    } else if (n->isIfStmt())             {r=            SubIfStmt(oldvar, newvar, n, parent ); 
    } else if (n->isCallExpr())           {r=          SubCallExpr(oldvar, newvar, n, parent );
    } else if (n->isCompoundStmt())       {r=      SubCompoundStmt(oldvar, newvar, n, parent );
    } else if (n->isMemberExpr())         {r=        SubMemberExpr(oldvar, newvar, n, parent );
      
    } else if (n->isFloatingLiteral())    {  //fprintf(stderr, "sub in FL\n"); // do nothing
    } else if (n->isIntegerLiteral())     {  // do nothing 
      
    } else {
      fprintf(stderr, "\nCG_chillBuilder.cc substituteChill() UNHANDLED statement of type ");
      n->dump(); printf("   "); n->print(); printf("\n"); fflush(stdout); 
      fprintf(stderr, "%s\n", n->getTypeString()); 
      exit(-1);
    }
    
    /*
      if (isa<DeclStmt>(s))                  {         SubDeclStmt(oldvar, newvar, n, parent );
      } else if (isa<UnaryOperator>(s))      {    SubUnaryOperator(oldvar, newvar, n, parent );
      } else if (isa<ForStmt>(s))            {             SubLoop(oldvar, newvar, n, parent );
    */  
    
    return r;
  }
  
  
  chillAST_node *SubABinaryOperator( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    chillAST_BinaryOperator *b = (chillAST_BinaryOperator *) n; 
    //fprintf(stderr,"SubABinaryOperator() 0x%x  subbing old variable %s in \n", b, oldvar); 
    

    //fprintf(stderr,"SubABinaryOperator() subbing old variable %s in \n", oldvar); 
    //if (b->lhs!=NULL  && b->rhs!=NULL) {
    //  b->print(); printf("\n"); fflush(stdout); 
    //} 
    
    chillAST_node *lhs = b->lhs;
    chillAST_node *rhs = b->rhs;
    
    //if (!strcmp(b->op, "=") && rhs->isBinaryOperator() ) { 
    //  chillAST_BinaryOperator *r = (chillAST_BinaryOperator *) rhs;  
    //  fprintf(stderr, "a(%p) = b(%p) %s c(%p)\n", lhs, r->lhs, r->op, r->rhs );
    //} 
    
    //fprintf(stderr, "op %s   rhs type ", b->op);
    //fprintf(stderr, "%s\n", rhs->getTypeString()); 
    //rhs->dump(); printf("\n"); fflush(stdout);
    
    
    b->lhs = substituteChill( oldvar, newvar, lhs, b);
    b->rhs = substituteChill( oldvar, newvar, rhs, b);
    return b;
  }
  


  chillAST_node *SubUnaryOperator( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    chillAST_UnaryOperator *u = (chillAST_UnaryOperator *) n; 
    chillAST_node *sub = u->subexpr; 
    u->subexpr = substituteChill( oldvar, newvar, sub, u);
    return u;
  }

  
  chillAST_node *SubDeclRefExpr( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    //fprintf(stderr, "SubDeclRefExpr() subbing statement of type %s\n", n->getTypeString());
    
    chillAST_DeclRefExpr *DRE = (chillAST_DeclRefExpr *) n;
    //const char *variable = DRE->declarationName; // should be the same as oldvar ?? 
    
    //fprintf(stderr, "looking for oldvar %s in old DRE code ", oldvar);
    //n->print(); printf("\n"); fflush(stdout); 
    
    //fprintf(stderr, "old DRE name was %s\n", DRE->declarationName);
    if (streq( oldvar,  DRE->declarationName)) { 
      //fprintf(stderr, "yep. replacing\n"); 
      
      
      //fprintf(stderr, "\nNEED TO REPLACE VARIABLE %s with new thing ", oldvar);  
      //newvar->printChillNodes(); 
      
      
      //  newvar->Dump();  printf("\n"); fflush(stdout); 
      //  //fprintf(stderr, " in statement of type %s\n",s->getTypeString());
      //} 
      
      vector<chillAST_node*> newnodes = newvar->chillnodes;
      //fprintf(stderr, "%d nodes in newvar\n", newnodes.size());
      chillAST_node *firstn = newnodes[0];
      firstn->parent = parent;
      return firstn;   // it's that simple!
      
    } 
    //else fprintf(stderr, "nope. not the right thing to replace\n\n");
    
    
    return DRE; // unchanged 
  } // subdeclrefexpr
  
  
  
  
  
  chillAST_node *SubArraySubscriptExpr( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    chillAST_ArraySubscriptExpr *ASE = (chillAST_ArraySubscriptExpr *) n; 
    
    //fprintf(stderr, "subASE   ASE 0x%x\n", ASE); 
    //fprintf(stderr, "SubArraySubscriptExpr subbing old variable %s with new thing in ASE 0x%x  ", oldvar, ASE);
    
    //ASE->print(); printf("\n"); fflush(stdout);
    
    chillAST_node *Base  = ASE->base;
    chillAST_node *Index = ASE->index;
    //fprintf(stderr, "Index is of type %s\n", Index->getTypeString()); 
    
    ASE->base  = substituteChill( oldvar, newvar, Base,  ASE);  // this should not do anything 
    ASE->index = substituteChill( oldvar, newvar, Index, ASE); // this should
    
    //if (Index != ASE->index) {
    //  fprintf(stderr, "ASE was "); 
    //  Base->print(); 
    //  printf("["); 
    //  Index->print();
    //  printf("]\n"); 
    //  printf("SWAPPED INDEX ASE 0x%x  is ", ASE); ASE->print(); printf("\n"); fflush(stdout); 
    //} 
    //else fprintf(stderr, "ASE  is "); ASE->print(); printf("\n"); fflush(stdout); 
    return ASE;
  }
  
  
  
  chillAST_node *SubImplicitCastExpr( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    //fprintf(stderr, "SubImplicitCastExpr subbing statement of type %s at 0x%x    parent 0x%x\n", n->getTypeString(), n, parent);
    chillAST_ImplicitCastExpr *IC = (chillAST_ImplicitCastExpr *) n; 
    chillAST_node *oldsub = IC->subexpr;
    IC->subexpr = substituteChill( oldvar, newvar, oldsub, IC); 
    
    //if (oldsub != IC->subexpr) { 
    //fprintf(stderr, "ImplicitCastExpr has CHANGED\n");
    //IC->print(); printf("\n"); fflush(stdout); 
    //fprintf(stderr, "ICE was "); 
    //oldsub->print(); 
    //printf("\nSWAPPED subexpr ICE 0x%x  is ", IC); IC->print(); printf("\n"); fflush(stdout); 
    //fprintf(stderr, "PARENT 0x%x is now ",IC->parent);
    //IC->parent->print(); printf("\n"); fflush(stdout); 
    //} 
    return IC; 
  }
  
  chillAST_node *SubCStyleCastExpr( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    //fprintf(stderr, "SubCStyleCastExpr()  subexpr is type ");
    chillAST_CStyleCastExpr *CSCE = (chillAST_CStyleCastExpr *) n;
    //fprintf(stderr, "%s\n", CSCE->subexpr->getTypeString()); 
    CSCE->subexpr = substituteChill( oldvar, newvar, CSCE->subexpr, CSCE);
    return CSCE;
  }
  
  
  chillAST_node *SubParenExpr( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    chillAST_ParenExpr *PE = (chillAST_ParenExpr *) n;
    PE->subexpr = substituteChill( oldvar, newvar, PE->subexpr, PE);
    return PE;
  }
  
  chillAST_node *SubCallExpr( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    chillAST_CallExpr *CE = (chillAST_CallExpr *) n;
    
    //fprintf(stderr, "substituting for oldvar %s in ", oldvar );
    //CE->print(); printf("\n"); fflush(stdout); 
    
    int nargs = CE->numargs;
    for (int i=0; i<nargs; i++) {
      CE->args[i] = substituteChill( oldvar, newvar, CE->args[i], CE); 
    }
    return CE; 
  }
  
  
  
  chillAST_node *SubReturnStmt( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    //fprintf(stderr, "SubReturnStmt()\n");
    
    chillAST_ReturnStmt *RS = (chillAST_ReturnStmt *)n;
    if (RS->returnvalue) RS->returnvalue = substituteChill(oldvar, newvar, RS->returnvalue, RS);
    return RS;
  }
  
  
  chillAST_node *SubIfStmt( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    //fprintf(stderr, "SubIfStmt()\n");
    chillAST_IfStmt *IS = (chillAST_IfStmt *)n;
    //IS->print(0, stderr); fprintf(stderr, "\n\n"); 
    chillAST_node *sub;
    if ( sub = IS->getCond() ) IS->setCond( substituteChill(oldvar, newvar, sub, IS)); 
    if ( sub = IS->getThen() ) IS->setThen( substituteChill(oldvar, newvar, sub, IS)); 
    sub = IS->getElse(); //fprintf(stderr, "sub(else) = %p\n", sub); 
    if ( sub = IS->getElse() ) IS->setElse( substituteChill(oldvar, newvar, sub, IS)); 
    
    return IS; 
  }
  
  
  chillAST_node *SubCompoundStmt( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    //fprintf(stderr, "SubCompoundStmt()\n");
    chillAST_CompoundStmt *CS = (chillAST_CompoundStmt *)n;
    
    int numchildren = CS->getNumChildren(); 
    for (int i=0; i<numchildren; i++) { 
      CS->setChild( i, substituteChill(oldvar, newvar, CS->getChild(i), CS )); 
    }
    
    return CS;
  }
  
  
  
  chillAST_node *SubMemberExpr( const char *oldvar, CG_chillRepr *newvar, chillAST_node *n, chillAST_node *parent = NULL ) {
    //fprintf(stderr, "SubMemberExpr(   oldvar %s   ) \n", oldvar);
    chillAST_MemberExpr *ME = (chillAST_MemberExpr *)n;
    
    ME->base =  substituteChill(oldvar, newvar, ME->base, ME ); 
    
    
    // 
    return ME; 
  }
  
  
  
  CG_chillBuilder::CG_chillBuilder() { 
    toplevel = NULL;
    currentfunction = NULL; // not very useful
    symtab_ = symtab2_ = NULL; 
  }
  
  CG_chillBuilder::CG_chillBuilder(chillAST_SourceFile *top, chillAST_FunctionDecl *func) { 
    //fprintf(stderr, "\nCG_chillBuilder::CG_chillBuilder()\n"); 
    toplevel = top;
    currentfunction = func;
    
    //fprintf(stderr, "\nfunction is:\n"); currentfunction->print(); printf("\n\n"); fflush(stdout); 

    symtab_  = &(currentfunction->parameters); // getSymbolTable();             // TODO rename 
    symtab2_ = currentfunction->getBody()->getSymbolTable(); // TODO rename
    
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
    fprintf(stderr, "CG_chillBuilder::CreatePlaceHolder()  TODO \n");
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
    fprintf(stderr, "\n\nin CG_xxxxBuilder.cc (OMEGA)  CG_xxxxBuilder::CreateSubstitutedStmt()\n");
    fprintf(stderr, "%d vars and %d substitutions\n", numvars, (int)subs.size());
    
    
    if (numvars != numsubs) {
      //fprintf(stderr, "umwut?\n"); exit(-1); 
    }
    
    
    //{
    //  vector<chillAST_node*> nodes = ((CG_chillRepr *) stmt)->getChillCode(); 
    //  // 
    //  fprintf(stderr, "%d nodes in old code. was:\n", nodes.size()); 
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
    //  fprintf(stderr, "sub %d  ", i); 
    //  if (subs[i]) {   ((CG_chillRepr *)subs[i])->Dump(); fflush( stdout );  }
    //  else  { 
    //    //int *crash = NULL;  *crash = 1; 
    //    fprintf(stderr, "(NULL  error!)"); 
    //  }
    //  //fprintf(stderr, "\n"); 
    //} 
    //fprintf(stderr, "\n"); 
    
    
    if (numsubs == 0) {
      
      vector<chillAST_node*> nodes = ((CG_chillRepr *) stmt)->getChillCode(); 
      
      // 
      //fprintf(stderr, "nosubs old code was:\n"); 
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
      
      //fprintf(stderr, "since nothing actually being substituted, this is just a clone\n"); 
      //fprintf(stderr, "old code was AND new code is:\n");
      //for (int i=0; i<nodes.size(); i++) { 
      //  fprintf(stderr, "stmt = ");
      //  nodes[i]->print();  fflush(stdout); 
      //  fprintf(stderr, "\n"); 
      //} 
      //fprintf(stderr, "cloning()\n"); 
      //return stmt->clone(); 
      
    }



    //fprintf(stderr, "numsubs %d\n", numsubs);
    
    // debugging: print the substitutions we'll do 
    
    //if (numsubs > 0) { 
    //  for (int i=0; i< numsubs; i++)        {
    //    fprintf(stderr, "subbing "); 
    //    if (subs[i]) {   
    //      ((CG_chillRepr *)subs[i])->Dump(); fflush( stdout ); 
    //      fprintf(stderr, "for  %s\n", vars[i].c_str() );
    //}        else  { 
    //      //int *crash = NULL;  *crash = 1; 
    //      fprintf(stderr, "(NULL  error!)"); 
    //    }
    //    //fprintf(stderr, "\n"); 
    //  }
    //  fprintf(stderr, "\n"); 
    //} 
    
    
    
    //fprintf(stderr, "OK, now to really substitute ...\n");  
    //CG_outputRepr *newstmt = stmt->clone();
    //CG_chillRepr *n = (CG_chillRepr *) newstmt; 
    //vector<chillAST_node*> newnodes =  n->getChillCode();  
    
    CG_chillRepr *old = (CG_chillRepr *) stmt; 
    vector<chillAST_node*> oldnodes = old->getChillCode();
    
    
    for (int j=0; j<numsubs; j++) { 
      if (subs[j] != NULL) {
        
        //fprintf(stderr, "substitution %d    %s -> ", j,vars[j].c_str()); 
        //if (subs[j]) {  ((CG_chillRepr *)subs[j])->Dump(); fflush( stdout );  }
        
        
        // find the type of thing we'll be using to replace the old variable
        CG_chillRepr *CRSub = (CG_chillRepr *)(subs[j]); 
        vector<chillAST_node*> nodes = CRSub->chillnodes;
        if (1 != nodes.size() )  { // always just one? 
          fprintf(stderr, "CG_chillBuilder::CreateSubstitutedStmt(), replacement is not one statement??\n");
          exit(-1);
        }
        chillAST_node *node = nodes[0]; // always just one? 
        
        for (int i=0; i<oldnodes.size(); i++) { 
          //fprintf(stderr, "   statement %d    ", i);
          //oldnodes[i]->print();  printf("\n\n"); fflush(stdout); 
          oldnodes[i] = substituteChill( vars[j].c_str(), CRSub, oldnodes[i]);
        }
      }
    }
    
    //fprintf(stderr, "\ncode after substituting variables:\n");
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
    //fprintf(stderr, "CG_chillBuilder::CreateAssignment()\n"); 
    if(lhs == NULL || rhs == NULL) {
      fprintf(stderr, "Code generation: Missing lhs or rhs\n");
      return NULL;
    }
    
    CG_chillRepr *clhs = (CG_chillRepr *) lhs;
    CG_chillRepr *crhs = (CG_chillRepr *) rhs;
    chillAST_node *lAST = clhs->chillnodes[0]; // always just one?
    chillAST_node *rAST = crhs->chillnodes[0]; // always just one?
    
    chillAST_BinaryOperator *bop = new chillAST_BinaryOperator(lAST->clone(), "=", rAST->clone(), NULL); // clone??
    
    delete lhs; delete rhs;
    return new CG_chillRepr(bop);
  }
  
  
  
  
  CG_outputRepr* CG_chillBuilder::CreatePlusAssignment(int indent,               // += 
                                                       CG_outputRepr *lhs,
                                                       CG_outputRepr *rhs) const {
    //fprintf(stderr, "CG_chillBuilder::CreatePlusAssignment()\n"); 
    if(lhs == NULL || rhs == NULL) {
      fprintf(stderr, "Code generation: Missing lhs or rhs\n");
      return NULL;
    }
    
    CG_chillRepr *clhs = (CG_chillRepr *) lhs;
    CG_chillRepr *crhs = (CG_chillRepr *) rhs;
    chillAST_node *lAST = clhs->chillnodes[0]; // always just one?
    chillAST_node *rAST = crhs->chillnodes[0]; // always just one?
    
    chillAST_BinaryOperator *bop = new chillAST_BinaryOperator(lAST->clone(), "+=", rAST->clone(), NULL); // clone??
    
    delete lhs; delete rhs;
    return new CG_chillRepr(bop);
  }
  
  
  
  
  //-----------------------------------------------------------------------------
  // function invocation generation
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateInvoke(const std::string &fname,
                                               std::vector<CG_outputRepr*> &list,
                                               bool is_array) const { // WHAT is an array?
    fprintf(stderr, "CG_roseBuilder::CreateInvoke( fname %s, ...)\n", fname.c_str()); 
    //fprintf(stderr, "%d things in list\n", list.size()); 
    
    // debugging output.  print the "call"
    //fprintf(stderr, "%s", fname.c_str());
    //if (is_array) fprintf(stderr, "["); else fprintf(stderr, "("); 
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
      //fprintf(stderr, "CG_chillBuilder::CreateInvoke() %s is_array\n", fname.c_str());
      const char *arrayname = fname.c_str(); 
      
      CG_chillRepr *CR = (CG_chillRepr *) list[0];
      chillAST_node *cast = CR->GetCode();

      //fprintf(stderr, "%s[",  arrayname);
      //cast->print(); printf("] ???\n"); fflush(stdout);
      
      // find the array variable  (scope ??  TODO) 
      chillAST_VarDecl *array = currentfunction->findArrayDecl( arrayname ); 
      if (!array) { 
        fprintf(stderr, "CG_chillBuilder::CreateInvoke(), can't find array %s\n", fname.c_str()); 
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
        //fprintf(stderr, "else\n"); 
        int last = list.size()-1;
        CG_outputRepr *CGOR; 
        CG_chillRepr  *CGCR;
        
        //fprintf(stderr, "going to create call to %s( ", fname.c_str());
        //for (int i=0; i<list.size(); i++) { 
        //  CGCR = (CG_chillRepr*) list[i];
        //  CGCR->chillnodes[0]->print(0, stderr);
        //  if (i<(list.size()-1)) fprintf(stderr, ", ");
        //} 
        //fprintf(stderr, ")\n"); 
        
        char macroname[32];
        char op; 
        
        if (fname == std::string("max"))  op = '>';
        else op = '<'; 
        
        // TODO >, check number of args etc 
        chillAST_node *ternary = lessthanmacro(  ((CG_chillRepr*) list[0])->chillnodes[0], 
                                                 ((CG_chillRepr*) list[1])->chillnodes[0]);  
        
        //fprintf(stderr, "just made ternary ");
        //ternary->print(0, stdout);
        
        
        CG_chillRepr *repr = new CG_chillRepr( ternary );
        //fprintf(stderr, "returning callexpr with ternary\n", macroname); 
        return repr;
      }
    }
    //else { // special case for reduce? 
    //} 
    else {
      //do a simple function call 
      fprintf(stderr, "building a function call expression\n"); 

      // try to find the function name, for a function in this file
      const char *name = fname.c_str(); 
      //fprintf(stderr, "fname '%s'\n", name);
      chillAST_SourceFile *src = toplevel; // todo don't be dumb
      
      chillAST_node *def = src->findCall(name);
      if (!def) { // can't find it
        fprintf(stderr, "CG_chillBuilder::CreateInvoke( %s ), can't find a function or macro by that name\n", name); 
        exit(-1); 
      }
      
      //fprintf(stderr, "%s is a %s\n", name, def->getTypeString()); 
      if (def->isMacroDefinition()) { 
        chillAST_CallExpr *CE = new chillAST_CallExpr( def, toplevel );
        int numparams = list.size(); 
        for (int i=0; i<numparams; i++) { 
          CG_chillRepr *CR = (CG_chillRepr *) list[i];
          CE->addArg( CR->GetCode() ); 
        }
        return  new CG_chillRepr( CE ); 
      }
      else if (def->isFunctionDecl()) { 
        // TODO are these cases exactly the same?
        chillAST_CallExpr *CE = new chillAST_CallExpr( def, toplevel );
        int numparams = list.size(); 
        for (int i=0; i<numparams; i++) { 
          CG_chillRepr *CR = (CG_chillRepr *) list[i];
          CE->addArg( CR->GetCode() ); 
        }
        return  new CG_chillRepr( CE ); 
      }
      else { 
      }


      // chillAST_CallExpr::chillAST_CallExpr(chillAST_node *function, chillAST_node *p );

      // todo addarg()
      //int numargs;
      //std::vector<class chillAST_node*> args;
      fprintf(stderr, "Code generation: invoke function io_call not implemented\n");
      return NULL;
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
    
    //fprintf(stderr, "in CG_chillBuilder.cc (OMEGA)   CG_chillBuilder::CreateAttribute()\n");
    //fprintf(stderr, "comment = '%s'\n", commentText.c_str()); 
    
    CG_chillRepr *CR = (CG_chillRepr *) control;
    int numnodes = CR->chillnodes.size(); 
    //fprintf(stderr, "%d chill nodes\n", numnodes); 
    if (numnodes > 0) { 
      //fprintf(stderr, "adding a comment to a %s\n", CR->chillnodes[0]->getTypeString()); 
      CR->chillnodes[0]->metacomment = strdup( commentText.c_str()); 
    }
    else { 
      fprintf(stderr, "CG_chillBuilder::CreateAttribute no chillnodes to attach comment to???\n");
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
    //fprintf(stderr, "CG_chillBuilder::CreateIf()\n"); 
    
    if (true_stmtList == NULL && false_stmtList == NULL) {
      delete guardList;
      return NULL;
    }
    else if (guardList == NULL) {  // this seems odd 
      return StmtListAppend(true_stmtList, false_stmtList);
    }
    
    vector<chillAST_node*> vectorcode =  static_cast<CG_chillRepr*>(guardList)->getChillCode();
    if (vectorcode.size() != 1 ) {
      fprintf(stderr, "CG_chillBuilder.cc IfStmt conditional is multiple statements?\n");
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
    
    
    chillAST_IfStmt *if_stmt = new chillAST_IfStmt( conditional, then_part, else_part, NULL); 
    
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
    fprintf(stderr, "\nCG_chillBuilder::CreateInductive()\n");
    if (index == NULL || lower == NULL || upper == NULL) {
      fprintf(stderr, "Code generation: invalid arguments to CreateInductive\n");
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
    //fprintf(stderr, "%d index nodes\n", nodes.size());
    chillAST_node *indexnode = nodes[0];
    if (!streq("DeclRefExpr", indexnode->getTypeString())) {
      fprintf(stderr, "CG_chillBuilder::CreateInductive index is not a DeclRefExpr\n"); 
      if (indexnode->isIntegerLiteral()) fprintf(stderr, "isIntegerLiteral()\n"); 

      fprintf(stderr, "index is %s\n", indexnode->getTypeString());
      indexnode->print(); printf("\n");   fflush(stdout);
      indexnode->dump();  printf("\n\n"); fflush(stdout);
      int *i = 0; int j = i[0];
      exit(-1); 
    }
    
    nodes = static_cast<CG_chillRepr*>(lower)->getChillCode();
    //fprintf(stderr, "%d lower nodes\n", nodes.size());
    chillAST_node *lowernode = nodes[0];
    //fprintf(stderr, "lower node is %s\n", lowernode->getTypeString()); 
    
    nodes = static_cast<CG_chillRepr*>(upper)->getChillCode();
    //fprintf(stderr, "%d upper nodes\n", nodes.size());
    chillAST_node *uppernode = nodes[0];
    //fprintf(stderr, "upper node is %s\n", uppernode->getTypeString()); 
    
    nodes = static_cast<CG_chillRepr*>(step)->getChillCode();
    //fprintf(stderr, "%d step nodes\n", nodes.size());
    chillAST_node *stepnode = nodes[0];
    //fprintf(stderr, "step  node is %s\n",  stepnode->getTypeString()); 
    
    // unclear is this will always be the same 
    // TODO error checking  && incr vs decr
    chillAST_BinaryOperator *init = new  chillAST_BinaryOperator( indexnode, "=", lowernode, NULL); 
    chillAST_BinaryOperator *cond = new  chillAST_BinaryOperator( indexnode, "<=", uppernode, NULL); 
    
    //chillAST_BinaryOperator *inc  = new  chillAST_BinaryOperator( indexnode, "+", stepnode, NULL); 
    chillAST_BinaryOperator *incr = new  chillAST_BinaryOperator( indexnode, "+=", stepnode, NULL); 
    
    chillAST_ForStmt *loop = new chillAST_ForStmt( init, cond, incr, NULL /* NULL BODY DANGER! */, NULL); 
    
    return new CG_chillRepr(loop); 
    
    /*    
    //vector<chillAST_node*> indexnodes = static_cast<CG_chillRepr*>(index)->getChillCode(); 
    chillAST_DeclRefExpr *index_decl
    Expr *lower_bound; //                 = static_cast<CG_chillRepr*>(lower)->getChillCode();
    Expr *upper_bound; //               = static_cast<CG_chillRepr*>(upper)->getChillCode();
    Expr *step_size  ; //                = static_cast<CG_chillRepr*>(step)->getChillCode();
    
    fprintf(stderr, "gonna die in CG_chillBuilder ~line 459\n");
    
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
    fprintf(stderr, "CG_chillBuilder::CreatePragmaAttribute()   TODO\n");
    exit(-1); 
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
    return stmt;
  }
  
  //-----------------------------------------------------------------------------
  // Prefetch Attribute
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreatePrefetchAttribute(CG_outputRepr* stmt, int looplevel, const std::string &arrName, int hint) const {
    fprintf(stderr, "CG_chillBuilder::CreatePrefetchAttribute()   TODO\n");
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
    //fprintf(stderr, "CG_chillBuilder::CreateLoop( indent %d)\n", indent); 
    
    if (stmtList == NULL) {
      delete control;
      return NULL;
    }
    else if (control == NULL) {
      fprintf(stderr, "Code generation: no inductive for this loop\n");
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
    fprintf(stderr, "CG_chillBuilder::CreateInt( %d )\n",i); 
    chillAST_IntegerLiteral *il = new chillAST_IntegerLiteral(i, NULL); // parent not available
    return new CG_chillRepr(il);
  }
  CG_outputRepr* CG_chillBuilder::CreateFloat(float f) const {
    //fprintf(stderr, "CG_chillBuilder::CreateFloat( %f )\n", f); 
    chillAST_FloatingLiteral *fl = new chillAST_FloatingLiteral(f, NULL); // parent not available
    return new CG_chillRepr(fl);
  }
  CG_outputRepr* CG_chillBuilder::CreateDouble(double d) const {
    //fprintf(stderr, "CG_chillBuilder::CreateInt( %f )\n",d); 
    chillAST_FloatingLiteral *dl = new chillAST_FloatingLiteral(d, NULL); // parent not available
    return new CG_chillRepr(dl);
  }
  
  
  //----------------------------------------------------------------------------------------
  bool CG_chillBuilder::isInteger(CG_outputRepr *op) const{
    CG_chillRepr *cr = (CG_chillRepr *)op;
    return cr->chillnodes[0]->isIntegerLiteral(); 
  }
  
  
  //----------------------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateIdent(const std::string &_s) const {
    fprintf(stderr, "CG_chillBuilder::CreateIdent( %s )\n", _s.c_str()); 
    
    bool already_parameter = symbolTableHasVariableNamed(symtab_,  _s.c_str());
    bool already_internal  = symbolTableHasVariableNamed(symtab2_, _s.c_str());
    if ( already_parameter ) { 
      fprintf(stderr, "%s was already a parameter??\n",  _s.c_str()); 
    } 
    if ( already_internal ) { 
      //fprintf(stderr, "%s was already defined in the function body\n",  _s.c_str()); 
      //printSymbolTable(symtab2_); printf("dammit\n"); fflush(stdout); 
    } 

    if ( (!already_parameter) && (! already_internal)) {  
      fprintf(stderr, "CG_roseBuilder.cc L919 adding symbol %s to symtab2_ because it was not already there\n", _s.c_str()); 
      
      //fprintf(stderr, "parameters were: %p\n", symtab_); 
      //printSymbolTable( symtab_ ); 
      //fprintf(stderr, "\nbody symbols were: %p\n", symtab2_); 
      //printSymbolTable( symtab2_ ); 
      //fprintf(stderr, "\n\n"); 
      //fprintf(stderr, "there were  already %d entries in body\n", symtab2_->size()); 

      // this is copying roseBuilder, but is probably wrong. it is assuming 
      // that the ident is a direct child of the current function 
      
      chillAST_VarDecl *vd = new chillAST_VarDecl( "int", _s.c_str(), "", currentfunction->getBody()); // parent not available  TODO 
      currentfunction->addVariableToSymbolTable( vd ); // use symtab2_  ?? 
    
      
      chillAST_DeclRefExpr *dre = new chillAST_DeclRefExpr( "int", _s.c_str(), (chillAST_node*)vd, NULL ); // parent not available
      //fprintf(stderr, "made a new chillRepr from "); dre->dump(); fflush(stdout);
      return new CG_chillRepr( dre );
    }


    // variable was already defined as either a parameter or internal variable to the function.

    // NOW WHAT??  gotta return something
    chillAST_VarDecl *vd = currentfunction->funcHasVariableNamed( _s.c_str() );
    //fprintf(stderr, "vd %p\n", vd); 

    chillAST_DeclRefExpr *dre = new chillAST_DeclRefExpr( "int", _s.c_str(), (chillAST_node*)vd, NULL ); // parent not available
    return new CG_chillRepr( dre );
  }
  



  
  //-----------------------------------------------------------------------------
  // binary arithmetic operations
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreatePlus(CG_outputRepr *lop,
                                             CG_outputRepr *rop) const {
    fprintf(stderr, "CG_chillBuilder::CreatePlus()\n"); 
    
    
    if(rop == NULL) return lop;     // ?? 
    else if(lop == NULL) return rop;
    
    chillAST_node *left  = ((CG_chillRepr*)lop)->chillnodes[0]; 
    chillAST_node *right = ((CG_chillRepr*)rop)->chillnodes[0]; 
    chillAST_BinaryOperator *bop = new chillAST_BinaryOperator( left, "+", right, NULL ); // parent not available
    return new CG_chillRepr( bop );
    /*
      Expr *lhs = static_cast<CG_chillRepr*>(lop)->GetExpression();
      Expr *rhs = static_cast<CG_chillRepr*>(rop)->GetExpression();
      
      // Not sure about type!!
      fprintf(stderr, "about to die in CG_chillBuilder ~line 628    CREATE PLUS\n"); 
      BinaryOperator *ins = new (astContext_)BinaryOperator(lhs,
      rhs, 
      BO_Add, 
      lhs->getType(), // qualifyier type 
      VK_LValue, //Expression Value Kind, following the C++11 scheme
      OK_Ordinary, // expression object kind, A further classification of the kind of object referenced by an l-value or x-value. 
      SourceLocation(),
      false );  // fpContractable  ?? 
      
      delete lop; delete rop;
      
      //fprintf(stderr, "                                                                               NEW binary operator 0x%x\n", ins);
      fprintf(stderr, "CG_chillBuilder::CreatePlus  ins 0x%x\n", ins); 
      return new CG_chillRepr(ins);
    */
  }
  
  //-----------------------------------------------------------------------------  
  CG_outputRepr* CG_chillBuilder::CreateMinus(CG_outputRepr *lop,
                                              CG_outputRepr *rop) const {
    //fprintf(stderr, "CG_chillBuilder::CreateMinus( lop %p   rop %p)\n", lop, rop); 
    fprintf(stderr, "CG_chillBuilder::CreateMinus()\n");
    
    if(rop == NULL) {
      fprintf(stderr, "CG_chillBuilder::CreateMinus(), right side is NULL\n"); 
      return lop; // from protonu's version. 

      int *i = 0;
      int j = i[0]; // segfault 
    }
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    if(clop == NULL) {  // this is really a unary operator ??? 
      //fprintf(stderr, "CG_chillBuilder::CreateMinus()  unary\n");
      chillAST_node *rAST = crop->chillnodes[0]; // always just one?
      chillAST_UnaryOperator *ins = new chillAST_UnaryOperator("-", true, rAST->clone(), NULL); // clone?
      delete crop;  // ?? note: the chillRepr, not the chillAST_node 
      return new CG_chillRepr(ins);
    } else {
      //fprintf(stderr, "binary\n");
      chillAST_node *lAST = clop->chillnodes[0]; // always just one?
      chillAST_node *rAST = crop->chillnodes[0]; // always just one?
      //lAST->print(); printf(" - ");
      //rAST->print(); printf("\n"); fflush(stdout); 
      
      chillAST_BinaryOperator *bop = new chillAST_BinaryOperator(lAST->clone(), "-", rAST->clone(), NULL); // clone??
      
      delete clop; delete crop; // ?? note: the chillReprs, not the chillAST_nodes
      return new CG_chillRepr(bop);
    }
  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateTimes(CG_outputRepr *lop,
                                              CG_outputRepr *rop) const {
    fprintf(stderr, "CG_chillBuilder::CreateTimes()\n"); 
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
    
    fprintf(stderr, "building "); 
    lAST->print(0, stderr); 
    fprintf(stderr, " * ");
    rAST->print(0, stderr);
    fprintf(stderr, "\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "*", rAST, NULL);
    delete lop; delete rop; // ?? 
    //fprintf(stderr, "CG_chillBuilder::CreateTimes() returning a CG_chillRepr with a binop inside\n");
    return new CG_chillRepr( binop );
  }
  
  
  
  //-----------------------------------------------------------------------------
  //  CG_outputRepr *CG_chillBuilder::CreateDivide(CG_outputRepr *lop, CG_outputRepr *rop) const {
  //    return CreateIntegerFloor(lop, rop);
  //  }
  
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateIntegerDivide(CG_outputRepr *lop,
                                                      CG_outputRepr *rop) const {
    //fprintf(stderr, "CG_chillBuilder::CreatIntegerDivide()\n"); 
    if (rop == NULL) {
      fprintf(stderr, "Code generation: divide by NULL\n");
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
    
    //fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //fprintf(stderr, " / ");
    //rAST->print(0, stderr);
    //fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "/", rAST, NULL);
    delete lop; delete rop; // ?? 
    return new CG_chillRepr( binop );
  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateIntegerFloor(CG_outputRepr* lop, CG_outputRepr* rop) const { 
    //fprintf(stderr, "CG_chillBuilder::CreateIntegerFloor()\n");
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //fprintf(stderr, " / ");
    //rAST->print(0, stderr);
    //fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "/", rAST, NULL);
    return new CG_chillRepr( binop );
  }
  
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateIntegerMod(CG_outputRepr *lop,
                                                   CG_outputRepr *rop) const {
    //fprintf(stderr, "CG_chillBuilder::CreateIntegerMod()   NEEDS WORK\n"); 
    //fprintf(stderr, "LHS "); lop->dump(); 
    //fprintf(stderr, "RHS "); rop->dump(); 
    
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
       fprintf(stderr, "gonna die in CG_chillBuilder.cc ~line 394\n"); 
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
    fprintf(stderr, "CG_chillBuilder::CreateAnd()\n");  
    if (rop == NULL)
      return lop;
    else if (lop == NULL)
      return rop;
    
    /* if (rop == NULL || lop == NULL ) {
       fprintf(stderr, "returning NULL!\n"); 
       return NULL;
       }*/
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //fprintf(stderr, " && ");
    //rAST->print(0, stderr);
    //fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "&&", rAST, NULL);
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
  //    fprintf(stderr, "about to die in CG_chillBuilder ~line 480\n"); 
  
  //    BinaryOperator *ins = NULL; // new (astContext_)BinaryOperator(op1, op2, BO_GE, op1->getType(), SourceLocation());
  
  //    delete lop; delete rop;
  //    return new CG_chillRepr(ins);
  //  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateLE(CG_outputRepr *lop,
                                           CG_outputRepr *rop) const {
    //fprintf(stderr, "CG_chillBuilder::CreateLE()\n");  
    if (rop == NULL || lop == NULL) {
      return NULL;           
    }            
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //fprintf(stderr, " <= ");
    //rAST->print(0, stderr);
    //fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "<=", rAST, NULL);
    delete lop; delete rop; // ?? 
    return new CG_chillRepr( binop );
  }
  
  
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateEQ(CG_outputRepr *lop,
                                           CG_outputRepr *rop) const {
    //fprintf(stderr, "CG_chillBuilder::CreateEQ()\n");  
    if (rop == NULL || lop == NULL) {
      return NULL;           
    }            
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //fprintf(stderr, " = ");
    //rAST->print(0, stderr);
    //fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "==", rAST, NULL);
    delete lop; delete rop; // ?? 
    return new CG_chillRepr( binop );
  }
  
  
  
  
  CG_outputRepr* CG_chillBuilder::CreateNEQ(CG_outputRepr *lop,
                                            CG_outputRepr *rop) const {
    //fprintf(stderr, "CG_chillBuilder::CreateNEQ()\n");  
    if (rop == NULL || lop == NULL) {
      return NULL;           
    }            
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    
    //fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //fprintf(stderr, " != ");
    //rAST->print(0, stderr);
    //fprintf(stderr, "  ??\n"); 
    
    chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, "!=", rAST, NULL);
    delete lop; delete rop; // ?? 
    return new CG_chillRepr( binop );
  }
  
  
  CG_outputRepr* CG_chillBuilder::CreateDotExpression(CG_outputRepr *lop,
                                                      CG_outputRepr *rop) const {
    //fprintf(stderr, "\nCG_chillBuilder::CreateDotExpression()\n");  
    if (rop == NULL || lop == NULL) {
      return NULL;           
    }            
    
    CG_chillRepr *clop = (CG_chillRepr *) lop;
    CG_chillRepr *crop = (CG_chillRepr *) rop;
    
    chillAST_node *lAST = clop->chillnodes[0]; // always just one?
    chillAST_node *rAST = crop->chillnodes[0]; // always just one?
    //fprintf(stderr, "left is %s,  right is %s\n", lAST->getTypeString(), rAST->getTypeString()); 
    
    if ( !rAST->isVarDecl()) { 
      fprintf(stderr, "CG_chillBuilder::CreateDotExpression() right is a %s, not a vardecl\n",
              rAST->getTypeString());
      exit(-1); 
    }
    chillAST_VarDecl *rvd = (chillAST_VarDecl *)rAST;
    //fprintf(stderr, "building "); 
    //lAST->print(0, stderr); 
    //fprintf(stderr, ".");
    //rAST->print(0, stderr);
    //fprintf(stderr, "  ??\n"); 
    
    //chillAST_BinaryOperator *binop = new chillAST_BinaryOperator( lAST, ".", rAST, NULL);
    
    
    // MemberExpr should be a DeclRefExpr on the left?
    chillAST_DeclRefExpr *DRE = NULL;
    if (lAST->isDeclRefExpr()) DRE = (chillAST_DeclRefExpr *)lAST; 
    if (lAST->isVarDecl()) { 
      // make a DeclRefExpr ?  probably an error upstream of here in this case
      DRE = new chillAST_DeclRefExpr( (chillAST_VarDecl *)lAST ); 
    }
    if (!DRE) { 
      fprintf(stderr, "CG_chillBuilder::CreateDotExpression(), can't create base\n");
      exit(-1); 
    }
    chillAST_MemberExpr *memexpr = new chillAST_MemberExpr( DRE, rvd->varname, NULL, NULL, CHILL_MEMBER_EXP_DOT );
    
    
    //delete lop; delete rop; // ??  
    return new CG_chillRepr( memexpr );
  }
  
  
  //-----------------------------------------------------------------------------
  // stmt list gen operations
  //-----------------------------------------------------------------------------
  CG_outputRepr* CG_chillBuilder::CreateStmtList(CG_outputRepr *singleton) const {
    //fprintf(stderr, "CG_chillBuilder::CreateStmtList()\n");  
    if(singleton == NULL) return NULL;
    
    exit(-1);                  // DFL 
    return( NULL ); 
    /* 
       StmtList *tnl = static_cast<CG_chillRepr *>(singleton)->GetCode();
       
       if(tnl->empty()) {
       StmtList foo;
       fprintf(stderr, "gonna die soon  CG_chillBuilder::CreateStmtList()\n");  exit(-1); 
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
    
    //fprintf(stderr, "CG_chillBuilder::StmtListAppend()\n"); 
    
    if(list1 == NULL) return list2;
    else if(list2 == NULL) return list1;
    
    CG_chillRepr *cr1 = (CG_chillRepr *)list1;
    CG_chillRepr *cr2 = (CG_chillRepr *)list2;
    
    int numtoadd = cr2->chillnodes.size();
    //fprintf(stderr, "before: %d nodes and %d nodes\n", cr1->chillnodes.size(), numtoadd ); 
    for (int i=0; i<numtoadd; i++){
      (cr1->chillnodes).push_back(cr2->chillnodes[i] );
    }
    //fprintf(stderr, "after %d nodes\n", cr1->chillnodes.size() ); 
    
    delete list2;
    return list1;
    
  }
  
  
  bool CG_chillBuilder::QueryInspectorType(const std::string &varName) const {
    fprintf(stderr, "CG_chillBuilder::QueryInspectorType( %s )\n", varName.c_str()); 
    int *i=0; int j= i[0]; 
    return false;
  }
  
  
  CG_outputRepr* CG_chillBuilder::CreateArrayRefExpression(const std::string &_s,
                                                           CG_outputRepr *rop) const {
    fprintf(stderr, "CG_chillBuilder::CreateArrayRefExpression()  DIE\n");
    fprintf(stderr, "string s  '%s'\n", _s.c_str());
    rop->dump(); 

    int *i=0; int j = i[0]; 
    exit(-1);
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
    
    if (!base)  {
      fprintf(stderr, "CG_chillBuilder::CreateArrayRefExpression(), left is %s\n", l->getTypeString()); 
      
      exit(-1);
    }
    
    
    
    chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( base, r, NULL, 0); // unique TODO 
    return new CG_chillRepr( ASE ); 
  }
  
  
  CG_outputRepr* CG_chillBuilder::ObtainInspectorData(const std::string &_s, const std::string &member_name) const{
    fprintf(stderr, "CG_chillBuilder::ObtainInspectorData( %s, %s)\n", 
            _s.c_str(), member_name.c_str());
    
    //WTF 

    return ObtainInspectorRange( _s, member_name ); 
  }
  
  
  CG_outputRepr *CG_chillBuilder::CreateAddressOf(CG_outputRepr* op) const {
    fprintf(stderr, "CG_chillBuilder::CreateAddressOf()\n");
    exit(-1);
  }
  
  CG_outputRepr* CG_chillBuilder::CreateBreakStatement() const { 
    fprintf(stderr, "CG_chillBuilder::CreateBreakStatement()\n");
    exit(-1);
  }
  
  
  CG_outputRepr *CG_chillBuilder::CreateStatementFromExpression(CG_outputRepr *exp) const { 
    fprintf(stderr, "CG_chillBuilder::CreateStatementFromExpression()\n");
    exit(-1);
  }
  
  


  CG_outputRepr *CG_chillBuilder::CreateStruct(const std::string struct_name,
                                               std::vector<std::string> data_members,
                                               std::vector<CG_outputRepr *> data_types)
  { 
    
    fprintf(stderr, "\nCG_chillBuilder::CreateStruct( %s )\n", struct_name.c_str()); 
    
/* WRONG - a typedef 
    // NEED TO ADD TYPEDEF TO ... SOMETHING 
    
    chillAST_TypedefDecl *tdd = new chillAST_TypedefDecl( ) ;
    
    tdd->setStructName(struct_name.c_str()); 
    tdd->setStruct( true ); 
    int n_memb = data_members.size();
    int n_data_types = data_types.size();
    for (int i=0; i<n_memb; i++) { 
      chillAST_VarDecl *vd;
      fprintf(stderr, "member %s type ", data_members[i].c_str()); 
      if (i <n_data_types) {
        vd = (chillAST_VarDecl *) ((CG_chillRepr *)data_types[i])->GetCode(); 
        vd->varname = strdup(  data_members[i].c_str() ); 
        bool simplepointer = (vd->numdimensions == 1 && !vd->knownArraySizes);
        if (simplepointer) fprintf(stderr, "pointer to "); 
        fprintf(stderr, "%s\n", vd->vartype );
        if (vd->numdimensions > 0 && vd->knownArraySizes) {
          for (int k=0; k<vd->numdimensions; k++) fprintf(stderr, "[%d]", vd->arraysizes[k]);
        }
      }
      else { 
        fprintf(stderr, "type int BY DEFAULT (bad idea)\n");
        vd = new chillAST_VarDecl( "int", data_members[i].c_str(), "", NULL);
      }
      // add vd to suparts of the struct typedef 
      tdd->subparts.push_back( vd ); 
      
      fprintf(stderr, "\n"); 
    }
    
    // put the typedef in the top level ... for now   TODO 
    toplevel->insertChild( 0, tdd); 
    return new CG_chillRepr( tdd ); 
*/


    chillAST_RecordDecl *rd = new chillAST_RecordDecl(struct_name.c_str(), toplevel);
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
      //fprintf(stderr, "%d member %s type ", i, data_members[i].c_str()); 
      if (i < n_data_types) { 
        // this should always happen, formerly, if no data type was 
        // specified, it was an int. bad idea
        vd = (chillAST_VarDecl *) ((CG_chillRepr *)data_types[i])->GetCode(); 

        // vd did not have a name before 
        vd->varname = strdup(  data_members[i].c_str() ); 

        vd->parent = rd;  // ??

        bool simplepointer = (vd->numdimensions == 1 && !vd->knownArraySizes);
        if (simplepointer) {  
          fprintf(stderr, "struct member %s is pointer to %s\n", vd->varname, vd->vartype);
          vd->arraypointerpart = strdup("*"); // ?? 
        }
        else { 
          //fprintf(stderr, "struct member %s is not a pointer TODO!\n", vd->varname); 
          fprintf(stderr, "struct member %s is %s\n", vd->varname, vd->vartype); 
          
          // it should be good to go ??? 
        }
        //vd->print(); printf("\n"); fflush(stdout); 
        //fprintf(stderr, "%s\n", vd->vartype );
        //if (vd->numdimensions > 0 && vd->knownArraySizes) {
        //  for (int k=0; k<vd->numdimensions; k++) fprintf(stderr, "[%d]", vd->arraysizes[k]);
        //} 
      }
      else { 
        fprintf(stderr, "int BY DEFAULT (bad idea) FIXME\n"); // TODO 
        vd = new chillAST_VarDecl( "int", data_members[i].c_str(), "", NULL);
      }
      rd->addSubpart( vd );
      //fprintf(stderr, "\n"); 
    }
    fprintf(stderr, "\n"); 
    return new CG_chillRepr( rd ); 
  }
  
  
  
  CG_outputRepr *CG_chillBuilder::CreateClassInstance(std::string name ,  // TODO can't make array
                                                      CG_outputRepr *class_def){
    fprintf(stderr, "CG_chillBuilder::CreateClassInstance( %s )\n", name.c_str()); 
    
    CG_chillRepr *CD = (CG_chillRepr *)class_def; 
    chillAST_node *n = CD->GetCode();
    //fprintf(stderr, "class def is of type %s\n", n->getTypeString());
    //n->print(); printf("\n"); fflush(stdout); 

    if (n->isTypeDefDecl()) { 
      chillAST_TypedefDecl *tdd = (chillAST_TypedefDecl *)n;
      //tdd->print(); printf("\n"); fflush(stdout);
      
      chillAST_VarDecl *vd = new chillAST_VarDecl( tdd, name.c_str(), "", NULL); 
      
      // we need to add this to function ??  TODO 
      //fprintf(stderr, "adding typedef instance to symbolTable\n");
      chillAST_SymbolTable *st =  currentfunction->getBody()->getSymbolTable();
      //printSymbolTable(st); 

      currentfunction->getBody()->addVariableToSymbolTable( vd ); // TODO 
      currentfunction->getBody()->insertChild(0, vd);  // TODO 
      //printSymbolTable(st); 
      
      return new CG_chillRepr( vd ); 
    }
    if  (n->isRecordDecl()) { 
      fprintf(stderr, "a RecordDecl\n"); 

      chillAST_RecordDecl *rd = (chillAST_RecordDecl *) n;
      rd->print(); printf("\n"); fflush(stdout);
      rd->dump(); printf("\n");  fflush(stdout);
      
      chillAST_VarDecl *vd = new chillAST_VarDecl( rd, name.c_str(), "", NULL);

      //fprintf(stderr, "CG_chillBuilder.cc, adding struct instance to body of function's symbolTable\n");


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

    fprintf(stderr, "ERROR: CG_chillBuilder::CreateClassInstance() not sent a class or struct\n"); 
    int *i=0; int j = i[0]; 
    return NULL; 
  }
  
  
  
  CG_outputRepr *CG_chillBuilder::lookup_member_data(CG_outputRepr* classtype, 
                                                     std::string varName, 
                                                     CG_outputRepr *instance) {
    
    
    //fprintf(stderr, "CG_chillBuilder::lookup_member_data( %s )\n", varName.c_str()); 
    
    chillAST_VarDecl* sub = NULL;

    CG_chillRepr *CR = (CG_chillRepr *)classtype;
    chillAST_node *classnode = CR->GetCode();
    //fprintf(stderr, "classnode is %s\n", classnode->getTypeString()); classnode->print(); printf("\n"); fflush(stdout); 
    if (! ( classnode->isTypeDefDecl() || 
            classnode->isRecordDecl() )) { 
      fprintf(stderr, "ERROR: CG_chillBuilder::lookup_member_data(), classnode is not a TypeDefDecl or a RecordDecl\n"); 
      exit(-1); 
    }


    CG_chillRepr *CI = (CG_chillRepr *)instance; 

    chillAST_node *in = CI->GetCode();
    //fprintf(stderr, "instance is %s\n", in->getTypeString()); 
    //in->print(); printf("\n"); fflush(stdout); 
    
    if ( !in->isVarDecl() ) { // error, instance needs to be a vardecl
      fprintf(stderr, "ERROR: CG_chillBuilder::lookup_member_data() instance needs to be a VarDecl, not a %s", in->getTypeString());
      exit(-1);
    }
    chillAST_VarDecl *vd = (chillAST_VarDecl *)in;
    if (vd->typedefinition != classnode && 
      vd->vardef != classnode) { 
      fprintf(stderr, "vd: typedef %p  vardev %p    classnode %p\n", vd->typedefinition, vd->vardef, classnode); 
      fprintf(stderr, "CG_chillBuilder::lookup_member_data(), instance is not of correct class \n");
      
      exit(-1);
    }
    
    

    if (classnode->isTypeDefDecl()){ 
      chillAST_TypedefDecl *tdd = (chillAST_TypedefDecl *)classnode;
      if ( !tdd->isAStruct() ) { 
        fprintf(stderr, "ERROR: CG_chillBuilder::lookup_member_data() instance must be a struct or class\n");
        exit(-1);
      }
      
      sub = tdd->findSubpart( varName.c_str() ); 
    }

    if (classnode->isRecordDecl()){ 
      chillAST_RecordDecl *rd = (chillAST_RecordDecl *)classnode;
      if ( !rd->isAStruct() ) { 
        fprintf(stderr, "ERROR: CG_chillBuilder::lookup_member_data() instance must be a struct or class\n");
        exit(-1);
      }
      
      //fprintf(stderr, "looking for member (subpart) %s in RecordDecl\n",  varName.c_str()); 
      sub = rd->findSubpart( varName.c_str() ); 
    }   

    if (!sub) {
      fprintf(stderr, "CG_chillBuilder::lookup_member_data(), variable %s is not submember of class/struct\n"); 
      exit(-1);
    }
    
    //fprintf(stderr, "subpart (member) %s is\n", varName.c_str()); sub->print(); printf("\n"); fflush(stdout);

    return( new CG_chillRepr( sub ) ); // the vardecl inside the struct typedef 
  }
  
  
  CG_outputRepr* CG_chillBuilder::CreatePointer(std::string  &name) const { 
    //fprintf(stderr, "CG_chillBuilder::CreatePointer( %s )\n", name.c_str()); 
    
    chillAST_VarDecl *vd = new chillAST_VarDecl( "int", name.c_str(), "*", currentfunction->getBody());
    //vd->print(); printf("\n"); fflush(stdout); 
    //vd->dump(); printf("\n"); fflush(stdout); 
    
    //printSymbolTable( currentfunction->getBody()->getSymbolTable() ); 

    chillAST_DeclRefExpr *dre = new chillAST_DeclRefExpr( vd ); // ?? 
    return new CG_chillRepr( dre );  // need a declrefexpr? 
  }
  

  CG_outputRepr* CG_chillBuilder::ObtainInspectorRange(const std::string &structname, const std::string &member) const {
    //fprintf(stderr, "CG_chillBuilder::ObtainInspectorRange(%s,  %s )\n", structname.c_str(), member.c_str()); 
    
    // find a struct/class with name structname and member member
    // return a Member access (or binary dot op )
    // seems like you need to know where (scoping) to look for the struct definition
    
    vector<chillAST_VarDecl*> decls;
    currentfunction->gatherVarDecls( decls );
    //fprintf(stderr, "\nfunc has %d vardecls  (looking for %s)\n", decls.size(), structname.c_str()); 
    
    chillAST_VarDecl *thestructvd = NULL;
    for (int i=0; i<decls.size(); i++) { 
      chillAST_VarDecl *vd = decls[i];
      //vd->print(); printf("\n"); fflush(stdout); 
      
      if (structname == vd->varname) { 
        //fprintf(stderr, "found it!\n"); 
        thestructvd = vd;
        break;
      }
    }
    
    if (!thestructvd) { 
      fprintf(stderr, "CG_chillBuilder::ObtainInspectorRange could not find variable named %s in current function\n", structname.c_str()); 
      exit(-1); 
    }
    
    // make sure the variable is a struct with a member with the correct name
    chillAST_RecordDecl *rd = thestructvd->getStructDef(); 
    if ( !rd ) { 
      fprintf(stderr, "CG_chillBuilder::ObtainInspectorRange(), variable %s is not a struct/class\n",  structname.c_str()); 
      exit(-1);
    }
    
    
    chillAST_VarDecl *sub = rd->findSubpart( member.c_str() ); 
    if (!sub) { 
      fprintf(stderr, "CG_chillBuilder::ObtainInspectorRange(), struct/class %s has no member named %s\n",  structname.c_str(), member.c_str()); 
      exit(-1); 
    }
    
    
    // build up a member expression  (or a binop with dot operation?? )
    // make a declrefexpr that refers to this variable definition
    chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( thestructvd ); 
    chillAST_MemberExpr *ME = new chillAST_MemberExpr( DRE, member.c_str(), NULL, NULL ); // uniq TODO 
    
    return new CG_chillRepr( ME ); 
  }
  
} // namespace


