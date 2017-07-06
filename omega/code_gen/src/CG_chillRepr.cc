/*****************************************************************************
 Copyright (C) 2008 University of Southern California. 
 All Rights Reserved.

 Purpose:
   omega holder for chill AST  implementaion

 Notes:

 History:
   02/01/06 - Chun Chen - created
   LLVM/CLANG interface created by Saurav Muralidharan
*****************************************************************************/

#include <code_gen/CG_chillRepr.h>
#include <stdio.h>
#include <stdlib.h>  // for exit()

namespace omega { 

  CG_chillRepr::~CG_chillRepr() {
  }
  
  //void CG_chillRepr::AppendStmt(Stmt *s) const {
  //      tree_node_list_->push_back(s);
  //}
  
  //void CG_chillRepr::AppendV(StmtList *sl) const {
  //      for(int i=0; i<sl->size(); ++i) tree_node_list_->push_back((*sl)[i]);
  //}
  

  chillAST_node * CG_chillRepr::GetCode() { 
    //debug_fprintf(stderr, "CG_chillRepr::GetCode() this %p   size %d\n", this, chillnodes.size()); 

    if (0 == chillnodes.size()) return NULL; // error?
        
    if (1 == chillnodes.size()) return chillnodes[0];

    // make a compoundstatement with all the code ???  probably should be that way already
    debug_fprintf(stderr, "CG_chillRepr GetCode(), multiple (%d) statements in the code??\n", chillnodes.size());
    for (int i=0; i<chillnodes.size(); i++) {
      debug_fprintf(stderr, "chillnode %d  %p\n", i, chillnodes[i] );
    }


    chillAST_CompoundStmt *CS = new chillAST_CompoundStmt();
    for (int i=0; i<chillnodes.size(); i++) {
      CS->addChild( chillnodes[i] );
    }
    return CS; 
  }
  


  CG_outputRepr* CG_chillRepr::clone() const {  // make a deep/shallow  COPY of all the nodes
    CG_chillRepr *newrepr = new  CG_chillRepr(); // empty
    
    for (int i=0; i<chillnodes.size(); i++) newrepr->addStatement( chillnodes[i]->clone() );

    // shallow (the submembers are the same )
    //for (int i=0; i<chillnodes.size(); i++) newrepr->addStatement( chillnodes[i] );
    //debug_fprintf(stderr, "done cloning\n"); 
    return newrepr; 
  }


  
  void CG_chillRepr::clear() {
    chillnodes.clear();
  }
  
  
  // TODO this is duplicated and shouldn't be here anyway 
  static std::string binops[] = {
    " ", " ",             // BO_PtrMemD, BO_PtrMemI,       // [C++ 5.5] Pointer-to-member operators.
  "*", "/", "%",        // BO_Mul, BO_Div, BO_Rem,       // [C99 6.5.5] Multiplicative operators.
  "+", "-",             // BO_Add, BO_Sub,               // [C99 6.5.6] Additive operators.
  "<<", ">>",           // BO_Shl, BO_Shr,               // [C99 6.5.7] Bitwise shift operators.
  "<", ">", "<=", ">=", // BO_LT, BO_GT, BO_LE, BO_GE,   // [C99 6.5.8] Relational operators.
  "==", "!=",           // BO_EQ, BO_NE,                 // [C99 6.5.9] Equality operators.
  "&",                  // BO_And,                       // [C99 6.5.10] Bitwise AND operator.
  "??",                 // BO_Xor,                       // [C99 6.5.11] Bitwise XOR operator.
  "|",                  // BO_Or,                        // [C99 6.5.12] Bitwise OR operator.
  "&&",                 // BO_LAnd,                      // [C99 6.5.13] Logical AND operator.
  "||",                 // BO_LOr,                       // [C99 6.5.14] Logical OR operator.
  "=", "*=",            // BO_Assign, BO_MulAssign,      // [C99 6.5.16] Assignment operators.
  "/=", "%=",           // BO_DivAssign, BO_RemAssign,
  "+=", "-=",           // BO_AddAssign, BO_SubAssign,
  "???", "???",         // BO_ShlAssign, BO_ShrAssign,
  "&&=", "???",         // BO_AndAssign, BO_XorAssign,
  "||=",                // BO_OrAssign,
  ","};                 // BO_Comma                      // [C99 6.5.17] Comma operator.


  static std::string unops[] = {
  "++", "--",           // [C99 6.5.2.4] Postfix increment and decrement
  "++", "--",           // [C99 6.5.3.1] Prefix increment and decrement
  "@",  "*",            // [C99 6.5.3.2] Address and indirection
  "+", "-",             // [C99 6.5.3.3] Unary arithmetic
  "~", "!",             // [C99 6.5.3.3] Unary arithmetic
  "__real", "__imag",   // "__real expr"/"__imag expr" Extension.
  "__extension"          // __extension__ marker.
  };



  //void CG_chillRepr::dump() const { Dump(); }


  void CG_chillRepr::Dump() const {
    CG_chillRepr *me = (CG_chillRepr *)this;  // ?? 
    //debug_fprintf(stderr, "repr of type ");
    //debug_fprintf(stderr, "%s\n", this->type()); 
    int numnodes = me->chillnodes.size();
    //debug_fprintf(stderr, "repr %p  %d nodes\n", this, numnodes); 
    for (int i=0; i<numnodes; i++) {
      me->chillnodes[i]->print();  printf("\n"); fflush(stdout);
    }
    return; 
  }  
  
} // namespace
