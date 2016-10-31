

#ifndef _CHILL_AST_H_
#define _CHILL_AST_H_


#define CHILL_INDENT_AMOUNT 2

#include "chill_io.hh"

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>  // std::vector 

#include <ir_enums.hh> // for IR_CONDITION_* 

using std::vector;
using std::string;

char *parseUnderlyingType( char *sometype );
char *parseArrayParts( char *sometype );
bool isRestrict( const char *sometype );
char *splitTypeInfo( char *underlyingtype );
char *ulhack(  char *brackets ); // change "1024UL" to "1024" 
char *restricthack( char *typeinfo ); // remove __restrict__ , MODIFIES the argument!


enum CHILL_ASTNODE_TYPE {
  CHILLAST_NODETYPE_UNKNOWN=0,
  CHILLAST_NODETYPE_SOURCEFILE,
  CHILLAST_NODETYPE_TYPEDEFDECL,
  CHILLAST_NODETYPE_VARDECL,
  //  CHILLAST_NODETYPE_PARMVARDECL,   not used any more 
  CHILLAST_NODETYPE_FUNCTIONDECL,
  CHILLAST_NODETYPE_RECORDDECL,     // struct or union (or class) 
  CHILLAST_NODETYPE_MACRODEFINITION,
  CHILLAST_NODETYPE_COMPOUNDSTMT,
  CHILLAST_NODETYPE_LOOP,               // AKA ForStmt
  CHILLAST_NODETYPE_TERNARYOPERATOR,
  CHILLAST_NODETYPE_BINARYOPERATOR,
  CHILLAST_NODETYPE_UNARYOPERATOR,
  CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR,
  CHILLAST_NODETYPE_MEMBEREXPR,          // structs/unions
  CHILLAST_NODETYPE_DECLREFEXPR,
  CHILLAST_NODETYPE_INTEGERLITERAL,
  CHILLAST_NODETYPE_FLOATINGLITERAL,
  CHILLAST_NODETYPE_IMPLICITCASTEXPR,
  CHILLAST_NODETYPE_RETURNSTMT,
  CHILLAST_NODETYPE_CALLEXPR,
  CHILLAST_NODETYPE_DECLSTMT,
  CHILLAST_NODETYPE_PARENEXPR,
  CHILLAST_NODETYPE_CSTYLECASTEXPR,
  CHILLAST_NODETYPE_CSTYLEADDRESSOF,
  CHILLAST_NODETYPE_IFSTMT,
  CHILLAST_NODETYPE_SIZEOF,
  CHILLAST_NODETYPE_MALLOC,
  CHILLAST_NODETYPE_FREE,
  CHILLAST_NODETYPE_PREPROCESSING, // comments, #define, #include, whatever else works
  CHILLAST_NODETYPE_NOOP,   // NO OP
  // CUDA specific
  CHILLAST_NODETYPE_CUDAMALLOC,
  CHILLAST_NODETYPE_CUDAFREE,
  CHILLAST_NODETYPE_CUDAMEMCPY,
  CHILLAST_NODETYPE_CUDAKERNELCALL,
  CHILLAST_NODETYPE_CUDASYNCTHREADS,
  CHILLAST_NODETYPE_NULL    // explicit non-statement 
  // TODO 
  
} ;

#define CHILLAST_NODETYPE_FORSTMT CHILLAST_NODETYPE_LOOP
#define CHILLAST_NODETYPE_TRANSLATIONUNIT CHILLAST_NODETYPE_SOURCEFILE

enum CHILL_FUNCTION_TYPE { 
  CHILL_FUNCTION_CPU = 0,
  CHILL_FUNCTION_GPU
};
  
enum CHILL_MEMBER_EXP_TYPE { 
  CHILL_MEMBER_EXP_DOT = 0,
  CHILL_MEMBER_EXP_ARROW
};
  
enum CHILL_PREPROCESSING_TYPE { 
  CHILL_PREPROCESSING_TYPEUNKNOWN = 0,
  CHILL_PREPROCESSING_COMMENT,
  CHILL_PREPROCESSING_POUNDDEFINE,
  CHILL_PREPROCESSING_POUNDINCLUDE,
  CHILL_PREPROCESSING_PRAGMA  // unused so far
}; 

enum CHILL_PREPROCESSING_POSITION { // when tied to another statement
  CHILL_PREPROCESSING_POSITIONUNKNOWN = 0,
  CHILL_PREPROCESSING_LINEBEFORE,       // previous line 
  CHILL_PREPROCESSING_LINEAFTER,        // next line 
  CHILL_PREPROCESSING_TOTHERIGHT,       // for this kind of comment, on same line
  CHILL_PREPROCESSING_IMMEDIATELYBEFORE // on same line 
}; 





extern const char* Chill_AST_Node_Names[];  // WARNING MUST BE KEPT IN SYNC WITH BELOW LIST 


// fwd declarations
class chillAST_node;         // the generic node. specific types derive from this
class chillAST_NULL;         // empty 
class chillAST_SourceFile;  // ast for an entire source file (translationunit)

class chillAST_TypedefDecl; 
class chillAST_VarDecl; 
//class chillAST_ParmVarDecl; 
class chillAST_FunctionDecl; 
class chillAST_RecordDecl;       // structs and unions (and classes?)
class chillAST_MacroDefinition;
class chillAST_CompoundStmt;  // just a bunch of other statements 
class chillAST_ForStmt;    // AKA a LOOP
class chillAST_TernaryOperator; 
class chillAST_BinaryOperator; 
class chillAST_ArraySubscriptExpr;
class chillAST_MemberExpr;
class chillAST_DeclRefExpr; 
class chillAST_IntegerLiteral;
class chillAST_FloatingLiteral;
class chillAST_UnaryOperator;
class chillAST_ImplicitCastExpr;
class chillAST_CStyleCastExpr;
class chillAST_CStyleAddressOf;
class chillAST_ReturnStmt; 
class chillAST_CallExpr; 
class chillAST_ParenExpr; 
class chillAST_Sizeof; 
class chillAST_Malloc;
class chillAST_Free; 
class chillAST_NoOp; 
class chillAST_CudaMalloc; 
class chillAST_CudaFree; 
class chillAST_CudaMemcpy; 
class chillAST_CudaKernelCall; 
class chillAST_CudaSyncthreads; 
class chillAST_Preprocessing;

typedef std::vector<chillAST_VarDecl *>         chillAST_SymbolTable;   //  typedef
typedef std::vector<chillAST_TypedefDecl *>     chillAST_TypedefTable;  //  typedef

bool symbolTableHasVariableNamed( chillAST_SymbolTable *table, const char *name ); // fwd decl 
chillAST_VarDecl *symbolTableFindVariableNamed( chillAST_SymbolTable *table, const char *name ); // fwd decl TODO too many similar named functions

void printSymbolTable( chillAST_SymbolTable *st ); // fwd decl 
void printSymbolTableMoreInfo( chillAST_SymbolTable *st ); // fwd decl 


chillAST_node           *lessthanmacro( chillAST_node *left,  chillAST_node *right);  // fwd declaration 
chillAST_SymbolTable    *addSymbolToTable( chillAST_SymbolTable *st, chillAST_VarDecl *vd ); // fwd decl
chillAST_TypedefTable   *addTypedefToTable( chillAST_TypedefTable *tt, chillAST_TypedefDecl *td ); // fwd decl


bool streq( const char *a, const char *b); // fwd decl
void chillindent( int i, FILE *fp );  // fwd declaration  
void insertNewDeclAtLocationOfOldIfNeeded( chillAST_VarDecl *newdecl, chillAST_VarDecl *olddecl); 

chillAST_DeclRefExpr *buildDeclRefExpr( chillAST_VarDecl  *); 






// an actual chill ast. 
// nodes based on clang AST which are in turn based on C++ 

class chillAST_node {   // generic node. a tree of these is the AST. this is virtual (can't instantiate)
public: 

  static int chill_scalar_counter;   // for manufactured scalars 
  static int chill_array_counter ;   // for manufactured arrays
  static int chill_pointer_counter ; // for manufactured arrays


  CHILL_ASTNODE_TYPE asttype;
  
  bool isSourceFile()         { return (asttype == CHILLAST_NODETYPE_SOURCEFILE); };
  bool isTypeDefDecl()        { return (asttype == CHILLAST_NODETYPE_TYPEDEFDECL); };
  bool isVarDecl()            { return (asttype == CHILLAST_NODETYPE_VARDECL); }; 
  bool isFunctionDecl()       { return (asttype == CHILLAST_NODETYPE_FUNCTIONDECL); }; 
  bool isRecordDecl()         { return (asttype == CHILLAST_NODETYPE_RECORDDECL); }; 
  bool isMacroDefinition()    { return (asttype == CHILLAST_NODETYPE_MACRODEFINITION); }; 
  bool isCompoundStmt()       { return (asttype == CHILLAST_NODETYPE_COMPOUNDSTMT); }; 
  bool isLoop()               { return (asttype == CHILLAST_NODETYPE_LOOP); };    // AKA ForStmt
  bool isForStmt()            { return (asttype == CHILLAST_NODETYPE_LOOP); };    // AKA Loop
  bool isIfStmt()             { return (asttype == CHILLAST_NODETYPE_IFSTMT); }; 
  bool isTernaryOperator()    { return (asttype == CHILLAST_NODETYPE_TERNARYOPERATOR);}; 
  bool isBinaryOperator()     { return (asttype == CHILLAST_NODETYPE_BINARYOPERATOR); }; 
  bool isUnaryOperator()      { return (asttype == CHILLAST_NODETYPE_UNARYOPERATOR); }; 
  bool isArraySubscriptExpr() { return (asttype == CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR); }; 
  bool isMemberExpr()         { return (asttype == CHILLAST_NODETYPE_MEMBEREXPR); }; 
  bool isDeclRefExpr()        { return (asttype == CHILLAST_NODETYPE_DECLREFEXPR); }; 
  bool isIntegerLiteral()     { return (asttype == CHILLAST_NODETYPE_INTEGERLITERAL); }; 
  bool isFloatingLiteral()    { return (asttype == CHILLAST_NODETYPE_FLOATINGLITERAL); }; 
  bool isImplicitCastExpr()   { return (asttype == CHILLAST_NODETYPE_IMPLICITCASTEXPR); }; 
  bool isReturnStmt()         { return (asttype == CHILLAST_NODETYPE_RETURNSTMT); }; 
  bool isCallExpr()           { return (asttype == CHILLAST_NODETYPE_CALLEXPR); }; 
  bool isParenExpr()          { return (asttype == CHILLAST_NODETYPE_PARENEXPR); }; 
  bool isSizeof()             { return (asttype == CHILLAST_NODETYPE_SIZEOF); }; 
  bool isMalloc()             { return (asttype == CHILLAST_NODETYPE_MALLOC); }; 
  bool isFree()               { return (asttype == CHILLAST_NODETYPE_FREE); }; 
  bool isPreprocessing()      { return (asttype == CHILLAST_NODETYPE_PREPROCESSING); }; 
  bool isNoOp()               { return (asttype == CHILLAST_NODETYPE_NOOP); }; 
  bool isNull()               { return (asttype == CHILLAST_NODETYPE_NULL); }; 
  bool isCStyleCastExpr()     { return (asttype == CHILLAST_NODETYPE_CSTYLECASTEXPR); }; 
  bool isCStyleAddressOf()    { return (asttype == CHILLAST_NODETYPE_CSTYLEADDRESSOF); }; 
  bool isCudaMalloc()         { return (asttype == CHILLAST_NODETYPE_CUDAMALLOC); }; 
  bool isCudaFree()           { return (asttype == CHILLAST_NODETYPE_CUDAFREE); }; 
  bool isCudaMemcpy()         { return (asttype == CHILLAST_NODETYPE_CUDAMEMCPY); }; 
  bool isCudaKERNELCALL()     { return (asttype == CHILLAST_NODETYPE_CUDAKERNELCALL); }; 
  bool isCudaSYNCTHREADS()    { return (asttype == CHILLAST_NODETYPE_CUDASYNCTHREADS); }; 

  bool isDeclStmt()           { return (asttype == CHILLAST_NODETYPE_DECLSTMT); }; // doesn't exist
  
  bool isConstant()           { return (asttype == CHILLAST_NODETYPE_INTEGERLITERAL) || (asttype == CHILLAST_NODETYPE_FLOATINGLITERAL); } 
    

  virtual bool isAssignmentOp() { return false; }; 
  virtual bool isComparisonOp() { return false; }; 
  virtual bool isNotLeaf()      { return false; };
  virtual bool isLeaf()         { return true;  };
  virtual bool isParmVarDecl()  { return false; };  

  virtual bool isPlusOp()       { return false; }; 
  virtual bool isMinusOp()      { return false; }; 
  virtual bool isPlusMinusOp()  { return false; }; 
  virtual bool isMultDivOp()    { return false; }; 

  virtual bool isAStruct() { return false; }; 
  virtual bool isAUnion()  { return false; };

  virtual bool hasSymbolTable() { return false; } ; // most nodes do NOT have a symbol table
  virtual bool hasTypedefTable() { return false; } ; // most nodes do NOT have a typedef table
  virtual chillAST_SymbolTable *getSymbolTable() { return NULL; } // most nodes do NOT have a symbol table

  virtual chillAST_VarDecl *findVariableNamed( const char *name ); // recursive 

  chillAST_RecordDecl *findRecordDeclNamed( const char *name ); // recursive
  
  // void addDecl( chillAST_VarDecl *vd); // recursive, adds to first  symbol table it can find 

  // TODO decide how to hide some data
  chillAST_node *parent; 
  bool isFromSourceFile;  // false = #included 
  char *filename;  // file this node is from

  void segfault() { debug_fprintf(stderr, "segfaulting on purpose\n"); int *i=0; int j = i[0]; }; // seg fault
  int getNumChildren() { return children.size(); }; 
  vector<chillAST_node*> children; 
  vector<chillAST_node*> getChildren() { return children; } ;  // not usually useful
  void                   setChildren( vector<chillAST_node*>&c ) { children = c; } ; // does not set parent. probably should
  chillAST_node *getChild( int which)                    { return children[which]; };
  void           setChild( int which, chillAST_node *n ) { children[which] = n; children[which]->parent = this; } ;
  
  char *metacomment; // for compiler internals, formerly a comment
  void setMetaComment( char *c ) { metacomment = strdup(c); }; 

  vector<chillAST_Preprocessing*> preprocessinginfo; 

  virtual void addChild( chillAST_node* c) {
    //if (c->isFunctionDecl()) debug_fprintf(stderr, "addchild FunctionDecl\n"); 
    c->parent = this;
    // check to see if it's already there
    for (int i=0; i<children.size(); i++) { 
      if (c == children[i]) {
        //debug_fprintf(stderr, "addchild ALREADY THERE\n"); 
        return; // already there
      }
    }
    children.push_back(c);
  } ;  // not usually useful

  virtual void insertChild(int i, chillAST_node* node) { 
    //debug_fprintf(stderr, "%s inserting child of type %s at location %d\n", getTypeString(), node->getTypeString(), i); 
    node->parent = this; 
    children.insert( children.begin()+i, node );
  };
  
  virtual void removeChild(int i) { 
    children.erase( children.begin()+i );
  };
  
  int findChild(  chillAST_node *c )  {   
    for (int i=0; i<children.size(); i++) { 
      if (children[i] == c) return i;
    }
    return -1;
  }

  virtual void replaceChild( chillAST_node *old, chillAST_node *newchild ) { 
    debug_fprintf(stderr,"(%s) forgot to implement replaceChild() ... using generic\n" ,Chill_AST_Node_Names[asttype]); 
    debug_fprintf(stderr, "%d children\n", children.size()); 
    for (int i=0; i<children.size(); i++) { 
      if (children[i] == old) { 
        children[i] = newchild;
        newchild->setParent( this );
        return; 
      }
    }
    debug_fprintf(stderr, "%s %p generic replaceChild called with oldchild that was not a child\n", 
            getTypeString(), this) ;
    debug_fprintf(stderr, "printing\n"); 
    print(); debug_fprintf(stderr, "\nchild: ");
    if (!old) debug_fprintf(stderr, "oldchild NULL!\n");
    old->print(); debug_fprintf(stderr, "\nnew: "); 
    newchild->print(); debug_fprintf(stderr, "\n"); 
    segfault(); // make easier for gdb
  };
  
  virtual void loseLoopWithLoopVar( char *var ) { 
    // walk tree. If a loop has this loop variable, replace the loop with the loop body, 
    // removing the loop.  The loop will be spread across a bunch of cores that will each
    // calculate their own loop variable.

    // things that can not have loops as substatements can have a null version of this method
    // things that have more complicated sets of "children" will have specialized versions

    // this is the generic version of the method. It just recurses among its children.
    // ForStmt is the only one that can actually remove itself. When it does, it will 
    // potentially change the children vector, which is not the simple array it might appear.
    // so you have to make a copy of the vector to traverse
    
    vector<chillAST_node*> dupe = children; // simple enough?
    //debug_fprintf(stderr, "node XXX has %d children\n", dupe.size()); 
    //debug_fprintf(stderr, "generic node %s has %d children\n", getTypeString(), dupe.size()); 
    for (int i=0; i<dupe.size(); i++) {  // recurse on all children
      dupe[i]->loseLoopWithLoopVar( var );
    }
  }

  virtual int evalAsInt() { 
    debug_fprintf(stderr,"(%s) can't be evaluated as an integer??\n", Chill_AST_Node_Names[asttype]);
    print(); debug_fprintf(stderr, "\n"); 
    segfault(); 
  }

  virtual const char* getUnderlyingType() { 
    debug_fprintf(stderr,"(%s) forgot to implement getUnderlyingType()\n", Chill_AST_Node_Names[asttype]);
    dump();
    print();
    debug_fprintf(stderr, "\n\n"); 
    segfault(); 
  }; 

  virtual chillAST_VarDecl* getUnderlyingVarDecl() { 
    debug_fprintf(stderr,"(%s) forgot to implement getUnderlyingVarDecl()\n", Chill_AST_Node_Names[asttype]);
    dump();
    print();
    debug_fprintf(stderr, "\n\n"); 
    segfault();
  }; 


  virtual chillAST_node *findref(){// find the SINGLE constant or data reference at this node or below
    debug_fprintf(stderr,"(%s) forgot to implement findref()\n" ,Chill_AST_Node_Names[asttype]); 
    dump();
    print();
    debug_fprintf(stderr, "\n\n"); 
    segfault();
  };

  virtual void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {
    debug_fprintf(stderr,"(%s) forgot to implement gatherArrayRefs()\n" ,Chill_AST_Node_Names[asttype]); 
    dump();
    print();
    debug_fprintf(stderr, "\n\n"); 
  };
 
  // TODO we MIGHT want the VarDecl // NOTHING IMPLEMENTS THIS? ??? 
  virtual void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
    debug_fprintf(stderr,"(%s) forgot to implement gatherScalarRefs()\n" ,Chill_AST_Node_Names[asttype]); 
    dump();
    print();
    debug_fprintf(stderr, "\n\n"); 
  };
 
  virtual void gatherLoopIndeces( std::vector<chillAST_VarDecl*> &indeces ) { // recursive walk parent links, looking for loops, and grabbing the declRefExpr in the loop init and cond. 
    // you can quit when you get to certain nodes

    //debug_fprintf(stderr, "%s::gatherLoopIndeces()\n", getTypeString()); 
    
    if (isSourceFile() || isFunctionDecl() ) return; // end of the line

    // just for debugging 
    //if (parent) {
    //  debug_fprintf(stderr, "%s has parent of type %s\n", getTypeString(), parent->getTypeString()); 
    //} 
    //else debug_fprintf(stderr, "this %s %p has no parent???\n", getTypeString(), this);


    if (!parent) return; // should not happen, but be careful

    // for most nodes, this just recurses upwards
    //debug_fprintf(stderr, "%s::gatherLoopIndeces() %p recursing up\n", this); 
    parent->gatherLoopIndeces( indeces );
  }


  chillAST_ForStmt* findContainingLoop() { // recursive walk parent links, looking for loops
    //debug_fprintf(stderr, "%s::findContainingLoop()   ", getTypeString()); 
    //if (parent) debug_fprintf(stderr, "parents is a %s\n", parent->getTypeString()); 
    //else debug_fprintf(stderr, "no parent\n"); 
    // do not check SELF type, as we may want to find the loop containing a loop
    if (!parent) return NULL;
    if (parent->isForStmt()) return (chillAST_ForStmt*)parent;
    return parent->findContainingLoop(); // recurse upwards
  }

  chillAST_node* findContainingNonLoop() { // recursive walk parent links, avoiding loops
    debug_fprintf(stderr, "%s::findContainingNonLoop()   ", getTypeString()); 
    //if (parent) debug_fprintf(stderr, "parent is a %s\n", parent->getTypeString()); 
    //else debug_fprintf(stderr, "no parent\n"); 
    // do not check SELF type, as we may want to find the loop containing a loop
    if (!parent) return NULL;
    if (parent->isCompoundStmt() && parent->getParent()->isForStmt()) return parent->getParent()->findContainingNonLoop(); // keep recursing
    if (parent->isForStmt()) return parent->findContainingNonLoop(); // keep recursing
    return (chillAST_node*)parent; // return non-loop 
  }

  // TODO gather loop init and cond (and if cond) like gatherloopindeces

  virtual void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ){  // both scalar and arrays
    debug_fprintf(stderr,"(%s) forgot to implement gatherDeclRefExpr()\n" ,Chill_AST_Node_Names[asttype]); 
  };



  virtual void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) { 
    debug_fprintf(stderr,"(%s) forgot to implement gatherVarUsage()\n" ,Chill_AST_Node_Names[asttype]); 
  }; 

  virtual void gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls ) { 
    debug_fprintf(stderr,"(%s) forgot to implement gatherVarLHSUsage()\n" ,Chill_AST_Node_Names[asttype]); 
  }; 


  virtual void gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {  // ACTUAL Declaration
    debug_fprintf(stderr,"(%s) forgot to implement gatherVarDecls()\n" ,Chill_AST_Node_Names[asttype]); 
  }; 

  
  virtual void gatherVarDeclsMore( vector<chillAST_VarDecl*> &decls ) {  // even if the decl itself is not in the ast. 
    debug_fprintf(stderr,"(%s) forgot to implement gatherVarDeclsMore()\n" ,Chill_AST_Node_Names[asttype]); 
  }; 

  virtual void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {  // ACTUAL Declaration
    debug_fprintf(stderr,"(%s) forgot to implement gatherScalarVarDecls()\n" ,Chill_AST_Node_Names[asttype]); 
  }; 

  virtual void gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {  // ACTUAL Declaration
    debug_fprintf(stderr,"(%s) forgot to implement gatherArrayVarDecls()\n" ,Chill_AST_Node_Names[asttype]); 
  }; 

  virtual chillAST_VarDecl *findArrayDecl( const char *name ) { // scoping TODO 
    if (!hasSymbolTable()) return parent->findArrayDecl( name ); // most things
    else
      debug_fprintf(stderr,"(%s) forgot to implement gatherArrayVarDecls()\n" ,Chill_AST_Node_Names[asttype]);
  }


  virtual void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
    debug_fprintf(stderr,"(%s) forgot to implement replaceVarDecls()\n" ,Chill_AST_Node_Names[asttype]); 
  }; 

  virtual bool findLoopIndexesToReplace( chillAST_SymbolTable *symtab, bool forcesync=false ) { 
    // this just looks for ForStmts with preferred index metacomment attached 
    debug_fprintf(stderr,"(%s) forgot to implement findLoopIndexesToReplace()\n" ,Chill_AST_Node_Names[asttype]); 
    return false; 
  }

  
  virtual chillAST_node* constantFold() {  // hacky. TODO. make nice
    debug_fprintf(stderr,"(%s) forgot to implement constantFold()\n" ,Chill_AST_Node_Names[asttype]); 
    exit(-1); ; 
  };

  virtual chillAST_node* clone() {   // makes a deep COPY (?)
    debug_fprintf(stderr,"(%s) forgot to implement clone()\n" ,Chill_AST_Node_Names[asttype]); 
    exit(-1); ; 
  };
  virtual void dump(  int indent=0,  FILE *fp = stderr ) { 
    fflush(fp); 
    fprintf(fp,"(%s) forgot to implement dump()\n" ,Chill_AST_Node_Names[asttype]); };// print ast
  
  virtual void print( int indent=0,  FILE *fp = stderr ) { 
    fflush(fp); 
    //debug_fprintf(stderr, "generic chillAST_node::print() called!\n"); 
    //debug_fprintf(stderr, "asttype is %d\n", asttype); 
    fprintf(fp, "\n");
    chillindent(indent, fp); 
    fprintf(fp,"(%s) forgot to implement print()\n" ,Chill_AST_Node_Names[asttype]); 
  };// print CODE 
  
  virtual void printName( int indent=0,  FILE *fp = stderr ) { 
    fflush(fp); 
    //debug_fprintf(stderr, "generic chillAST_node::printName() called!\n"); 
    //debug_fprintf(stderr, "asttype is %d\n", asttype); 
    fprintf(fp, "\n");
    chillindent(indent, fp); 
    fprintf(fp,"(%s) forgot to implement printName()\n" ,Chill_AST_Node_Names[asttype]); 
  };// print CODE 

  virtual char *stringRep(int indent=0 ) {  // the ast's print version
    fflush(stdout);
    // chillindent(indent, fp);  TODO 
    debug_fprintf(stderr,"(%s) forgot to implement stringRep()\n" ,Chill_AST_Node_Names[asttype]);
    segfault(); 
  }


  virtual void printonly( int indent=0,  FILE *fp = stderr ) { print( indent, fp); }; 

  //virtual void printString( std::string &s ) { 
  //  debug_fprintf(stderr,"(%s) forgot to implement printString()\n" ,Chill_AST_Node_Names[asttype]);
  //}


  virtual void get_top_level_loops( std::vector<chillAST_ForStmt *> &loops) {
    int n = children.size(); 
    //debug_fprintf(stderr, "get_top_level_loops of a %s with %d children\n", getTypeString(), n); 
    for (int i=0; i<n; i++) { 
      //debug_fprintf(stderr, "child %d is a %s\n", i, children[i]->getTypeString()); 
      if (children[i]->isForStmt()) {
        loops.push_back( ((chillAST_ForStmt *)(children[i])) );
      }
    }
    //debug_fprintf(stderr, "found %d top level loops\n", loops.size()); 
  }


  virtual void repairParentChild() {  // for nodes where all subnodes are children
    int n = children.size(); 
    for (int i=0; i<n; i++) { 
      if (children[i]->parent != this) { 
        debug_fprintf(stderr, "fixing child %s that didn't know its parent\n", children[i]->getTypeString()); 
        children[i]->parent = this; 
      }
    }
  }



  virtual void get_deep_loops( std::vector<chillAST_ForStmt *> &loops) { // this is probably broken - returns ALL loops under it
    int n = children.size(); 
    //debug_fprintf(stderr, "get_deep_loops of a %s with %d children\n", getTypeString(), n); 
    for (int i=0; i<n; i++) { 
      //debug_fprintf(stderr, "child %d is a %s\n", i, children[i]->getTypeString()); 
      children[i]->get_deep_loops( loops ); 
    }
    //debug_fprintf(stderr, "found %d deep loops\n", loops.size()); 
  }


  // generic for chillAST_node with children
  virtual void find_deepest_loops( std::vector<chillAST_ForStmt *> &loops) { // returns DEEPEST nesting of loops 
    std::vector<chillAST_ForStmt *>deepest; // deepest below here 
    
    int n = children.size(); 
    //debug_fprintf(stderr, "find_deepest_loops of a %s with %d children\n", getTypeString(), n); 
    for (int i=0; i<n; i++) { 
      std::vector<chillAST_ForStmt *> subloops;  // loops below here among a child of mine 
      
      //debug_fprintf(stderr, "child %d is a %s\n", i, children[i]->getTypeString()); 
      children[i]->find_deepest_loops( subloops );
      
      if (subloops.size() > deepest.size()) { 
        deepest = subloops;
      }
    }
    
    // append deepest we see at this level to loops 
    for ( int i=0; i<deepest.size(); i++) { 
      loops.push_back( deepest[i] );
    }

    //debug_fprintf(stderr, "found %d deep loops\n", loops.size()); 
    
  }




  const char *getTypeString() { return Chill_AST_Node_Names[asttype]; } ; 
  int  getType() { return asttype; }; 
  void setParent( chillAST_node *p) { parent = p; } ;
  chillAST_node  *getParent() { return parent; } ;
  
  chillAST_SourceFile *getSourceFile() { 
    if (isSourceFile()) return ((chillAST_SourceFile *)this);
    if (parent != NULL) return parent->getSourceFile(); 
    debug_fprintf(stderr, "UHOH, getSourceFile() called on node %p %s that does not have a parent and is not a source file\n", this, this->getTypeString());
    this->print(); printf("\n\n"); fflush(stdout); 
    exit(-1);
  }
  
  virtual chillAST_node *findDatatype( char *t ) { 
    //debug_fprintf(stderr, "%s looking for datatype %s\n", getTypeString(), t); 
    if (parent != NULL) return parent->findDatatype(t); // most nodes do this
    return NULL; 
  }


  virtual chillAST_SymbolTable *addVariableToSymbolTable( chillAST_VarDecl *vd ) { 
    if (!parent) { 
      debug_fprintf(stderr, "%s with no parent addVariableToSymbolTable()\n", getTypeString()); 
      exit(-1);
    }
    //debug_fprintf(stderr, "%s::addVariableToSymbolTable() (default) headed up\n",  getTypeString()); 
    return parent->addVariableToSymbolTable( vd ); // default, defer to parent 
  }

  virtual void addTypedefToTypedefTable( chillAST_TypedefDecl *tdd ) { 
    parent->addTypedefToTypedefTable( tdd ); // default, defer to parent 
  }

  void walk_parents() { 
    debug_fprintf(stderr, "wp: (%s)  ", getTypeString()); 
    print(); printf("\n");  fflush(stdout); 
    if (isSourceFile()) { debug_fprintf(stderr, "(top sourcefile)\n\n"); return;}

    if (parent) parent->walk_parents();
    else debug_fprintf(stderr, "UHOH, %s has no parent??\n", getTypeString());
    return; 
  }

  virtual chillAST_node *getEnclosingStatement( int level = 0 );
  virtual chillAST_VarDecl *multibase() { 
    debug_fprintf(stderr,"(%s) forgot to implement multibase()\n", Chill_AST_Node_Names[asttype]);
    exit(-1);
  }
  virtual chillAST_node *multibase2() {  
    debug_fprintf(stderr,"(%s) forgot to implement multibase2()\n", Chill_AST_Node_Names[asttype]);
    exit(-1);
  }

  
  virtual void gatherStatements( std::vector<chillAST_node*> &statements ) { 
    debug_fprintf(stderr,"(%s) forgot to implement gatherStatements()\n" ,Chill_AST_Node_Names[asttype]); 
    dump();fflush(stdout); 
    print();
    debug_fprintf(stderr, "\n\n"); 
  }


  virtual bool isSameAs( chillAST_node *other ){  // for tree comparison 
    debug_fprintf(stderr,"(%s) forgot to implement isSameAs()\n" ,Chill_AST_Node_Names[asttype]); 
    dump(); fflush(stdout); 
    print();
    debug_fprintf(stderr, "\n\n");   }

  void printPreprocBEFORE( int indent, FILE *fp );
  void printPreprocAFTER( int indent, FILE *fp );



};

class chillAST_NULL: public chillAST_node {  // NOOP?
public:
  chillAST_NULL(chillAST_node *p = NULL)  {  parent = p; asttype = CHILLAST_NODETYPE_NULL; };
  void print( int indent=0,  FILE *fp = stderr ) { 
    chillindent( indent, fp );
    fprintf(fp, "/* (NULL statement); */ ");
    fflush(fp);
  }
  void dump(  int indent=0,  FILE *fp = stderr ) {
    chillindent( indent, fp );
    fprintf(fp, "(NULL statement) "); fflush(fp);
  }
};


class chillAST_Preprocessing: public chillAST_node { 
public:
  // variables that are special for this type of node
  CHILL_PREPROCESSING_POSITION position;
  CHILL_PREPROCESSING_TYPE pptype;
  char *blurb;

  // constructors
  chillAST_Preprocessing(); // not sure what this is good for
  chillAST_Preprocessing( CHILL_PREPROCESSING_POSITION pos, CHILL_PREPROCESSING_TYPE t, char *text ); 
  
  // other methods particular to this type of node
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  //void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
}; 


//typedef is a keyword in the C and C++ programming languages. The purpose of typedef is to assign alternative names to existing types, most often those whose standard declaration is cumbersome, potentially confusing, or likely to vary from one implementation to another. 
class chillAST_TypedefDecl: public chillAST_node { 
private:
  bool isStruct;
  bool isUnion;
  char *structname;  // get rid of this? 

public:
  char *newtype; // the new type name  ?? 
  char *underlyingtype;  // float, int, "struct bubba" ? 
  char *arraypart;  // string like "[1234][56]"  ?? 

  chillAST_RecordDecl *rd;  // if it's a struct, point to the recorddecl ??
  // TODO what if   "typedef int[10] tenints; " ?? 
  void setStructInfo( chillAST_RecordDecl *arrdee ) { rd = arrdee; };
  chillAST_RecordDecl * getStructDef();


  bool isAStruct() { return isStruct; }; 
  bool isAUnion()  { return isUnion;  };
  void setStruct(bool tf) { isStruct = tf; debug_fprintf(stderr, "%s isStruct %d\n", structname, isStruct);  }; 
  void setUnion( bool tf) { isUnion  = tf; };
  void setStructName( const char *newname) { structname = strdup(newname); }; 
  char *getStructName( ) { return structname; }; 
  bool nameis( const char *n ) { return !strcmp(n, structname); }; 

  // special for struct/unions     rethink TODO 
  vector<chillAST_VarDecl *> subparts;
  chillAST_VarDecl *findSubpart( const char *name );

  //TODO hide data, set/get type and alias
  chillAST_TypedefDecl();
  chillAST_TypedefDecl(char *t, char *nt, chillAST_node *p);
  chillAST_TypedefDecl(char *t, char *nt, char *a, chillAST_node *par); 
  
  const char* getUnderlyingType() { debug_fprintf(stderr, "TypedefDecl getUnderLyingType()\n"); return underlyingtype; }; 
  //virtual chillAST_VarDecl* getUnderlyingVarDecl() { return this; }; // ?? 

  void dump(  int indent=0,  FILE *fp = stderr ) { 
    fprintf(fp, "(TypedefDecl %s %s %s)\n",  underlyingtype, newtype, arraypart); };
  void print( int indent=0,  FILE *fp = stderr ) ;
  //void printString( std::string &s );

};


class chillAST_VarDecl: public chillAST_node { 
public:
  char *vartype; // should probably be an enum, except it's used for unnamed structs too

  chillAST_RecordDecl  *vardef;// the thing that says what the struct looks like
  chillAST_TypedefDecl *typedefinition; // NULL for float, int, etc.
  chillAST_RecordDecl * getStructDef(); // TODO make vardef private?

  //bool insideAStruct;  // this variable is itself part of a struct
  
  char *underlyingtype;
  char *varname;
  char *arraypart;           // [ 12 ] [ 34 ] if that is how it was defined
  char *arraypointerpart;
  char *arraysetpart; 
  void splitarraypart();

  int numdimensions;
  int *arraysizes;       // TODO 
  bool knownArraySizes;  //  if this float *a or float a[128] ?  true means we know ALL dimensions
  int cudamallocsize;      // usually specified in lua/python transformation file 

  bool isRestrict;  // C++ __restrict__ 
  bool isShared; // CUDA  __shared__
  bool isDevice; // CUDA  __device__
  bool isStruct; 
  
  int isAParameter; 
  bool byreference;
  void setByReference( bool tf ) { byreference = tf; debug_fprintf(stderr, "byref %d\n", tf); };

  bool isABuiltin; // if variable is builtin, we don't need to declare it
  void *uniquePtr;  // DO NOT REFERENCE THROUGH THIS! just used to differentiate declarations 
  bool isArray() { return (numdimensions != 0); }; 
  bool isAStruct() { return (isStruct || (typedefinition && typedefinition->isAStruct())); }
  void setStruct( bool b ) {isStruct = b;/*debug_fprintf(stderr,"vardecl %s IS A STRUCT\n",varname);*/ };
  bool isPointer() { return isArray() && !knownArraySizes; }  // 

  bool knowAllDimensions() { return knownArraySizes; } ; 

  chillAST_node *init;
  void setInit( chillAST_node *i ) { init = i; i->setParent(this); };
  bool hasInit() { return init != NULL; };
  chillAST_node *getInit() { return init; };
  
  chillAST_VarDecl();
  chillAST_VarDecl( const char *t,  const char *n, const char *a, chillAST_node *p);
  chillAST_VarDecl( const char *t,  const char *n, const char *a, void *ptr, chillAST_node *p);
  chillAST_VarDecl( chillAST_TypedefDecl *tdd, const char *n, const char *arraypart, chillAST_node *par); 
  chillAST_VarDecl( chillAST_RecordDecl *astruct, const char *n, const char *arraypart, chillAST_node *par); 

  void dump(  int indent=0,  FILE *fp = stderr );
  void print( int indent=0,  FILE *fp = stderr );
  void printName( int indent=0,  FILE *fp = stderr ); 
  bool isParmVarDecl() { return( isAParameter == 1 ); };
  bool isBuiltin()     { return( isABuiltin == 1 ); };  // designate variable as a builtin
  void setLocation( void *ptr ) { uniquePtr = ptr; } ; 


  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;
  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {}; // does nothing
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {}; // does nothing
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) {}; 
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here
  const char* getUnderlyingType() {  /* debug_fprintf(stderr, "VarDecl getUnderLyingType()\n"); */return underlyingtype; }; 
  virtual chillAST_VarDecl* getUnderlyingVarDecl() { return this; }; 

  chillAST_node* constantFold();
  chillAST_node* clone();  

};


class chillAST_DeclRefExpr: public chillAST_node { 
public:
  // variables that are special for this type of node
  char *declarationType; 
  char *declarationName; 
  chillAST_node *decl; // the declaration of this variable or function ... uhoh
  //char *functionparameters;  // TODO probably should split this node into 2 types, one for variables, one for functions

  // constructors
  chillAST_DeclRefExpr(); 
  chillAST_DeclRefExpr( const char *variablename, chillAST_node *p); 
  chillAST_DeclRefExpr( const char *vartype, const char *variablename, chillAST_node *p); 
  chillAST_DeclRefExpr( const char *vartype, const char *variablename, chillAST_node *dec, chillAST_node *p); 
  chillAST_DeclRefExpr( chillAST_VarDecl *vd, chillAST_node *p=NULL); 
  chillAST_DeclRefExpr( chillAST_FunctionDecl *fd, chillAST_node *p=NULL); 
  
  // other methods particular to this type of node
  bool operator!=( chillAST_DeclRefExpr &other ) { return decl != other.decl ; }; 
  bool operator==( chillAST_DeclRefExpr &other ) { return decl == other.decl ; }; // EXACT SAME VARECL BY ADDRESS
  
  chillAST_node *getDecl() { return decl; };

  chillAST_VarDecl *getVarDecl() { 
    if (!decl) return NULL; // should never happen 
    if (decl->isVarDecl()) return (chillAST_VarDecl *)decl;
    return NULL; 
  }; 
  
  chillAST_FunctionDecl *getFunctionDecl() { 
    if (!decl) return NULL; // should never happen 
    if (decl->isFunctionDecl()) return (chillAST_FunctionDecl *)decl;
    return NULL; 
  }; 
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE  
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast   
  char *stringRep(int indent=0 );

  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {}; // do nothing
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento );

  // this is the AST node where these 2 differ 
  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) {};  // does nothing, to get the cvardecl using this method, the actual vardecl must be in the AST 
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ); // returns the decl this declrefexpr references, even if the decl is not in the AST 


  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ); 
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 
  chillAST_node *findref(){return this;}// find the SINGLE constant or data reference at this node or below
  
  const char* getUnderlyingType() {debug_fprintf(stderr, "DeclRefExpr getUnderLyingType()\n"); return decl->getUnderlyingType();}; 

  virtual chillAST_VarDecl* getUnderlyingVarDecl() { return decl->getUnderlyingVarDecl(); } // functions?? TODO 

  chillAST_VarDecl* multibase();
  chillAST_node *multibase2() { return (chillAST_node *)multibase(); } 
}; 





class chillAST_CompoundStmt: public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_SymbolTable  *symbol_table;  // symbols defined inside this compound statement 
  chillAST_TypedefTable *typedef_table;

  bool hasSymbolTable()  { return true; } ;
  bool hasTypeDefTable() { return true; } ;
  chillAST_node *findDatatype( char *t ) { 
    debug_fprintf(stderr, "chillAST_CompoundStmt::findDatatype( %s )\n", t); 
    if (typedef_table) { 
      for (int i=0; i< typedef_table->size(); i++)  {
        chillAST_TypedefDecl *tdd = (*typedef_table)[i];
        if (tdd->nameis( t )) return tdd;
      }
    }
    if (parent) return parent->findDatatype(t);
    return NULL; // should not happen 
  }

  chillAST_SymbolTable *getSymbolTable() { return symbol_table; }

  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) {   // chillAST_CompoundStmt method
    //debug_fprintf(stderr, "\nchillAST_CompoundStmt addVariableToSymbolTable( %s )\n", vd->varname);
    symbol_table = addSymbolToTable( symbol_table, vd ); 
    //printSymbolTable(  symbol_table );
    return symbol_table;
  }

  void addTypedefToTypedefTable( chillAST_TypedefDecl *tdd ) { 
    typedef_table = addTypedefToTable( typedef_table, tdd );
  }

  // constructors
  chillAST_CompoundStmt(); // never has any args ???
  
  // other methods particular to this type of node
  
  
  // required methods 
  void replaceChild( chillAST_node *old, chillAST_node *newchild );
  void dump(  int indent=0,  FILE *fp = stderr );
  void print( int indent=0,  FILE *fp = stderr );
  chillAST_node* constantFold();
  chillAST_node* clone(); 

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ); 
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ); 
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ); 
  void loseLoopWithLoopVar( char *var ); // special case this for not for debugging

  void gatherStatements( std::vector<chillAST_node*> &statements );
}; 




class chillAST_RecordDecl: public chillAST_node {  // declaration of the shape of a struct or union 
private:
  char *name;  // could be NULL? for unnamed structs?
  char *originalname; 
  bool isStruct;

  bool isUnion;
  vector<chillAST_VarDecl *> subparts;
  
public:
  chillAST_RecordDecl();
  chillAST_RecordDecl( const char *nam, chillAST_node *p ); 
  chillAST_RecordDecl( const char *nam, const char *orig, chillAST_node *p ); 

  void  setName( const char *newname) { name = strdup(newname); }; 
  char *getName( ) { return name; }; 
  
  bool isAUnion()  { return isUnion;  };
  bool isAStruct() { return isStruct; }; 
  bool isUnnamed;
  void setUnnamed( bool b ) { isUnnamed = b; };


  void setStruct(bool tf) { isStruct = tf; }; 
  //debug_fprintf(stderr, "%s isStruct %d\n", structname, isStruct);  }; 
  void setUnion( bool tf) { isUnion  = tf; };

  chillAST_SymbolTable *addVariableToSymbolTable( chillAST_VarDecl *vd ); //  RecordDecl does NOTHING
  
  int numSubparts() { return subparts.size(); }; 
  void addSubpart( chillAST_VarDecl *s ) { subparts.push_back(s); }; 
  chillAST_VarDecl *findSubpart( const char *name );
  chillAST_VarDecl *findSubpartByType( const char *typ );

  void dump(  int indent=0,  FILE *fp = stderr );
  void print( int indent=0,  FILE *fp = stderr ) ;
  void printStructure( int indent=0,  FILE *fp = stderr ) ;
};




class chillAST_FunctionDecl: public chillAST_node { 
private:
  chillAST_CompoundStmt *body; // always a compound statement? 
  CHILL_FUNCTION_TYPE function_type;  // CHILL_FUNCTION_CPU or  CHILL_FUNCTION_GPU
  bool externfunc;   // function is external 
  bool builtin;      // function is a builtin
  bool forwarddecl; 

public:
  char *returnType;
  char *functionName;

  // parameters
  int numParameters() { return parameters.size(); } ; 
  chillAST_SymbolTable parameters;

  // this is probably a mistake, but symbol_table here is pointing to BODY'S symbol table
  //chillAST_SymbolTable  *symbol_table;  // symbols defined inside this function. REALLY the body's symbol table?

  chillAST_TypedefTable *typedef_table; // function typedef table


  bool hasSymbolTable() { return true; } ; // COULD HAVE
  bool hasTypeDefTable(){ return true; } ; // COULD HAVE 


  //char *parametertypes; // a single string?? 
  void printParameterTypes( FILE *fp ); 
  void setName( char *n ) { functionName = strdup(n); /* probable memory leak */ }; 

  void setBuiltin() { builtin = true; } ; // designate function as a builtin
  bool isBuiltin()  { return builtin; } ; // report whether is a builtin 

  void setExtern() { externfunc = true; }; // designate function as external 
  bool isExtern()  { return externfunc; }; // report whether function is external

  void setForward() { forwarddecl = true; }; // designate function as fwd declaration
  bool isForward()  { return forwarddecl; }; // report whether function is external

  bool isFunctionCPU() { return( function_type == CHILL_FUNCTION_CPU ); };
  bool isFunctionGPU() { return( function_type == CHILL_FUNCTION_GPU ); };
  void setFunctionCPU() { function_type = CHILL_FUNCTION_CPU; };
  void setFunctionGPU() { function_type = CHILL_FUNCTION_GPU; };

  void *uniquePtr;  // DO NOT REFERENCE THROUGH THIS! USED AS A UNIQUE ID

  
  

  chillAST_FunctionDecl(); //  { asttype = CHILLAST_NODETYPE_FUNCTIONDECL; numparameters = 0;}; 
  chillAST_FunctionDecl(const char *rt, const char *fname, chillAST_node *p=NULL ) ;
  chillAST_FunctionDecl(const char *rt, const char *fname, chillAST_node *p, void *unique ) ;
  
  void addParameter( chillAST_VarDecl *p); 
  chillAST_VarDecl *hasParameterNamed( const char *name ); 
  chillAST_VarDecl *findParameterNamed( const char *name ) { return hasParameterNamed( name ); }; 

  void addDecl( chillAST_VarDecl *vd);  // just adds to symbol table?? TODO 

  chillAST_VarDecl *funcHasVariableNamed( const char *name );  // functiondecl::hasVariableNamed
  //chillAST_VarDecl *findVariableNamed( const char *name ) { return hasVariableNamed( name ); }; 

  void addChild(chillAST_node* node); // special because inserts into BODY
  void insertChild(int i, chillAST_node* node); // special because inserts into BODY

  void setBody( chillAST_node * bod );  
  chillAST_CompoundStmt *getBody() { return( body); }
  
  void print( int indent=0,  FILE *fp = stderr ); // in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr ); // in chill_ast.cc

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );
  chillAST_VarDecl *findArrayDecl( const char *name ); 
  //void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ); 
  //void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) 

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ); 
  void cleanUpVarDecls();   

  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ); 

  chillAST_node* constantFold();

  chillAST_node *findDatatype( char *t ) { 
    //debug_fprintf(stderr, "%s looking for datatype %s\n", getTypeString(), t); 
    if (!typedef_table) { // not here
      if (parent) return parent->findDatatype(t); // not here, check parents
      else return NULL; // not defined here and no parent 
    }
    
    //debug_fprintf(stderr, "%d typedefs\n", typedef_table->size());
    for (int i=0; i< typedef_table->size(); i++)  {
      chillAST_TypedefDecl *tdd = (*typedef_table)[i];
      if ( tdd->nameis( t )) return tdd;
    }
    if (parent) return parent->findDatatype(t);
    return NULL; // should not happen 
  }

  chillAST_SymbolTable *getParameterSymbolTable() { return &parameters; }
  chillAST_SymbolTable *getSymbolTable() { return body->getSymbolTable(); }  //symbol_table; } // 
  void setSymbolTable( chillAST_SymbolTable *tab ) { 
    // no longer keeping a local ?? symbol_table = tab;
    if (!body) { // can never happen now 
      body = new chillAST_CompoundStmt(); 
    } // only if func is empty!
    body->symbol_table = tab; 
  }

  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) {  // chillAST_FunctionDecl method 
    //debug_fprintf(stderr, "\nchillAST_FunctionDecl addVariableToSymbolTable( %s )\n", vd->varname);
    
    // this is all dealing with the body's symbol table
    // the function has a symbol table called "parameters" but that is a special case

    addSymbolToTable( getSymbolTable(), vd ); 
    if (!vd->parent) { 
      //debug_fprintf(stderr, "setting parent of vardecl to be the function whose symbol table it is going into\n"); // ?? 
      vd->setParent( this );
      insertChild(0,vd);  
    }
    //printSymbolTable( getSymbolTable() ); 
    return getSymbolTable();
  }


  void addTypedefToTypedefTable( chillAST_TypedefDecl *tdd ) { 
    typedef_table = addTypedefToTable( typedef_table, tdd );
  }

  void replaceChild( chillAST_node *old, chillAST_node *newchild ) { 
    body->replaceChild( old, newchild ); 
  }
};  // end FunctionDecl 




class chillAST_SourceFile: public chillAST_node { 
public:

  // constructors
  chillAST_SourceFile();                       //  defined in chill_ast.cc 
  chillAST_SourceFile(const char *filename );  //  defined in chill_ast.cc 
  
  ~chillAST_SourceFile();                       //  defined in chill_ast.cc

void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void printToFile( char *filename = NULL ); 
  
  char *SourceFileName; // where this originated
  char *FileToWrite; 
  char *frontend;

  void setFileToWrite( char *f ) { FileToWrite =  strdup( f ); }; 
  
  void setFrontend( const char *compiler ) { if (frontend) free(frontend); frontend = strdup(compiler); } 
  // get, set filename ? 

  chillAST_SymbolTable  *global_symbol_table;  // (global) symbols defined inside this source file 
  chillAST_TypedefTable *global_typedef_table; // source file 
  chillAST_VarDecl *findVariableNamed( const char *name ); // looks in global_symbol_table;

  bool hasSymbolTable()  { return true; } ;  // "has" vs "can have"    TODO 
  bool hasTypeDefTable() { return true; } ;

  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) {  // chillAST_SourceFile method
    global_symbol_table = addSymbolToTable( global_symbol_table, vd );
    return global_symbol_table;
  }

  void addTypedefToTypedefTable( chillAST_TypedefDecl *tdd ) {
    assert(this->global_typedef_table != NULL);
    this->global_typedef_table = addTypedefToTable( this->global_typedef_table, tdd );
  }

  chillAST_node *findDatatype( char *type_name ) {
    // Look for name in global typedefs
    assert(this->global_typedef_table != NULL);
    for (int i=0; i < this->global_typedef_table->size(); i++)  {
      chillAST_TypedefDecl *tdd = (*this->global_typedef_table)[i];
      if (tdd->nameis( type_name )) {
        return (chillAST_node *)tdd;
      }
    }
    return NULL;
  }

  vector< chillAST_FunctionDecl *>     functions;  // at top level, or anywhere?
  vector< chillAST_MacroDefinition *>  macrodefinitions;

  chillAST_MacroDefinition* findMacro( const char *name ); // TODO ignores arguments
  chillAST_FunctionDecl *findFunction( const char *name ); // TODO ignores arguments
  chillAST_node *findCall( const char *name ); 
  void addMacro(chillAST_MacroDefinition* md) {
    macrodefinitions.push_back(md);
    //debug_fprintf(stderr, "addMacro(), now %d macros\n", macrodefinitions.size()); 
  }
  void addFunc(chillAST_FunctionDecl* fd) { 
    //debug_fprintf(stderr, "chillAST_SourceFile::addFunc( %s %p)\n", fd->functionName, fd);

    bool already = false; 
    for (int i=0; i<functions.size(); i++) { 
      //debug_fprintf(stderr, "function %d is %s %p\n", i, functions[i]->functionName, functions[i]); 
      if (functions[i] == fd) { 
        //debug_fprintf(stderr, "function %s was already in source functions\n", fd->functionName); 
        already = true;
      }
    }
    if (!already) functions.push_back(fd);

    // PROBABLY fd was created with sourcefile as its parent. Don't add it twice
    addChild( (chillAST_node *)fd); }

};


/* 
   class chillAST_VarDecl: public chillAST_node {  // now a SINGLE DECL.  multiples in 
   public:
   int howmany;  // usually 1 but sometimes multiple declarations in a decl;
   std::vector<class chillAST_SingleVarDecl*> decls;
   
   chillAST_VarDecl();
   chillAST_VarDecl( char *t,  char *n, char *a);
   void addDecl( char *t,  char *n, char *a);
   
   void dump(  int indent=0,  FILE *fp = stderr );
   void print( int indent=0,  FILE *fp = stderr );
   }; 
*/


/* 
class chillAST_ParmVarDecl: public chillAST_node {  // no longer used? 
public:
  char *vartype; // should probably be an enum
  char *varname;
  char *arraypart;
  int numdimensions; // TODO 
  int *arraysizes;   // TODO 
  // hasDefaultArg
  // getDefaultArg
  
  chillAST_ParmVarDecl(); 
  chillAST_ParmVarDecl( const char *type, const char *name, const char *ap, chillAST_node *p );
  
  void dump(  int indent=0,  FILE *fp = stderr ) { 
    fprintf(fp, "(2VarDecl'%s' '%s' '%s')",  vartype, varname, arraypart); };
  void print( int indent=0,  FILE *fp = stderr );
};
*/



class chillAST_MacroDefinition: public chillAST_node { 
private:
  chillAST_node *body; // rhs      always a compound statement? 
  chillAST_SymbolTable *symbol_table;
public:
  char *macroName;
  char *rhsString; 

  // parameters - these will be odd, in that they HAVE NO TYPE
  int numParameters() { return parameters.size(); } ; 
  std::vector<chillAST_VarDecl *>parameters;
  
  void setName( char *n ) { macroName = strdup(n); /* probable memory leak */ }; 
  void setRhsString( char *n ) { rhsString = strdup(n); /* probable memory leak */ }; 
  char *getRhsString() { return rhsString; }

  chillAST_MacroDefinition();
  chillAST_MacroDefinition( const char *name, chillAST_node *par);
  chillAST_MacroDefinition( const char *name, const char *rhs, chillAST_node *par);
  
  void addParameter( chillAST_VarDecl *p);  // parameters have no TYPE ??
  chillAST_VarDecl *hasParameterNamed( const char *name ); 
  chillAST_VarDecl *findParameterNamed( const char *name ) { return hasParameterNamed( name ); };
  void addChild(chillAST_node* node); // special because inserts into BODY
  void insertChild(int i, chillAST_node* node); // special because inserts into BODY
  
  void setBody( chillAST_node * bod );  
  chillAST_node *getBody() { return( body); }
  
  void print( int indent=0,  FILE *fp = stderr ); // in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr ); // in chill_ast.cc
  
  bool hasSymbolTable() { return true; } ;

  //const std::vector<chillAST_VarDecl *> getSymbolTable() { return symbol_table; }
  chillAST_SymbolTable *getSymbolTable() { return symbol_table; }
  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) {  // chillAST_MacroDefinition method  ??
    //debug_fprintf(stderr, "\nchillAST_MacroDefinition addVariableToSymbolTable( %s )\n", vd->varname);
    symbol_table = addSymbolToTable( symbol_table, vd ); 
    //printSymbolTable(  symbol_table );
    return symbol_table;
  }


  chillAST_node* clone();

  // none of these make sense for macros 
  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ){};
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ){};
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ){};
  chillAST_VarDecl *findArrayDecl( const char *name ){};
  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ){};
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ){}; 
  void cleanUpVarDecls();   
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){}; 
  chillAST_node* constantFold(){};
};
  
  
  



class chillAST_ForStmt: public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_node *init;
  chillAST_node *cond;
  chillAST_node *incr;
  chillAST_node *body; // always a compoundstmt? 
  IR_CONDITION_TYPE conditionoperator;  // from ir_code.hh

  chillAST_SymbolTable *symbol_table; // symbols defined inside this forstmt (in init but not body?) body is compound stmt 
   bool hasSymbolTable() { return true; } ;

  // constructors
  chillAST_ForStmt();
  chillAST_ForStmt(  chillAST_node *ini, chillAST_node *con, chillAST_node *inc, chillAST_node *bod, chillAST_node *p); 
  
  // other methods particular to this type of node
  void addSyncs();
  void removeSyncComment(); 
  chillAST_node *getInit() { return init; };
  chillAST_node *getCond() { return cond; };
  chillAST_node *getInc()  { return incr; };
  chillAST_node *getBody() { //debug_fprintf(stderr, "chillAST_ForStmt::getBody(), returning a chillAST_node of type %s\n", body->getTypeString()); 
    return body; }; 
  void setBody( chillAST_node *b ) { body = b;  b->parent = this; };
  
  bool isNotLeaf() { return true; }; 
  bool isLeaf()    { return false; }; 

  
  // required methods that I can't seem to get to inherit
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void printControl( int indent=0,  FILE *fp = stderr );  // print just for ( ... ) but not body 

  chillAST_node* constantFold();
  chillAST_node* clone(); 

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ); 
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ); 
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl); // will get called on inner loops 
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false );

  void gatherLoopIndeces( std::vector<chillAST_VarDecl*> &indeces );
  void gatherLoopVars(  std::vector<std::string> &loopvars );  // gather as strings ??

  void get_deep_loops( std::vector<chillAST_ForStmt *> &loops) { // chillAST_ForStmt version 
    // ADD MYSELF!
    loops.push_back( this );

    int n = body->children.size(); 
    //debug_fprintf(stderr, "get_deep_loops of a %s with %d children\n", getTypeString(), n); 
    for (int i=0; i<n; i++) { 
      //debug_fprintf(stderr, "child %d is a %s\n", i, body->children[i]->getTypeString()); 
      body->children[i]->get_deep_loops( loops ); 
    }
    //debug_fprintf(stderr, "found %d deep loops\n", loops.size()); 
  }


  void find_deepest_loops( std::vector<chillAST_ForStmt *> &loops) { 
    std::vector<chillAST_ForStmt *> b; // deepest loops below me

    int n = body->children.size(); 
    for (int i=0; i<n; i++) { 
      std::vector<chillAST_ForStmt *> l; // deepest loops below one child
      body->children[i]->find_deepest_loops( l ); 
      if ( l.size() > b.size() ) { // a deeper nesting than we've seen
        b = l;
      }
    }

    loops.push_back( this ); // add myself
    for (int i=0; i<b.size(); i++) loops.push_back(b[i]);
  }


  void loseLoopWithLoopVar( char *var ); // chillAST_ForStmt
  void replaceChild( chillAST_node *old, chillAST_node *newchild ) ; 

  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) {   // chillAST_ForStmt method 
    //debug_fprintf(stderr, "\nchillAST_ForStmt addVariableToSymbolTable( %s )\n", vd->varname);
    symbol_table = addSymbolToTable( symbol_table, vd ); 
    //printSymbolTable(  symbol_table );
    return symbol_table;
  }

  void gatherStatements( std::vector<chillAST_node*> &statements );
  bool lowerBound( int &l ); 
  bool upperBound( int &u );

}; 



class chillAST_TernaryOperator: public chillAST_node { 
public:
  // variables that are special for this type of node
  char *op;            // TODO need enum  so far, only "?" conditional operator
  chillAST_node *condition;
  chillAST_node *lhs;        // keep name from binary 
  chillAST_node *rhs;
  
  
  // constructors
  chillAST_TernaryOperator();
  chillAST_TernaryOperator(const char *op, chillAST_node *cond, chillAST_node *lhs, chillAST_node *rhs, chillAST_node *p=NULL); 
  
  // other methods particular to this type of node
  bool isNotLeaf() { return true; }; 
  bool isLeaf()    { return false; }; 
  
  
  char          *getOp()  { return op; };  // dangerous. could get changed!
  chillAST_node *getCond() { return condition; }; 
  chillAST_node *getRHS() { return rhs; }; 
  chillAST_node *getLHS() { return lhs; };

  void setCond( chillAST_node *newc ) { condition = newc;  newc->setParent( this ); } 
  void setLHS( chillAST_node *newlhs ) { lhs = newlhs;  newlhs->setParent( this ); } 
  void setRHS( chillAST_node *newrhs ) { rhs = newrhs;  newrhs->setParent( this ); } 

  
  
  
  // required methods that I can't seem to get to inherit
  void dump(  int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void printonly( int indent=0,  FILE *fp = stderr );

  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void replaceChild( chillAST_node *old, chillAST_node *newchild ) ; 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ); 
  void gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls );
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 
  void loseLoopWithLoopVar( char *var ){}; // ternop can't have loop as child? 
}; 



class chillAST_BinaryOperator: public chillAST_node { 
public:
  // variables that are special for this type of node
  char *op;            // TODO need enum
  chillAST_node *lhs;
  chillAST_node *rhs;
  
  
  // constructors
  chillAST_BinaryOperator();
  chillAST_BinaryOperator(chillAST_node *lhs, const char *op, chillAST_node *rhs, chillAST_node *p=NULL); 
  
  // other methods particular to this type of node
  int evalAsInt();
  chillAST_IntegerLiteral *evalAsIntegerLiteral(); 

  bool isNotLeaf() { return true; }; 
  bool isLeaf()    { return false; }; 
  
  chillAST_node *getRHS() { return rhs; }; 
  chillAST_node *getLHS() { return lhs; };
  void setLHS( chillAST_node *newlhs ) { lhs = newlhs;  newlhs->setParent( this ); } 
  void setRHS( chillAST_node *newrhs ) { rhs = newrhs;  newrhs->setParent( this ); } 

  char          *getOp()  { return op; };  // dangerous. could get changed!
  bool isAugmentedAssignmentOp() { 
    return 
      (!strcmp(op, "*=")) || // BO_MulAssign, 
      (!strcmp(op, "/=")) || // BO_DivAssign
      (!strcmp(op, "%=")) || // BO_RemAssign
      (!strcmp(op, "+=")) || // BO_AddAssign 
      (!strcmp(op, "-=")) || // BO_SubAssign
      
      (!strcmp(op, "<<=")) || // BO_ShlAssign
      (!strcmp(op, ">>=")) || // BO_ShrAssign
      (!strcmp(op, "&&=")) || // BO_AndAssign
      (!strcmp(op, "||=")) || // BO_OrAssign
      
      (!strcmp(op, "^="))    // BO_XorAssign 
      ; 
  }
  bool isAssignmentOp() { 
    return( (!strcmp(op, "=")) ||  // BO_Assign,
             isAugmentedAssignmentOp() ); 
  }; 
  bool isComparisonOp() { return (!strcmp(op,"<")) || (!strcmp(op,">")) || (!strcmp(op,"<=")) || (!strcmp(op,">=")); }; 
  
  bool isPlusOp()  { return (!strcmp(op,"+")); };
  bool isMinusOp() { return (!strcmp(op,"-")); };
  bool isPlusMinusOp() { return (!strcmp(op,"+")) || (!strcmp(op,"-")); };
  bool isMultDivOp()   { return (!strcmp(op,"*")) || (!strcmp(op,"/")); };
  
  bool isStructOp() { return (!strcmp(op,".")) || (!strcmp(op,"->")); }; 
  
  
  // required methods that I can't seem to get to inherit
  void dump(  int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void printonly( int indent=0,  FILE *fp = stderr );
  char *stringRep(int indent=0 );

  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void replaceChild( chillAST_node *old, chillAST_node *newchild ) ; 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ); // chillAST_BinaryOperator
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ); 
  void gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls );
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 
  void loseLoopWithLoopVar( char *var ){}; // binop can't have loop as child? 

  void gatherStatements( std::vector<chillAST_node*> &statements ); // 

  bool isSameAs( chillAST_node *other );

}; 








class chillAST_ArraySubscriptExpr: public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_node *base;  // always a decl ref expr? No, for multidimensional array, is another ASE 
  chillAST_node *index;
  bool imwrittento;
  bool imreadfrom; // WARNING: ONLY used when both writtento and readfrom are true  x += 1 and so on
  chillAST_VarDecl *basedecl; // the vardecl that this refers to
  void *uniquePtr;  // DO NOT REFERENCE THROUGH THIS!
  
  // constructors
  chillAST_ArraySubscriptExpr(); 
  chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, chillAST_node *p, void *unique);
  chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, bool writtento, chillAST_node *p, void *unique);
  
  chillAST_ArraySubscriptExpr( chillAST_VarDecl *v, std::vector<chillAST_node *> indeces, chillAST_node *p); 
  
  // other methods particular to this type of node
  bool operator!=( const chillAST_ArraySubscriptExpr& ) ; 
  bool operator==( const chillAST_ArraySubscriptExpr& ) ; 
  chillAST_VarDecl *multibase();  // method for finding the basedecl 
  chillAST_node *multibase2() { return  base->multibase2();  }

  chillAST_node *getIndex(int dim);
  void gatherIndeces( std::vector< chillAST_node * > &ind ); 

  void replaceChild( chillAST_node *old, chillAST_node *newchild ); // will examine index

  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void printonly( int indent=0,  FILE *fp = stderr );
  void print( int indent=0,  FILE *fp = stderr ) const;  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  char *stringRep(int indent=0 );

  chillAST_node* constantFold();
  chillAST_node* clone(); 
  chillAST_node *findref(){return this;}// find the SINGLE constant or data reference at this node or below
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ); 
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

  const char* getUnderlyingType() { 
    //debug_fprintf(stderr, "ASE getUnderlyingType() base of type %s\n", base->getTypeString()); base->print(); printf("\n"); fflush(stdout); 
    return base->getUnderlyingType(); }; 

  virtual chillAST_VarDecl* getUnderlyingVarDecl() { return base->getUnderlyingVarDecl(); };

}; 



class chillAST_MemberExpr: public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_node *base;  // always a decl ref expr? No, can be Array Subscript Expr
  char *member; 
  char *printstring; 

  chillAST_VarDecl *basedecl; // the vardecl that this refers to
  void *uniquePtr;  // DO NOT REFERENCE THROUGH THIS!

  CHILL_MEMBER_EXP_TYPE exptype; 
  

  // constructors
  chillAST_MemberExpr(); 
  chillAST_MemberExpr( chillAST_node *bas, const char *mem, chillAST_node *p, void *unique, CHILL_MEMBER_EXP_TYPE t=CHILL_MEMBER_EXP_DOT);
  
  // other methods particular to this type of node
  bool operator!=( const chillAST_MemberExpr& ) ; 
  bool operator==( const chillAST_MemberExpr& ) ; 
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void printonly( int indent=0,  FILE *fp = stderr );
  void print( int indent=0,  FILE *fp = stderr ) const;  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  char *stringRep( int indent = 0);
 
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ); 
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

  chillAST_VarDecl* getUnderlyingVarDecl(); 

  void replaceChild( chillAST_node *old, chillAST_node *newchild );

  void setType( CHILL_MEMBER_EXP_TYPE t ) { exptype = t; };
  CHILL_MEMBER_EXP_TYPE getType( CHILL_MEMBER_EXP_TYPE t ) { return exptype; };

  chillAST_VarDecl* multibase();   // this one will return the member decl 
  chillAST_node*    multibase2();  // this one will return the member expression
}; 




class chillAST_IntegerLiteral: public chillAST_node { 
public:
  // variables that are special for this type of node
  int value;
  
  // constructors
  chillAST_IntegerLiteral(int val, chillAST_node *p = NULL); 
  
  // other methods particular to this type of node
  int evalAsInt() { return value; } 

  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ){}; // does nothing
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ){}; // does nothing

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ){}; // does nothing 
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ){}; // does nothing
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ){}; // does nothing

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {}; // does nothing
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {};  // does nothing 
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) {};
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

  chillAST_node *findref(){return this;}// find the SINGLE constant or data reference at this node or below
}; 


class chillAST_FloatingLiteral: public chillAST_node { 
public:
  // variables that are special for this type of node
  float value;
  double doublevalue;
  int float0double1; 

  char *allthedigits; // if not NULL, use this as printable representation
  int precision;   // float == 1, double == 2, ??? 

  // constructors
  chillAST_FloatingLiteral( float  val,                                 chillAST_node *p); 
  chillAST_FloatingLiteral( double val,                                 chillAST_node *p); 
  chillAST_FloatingLiteral( float  val, int pre,                        chillAST_node *p); 
  chillAST_FloatingLiteral( double val, int pre,                        chillAST_node *p); 
  chillAST_FloatingLiteral( float  val, const char *printable,          chillAST_node *p); 
  chillAST_FloatingLiteral( float  val, int pre, const char *printable, chillAST_node *p); 
  chillAST_FloatingLiteral( chillAST_FloatingLiteral *old ); 
  
  // other methods particular to this type of node
  void setPrecision( int precis ) { precision = precis; }; 
  int getPrecision() { return precision; } 
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ){}; // does nothing
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ){}; // does nothing

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ){}; // does nothing 
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ){}; // does nothing ;
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ){}; // does nothing ;

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {}; // does nothing 
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ){}; // does nothing 
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){};
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 
  chillAST_node *findref(){return this;};// find the SINGLE constant or data reference at this node or below

 bool isSameAs( chillAST_node *other ); 
}; 




class chillAST_UnaryOperator: public chillAST_node { 
public:
  // variables that are special for this type of node
  char *op; // TODO enum
  bool prefix; // or post
  chillAST_node *subexpr;
  
  // constructors
  chillAST_UnaryOperator( const char *oper, bool pre, chillAST_node *sub, chillAST_node *p ); 
  
  // other methods particular to this type of node
  bool isAssignmentOp() { 
    return( (!strcmp(op, "++")) || 
            (!strcmp(op, "--")) );   // are there more ???  TODO 
  }
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ); // chillAST_UnaryOperator

  void gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls );

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

  int evalAsInt();
  bool isSameAs( chillAST_node *other );

}; 





class chillAST_ImplicitCastExpr: public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_node *subexpr;
  
  // constructors
  chillAST_ImplicitCastExpr(chillAST_node *sub, chillAST_node *p); 
  
  // other methods particular to this type of node
  bool isNotLeaf() { return true; }; 
  bool isLeaf()    { return false; }; 
  
  // required methods that I can't seem to get to inherit
  void replaceChild( chillAST_node *old, chillAST_node *newchild );
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void printonly( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr ) { print( indent, fp); };  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 
  chillAST_VarDecl *multibase(); // just recurse on subexpr

}; 



class chillAST_CStyleCastExpr: public chillAST_node { 
public:
  // variables that are special for this type of node
  char * towhat; 
  chillAST_node *subexpr;
  // constructors
  chillAST_CStyleCastExpr(const char *to, chillAST_node *sub, chillAST_node *p=NULL); 
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void replaceChild( chillAST_node *old, chillAST_node *newchild );
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 
  chillAST_node *findref(){return subexpr;};// find the SINGLE constant or data reference at this node or below

}; 


class chillAST_CStyleAddressOf: public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_node *subexpr;
  // constructors
  chillAST_CStyleAddressOf(chillAST_node *sub, chillAST_node *p=NULL); 
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

  
}; 


class chillAST_CudaMalloc:public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_node *devPtr;  // Pointer to allocated device memory 
  chillAST_node *sizeinbytes;

  // constructors
  chillAST_CudaMalloc(chillAST_node *devmemptr, chillAST_node *size, chillAST_node *p=NULL); 
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  //void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

}; 


class chillAST_CudaFree:public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_VarDecl *variable;  

  // constructors
  chillAST_CudaFree(chillAST_VarDecl *var, chillAST_node *p=NULL); 
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  //void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

}; 





class chillAST_Malloc:public chillAST_node {   // malloc( sizeof(int) * 2048 ); 
public:
  // variables that are special for this type of node
  char *thing;  // to void if this is null  ,  sizeof(thing) if it is not 
  chillAST_node *sizeexpr; // bytes

  // constructors
  chillAST_Malloc(chillAST_node *size, chillAST_node *p=NULL); 
  chillAST_Malloc(char *thething, chillAST_node *numthings, chillAST_node *p=NULL); // malloc (sizeof(int) *1024)

  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  //void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

};




class chillAST_Free:public chillAST_node { 
public:
 



}; 




class chillAST_CudaMemcpy:public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_VarDecl *dest;  // Pointer to allocated device memory 
  chillAST_VarDecl *src;
  chillAST_node *size;
  char *cudaMemcpyKind;  // could use the actual enum

  // constructors
  chillAST_CudaMemcpy(chillAST_VarDecl *d, chillAST_VarDecl *s, chillAST_node *size, char *kind, chillAST_node *p=NULL); 
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  //void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

}; 


class chillAST_CudaSyncthreads:public chillAST_node { 
public:
  // variables that are special for this type of node

  // constructors
  chillAST_CudaSyncthreads(chillAST_node *p=NULL); 
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  //chillAST_node* constantFold() {};
  //chillAST_node* clone(); 
  //void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ){};
  //void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) {}; // does nothing 
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {}; // does nothing 
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) {}; // does nothing 

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {}; // does nothing
  //void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  //bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; 

}; 


 
class chillAST_ReturnStmt: public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_node *returnvalue;
  
  // constructors
  chillAST_ReturnStmt( chillAST_node *retval, chillAST_node *p ); 
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone(); 

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

}; 



class chillAST_CallExpr: public chillAST_node {  // a function call 
public:
  // variables that are special for this type of node
  chillAST_node *callee;   // the function declaration (what about builtins?)
  int numargs;
  std::vector<class chillAST_node*> args;
  chillAST_VarDecl *grid;
  chillAST_VarDecl *block;

  // constructors
  chillAST_CallExpr(chillAST_node *function, chillAST_node *p );
  void addArg(  chillAST_node *newarg  ); 
  
  // other methods particular to this type of node
  // TODO get/set grid, block
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold();
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 
  chillAST_node* clone();
}; 



class chillAST_ParenExpr: public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_node *subexpr;
  
  // constructors
  chillAST_ParenExpr( chillAST_node *sub, chillAST_node *p=NULL ); 
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone();
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

}; 


class chillAST_Sizeof: public chillAST_node { 
public:
  // variables that are special for this type of node
  char *thing;  
  
  // constructors
  chillAST_Sizeof( char *t, chillAST_node *p = NULL ); 
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
  chillAST_node* constantFold();
  chillAST_node* clone();
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento );
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){};
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; }; // no loops under here 

}; 



class chillAST_NoOp: public chillAST_node { 
public:
  chillAST_NoOp( chillAST_node *p = NULL ); //  { parent = p; };

  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr ) {};  // print CODE   in chill_ast.cc
  void dump(  int indent=0,  FILE *fp = stderr ) {};  // print ast    in chill_ast.cc
  chillAST_node* constantFold() {};
  chillAST_node* clone() { return new chillAST_NoOp( parent ); }; // ?? 

  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {};
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {};

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ){};
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ){};
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ){};

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ){};
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ){};
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){};
  bool findLoopIndexesToReplace( chillAST_SymbolTable *symtab, bool forcesync=false ){ return false; };//no loops under here 
};



class chillAST_IfStmt: public chillAST_node { 
public:
  // variables that are special for this type of node
  chillAST_node *cond;
  chillAST_node *thenpart;
  chillAST_node *elsepart;
  IR_CONDITION_TYPE conditionoperator;  // from ir_code.hh
  
  // constructors
  chillAST_IfStmt();
  chillAST_IfStmt( chillAST_node *c, chillAST_node *t, chillAST_node *e, chillAST_node *p); 
  
  // other methods particular to this type of node
  chillAST_node *getCond() { return cond; };
  chillAST_node *getThen() { return thenpart; };
  chillAST_node *getElse() { return elsepart; };

  void setCond( chillAST_node *b ) {     cond = b;  if (cond)     cond->parent = this; };
  void setThen( chillAST_node *b ) { thenpart = b;  if (thenpart) thenpart->parent = this; };
  void setElse( chillAST_node *b ) { elsepart = b;  if (elsepart) elsepart->parent = this; };
  
  // required methods that I can't seem to get to inherit
  void dump(  int indent=0,  FILE *fp = stderr ); 
  void print( int indent=0,  FILE *fp = stderr ); 

  chillAST_node* constantFold();
  chillAST_node* clone(); 

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls );
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls );
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls );
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ); 
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) ;

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls );
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs );
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ); 

  void gatherStatements( std::vector<chillAST_node*> &statements );
 
}; 








class chillAST_something: public chillAST_node { 
public:
  // variables that are special for this type of node
  
  // constructors
  chillAST_something(); 
  
  // other methods particular to this type of node
  

  // required methods that I can't seem to get to inherit
  void print( int indent=0,  FILE *fp = stderr );  // print CODE   in chill_ast.cc
  //void dump(  int indent=0,  FILE *fp = stderr );  // print ast    in chill_ast.cc
}; 




chillAST_FunctionDecl *findFunctionDecl( chillAST_node *node, const char *procname); 



#endif

