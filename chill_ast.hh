

#ifndef _CHILL_AST_H_
#define _CHILL_AST_H_


#define CHILL_INDENT_AMOUNT 2

#include "chill_io.hh"
#include "chill_error.hh"

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>  // std::vector 
#include <type_traits>

#include <ir_enums.hh> // for IR_CONDITION_*

using std::vector;
using std::string;

//! Parse to the most basic type
char *parseUnderlyingType( char *sometype );

char *parseArrayParts( char *sometype );

bool isRestrict( const char *sometype );


//! return the bracketed part of a type
char *splitTypeInfo( char *underlyingtype );

//! remove UL from numbers, MODIFIES the argument!
/*!
 * change "1024UL" to "1024" 
 */
char *ulhack(  char *brackets );

//! remove __restrict__ , MODIFIES the argument!
char *restricthack( char *typeinfo );

#ifdef chillast_nodetype
#error "chillast_nodetype already defined"
#else
#define chillast_nodetype(n, s)                     CHILLAST_NODETYPE_##n,
#define chillast_nodetype_alias(a, b)               CHILLAST_NODETYPE_##a = CHILLAST_NODETYPE_##b,
#endif

enum CHILL_ASTNODE_TYPE {
  CHILLAST_NODETYPE_UNKNOWN = 0,
#include "chill_ast.def"
};

#undef chillast_nodetype
#undef chillast_nodetype_alias

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

//! When tied to another statement
enum CHILL_PREPROCESSING_POSITION {
  CHILL_PREPROCESSING_POSITIONUNKNOWN = 0,
  CHILL_PREPROCESSING_LINEBEFORE,       //!< previous line
  CHILL_PREPROCESSING_LINEAFTER,        //!< next line
  CHILL_PREPROCESSING_TOTHERIGHT,       //!< for this kind of comment, on same line
  CHILL_PREPROCESSING_IMMEDIATELYBEFORE //!< on same line
}; 





extern const char* Chill_AST_Node_Names[];  // WARNING MUST BE KEPT IN SYNC WITH BELOW LIST 


// fwd declarations

//! the generic node. specific types derive from this
class chillAST_node;         

//! empty node
class chillAST_NULL;

//! ast for an entire source file (translationunit)
class chillAST_SourceFile;

//! C++ typedef node
class chillAST_TypedefDecl; 

//! Variable declaration
class chillAST_VarDecl; 

//class chillAST_ParmVarDecl; 

//! Function declaration & definition
class chillAST_FunctionDecl; 

//! structs and unions (and classes?)
class chillAST_RecordDecl;

//! Macro definition
class chillAST_MacroDefinition;

//! A sequence of statements
class chillAST_CompoundStmt;

//! a for loop
class chillAST_ForStmt;

//! a while loop
class chillAST_WhileStmt;

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
typedef std::vector<chillAST_node *>            chillAST_NodeList;      //  typedef

chillAST_VarDecl *symbolTableFindVariableNamed( chillAST_SymbolTable *table, const char *name ); // fwd decl

void printSymbolTable( chillAST_SymbolTable *st ); // fwd decl 
void printSymbolTableMoreInfo( chillAST_SymbolTable *st ); // fwd decl 


chillAST_node           *minmaxTernary( const char * op, chillAST_node *left,  chillAST_node *right);  // fwd declaration
chillAST_SymbolTable    *addSymbolToTable( chillAST_SymbolTable *st, chillAST_VarDecl *vd ); // fwd decl
chillAST_TypedefTable   *addTypedefToTable( chillAST_TypedefTable *tt, chillAST_TypedefDecl *td ); // fwd decl


bool streq( const char *a, const char *b); // fwd decl
void chillindent( int i, FILE *fp );  // fwd declaration  
void insertNewDeclAtLocationOfOldIfNeeded( chillAST_VarDecl *newdecl, chillAST_VarDecl *olddecl); 

// an actual chill ast.
// nodes based on clang AST which are in turn based on C++ 

//! generic node of the actual chillAST, a multiway tree node.
class chillAST_node {
public: 

  //! for manufactured scalars 
  static int chill_scalar_counter;
  //! for manufactured arrays
  static int chill_array_counter;
  //! for manufactured arrays
  static int chill_pointer_counter;

  virtual ~chillAST_node() = default;

  bool isSourceFile()         { return (getType() == CHILLAST_NODETYPE_SOURCEFILE); };
  bool isTypeDefDecl()        { return (getType() == CHILLAST_NODETYPE_TYPEDEFDECL); };
  bool isVarDecl()            { return (getType() == CHILLAST_NODETYPE_VARDECL); };
  bool isFunctionDecl()       { return (getType() == CHILLAST_NODETYPE_FUNCTIONDECL); };
  bool isRecordDecl()         { return (getType() == CHILLAST_NODETYPE_RECORDDECL); };
  bool isMacroDefinition()    { return (getType() == CHILLAST_NODETYPE_MACRODEFINITION); };
  bool isCompoundStmt()       { return (getType() == CHILLAST_NODETYPE_COMPOUNDSTMT); };
  bool isLoop()               { return (getType() == CHILLAST_NODETYPE_LOOP); };    // AKA ForStmt
  bool isForStmt()            { return (getType() == CHILLAST_NODETYPE_LOOP); };    // AKA Loop
  bool isWhileStmt()          { return (getType() == CHILLAST_NODETYPE_WHILESTMT); };
  bool isIfStmt()             { return (getType() == CHILLAST_NODETYPE_IFSTMT); };
  bool isTernaryOperator()    { return (getType() == CHILLAST_NODETYPE_TERNARYOPERATOR);};
  bool isBinaryOperator()     { return (getType() == CHILLAST_NODETYPE_BINARYOPERATOR); };
  bool isUnaryOperator()      { return (getType() == CHILLAST_NODETYPE_UNARYOPERATOR); };
  bool isArraySubscriptExpr() { return (getType() == CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR); };
  bool isMemberExpr()         { return (getType() == CHILLAST_NODETYPE_MEMBEREXPR); };
  bool isDeclRefExpr()        { return (getType() == CHILLAST_NODETYPE_DECLREFEXPR); };
  bool isIntegerLiteral()     { return (getType() == CHILLAST_NODETYPE_INTEGERLITERAL); };
  bool isFloatingLiteral()    { return (getType() == CHILLAST_NODETYPE_FLOATINGLITERAL); };
  bool isImplicitCastExpr()   { return (getType() == CHILLAST_NODETYPE_IMPLICITCASTEXPR); };
  bool isReturnStmt()         { return (getType() == CHILLAST_NODETYPE_RETURNSTMT); };
  bool isCallExpr()           { return (getType() == CHILLAST_NODETYPE_CALLEXPR); };
  bool isParenExpr()          { return (getType() == CHILLAST_NODETYPE_PARENEXPR); };
  bool isSizeof()             { return (getType() == CHILLAST_NODETYPE_SIZEOF); };
  bool isMalloc()             { return (getType() == CHILLAST_NODETYPE_MALLOC); };
  bool isFree()               { return (getType() == CHILLAST_NODETYPE_FREE); };
  bool isPreprocessing()      { return (getType() == CHILLAST_NODETYPE_PREPROCESSING); };
  bool isNoOp()               { return (getType() == CHILLAST_NODETYPE_NOOP); };
  bool isNull()               { return (getType() == CHILLAST_NODETYPE_NULL); };
  bool isCStyleCastExpr()     { return (getType() == CHILLAST_NODETYPE_CSTYLECASTEXPR); };
  bool isCStyleAddressOf()    { return (getType() == CHILLAST_NODETYPE_CSTYLEADDRESSOF); };
  bool isCudaMalloc()         { return (getType() == CHILLAST_NODETYPE_CUDAMALLOC); };
  bool isCudaFree()           { return (getType() == CHILLAST_NODETYPE_CUDAFREE); };
  bool isCudaMemcpy()         { return (getType() == CHILLAST_NODETYPE_CUDAMEMCPY); };
  bool isCudaKERNELCALL()     { return (getType() == CHILLAST_NODETYPE_CUDAKERNELCALL); };
  bool isCudaSYNCTHREADS()    { return (getType() == CHILLAST_NODETYPE_CUDASYNCTHREADS); };
  bool isDeclStmt()           { return (getType() == CHILLAST_NODETYPE_DECLSTMT); }; // doesn't exist
  bool isConstant()           { return (getType() == CHILLAST_NODETYPE_INTEGERLITERAL) || (getType() == CHILLAST_NODETYPE_FLOATINGLITERAL); }
    

  virtual bool isAssignmentOp() { return false; }; 
  virtual bool isComparisonOp() { return false; }; 
  virtual bool isNotLeaf()      { return false; };
  virtual bool isLeaf()         { return true;  };
  virtual bool isParmVarDecl()  { return false; };  

  virtual bool isPlusOp()       { return false; }; 
  virtual bool isMinusOp()      { return false; }; 
  virtual bool isPlusMinusOp()  { return false; }; 
  virtual bool isMultDivOp()    { return false; };
  virtual bool isRemOp()        { return false; };

  virtual bool isAStruct() { return false; }; 
  virtual bool isAUnion()  { return false; };

  virtual bool hasSymbolTable() { return false; } ; // most nodes do NOT have a symbol table
  virtual bool hasTypedefTable() { return false; } ; // most nodes do NOT have a typedef table
  virtual chillAST_SymbolTable *getSymbolTable() { return NULL; } // most nodes do NOT have a symbol table

  virtual chillAST_VarDecl *findVariableNamed( const char *name ); // recursive

  chillAST_RecordDecl *findRecordDeclNamed( const char *name ); // recursive
  
  // TODO decide how to hide some data

  //! this Node's parent
  chillAST_node *parent; 
  //! whether it is from a source file, when false it is from included files
  bool isFromSourceFile;
  //! the name of file this node from
  char *filename;

  int getNumChildren() { return children.size(); }; 
  vector<chillAST_node*> children;
  vector<chillAST_node*> &getChildren() { return children; } ;  // not usually useful
  chillAST_node *getChild( int which)                    { return children[which]; };
  void           setChild( int which, chillAST_node *n ) { children[which] = n; if(n) n->parent = this; } ;
  
  //! for compiler internals, formerly a comment
  char *metacomment;
  void setMetaComment( char *c ) { metacomment = strdup(c); }; 

  vector<chillAST_Preprocessing*> preprocessinginfo; 

  virtual void addChild( chillAST_node* c) {
    c->parent = this;
    // check to see if it's already there
    for (auto i: children)
      if (i == c) {
        debug_printf("addChild(): Child already exist");
        return;
      }
      // assert(true || children[i] != c && "child already exist");
    children.push_back(c);
  }  // not usually useful
  virtual void addChildren( chillAST_NodeList nl ) {
    for (auto i: nl)
      addChild(i);
  }
  virtual void insertChild(int i, chillAST_node* node) {
    node->parent = this;
    children.insert( children.begin()+i, node );
  };
  
  void removeChild(int i) {
    children.erase( children.begin()+i );
  };

  /**
   * @brief prepend a statement to the begining of a block, the body of a loop, or the body of a function
   */
  virtual void prependStatement(chillAST_node* stmt) {
    __throw_runtime_error(std::string("Not implemented for ") + getTypeString());
  }
  /**
   * @brief append a statement to the end of a block, the body of a loop, or the body of a function
   */
  virtual void appendStatement(chillAST_node* stmt) {
    __throw_runtime_error(std::string("Not implemented for ") + getTypeString());
  }

  int findChild(  chillAST_node *c )  {   
    for (int i=0; i<children.size(); i++) { 
      if (children[i] == c) return i;
    }
    return -1;
  }

  virtual void replaceChild( chillAST_node *old, chillAST_node *newchild ) {
    int pos = findChild(old);
    if (pos >= 0) setChild(pos,newchild);
    else debug_fprintf(stderr, "Replacing a non-child");
  };

  /**
   * @brief Replace the loop with the loop body, if loop is with this variable.
   *
   * The loop will be spread across a bunch of cores that will each
   * calculate their own loop variable.
   *
   * @param var
   */
  virtual void loseLoopWithLoopVar( char *var ) { 
    // things that can not have loops as substatements should have a null version of this method
    
    __throw_runtime_error(std::string("looseLoopWithLoopVar called on node of type") + this->getTypeString());
  }

  virtual int evalAsInt() {
    __throw_runtime_error(std::string("Not implemented for ") + getTypeString());
  }

  virtual const char* getUnderlyingType() {
    __throw_runtime_error(std::string("Not implemented for ") + getTypeString());
  };

  virtual chillAST_VarDecl* getUnderlyingVarDecl() {
    __throw_runtime_error(std::string("Not implemented for ") + getTypeString());
  };

  virtual void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {
    debug_fprintf(stderr,"(%s) forgot to implement gatherArrayRefs()\n" ,Chill_AST_Node_Names[getType()]);
    dump();
    print();
    debug_fprintf(stderr, "\n\n"); 
  };
 
  // TODO we MIGHT want the VarDecl // NOTHING IMPLEMENTS THIS? ??? 
  virtual void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
    debug_fprintf(stderr,"(%s) forgot to implement gatherScalarRefs()\n" ,Chill_AST_Node_Names[getType()]);
    dump();
    print();
    debug_fprintf(stderr, "\n\n"); 
  };
 
  //! recursively walking parent links, looking for loops and grabbing the declRefExpr in the loop init and cond
  virtual void gatherLoopIndeces( std::vector<chillAST_VarDecl*> &indeces ) {
    // you can quit when you get to certain nodes
    if (isSourceFile() || isFunctionDecl() ) return; // end of the line

    if (!parent) return; // should not happen, but be careful

    // for most nodes, this just recurses upwards
    parent->gatherLoopIndeces( indeces );
  }


  //! recursively walking parent links, looking for loops
  chillAST_ForStmt* findContainingLoop() {
    // do not check SELF type, as we may want to find the loop containing a loop
    if (!parent) return nullptr;
    if (parent->isForStmt()) return (chillAST_ForStmt*)parent;
    return parent->findContainingLoop(); // recurse upwards
  }

  //! recursively walking parent links, avoiding loops
  chillAST_node* findContainingNonLoop() {
    debug_fprintf(stderr, "%s::findContainingNonLoop()   ", getTypeString());
    //if (parent) debug_fprintf(stderr, "parent is a %s\n", parent->getTypeString()); 
    //else debug_fprintf(stderr, "no parent\n"); 
    // do not check SELF type, as we may want to find the loop containing a loop
    if (!parent) return nullptr;
    if (parent->isCompoundStmt() && parent->getParent()->isForStmt()) return parent->getParent()->findContainingNonLoop(); // keep recursing
    if (parent->isForStmt()) return parent->findContainingNonLoop(); // keep recursing
    return (chillAST_node*)parent; // return non-loop 
  }

  // TODO gather loop init and cond (and if cond) like gatherloopindeces
  //! gather both scalar and array references
  virtual void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ){
    debug_fprintf(stderr,"(%s) forgot to implement gatherDeclRefExpr()\n" ,Chill_AST_Node_Names[getType()]);
    for (int i = 0; i<getNumChildren(); ++i)
      if (getChild(i)) getChild(i)->gatherDeclRefExprs(refs);
  };

  virtual void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) { 
    debug_fprintf(stderr,"(%s) forgot to implement gatherVarUsage()\n" ,Chill_AST_Node_Names[getType()]);
    for (int i = 0; i<getNumChildren(); ++i)
      if (getChild(i)) getChild(i)->gatherVarUsage(decls);
  };

  //! gather all variable that is used as a lefthand side operand
  virtual void gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls ) { 
    debug_fprintf(stderr,"(%s) forgot to implement gatherVarLHSUsage()\n" ,Chill_AST_Node_Names[getType()]);
  }; 

  //! gather ACTUAL variable declarations
  virtual void gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
    debug_fprintf(stderr,"(%s) uses default gatherVarDecls()\n" ,Chill_AST_Node_Names[getType()]);
    for (int i = 0; i<getNumChildren(); ++i)
      if (getChild(i)) getChild(i)->gatherVarDecls(decls);
  };

  virtual void gatherVarDeclsMore( vector<chillAST_VarDecl*> &decls ) {  // even if the decl itself is not in the ast. 
    debug_fprintf(stderr,"(%s) forgot to implement gatherVarDeclsMore()\n" ,Chill_AST_Node_Names[getType()]);
  };

  //! gather ACTUAL scalar variable declarations
  virtual void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
    debug_fprintf(stderr,"(%s) forgot to implement gatherScalarVarDecls()\n" ,Chill_AST_Node_Names[getType()]);
    for (int i = 0; i<getNumChildren(); ++i)
      if (getChild(i)) getChild(i)->gatherScalarVarDecls(decls);
  };

  //! gather ACTUAL array variable declarations
  virtual void gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
    debug_fprintf(stderr,"(%s) forgot to implement gatherArrayVarDecls()\n" ,Chill_AST_Node_Names[getType()]);
    for (int i = 0; i<getNumChildren(); ++i)
      if (getChild(i)) getChild(i)->gatherArrayVarDecls(decls);
  };

  virtual chillAST_VarDecl *findArrayDecl( const char *name ) { // scoping TODO 
    if (!hasSymbolTable()) return parent->findArrayDecl( name ); // most things
    debug_fprintf(stderr,"(%s) forgot to implement gatherArrayVarDecls()\n" ,Chill_AST_Node_Names[getType()]);
    return nullptr;
  }


  virtual void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
    debug_fprintf(stderr,"(%s) forgot to implement replaceVarDecls()\n" ,Chill_AST_Node_Names[getType()]);
  }; 

  //! this just looks for ForStmts with preferred index metacomment attached 
  virtual bool findLoopIndexesToReplace( chillAST_SymbolTable *symtab, bool forcesync=false ) { 
    debug_fprintf(stderr,"(%s) forgot to implement findLoopIndexesToReplace()\n" ,Chill_AST_Node_Names[getType()]);
    return false; 
  }

 
  //! Folding constant, to some degree
  /*!
   * We should need to delegate this to the backend compiler
   * @return This node
   */
  virtual chillAST_node* constantFold() {  // hacky. TODO. make nice
    __throw_runtime_error(std::string("Not implemented for ") + getTypeString());
  };

  virtual chillAST_node* clone() {   // makes a deep COPY (?)
    __throw_runtime_error(std::string("Not implemented for ") + getTypeString());
  };

  void dump( int indent=0,  std::ostream& o = std::cerr );

  void print( int indent=0,  std::ostream& o = std::cerr );

  void dump( int indent,  FILE *fp ) __attribute_deprecated__ {
    if (fp == stderr)
      dump(indent, std::cerr);
    else if(fp == stdout)
      dump(indent, std::cout);
    else
      chill_error_printf("Printing to somewhere other than stderr/stdout using deprecated print");
  }

  void print( int indent,  FILE *fp ) __attribute_deprecated__ {
    if (fp == stderr)
      print(indent, std::cerr);
    else if(fp == stdout)
      print(indent, std::cout);
    else
      chill_error_printf("Printing to somewhere other than stderr/stdout using deprecated print");
  }

  //! the ast's print version
  virtual std::string stringRep(int indent=0 );


  virtual void get_top_level_loops( std::vector<chillAST_ForStmt *> &loops) {
    for (auto c: children) {
      if (c->isForStmt()) {
        loops.push_back( (chillAST_ForStmt *)c );
      }
    }
  }

  virtual void repairParentChild() {  // for nodes where all subnodes are children
    for (auto c: children) {
      if (c->parent != this) {
        debug_fprintf(stderr, "fixing child %s that didn't know its parent\n", c->getTypeString());
        c->parent = this;
      }
    }
  }

  //! generic for chillAST_node with children
  virtual void find_deepest_loops( std::vector<chillAST_ForStmt *> &loops) { // returns DEEPEST nesting of loops 
    // TODO hide implementation
    std::vector<chillAST_ForStmt *>deepest; // deepest below here 
    
    for (auto i: children) {
      std::vector<chillAST_ForStmt *> subloops;  // loops below here among a child of mine 
      
      i->find_deepest_loops( subloops );
      
      if (subloops.size() > deepest.size()) { 
        deepest = subloops;
      }
    }
    
    // append deepest we see at this level to loops
    std::copy(deepest.begin(), deepest.end(), std::back_inserter(loops));
  }

  const char *getTypeString() { return Chill_AST_Node_Names[getType()]; } ;
  virtual CHILL_ASTNODE_TYPE getType() { return CHILLAST_NODETYPE_UNKNOWN; };
  void setParent( chillAST_node *p) { parent = p; } ;
  chillAST_node  *getParent() { return parent; } ;
  
  chillAST_SourceFile *getSourceFile() { 
    if (isSourceFile()) return ((chillAST_SourceFile *)this);
    if (parent != nullptr) return parent->getSourceFile();
    this->print(); fprintf(stderr, "\n\n");
    __throw_runtime_error(std::string("Can't get sourcefile from ") + this->getTypeString());
  }
  
  virtual chillAST_node *findDatatype( char *t ) { 
    if (parent != nullptr) return parent->findDatatype(t); // most nodes do this
    return nullptr;
  }


  virtual chillAST_SymbolTable *addVariableToSymbolTable( chillAST_VarDecl *vd ) { 
    if (!parent)
      __throw_runtime_error(std::string("Without a parent, can't add to symbol-table from ") + this->getTypeString());
    return parent->addVariableToSymbolTable( vd ); // default, defer to parent
  }

  virtual void addTypedefToTypedefTable( chillAST_TypedefDecl *tdd ) { 
    parent->addTypedefToTypedefTable( tdd ); // default, defer to parent 
  }

  virtual chillAST_node *getEnclosingStatement();
   /**
   * @brief Find the base declaration that this node refers to
   *
   * This will step through:
   *    * ArraySubscriptExpression
   *    * MemberExpression
   */
  virtual chillAST_VarDecl *multibase() {
     __throw_runtime_error(std::string("Not implemented for ") + getTypeString());
  }

  //! Get a vector of statements
  virtual void gatherStatements( std::vector<chillAST_node*> &statements ) { 
    debug_fprintf(stderr,"(%s) forgot to implement gatherStatements()\n" ,Chill_AST_Node_Names[getType()]);
    dump();fflush(stdout); 
    print();
    debug_fprintf(stderr, "\n\n"); 
  }


  virtual bool isSameAs( chillAST_node *other ) {  // for tree comparison
    debug_fprintf(stderr,"(%s) forgot to implement isSameAs()\n" ,Chill_AST_Node_Names[getType()]);
    dump(); fflush(stdout);
    print();
    debug_fprintf(stderr, "\n\n");
    return true;
  }

  //! Base constructor for all inherited class
  chillAST_node() {
    parent = NULL;
    metacomment = NULL;
    isFromSourceFile = true; // default
    filename = NULL;
  }

  template<typename ASTDestClass>
  ASTDestClass* as() {
      return dynamic_cast<ASTDestClass*>(this);
  }
};

/**
 * \brief Generic class to handle children in chillAST nodes
 *
 * Storing the positional information to look up the child in Children.
 */
template<typename ASTNodeClass>
class chillAST_Child {
private:
  chillAST_node* _parent;
  int _pos;

  ASTNodeClass* get() const {
    return dynamic_cast<ASTNodeClass*>(_parent->getChild(_pos));
  }
public:
  chillAST_Child(chillAST_node* parent,int pos):_parent(parent),_pos(pos) {
    while (parent->getNumChildren() <= pos)
      parent->children.push_back(NULL);
    // This will ensure that ASTNodeClass is a subclass of chillAST_node
    parent->setChild(pos, NULL);
  }

  chillAST_Child(const chillAST_Child<ASTNodeClass>&) = delete;
  chillAST_Child(const chillAST_Child<ASTNodeClass>&&) = delete;

  //! Assignment operator will set the child
  ASTNodeClass* operator=(ASTNodeClass *ptr) {
    _parent->setChild(_pos, ptr);
    return ptr;
  }

  //! Assignment operator from another child of the same type
  ASTNodeClass* operator=(const chillAST_Child<ASTNodeClass>& other) {
      _parent->setChild(_pos, (ASTNodeClass*)other);
      return (ASTNodeClass*)other;
  }

  //! Equality operator for "not null" and "null" checks
  bool operator==(std::nullptr_t) { return !get(); }

  //! Behaving like a pointer with arrow operator
  ASTNodeClass* operator ->() const { return get(); }

  //! Implicit conversion to the default type, or base of default type
  template<typename DestASTNodeClass,
           typename std::enable_if<std::is_base_of<chillAST_node, DestASTNodeClass>::value, int>::type = 0>
  operator DestASTNodeClass* () const {
      return dynamic_cast<DestASTNodeClass*>(get());
  }

  //! Boolean conversion for "not null" and "null" checks
  operator bool () const {
      return get() != nullptr;
  }

  //! Dereferencing this pointer
  ASTNodeClass& operator *() const { return *get(); }
};

class chillAST_NULL: public chillAST_node {  // NOOP?
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_NULL;}
};


class chillAST_Preprocessing: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_PREPROCESSING;}
  // variables that are special for this type of node
  CHILL_PREPROCESSING_POSITION position;
  CHILL_PREPROCESSING_TYPE pptype;
  char *blurb;

  // constructors
  chillAST_Preprocessing(); // not sure what this is good for
  chillAST_Preprocessing( CHILL_PREPROCESSING_POSITION pos, CHILL_PREPROCESSING_TYPE t, char *text );
  
  // other methods particular to this type of node
  
};


//typedef is a keyword in the C and C++ programming languages. The purpose of typedef is to assign alternative names to existing types, most often those whose standard declaration is cumbersome, potentially confusing, or likely to vary from one implementation to another. 
class chillAST_TypedefDecl: public chillAST_node { 
private:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_TYPEDEFDECL;}
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


  bool isAStruct() override { return isStruct; };
  bool isAUnion()  override { return isUnion;  };
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
  chillAST_TypedefDecl(const char *t, const char *nt);
  chillAST_TypedefDecl(const char *t, const char *nt, const char *a);
  
  const char* getUnderlyingType() override { debug_fprintf(stderr, "TypedefDecl getUnderLyingType()\n"); return underlyingtype; };
};


class chillAST_VarDecl: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_VARDECL;}
  char *vartype; //!< interchangabe with underlying type

  chillAST_RecordDecl  *vardef;// the thing that says what the struct looks like
  chillAST_TypedefDecl *typedefinition; // NULL for float, int, etc.
  chillAST_RecordDecl * getStructDef(); // TODO make vardef private?

  //bool insideAStruct;  // this variable is itself part of a struct
  
  char *underlyingtype;   //!< the base type of the variable
  char *varname;          //!< Variable name
  char *arraypointerpart; //!< Pointer part of the array such as '***'
  char *arraysetpart; 

  int numdimensions;      //!< The total number of dimensions, some might be unbounded as specified in '**'
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
  bool isAStruct() override { return (isStruct || (typedefinition && typedefinition->isAStruct())); }
  void setStruct( bool b ) {isStruct = b;/*debug_fprintf(stderr,"vardecl %s IS A STRUCT\n",varname);*/ };
  bool isPointer() { return numdimensions > getNumChildren(); }  //

  chillAST_node *init;
  void setInit( chillAST_node *i ) { init = i; i->setParent(this); };
  bool hasInit() { return init != nullptr; };
  chillAST_node *getInit() { return init; };
  int            getArrayDimensions()                  { return this->getChildren().size(); }
  chillAST_node *getArraySize(int i)                   { return this->getChild(i); }
  int            getArraySizeAsInt(int i)              { return this->getArraySize(i)->evalAsInt(); }
  void           setArraySize(int i, chillAST_node* s) { this->setChild(i, s); }
  void           convertArrayToPointer();
  
  chillAST_VarDecl();
  /**
   * @brief Base constructor for VarDecl
   * @param t the base type, such as "int"
   * @param ap the array pointer part, such as "**"
   * @param n the variable name
   * @param arraypart the explicit array sizes as a vector
   * @param ptr Unique pointer
   */
  chillAST_VarDecl( const char *t, const char *ap,  const char *n, chillAST_NodeList arraypart = chillAST_NodeList(), void *ptr = nullptr);
  chillAST_VarDecl( chillAST_TypedefDecl *tdd, const char *ap, const char *n, chillAST_NodeList arraypart = chillAST_NodeList());
  chillAST_VarDecl( chillAST_RecordDecl *astruct, const char *ap, const char *n, chillAST_NodeList arraypart = chillAST_NodeList());

  bool isParmVarDecl() override { return( isAParameter == 1 ); };
  bool isBuiltin()     { return( isABuiltin == 1 ); };  // designate variable as a builtin
  void setLocation( void *ptr ) { uniquePtr = ptr; } ;


  void gatherVarDecls(vector<chillAST_VarDecl *> &decls) override;
  void gatherVarDeclsMore(vector<chillAST_VarDecl *> &decls) override { gatherVarDecls(decls); };
  void gatherScalarVarDecls(vector<chillAST_VarDecl *> &decls) override;
  void gatherArrayVarDecls(vector<chillAST_VarDecl *> &decls) override;
  void gatherVarUsage(vector<chillAST_VarDecl *> &decls) override { if (init) init->gatherVarUsage(decls); };
  void gatherDeclRefExprs(vector<chillAST_DeclRefExpr *> &refs) override { if (init) init->gatherDeclRefExprs(refs); };

  void replaceVarDecls(chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override {
    init->replaceVarDecls(olddecl, newdecl);
  };

  bool findLoopIndexesToReplace(chillAST_SymbolTable *symtab, bool forcesync = false) override { return false; };
  const char* getUnderlyingType() override { return underlyingtype; };
  chillAST_VarDecl* getUnderlyingVarDecl() override { return this; };

  chillAST_node* constantFold() override;
  chillAST_node* clone() override;

  void loseLoopWithLoopVar(char *var ) override;

};


class chillAST_DeclRefExpr: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_DECLREFEXPR;}
  // variables that are special for this type of node
  char *declarationType; 
  char *declarationName; 
  chillAST_node *decl; // the declaration of this variable or function ... uhoh
  //char *functionparameters;  // TODO probably should split this node into 2 types, one for variables, one for functions

  // constructors
  chillAST_DeclRefExpr(); 
  explicit chillAST_DeclRefExpr( const char *variablename);
  chillAST_DeclRefExpr( const char *vartype, const char *variablename);
  chillAST_DeclRefExpr( const char *vartype, const char *variablename, chillAST_node *dec);
  explicit chillAST_DeclRefExpr( chillAST_node *d);

  // other methods particular to this type of node
  bool operator!=( chillAST_DeclRefExpr &other ) { return decl != other.decl ; }; 
  bool operator==( chillAST_DeclRefExpr &other ) { return decl == other.decl ; }; // EXACT SAME VARECL BY ADDRESS
  
  chillAST_node *getDecl() { return decl; };

  chillAST_VarDecl *getVarDecl() { 
    if (!decl) return nullptr; // should never happen
    if (decl->isVarDecl()) return (chillAST_VarDecl *)decl;
    return nullptr;
  }; 
  
  chillAST_FunctionDecl *getFunctionDecl() { 
    if (!decl) return nullptr; // should never happen
    if (decl->isFunctionDecl()) return (chillAST_FunctionDecl *)decl;
    return nullptr;
  }; 
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override {}; // do nothing
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  // this is the AST node where these 2 differ 
  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) override {};  // does nothing, to get the cvardecl using this method, the actual vardecl must be in the AST
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override; // returns the decl this declrefexpr references, even if the decl is not in the AST


  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) override;
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) override;

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) override;
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) override;
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

  const char* getUnderlyingType() override {
    debug_fprintf(stderr, "DeclRefExpr getUnderLyingType()\n");
    return decl->getUnderlyingType();
  };

  chillAST_VarDecl* getUnderlyingVarDecl() override { return decl->getUnderlyingVarDecl(); } // functions?? TODO

  chillAST_VarDecl* multibase() override;
  chillAST_node *multibase2() { return (chillAST_node *)multibase(); } 
}; 





class chillAST_CompoundStmt: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_COMPOUNDSTMT;}
  // variables that are special for this type of node
  chillAST_SymbolTable  *symbol_table;  // symbols defined inside this compound statement 
  chillAST_TypedefTable *typedef_table;

  bool hasSymbolTable() override { return true; } ;
  bool hasTypeDefTable() { return true; } ;
  chillAST_node *findDatatype( char *t ) override {
    debug_fprintf(stderr, "chillAST_CompoundStmt::findDatatype( %s )\n", t); 
    if (typedef_table) { 
      for (int i=0; i< typedef_table->size(); i++)  {
        chillAST_TypedefDecl *tdd = (*typedef_table)[i];
        if (tdd->nameis( t )) return tdd;
      }
    }
    if (parent) return parent->findDatatype(t);
    return nullptr; // should not happen
  }

  chillAST_SymbolTable *getSymbolTable() override { return symbol_table; }

  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) override {   // chillAST_CompoundStmt method
    symbol_table = addSymbolToTable( symbol_table, vd );
    return symbol_table;
  }

  void addTypedefToTypedefTable( chillAST_TypedefDecl *tdd ) override {
    typedef_table = addTypedefToTable( typedef_table, tdd );
  }

  // constructors
  chillAST_CompoundStmt(); // never has any args ???

  // other methods particular to this type of node
  
  
  // required methods 
  void replaceChild( chillAST_node *old, chillAST_node *newchild ) override;
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override;
  void loseLoopWithLoopVar( char *var ) override;

  void gatherStatements( std::vector<chillAST_node*> &statements ) override;
}; 




class chillAST_RecordDecl: public chillAST_node {  // declaration of the shape of a struct or union 
private:
  virtual CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_RECORDDECL;}
  char *name;  // could be NULL? for unnamed structs?
  char *originalname; 
  bool isStruct;

  bool isUnion;
  vector<chillAST_VarDecl *> subparts;
  
public:
  chillAST_RecordDecl();
  explicit chillAST_RecordDecl( const char *nam);
  chillAST_RecordDecl( const char *nam, const char *orig);

  void  setName( const char *newname) { name = strdup(newname); }; 
  char *getName( ) { return name; };
  vector<chillAST_VarDecl *> &getSubparts() {return subparts;}
  
  bool isAUnion()  override { return isUnion;  };
  bool isAStruct() override { return isStruct; };

  void setStruct(bool tf) { isStruct = tf; }; 
  //debug_fprintf(stderr, "%s isStruct %d\n", structname, isStruct);  }; 
  void setUnion( bool tf) { isUnion  = tf; };

  chillAST_SymbolTable *addVariableToSymbolTable( chillAST_VarDecl *vd ) override; //  RecordDecl does NOTHING
  
  int numSubparts() { return subparts.size(); }; 
  void addSubpart( chillAST_VarDecl *s ) { subparts.push_back(s); }; 
  chillAST_VarDecl *findSubpart( const char *name );
  chillAST_VarDecl *findSubpartByType( const char *typ );

  void printStructure( int indent=0,  FILE *fp = stderr ) ;
};




class chillAST_FunctionDecl: public chillAST_node { 
private:
  virtual CHILL_ASTNODE_TYPE getType() {return CHILLAST_NODETYPE_FUNCTIONDECL;}
  chillAST_Child<chillAST_CompoundStmt> body; // always a compound statement?
  CHILL_FUNCTION_TYPE function_type;  // CHILL_FUNCTION_CPU or  CHILL_FUNCTION_GPU
  bool externfunc;   // function is external 
  bool builtin;      // function is a builtin
  bool forwarddecl;  // function is a forward declaration

public:
  char *returnType;
  char *functionName;

  // parameters
  int numParameters() { return parameters.size(); } ; 
  chillAST_SymbolTable parameters;

  chillAST_TypedefTable *typedef_table; // function typedef table

  bool hasSymbolTable() override { return true; } ; // COULD HAVE
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

  
  

  chillAST_FunctionDecl(); //  { getType() = CHILLAST_NODETYPE_FUNCTIONDECL; numparameters = 0;};
  chillAST_FunctionDecl(const char *rt, const char *fname ) ;
  chillAST_FunctionDecl(const char *rt, const char *fname, void *unique ) ;
  
  void addParameter( chillAST_VarDecl *p); 
  chillAST_VarDecl *hasParameterNamed( const char *name ); 
  chillAST_VarDecl *findParameterNamed( const char *name ) { return hasParameterNamed( name ); }; 

  void addDecl( chillAST_VarDecl *vd);  // just adds to symbol table?? TODO 

  chillAST_VarDecl *funcHasVariableNamed( const char *name );  // functiondecl::hasVariableNamed
  //chillAST_VarDecl *findVariableNamed( const char *name ) { return hasVariableNamed( name ); }; 

  void addChild(chillAST_node* node) override {assert(false && "Function declaration has no child");} // special because inserts into BODY
  void insertChild(int i, chillAST_node* node) override {assert(false && "Function declaration has no child");} // special because inserts into BODY

  void setBody( chillAST_node * bod );  
  chillAST_CompoundStmt *getBody() { return body; }
  
  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) override;
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) override;
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) override;
  chillAST_VarDecl *findArrayDecl( const char *name ) override;

  void cleanUpVarDecls();

  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override;

  chillAST_node* constantFold() override;

  chillAST_node *findDatatype( char *t ) override {
    //debug_fprintf(stderr, "%s looking for datatype %s\n", getTypeString(), t); 
    if (!typedef_table) { // not here
      if (parent) return parent->findDatatype(t); // not here, check parents
      else return nullptr; // not defined here and no parent
    }
    
    //debug_fprintf(stderr, "%d typedefs\n", typedef_table->size());
    for (auto tdd: *typedef_table)
      if ( tdd->nameis( t )) return tdd;
    if (parent) return parent->findDatatype(t);
    return nullptr; // should not happen
  }

  chillAST_SymbolTable *getParameterSymbolTable() { return &parameters; }
  chillAST_SymbolTable *getSymbolTable() override { return body->getSymbolTable(); }
  void setSymbolTable( chillAST_SymbolTable *tab ) {
    // no longer keeping a local ?? symbol_table = tab;
    if (!body) { // can never happen now
      body = new chillAST_CompoundStmt();
    } // only if func is empty!
    body->symbol_table = tab;
  }

  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) override {
    //debug_fprintf(stderr, "\nchillAST_FunctionDecl addVariableToSymbolTable( %s )\n", vd->varname);
    
    // this is all dealing with the body's symbol table
    // the function has a symbol table called "parameters" but that is a special case

    addSymbolToTable( getSymbolTable(), vd ); 
    return getSymbolTable();
  }


  void addTypedefToTypedefTable( chillAST_TypedefDecl *tdd ) override {
    typedef_table = addTypedefToTable( typedef_table, tdd );
  }

  void replaceChild( chillAST_node *old, chillAST_node *newchild ) override {
    body->replaceChild( old, newchild ); 
  }

  void prependStatement(chillAST_node* stmt) override {
      body->insertChild(0, stmt);
  }

  void appendStatement(chillAST_node* stmt) override {
      body->addChild(stmt);
  }
};  // end FunctionDecl 




class chillAST_SourceFile: public chillAST_node { 
public:
  virtual CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_SOURCEFILE;}

  // constructors
  chillAST_SourceFile();                       //  defined in chill_ast.cc 
  explicit chillAST_SourceFile(const char *filename );  //  defined in chill_ast.cc
  
  ~chillAST_SourceFile() override;                       //  defined in chill_ast.cc

  void printToFile( char *filename = nullptr );
  
  char *SourceFileName; // where this originated
  char *FileToWrite; 
  char *frontend;

  void setFileToWrite( char *f ) { FileToWrite =  strdup( f ); }; 
  
  void setFrontend( const char *compiler ) { if (frontend) free(frontend); frontend = strdup(compiler); } 
  // get, set filename ? 

  chillAST_SymbolTable  *global_symbol_table;  // (global) symbols defined inside this source file 
  chillAST_TypedefTable *global_typedef_table; // source file 
  chillAST_VarDecl *findVariableNamed( const char *name ) override; // looks in global_symbol_table;

  bool hasSymbolTable() override { return true; } ;  // "has" vs "can have"    TODO
  bool hasTypeDefTable() { return true; } ;

  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) override {
    global_symbol_table = addSymbolToTable( global_symbol_table, vd );
    return global_symbol_table;
  }

  void addTypedefToTypedefTable( chillAST_TypedefDecl *tdd ) override {
    assert(global_typedef_table != nullptr);
    global_typedef_table = addTypedefToTable( global_typedef_table, tdd );
  }

  chillAST_node *findDatatype( char *type_name ) override {
    // Look for name in global typedefs
    assert(this->global_typedef_table != nullptr);
    for (chillAST_TypedefDecl* tdd: *global_typedef_table)
      if (tdd->nameis( type_name ))
        return static_cast<chillAST_node *>(tdd);
    return nullptr;
  }

  vector< chillAST_FunctionDecl *>     functions;  // at top level, or anywhere?
  vector< chillAST_MacroDefinition *>  macrodefinitions;

  chillAST_MacroDefinition* findMacro( const char *name ); // TODO ignores arguments
  chillAST_FunctionDecl *findFunction( const char *name ); // TODO ignores arguments
  chillAST_node *findCall( const char *name ); 
  void addChild(chillAST_node *n) override {
    if (n->isMacroDefinition()) {
      insertChild(0,n);
      macrodefinitions.push_back((chillAST_MacroDefinition*)n);
    } else {
      if (n->isFunctionDecl()) {
        bool already = false;
        for (auto func: functions)
          if (func == n)
            already = true;
        if (!already) functions.push_back((chillAST_FunctionDecl*)n);
      }
      chillAST_node::addChild(n);
    }
  }
};

class chillAST_MacroDefinition: public chillAST_node {
private:
  chillAST_node *body; // rhs      always a compound statement? 
  chillAST_SymbolTable *symbol_table;
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_MACRODEFINITION;}
  char *macroName;
  char *rhsString; 

  // parameters - these will be odd, in that they HAVE NO TYPE
  int numParameters() { return parameters.size(); } ; 
  std::vector<chillAST_VarDecl *>parameters;
  
  void setName( char *n ) { macroName = strdup(n); /* probable memory leak */ }; 
  void setRhsString( char *n ) { rhsString = strdup(n); /* probable memory leak */ }; 
  char *getRhsString() { return rhsString; }

  chillAST_MacroDefinition();
  explicit chillAST_MacroDefinition( const char *name);
  chillAST_MacroDefinition( const char *name, const char *rhs);
  
  void addParameter( chillAST_VarDecl *p);  // parameters have no TYPE ??
  chillAST_VarDecl *hasParameterNamed( const char *name ); 
  chillAST_VarDecl *findParameterNamed( const char *name ) { return hasParameterNamed( name ); };
  void addChild(chillAST_node* node) override; // special because inserts into BODY
  void insertChild(int i, chillAST_node* node) override; // special because inserts into BODY
  
  void setBody( chillAST_node * bod );  
  chillAST_node *getBody() { return( body); }
  
  bool hasSymbolTable() override { return true; };

  //const std::vector<chillAST_VarDecl *> getSymbolTable() { return symbol_table; }
  chillAST_SymbolTable *getSymbolTable() override { return symbol_table; }
  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) override {  // chillAST_MacroDefinition method  ??
    //debug_fprintf(stderr, "\nchillAST_MacroDefinition addVariableToSymbolTable( %s )\n", vd->varname);
    symbol_table = addSymbolToTable( symbol_table, vd ); 
    //printSymbolTable(  symbol_table );
    return symbol_table;
  }


  chillAST_node* clone() override;

  // none of these make sense for macros 
  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) override {};
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) override {};
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) override {};
  chillAST_VarDecl *findArrayDecl( const char *name ) override { return nullptr; };
  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) override {};
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) override {};
  void cleanUpVarDecls();
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override {};
  chillAST_node* constantFold() override {};
};
  
  
  



class chillAST_ForStmt: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_LOOP;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> init,cond,incr,body;
  // FIXME: Should not be the responsibility of this
  IR_CONDITION_TYPE conditionoperator;  // from ir_code.hh
  char* pragma;

  chillAST_SymbolTable *symbol_table; // symbols defined inside this forstmt (in init but not body?) body is compound stmt 
  bool hasSymbolTable() override { return true; } ;

  // constructors
  chillAST_ForStmt();
  chillAST_ForStmt(chillAST_node *ini, chillAST_node *con, chillAST_node *inc, chillAST_node *bod);
  
  // other methods particular to this type of node
  void addSyncs();
  void removeSyncComment();
  // TODO: deprecating
  chillAST_node *getInit() { return init; };
  chillAST_node *getCond() { return cond; };
  chillAST_node *getInc()  { return incr; };
  chillAST_node *getBody() { return body; };
  void setBody( chillAST_node *b ) { body = b;  b->parent = this; };
  
  bool isNotLeaf() override { return true; };
  bool isLeaf()    override { return false; };

  
  // required methods that I can't seem to get to inherit
  void printControl( int indent=0,  FILE *fp = stderr );  // print just for ( ... ) but not body

  chillAST_node* constantFold() override;
  chillAST_node* clone() override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override; // will get called on inner loops
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override;

  void gatherLoopIndeces( std::vector<chillAST_VarDecl*> &indeces ) override;
  void gatherLoopVars(  std::vector<std::string> &loopvars );  // gather as strings ??

  void find_deepest_loops( std::vector<chillAST_ForStmt *> &loops) override {
    std::vector<chillAST_ForStmt *> b; // deepest loops below me

    for (auto c: body->children) {
      std::vector<chillAST_ForStmt *> l; // deepest loops below one child
      c->find_deepest_loops( l );
      if ( l.size() > b.size() ) // a deeper nesting than we've seen
        b = l;
    }

    loops.push_back( this ); // add myself
    for (auto l: b) loops.push_back(l);
  }


  void loseLoopWithLoopVar( char *var ) override; // chillAST_ForStmt
  void replaceChild( chillAST_node *old, chillAST_node *newchild ) override;

  chillAST_SymbolTable* addVariableToSymbolTable( chillAST_VarDecl *vd ) override {   // chillAST_ForStmt method
    //debug_fprintf(stderr, "\nchillAST_ForStmt addVariableToSymbolTable( %s )\n", vd->varname);
    symbol_table = addSymbolToTable( symbol_table, vd ); 
    //printSymbolTable(  symbol_table );
    return symbol_table;
  }

  void gatherStatements( std::vector<chillAST_node*> &statements ) override;
  bool lowerBound( int &l ); 
  bool upperBound( int &u );

  void prependStatement(chillAST_node* stmt) override {
      body->insertChild(0, stmt);
  }

  void appendStatement(chillAST_node* stmt) override {
      body->addChild(stmt);
  }

}; 


class chillAST_WhileStmt: public chillAST_node {
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_WHILESTMT;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> cond, body;

  // constructors
  chillAST_WhileStmt():cond(this, 0), body(this, 1) {};
  chillAST_WhileStmt(chillAST_node *cond, chillAST_node *body);

  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override {return this;}
  chillAST_node* clone() override;

  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here
  void loseLoopWithLoopVar( char *var ) override {};

  void prependStatement(chillAST_node* stmt) override {
      body->insertChild(0, stmt);
  }
  void appendStatement(chillAST_node* stmt) override {
      body->addChild(stmt);
  }

};


class chillAST_TernaryOperator: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_TERNARYOPERATOR;}
  // variables that are special for this type of node
  // TODO need enum  so far, only "?" conditional operator
  char *op;
  chillAST_Child<chillAST_node> condition, lhs, rhs;

  // constructors
  chillAST_TernaryOperator();
  chillAST_TernaryOperator(const char *op, chillAST_node *cond, chillAST_node *lhs, chillAST_node *rhs);
  
  // other methods particular to this type of node
  bool isNotLeaf() override { return true; };
  bool isLeaf()    override { return false; };
  
  
  char          *getOp()  { return op; };  // dangerous. could get changed!
  chillAST_node *getCond() { return condition; }; 
  chillAST_node *getRHS() { return rhs; }; 
  chillAST_node *getLHS() { return lhs; };

  void setCond( chillAST_node *newc ) { condition = newc;  newc->setParent( this ); } 
  void setLHS( chillAST_node *newlhs ) { lhs = newlhs;  newlhs->setParent( this ); } 
  void setRHS( chillAST_node *newrhs ) { rhs = newrhs;  newrhs->setParent( this ); } 

  
  
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void replaceChild( chillAST_node *old, chillAST_node *newchild ) override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  void gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls ) override;
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here
  void loseLoopWithLoopVar( char *var ) override {}; // ternop can't have loop as child
};



class chillAST_BinaryOperator: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_BINARYOPERATOR;}
  // variables that are special for this type of node
  char *op;            // TODO need enum
  chillAST_Child<chillAST_node> lhs;
  chillAST_Child<chillAST_node> rhs;
  
  
  // constructors
  chillAST_BinaryOperator();
  chillAST_BinaryOperator(chillAST_node *lhs, const char *op, chillAST_node *rhs);
  
  // other methods particular to this type of node
  int evalAsInt() override;

  bool isNotLeaf() override { return true; };
  bool isLeaf()    override { return false; };
  
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
  bool isAssignmentOp() override {
    return( (!strcmp(op, "=")) ||  // BO_Assign,
             isAugmentedAssignmentOp() ); 
  }; 
  bool isComparisonOp() override { return (!strcmp(op,"<")) || (!strcmp(op,">")) || (!strcmp(op,"<=")) || (!strcmp(op,">=")); };
  
  bool isPlusOp()  override { return (!strcmp(op,"+")); };
  bool isMinusOp() override { return (!strcmp(op,"-")); };
  bool isPlusMinusOp() override { return (!strcmp(op,"+")) || (!strcmp(op,"-")); };
  bool isMultDivOp()   override { return (!strcmp(op,"*")) || (!strcmp(op,"/")); };
  bool isRemOp()       override { return (!strcmp(op,"&")); }
  
  bool isStructOp() { return (!strcmp(op,".")) || (!strcmp(op,"->")); }; 
  
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override; // chillAST_BinaryOperator
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  void gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls ) override;
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here
  void loseLoopWithLoopVar( char *var ) override {}; // binop can't have loop as child?

  void gatherStatements( std::vector<chillAST_node*> &statements ) override; //

  bool isSameAs( chillAST_node *other ) override;

}; 








class chillAST_ArraySubscriptExpr: public chillAST_node {
private:
  chillAST_VarDecl *basedecl; //<! the vardecl that this refers to
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> base;  // always a decl ref expr? No, for multidimensional array, is another ASE
  chillAST_Child<chillAST_node> index;
  bool imwrittento;
  bool imreadfrom; // WARNING: ONLY used when both writtento and readfrom are true  x += 1 and so on
  void *uniquePtr;  // DO NOT REFERENCE THROUGH THIS!
  
  // constructors
  chillAST_ArraySubscriptExpr(); 
  chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, void *unique);
  chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, bool writtento, void *unique);
  
  chillAST_ArraySubscriptExpr( chillAST_VarDecl *v, std::vector<chillAST_node *> indeces, chillAST_node *p); 
  
  // other methods particular to this type of node
  bool operator!=( const chillAST_ArraySubscriptExpr& ) ; 
  bool operator==( const chillAST_ArraySubscriptExpr& ) ;

  chillAST_VarDecl *multibase() override;

  chillAST_node *getIndex(int dim);
  void gatherIndeces( std::vector< chillAST_node * > &ind ); 

  void replaceChild( chillAST_node *old, chillAST_node *newchild ) override; // will examine index

  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

  const char* getUnderlyingType() override {
    //debug_fprintf(stderr, "ASE getUnderlyingType() base of type %s\n", base->getTypeString()); base->print(); printf("\n"); fflush(stdout); 
    return base->getUnderlyingType(); }; 

  virtual chillAST_VarDecl* getUnderlyingVarDecl() override { return base->getUnderlyingVarDecl(); };

}; 



class chillAST_MemberExpr: public chillAST_node {
private:
  chillAST_VarDecl *basedecl; //!< the vardecl that this refers to
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_MEMBEREXPR;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> base;  // always a decl ref expr? No, can be Array Subscript Expr
  char *member; 
  char *printstring; 

  void *uniquePtr;  // DO NOT REFERENCE THROUGH THIS!

  CHILL_MEMBER_EXP_TYPE exptype; 
  

  // constructors
  chillAST_MemberExpr(); 
  chillAST_MemberExpr( chillAST_node *bas, const char *mem, void *unique, CHILL_MEMBER_EXP_TYPE t=CHILL_MEMBER_EXP_DOT);
  
  // other methods particular to this type of node
  bool operator!=( const chillAST_MemberExpr& ) ; 
  bool operator==( const chillAST_MemberExpr& ) ; 
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) override;
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) override;
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) override;

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

  chillAST_VarDecl* getUnderlyingVarDecl() override;

  void replaceChild( chillAST_node *old, chillAST_node *newchild ) override;

  void setType( CHILL_MEMBER_EXP_TYPE t ) { exptype = t; };
  CHILL_MEMBER_EXP_TYPE getType( CHILL_MEMBER_EXP_TYPE t ) { return exptype; };

  chillAST_VarDecl* multibase() override;   // this one will return the member decl
};




class chillAST_IntegerLiteral: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_INTEGERLITERAL;}
  // variables that are special for this type of node
  // TODO precision limited
  int value;
  
  // constructors
  explicit chillAST_IntegerLiteral(int val);
  
  // other methods particular to this type of node
  int evalAsInt() override { return value; }

  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ) override {}; // does nothing
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override {}; // does nothing

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) override {};  // does nothing
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override {};
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here
};


class chillAST_FloatingLiteral: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_FLOATINGLITERAL;}
  // variables that are special for this type of node
  // FIXME: two conflicting flag for almost the same thing
  double value;

  char *allthedigits; // if not NULL, use this as printable representation
  int precision;   // float == 1, double == 2

  // constructors
  explicit chillAST_FloatingLiteral( float  val);
  explicit chillAST_FloatingLiteral( double val);
  chillAST_FloatingLiteral( double val, int pre);
  chillAST_FloatingLiteral( double val, const char *printable);
  chillAST_FloatingLiteral( double val, int pre, const char *printable);
  explicit chillAST_FloatingLiteral( chillAST_FloatingLiteral *old );
  
  // other methods particular to this type of node
  void setPrecision( int precis ) { precision = precis; }; 
  int getPrecision() { return precision; } 
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ) override {}; // does nothing
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override {}; // does nothing

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing ;
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing ;

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) override {}; // does nothing
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override {};
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here
  bool isSameAs( chillAST_node *other ) override;
}; 




class chillAST_UnaryOperator: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_UNARYOPERATOR;}
  // variables that are special for this type of node
  char *op; // TODO enum
  bool prefix; // or post
  chillAST_Child<chillAST_node> subexpr;
  
  // constructors
  chillAST_UnaryOperator( const char *oper, bool pre, chillAST_node *sub );
  
  // other methods particular to this type of node
  bool isAssignmentOp() override {
    return( (!strcmp(op, "++")) || 
            (!strcmp(op, "--")) );   // are there more ???  TODO 
  }
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override; // chillAST_UnaryOperator

  void gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls ) override;

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;
  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

  int evalAsInt() override;
  bool isSameAs( chillAST_node *other ) override;

}; 





class chillAST_ImplicitCastExpr: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_IMPLICITCASTEXPR;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> subexpr;
  
  // constructors
  explicit chillAST_ImplicitCastExpr(chillAST_node *sub);
  
  // other methods particular to this type of node
  bool isNotLeaf() override { return true; };
  bool isLeaf()    override { return false; };
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here
  chillAST_VarDecl *multibase() override; // just recurse on subexpr

}; 



class chillAST_CStyleCastExpr: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_CSTYLECASTEXPR;}
  // variables that are special for this type of node
  char * towhat; 
  chillAST_Child<chillAST_node> subexpr;
  // constructors
  chillAST_CStyleCastExpr(const char *to, chillAST_node *sub);
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here
};


class chillAST_CStyleAddressOf: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_CSTYLEADDRESSOF;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> subexpr;
  // constructors
  explicit chillAST_CStyleAddressOf(chillAST_node *sub);
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

  
}; 


class chillAST_CudaMalloc:public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_CUDAMALLOC;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> devPtr;  // Pointer to allocated device memory
  chillAST_Child<chillAST_node> sizeinbytes;

  // constructors
  chillAST_CudaMalloc(chillAST_node *devmemptr, chillAST_node *size);
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

}; 


class chillAST_CudaFree:public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_CUDAFREE;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_VarDecl> variable;

  // constructors
  explicit chillAST_CudaFree(chillAST_VarDecl *var);
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

}; 





class chillAST_Malloc:public chillAST_node {   // malloc( sizeof(int) * 2048 ); 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_MALLOC;}
  // variables that are special for this type of node
  char *thing;  // to void if this is null  ,  sizeof(thing) if it is not 
  chillAST_Child<chillAST_node> sizeexpr;

  // constructors
  explicit chillAST_Malloc(chillAST_node *size);
  chillAST_Malloc(char *thething, chillAST_node *numthings); // malloc (sizeof(int) *1024)

  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

};




class chillAST_Free:public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_FREE;}
};




class chillAST_CudaMemcpy:public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_CUDAMEMCPY;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_VarDecl> dest;
  chillAST_Child<chillAST_VarDecl> src;
  chillAST_Child<chillAST_node> size;
  char *cudaMemcpyKind;  // could use the actual enum

  // constructors
  chillAST_CudaMemcpy(chillAST_VarDecl *d, chillAST_VarDecl *s, chillAST_node *size, char *kind);
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

}; 


class chillAST_CudaSyncthreads:public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_CUDASYNCTHREADS;}
  // variables that are special for this type of node

  // other methods particular to this type of node

  // required methods that I can't seem to get to inherit
  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing
  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) override {}; // does nothing

  void loseLoopWithLoopVar( char *var ) override { /* do nothing */ }
};


 
class chillAST_ReturnStmt: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_RETURNSTMT;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> returnvalue;

  // constructors
  explicit chillAST_ReturnStmt( chillAST_node *retval );
  
  // other methods particular to this type of node
  
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

}; 



class chillAST_CallExpr: public chillAST_node {  // a function call 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_CALLEXPR;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> callee;   // the function declaration (what about builtins?)
  int numargs;
  std::vector<class chillAST_node*> args;
  chillAST_VarDecl *grid;
  chillAST_VarDecl *block;

  // constructors
  explicit chillAST_CallExpr(chillAST_node *function);
  void addArg(  chillAST_node *newarg  ); 
  
  // other methods particular to this type of node
  // TODO get/set grid, block
  
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDecls      ( vector<chillAST_VarDecl*> &decls ) override;
  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  void gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) override;

  void gatherVarUsage( vector<chillAST_VarDecl*> &decls ) override;
  void gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) override;
  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here
  chillAST_node* clone() override;
}; 



class chillAST_ParenExpr: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_PARENEXPR;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> subexpr;
  
  // constructors
  explicit chillAST_ParenExpr( chillAST_node *sub );
  
  // other methods particular to this type of node

  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override;
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

}; 


class chillAST_Sizeof: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_SIZEOF;}
  // variables that are special for this type of node
  char *thing;  
  
  // constructors
  explicit chillAST_Sizeof( char *t );
  
  // other methods particular to this type of node

  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override;
  chillAST_node* clone() override;
  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override {};
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; }; // no loops under here

}; 



class chillAST_NoOp: public chillAST_node { 
public:
  virtual CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_NOOP;}
  // required methods that I can't seem to get to inherit
  chillAST_node* constantFold() override {};
  chillAST_node* clone() override { return new chillAST_NoOp( ); }; // ??

  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override {};
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override {};

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); };

  void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) override {};
  bool findLoopIndexesToReplace( chillAST_SymbolTable *symtab, bool forcesync=false ) override { return false; };//no loops under here
};



class chillAST_IfStmt: public chillAST_node { 
public:
  CHILL_ASTNODE_TYPE getType() override {return CHILLAST_NODETYPE_IFSTMT;}
  // variables that are special for this type of node
  chillAST_Child<chillAST_node> cond, thenpart, elsepart;
  IR_CONDITION_TYPE conditionoperator;  // from ir_code.hh
  
  // constructors
  chillAST_IfStmt();
  chillAST_IfStmt( chillAST_node *c, chillAST_node *t, chillAST_node *e );
  
  // other methods particular to this type of node
  chillAST_node *getCond() { return cond; };
  chillAST_node *getThen() { return thenpart; };
  chillAST_node *getElse() { return elsepart; };

  void setCond( chillAST_node *b ) {     cond = b;  if (cond)     cond->parent = this; };
  void setThen( chillAST_node *b ) { thenpart = b;  if (thenpart) thenpart->parent = this; };
  void setElse( chillAST_node *b ) { elsepart = b;  if (elsepart) elsepart->parent = this; };
  
  // required methods that I can't seem to get to inherit

  chillAST_node* constantFold() override;
  chillAST_node* clone() override;

  void gatherVarDeclsMore  ( vector<chillAST_VarDecl*> &decls ) override { gatherVarDecls(decls); } ;

  void gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) override;
  void gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) override;

  //void replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl);
  bool findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync=false ) override;

  void gatherStatements( std::vector<chillAST_node*> &statements ) override;

  void loseLoopWithLoopVar(char *var) override;
 
}; 

chillAST_FunctionDecl *findFunctionDecl( chillAST_node *node, const char *procname); 



#endif

