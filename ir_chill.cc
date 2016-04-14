

/*****************************************************************************
  Copyright (C) 2009-2010 University of Utah
  All Rights Reserved.

Purpose:   chill Intermediate Representation    no knowledge of the front end parser! 


 *****************************************************************************/

#include <typeinfo>
#include <sstream>
#include "ir_chill.hh"
#include "loop.hh"
#include "chill_error.hh"

#include "code_gen/CG_chillRepr.h"
#include "code_gen/CG_chillBuilder.h"

#include "chill_ast.hh"

vector<chillAST_VarDecl *> VariableDeclarations; 
vector<chillAST_FunctionDecl *> FunctionDeclarations; 

using namespace omega;
using namespace std;

#define DUMPFUNC(x, y) cerr << "In function " << x << "\n"; y->dump(); 


static string binops[] = {
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


static string unops[] = {
  "++", "--",           // [C99 6.5.2.4] Postfix increment and decrement
  "++", "--",           // [C99 6.5.3.1] Prefix increment and decrement
  "@",  "*",            // [C99 6.5.3.2] Address and indirection
  "+", "-",             // [C99 6.5.3.3] Unary arithmetic
  "~", "!",             // [C99 6.5.3.3] Unary arithmetic
  "__real", "__imag",   // "__real expr"/"__imag expr" Extension.
  "__extension"          // __extension__ marker.
};

// forward defs


void printsourceline( const char *filename, int line  );

void Indent( int level ) {
  for (int i=0; i<level; i++) fprintf(stderr, "    ");
}




// ----------------------------------------------------------------------------
// Class: IR_chillScalarSymbol
// ----------------------------------------------------------------------------
 
string IR_chillScalarSymbol::name() const {
  return string(chillvd->varname);  // CHILL 
}
 

// Return size in bytes
int IR_chillScalarSymbol::size() const {
  fprintf(stderr, "IR_chillScalarSymbol::size()  probably WRONG\n"); 
  return (8); // bytes?? 
}


bool IR_chillScalarSymbol::operator==(const IR_Symbol &that) const {
  //fprintf(stderr, "IR_chillScalarSymbol::operator==  probably WRONG\n"); 
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillScalarSymbol *l_that = static_cast<const IR_chillScalarSymbol *>(&that);
  return this->chillvd == l_that->chillvd;
}

IR_Symbol *IR_chillScalarSymbol::clone() const {
  return new IR_chillScalarSymbol(ir_, chillvd );  // clone
}

// ----------------------------------------------------------------------------
// Class: IR_chillArraySymbol
// ----------------------------------------------------------------------------

string IR_chillArraySymbol::name() const {
  return string( strdup( chillvd ->varname)); 
}


int IR_chillArraySymbol::elem_size() const {
  fprintf(stderr, "IR_chillArraySymbol::elem_size()  TODO\n");  exit(-1); 
  return 8;  // TODO 
}


int IR_chillArraySymbol::n_dim() const {
  //fprintf(stderr, "IR_chillArraySymbol::n_dim()\n");
  //fprintf(stderr, "variable %s %s %s\n", chillvd->vartype, chillvd->varname, chillvd->arraypart);
  //fprintf(stderr, "IR_chillArraySymbol::n_dim() %d\n", chillvd->numdimensions); 
  //fprintf(stderr, "IR_chillArraySymbol::n_dim()  TODO \n"); exit(-1); 
  return chillvd->numdimensions; 
}

IR_CONSTANT_TYPE IR_chillArraySymbol::elem_type() const { 
  const char *type = chillvd->underlyingtype;
  if (!strcmp(type, "int"))   return IR_CONSTANT_INT; // should be stored instead of a stings
  if (!strcmp(type, "float")) return IR_CONSTANT_FLOAT;
  return IR_CONSTANT_UNKNOWN;
}

// TODO
CG_outputRepr *IR_chillArraySymbol::size(int dim) const {
  fprintf(stderr, "IR_chillArraySymbol::n_size()  TODO \n"); exit(-1); 
  return NULL;
}


bool IR_chillArraySymbol::operator!=(const IR_Symbol &that) const {
  //fprintf(stderr, "IR_chillArraySymbol::operator!=   NOT EQUAL\n"); 
  //chillAST_VarDecl *chillvd;
  return chillvd != ((IR_chillArraySymbol*)&that)->chillvd ; 
}

bool IR_chillArraySymbol::operator==(const IR_Symbol &that) const {
  //fprintf(stderr, "IR_chillArraySymbol::operator==   EQUAL\n"); 
  //chillAST_VarDecl *chillvd;
  return chillvd == ((IR_chillArraySymbol*)&that)->chillvd ; 
  /*
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillArraySymbol *l_that = static_cast<const IR_chillArraySymbol *>(&that);
  return this->vd_ == l_that->vd_ && this->offset_ == l_that->offset_;
  */
}


IR_Symbol *IR_chillArraySymbol::clone() const {
  return new IR_chillArraySymbol(ir_, chillvd, offset_);
}

// ----------------------------------------------------------------------------
// Class: IR_chillConstantRef
// ----------------------------------------------------------------------------

bool IR_chillConstantRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillConstantRef *l_that = static_cast<const IR_chillConstantRef *>(&that);
  
  if (this->type_ != l_that->type_)
    return false;
  
  if (this->type_ == IR_CONSTANT_INT)
    return this->i_ == l_that->i_;
  else
    return this->f_ == l_that->f_;
}


CG_outputRepr *IR_chillConstantRef::convert() {
  if (type_ == IR_CONSTANT_INT) {

    fprintf(stderr, "IR_chillConstantRef::convert() unimplemented\n");  exit(-1); 
    
    // TODO 
    /*
    BuiltinType *bint = new BuiltinType(BuiltinType::Int);
    IntegerLiteral *ilit = new (astContext_)IntegerLiteral(*astContext_, llvm::APInt(32, i_), bint->desugar(), SourceLocation());
    CG_chillRepr *result = new CG_chillRepr(ilit);
    delete this;
    return result;
    */
  }
  else
    throw ir_error("constant type not supported");
}


IR_Ref *IR_chillConstantRef::clone() const {
  if (type_ == IR_CONSTANT_INT)
    return new IR_chillConstantRef(ir_, i_);
  else if (type_ == IR_CONSTANT_FLOAT)
    return new IR_chillConstantRef(ir_, f_);
  else
    throw ir_error("constant type not supported");
}

// ----------------------------------------------------------------------------
// Class: IR_chillScalarRef
// ----------------------------------------------------------------------------

bool IR_chillScalarRef::is_write() const {
  return op_pos_ ==  OP_DEST; // 2 other alternatives: OP_UNKNOWN, OP_SRC
}


IR_ScalarSymbol *IR_chillScalarRef::symbol() const {
  chillAST_VarDecl *vd = NULL;
  if (chillvd) vd = chillvd; 
  return new IR_chillScalarSymbol(ir_, vd); // IR_chillScalarRef::symbol()
}


bool IR_chillScalarRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillScalarRef *l_that = static_cast<const IR_chillScalarRef *>(&that);
  
  return this->chillvd == l_that->chillvd;
}


CG_outputRepr *IR_chillScalarRef::convert() {
  if (!dre) fprintf(stderr, "IR_chillScalarRef::convert()   CHILL SCALAR REF has no dre\n"); 
  CG_chillRepr *result = new CG_chillRepr(dre);
  delete this;
  return result;
}

IR_Ref * IR_chillScalarRef::clone() const {
  if (dre) return new IR_chillScalarRef(ir_, dre); // use declrefexpr if it exists
  return new IR_chillScalarRef(ir_, chillvd); // uses vardecl
}


// ----------------------------------------------------------------------------
// Class: IR_chillArrayRef
// ----------------------------------------------------------------------------

string IR_chillArrayRef::name() const {
  if (!printable) { 
    //fprintf(stderr, "IR_chillArrayRef::name(), bailing\n");
    return IR_ArrayRef::name(); 
  }
  return string(printable);  // CHILL 
}


bool IR_chillArrayRef::is_write() const {
  
  return (iswrite); // TODO 
}


// TODO
CG_outputRepr *IR_chillArrayRef::index(int dim) const {
  //fprintf(stderr, "IR_chillArrayRef::index( %d )  \n", dim); 
  //chillASE->print(); printf("\n"); fflush(stdout); 
  //chillASE->getIndex(dim)->print(); printf("\n"); fflush(stdout); 
  return new CG_chillRepr( chillASE->getIndex(dim) );
}


IR_ArraySymbol *IR_chillArrayRef::symbol() const {
  //fprintf(stderr, "IR_chillArrayRef::symbol()\n"); 
  //chillASE->print(); printf("\n"); fflush(stdout); 
  //fprintf(stderr, "base:  ");  chillASE->base->print();  printf("\n"); fflush(stdout); 

  
  chillAST_node *mb = chillASE->multibase(); 
  chillAST_VarDecl *vd = (chillAST_VarDecl*)mb;
  //fprintf(stderr, "symbol: '%s'\n", vd->varname);

  //fprintf(stderr, "IR_chillArrayRef symbol: '%s%s'\n", vd->varname, vd->arraypart); 
  //fprintf(stderr, "numdimensions %d\n", vd->numdimensions); 
  IR_ArraySymbol *AS =  new IR_chillArraySymbol(ir_, vd); 
  //fprintf(stderr, "ir_chill.cc returning IR_chillArraySymbol 0x%x\n", AS); 
  return  AS;
/*
  chillAST_node *b = chillASE->base;
  fprintf(stderr, "base of type %s\n", b->getTypeString()); 
  //b->print(); printf("\n"); fflush(stdout); 
  if (b->asttype == CHILLAST_NODETYPE_IMPLICITCASTEXPR) {
    b = ((chillAST_ImplicitCastExpr*)b)->subexpr;
    fprintf(stderr, "base of type %s\n", b->getTypeString()); 
  }
  
  if (b->asttype == CHILLAST_NODETYPE_DECLREFEXPR)  {
    if (NULL == ((chillAST_DeclRefExpr*)b)->decl) { 
      fprintf(stderr, "IR_chillArrayRef::symbol()  var decl = 0x%x\n", ((chillAST_DeclRefExpr*)b)->decl); 
      exit(-1); 
    }
    return new IR_chillArraySymbol(ir_, ((chillAST_DeclRefExpr*)b)->decl); // -> decl?
  }
  if (b->asttype ==  CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR)  { // multidimensional array
    return (
  }
  fprintf(stderr, "IR_chillArrayRef::symbol() can't handle\n");
  fprintf(stderr, "base of type %s\n", b->getTypeString()); 
  exit(-1); 
  return NULL; 
*/
}


bool IR_chillArrayRef::operator!=(const IR_Ref &that) const {
  //fprintf(stderr, "IR_chillArrayRef::operator!=\n"); 
  bool op = (*this) == that; // opposite
  return !op;
}
  
void IR_chillArrayRef::Dump() const { 
  //fprintf(stderr, "IR_chillArrayRef::Dump()  this 0x%x  chillASE 0x%x\n", this, chillASE); 
  chillASE->print(); printf("\n");fflush(stdout);
}


bool IR_chillArrayRef::operator==(const IR_Ref &that) const {
  //fprintf(stderr, "IR_chillArrayRef::operator==\n"); 
  //printf("I am\n"); chillASE->print(); printf("\n"); 
  const IR_chillArrayRef *l_that = static_cast<const IR_chillArrayRef *>(&that);
  const chillAST_ArraySubscriptExpr* thatASE = l_that->chillASE;
  //printf("other is:\n");  thatASE->print(); printf("\n"); fflush(stdout);
  //fprintf(stderr, "addresses are 0x%x  0x%x\n", chillASE, thatASE ); 
  return (*chillASE) == (*thatASE);
  /*

  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillArrayRef *l_that = static_cast<const IR_chillArrayRef *>(&that);
  
  return this->as_ == l_that->as_;
  */
}


CG_outputRepr *IR_chillArrayRef::convert() {
  //fprintf(stderr, "IR_chillArrayRef::convert()\n"); 
  CG_chillRepr *result = new  CG_chillRepr( chillASE->clone() ); 
//  CG_chillRepr *temp = new CG_chillRepr(static_cast<Expr*>(this->as_));
//  CG_outputRepr *result = temp->clone();
  //delete this;  // if you do this, and call it twice, you're DEAD 
  return result;
}


IR_Ref *IR_chillArrayRef::clone() const {
  return new IR_chillArrayRef(ir_, chillASE, iswrite);
}


// ----------------------------------------------------------------------------
// Class: IR_chillLoop
// ----------------------------------------------------------------------------
IR_chillLoop::IR_chillLoop(const IR_Code *ir, chillAST_ForStmt *achillforstmt) { 
  //fprintf(stderr, "IR_chillLoop::IR_chillLoop()\n"); 
  //fprintf(stderr, "loop is:\n");   achillforstmt->print(); 

  ir_ = ir; 
  chillforstmt = achillforstmt;

  chillAST_BinaryOperator *init = (chillAST_BinaryOperator *)chillforstmt->getInit();
  chillAST_BinaryOperator *cond = (chillAST_BinaryOperator *)chillforstmt->getCond();
  // check to be sure  (assert) 
  if (!init->isAssignmentOp() || !cond->isComparisonOp() ) {
    fprintf(stderr, "ir_chill.cc, malformed loop init or cond:\n");
    achillforstmt->print(); 
    exit(-1); 
  }

  chilllowerbound = init->getRHS();
  chillupperbound = cond->getRHS();
  conditionoperator = achillforstmt->conditionoperator; 
  
  chillAST_node *inc  = chillforstmt->getInc();
  // check the increment
  //fprintf(stderr, "increment is of type %s\n", inc->getTypeString()); 
  //inc->print(); printf("\n"); fflush(stdout);

  if (inc->asttype == CHILLAST_NODETYPE_UNARYOPERATOR) { 
    if (!strcmp(((chillAST_UnaryOperator *) inc)->op, "++")) step_size_ = 1;
    else  step_size_ = -1;
  }
  else if (inc->asttype == CHILLAST_NODETYPE_BINARYOPERATOR) { 
    int beets = false;  // slang
    chillAST_BinaryOperator *bop = (chillAST_BinaryOperator *) inc;
    if (bop->isAssignmentOp()) {        // I=I+1   or similar
      chillAST_node *rhs = bop->getRHS();  // (I+1)
      // TODO looks like this will fail for I=1+I or I=J+1 etc. do more checking
      
      char *assop =  bop->getOp(); 
      //fprintf(stderr, "'%s' is an assignment op\n", bop->getOp()); 
      if (streq(assop, "+=") || streq(assop, "-=")) {
        chillAST_node *stride = rhs;
        //fprintf(stderr, "stride is of type %s\n", stride->getTypeString());
        if  (stride->isIntegerLiteral()) {
          int val = ((chillAST_IntegerLiteral *)stride)->value;
          if      (streq( assop, "+=")) step_size_ =  val;
          else if (streq( assop, "-=")) step_size_ = -val;
          else beets = true; 
        }
        else beets = true;  // += or -= but not constant stride
      }
      else if (rhs->isBinaryOperator()) { 
        chillAST_BinaryOperator *binoprhs = (chillAST_BinaryOperator *)rhs;
        chillAST_node *intlit =  binoprhs->getRHS();
        if (intlit->isIntegerLiteral()) {
          int val = ((chillAST_IntegerLiteral *)intlit)->value;
          if      (!strcmp( binoprhs->getOp(), "+")) step_size_ =  val;
          else if (!strcmp( binoprhs->getOp(), "-")) step_size_ = -val;
          else beets = true; 
        }
        else beets = true;
      }
      else beets = true;
    }
    else beets = true;

    if (beets) {
      fprintf(stderr, "malformed loop increment (or more likely unhandled case)\n");
      inc->print(); 
      exit(-1); 
    }
  } // binary operator 
  else { 
    fprintf(stderr, "IR_chillLoop constructor, unhandled loop increment\n");
      inc->print(); 
      exit(-1); 
  }
  //inc->print(0, stderr);fprintf(stderr, "\n"); 

  chillAST_DeclRefExpr *dre = (chillAST_DeclRefExpr *)init->getLHS();
  if (!dre->isDeclRefExpr()) { 
    fprintf(stderr, "malformed loop init.\n"); 
    init->print(); 
  }

  chillindex = dre; // the loop index variable

  //fprintf(stderr, "\n\nindex is ");  dre->print(0, stderr);  fprintf(stderr, "\n"); 
  //fprintf(stderr, "init is   "); 
  //chilllowerbound->print(0, stderr);  fprintf(stderr, "\n");
  //fprintf(stderr, "condition is  %s ", "<"); 
  //chillupperbound->print(0, stderr);  fprintf(stderr, "\n");
  //fprintf(stderr, "step size is %d\n\n", step_size_) ; 

  chillbody = achillforstmt->getBody(); 

  //fprintf(stderr, "IR_chillLoop::IR_chillLoop() DONE\n"); 
}


CG_outputRepr *IR_chillLoop::lower_bound() const {
  //fprintf(stderr, "IR_chillLoop::lower_bound()\n"); 
  return new CG_chillRepr(chilllowerbound);
}

CG_outputRepr *IR_chillLoop::upper_bound() const {
  //fprintf(stderr, "IR_chillLoop::upper_bound()\n"); 
  return new CG_chillRepr(chillupperbound);
}

IR_Block *IR_chillLoop::body() const {
  //fprintf(stderr, "IR_chillLoop::body()\n");
  //assert(isa<CompoundStmt>(tf_->getBody()));
  //fprintf(stderr, "returning a chillBLOCK corresponding to the body of the loop\n"); 
  //fprintf(stderr, "body type %s\n", chillbody->getTypeString()); 
  return new IR_chillBlock(ir_, chillbody ) ; // static_cast<CompoundStmt *>(tf_->getBody()));
}

IR_Control *IR_chillLoop::clone() const {
  //fprintf(stderr, "IR_chillLoop::clone()\n"); 
  //chillforstmt->print(); fflush(stdout); 
  return new IR_chillLoop(ir_, chillforstmt);
}

IR_CONDITION_TYPE IR_chillLoop::stop_cond() const {
  chillAST_BinaryOperator *loopcondition = (chillAST_BinaryOperator*) chillupperbound;
  //fprintf(stderr, "IR_chillLoop::stop_cond()\n"); 
  return conditionoperator; 
}

IR_Block *IR_chillLoop::convert() { // convert the loop to a block 
  //fprintf(stderr, "IR_chillLoop::convert()   maybe \n"); 
  return new IR_chillBlock( ir_, chillbody ); // ?? 
}

void IR_chillLoop::dump() const { 
  fprintf(stderr, "TODO:  IR_chillLoop::dump()\n"); exit(-1); 
}


// ----------------------------------------------------------------------------
// Class: IR_chillBlock
// ----------------------------------------------------------------------------
CG_outputRepr *IR_chillBlock::original() const {
  fprintf(stderr, "IR_chillBlock::original()  TODO \n"); 
  exit(-1); 
  return NULL;
}



CG_outputRepr *IR_chillBlock::extract() const {
  fflush(stdout); 
  //fprintf(stderr, "IR_chillBlock::extract()\n"); 
  //CG_chillRepr *tnl =  new CG_chillRepr(getStmtList());

  // if the block refers to a compound statement, return the next level
  // of statements ;  otherwise just return a repr of the statements

  chillAST_node *code = chillAST;
  //if (chillAST != NULL) fprintf(stderr, "block has chillAST of type %s\n",code->getTypeString()); 
  //fprintf(stderr, "block has %d exploded statements\n", statements.size()); 

  CG_chillRepr *OR; 
  if (0 == statements.size()) { 
    OR = new CG_chillRepr(code); // presumably a compound statement ??
  }
  else { 
    //fprintf(stderr, "adding a statement from IR_chillBlock::extract()\n"); 
    OR = new CG_chillRepr(); // empty of statements
    for (int i=0; i<statements.size(); i++) OR->addStatement( statements[i] ); 
  }

  fflush(stdout); 
  //fprintf(stderr, "IR_chillBlock::extract() LEAVING\n"); 
  return OR;
}

IR_Control *IR_chillBlock::clone() const {
  //fprintf(stderr, "IR_chillBlock::clone()\n"); 
  //fprintf(stderr, "IR_chillBlock::clone()  %d statements\n", statements.size());
  return new IR_chillBlock( this );  // shallow copy ? 
}

void IR_chillBlock::dump() const { 
  fprintf(stderr, "IR_chillBlock::dump()  TODO\n");  return;
}


vector<chillAST_node*> IR_chillBlock::getStmtList() const {
  //fprintf(stderr, "IR_chillBlock::getStmtList()\n");
  return statements; // ?? 
}


void IR_chillBlock::addStatement( chillAST_node* s ) {
  statements.push_back( s );
  //fprintf(stderr, "IR_chillBlock::addStatement()  added statement of type %s\n", s->getTypeString());
  //fprintf(stderr, "IR_chillBlock::addStatement()  now have %d statements\n", statements.size()); 
}





void findmanually( chillAST_node *node, char *procname, vector<chillAST_node*>& procs ) {
  //fprintf(stderr, "findmanually()                CHILL AST node of type %s\n", node->getTypeString()); 
  
  if (node->asttype == CHILLAST_NODETYPE_FUNCTIONDECL ) { 
    char *name = ((chillAST_FunctionDecl *) node)->functionName;
    //fprintf(stderr, "node name 0x%x  ", name);
    //fprintf(stderr, "%s     procname ", name); 
    //fprintf(stderr, "0x%x  ", procname);
    //fprintf(stderr, "%s\n", procname); 
    if (!strcmp( name, procname)) {
      //fprintf(stderr, "found procedure %s\n", procname ); 
      procs.push_back( node ); 
      // quit recursing. probably not correct in some horrible case
      return; 
    }
    //else fprintf(stderr, "this is not the function we're looking for\n"); 
  }


  // this is where the children can be used effectively. 
  // we don't really care what kind of node we're at. We just check the node itself
  // and then its children is needed. 

  int numc = node->children.size();  
  //fprintf(stderr, "%d children\n", numc);

  for (int i=0; i<numc; i++) {
    //fprintf(stderr, "node of type %s is recursing to child %d\n",  node->getTypeString(), i); 
    findmanually( node->children[i], procname, procs );
  }
  return; 
}


// ----------------------------------------------------------------------------
// Class: IR_chillIf
// ----------------------------------------------------------------------------
CG_outputRepr *IR_chillIf::condition() const {
  return new CG_chillRepr( chillif->cond ); 
}

IR_Block *IR_chillIf::then_body() const {
  fprintf(stderr, "IR_chillIf::then_body() making a block with then part\n");
  chillif->thenpart->print(0, stderr); fprintf(stderr, "\n"); 
  return new IR_chillBlock(ir_, chillif->thenpart); 
}

IR_Block *IR_chillIf::else_body() const {
  if (chillif->elsepart) return  new IR_chillBlock(ir_, chillif->elsepart);
  return NULL; 
}


IR_Block *IR_chillIf::convert() { // change this ir_if to an ir_block ??
  const IR_Code *ir = ir_;
  chillAST_IfStmt *i = chillif;
	delete this;
	return new IR_chillBlock( ir, i );
}

IR_Control *IR_chillIf::clone() const {
  return  new IR_chillIf(ir_, chillif);
}




// ----------------------------------------------------------------------------
// Class: IR_chillCode
// ----------------------------------------------------------------------------

IR_chillCode::IR_chillCode() { 
  //fprintf(stderr, "IR_chillCode::IR_chillCode() NO PARAMETERS\n"); 
  filename = NULL; 
  procedurename = strdup("hellifiknow"); // NULL; 
  //scriptname = NULL; 
  outputname = NULL;
  entire_file_AST = NULL;
  chillfunc = NULL; 
  ir_pointer_counter = 0;
  ir_array_counter = 0;
}

IR_chillCode::IR_chillCode(const char *fname, char *proc_name): IR_Code() {
  //fprintf(stderr, "\nIR_chillCode::IR_chillCode()\n\n"); 
  //fprintf(stderr, "IR_chillCode::IR_chillCode( filename %s, procedure %s )\n", filename, proc_name);
  
  filename = strdup(fname); // filename is internal to IR_chillCode
  procedurename = strdup(proc_name);
  //scriptname = NULL;
  outputname = NULL;
  entire_file_AST = NULL;
  chillfunc = NULL; 
}

IR_chillCode::IR_chillCode(const char *fname, char *proc_name, char *script_name): IR_Code() {
  //fprintf(stderr, "\nIR_chillCode::IR_chillCode()\n\n"); 
  //fprintf(stderr, "IR_chillCode::IR_chillCode( filename %s, procedure %s, script %s )\n", fname, proc_name, script_name);
  
  filename = strdup(fname); // filename is internal to IR_chillCode
  procedurename = strdup(proc_name);
  //scriptname = strdup(script_name); 
  outputname = NULL;
  entire_file_AST = NULL;
  chillfunc = NULL; 
}


IR_chillCode::~IR_chillCode() {
  //func_->print(llvm::outs(), 4); // printing as part of the destructor !! 
  //fprintf(stderr, "IR_chillCode::~IR_chillCode()\noutput happening as part of the destructor !!\n");
  
  if (!chillfunc) { 
    //fprintf(stderr, "in IR_chillCode::~IR_chillCode(), chillfunc is NULL?\n"); 
    return;
  }
  //chillfunc->dump(); 
  //chillfunc->print(); 

  //fprintf(stderr, "Constant Folding before\n"); 
  //chillfunc->print(); 
  //chillfunc->constantFold(); 
  //fprintf(stderr, "\nConstant Folding after\n"); 
  //chillfunc->print(); 

  chillfunc->cleanUpVarDecls(); 

  chillAST_SourceFile *src = chillfunc->getSourceFile(); 
  //chillAST_node *p = chillfunc->parent; // should be translationDeclUnit
  if (src) { 
    //src->print(); // tmp
    if (src->isSourceFile()) src->printToFile(  ); 
  }
}





//TODO
IR_ScalarSymbol *IR_chillCode::CreateScalarSymbol(const IR_Symbol *sym, int i) {
  //fprintf(stderr, "IR_chillCode::CreateScalarSymbol()\n");  
  if (typeid(*sym) == typeid( IR_chillScalarSymbol ) ) {  // should be the case ??? 
    //fprintf(stderr, "IR_chillCode::CreateScalarSymbol() from a scalar symbol\n"); 
    //fprintf(stderr, "(typeid(*sym) == typeid( IR_chillScalarSymbol )\n"); 
    const IR_chillScalarSymbol *CSS = (IR_chillScalarSymbol*) sym;
    chillAST_VarDecl *vd = CSS->chillvd;
    
    // do we have to check to see if it's already there? 
     VariableDeclarations.push_back(vd);
     chillAST_node *bod = chillfunc->getBody(); // always a compoundStmt ?? 
     bod->insertChild(0, vd);
     //fprintf(stderr, "returning ... really\n"); 
    return new IR_chillScalarSymbol( this, CSS->chillvd); // CSS->clone(); 
  }

  // ?? 
  if (typeid(*sym) == typeid( IR_chillArraySymbol ) ) {  
    //fprintf(stderr, "IR_chillCode::CreateScalarSymbol() from an array symbol?\n"); 
    const IR_chillArraySymbol *CAS = (IR_chillArraySymbol*) sym;
    //fprintf(stderr, "CAS 0x%x   chillvd = 0x%x\n", CAS, CAS->chillvd);
    //fprintf(stderr, "\nthis is the SYMBOL?: \n"); 
    //CAS->print();
    //CAS->dump();

    chillAST_VarDecl *vd = CAS->chillvd; 
    //fprintf(stderr, "\nthis is the var decl?: "); 
    //vd->print(); printf("\n"); 
    //vd->dump(); printf("\n\n");
    fflush(stdout);  
    
    // figure out the base type (probably float) of the array
    char *basetype = vd->underlyingtype;
    //fprintf(stderr, "scalar will be of type SgType%s\n", basetype);   

    char tmpname[128];
    sprintf(tmpname, "newVariable%i\0", vd->chill_scalar_counter++); 
    chillAST_VarDecl * scalarvd = new chillAST_VarDecl( basetype, tmpname,  "",  NULL);  // TODO parent
    //scalarvd->print(); printf("\n"); fflush(stdout); 

    //fprintf(stderr, "VarDecl has parent that is a NULL\n"); 

    return (IR_ScalarSymbol *) (new IR_chillScalarSymbol( this, scalarvd)); // CSS->clone(); 
  }
  
  fprintf(stderr, "IR_chillCode::CreateScalarSymbol(), passed a sym that is not a chill scalar symbol OR an array symbol???\n"); 
  int *n = NULL;
  n[0] = 1;
  exit(-1); 
  return NULL;
}


IR_ArraySymbol *IR_chillCode::CreateArraySymbol(const IR_Symbol *sym, vector<CG_outputRepr *> &size, int i) {
  fprintf(stderr, "IR_chillCode::CreateArraySymbol()\n");  

  // build a new array name 
  char namestring[128];

  sprintf(namestring, "_P%d\0", entire_file_AST->chill_array_counter++);
  fprintf(stderr, "creating Array %s\n", namestring); 
    
  char arraypart[100];
  char *s = &arraypart[0];

  for (int i=0; i<size.size(); i++) { 
    CG_outputRepr *OR = size[i];
    CG_chillRepr * CR = (CG_chillRepr * ) OR;
    //fprintf(stderr, "%d chillnodes\n", CR->chillnodes.size()); 
    
    // this SHOULD be 1 chillnode of type IntegerLiteral (per dimension)
    int numnodes = CR->chillnodes.size();
    if (1 != numnodes) { 
      fprintf(stderr, 
              "IR_chillCode::CreateArraySymbol() array dimension %d has %d chillnodes\n", 
              i, numnodes );
      exit(-1);
    }

    chillAST_node *nodezero = CR->chillnodes[0];
    if (!nodezero->isIntegerLiteral())  {
      fprintf(stderr, "IR_chillCode::CreateArraySymbol() array dimension %d not an IntegerLiteral\n", i);
      exit(-1);
    }

    chillAST_IntegerLiteral *IL = (chillAST_IntegerLiteral *)nodezero;
    int val = IL->value;
    sprintf(s, "[%d]\0", val); 
    s = &arraypart[ strlen(arraypart) ];
  }
  //fprintf(stderr, "arraypart '%s'\n", arraypart); 

  chillAST_VarDecl *vd = new chillAST_VarDecl( "float",  namestring, arraypart, NULL); // todo type from sym

  // put decl in some symbol table
  VariableDeclarations.push_back(vd);
  // insert decl in the IR_code body
  chillAST_node *bod = chillfunc->getBody(); // always a compoundStmt ?? 
  bod->insertChild(0, vd);

  return new IR_chillArraySymbol( this, vd); 
}

// TODO 
vector<IR_ScalarRef *> IR_chillCode::FindScalarRef(const CG_outputRepr *repr) const {
  vector<IR_ScalarRef *> scalars;
  fprintf(stderr, "IR_chillCode::FindScalarRef() DIE\n");  exit(-1); 
  return scalars;
}



IR_ScalarRef *IR_chillCode::CreateScalarRef(const IR_ScalarSymbol *sym) {
  //fprintf(stderr, "\n***** ir_chill.cc IR_chillCode::CreateScalarRef( sym %s )\n", sym->name().c_str()); 
  //DeclRefExpr *de = new (vd->getASTContext())DeclRefExpr(static_cast<ValueDecl*>(vd), vd->getType(), SourceLocation());
  //fprintf(stderr, "sym 0x%x\n", sym); 

  IR_chillScalarRef *sr = new IR_chillScalarRef(this, buildDeclRefExpr(((IR_chillScalarSymbol*)sym)->chillvd)); // uses VarDecl to mak a declrefexpr
  //fprintf(stderr, "returning ScalarRef with dre 0x%x\n", sr->dre); 
  return sr; 
  //return (IR_ScalarRef *)NULL;
}



IR_ArrayRef *IR_chillCode::CreateArrayRef(const IR_ArraySymbol *sym, vector<CG_outputRepr *> &index) {
  //fprintf(stderr, "IR_chillCode::CreateArrayRef()   ir_chill.cc\n"); 
  //fprintf(stderr, "sym->n_dim() %d   index.size() %d\n", sym->n_dim(), index.size()); 

  int t;
  if(sym->n_dim() != index.size()) {
    throw invalid_argument("incorrect array symbol dimensionality   dim != size    ir_chill.cc L2359");
  }

  const IR_chillArraySymbol *c_sym = static_cast<const IR_chillArraySymbol *>(sym);
  chillAST_VarDecl *vd = c_sym->chillvd;
  vector<chillAST_node *> inds;

  //fprintf(stderr, "%d array indeces\n", sym->n_dim()); 
  for (int i=0; i< index.size(); i++) { 
    CG_chillRepr *CR = (CG_chillRepr *)index[i];
   
    int numnodes = CR->chillnodes.size();
    if (1 != numnodes) { 
      fprintf(stderr, 
              "IR_chillCode::CreateArrayRef() array dimension %d has %d chillnodes\n", 
              i, numnodes );
      exit(-1);
    }

    inds.push_back( CR->chillnodes[0] );

    /* 
       chillAST_node *nodezero = CR->chillnodes[0];
    if (!nodezero->isIntegerLiteral())  {
      fprintf(stderr,"IR_chillCode::CreateArrayRef() array dimension %d not an IntegerLiteral\n",i);
      fprintf(stderr, "it is a %s\n", nodezero->getTypeString()); 
      nodezero->print(); printf("\n"); fflush(stdout); 
      exit(-1);
    }

    chillAST_IntegerLiteral *IL = (chillAST_IntegerLiteral *)nodezero;
    int val = IL->value;
    inds.push_back( val );
    */
  }

  // now we've got the vardecl AND the indeces to make a chillAST that represents the array reference
  chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( vd, inds, NULL );

  return new IR_chillArrayRef( this, ASE, 0 ); 
}



// find all array references ANYWHERE in this block of code  ?? 
vector<IR_ArrayRef *> IR_chillCode::FindArrayRef(const CG_outputRepr *repr) const {
  //fprintf(stderr, "FindArrayRef()  ir_chill.cc\n"); 
  vector<IR_ArrayRef *> arrays;
  const CG_chillRepr *crepr = static_cast<const CG_chillRepr *>(repr); 
  vector<chillAST_node*> chillstmts = crepr->getChillCode();

  //fprintf(stderr, "there are %d chill statements in this repr\n", chillstmts.size()); 

  vector<chillAST_ArraySubscriptExpr*> refs; 
  for (int i=0; i<chillstmts.size(); i++) { 
    //fprintf(stderr, "\nchillstatement %d = ", i); chillstmts[i]->print(0, stderr); fprintf(stderr, "\n"); 
    chillstmts[i]->gatherArrayRefs( refs, false );
  }
  //fprintf(stderr, "%d total refs\n", refs.size());
  for (int i=0; i<refs.size(); i++) { 
    if (refs[i]->imreadfrom) { 
      //fprintf(stderr, "ref[%d] going to be put in TWICE, as both read and write\n", i); 
      arrays.push_back( new IR_chillArrayRef( this, refs[i], 0 ) );  // UGLY TODO dual usage of a ref in "+="
    }
    arrays.push_back( new IR_chillArrayRef( this, refs[i], refs[i]->imwrittento ) ); // this is wrong
    // we need to know whether this reference will be written, etc. 
  }

  /* 
  if(chillstmts.size() > 1) {
    for(int i=0; i<tnl->size(); ++i) {
      CG_chillRepr *r = new CG_chillRepr((*tnl)[i]);
      vector<IR_ArrayRef *> a = FindArrayRef(r);
      delete r;
      copy(a.begin(), a.end(), back_inserter(arrays));
    }
  } else if(chillstmts.size() == 1) {
    Stmt *s = (*tnl)[0];
    
    if(CompoundStmt *cs = dyn_cast<CompoundStmt>(s)) {
      for(CompoundStmt::body_iterator bi = cs->body_begin(); bi != cs->body_end(); ++bi) {
        CG_chillRepr *r = new CG_chillRepr(*bi);
        vector<IR_ArrayRef *> a = FindArrayRef(r);
        delete r;
        copy(a.begin(), a.end(), back_inserter(arrays));
      }
    } else if(ForStmt *fs = dyn_cast<ForStmt>(s)) {
      CG_chillRepr *r = new CG_chillRepr(fs->getBody());
      vector<IR_ArrayRef *> a = FindArrayRef(r);
      delete r;
      copy(a.begin(), a.end(), back_inserter(arrays));
    } else if(IfStmt *ifs = dyn_cast<IfStmt>(s)) {
      CG_chillRepr *r = new CG_chillRepr(ifs->getCond());
      vector<IR_ArrayRef *> a = FindArrayRef(r);
      delete r;
      copy(a.begin(), a.end(), back_inserter(arrays));
      r = new CG_chillRepr(ifs->getThen());
      a = FindArrayRef(r);
      delete r;
      copy(a.begin(), a.end(), back_inserter(arrays));
      if(Stmt *s_else = ifs->getElse()) {
        r = new CG_chillRepr(s_else);
        a = FindArrayRef(r);
        delete r;
        copy(a.begin(), a.end(), back_inserter(arrays));
      }
    } else if(Expr *e = dyn_cast<Expr>(s)) {
      CG_chillRepr *r = new CG_chillRepr(static_cast<Expr*>(s));
      vector<IR_ArrayRef *> a = FindArrayRef(r);
      delete r;
      copy(a.begin(), a.end(), back_inserter(arrays));
    } else throw ir_error("control structure not supported");
  }
  */
/* 
  else { // We have an expression
    Expr *op = static_cast<const CG_chillRepr *>(repr)->GetExpression();
    if(0) { // TODO: Handle pointer reference exp. here
    } else if(BinaryOperator *bop = dyn_cast<BinaryOperator>(op)) {
      CG_chillRepr *r1 = new CG_chillRepr(bop->getLHS());
      vector<IR_ArrayRef *> a1 = FindArrayRef(r1);
      delete r1;      
      copy(a1.begin(), a1.end(), back_inserter(arrays));
      CG_chillRepr *r2 = new CG_chillRepr(bop->getRHS());
      vector<IR_ArrayRef *> a2 = FindArrayRef(r2);
      delete r2;
      copy(a2.begin(), a2.end(), back_inserter(arrays));
    } else if(UnaryOperator *uop = dyn_cast<UnaryOperator>(op)) {
      CG_chillRepr *r1 = new CG_chillRepr(uop->getSubExpr());
      vector<IR_ArrayRef *> a1 = FindArrayRef(r1);
      delete r1;      
      copy(a1.begin(), a1.end(), back_inserter(arrays));
    } //else throw ir_error("Invalid expr. type passed to FindArrayRef");
  }
  */
  return arrays;
}



vector<IR_PointerArrayRef *> IR_chillCode::FindPointerArrayRef(const CG_outputRepr *repr) const
{
  fprintf(stderr, "IR_chillCode::FindPointerArrayRef()\n");
  
  fprintf(stderr, "here is the code I'm look for a pointerarrayref in, though:\n");
  CG_chillRepr * CR = (CG_chillRepr * ) repr;
  CR-> printChillNodes(); printf("\n"); fflush(stdout); 

  vector<chillAST_ArraySubscriptExpr*> refs; 

  int numnodes = CR->chillnodes.size();
  for (int i=0; i<numnodes; i++) { 
    CR->chillnodes[i]->gatherArrayRefs( refs, false );
    printf("\n");
  } 
  fprintf(stderr, "%d array refs\n", refs.size()); 

  // now look for ones where the base is an array with unknowns sizes  int *i;
  vector<IR_PointerArrayRef *> IRPAR;
  int numrefs = refs.size();
  for (int i=0; i<numrefs; i++) { 
    refs[i]->print(0,stderr); fprintf(stderr, "\n"); 
    chillAST_VarDecl *vd = refs[i]->multibase(); 
    vd->print(0,stderr); fprintf(stderr, "\n"); 
    vd->dump(); fflush(stdout); 
    if (vd->isPointer()) { 
      IRPAR.push_back( new IR_rosePointerArrayRef( this, refs[i], refs[i]->imwrittento ) ); 
    }
  }
  fprintf(stderr, "%d pointer array refs\n", IRPAR.size());

  return IRPAR; 

}




vector<IR_Control *> IR_chillCode::FindOneLevelControlStructure(const IR_Block *block) const {
  fprintf(stderr, "\nIR_chillCode::FindOneLevelControlStructure() yep CHILLcode\n"); 
  
  vector<IR_Control *> controls;
  IR_chillBlock *CB = (IR_chillBlock *) block; 

  const IR_chillBlock *R_IR_CB = (const IR_chillBlock *) block;
  vector<chillAST_node*> statements = R_IR_CB->getStmtList(); 
  int ns = statements.size();  // number of statements if block has a vec of statements instead of a single AST
  fprintf(stderr, "%d statements\n", ns);
  
  vector<chillAST_node *> children; // we will populate this. IR_Block has multiple ways of storing its contents, for undoubtedly historical reasons.  it can be an AST node, or a vector of them.
  
  // if IR_Block has statements, those are them. otherwise the code is in an AST
  if (0 < ns) {
    //fprintf(stderr, "load children with %d statements\n", ns); 
    
    for (int i=0; i<ns; i++) { 
      //fprintf(stderr, "statement %d (%p):   ", i, statements[i]); statements[i]->print(); printf("\n"); fflush(stdout); 
      children.push_back( statements[i] ); 
    }
    exit(-1);  // ?? 
  }
  else { 
    //fprintf(stderr, "there is a single AST ?\n"); 
    // we will look at the AST 
    chillAST_node *blockast = R_IR_CB->getChillAST();
    //fprintf(stderr, "basic block %p %p is:\n", blockast, R_IR_CB->chillAST ); 
    if (!blockast) { 
      fprintf(stderr, "blockast is NULL\n"); 
      // this should never happen. we have an IR_Block with no statements and no AST
      fprintf(stderr, "THIS SHOULD NEVER HAPPEN ir_chill.cc\n"); 
      return controls; // ?? 
    }
    
    // we know we have an AST.  see what the top node is
    //fprintf(stderr, "block ast of type %s\n", blockast->getTypeString()); blockast->print(); printf("\n\n");  fflush(stdout);
    
    if (blockast->isIfStmt()) { 
      //fprintf(stderr, "found a top level Basic Block If Statement.  this will be the only control structure\n"); 
      controls.push_back(new IR_roseIf(this, blockast));
      return controls;
    }
    
    if (blockast->isForStmt()) { 
      //fprintf(stderr, "found a top level Basic Block For Statement.  this will be the only control structure\n"); 
      controls.push_back(new IR_roseLoop(this, (chillAST_ForStmt *)blockast));
      return controls;
    }
    
    
    if  (blockast->isCompoundStmt()) { 
      //fprintf(stderr, "found a top level Basic Block Compound Statement\n"); 
      children = blockast->getChildren();
    }
    else  if (blockast->isFunctionDecl()) { // why did I do this? It is not in the rose version 
      //fprintf(stderr, "blockast is a Functiondecl\n"); 
      chillAST_FunctionDecl *FD =  (chillAST_FunctionDecl *)blockast;
      chillAST_node *bod = FD->getBody(); 
      children = bod->getChildren(); 
    }
    else { 
      // if the AST node is not one of these, ASSUME that it is just a single statement
      // so, no control statements underneath the block.
      return controls; // controls is empty, and this is checked in the caller
      
      //fprintf(stderr, "ir_rose.cc UNHANDLED blockast type %s\n", blockast->getTypeString()); 
      //int *i=0; int j=i[0]; 
      //exit(-1); 
    }
  }
  
  // OK, at this point, we have children of the IR_Block in the vector called children.
  // we don't care any more what the top thing is
  
  int numchildren = children.size(); 
  fprintf(stderr, "basic block has %d statements\n", numchildren);
  fprintf(stderr, "basic block is:\n");
  fprintf(stderr, "{\n");
  for (int n =0; n<numchildren; n++) { 
    children[n]->print(0,stderr); fprintf(stderr, ";\n"); 
  }
  fprintf(stderr, "}\n");
  
  int startofrun = -1;
  
  for (int i=0; i<numchildren; i++) { 
    fprintf(stderr, "child %d/%d  is of type %s\n", i, numchildren, children[i]->getTypeString());

    CHILL_ASTNODE_TYPE typ = children[i]->asttype;
    if (typ == CHILLAST_NODETYPE_LOOP) {
      fprintf(stderr, "loop\n"); 
      // we will add the loop as a control, but before we can do that, 
      // add any group of non-special
      if (startofrun != -1) {
        fprintf(stderr, "there was a run of statements %d to %d before the Loop\n", startofrun, i); 
        IR_roseBlock *rb = new IR_roseBlock(this); // empty
        //fprintf(stderr, "rb %p   startofrun %d   i %d\n", rb, startofrun, i); 
        int count = 0; 
        for (int j=startofrun; j<i; j++) { 
          fprintf(stderr, "j %d   ", j); children[j]->print(); printf("\n"); fflush(stdout); 
          rb->addStatement( children[j] ); 
          count++;
        }
        fprintf(stderr, "added %d statements to the formerly empty Block %p\n", count, rb);
        if (count == 0) { 
          int *k = 0;
          int l = k[0]; 
        }
        controls.push_back( rb );
        startofrun = -1;
      }
      // then add the loop itself 
      controls.push_back(new IR_roseLoop(this, children[i]));
      //fprintf(stderr, "roseLoop %p\n", controls[ -1+controls.size()] ); 
    }
    else if (typ == CHILLAST_NODETYPE_IFSTMT ) {
        //fprintf(stderr, "if\n"); 
        // we will add the if as a control, but before we can do that, 
        // add any group of non-special
        if (startofrun != -1) {
          //fprintf(stderr, "there was a run of statements before the IF\n"); 
          IR_roseBlock *rb = new IR_roseBlock(this); // empty
          //fprintf(stderr, "rb %p\n", rb); 
          for (int j=startofrun; j<i; j++) rb->addStatement( children[j] ); 
          controls.push_back( rb );
          startofrun = -1;
        }
        //else fprintf(stderr, "there was no run of statements before the IF\n"); 
        // then add the if itself 
        //fprintf(stderr, "adding the IF to controls\n"); 
        controls.push_back(new IR_roseIf(this, children[i])); 
        //fprintf(stderr, "roseIf %p\n", controls[ -1+controls.size()] ); 
    }
    
    else if (startofrun == -1) { // straight line code, starting a new run of statements
      //fprintf(stderr, "starting a run at %d\n", i); 
      startofrun = i;
    }
  } // for i (children statements) 

  // at the end, see if the block ENDED with a run of non-special statements.
  // if so, add that run as a control. 
  if (startofrun != -1) {
    int num = numchildren-startofrun;
    //fprintf(stderr, "adding final run of %d statements starting with %d\n", num, startofrun); 
    IR_roseBlock *rb = new IR_roseBlock(this); // empty
    if (num == 1) rb->setChillAst( children[0] ); 
    else {
      for (int j=startofrun; j<numchildren; j++) rb->addStatement( children[j] ); 
    }
    controls.push_back( rb );
  }
  
  fprintf(stderr, "ir_chill.cc returning vector of %d controls\n", controls.size() );
  for (int i=0; i<controls.size(); i++) { 
    fprintf(stderr, "%2d   an ", i);
    if (controls[i]->type() == IR_CONTROL_BLOCK) fprintf(stderr, "IR_CONTROL_BLOCK\n");
    if (controls[i]->type() == IR_CONTROL_WHILE) fprintf(stderr, "IR_CONTROL_WHILE\n");
    if (controls[i]->type() == IR_CONTROL_LOOP)  fprintf(stderr, "IR_CONTROL_LOOP\n");
    if (controls[i]->type() == IR_CONTROL_IF)    fprintf(stderr, "IR_CONTROL_IF\n");
    
  }
  fprintf(stderr, "\n"); 
  return controls;
}




IR_Block *IR_chillCode::MergeNeighboringControlStructures(const vector<IR_Control *> &controls) const {
  fprintf(stderr, "IR_chillCode::MergeNeighboringControlStructures  %d controls\n", controls.size());

  if (controls.size() == 0)
    return NULL;
  
  IR_chillBlock *CBlock =  new IR_chillBlock(controls[0]->ir_); // the thing we're building

  vector<chillAST_node*> statements;
  chillAST_node *parent = NULL; 
   for (int i = 0; i < controls.size(); i++) {
    switch (controls[i]->type()) {
    case IR_CONTROL_LOOP: {
      fprintf(stderr, "control %d is IR_CONTROL_LOOP\n", i); 
      chillAST_ForStmt *loop =  static_cast<IR_chillLoop *>(controls[i])->chillforstmt;
      if (parent == NULL) {
        parent = loop->parent;
      } else {
        if (parent != loop->parent) { 
          throw ir_error("controls to merge not at the same level");
        }
      }
      CBlock->addStatement( loop );
      break;
     }
    case IR_CONTROL_BLOCK: {
      fprintf(stderr, "control %d is IR_CONTROL_BLOCK\n", i); 
      IR_chillBlock *CB =  static_cast<IR_chillBlock*>(controls[i]);
      vector<chillAST_node*> blockstmts = CB->statements;
      if (statements.size() != 0) { 
        for (int j=0; j< blockstmts.size(); j++) {
          if (parent == NULL) {
            parent = blockstmts[j]->parent;
          }
          else { 
            if (parent !=  blockstmts[j]->parent) { 
              throw ir_error("ir_chill.cc  IR_chillCode::MergeNeighboringControlStructures  controls to merge not at the same level");
            }
          }
          CBlock->addStatement( blockstmts[j] );
        }
      }
      else {
        if (CB->getChillAST())  CBlock->addStatement(CBlock->getChillAST()); // if this is a block, add theblock's statements? 
        else { // should never happen
          fprintf(stderr, "WARNING: ir_chill.cc  IR_chillCode::MergeNeighboringControlStructures");
          fprintf(stderr, "    empty IR_CONTROL_BLOCK \n");
        }
      }
      break;
    }
    default:
      throw ir_error("unrecognized control to merge");
    }
   } // for each control

   return CBlock; 
}


IR_Block *IR_chillCode::GetCode() const {    // return IR_Block corresponding to current function?
  //fprintf(stderr, "IR_chillCode::GetCode()\n"); 
  //Stmt *s = func_->getBody();  // chill statement, and chill getBody
  //fprintf(stderr, "chillfunc 0x%x\n", chillfunc);

  chillAST_node *bod = chillfunc->getBody();  // chillAST 

  //fprintf(stderr, "got the function body??\n");

  //fprintf(stderr, "printing the function getBody()\n"); 
  //fprintf(stderr, "sourceManager 0x%x\n", sourceManager); 
  //bod->print(); 

  return new IR_chillBlock(this, chillfunc ) ; 
}


void IR_chillCode::ReplaceCode(IR_Control *old, CG_outputRepr *repr) {
  fflush(stdout); 
  fprintf(stderr, "IR_chillCode::ReplaceCode( old, *repr)\n"); 

  CG_chillRepr *chillrepr = (CG_chillRepr *) repr;
  vector<chillAST_node*>  newcode = chillrepr->getChillCode();
  int numnew = newcode.size();

  //fprintf(stderr, "new code (%d) is\n", numnew); 
  //for (int i=0; i<numnew; i++) { 
  //  newcode[i]->print(0, stderr);
  //  fprintf(stderr, "\n"); 
  //} 

  struct IR_chillLoop* cloop;

  vector<chillAST_VarDecl*> olddecls;
  chillfunc->gatherVarDecls( olddecls );
  //fprintf(stderr, "\n%d old decls   they are:\n", olddecls.size()); 
  //for (int i=0; i<olddecls.size(); i++) {
  //  fprintf(stderr, "olddecl[%d]  ox%x  ",i, olddecls[i]); 
  //  olddecls[i]->print(); printf("\n"); fflush(stdout); 
  //} 


  //fprintf(stderr, "num new stmts %d\n", numnew); 
  //fprintf(stderr, "new code we're look for decls in:\n"); 
  vector<chillAST_VarDecl*> decls;
  for (int i=0; i<numnew; i++)  {
    //newcode[i]->print(0,stderr);
    //fprintf(stderr, "\n"); 
    newcode[i]->gatherVarUsage( decls );
  }

  //fprintf(stderr, "\n%d new vars used  they are:\n", decls.size()); 
  //for (int i=0; i<decls.size(); i++) {
  //  fprintf(stderr, "decl[%d]  ox%x  ",i, decls[i]); 
  //  decls[i]->print(); printf("\n"); fflush(stdout); 
  //} 


  for (int i=0; i<decls.size(); i++) {
    //fprintf(stderr, "\nchecking "); decls[i]->print(); printf("\n"); fflush(stdout); 
    int inthere = 0; 
    for (int j=0; j<VariableDeclarations.size(); j++) { 
      if (VariableDeclarations[j] == decls[i]) { 
        //fprintf(stderr, "it's in the Variable Declarations()\n");
      }
    }
    for (int j=0; j<olddecls.size(); j++) { 
      if (decls[i] == olddecls[j]) { 
        //fprintf(stderr, "it's in the olddecls (exactly)\n");
        inthere = 1;
      }
      if (streq(decls[i]->varname, olddecls[j]->varname)) { 
        if (streq(decls[i]->arraypart, olddecls[j]->arraypart)) { 
          //fprintf(stderr, "it's in the olddecls (INEXACTLY)\n");
          inthere = 1;
        }
      }
    }
    if (!inthere) {
      //fprintf(stderr, "inserting decl[%d] for ",i); decls[i]->print(); printf("\n");fflush(stdout); 
      chillfunc->getBody()->insertChild(0, decls[i]); 
      olddecls.push_back( decls[i] ); 
    }
  }
  
  chillAST_node *par;
  switch (old->type()) {
  case IR_CONTROL_LOOP: 
    {
      //fprintf(stderr, "old is IR_CONTROL_LOOP\n"); 
      cloop = (struct IR_chillLoop* )old;
      chillAST_ForStmt *forstmt = cloop->chillforstmt;

      fprintf(stderr, "old was\n");
      forstmt->print(); printf("\n"); fflush(stdout);

      //fprintf(stderr, "\nnew code is\n");
      //for (int i=0; i<numnew; i++) { newcode[i]->print(); printf("\n"); } 
      //fflush(stdout);
      

      par = forstmt->parent;
      if (!par) {
        fprintf(stderr, "old parent was NULL\n"); 
        fprintf(stderr, "ir_chill.cc that will not work very well.\n");
        exit(-1); 
      }

      

      fprintf(stderr, "\nold parent was\n\n{\n"); 
      par->print(); printf("\n"); fflush(stdout);
      fprintf(stderr, "\n}\n"); 

      vector<chillAST_node*>  oldparentcode = par->getChildren(); // probably only works for compoundstmts
      //fprintf(stderr, "ir_chill.cc oldparentcode\n"); 

      // find loop in the parent
      int index = -1;
      int numstatements = oldparentcode.size();
      for (int i=0; i<numstatements; i++) if (oldparentcode[i] == forstmt) { index = i; }
      if (index == -1) { 
        fprintf(stderr, "ir_chill.cc can't find the loop in its parent\n"); 
        exit(-1); 
      }
      //fprintf(stderr, "loop is index %d\n", index); 

      // insert the new code
      par->setChild(index, newcode[0]);    // overwrite old stmt
      //fprintf(stderr, "inserting %s 0x%x as index %d of 0x%x\n", newcode[0]->getTypeString(), newcode[0], index, par); 
      // do we need to update the IR_cloop? 
      cloop->chillforstmt = (chillAST_ForStmt*) newcode[0]; // ?? DFL 



      //printf("inserting "); newcode[0]->print(); printf("\n"); 
      if (numnew > 1){ 
        //oldparentcode.insert( oldparentcode.begin()+index+1, numnew-1, NULL); // allocate in bulk
        
        // add the rest of the new statements
        for (int i=1; i<numnew; i++) {
          printf("inserting "); newcode[i]->print(); printf("\n"); 
          par->insertChild( index+i, newcode[i] );  // sets parent
        }
      }

      // TODO add in (insert) variable declarations that go with the new loops
      

      fflush(stdout); 
    }
    break; 
  case IR_CONTROL_BLOCK:
    fprintf(stderr, "old is IR_CONTROL_BLOCK\n"); 
    fprintf(stderr, "IR_chillCode::ReplaceCode() stubbed out\n"); 
    exit(-1); 
    //tf_old = static_cast<IR_chillBlock *>(old)->getStmtList()[0];
    break; 
  default:
    throw ir_error("control structure to be replaced not supported");
    break;    
  }
  
  fflush(stdout); 
  //fprintf(stderr, "\nafter inserting %d statements into the Chill IR,", numnew);
  fprintf(stderr, "\nnew parent2 is\n\n{\n");
  vector<chillAST_node*>  newparentcode = par->getChildren();
  for (int i=0; i<newparentcode.size(); i++) { 
    fflush(stdout); 
    //fprintf(stderr, "%d ", i); 
    newparentcode[i]->print(); printf(";\n"); fflush(stdout); 
  }



  fprintf(stderr, "}\n"); 

}




void IR_chillCode::ReplaceExpression(IR_Ref *old, CG_outputRepr *repr) {
  fprintf(stderr, "IR_chillCode::ReplaceExpression()\n");

  if (typeid(*old) == typeid(IR_chillArrayRef)) {
    //fprintf(stderr, "expressions is IR_chillArrayRef\n"); 
    IR_chillArrayRef *CAR = (IR_chillArrayRef *)old;
    chillAST_ArraySubscriptExpr* CASE = CAR->chillASE;
    printf("\nreplacing old ASE %p   ", CASE); CASE->print(); printf("\n"); fflush(stdout);

    CG_chillRepr *crepr = (CG_chillRepr *)repr;
    if (crepr->chillnodes.size() != 1) { 
      fprintf(stderr, "IR_chillCode::ReplaceExpression(), replacing with %d chillnodes???\n"); 
      //exit(-1);
    }
    
    chillAST_node *newthing = crepr->chillnodes[0]; 
    fprintf(stderr, "with new "); newthing->print(); printf("\n\n"); fflush(stdout);

    if (!CASE->parent) { 
      fprintf(stderr, "IR_chillCode::ReplaceExpression()  old has no parent ??\n"); 
      exit(-1); 
    }

    fprintf(stderr, "OLD parent = "); // of type %s\n", CASE->parent->getTypeString()); 
    if (CASE->parent->isImplicitCastExpr()) CASE->parent->parent->print(); 
    else CASE->parent->print(); 
    printf("\n"); fflush(stdout); 

    //CASE->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->parent->print(); printf("\n"); fflush(stdout); 

    CASE->parent->replaceChild( CASE, newthing ); 

    fprintf(stderr, "after (chill) replace parent is "); // of type %s\n", CASE->parent->getTypeString()); 
    if (CASE->parent->isImplicitCastExpr()) CASE->parent->parent->print(); 
    else CASE->parent->print(); 
    printf("\n\n"); fflush(stdout); 



    //CASE->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->parent->print(); printf("\n"); fflush(stdout); 


  }
  else  if (typeid(*old) == typeid(IR_chillScalarRef)) {
    fprintf(stderr, "IR_chillCode::ReplaceExpression()  IR_chillScalarRef unhandled\n"); 
  }
  else { 
    fprintf(stderr, "UNKNOWN KIND OF REF\n"); exit(-1); 
  }

  delete old;
}


// TODO 
IR_CONDITION_TYPE IR_chillCode::QueryBooleanExpOperation(const CG_outputRepr *repr) const {
  return IR_COND_UNKNOWN;
}



IR_OPERATION_TYPE IR_chillCode::QueryExpOperation(const CG_outputRepr *repr) const {
  //fprintf(stderr, "IR_chillCode::QueryExpOperation()\n");

  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  chillAST_node *node = crepr->chillnodes[0];
  //fprintf(stderr, "chillAST node type %s\n", node->getTypeString());

  // really need to be more rigorous than this hack  // TODO 
  if (node->isImplicitCastExpr()) node = ((chillAST_ImplicitCastExpr*)node)->subexpr;
  if (node->isCStyleCastExpr())   node = ((chillAST_CStyleCastExpr*)  node)->subexpr;
  if (node->isParenExpr())        node = ((chillAST_ParenExpr*)       node)->subexpr;

  if (node->isIntegerLiteral() || node->isFloatingLiteral())  return IR_OP_CONSTANT;
  else if (node->isBinaryOperator() || node->isUnaryOperator()) {
    char *opstring;
    if (node->isBinaryOperator()) 
      opstring= ((chillAST_BinaryOperator*)node)->op; // TODO enum
    else
      opstring= ((chillAST_UnaryOperator*)node)->op; // TODO enum
      
    if (!strcmp(opstring, "+")) return IR_OP_PLUS;
    if (!strcmp(opstring, "-")) return IR_OP_MINUS;
    if (!strcmp(opstring, "*")) return IR_OP_MULTIPLY;
    if (!strcmp(opstring, "/")) return IR_OP_DIVIDE;
    if (!strcmp(opstring, "=")) return IR_OP_ASSIGNMENT;

    fprintf(stderr, "ir_chill.cc  IR_chillCode::QueryExpOperation() UNHANDLED Binary(or Unary)Operator op type (%s)\n", opstring); 
    exit(-1);
  }
  else if (node->isDeclRefExpr() ) return  IR_OP_VARIABLE; // ?? 
  //else if (node->is ) return  something;
  else { 
    fprintf(stderr, "IR_chillCode::QueryExpOperation()  UNHANDLED NODE TYPE %s\n", node->getTypeString());
    exit(-1); 
  }

  /* CHILL 
  Expr *e = static_cast<const CG_chillRepr *>(repr)->GetExpression();
  if(isa<IntegerLiteral>(e) || isa<FloatingLiteral>(e)) return IR_OP_CONSTANT;
  else if(isa<DeclRefExpr>(e)) return IR_OP_VARIABLE;
  else if(BinaryOperator *bop = dyn_cast<BinaryOperator>(e)) {
    switch(bop->getOpcode()) {
    case BO_Assign: return IR_OP_ASSIGNMENT;
    case BO_Add: return IR_OP_PLUS;
    case BO_Sub: return IR_OP_MINUS;
    case BO_Mul: return IR_OP_MULTIPLY;
    case BO_Div: return IR_OP_DIVIDE;
    default: return IR_OP_UNKNOWN;
    }
  } else if(UnaryOperator *uop = dyn_cast<UnaryOperator>(e)) {
    switch(uop->getOpcode()) {
    case UO_Minus: return IR_OP_NEGATIVE;
    case UO_Plus: return IR_OP_POSITIVE;
    default: return IR_OP_UNKNOWN;
    }
  } else if(ConditionalOperator *cop = dyn_cast<ConditionalOperator>(e)) {
    BinaryOperator *bop;
    if(bop = dyn_cast<BinaryOperator>(cop->getCond())) {
      if(bop->getOpcode() == BO_GT) return IR_OP_MAX;
      else if(bop->getOpcode() == BO_LT) return IR_OP_MIN;
    } else return IR_OP_UNKNOWN;
    
  } 
  
  else if(e == NULL) return IR_OP_NULL;
  else return IR_OP_UNKNOWN;
  }
   END CLANG */
}


vector<CG_outputRepr *> IR_chillCode::QueryExpOperand(const CG_outputRepr *repr) const { 
  //fprintf(stderr, "IR_chillCode::QueryExpOperand() chill\n"); 
  vector<CG_outputRepr *> v;
  
  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  //Expr *e = static_cast<const CG_chillRepr *>(repr)->GetExpression(); wrong.. CLANG
  chillAST_node *e = crepr->chillnodes[0]; // ?? 
  //e->print(); printf("\n"); fflush(stdout); 

  // really need to be more rigorous than this hack  // TODO 
  if (e->isImplicitCastExpr()) e = ((chillAST_ImplicitCastExpr*)e)->subexpr;
  if (e->isCStyleCastExpr())   e = ((chillAST_CStyleCastExpr*)  e)->subexpr;
  if (e->isParenExpr())        e = ((chillAST_ParenExpr*)       e)->subexpr;


  //if(isa<IntegerLiteral>(e) || isa<FloatingLiteral>(e) || isa<DeclRefExpr>(e)) {
  if (e->isIntegerLiteral() || e->isFloatingLiteral() || e->isDeclRefExpr() ) { 
    //fprintf(stderr, "it's a constant\n"); 
    CG_chillRepr *repr = new CG_chillRepr(e);
    v.push_back(repr);
    //} else if(BinaryOperator *bop = dyn_cast<BinaryOperator>(e)) {
  } else if (e->isBinaryOperator()) { 
    //fprintf(stderr, "ir_chill.cc BOP TODO\n"); exit(-1); // 
    chillAST_BinaryOperator *bop = (chillAST_BinaryOperator*)e;
    char *op = bop->op;  // TODO enum for operator types
    if (streq(op, "=")) { 
      v.push_back(new CG_chillRepr( bop->rhs ));  // for assign, return RHS
    }
    else if (streq(op, "+") || streq(op, "-") || streq(op, "*") || streq(op, "/") ) {
      v.push_back(new CG_chillRepr( bop->lhs ));  // for +*-/ return both lhs and rhs
      v.push_back(new CG_chillRepr( bop->rhs )); 
    }
    else { 
      fprintf(stderr, "ir_chill.cc  IR_chillCode::QueryExpOperand() Binary Operator  UNHANDLED op (%s)\n", op); 
      exit(-1);
    }
  } // BinaryOperator
  else if  (e->isUnaryOperator()) { 
    CG_chillRepr *repr;
    chillAST_UnaryOperator *uop = (chillAST_UnaryOperator*)e;
    char *op = uop->op; // TODO enum
    if (streq(op, "+") || streq(op, "-")) {
      v.push_back( new CG_chillRepr( uop->subexpr ));
    }
    else { 
      fprintf(stderr, "ir_chill.cc  IR_chillCode::QueryExpOperand() Unary Operator  UNHANDLED op (%s)\n", op); 
      exit(-1);
    }
  } // unaryoperator
  else { 
    fprintf(stderr, "ir_chill.cc  IR_chillCode::QueryExpOperand() UNHANDLED node type %s\n", e->getTypeString()); 
    exit(-1); 
  }
    

    /*   
  Expr *op1, *op2;
    switch(bop->getOpcode()) {
    case BO_Assign:
      op2 = bop->getRHS();
      repr = new CG_chillRepr(op2);
      v.push_back(repr);
      break;
    case BO_Add:
    case BO_Sub:
    case BO_Mul:
    case BO_Div:
      op1 = bop->getLHS();
      repr = new CG_chillRepr(op1);
      v.push_back(repr);
      op2 = bop->getRHS();
      repr = new CG_chillRepr(op2);
      v.push_back(repr);
      break;
    default:
      throw ir_error("operation not supported");
    }
    */
    //} else if(UnaryOperator *uop = dyn_cast<UnaryOperator>(e)) {
    //} else if(e->isUnaryOperator()) { 
    /* 
    CG_chillRepr *repr;
    
    switch(uop->getOpcode()) {
    case UO_Minus:
    case UO_Plus:
      op1 = uop->getSubExpr();
      repr = new CG_chillRepr(op1);
      v.push_back(repr);
      break;
    default:
      throw ir_error("operation not supported");
    }
    */
    //} else if(ConditionalOperator *cop = dyn_cast<ConditionalOperator>(e)) {
    //CG_chillRepr *repr;
    
    // TODO: Handle conditional operator here
    //} else  throw ir_error("operand type UNsupported");
  
  return v;
}

IR_Ref *IR_chillCode::Repr2Ref(const CG_outputRepr *repr) const {
  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  chillAST_node *node = crepr->chillnodes[0]; 
  
  //Expr *e = static_cast<const CG_chillRep *>(repr)->GetExpression();

  if(node->isIntegerLiteral()) { 
    // FIXME: Not sure if it'll work in all cases (long?)
    int val = ((chillAST_IntegerLiteral*)node)->value; 
    return new IR_chillConstantRef(this, static_cast<coef_t>(val) ); 
  } else if(node->isFloatingLiteral()) { 
    float val = ((chillAST_FloatingLiteral*)node)->value; 
    return new IR_chillConstantRef(this, val );
  } else if(node->isDeclRefExpr()) { 
    //fprintf(stderr, "ir_chill.cc  IR_chillCode::Repr2Ref()  declrefexpr TODO\n"); exit(-1); 
    return new IR_chillScalarRef(this, (chillAST_DeclRefExpr*)node);  // uses DRE
  } else  { 
    fprintf(stderr, "ir_chill.cc IR_chillCode::Repr2Ref() UNHANDLED node type %s\n", node->getTypeString()); 
    exit(-1); 
    //assert(0);
  }
}

