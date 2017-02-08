

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
  for (int i=0; i<level; i++) debug_fprintf(stderr, "    ");
}

namespace {

  char *irTypeString(IR_CONSTANT_TYPE t) {
    switch (t) {
      case IR_CONSTANT_INT:
        return strdup("int");
        break;
      case IR_CONSTANT_FLOAT:
        return strdup("float");
        break;
      case IR_CONSTANT_DOUBLE:
        return strdup("double");
        break; // ??

      case IR_CONSTANT_UNKNOWN:
      default:
        debug_fprintf(stderr, "irTypeString() unknown IR_CONSTANT_TYPE\n");
        exit(-1);
    }
    return NULL; // unreachable
  }
}

// ----------------------------------------------------------------------------
// Class: IR_chillScalarSymbol
// ----------------------------------------------------------------------------
 
string IR_chillScalarSymbol::name() const {
  return string(chillvd->varname);  // CHILL 
}
 

// Return size in bytes
int IR_chillScalarSymbol::size() const {
  debug_fprintf(stderr, "IR_chillScalarSymbol::size()  probably WRONG\n"); 
  return (8); // bytes?? 
}


bool IR_chillScalarSymbol::operator==(const IR_Symbol &that) const {
  //debug_fprintf(stderr, "IR_chillScalarSymbol::operator==  probably WRONG\n"); 
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
  debug_fprintf(stderr, "IR_chillArraySymbol::elem_size()  TODO\n");  exit(-1); 
  return 8;  // TODO 
}


int IR_chillArraySymbol::n_dim() const {
  //debug_fprintf(stderr, "IR_chillArraySymbol::n_dim()\n");
  //debug_fprintf(stderr, "variable %s %s %s\n", chillvd->vartype, chillvd->varname, chillvd->arraypart);
  //debug_fprintf(stderr, "IR_chillArraySymbol::n_dim() %d\n", chillvd->numdimensions); 
  //debug_fprintf(stderr, "IR_chillArraySymbol::n_dim()  TODO \n"); exit(-1); 
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
  debug_fprintf(stderr, "IR_chillArraySymbol::n_size()  TODO \n"); exit(-1); 
  return NULL;
}


bool IR_chillArraySymbol::operator!=(const IR_Symbol &that) const {
  //debug_fprintf(stderr, "IR_chillArraySymbol::operator!=   NOT EQUAL\n"); 
  //chillAST_VarDecl *chillvd;
  return chillvd != ((IR_chillArraySymbol*)&that)->chillvd ; 
}

bool IR_chillArraySymbol::operator==(const IR_Symbol &that) const {
  //debug_fprintf(stderr, "IR_chillArraySymbol::operator==   EQUAL\n"); 
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
// Class: IR_chillPointerSymbol
// ----------------------------------------------------------------------------

std::string IR_chillPointerSymbol::name() const {
  debug_fprintf(stderr, "IR_rosePointerSymbol::name()\n");
	return name_;
}



IR_CONSTANT_TYPE IR_chillPointerSymbol::elem_type() const {
	char *typ = chillvd->vartype;
  if (!strcmp("int", typ)) return IR_CONSTANT_INT;
  else  if (!strcmp("float", typ)) return IR_CONSTANT_FLOAT;
  else  if (!strcmp("double", typ)) return IR_CONSTANT_DOUBLE;
  return IR_CONSTANT_UNKNOWN;
}



int IR_chillPointerSymbol::n_dim() const {
	return dim_;
}


void IR_chillPointerSymbol::set_size(int dim, omega::CG_outputRepr*)  {
  dims.resize(dim);
};

omega::CG_outputRepr *IR_chillPointerSymbol::size(int dim) const {
	return dims[dim]; // will fail because often we don't have a size for a given dimension
}


bool IR_chillPointerSymbol::operator==(const IR_Symbol &that) const {
	if (typeid(*this) != typeid(that)) return false;

	const IR_chillPointerSymbol *ps_that = static_cast<const IR_chillPointerSymbol *>(&that);
	return this->chillvd == ps_that->chillvd;
}



IR_Symbol *IR_chillPointerSymbol::clone() const {
	return new IR_chillPointerSymbol(ir_, chillvd);
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

    debug_fprintf(stderr, "IR_chillConstantRef::convert() unimplemented\n");  exit(-1); 
    
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
  if (!dre) debug_fprintf(stderr, "IR_chillScalarRef::convert()   CHILL SCALAR REF has no dre\n"); 
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
    //debug_fprintf(stderr, "IR_chillArrayRef::name(), bailing\n");
    return IR_ArrayRef::name(); 
  }
  return string(printable);  // CHILL 
}


bool IR_chillArrayRef::is_write() const {
  
  return (iswrite); // TODO 
}


// TODO
CG_outputRepr *IR_chillArrayRef::index(int dim) const {
  //debug_fprintf(stderr, "IR_chillArrayRef::index( %d )  \n", dim); 
  //chillASE->print(); printf("\n"); fflush(stdout); 
  //chillASE->getIndex(dim)->print(); printf("\n"); fflush(stdout); 
  return new CG_chillRepr( chillASE->getIndex(dim) );
}


IR_ArraySymbol *IR_chillArrayRef::symbol() const {
  //debug_fprintf(stderr, "IR_chillArrayRef::symbol()\n"); 
  //chillASE->print(); printf("\n"); fflush(stdout); 
  //debug_fprintf(stderr, "base:  ");  chillASE->base->print();  printf("\n"); fflush(stdout); 

  
  chillAST_node *mb = chillASE->multibase(); 
  chillAST_VarDecl *vd = (chillAST_VarDecl*)mb;
  //debug_fprintf(stderr, "symbol: '%s'\n", vd->varname);

  //debug_fprintf(stderr, "IR_chillArrayRef symbol: '%s%s'\n", vd->varname, vd->arraypart); 
  //debug_fprintf(stderr, "numdimensions %d\n", vd->numdimensions); 
  IR_ArraySymbol *AS =  new IR_chillArraySymbol(ir_, vd); 
  //debug_fprintf(stderr, "ir_chill.cc returning IR_chillArraySymbol 0x%x\n", AS); 
  return  AS;
/*
  chillAST_node *b = chillASE->base;
  debug_fprintf(stderr, "base of type %s\n", b->getTypeString()); 
  //b->print(); printf("\n"); fflush(stdout); 
  if (b->asttype == CHILLAST_NODETYPE_IMPLICITCASTEXPR) {
    b = ((chillAST_ImplicitCastExpr*)b)->subexpr;
    debug_fprintf(stderr, "base of type %s\n", b->getTypeString()); 
  }
  
  if (b->asttype == CHILLAST_NODETYPE_DECLREFEXPR)  {
    if (NULL == ((chillAST_DeclRefExpr*)b)->decl) { 
      debug_fprintf(stderr, "IR_chillArrayRef::symbol()  var decl = 0x%x\n", ((chillAST_DeclRefExpr*)b)->decl); 
      exit(-1); 
    }
    return new IR_chillArraySymbol(ir_, ((chillAST_DeclRefExpr*)b)->decl); // -> decl?
  }
  if (b->asttype ==  CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR)  { // multidimensional array
    return (
  }
  debug_fprintf(stderr, "IR_chillArrayRef::symbol() can't handle\n");
  debug_fprintf(stderr, "base of type %s\n", b->getTypeString()); 
  exit(-1); 
  return NULL; 
*/
}


bool IR_chillArrayRef::operator!=(const IR_Ref &that) const {
  //debug_fprintf(stderr, "IR_chillArrayRef::operator!=\n"); 
  bool op = (*this) == that; // opposite
  return !op;
}
  
void IR_chillArrayRef::Dump() const { 
  //debug_fprintf(stderr, "IR_chillArrayRef::Dump()  this 0x%x  chillASE 0x%x\n", this, chillASE); 
  chillASE->print(); printf("\n");fflush(stdout);
}


bool IR_chillArrayRef::operator==(const IR_Ref &that) const {
  //debug_fprintf(stderr, "IR_chillArrayRef::operator==\n"); 
  //printf("I am\n"); chillASE->print(); printf("\n"); 
  const IR_chillArrayRef *l_that = static_cast<const IR_chillArrayRef *>(&that);
  const chillAST_ArraySubscriptExpr* thatASE = l_that->chillASE;
  //printf("other is:\n");  thatASE->print(); printf("\n"); fflush(stdout);
  //debug_fprintf(stderr, "addresses are 0x%x  0x%x\n", chillASE, thatASE ); 
  return (*chillASE) == (*thatASE);
  /*

  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_chillArrayRef *l_that = static_cast<const IR_chillArrayRef *>(&that);
  
  return this->as_ == l_that->as_;
  */
}


CG_outputRepr *IR_chillArrayRef::convert() {
  //debug_fprintf(stderr, "IR_chillArrayRef::convert()\n"); 
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
  //debug_fprintf(stderr, "IR_chillLoop::IR_chillLoop()\n"); 
  //debug_fprintf(stderr, "loop is:\n");   achillforstmt->print(); 

  ir_ = ir; 
  chillforstmt = achillforstmt;

  chillAST_BinaryOperator *init = (chillAST_BinaryOperator *)chillforstmt->getInit();
  chillAST_BinaryOperator *cond = (chillAST_BinaryOperator *)chillforstmt->getCond();
  // check to be sure  (assert) 
  if (!init->isAssignmentOp() || !cond->isComparisonOp() ) {
    debug_fprintf(stderr, "ir_chill.cc, malformed loop init or cond:\n");
    achillforstmt->print(); 
    exit(-1); 
  }

  chilllowerbound = init->getRHS();
  chillupperbound = cond->getRHS();
  conditionoperator = achillforstmt->conditionoperator; 
  
  chillAST_node *inc  = chillforstmt->getInc();
  // check the increment
  //debug_fprintf(stderr, "increment is of type %s\n", inc->getTypeString()); 
  //inc->print(); printf("\n"); fflush(stdout);

  if (inc->getType() == CHILLAST_NODETYPE_UNARYOPERATOR) {
    if (!strcmp(((chillAST_UnaryOperator *) inc)->op, "++")) step_size_ = 1;
    else  step_size_ = -1;
  }
  else if (inc->getType() == CHILLAST_NODETYPE_BINARYOPERATOR) {
    int beets = false;  // slang
    chillAST_BinaryOperator *bop = (chillAST_BinaryOperator *) inc;
    if (bop->isAssignmentOp()) {        // I=I+1   or similar
      chillAST_node *rhs = bop->getRHS();  // (I+1)
      // TODO looks like this will fail for I=1+I or I=J+1 etc. do more checking
      
      char *assop =  bop->getOp(); 
      //debug_fprintf(stderr, "'%s' is an assignment op\n", bop->getOp()); 
      if (streq(assop, "+=") || streq(assop, "-=")) {
        chillAST_node *stride = rhs;
        //debug_fprintf(stderr, "stride is of type %s\n", stride->getTypeString());
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
      debug_fprintf(stderr, "malformed loop increment (or more likely unhandled case)\n");
      inc->print(); 
      exit(-1); 
    }
  } // binary operator 
  else { 
    debug_fprintf(stderr, "IR_chillLoop constructor, unhandled loop increment\n");
      inc->print(); 
      exit(-1); 
  }
  //inc->print(0, stderr);debug_fprintf(stderr, "\n"); 

  chillAST_DeclRefExpr *dre = (chillAST_DeclRefExpr *)init->getLHS();
  if (!dre->isDeclRefExpr()) { 
    debug_fprintf(stderr, "malformed loop init.\n"); 
    init->print(); 
  }

  chillindex = dre; // the loop index variable

  //debug_fprintf(stderr, "\n\nindex is ");  dre->print(0, stderr);  debug_fprintf(stderr, "\n"); 
  //debug_fprintf(stderr, "init is   "); 
  //chilllowerbound->print(0, stderr);  debug_fprintf(stderr, "\n");
  //debug_fprintf(stderr, "condition is  %s ", "<"); 
  //chillupperbound->print(0, stderr);  debug_fprintf(stderr, "\n");
  //debug_fprintf(stderr, "step size is %d\n\n", step_size_) ; 

  chillbody = achillforstmt->getBody(); 

  //debug_fprintf(stderr, "IR_chillLoop::IR_chillLoop() DONE\n"); 
}


CG_outputRepr *IR_chillLoop::lower_bound() const {
  //debug_fprintf(stderr, "IR_chillLoop::lower_bound()\n"); 
  return new CG_chillRepr(chilllowerbound);
}

CG_outputRepr *IR_chillLoop::upper_bound() const {
  //debug_fprintf(stderr, "IR_chillLoop::upper_bound()\n"); 
  return new CG_chillRepr(chillupperbound);
}

IR_Block *IR_chillLoop::body() const {
  //debug_fprintf(stderr, "IR_chillLoop::body()\n");
  //assert(isa<CompoundStmt>(tf_->getBody()));
  //debug_fprintf(stderr, "returning a chillBLOCK corresponding to the body of the loop\n"); 
  //debug_fprintf(stderr, "body type %s\n", chillbody->getTypeString()); 
  return new IR_chillBlock(ir_, chillbody ) ; // static_cast<CompoundStmt *>(tf_->getBody()));
}

IR_Control *IR_chillLoop::clone() const {
  //debug_fprintf(stderr, "IR_chillLoop::clone()\n"); 
  //chillforstmt->print(); fflush(stdout); 
  return new IR_chillLoop(ir_, chillforstmt);
}

IR_CONDITION_TYPE IR_chillLoop::stop_cond() const {
  chillAST_BinaryOperator *loopcondition = (chillAST_BinaryOperator*) chillupperbound;
  //debug_fprintf(stderr, "IR_chillLoop::stop_cond()\n"); 
  return conditionoperator; 
}

IR_Block *IR_chillLoop::convert() { // convert the loop to a block 
  //debug_fprintf(stderr, "IR_chillLoop::convert()   maybe \n"); 
  return new IR_chillBlock( ir_, chillbody ); // ?? 
}

void IR_chillLoop::dump() const { 
  debug_fprintf(stderr, "TODO:  IR_chillLoop::dump()\n"); exit(-1); 
}


// ----------------------------------------------------------------------------
// Class: IR_chillBlock
// ----------------------------------------------------------------------------
CG_outputRepr *IR_chillBlock::original() const {
  debug_fprintf(stderr, "IR_chillBlock::original()  TODO \n"); 
  exit(-1); 
  return NULL;
}



CG_outputRepr *IR_chillBlock::extract() const {
  fflush(stdout); 

  CG_chillRepr *OR; 
  OR = new CG_chillRepr(); // empty of statements
  for (int i=0; i<statements.size(); i++) OR->addStatement( statements[i] );

  return OR;
}

IR_Control *IR_chillBlock::clone() const {
  //debug_fprintf(stderr, "IR_chillBlock::clone()\n"); 
  //debug_fprintf(stderr, "IR_chillBlock::clone()  %d statements\n", statements.size());
  return new IR_chillBlock( this );  // shallow copy ? 
}

void IR_chillBlock::dump() const { 
  debug_fprintf(stderr, "IR_chillBlock::dump()  TODO\n");  return;
}


vector<chillAST_node*> IR_chillBlock::getStmtList() const {
  //debug_fprintf(stderr, "IR_chillBlock::getStmtList()\n");
  return statements; // ?? 
}


void IR_chillBlock::addStatement( chillAST_node* s ) {
  statements.push_back( s );
  //debug_fprintf(stderr, "IR_chillBlock::addStatement()  added statement of type %s\n", s->getTypeString());
  //debug_fprintf(stderr, "IR_chillBlock::addStatement()  now have %d statements\n", statements.size()); 
}





void findmanually( chillAST_node *node, char *procname, vector<chillAST_node*>& procs ) {
  //debug_fprintf(stderr, "findmanually()                CHILL AST node of type %s\n", node->getTypeString()); 
  
  if (node->getType() == CHILLAST_NODETYPE_FUNCTIONDECL ) {
    char *name = ((chillAST_FunctionDecl *) node)->functionName;
    //debug_fprintf(stderr, "node name 0x%x  ", name);
    //debug_fprintf(stderr, "%s     procname ", name); 
    //debug_fprintf(stderr, "0x%x  ", procname);
    //debug_fprintf(stderr, "%s\n", procname); 
    if (!strcmp( name, procname)) {
      //debug_fprintf(stderr, "found procedure %s\n", procname ); 
      procs.push_back( node ); 
      // quit recursing. probably not correct in some horrible case
      return; 
    }
    //else debug_fprintf(stderr, "this is not the function we're looking for\n"); 
  }


  // this is where the children can be used effectively. 
  // we don't really care what kind of node we're at. We just check the node itself
  // and then its children is needed. 

  int numc = node->children.size();  
  //debug_fprintf(stderr, "%d children\n", numc);

  for (int i=0; i<numc; i++) {
    //debug_fprintf(stderr, "node of type %s is recursing to child %d\n",  node->getTypeString(), i); 
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
  debug_fprintf(stderr, "IR_chillIf::then_body() making a block with then part\n");
  chillif->thenpart->print(0, stderr); debug_fprintf(stderr, "\n"); 
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
  //debug_fprintf(stderr, "IR_chillCode::IR_chillCode() NO PARAMETERS\n"); 
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
  //debug_fprintf(stderr, "\nIR_chillCode::IR_chillCode()\n\n"); 
  //debug_fprintf(stderr, "IR_chillCode::IR_chillCode( filename %s, procedure %s )\n", filename, proc_name);
  
  filename = strdup(fname); // filename is internal to IR_chillCode
  procedurename = strdup(proc_name);
  //scriptname = NULL;
  outputname = NULL;
  entire_file_AST = NULL;
  chillfunc = NULL; 
}

IR_chillCode::IR_chillCode(const char *fname, char *proc_name, char *script_name): IR_Code() {
  //debug_fprintf(stderr, "\nIR_chillCode::IR_chillCode()\n\n"); 
  //debug_fprintf(stderr, "IR_chillCode::IR_chillCode( filename %s, procedure %s, script %s )\n", fname, proc_name, script_name);
  
  filename = strdup(fname); // filename is internal to IR_chillCode
  procedurename = strdup(proc_name);
  //scriptname = strdup(script_name); 
  outputname = NULL;
  entire_file_AST = NULL;
  chillfunc = NULL; 
}


IR_chillCode::~IR_chillCode() {
  if (!chillfunc) {
    return;
  }

  chillfunc->cleanUpVarDecls(); 

  chillAST_SourceFile *src = chillfunc->getSourceFile(); 
  if (src) {
    if (src->isSourceFile()) src->printToFile(outputname);
  }
}





//TODO
IR_ScalarSymbol *IR_chillCode::CreateScalarSymbol(const IR_Symbol *sym, int i) {
  //debug_fprintf(stderr, "IR_chillCode::CreateScalarSymbol()\n");  
  if (typeid(*sym) == typeid( IR_chillScalarSymbol ) ) {  // should be the case ??? 
    //debug_fprintf(stderr, "IR_chillCode::CreateScalarSymbol() from a scalar symbol\n"); 
    //debug_fprintf(stderr, "(typeid(*sym) == typeid( IR_chillScalarSymbol )\n"); 
    const IR_chillScalarSymbol *CSS = (IR_chillScalarSymbol*) sym;
    chillAST_VarDecl *vd = CSS->chillvd;
    
    // do we have to check to see if it's already there? 
     VariableDeclarations.push_back(vd);
     chillAST_node *bod = chillfunc->getBody(); // always a compoundStmt ?? 
     bod->insertChild(0, vd);
     //debug_fprintf(stderr, "returning ... really\n"); 
    return new IR_chillScalarSymbol( this, CSS->chillvd); // CSS->clone(); 
  }

  // ?? 
  if (typeid(*sym) == typeid( IR_chillArraySymbol ) ) {  
    //debug_fprintf(stderr, "IR_chillCode::CreateScalarSymbol() from an array symbol?\n"); 
    const IR_chillArraySymbol *CAS = (IR_chillArraySymbol*) sym;
    //debug_fprintf(stderr, "CAS 0x%x   chillvd = 0x%x\n", CAS, CAS->chillvd);
    //debug_fprintf(stderr, "\nthis is the SYMBOL?: \n"); 
    //CAS->print();
    //CAS->dump();

    chillAST_VarDecl *vd = CAS->chillvd; 
    //debug_fprintf(stderr, "\nthis is the var decl?: "); 
    //vd->print(); printf("\n"); 
    //vd->dump(); printf("\n\n");
    fflush(stdout);  
    
    // figure out the base type (probably float) of the array
    char *basetype = vd->underlyingtype;
    //debug_fprintf(stderr, "scalar will be of type SgType%s\n", basetype);   

    char tmpname[128];
    sprintf(tmpname, "newVariable%i\0", vd->chill_scalar_counter++); 
    chillAST_VarDecl * scalarvd = new chillAST_VarDecl( basetype, tmpname,  "",  NULL);  // TODO parent
    //scalarvd->print(); printf("\n"); fflush(stdout); 

    //debug_fprintf(stderr, "VarDecl has parent that is a NULL\n"); 

    return (IR_ScalarSymbol *) (new IR_chillScalarSymbol( this, scalarvd)); // CSS->clone(); 
  }
  
  debug_fprintf(stderr, "IR_chillCode::CreateScalarSymbol(), passed a sym that is not a chill scalar symbol OR an array symbol???\n"); 
  int *n = NULL;
  n[0] = 1;
  exit(-1); 
  return NULL;
}

// TODO what is memory_type
IR_ScalarSymbol *IR_chillCode::CreateScalarSymbol(IR_CONSTANT_TYPE type, int memory_type, std::string name){

  char *basetype = irTypeString( type ); // float or int usually

  chillAST_VarDecl * scalarvd = new chillAST_VarDecl( basetype, name.c_str(),  "",  NULL); // TODO parent

  return (IR_ScalarSymbol *) (new IR_chillScalarSymbol( this, scalarvd));

}

IR_ArraySymbol *IR_chillCode::CreateArraySymbol(const IR_Symbol *sym, vector<CG_outputRepr *> &size, int i) {
  debug_fprintf(stderr, "IR_chillCode::CreateArraySymbol()\n");  

  // build a new array name 
  char namestring[128];

  sprintf(namestring, "_P%d\0", entire_file_AST->chill_array_counter++);
  debug_fprintf(stderr, "creating Array %s\n", namestring); 
    
  char arraypart[100];
  char *s = &arraypart[0];

  for (int i=0; i<size.size(); i++) { 
    CG_outputRepr *OR = size[i];
    CG_chillRepr * CR = (CG_chillRepr * ) OR;
    //debug_fprintf(stderr, "%d chillnodes\n", CR->chillnodes.size()); 
    
    // this SHOULD be 1 chillnode of type IntegerLiteral (per dimension)
    int numnodes = CR->chillnodes.size();
    if (1 != numnodes) { 
      debug_fprintf(stderr, 
              "IR_chillCode::CreateArraySymbol() array dimension %d has %d chillnodes\n", 
              i, numnodes );
      exit(-1);
    }

    chillAST_node *nodezero = CR->chillnodes[0];
    if (!nodezero->isIntegerLiteral())  {
      debug_fprintf(stderr, "IR_chillCode::CreateArraySymbol() array dimension %d not an IntegerLiteral\n", i);
      exit(-1);
    }

    chillAST_IntegerLiteral *IL = (chillAST_IntegerLiteral *)nodezero;
    int val = IL->value;
    sprintf(s, "[%d]\0", val); 
    s = &arraypart[ strlen(arraypart) ];
  }
  //debug_fprintf(stderr, "arraypart '%s'\n", arraypart); 

  chillAST_VarDecl *vd = new chillAST_VarDecl( "float",  namestring, arraypart, NULL); // todo type from sym

  // put decl in some symbol table
  VariableDeclarations.push_back(vd);
  // insert decl in the IR_code body
  chillAST_node *bod = chillfunc->getBody(); // always a compoundStmt ?? 
  bod->insertChild(0, vd);

  return new IR_chillArraySymbol( this, vd); 
}

omega::CG_outputRepr*  IR_chillCode::CreateArrayRefRepr(const IR_ArraySymbol *sym,
                                                       std::vector<omega::CG_outputRepr *> &index) {
  //debug_fprintf(stderr, "IR_roseCode::CreateArrayRefRepr()\n");
  IR_chillArrayRef *RAR = (IR_chillArrayRef *)CreateArrayRef(sym, index);
  return new omega::CG_chillRepr(RAR->chillASE);
};

IR_ArraySymbol *IR_chillCode::CreateArraySymbol(CG_outputRepr *type,
                                               std::vector<omega::CG_outputRepr *> &size) {

  debug_fprintf(stderr, "IR_chillCode::CreateArraySymbol 2( outputRepr, vector of outputreprs! size)\n");
  exit(-1);
}

IR_ArraySymbol *IR_chillCode::CreateArraySymbol(omega::CG_outputRepr *size, const IR_Symbol *sym){
  debug_fprintf(stderr, "IR_chillCode::CreateArraySymbol 3( outputRepr, sym )\n");
  exit(-1);
}

IR_PointerSymbol *IR_chillCode::CreatePointerSymbol(const IR_Symbol *sym,
                                                   std::vector<omega::CG_outputRepr *> &size_repr)
{
  debug_fprintf(stderr, "IR_roseCode::CreatePointerSymbol 2()\n");
  debug_fprintf(stderr, "symbol name %s\n", sym->name().c_str());

  char *typ = NULL;

  if (sym->isScalar()) {
    debug_fprintf(stderr, "scalar\n");
    IR_roseScalarSymbol *RSS = (IR_roseScalarSymbol *)sym;
    chillAST_VarDecl *vd = RSS->chillvd;
    debug_fprintf(stderr, "vd vartype %s     ", vd->vartype);
    debug_fprintf(stderr, "underlyingtype %s\n", vd->underlyingtype);
    typ = strdup(vd->vartype);
  }
  else if (sym->isArray()) {
    debug_fprintf(stderr, "array symbol at top,  array or pointer\n");
    IR_roseArraySymbol *RAS = (IR_roseArraySymbol *)sym;
    chillAST_VarDecl *vd = RAS->chillvd;
    debug_fprintf(stderr, "vd vartype %s     ", vd->vartype);
    debug_fprintf(stderr, "underlyingtype %s\n", vd->underlyingtype);
    typ = strdup(vd->vartype);
  }
  else if (sym->isPointer()) {
    debug_fprintf(stderr, "pointer symbol at top,  array or pointer  (TODO)\n");
    IR_rosePointerSymbol *RPS = (IR_rosePointerSymbol *)sym;
    chillAST_VarDecl *vd = RPS->chillvd;
    debug_fprintf(stderr, "vd vartype %s     ", vd->vartype);
    debug_fprintf(stderr, "underlyingtype %s\n", vd->underlyingtype);
    typ = strdup(vd->vartype);
  }
  else debug_fprintf(stderr, "unknown symbol type at top\n");

  if (!typ) {
    debug_fprintf(stderr, "ir_rose.cc BAD TYPE\n");
    exit(-1);
  }

  debug_fprintf(stderr, "symbol type is %s\n", typ);


  debug_fprintf(stderr, "with %d indirections\n", (int)size_repr.size());
  std::string asterisks = "";
  for (int i = 0; i < size_repr.size(); i++) {
    asterisks = asterisks + "*";
  }

  std::string s = std::string("_P_DATA")
    + omega::to_string(getAndIncrementPointerCounter());
  debug_fprintf(stderr, "defining s %s\n", s.c_str());

  chillAST_VarDecl *vd = new chillAST_VarDecl( typ, s.c_str(), asterisks.c_str(), NULL);
  vd->print(0,stderr); debug_fprintf(stderr, "\n");

  // TODO parent? symbol table?
  chillfunc->getBody()->insertChild( 0, vd);  // is this always the right function to add to?
  chillfunc->addVariableToSymbolTable( vd ); // always right?


  return new IR_chillPointerSymbol(this, vd);
}



IR_PointerSymbol *IR_chillCode::CreatePointerSymbol(const IR_CONSTANT_TYPE type,
                                                   std::vector<CG_outputRepr *> &size_repr,
                                                   std::string name) {
  debug_fprintf(stderr, "\nIR_chillCode::CreatePointerSymbol()  TODO \n");


  // this creates a definition like
  //   int *i;
  //  float ***array;
  // it does NOT use the sizes in size_repr

  char *basetype = irTypeString( type ); // float or int usually
  std::string n;
  if(name == "") {
    debug_fprintf(stderr, "creating a P_DATA name, since none was sent in\n");
    n = std::string("_P_DATA")
      + omega::to_string( getAndIncrementPointerCounter() );
    debug_fprintf(stderr, "%s\n", n.c_str());
  }
  else
    n = name;

  debug_fprintf(stderr,"*s *%s;\n",basetype, n.c_str());

  char arraypart[100];
  char *s = &arraypart[0];
  for (int i=0; i<size_repr.size(); i++) arraypart[i] = '*';
  arraypart[size_repr.size()] = '\0';

  chillAST_VarDecl *vd = new  chillAST_VarDecl( basetype, n.c_str(), arraypart, NULL);

  vd->print(0, stderr); debug_fprintf(stderr, "\n");

  // put this in current function?  (seems wrong)  TODO
  return new IR_chillPointerSymbol( this, vd );
}


IR_PointerSymbol *IR_chillCode::CreatePointerSymbol(omega::CG_outputRepr *type,
                                                   std::vector<omega::CG_outputRepr *> &size_repr)
{
  debug_fprintf(stderr, "IR_chillCode::CreatePointerSymbol 3()  TODO \n");
  exit(-1);
}


vector<IR_ScalarRef *> IR_chillCode::FindScalarRef(const CG_outputRepr *repr) const {
  std::vector<IR_ScalarRef *> scalars;

  CG_chillRepr *CR = (CG_chillRepr *) repr;
  chillAST_node * chillcode = CR->GetCode();

  vector<chillAST_DeclRefExpr*> refs;
  chillcode-> gatherDeclRefExprs(refs);

  int numdecls = refs.size();
  for (int i=0; i<numdecls; i++) {
    IR_chillScalarRef *r = new IR_chillScalarRef( this, refs[i] );
    scalars.push_back( r );
  }

  return scalars;
}



IR_ScalarRef *IR_chillCode::CreateScalarRef(const IR_ScalarSymbol *sym) {
  //debug_fprintf(stderr, "\n***** ir_chill.cc IR_chillCode::CreateScalarRef( sym %s )\n", sym->name().c_str()); 
  //DeclRefExpr *de = new (vd->getASTContext())DeclRefExpr(static_cast<ValueDecl*>(vd), vd->getType(), SourceLocation());
  //debug_fprintf(stderr, "sym 0x%x\n", sym); 

  IR_chillScalarRef *sr = new IR_chillScalarRef(this, buildDeclRefExpr(((IR_chillScalarSymbol*)sym)->chillvd)); // uses VarDecl to mak a declrefexpr
  //debug_fprintf(stderr, "returning ScalarRef with dre 0x%x\n", sr->dre); 
  return sr; 
  //return (IR_ScalarRef *)NULL;
}



IR_ArrayRef *IR_chillCode::CreateArrayRef(const IR_ArraySymbol *sym, vector<CG_outputRepr *> &index) {
  //debug_fprintf(stderr, "IR_chillCode::CreateArrayRef()   ir_chill.cc\n"); 
  //debug_fprintf(stderr, "sym->n_dim() %d   index.size() %d\n", sym->n_dim(), index.size()); 

  int t;
  if(sym->n_dim() != index.size()) {
    throw invalid_argument("incorrect array symbol dimensionality   dim != size    ir_chill.cc L2359");
  }

  const IR_chillArraySymbol *c_sym = static_cast<const IR_chillArraySymbol *>(sym);
  chillAST_VarDecl *vd = c_sym->chillvd;
  vector<chillAST_node *> inds;

  //debug_fprintf(stderr, "%d array indeces\n", sym->n_dim()); 
  for (int i=0; i< index.size(); i++) { 
    CG_chillRepr *CR = (CG_chillRepr *)index[i];
   
    int numnodes = CR->chillnodes.size();
    if (1 != numnodes) { 
      debug_fprintf(stderr, 
              "IR_chillCode::CreateArrayRef() array dimension %d has %d chillnodes\n", 
              i, numnodes );
      exit(-1);
    }

    inds.push_back( CR->chillnodes[0] );

    /* 
       chillAST_node *nodezero = CR->chillnodes[0];
    if (!nodezero->isIntegerLiteral())  {
      debug_fprintf(stderr,"IR_chillCode::CreateArrayRef() array dimension %d not an IntegerLiteral\n",i);
      debug_fprintf(stderr, "it is a %s\n", nodezero->getTypeString()); 
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
  //debug_fprintf(stderr, "FindArrayRef()  ir_chill.cc\n"); 
  vector<IR_ArrayRef *> arrays;
  const CG_chillRepr *crepr = static_cast<const CG_chillRepr *>(repr); 
  vector<chillAST_node*> chillstmts = crepr->getChillCode();

  //debug_fprintf(stderr, "there are %d chill statements in this repr\n", chillstmts.size()); 

  vector<chillAST_ArraySubscriptExpr*> refs; 
  for (int i=0; i<chillstmts.size(); i++) { 
    //debug_fprintf(stderr, "\nchillstatement %d = ", i); chillstmts[i]->print(0, stderr); debug_fprintf(stderr, "\n"); 
    chillstmts[i]->gatherArrayRefs( refs, false );
  }
  //debug_fprintf(stderr, "%d total refs\n", refs.size());
  for (int i=0; i<refs.size(); i++) { 
    if (refs[i]->imreadfrom) { 
      //debug_fprintf(stderr, "ref[%d] going to be put in TWICE, as both read and write\n", i); 
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
  debug_fprintf(stderr, "IR_chillCode::FindPointerArrayRef()\n");
  
  debug_fprintf(stderr, "here is the code I'm look for a pointerarrayref in, though:\n");
  CG_chillRepr * CR = (CG_chillRepr * ) repr;
  CR-> printChillNodes(); printf("\n"); fflush(stdout); 

  vector<chillAST_ArraySubscriptExpr*> refs; 

  int numnodes = CR->chillnodes.size();
  for (int i=0; i<numnodes; i++) { 
    CR->chillnodes[i]->gatherArrayRefs( refs, false );
    printf("\n");
  } 
  debug_fprintf(stderr, "%d array refs\n", refs.size()); 

  // now look for ones where the base is an array with unknowns sizes  int *i;
  vector<IR_PointerArrayRef *> IRPAR;
  int numrefs = refs.size();
  for (int i=0; i<numrefs; i++) { 
    refs[i]->print(0,stderr); debug_fprintf(stderr, "\n"); 
    chillAST_VarDecl *vd = refs[i]->multibase(); 
    vd->print(0,stderr); debug_fprintf(stderr, "\n"); 
    vd->dump(); fflush(stdout); 
    if (vd->isPointer()) { 
      IRPAR.push_back( new IR_rosePointerArrayRef( this, refs[i], refs[i]->imwrittento ) ); 
    }
  }
  debug_fprintf(stderr, "%d pointer array refs\n", IRPAR.size());

  return IRPAR; 

}




vector<IR_Control *> IR_chillCode::FindOneLevelControlStructure(const IR_Block *block) const {
  debug_fprintf(stderr, "\nIR_chillCode::FindOneLevelControlStructure() yep CHILLcode\n"); 
  
  vector<IR_Control *> controls;
  IR_chillBlock *CB = (IR_chillBlock *) block;
  int numstmts = (int)(CB->statements.size());
  bool unwrap = false;

  chillAST_node *blockast = NULL;

  if (numstmts == 0) return controls;
  else if (numstmts == 1) blockast = CB->statements[0]; // a single statement

  // build up a vector of "controls".
  // a run of straight-line code (statements that can't cause branching) will be
  // bundled up into an IR_Block
  // ifs and loops will get their own entry
  const std::vector<chillAST_node *> *children = NULL;
  if (blockast) {
    if (blockast->isFunctionDecl()) {
      chillAST_FunctionDecl *FD = (chillAST_FunctionDecl *) blockast;
      chillAST_node *bod = FD->getBody();
      children = &bod->getChildren();
      unwrap = true;
    }
    if (blockast->isCompoundStmt()) {
      children = &blockast->getChildren();
      unwrap = true;
    }
    if (blockast->isForStmt()) {
      controls.push_back(new IR_chillLoop(this, (chillAST_ForStmt *) blockast));
      return controls;
    }
  }
  if (!children)
    children = &(CB->statements);

  int numchildren = children->size();
  int ns;
  IR_chillBlock *basicblock = new IR_chillBlock(this); // no statements
  for (int i = 0; i < numchildren; i++) {
    CHILL_ASTNODE_TYPE typ = (*children)[i]->getType();
    if (typ == CHILLAST_NODETYPE_LOOP) {
      ns = basicblock->numstatements();
      if (ns) {
        controls.push_back(basicblock);
        basicblock = new IR_chillBlock(this); // start a new one
      }

      controls.push_back(new IR_chillLoop(this, (chillAST_ForStmt *) (*children)[i]));
    } else if (typ == CHILLAST_NODETYPE_IFSTMT ) {
      ns = basicblock->numstatements();
      if (ns) {
        controls.push_back(basicblock);
        basicblock = new IR_chillBlock(this); // start a new one
      }
      controls.push_back(new IR_chillIf(this, (chillAST_IfStmt *) (*children)[i]));
    } else
      basicblock->addStatement((*children)[i]);
  } // for each child
  ns = basicblock->numstatements();
  if (ns != 0 && (unwrap || ns != numchildren))
    controls.push_back(basicblock);

  return controls;

}




IR_Block *IR_chillCode::MergeNeighboringControlStructures(const vector<IR_Control *> &controls) const {
  debug_fprintf(stderr, "IR_chillCode::MergeNeighboringControlStructures  %d controls\n", controls.size());

  if (controls.size() == 0)
    return NULL;
  
  IR_chillBlock *CBlock =  new IR_chillBlock(controls[0]->ir_); // the thing we're building

  vector<chillAST_node*> statements;
  chillAST_node *parent = NULL; 
   for (int i = 0; i < controls.size(); i++) {
    switch (controls[i]->type()) {
    case IR_CONTROL_LOOP: {
      debug_fprintf(stderr, "control %d is IR_CONTROL_LOOP\n", i);
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
      debug_fprintf(stderr, "control %d is IR_CONTROL_BLOCK\n", i);
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
      break;
    }
    default:
      throw ir_error("unrecognized control to merge");
    }
   } // for each control

   return CBlock; 
}


IR_Block *IR_chillCode::GetCode() const {    // return IR_Block corresponding to current function?
  //debug_fprintf(stderr, "IR_chillCode::GetCode()\n"); 
  //Stmt *s = func_->getBody();  // chill statement, and chill getBody
  //debug_fprintf(stderr, "chillfunc 0x%x\n", chillfunc);

  chillAST_node *bod = chillfunc->getBody();  // chillAST 

  //debug_fprintf(stderr, "got the function body??\n");

  //debug_fprintf(stderr, "printing the function getBody()\n"); 
  //debug_fprintf(stderr, "sourceManager 0x%x\n", sourceManager); 
  //bod->print(); 

  return new IR_chillBlock(this, chillfunc ) ; 
}


void IR_chillCode::ReplaceCode(IR_Control *old, CG_outputRepr *repr) {
  fflush(stdout); 
  debug_fprintf(stderr, "IR_chillCode::ReplaceCode( old, *repr)\n"); 

  CG_chillRepr *chillrepr = (CG_chillRepr *) repr;
  vector<chillAST_node*>  newcode = chillrepr->getChillCode();
  int numnew = newcode.size();

  struct IR_chillLoop* cloop;

  vector<chillAST_VarDecl*> olddecls;
  chillfunc->gatherVarDecls( olddecls );

  vector<chillAST_VarDecl*> decls;
  for (int i=0; i<numnew; i++)
    newcode[i]->gatherVarUsage( decls );


  for (int i = 0; i < decls.size(); i++) {
    int inthere = 0;
    for (int j = 0; j < VariableDeclarations.size(); j++)
      if (VariableDeclarations[j] == decls[i])
        inthere = 1;
    for (int j = 0; j < olddecls.size(); j++) {
      if (decls[i] == olddecls[j])
        inthere = 1;
      if (!strcmp(decls[i]->varname, olddecls[j]->varname))
        if (!strcmp(decls[i]->arraypart, olddecls[j]->arraypart))
          inthere = 1;
    }
    if (!inthere) {
      chillfunc->getBody()->insertChild(0, decls[i]);
      olddecls.push_back(decls[i]);
    }
  }

  chillAST_node *par;
  switch (old->type()) {
    case IR_CONTROL_LOOP: {
      cloop = (struct IR_chillLoop *) old;
      chillAST_ForStmt *forstmt = cloop->chillforstmt;

      par = forstmt->getParent();
      if (!par) {
        chill_error_printf("old parent was NULL\n");
        chill_error_printf("ir_clang.cc that will not work very well.\n");
        exit(-1);
      }

      std::vector<chillAST_node *> *oldparentcode = &par->getChildren(); // probably only works for compoundstmts

      // find loop in the parent
      int numstatements = oldparentcode->size();
      int index = par->findChild(forstmt);
      if (index < 0) {
        chill_error_printf("can't find the loop in its parent\n");
        exit(-1);
      }
      // insert the new code
      par->setChild(index, newcode[0]);    // overwrite old stmt
      // do we need to update the IR_cloop?
      cloop->chillforstmt = (chillAST_ForStmt *) newcode[0]; // ?? DFL

      if (numnew > 1)
        // add the rest of the new statements
        for (int i = 1; i < numnew; i++)
          par->insertChild(index + i, newcode[i]);  // sets parent

      // TODO add in (insert) variable declarations that go with the new loops

      fflush(stdout);
    }
      break;
    case IR_CONTROL_BLOCK: {
      par = ((IR_chillBlock*)old)->statements[0]->getParent();
      if (!par) {
        chill_error_printf("old parent was NULL\n");
        chill_error_printf("ir_clang.cc that will not work very well.\n");
        exit(-1);
      }
      IR_chillBlock *cblock = (struct IR_chillBlock *) old;
      std::vector<chillAST_node *> *oldparentcode = &par->getChildren(); // probably only works for compoundstmts
      int index = par->findChild(cblock->statements[0]);
      for (int i = 0;i<cblock->numstatements();++i) // delete all current statements
        par->removeChild(par->findChild(cblock->statements[i]));
      for (int i = 0; i < numnew; i++)
        par->insertChild(index + i, newcode[i]);  // insert New child
      // TODO add in (insert) variable declarations that go with the new loops
      break;
    }
    default:
      throw ir_error("control structure to be replaced not supported");
      break;
  }


}




void IR_chillCode::ReplaceExpression(IR_Ref *old, CG_outputRepr *repr) {
  debug_fprintf(stderr, "IR_chillCode::ReplaceExpression()\n");
  if (typeid(*old) == typeid(IR_chillArrayRef)) {
    //debug_fprintf(stderr, "expressions is IR_chillArrayRef\n"); 
    IR_chillArrayRef *CAR = (IR_chillArrayRef *)old;
    chillAST_ArraySubscriptExpr* CASE = CAR->chillASE;
    printf("\nreplacing old ASE %p   ", CASE); CASE->print(); printf("\n"); fflush(stdout);

    CG_chillRepr *crepr = (CG_chillRepr *)repr;
    if (crepr->chillnodes.size() != 1) { 
      debug_fprintf(stderr, "IR_chillCode::ReplaceExpression(), replacing with %d chillnodes???\n"); 
      //exit(-1);
    }
    
    chillAST_node *newthing = crepr->chillnodes[0]; 
    debug_fprintf(stderr, "with new "); newthing->print(); printf("\n\n"); fflush(stdout);

    if (!CASE->parent) { 
      debug_fprintf(stderr, "IR_chillCode::ReplaceExpression()  old has no parent ??\n"); 
      exit(-1); 
    }

    debug_fprintf(stderr, "OLD parent = "); // of type %s\n", CASE->parent->getTypeString()); 
    if (CASE->parent->isImplicitCastExpr()) CASE->parent->parent->print(); 
    else CASE->parent->print(); 
    printf("\n"); fflush(stdout); 

    //CASE->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->parent->print(); printf("\n"); fflush(stdout); 

    CASE->parent->replaceChild( CASE, newthing ); 

    debug_fprintf(stderr, "after (chill) replace parent is "); // of type %s\n", CASE->parent->getTypeString()); 
    if (CASE->parent->isImplicitCastExpr()) CASE->parent->parent->print(); 
    else CASE->parent->print(); 
    printf("\n\n"); fflush(stdout); 



    //CASE->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->print(); printf("\n"); fflush(stdout); 
    //CASE->parent->parent->parent->print(); printf("\n"); fflush(stdout); 


  }
  else  if (typeid(*old) == typeid(IR_chillScalarRef)) {
    debug_fprintf(stderr, "IR_chillCode::ReplaceExpression()  IR_chillScalarRef unhandled\n"); 
  }
  else { 
    debug_fprintf(stderr, "UNKNOWN KIND OF REF\n"); exit(-1); 
  }

  delete old;
}


// TODO 
IR_CONDITION_TYPE IR_chillCode::QueryBooleanExpOperation(const CG_outputRepr *repr) const {
  return IR_COND_UNKNOWN;
}



IR_OPERATION_TYPE IR_chillCode::QueryExpOperation(const CG_outputRepr *repr) const {
  //debug_fprintf(stderr, "IR_chillCode::QueryExpOperation()\n");

  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  chillAST_node *node = crepr->chillnodes[0];
  //debug_fprintf(stderr, "chillAST node type %s\n", node->getTypeString());

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

    debug_fprintf(stderr, "ir_chill.cc  IR_chillCode::QueryExpOperation() UNHANDLED Binary(or Unary)Operator op type (%s)\n", opstring); 
    exit(-1);
  }
  else if (node->isDeclRefExpr() ) return  IR_OP_VARIABLE; // ?? 
  //else if (node->is ) return  something;
  else { 
    debug_fprintf(stderr, "IR_chillCode::QueryExpOperation()  UNHANDLED NODE TYPE %s\n", node->getTypeString());
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
  //debug_fprintf(stderr, "IR_chillCode::QueryExpOperand() chill\n"); 
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
    //debug_fprintf(stderr, "it's a constant\n"); 
    CG_chillRepr *repr = new CG_chillRepr(e);
    v.push_back(repr);
    //} else if(BinaryOperator *bop = dyn_cast<BinaryOperator>(e)) {
  } else if (e->isBinaryOperator()) { 
    //debug_fprintf(stderr, "ir_chill.cc BOP TODO\n"); exit(-1); // 
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
      debug_fprintf(stderr, "ir_chill.cc  IR_chillCode::QueryExpOperand() Binary Operator  UNHANDLED op (%s)\n", op); 
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
      debug_fprintf(stderr, "ir_chill.cc  IR_chillCode::QueryExpOperand() Unary Operator  UNHANDLED op (%s)\n", op); 
      exit(-1);
    }
  } // unaryoperator
  else { 
    debug_fprintf(stderr, "ir_chill.cc  IR_chillCode::QueryExpOperand() UNHANDLED node type %s\n", e->getTypeString()); 
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
    //debug_fprintf(stderr, "ir_chill.cc  IR_chillCode::Repr2Ref()  declrefexpr TODO\n"); exit(-1); 
    return new IR_chillScalarRef(this, (chillAST_DeclRefExpr*)node);  // uses DRE
  } else  { 
    debug_fprintf(stderr, "ir_chill.cc IR_chillCode::Repr2Ref() UNHANDLED node type %s\n", node->getTypeString()); 
    exit(-1); 
    //assert(0);
  }
}

