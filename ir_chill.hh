#ifndef IR_CHILL_HH
#define IR_CHILL_HH

// INTERMEDIATE REPRESENTATION WITH CHILLAST INTERNALS 


#include <omega.h>
#include "ir_code.hh"
#include "chill_error.hh"
#include "chill_ast.hh"
#include "chill_io.hh"
#include "parser.h"


extern vector<chillAST_VarDecl *> VariableDeclarations;
extern vector<chillAST_FunctionDecl *> FunctionDeclarations;

void findmanually( chillAST_node *node, char *procname, vector<chillAST_node*>& procs );

extern vector<chillAST_VarDecl *> VariableDeclarations;  // a global.   TODO 

struct IR_chillScalarSymbol: public IR_ScalarSymbol {
  chillAST_VarDecl *chillvd; 

  IR_chillScalarSymbol(const IR_Code *ir, chillAST_VarDecl *vd) {
    debug_fprintf(stderr, "making CHILL scalar symbol %s\n", vd->varname); 
    ir_ = ir;
    chillvd = vd;
  }
  
  std::string name() const;
  int size() const;
  bool operator==(const IR_Symbol &that) const;
  IR_Symbol *clone() const;
};

struct IR_chillFunctionSymbol: public IR_FunctionSymbol {
  chillAST_DeclRefExpr* fs_;

  IR_chillFunctionSymbol(const IR_Code *ir, chillAST_DeclRefExpr *fs ) {
    ir_ = ir;
    fs_ = fs;
  }

  std::string name() const;
  bool operator==(const IR_Symbol &that) const;
  IR_Symbol *clone() const;
};

struct IR_chillArraySymbol: public IR_ArraySymbol {
  //int indirect_;             // what was this? 
  int offset_;                 // what is this?
  chillAST_node *base;          // Usually a vardecl but can be a member expression
  chillAST_VarDecl *chillvd; 

  IR_chillArraySymbol(const IR_Code *ir, chillAST_VarDecl *vd, int offset = 0) {
    ir_ = ir;
    base = vd;
    chillvd = vd; 
    //indirect_ = indirect;
    offset_ = offset;
  }

  IR_chillArraySymbol(const IR_Code *ir, chillAST_node *n, int offset = 0) {
    ir_ = ir;
    base = n;
    chillvd = n->multibase();
    offset_ = offset;
  }

  // No Fortran support!
  IR_ARRAY_LAYOUT_TYPE layout_type() const {
    return IR_ARRAY_LAYOUT_ROW_MAJOR;
  }

  std::string name() const;
  int elem_size() const;
  IR_CONSTANT_TYPE elem_type() const;
  int n_dim() const;
  omega::CG_outputRepr *size(int dim) const;
  bool operator!=(const IR_Symbol &that) const;
  bool operator==(const IR_Symbol &that) const;
  IR_Symbol *clone() const;
  
};



struct IR_chillConstantRef: public IR_ConstantRef {
  union {
    omega::coef_t i_;
    double f_;
  };
  
  IR_chillConstantRef(const IR_Code *ir, omega::coef_t i) {
    ir_ = ir;
    type_ = IR_CONSTANT_INT;
    i_ = i;
  }
  IR_chillConstantRef(const IR_Code *ir, double f) {
    ir_ = ir;
    type_ = IR_CONSTANT_FLOAT;
    f_ = f;
  }
  omega::coef_t integer() const {
    assert(is_integer());
    return i_;
  }
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
  
};

enum OP_POSITION { OP_DEST =-1, OP_UNKNOWN, OP_SRC };
#define OP_LEFT  OP_DEST
#define OP_RIGHT OP_SRC

struct IR_chillScalarRef: public IR_ScalarRef {
  /*!
   * @brief the position of the operand
   * -1 means destination operand, 0== unknown, 1 == source operand
   */
  OP_POSITION op_pos_;
  //chillAST_BinaryOperator *bop;  // binary op that contains this scalar? 
  chillAST_DeclRefExpr    *dre;   //!< declrefexpr that uses this scalar ref, if that exists
  chillAST_VarDecl        *chillvd; //!< the vardecl for this scalar 
  
  IR_chillScalarRef(const IR_Code *ir, chillAST_BinaryOperator *ins, OP_POSITION pos) {
    debug_fprintf(stderr, "\n*****                         new IR_xxxxScalarRef( ir, ins, pos ) *****\n\n"); 
    exit(-1);
    // this constructor takes a binary operation and an indicator of which side of the op to use,
    // and finds the scalar in the lhs or rhs of the binary op. 
    ir_ = ir;
    dre = NULL;
    //bop = ins; //   do we need this? 
    if (pos == OP_LEFT) { 
      chillAST_node *lhs = ins->lhs;
      if (lhs->isDeclRefExpr()) { 
        chillAST_DeclRefExpr *DRE = (chillAST_DeclRefExpr *) lhs;
        dre = DRE; 
        chillvd = DRE->getVarDecl();
      }
      else if (lhs->isVarDecl()) { 
        chillvd = (chillAST_VarDecl *)lhs;
      }
      else { 
        debug_fprintf(stderr, "IR_chillScalarRef constructor, I'm confused\n"); exit(-1); 
      }
    }
    else { 
      chillAST_node *rhs = ins->rhs;
      if (rhs->isDeclRefExpr()) { 
        chillAST_DeclRefExpr *DRE = (chillAST_DeclRefExpr *) rhs;
        dre = DRE;
        chillvd = DRE->getVarDecl();
      }
      else if (rhs->isVarDecl()) { 
        chillvd = (chillAST_VarDecl *)rhs;
      }
      else { 
        debug_fprintf(stderr, "IR_chillScalarRef constructor, I'm confused\n"); exit(-1); 
      }
    }
    op_pos_ = pos;
  }

  IR_chillScalarRef(const IR_Code *ir, chillAST_DeclRefExpr *d) { 
    debug_fprintf(stderr, "\n*****                         new IR_xxxxScalarRef( ir, REF EXPR sym %s ) *****\n\n", d->getVarDecl()->varname);
    ir_ = ir;
    dre = d;
    chillvd = d->getVarDecl();
    op_pos_ = OP_UNKNOWN; 
  }

  IR_chillScalarRef(const IR_Code *ir, chillAST_VarDecl *vardecl) { 
    debug_fprintf(stderr, "\n*****                         new IR_xxxxScalarRef( ir, sym 0x1234567 ) ***** THIS SHOULD NEVER HAPPEN\n\n"); 
    ir_ = ir;
    dre = NULL;
    chillvd = vardecl;
    op_pos_ = OP_UNKNOWN; 
  }

  
  bool is_write() const;
  IR_ScalarSymbol *symbol() const;
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
};

struct IR_chillFunctionRef: public IR_FunctionRef {

  chillAST_DeclRefExpr *vs_;

  int is_write_;

  IR_chillFunctionRef(const IR_Code *ir, chillAST_DeclRefExpr *ins) {
    ir_ = ir;
    vs_ = ins;
    is_write_ = 0;
  }
  bool is_write() const;
  IR_FunctionSymbol *symbol() const;
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
};

struct IR_chillArrayRef: public IR_ArrayRef {
  //DeclRefExpr *as_; 
  //chillAST_DeclRefExpr *chillDRE;
  chillAST_ArraySubscriptExpr* chillASE; 
  char *printable; 
  int iswrite; 
  

  IR_chillArrayRef(const IR_Code *ir, chillAST_ArraySubscriptExpr *ase, int write ) { 
    debug_fprintf(stderr, "IR_chillArrayRef::IR_chillArrayRef()  write %d\n", write); 
    ir_ = ir;
    chillASE = ase; 
    iswrite = write;  // ase->imwrittento;
    printable = NULL;
  }

  IR_chillArrayRef(const IR_Code *ir, chillAST_ArraySubscriptExpr *ase, const char *printname, int write ) { 
    //debug_fprintf(stderr, "IR_chillArrayRef::IR_chillArrayRef()  write %d\n", write); 
    ir_ = ir;
    chillASE = ase; 
    iswrite = write;  // ase->imwrittento;
    printable = NULL; 
    if (printname) printable = strdup(printname);
  }

  std::string name() const; // overriding virtual one in ir_code.hh 
  bool is_write() const;
  omega::CG_outputRepr *index(int dim) const;
  IR_ArraySymbol *symbol() const;
  bool operator!=(const IR_Ref &that) const;
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
  virtual void Dump() const;
};

struct IR_chillPointerArrayRef: public IR_PointerArrayRef { // exactly the same as arrayref ???
  chillAST_ArraySubscriptExpr* chillASE;
  int iswrite;

  IR_chillPointerArrayRef(const IR_Code *ir, chillAST_ArraySubscriptExpr *ase, bool write ) {
    //debug_fprintf(stderr, "IR_XXXXPointerArrayRef::IR_XXXXArrayRef() '%s' write %d\n\n", ase->basedecl->varname, write);
    ir_ = ir;
    chillASE = ase;
    // dies? ase->dump(); fflush(stdout);

    iswrite = write;  // ase->imwrittento;
  }


  bool is_write() const  { return iswrite; };
  omega::CG_outputRepr *index(int dim) const;
  IR_PointerSymbol *symbol() const;
  bool operator==(const IR_Ref &that) const;
  bool operator!=(const IR_Ref &that) const; // not the opposite logic to ==     TODO
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
  virtual void Dump() const;
};

struct IR_chillLoop: public IR_Loop {
  int step_size_;

  chillAST_DeclRefExpr *chillindex;   // the loop index variable  (I)  // was DeclRefExpr
  chillAST_ForStmt     *chillforstmt; 
// Mahdi: Change to correct embedded iteration space: from Tuowen's topdown branch
// next 2 lines are commented in Tuowen's branch, they related to current way of 
// generating iteration space that Tuowen may want to keep, I wanted to keep them, but
// variables have the same name as the newly introduced ones  
//  chillAST_node        *chilllowerbound;
//  chillAST_node        *chillupperbound;
  omega::CG_outputRepr *chilllowerbound; 
  omega::CG_outputRepr *chillupperbound;
  bool well_formed; //<! Declare whether the loop is gonna be parse-able or not

  chillAST_node        *chillbody;    // presumably a compound statement, but not guaranteeed
  IR_CONDITION_TYPE conditionoperator;

  IR_chillLoop(const IR_Code *ir, chillAST_ForStmt *forstmt);

  ~IR_chillLoop() {}
  IR_ScalarSymbol *index() const { return new IR_chillScalarSymbol(ir_, chillindex->getVarDecl()); }
  omega::CG_outputRepr *lower_bound() const;
  omega::CG_outputRepr *upper_bound() const;
  IR_CONDITION_TYPE stop_cond() const;
  IR_Block *body() const;
  
  // Handle following types of increment expressions:
  
  // Unary increment/decrement
  // i += K OR i -= K
  // i = i + K OR i = i - K
  // where K is positive
  int step_size() const { return step_size_; }  // K is always an integer ???
  IR_Control *clone() const;
  IR_Block *convert() ;
  virtual void dump() const; 
};




struct IR_chillBlock: public IR_Block {   // ONLY ONE OF bDecl or cs_ will be nonNULL  ?? 
public:
  vector<chillAST_node *>statements;
  
  // Block is a basic block?? (no, just a chunk of code )
  
  IR_chillBlock() { 
    ir_ = NULL;
  }


  IR_chillBlock(const IR_Code *ir, chillAST_node *ast) { 
    ir_ = ir;
    if (ast != NULL)
      statements.push_back(ast);
  }

  IR_chillBlock(const IR_Code *ir) { //  : cs_(NULL), bDecl_(NULL) {
    ir_ = ir;
  }
  
  IR_chillBlock( const IR_chillBlock *CB ) {  // clone existing IR_chillBlock
    ir_ = CB->ir_;
    for (int i=0; i<CB->statements.size(); i++) statements.push_back( CB->statements[i] ); 
  }


  ~IR_chillBlock() {} // leaves AST and statements intact

  omega::CG_outputRepr *extract() const;
  omega::CG_outputRepr *original() const;
  IR_Control *clone() const;

  vector<chillAST_node*> getStmtList() const;
  int numstatements() { return statements.size(); } ; 
  void addStatement( chillAST_node* s ); 

  void dump() const; 
  
};




struct IR_chillIf: public IR_If {

  chillAST_IfStmt  *chillif; 
  
  IR_chillIf(const IR_Code *ir, chillAST_node *i) {
    debug_fprintf(stderr, "IR_chillIf::IR_chillIf( ir, chillast_node )\n");
    ir_ = ir;
    if (!i->isIfStmt()) { 
      debug_fprintf(stderr, "IR_chillIf::IR_chillIf( ir, chillast_node ) node is not an ifstmt\n");
      debug_fprintf(stderr, "it is a %s\n", i->getTypeString());
      i->print(0, std::cerr); debug_fprintf(stderr, "\n\n");
    }
    chillif = (chillAST_IfStmt *)i;
  }


  IR_chillIf(const IR_Code *ir, chillAST_IfStmt *i) {
    debug_fprintf(stderr, "IR_chillIf::IR_chillIf( ir, chillast_IfStmt )\n"); 
    ir_ = ir;
    if (!i->isIfStmt()) { 
      debug_fprintf(stderr, "IR_chillIf::IR_chillIf( ir, chillast_IfStmt ) node is not an ifstmt\n");
      debug_fprintf(stderr, "it is a %s\n", i->getTypeString());
      i->print(0, std::cerr); debug_fprintf(stderr, "\n\n");
    }
    chillif = i;
  }


  ~IR_chillIf() { // leave ast alone
  }


  omega::CG_outputRepr *condition() const;
  IR_Block *then_body() const;
  IR_Block *else_body() const;
  IR_Block *convert();
  IR_Control *clone() const;
};


struct IR_chillPointerSymbol: public IR_PointerSymbol {
  chillAST_VarDecl *chillvd;

  std::string name_;  // these could be gotten by looking at the vardecl
  int dim_;
  std::vector<omega::CG_outputRepr *> dims; // ???

 	IR_chillPointerSymbol(const IR_Code *ir, chillAST_VarDecl *v ) {
    ir_ = ir;
    chillvd = v;

    name_ = chillvd->varname;
    dim_ = chillvd->numdimensions;
    dims.resize(dim_);
    // TODO set sizes
  };

  std::string name() const;
  int n_dim() const;

	bool operator==(const IR_Symbol &that) const;
	omega::CG_outputRepr *size(int dim) const ;
	void set_size(int dim, omega::CG_outputRepr*) ;
  IR_Symbol *clone() const;
	IR_CONSTANT_TYPE elem_type() const;
};

class IR_chillCode: public IR_Code{   // for an entire file?  A single function?
protected:

  //  
  char *filename;
  char *outputname;    // so we can output different results from one source, using different scripts
  chill::Parser *parser;

  std::vector<chillAST_VarDecl> entire_file_symbol_table;
  // loop symbol table??   for (int i=0;  ... )  ??


  // TODO yeah, these need to be associated with a sourcefile ??
  std::map<std::string, chillAST_node *> defined_macros;  // TODO these need to be in a LOCATION

public:
  chillAST_SourceFile *entire_file_AST;
  chillAST_FunctionDecl * chillfunc;   // the function we're currenly modifying

  char *procedurename;

  IR_chillCode(chill::Parser *parser, const char *filename, const char *proc_name, const char * dest_name);
  ~IR_chillCode();

  void setOutputName( const char *name ) { outputname = strdup(name); }

  virtual omega::CG_outputRepr *RetrieveMacro(std::string s);
  IR_ScalarSymbol *CreateScalarSymbol(const IR_Symbol *sym, int i);
  IR_ScalarSymbol *CreateScalarSymbol(IR_CONSTANT_TYPE type, int memory_type = 0, std::string name = "");

  IR_ArraySymbol  *CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size, int i);
  IR_ArraySymbol *CreateArraySymbol(omega::CG_outputRepr *type,
                                    std::vector<omega::CG_outputRepr *> &size_repr);
  IR_ArraySymbol *CreateArraySymbol(omega::CG_outputRepr *size, const IR_Symbol *sym);

  IR_PointerSymbol *CreatePointerSymbol(const IR_Symbol *sym,
                                        std::vector<omega::CG_outputRepr *> &size_repr);
  IR_PointerSymbol *CreatePointerSymbol(const IR_CONSTANT_TYPE type,
                                        std::vector<omega::CG_outputRepr *> &size_repr,
                                        std::string name="");
  IR_PointerSymbol *CreatePointerSymbol(omega::CG_outputRepr *type,
                                        std::vector<omega::CG_outputRepr *> &size_repr);

  IR_ScalarRef *CreateScalarRef(const IR_ScalarSymbol *sym);
  IR_ArrayRef *CreateArrayRef(const IR_ArraySymbol *sym, std::vector<omega::CG_outputRepr *> &index);
  omega::CG_outputRepr*  CreateArrayRefRepr(const IR_ArraySymbol *sym,
                                            std::vector<omega::CG_outputRepr *> &index);

  void CreateDefineMacro(std::string s,std::string args,  omega::CG_outputRepr *repr);
  void CreateDefineMacro(std::string s,std::string args, std::string repr);

  void CreateDefineMacro(std::string s,std::vector<std::string>args, omega::CG_outputRepr *repr);

  int ArrayIndexStartAt() { return 0;} // TODO FORTRAN

  std::vector<IR_ScalarRef *> FindScalarRef(const omega::CG_outputRepr *repr) const;
  std::vector<IR_ArrayRef *> FindArrayRef(const omega::CG_outputRepr *repr) const;
  virtual std::vector<IR_Loop *> FindLoops(omega::CG_outputRepr *repr);
  std::vector<IR_PointerArrayRef *> FindPointerArrayRef(const omega::CG_outputRepr *repr) const;
  IR_PointerArrayRef *CreatePointerArrayRef(IR_PointerSymbol *sym,
                                            std::vector<omega::CG_outputRepr *> &index);

  std::vector<IR_Control *> FindOneLevelControlStructure(const IR_Block *block) const;
  IR_Block *MergeNeighboringControlStructures(const std::vector<IR_Control *> &controls) const;
  bool parent_is_array(IR_ArrayRef *a); // looking for nested array refs??

  bool FromSameStmt(IR_ArrayRef *A, IR_ArrayRef *B);
  void printStmt(const omega::CG_outputRepr *repr);
  int getStmtType(const omega::CG_outputRepr *repr);
  IR_OPERATION_TYPE getReductionOp(const omega::CG_outputRepr *repr);
  IR_Control *  FromForStmt(const omega::CG_outputRepr *repr);

  IR_Block *GetCode() const;
  IR_Control* GetCode(omega::CG_outputRepr*) const; // what is this ???
  void ReplaceCode(IR_Control *old, omega::CG_outputRepr *repr);
  void ReplaceExpression(IR_Ref *old, omega::CG_outputRepr *repr);

  IR_CONDITION_TYPE QueryBooleanExpOperation(const omega::CG_outputRepr*) const;
  IR_OPERATION_TYPE QueryExpOperation(const omega::CG_outputRepr *repr) const;
  std::vector<omega::CG_outputRepr *> QueryExpOperand(const omega::CG_outputRepr *repr) const;
  IR_Ref *Repr2Ref(const omega::CG_outputRepr *) const;

  omega::CG_outputRepr *CreateArrayType(IR_CONSTANT_TYPE type, omega::CG_outputRepr* size);
  omega::CG_outputRepr *CreatePointerType(IR_CONSTANT_TYPE type);
  omega::CG_outputRepr *CreatePointerType(omega::CG_outputRepr *type);
  omega::CG_outputRepr *CreateScalarType(IR_CONSTANT_TYPE type);

  bool ReplaceRHSExpression(omega::CG_outputRepr *code, IR_Ref *ref);

  omega::CG_outputRepr * GetRHSExpression(omega::CG_outputRepr *code);
  omega::CG_outputRepr * GetLHSExpression(omega::CG_outputRepr *code);
  omega::CG_outputRepr *CreateMalloc(const IR_CONSTANT_TYPE type, std::string lhs,
                                     omega::CG_outputRepr * size_repr);
  omega::CG_outputRepr *CreateMalloc  (omega::CG_outputRepr *type, std::string lhs,
                                       omega::CG_outputRepr * size_repr);
  omega::CG_outputRepr *CreateFree( omega::CG_outputRepr *exp);
  friend class IR_chillArraySymbol;
  friend class IR_chillArrayRef;
};



#endif
