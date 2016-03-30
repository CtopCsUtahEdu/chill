#ifndef IR_CHILL_HH
#define IR_CHILL_HH

// INTERMEDIATE REPRESENTATION WITH CHILLAST INTERNALS 


#include <omega.h>
#include "ir_code.hh"
#include "chill_error.hh"
#include "chill_ast.hh"


void findmanually( chillAST_node *node, char *procname, vector<chillAST_node*>& procs ); 

extern vector<chillAST_VarDecl *> VariableDeclarations;  // a global.   TODO 

struct IR_chillScalarSymbol: public IR_ScalarSymbol {
  chillAST_VarDecl *chillvd; 

  IR_chillScalarSymbol(const IR_Code *ir, chillAST_VarDecl *vd) {
    fprintf(stderr, "making CHILL scalar symbol %s\n", vd->varname); 
    ir_ = ir;
    chillvd = vd;
  }
  
  std::string name() const;
  int size() const;
  bool operator==(const IR_Symbol &that) const;
  IR_Symbol *clone() const;
};



struct IR_chillArraySymbol: public IR_ArraySymbol {
  //int indirect_;             // what was this? 
  int offset_;                 // what is this? 
  chillAST_VarDecl *chillvd; 

  IR_chillArraySymbol(const IR_Code *ir, chillAST_VarDecl *vd, int offset = 0) {
    //if ( vd == 0 ) 
    //fprintf(stderr, "IR_chillArraySymbol::IR_chillArraySymbol (%s)  vd 0x%x\n", vd->varname, vd); 
    ir_ = ir;
    chillvd = vd; 
    //indirect_ = indirect;
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
  OP_POSITION op_pos_; // -1 means destination operand, 0== unknown, 1 == source operand
  //chillAST_BinaryOperator *bop;  // binary op that contains this scalar? 
  chillAST_DeclRefExpr    *dre;   // declrefexpr that uses this scalar ref, if that exists
  chillAST_VarDecl        *chillvd; // the vardecl for this scalar 
  
  IR_chillScalarRef(const IR_Code *ir, chillAST_BinaryOperator *ins, OP_POSITION pos) {
    fprintf(stderr, "\n*****                         new IR_xxxxScalarRef( ir, ins, pos ) *****\n\n"); 
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
        fprintf(stderr, "IR_chillScalarRef constructor, I'm confused\n"); exit(-1); 
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
        fprintf(stderr, "IR_chillScalarRef constructor, I'm confused\n"); exit(-1); 
      }
    }
    op_pos_ = pos;
  }

  IR_chillScalarRef(const IR_Code *ir, chillAST_DeclRefExpr *d) { 
    // fprintf(stderr, "\n*****                         new IR_xxxxScalarRef( ir, REF EXPR sym %s ) *****\n\n", d->getVarDecl()->varname); 
    //fprintf(stderr, "new IR_chillScalarRef with a DECLREFEXPR  (has dre) \n"); 
    ir_ = ir;
    dre = d;
    //bop = NULL;
    chillvd = d->getVarDecl(); 
    op_pos_ = OP_UNKNOWN; 

    //fprintf(stderr, "\nScalarRef has:\n"); 
    //fprintf(stderr, "assignment op  DOESNT EXIST\n"); 
    //fprintf(stderr, "ins_pos %d\n", ins_pos_); 
    //fprintf(stderr, "op_pos %d\n", op_pos_); 
    //fprintf(stderr, "ref expr dre = 0x%x\n", dre); 
  }

  IR_chillScalarRef(const IR_Code *ir, chillAST_VarDecl *vardecl) { 
    fprintf(stderr, "\n*****                         new IR_xxxxScalarRef( ir, sym 0x1234567 ) ***** THIS SHOULD NEVER HAPPEN\n\n"); 
    fprintf(stderr, "vardecl %s\n", vardecl->varname); 
    ir_ = ir;
    dre = NULL;  fprintf(stderr, "new IR_chillScalarRef with a vardecl but no dre\n"); 
    //bop = NULL;
    chillvd = vardecl; 
    op_pos_ = OP_UNKNOWN; 

    //fprintf(stderr, "\nScalarRef has:\n"); 
    //fprintf(stderr, "assignment op  DOESNT EXIST\n"); 
    //fprintf(stderr, "ins_pos %d\n", ins_pos_); 
    //fprintf(stderr, "op_pos %d\n", op_pos_); 
    //fprintf(stderr, "ref expr dre = 0x%x\n", dre); 
  }

  
  bool is_write() const;
  IR_ScalarSymbol *symbol() const;
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
    fprintf(stderr, "IR_chillArrayRef::IR_chillArrayRef()  write %d\n", write); 
    ir_ = ir;
    chillASE = ase; 
    iswrite = write;  // ase->imwrittento;
    printable = NULL;
  }

  IR_chillArrayRef(const IR_Code *ir, chillAST_ArraySubscriptExpr *ase, const char *printname, int write ) { 
    //fprintf(stderr, "IR_chillArrayRef::IR_chillArrayRef()  write %d\n", write); 
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



struct IR_chillLoop: public IR_Loop {
  int step_size_;

  chillAST_DeclRefExpr *chillindex;   // the loop index variable  (I)  // was DeclRefExpr
  chillAST_ForStmt     *chillforstmt; 
  chillAST_node        *chilllowerbound;
  chillAST_node        *chillupperbound;
  chillAST_node        *chillbody;    // presumably a compound statement, but not guaranteeed
  IR_CONDITION_TYPE conditionoperator;

  IR_chillLoop(const IR_Code *ir, chillAST_ForStmt *forstmt);

  virtual ~IR_chillLoop() {}
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
private:
  chillAST_node *chillAST;             // how about for now we say if there are statements, which is presumably the top level of statements from ... somewhere, otherwise the code is in   chillAST
public:
  vector<chillAST_node *>statements;
  
  // Block is a basic block?? (no, just a chunk of code )
  
  IR_chillBlock() { 
    ir_ = NULL;
    chillAST = NULL; 
  }


  IR_chillBlock(const IR_Code *ir, chillAST_node *ast) { 
    ir_ = ir;
    chillAST = ast; 
  }

  IR_chillBlock(const IR_Code *ir) { //  : cs_(NULL), bDecl_(NULL) {
    chillAST = NULL;
    ir_ = ir;
  }
  
  IR_chillBlock( const IR_chillBlock *CB ) {  // clone existing IR_chillBlock
    ir_ = CB->ir_;
    for (int i=0; i<CB->statements.size(); i++) statements.push_back( CB->statements[i] ); 
    chillAST = CB->chillAST; 
  }


  virtual ~IR_chillBlock() {} // leaves AST and statements intact

  omega::CG_outputRepr *extract() const;
  omega::CG_outputRepr *original() const;
  IR_Control *clone() const;

  vector<chillAST_node*> getStmtList() const;
  int numstatements() { return statements.size(); } ; 
  void addStatement( chillAST_node* s ); 

  void dump() const; 
  
  virtual chillAST_node *getChillAST() const { return chillAST; } 
  virtual void setChillAST( chillAST_node *n) { chillAST = n; }; 
};




struct IR_chillIf: public IR_If {

  chillAST_IfStmt  *chillif; 
  
  IR_chillIf(const IR_Code *ir, chillAST_node *i) {
    fprintf(stderr, "IR_chillIf::IR_chillIf( ir, chillast_node )\n");
    ir_ = ir;
    if (!i->isIfStmt()) { 
      fprintf(stderr, "IR_chillIf::IR_chillIf( ir, chillast_node ) node is not an ifstmt\n");
      fprintf(stderr, "it is a %s\n", i->getTypeString());
      i->print(0, stderr); fprintf(stderr, "\n\n");
    }
    chillif = (chillAST_IfStmt *)i;
  }


  IR_chillIf(const IR_Code *ir, chillAST_IfStmt *i) {
    fprintf(stderr, "IR_chillIf::IR_chillIf( ir, chillast_IfStmt )\n"); 
    ir_ = ir;
    if (!i->isIfStmt()) { 
      fprintf(stderr, "IR_chillIf::IR_chillIf( ir, chillast_IfStmt ) node is not an ifstmt\n");
      fprintf(stderr, "it is a %s\n", i->getTypeString());
      i->print(0, stderr); fprintf(stderr, "\n\n");
    }
    chillif = i;
  }


  virtual ~IR_chillIf() { // leave ast alone
  }


  omega::CG_outputRepr *condition() const;
  IR_Block *then_body() const;
  IR_Block *else_body() const;
  IR_Block *convert();
  IR_Control *clone() const;
};





class IR_chillCode: public IR_Code{   // for an entire file?  A single function? 
protected:

  //  
  char *filename;
  char *outputname;    // so we can output different results from one source, using different scripts

  chillAST_node *entire_file_AST;
  chillAST_FunctionDecl * chillfunc;   // the function we're currenly modifying

  std::vector<chillAST_VarDecl> entire_file_symbol_table;
  // loop symbol table??   for (int i=0;  ... )  ??


public:
  char *procedurename;

  IR_chillCode(); 
  IR_chillCode(const char *filename, char *proc_name);
  IR_chillCode(const char *filename, char *proc_name, char *script_name);
  virtual ~IR_chillCode();

  void setOutputName( const char *name ) { outputname = strdup(name); } 

  IR_ScalarSymbol *CreateScalarSymbol(const IR_Symbol *sym, int i);
  IR_ArraySymbol  *CreateArraySymbol(const IR_Symbol *sym, std::vector<omega::CG_outputRepr *> &size, int i);

  IR_ScalarRef *CreateScalarRef(const IR_ScalarSymbol *sym);
  IR_ArrayRef *CreateArrayRef(const IR_ArraySymbol *sym, std::vector<omega::CG_outputRepr *> &index);
  omega::CG_outputRepr*  CreateArrayRefRepr(const IR_ArraySymbol *sym,
                                            std::vector<omega::CG_outputRepr *> &index) { 
    fprintf(stderr, "IR_chillCode::CreateArrayRefRepr() not implemented\n");
    exit(-1);
    return NULL;
  }


  int ArrayIndexStartAt() { return 0;} // TODO FORTRAN

  std::vector<IR_ScalarRef *> FindScalarRef(const omega::CG_outputRepr *repr) const;
  std::vector<IR_ArrayRef *> FindArrayRef(const omega::CG_outputRepr *repr) const;

  std::vector<IR_PointerArrayRef *> FindPointerArrayRef(const omega::CG_outputRepr *repr) const;

  std::vector<IR_Control *> FindOneLevelControlStructure(const IR_Block *block) const;
  IR_Block *MergeNeighboringControlStructures(const std::vector<IR_Control *> &controls) const;

  IR_Block *GetCode() const;
  void ReplaceCode(IR_Control *old, omega::CG_outputRepr *repr);
  void ReplaceExpression(IR_Ref *old, omega::CG_outputRepr *repr);

  IR_CONDITION_TYPE QueryBooleanExpOperation(const omega::CG_outputRepr*) const;
  IR_OPERATION_TYPE QueryExpOperation(const omega::CG_outputRepr *repr) const;
  std::vector<omega::CG_outputRepr *> QueryExpOperand(const omega::CG_outputRepr *repr) const;
  IR_Ref *Repr2Ref(const omega::CG_outputRepr *) const;




  friend class IR_chillArraySymbol;
  friend class IR_chillArrayRef;
};



#endif
