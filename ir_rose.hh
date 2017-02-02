#ifndef IR_ROSE_HH
#define IR_ROSE_HH

#include <omega.h>
#include <code_gen/CG_chillRepr.h>  // just for CreateArrayRefRepr.  probably a bad idea 

#include "chill_ast.hh"  // needed now that we're going immediately from rose ast to chill ast 
#include "chill_io.hh"

#include "ir_code.hh"
#include "ir_rose_utils.hh"
#include <AstInterface_ROSE.h>
#include "chill_error.hh"
#include "staticSingleAssignment.h"
#include "VariableRenaming.h"
#include "ssaUnfilteredCfg.h"
#include "virtualCFG.h"


// IR_roseCode is built on IR_chillCode
#include "ir_chill.hh" 

// forward declarations 
chillAST_node * ConvertRoseFile(  SgGlobal *sg, const char *filename ); // the entire file 
chillAST_node * ConvertRoseFunctionDecl( SgFunctionDeclaration *D , chillAST_node *parent );
chillAST_node * ConvertRoseParamVarDecl( SgInitializedName *vardecl, chillAST_node *p );
chillAST_node * ConvertRoseInitName( SgInitializedName *vardecl, chillAST_node *p );
chillAST_node * ConvertRoseVarDecl( SgVariableDeclaration *vardecl, chillAST_node *p ); // stupid name TODO 
chillAST_node * ConvertRoseForStatement( SgForStatement *forstatement, chillAST_node *p );
chillAST_node * ConvertRoseExprStatement( SgExprStatement *exprstatement, chillAST_node *p );
chillAST_node * ConvertRoseBinaryOp( SgBinaryOp *binaryop, chillAST_node *p );
chillAST_node * ConvertRoseMemberExpr( SgBinaryOp *binaryop, chillAST_node *); // binop! a.b
chillAST_node * ConvertRoseArrowExp  ( SgBinaryOp *binaryop, chillAST_node *); // binop! a->b
char *          ConvertRoseMember( SgVarRefExp* memb, chillAST_node *base ); // TODO 
chillAST_node * ConvertRoseUnaryOp( SgUnaryOp *unaryop, chillAST_node *p ); 
chillAST_node * ConvertRoseVarRefExp( SgVarRefExp *varrefexp, chillAST_node *p );
chillAST_node * ConvertRoseIntVal( SgIntVal *riseintval, chillAST_node *p );
chillAST_node * ConvertRoseFloatVal( SgFloatVal *rosefloatval, chillAST_node *p );
chillAST_node * ConvertRoseDoubleVal( SgDoubleVal *rosecdoubleval, chillAST_node *p );
chillAST_node * ConvertRoseBasicBlock( SgBasicBlock *bb, chillAST_node *p );
chillAST_node * ConvertRoseFunctionCallExp( SgFunctionCallExp*, chillAST_node *p);
chillAST_node * ConvertRoseReturnStmt( SgReturnStmt *rs, chillAST_node *p );
chillAST_node * ConvertRoseArrayRefExp( SgPntrArrRefExp *roseARE, chillAST_node *p ); 
chillAST_node * ConvertRoseCastExp( SgCastExp *roseCE, chillAST_node *p );
chillAST_node * ConvertRoseAssignInitializer( SgAssignInitializer *roseAI, chillAST_node *p );
// TODO 
chillAST_node * ConvertRoseStructDefinition( SgClassDefinition *def, chillAST_node *p );
chillAST_node * ConvertRoseStructDeclaration( SgClassDeclaration *dec, chillAST_node *p );


chillAST_node * ConvertRoseIfStmt( SgIfStmt *ifstatement, chillAST_node *p); 

chillAST_node * ConvertRoseTypeDefDecl( SgTypedefDeclaration *TDD, chillAST_node * );

//chillAST_node * ConvertRoseRecordDecl( clang::RecordDecl *D, chillAST_node * );
//chillAST_node * ConvertRoseDeclStmt( clang::DeclStmt *clangDS, chillAST_node * );
//chillAST_node * ConvertRoseCompoundStmt( clang::CompoundStmt *clangCS, chillAST_node * );

//chillAST_node * ConvertRoseDeclRefExpr( clang::DeclRefExpr * clangDRE, chillAST_node * );
//chillAST_node * ConvertRoseCStyleCastExpr( clang::CStyleCastExpr *clangICE, chillAST_node * );
//chillAST_node * ConvertRoseIfStmt( clang::IfStmt *clangIS , chillAST_node *);
chillAST_node * ConvertRoseGenericAST( SgNode *n, chillAST_node *parent );


extern vector<chillAST_VarDecl *> VariableDeclarations; 
extern vector<chillAST_FunctionDecl *> FunctionDeclarations; 

// forward definitions. things defined in this file 
struct IR_roseScalarSymbol; 
struct IR_roseArraySymbol;  
struct IR_rosePointerSymbol;
//struct IR_roseConstantRef:  public IR_ConstantRef ;
//struct IR_roseScalarRef:    public IR_ScalarRef ;
//struct IR_roseArrayRef:     public IR_ArrayRef ;
//struct IR_roseLoop:         public IR_Loop ;
//struct IR_roseBlock:        public IR_chillBlock ;
//struct IR_roseIf:           public IR_If ;
//struct IR_roseCCast:        public IR_CCast; 


struct IR_roseScalarSymbol: public IR_ScalarSymbol {
  chillAST_VarDecl *chillvd; 
  IR_roseScalarSymbol(const IR_Code *ir, chillAST_VarDecl *vd) {
    //debug_fprintf(stderr, "making ROSE scalar symbol %s\n", vd->varname); 
    ir_ = ir;
    chillvd = vd; // using chill internals ... 
  }
  
  
  std::string name() const;
  int size() const;
  bool operator==(const IR_Symbol &that) const;
  IR_Symbol *clone() const;
};



struct IR_roseArraySymbol: public IR_ArraySymbol {
 
  chillAST_node  *base;  // usually a vardecl but can be a member expression 
  chillAST_VarDecl *chillvd; 
  
  IR_roseArraySymbol(const IR_Code *ir, chillAST_VarDecl *vd, int offset = 0) {
    //debug_fprintf(stderr, "IR_roseArraySymbol::IR_roseArraySymbol (%s)\n", vd->varname); 
    ir_     = ir;
    base = (chillAST_node *)vd; 
    chillvd = vd;
    //debug_fprintf(stderr, "\nmade new  IR_roseArraySymbol %p\n", this); 
    //offset_ = offset;
  }

  
  IR_roseArraySymbol(const IR_Code *ir, chillAST_node *n, int offset = 0) {
    //debug_fprintf(stderr, "IR_roseArraySymbol::IR_roseArraySymbol (%s)\n", vd->varname); 
    ir_     = ir;
    base = n;
    chillvd = n ->multibase();
    //debug_fprintf(stderr, "\nmade new  IR_roseArraySymbol %p\n", this); 
    //offset_ = offset;
  }

  
  ~IR_roseArraySymbol() { /* debug_fprintf(stderr, "deleting  IR_roseArraySymbol %p\n", this);*/ } 
  
  std::string name() const;  // IR_roseArraySymbol
  int elem_size() const;
  int n_dim() const;
  IR_CONSTANT_TYPE elem_type() const;
  omega::CG_outputRepr *size(int dim) const;          // why not int ? 
  bool operator==(const IR_Symbol &that) const;
  IR_ARRAY_LAYOUT_TYPE layout_type() const;
  IR_Symbol *clone() const;
};


struct IR_rosePointerSymbol: public IR_PointerSymbol {
  chillAST_VarDecl *chillvd; 

  std::string name_;  // these could be gotten by looking at the vardecl
  int dim_;
  std::vector<omega::CG_outputRepr *> dims; // ??? 

 	IR_rosePointerSymbol(const IR_Code *ir, chillAST_VarDecl *v ) {
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




struct IR_roseConstantRef: public IR_ConstantRef {
  union {
    omega::coef_t i_;
    double f_;
  };
  
  IR_roseConstantRef(const IR_Code *ir, omega::coef_t i) {
    ir_ = ir;
    type_ = IR_CONSTANT_INT;
    i_ = i;
  }
  IR_roseConstantRef(const IR_Code *ir, double f) {
    ir_ = ir;
    type_ = IR_CONSTANT_FLOAT;
    f_ = f;
  };
  omega::coef_t integer() const {
    assert(is_integer());
    return i_;
  };
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
  
};

struct IR_roseScalarRef: public IR_ScalarRef {
  OP_POSITION op_pos_; // -1 means destination operand, 0== unknown, 1 == source operand
  chillAST_DeclRefExpr    *dre;   // declrefexpr that uses this scalar ref, if that exists
  chillAST_VarDecl        *chillvd; // the vardecl for this scalar 
  
  int is_write_;
  
  
  IR_roseScalarRef(const IR_Code *ir, chillAST_DeclRefExpr *d) { 
    ir_ = ir;
    dre = d;
    //bop = NULL;
    chillvd = d->getVarDecl(); 
    op_pos_ = OP_UNKNOWN; 
  }
  
  IR_roseScalarRef(const IR_Code *ir, chillAST_VarDecl *vd) { 
    ir_ = ir;
    dre = NULL;  // ?? 
    //bop = NULL;
    chillvd = vd;
    op_pos_ = OP_UNKNOWN; 
  }
  
  
  bool is_write() const;
  IR_ScalarSymbol *symbol() const;
  bool operator==(const IR_Ref &that) const;
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
};



struct IR_roseArrayRef: public IR_ArrayRef {
  chillAST_ArraySubscriptExpr* chillASE; 
  bool iswrite; 
  
  IR_roseArrayRef(const IR_Code *ir, chillAST_ArraySubscriptExpr *ase, bool write ) { 
    //debug_fprintf(stderr, "IR_XXXXArrayRef::IR_XXXXArrayRef() '%s' write %d\n\n", ase->basedecl->varname, write); 
    ir_ = ir;
    chillASE = ase; 
    // dies? ase->dump(); fflush(stdout); 
    
    iswrite = write;  // ase->imwrittento;
  }
  
  
  bool is_write() const;
  omega::CG_outputRepr *index(int dim) const;
  IR_ArraySymbol *symbol() const;
  bool operator==(const IR_Ref &that) const;
  bool operator!=(const IR_Ref &that) const; // not the opposite logic to ==     TODO 
  omega::CG_outputRepr *convert();
  IR_Ref *clone() const;
  virtual void Dump() const;
};

struct IR_rosePointerArrayRef: public IR_PointerArrayRef { // exactly the same as arrayref ??? 
  chillAST_ArraySubscriptExpr* chillASE; 
  int iswrite; 
  
  IR_rosePointerArrayRef(const IR_Code *ir, chillAST_ArraySubscriptExpr *ase, bool write ) { 
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




struct IR_roseLoop: public IR_Loop {
  int step_size_;
  
  chillAST_DeclRefExpr *chillindex;   // the loop index variable  (I)  // was DeclRefExpr
  chillAST_ForStmt     *chillforstmt; 
  chillAST_node        *chilllowerbound;
  chillAST_node        *chillupperbound;
  chillAST_node        *chillbody;    // presumably a compound statement, but not guaranteeed
  IR_CONDITION_TYPE conditionoperator;
  
  IR_roseLoop(const IR_Code *ir, chillAST_node *forstmt);
  
  IR_ScalarSymbol *index() const;
  omega::CG_outputRepr *lower_bound() const;
  omega::CG_outputRepr *upper_bound() const;
  IR_CONDITION_TYPE stop_cond() const;
  IR_Block *body() const;
  IR_Block *convert();
  int step_size() const;  
  IR_Control *clone() const;
};

extern SgProject *OneAndOnlySageProject;  // a global

class IR_roseCode: public IR_chillCode {
protected:
  // things that start with Sg are rose-specific (Sage, actually)
  SgProject* project;
  SgSourceFile* file;
  SgGlobal *root;
  SgGlobal *firstScope;
  SgSymbolTable* symtab_;
  SgSymbolTable* symtab2_;
  SgSymbolTable* symtab3_;
  SgDeclarationStatementPtrList::iterator p;
  SgFunctionDeclaration *func;
  
  
  bool is_fortran_;
  int i_;
  StaticSingleAssignment *ssa_for_scalar;
  ssa_unfiltered_cfg::SSA_UnfilteredCfg *main_ssa;
  VariableRenaming *varRenaming_for_scalar;
  
  
  std::vector<chillAST_VarDecl> entire_file_symbol_table;
  
  // TODO yeah, these need to be associated with a sourcefile ?? 
  std::map<std::string, chillAST_node *> defined_macros;  // TODO these need to be in a LOCATION 
  
public:

  void print() { chillfunc->print(); printf("\n"); fflush(stdout); }; 
  
  IR_roseCode(const char *filename, const char* proc_name, const char* dest_name = NULL );
  ~IR_roseCode();

  int ArrayIndexStartAt() {
    if (is_fortran_)
      return 1;
    else
      return 0;
  }
  
  void populateLists(SgNode* tnl_1, SgStatementPtrList* list_1,
                     SgStatementPtrList& output_list_1);
  void populateScalars(const omega::CG_outputRepr *repr1,
                       std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
                       std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
                       std::set<std::string> &indices, std::vector<std::string> &index);
  //        std::set<std::string> &def_vars);
  /*void findDefinitions(SgStatementPtrList &list_1,
    std::set<VirtualCFG::CFGNode> &reaching_defs_1,
    std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
    std::set<std::string> &def_vars);
  */
  /*    void checkDependency(SgStatementPtrList &output_list_1,
        std::vector<DependenceVector> &dvs1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
        std::vector<std::string> &index, int i, int j);
        void checkSelfDependency(SgStatementPtrList &output_list_1,
        std::vector<DependenceVector> &dvs1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
        std::vector<std::string> &index, int i, int j);
        void checkWriteDependency(SgStatementPtrList &output_list_1,
        std::vector<DependenceVector> &dvs1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &read_scalars_1,
        std::map<SgVarRefExp*, IR_ScalarRef*> &write_scalars_1,
        std::vector<std::string> &index, int i, int j);
  */
  std::vector<IR_ScalarRef *> FindScalarRef(const omega::CG_outputRepr *repr) const;
  bool parent_is_array(IR_ArrayRef *a); // looking for nested array refs??
  
  IR_Block*   GetCode() const;
  IR_Control* GetCode(omega::CG_outputRepr*) const; // what is this ??? 
  
  IR_OPERATION_TYPE QueryExpOperation(const omega::CG_outputRepr *repr) const;
  IR_CONDITION_TYPE QueryBooleanExpOperation(
                                             const omega::CG_outputRepr *repr) const;
  std::vector<omega::CG_outputRepr *> QueryExpOperand(
                                                      const omega::CG_outputRepr *repr) const;
  /*    std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> >
        FindScalarDeps(const omega::CG_outputRepr *repr1,
        const omega::CG_outputRepr *repr2, std::vector<std::string> index,
        int i, int j);
  */
  void finalizeRose();
  
  // Manu:: Added functions required for reduction operation
  // virtual omega::CG_outputRepr * FromSameStmt(IR_ArrayRef *A, IR_ArrayRef *B) = 0;
  bool FromSameStmt(IR_ArrayRef *A, IR_ArrayRef *B);
  void printStmt(const omega::CG_outputRepr *repr);
  int getStmtType(const omega::CG_outputRepr *repr);
  IR_OPERATION_TYPE getReductionOp(const omega::CG_outputRepr *repr);
  IR_Control *  FromForStmt(const omega::CG_outputRepr *repr);
  
  // Manu:: Added functions for scalar expansion
  // TODO   
  IR_PointerArrayRef *CreatePointerArrayRef(IR_PointerSymbol *sym,
                                            std::vector<omega::CG_outputRepr *> &index); 
  void CreateDefineMacro(std::string s,std::string args,  omega::CG_outputRepr *repr);
  void CreateDefineMacro(std::string s,std::string args, std::string repr);
  
  void CreateDefineMacro(std::string s,std::vector<std::string>args, omega::CG_outputRepr *repr);
  
  omega::CG_outputRepr *CreateArrayType(IR_CONSTANT_TYPE type, omega::CG_outputRepr* size);
  omega::CG_outputRepr *CreatePointerType(IR_CONSTANT_TYPE type);
  omega::CG_outputRepr *CreatePointerType(omega::CG_outputRepr *type);
  omega::CG_outputRepr *CreateScalarType(IR_CONSTANT_TYPE type);
  
  //std::vector<IR_PointerArrayRef *> FindPointerArrayRef(const omega::CG_outputRepr *repr) const; // inherit from chillcode ?? 
  
  
  bool ReplaceRHSExpression(omega::CG_outputRepr *code, IR_Ref *ref);
  bool ReplaceLHSExpression(omega::CG_outputRepr *code, IR_ArrayRef *ref);
  omega::CG_outputRepr * GetRHSExpression(omega::CG_outputRepr *code);
  omega::CG_outputRepr * GetLHSExpression(omega::CG_outputRepr *code);
  omega::CG_outputRepr *CreateMalloc(const IR_CONSTANT_TYPE type, std::string lhs,
                                     omega::CG_outputRepr * size_repr);
  omega::CG_outputRepr *CreateMalloc  (omega::CG_outputRepr *type, std::string lhs,
                                       omega::CG_outputRepr * size_repr);
  omega::CG_outputRepr *CreateFree( omega::CG_outputRepr *exp);
  
  //static int rose_pointer_counter ; // for manufactured arrays
  
  friend class IR_roseArraySymbol;
  friend class IR_roseArrayRef;
};


#endif
