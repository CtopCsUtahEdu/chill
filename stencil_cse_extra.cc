/*****************************************************************************
 Copyright (C) 2014 University of Utah
 All Rights Reserved.

 Purpose:
 Looks at intersection of stencil operators to perform stencil common subexpression
 elimination.  

 Notes:
 Similar approach was called Array Subexpression Elminiation (ASE) by the ZPL group
 in their paper "Eliminating Redundancies in Sum-of-Product Array Computations"
 
 
 History:
 2/2014 Created by Protonu Basu.
*****************************************************************************/

#include "loop.hh"
#include "chill_error.hh"
#include <omega.h>
#include "omegatools.hh"
#include <string.h>
#include <code_gen/CG_outputRepr.h>
#include <code_gen/CG_roseRepr.h>
#include <code_gen/CG_roseBuilder.h>
#include "ir_rose.hh"
#include "ir_rose_utils.hh"
#include "rose.h"
#include <set>

using namespace omega;
using namespace SageBuilder;
using namespace SageInterface;

//TODO: I should create a class to hold the stencil operator
//It should store the coefficients, the shape of the stencil
//functions to query its shape and symmetry properties
extern double *** stencil_coeff_3d;
extern double **  stencil_coeff_2d;
extern double *   stencil_coeff_d;


extern char **** char_coeff_3d;
extern void BuildStencilCoefficient(int dim_x, int dim_y, int dim_z);








void Loop::Stencil_ASE_padded(int stmt_num) // enclosing loop outermost
{
  
  debug_fprintf(stderr, "\nLoop::Stencil_ASE_padded(stmt_num %d)\n", stmt_num); 
  //First things first
  //Find the loop bounds of the inner-loop
  
  delete last_compute_cgr_;
  last_compute_cgr_ = NULL;
  delete last_compute_cg_;
  last_compute_cg_ = NULL;
  
  // find the shape 
  BuildStencilCoefficient(1,1,1);
  
  static int call_counter = 0;
  
  IR_ArraySymbol *sym = NULL;
  
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invalid statement number " + to_string(stmt_num));
  
  int depth = stmt[stmt_num].loop_level.size();     // probably 4 nested loops  time + 3 dimensions
  LoopLevel _loop= stmt[stmt_num].loop_level[depth-1]; // innermost?
  
  int init_num_stmts = stmt.size();
  int final_num_stmts;
  
  
  
  //stmt[stmt_num].IS.print();
  //printf("modified transform after applying the xform\n");
  Relation NewIS = getNewIS(stmt_num);
  //NewIS.print();
  
  //Find the inner-loop and get its bounds
  printf("depth of the loop nest is %d\n", depth);
  
  //Let's save the names of the loop indices
  //may not be the most elegant method, but it works
  std::vector<std::string> loop_idxs_names;
  for(int i=0; i<depth; i++)
  {
    printf("the loop indices is %s\n", stmt[stmt_num].IS.set_var(i+1)->name().c_str());
    loop_idxs_names.push_back(stmt[stmt_num].IS.set_var(i+1)->name().c_str());
  }
  
  int variable_pos_newIS = 2*depth;
  int const_bound;
  int inner_loop_lower_bound = -1000;//setting it to unreasonable values for now
  int inner_loop_upper_bound = -1000;
  
  // find the innermost loop bounds 
  
  //How do I just look at the inner-most loop(?)
  //Not pretty, but I'll just go over the xformed IS
  for (DNF_Iterator di(NewIS.query_DNF()); di; di++){
    for (GEQ_Iterator gi= (*di)->GEQs(); gi; gi++) {
      int coef = (*gi).get_coef(NewIS.set_var(variable_pos_newIS));
      if(coef)
      {
        if(coef > 0)//lower bound
        {
          const_bound = (*gi).get_const();
          inner_loop_lower_bound = - (const_bound) / coef; 
        }
        if(coef < 0)//upper bound
        {
          const_bound = (*gi).get_const();
          inner_loop_upper_bound = - (const_bound) / coef; 
        }
      }
    }
  }
  // got inner_loop_upper/lower_bound
  
  
  //Dbg...
  printf("the lower bound of the inner loop is %d\n",inner_loop_lower_bound);
  printf("the upper bound of the inner loop is %d\n",inner_loop_upper_bound);
  
  
  //New Part is to figure out the number of linear buffers to allocate
  //for box-shaped stencils, we should allocate (2*r+1) buffers, where
  //r is the radius of the stencil
  
  
  int  _radius  = 1; //hardcode this for now    get from stencil   -2 to +2   radius is 2
  int  _buffers = 1 + (2* _radius); //hardcode this for now   2*radius+1 
  
  Relation original_IS = copy(stmt[stmt_num].IS);
  std::vector<Statement> temp_statement_buffer;
  
  
  NumLinearStencilBuffers = _buffers;
  StencilBufferSize = (inner_loop_upper_bound - inner_loop_lower_bound ) +1 + 2*_radius;//radius is for padding
  StencilBuffersRequired = _buffers;
  
  
  //Adding buffers to the function body
  SgSymbolTable* parameter_symtab;
  SgSymbolTable* body_symtab;
  SgSymbolTable* root_symtab;
  
  std::vector<SgSymbolTable *> symtabs = ((IR_roseCode *) ir)->getsymtabs();
  
  root_symtab      = symtabs[0];
  parameter_symtab = symtabs[1];
  body_symtab      = symtabs[2];
  
  
  SgFunctionDeclaration * fn= ((IR_roseCode *) ir)->get_func();
  SgScopeStatement* func_body = fn->get_definition()->get_body();
  
  std::string buff("buffer_");
  std::string tmpbuff;
  std::string t_reg("treg_");
  std::string tpreg;
  char num_buf[21];
  
  std::vector<SgVariableSymbol *> buff_syms;
  std::vector<SgVariableSymbol *> treg_syms;
  
  // create definition of the buffers 
  for (int i=0; i<StencilBuffersRequired; i++)
  {
    sprintf(num_buf, "%d",i);
    tmpbuff = buff + num_buf;  // buffer_1, buffer_2 etc
    
    //THIS IS FOR OMP CODE GENERATION ONLY
    if(!call_counter){
      std::string *tp_string = new std::string(tmpbuff);
      omp_thrd_private.insert(tp_string->c_str());
    }
    
    //We should create an array of locks
    //TO DO: the dimension of this array should depend on the input
    SgType * tp = new SgTypeDouble();
    //SgVariableDeclaration *locks_defn = buildVariableDeclaration(tmpbuff.c_str(), buildArrayType(tp, buildIntVal(64)));
    SgVariableDeclaration *locks_defn = buildVariableDeclaration(tmpbuff.c_str(), buildArrayType(tp, buildIntVal(StencilBufferSize)));
    
    
    SgInitializedNamePtrList& _variables = locks_defn->get_variables();
    SgInitializedNamePtrList::const_iterator _j = _variables.begin();
    SgInitializedName* _initializedName = *_j;
    SgVariableSymbol* lcks = new SgVariableSymbol(_initializedName);
    lcks->set_parent(body_symtab);
    body_symtab->insert(tmpbuff.c_str(), lcks );
    
    if(!call_counter)
      prependStatement(locks_defn, func_body);
    
    buff_syms.push_back(lcks);
  }
  //Finish adding the buffer declarations  buffer_[123]
  
  
// this is in stencilinfo coeficients 
  
  //Let's hack something for the constant coefficients
  char *_cffs[4] ={"a","b","c","d"};
  double cf[4] ={-128,14,3,1};
  std::vector<SgVariableSymbol *> scalar_cffs;
  int num_scalar_cffs=4;
  for (int i=0; i<num_scalar_cffs; i++)
  {
    SgType * tp = new SgTypeDouble();
    SgVariableDeclaration *cff_defn = buildVariableDeclaration(_cffs[i], buildDoubleType());
    SgInitializedNamePtrList& _variables = cff_defn->get_variables();
    SgInitializedNamePtrList::const_iterator _j = _variables.begin();
    SgInitializedName* _initializedName = *_j;
    SgVariableSymbol* lcks = new SgVariableSymbol(_initializedName);
    lcks->set_parent(body_symtab);
    body_symtab->insert(_cffs[i], lcks );
    SgStatement * cffdiv = buildAssignStatement(buildVarRefExp(lcks),buildDivideOp(buildDoubleVal(cf[i]), buildDoubleVal(30)));
    if(!call_counter){
      prependStatement(cffdiv, func_body);
      prependStatement(cff_defn, func_body);
    }
    scalar_cffs.push_back(lcks);
  }
  
  
  
  SgVariableSymbol ****scalar_cffs_3d_array;
  scalar_cffs_3d_array = new SgVariableSymbol ***[2*_radius+1];
  for(int i=0; i<2*_radius+1; i++) scalar_cffs_3d_array[i] = new SgVariableSymbol **[2*_radius+1];
  for(int i=0; i<2*_radius+1; i++) {
    for(int j=0; j<2*_radius+1; j++) {
      scalar_cffs_3d_array[i][j] = new SgVariableSymbol *[2*_radius+1];
    }
  }
  
  for(int k=0; k<2*_radius+1; k++){
    for(int j=0; j<2*_radius+1; j++){
      for(int i=0; i<2*_radius+1; i++){
        
        char *s = char_coeff_3d[k][j][i];
        int t;
        for(t=0; t<num_scalar_cffs; t++)
        { if(strcmp(s, scalar_cffs[t]->get_name().str()) == 0) break;}
        
        scalar_cffs_3d_array[k][j][i] = scalar_cffs[t];
      }
    }
  }
  //End Constant Coeff stuff
  
  
  //Adding temporary registers
  //when there is symmetry across x,y,z, and diagonals (27 point stencil)
  //#registers = sum(1+2+...+(radius+1)) = (r+1)(r+2)/2
  
  int num_temp_regs = (_radius+1)*(_radius+2) / 2;  // create treg_# variables as doubles 
  for (int i=0; i<num_temp_regs; i++)
  {
    sprintf(num_buf, "%d",i);
    tpreg = t_reg + num_buf;
    
    
    //THIS IS FOR OMP CODE GENERATION ONLY
    if(!call_counter){
      std::string *tp_string = new std::string(tpreg);
      omp_thrd_private.insert(tp_string->c_str());
    }
    
    //We should create an array of locks
    //TO DO: the dimension of this array should depend on the input
    SgType * tp = new SgTypeDouble();
    SgVariableDeclaration *locks_defn = buildVariableDeclaration(tpreg.c_str(), buildDoubleType());
    
    
    SgInitializedNamePtrList& _variables = locks_defn->get_variables();
    SgInitializedNamePtrList::const_iterator _j = _variables.begin();
    SgInitializedName* _initializedName = *_j;
    SgVariableSymbol* lcks = new SgVariableSymbol(_initializedName);
    lcks->set_parent(body_symtab);
    body_symtab->insert(tpreg.c_str(), lcks );
    if(!call_counter)
      prependStatement(locks_defn, func_body);
    
    treg_syms.push_back(lcks);  // treg_0, treg_1, treg_2 ??? 
  }
  // end of tregs
  
  
  //Make a new Statement
  //Set its iteration space correctly                  // here 
  //Add it to the loop nest
  
  Statement init_pipeline;  // evaluate tregs
  init_pipeline.IS = copy(stmt[stmt_num].IS); 
  init_pipeline.xform = copy(stmt[stmt_num].xform);
  init_pipeline.loop_level = stmt[stmt_num].loop_level;
  init_pipeline.ir_stmt_node = stmt[stmt_num].ir_stmt_node;
  
  // Create the new statement
  CG_outputBuilder *ocg_init = ir->builder();
  std::vector<CG_outputRepr *> index2;
  std::vector<IR_ArrayRef *> access = ir->FindArrayRef(stmt[stmt_num].code);
  
  //Let's get the array ref that is read in this statement   (this is also already in stencilInfo ?)
  IR_ArrayRef *read_;
  IR_ArraySymbol *sym_read;
  std::vector<CG_outputRepr *> index1;
  
  IR_ArrayRef *write_;
  IR_ArraySymbol *sym_write;
  
  IR_ArrayRef *stencil_output;
  IR_ArraySymbol *stencil_output_sym;
  
  
  for (int i = 0; i < access.size(); i++) {
    IR_ArrayRef *a = access[i];
    if(!a->is_write()){
      read_ = a;
      sym_read = a->symbol();
    }
    if(a->is_write()){
      stencil_output = a;
      stencil_output_sym = a->symbol();
    }
  }
  //Read array stuff ends here..
  //Error checking needed in the code above 
  //in case more than one array is read
  
  
  //Let's make a RHS for the initialization here....
  CG_outputRepr *read_ref;
  std::vector<CG_outputRepr *> read_idxs(depth);
  for(int i=1; i<=depth; i++){
    read_idxs[i-1] = ir->builder()->CreateIdent(init_pipeline.IS.set_var(i)->name());
  }
  
  
  
  printf("the array being read is:%s\n", read_->symbol()->name().c_str());
  printf("its dimension is:%d\n", read_->symbol()->n_dim());
  
  printf("WARNING!!:: input array dimension read as:%d but setting it to:%d",read_->symbol()->n_dim(),3 );
  int in_dim = 3;
  
  CG_outputBuilder *ocg = ir->builder();
  
  /**********************************************
   **********************************************/
  std::vector<CG_outputRepr *> indices;
  IR_roseArraySymbol *rose_buffr;
  CG_outputRepr *buf_idx = ir->builder()->CreateIdent(init_pipeline.IS.set_var(depth)->name());
  IR_ArrayRef * buff_arr_ref ;
  
  CG_outputRepr * warmup_pipeline = NULL;
  CG_outputRepr *init_rhs =NULL;
  CG_outputRepr *sum_products_regs =NULL;
  
  int _ofst;
  //Let's get the skeleton up for setting up the initialization code
  for (int r = -_radius; r<_radius; r++){
    
    _ofst = 0;
    int _x =r;
    
    IR_roseScalarSymbol *rose_reg;
    IR_ScalarRef *rose_reg_ref;
    CG_outputRepr *reg_assignments=NULL;
    
    int _ctr =0;
    for (int z=0; z<=_radius; z++){
      for (int y=0; y<=z; y++){
        
        if(z==0 && y==0){
          
          CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
          CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
          CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x)); // i+- 1
          
          std::vector<CG_outputRepr *> _idxs;
          _idxs.push_back(k_idx);
          _idxs.push_back(j_idx);
          _idxs.push_back(i_idx);
          
          CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone());
          _ctr++;
          
          
        }
        else if(y==0 && z!=0)
        {
          
          CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
          CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
          CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x));
          CG_outputRepr *k_idx_positive = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(z));//K+z
          CG_outputRepr *j_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(y));//j+y
          CG_outputRepr *k_idx_negative = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-z));//k-z
          CG_outputRepr *j_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-y));//j-y
          
          //These two are for reflection on diagonal
          CG_outputRepr *jz_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(z));//j+z
          CG_outputRepr *jz_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-z));//j-z
          
          
          //_in[k+z][j][x]
          std::vector<CG_outputRepr *> _idxs;
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx->clone());
          _idxs.push_back(i_idx->clone());
          
          CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          
          
          //_in[k-z][j][x]::reflection on y-axis
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          //_in[k][j+z][x]::reflection on y=z
          _idxs.clear();
          _idxs.push_back(k_idx->clone());
          _idxs.push_back(jz_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          //_in[k][j-z][x]::reflection on y=-z  
          _idxs.clear();
          _idxs.push_back(k_idx->clone());
          _idxs.push_back(jz_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                                ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
          _ctr++;
          
        }
        else if (z==y && z!=0 && y!=0) //points on the diagonal
        {
          CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
          CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
          CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x)); // i+x
          CG_outputRepr *k_idx_positive = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(z));//K+z
          CG_outputRepr *j_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(y));//j+y
          CG_outputRepr *k_idx_negative = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-z));//k-z
          CG_outputRepr *j_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-y));//j-y
          
          //_in[k+z][j+y][x]
          std::vector<CG_outputRepr *> _idxs;
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          
          //_in[k-z][j+y][x]::reflection on y-axis
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          //_in[k-z][j-y][x]::reflection on y=-z
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          //_in[k+z][j-y][x]::reflection on z-axis
          _idxs.clear();
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                                ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
          _ctr++;
        } // diagonal 
      } // for y
    } // for z
    
    if(warmup_pipeline == NULL) warmup_pipeline = reg_assignments->clone();
    else warmup_pipeline = ocg->StmtListAppend( warmup_pipeline->clone(),reg_assignments->clone());
    
    
    for(int n= r; n >= -_radius ; n--){
      
      //The plane we are working on is _in[K][J][r]
      //The plane of coefficients we use: coeff[K][J][n] 
      //The buffer we write to: ....TO FIGURE OUT....
      
      sum_products_regs =NULL;
      
      _ctr =0;
      for (int z=0; z<=_radius; z++){
        for (int y=0; y<=z; y++){
          
          rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          
          IR_ScalarRef *scalar_cff_ref = ir->CreateScalarRef(new IR_roseScalarSymbol (ir, scalar_cffs_3d_array[z+_radius][y+_radius][n+1]));
          
          //CG_outputRepr* read_var = ocg->CreateTimes(rose_reg_ref->convert()->clone(),
          //      ocg->CreateDouble(stencil_coeff_3d[z+_radius][y+_radius][n+1]));
          
          CG_outputRepr* read_var = ocg->CreateTimes(rose_reg_ref->convert()->clone(),
                                                     scalar_cff_ref->convert()->clone());
          
          if (!sum_products_regs) 
            sum_products_regs = read_var;
          else  
            sum_products_regs = ocg->CreatePlus(sum_products_regs, read_var);
          _ctr++;
        }
      }
      
      //Create the buffer and its index  (the LHS of the assignment?) 
      rose_buffr = new IR_roseArraySymbol(ir, buff_syms[n+_radius]);
      CG_outputRepr *idx_offset = ocg->CreatePlus(buf_idx->clone(), ocg->CreateInt(_ofst));
      idx_offset = ocg->CreatePlus(idx_offset->clone(), ocg->CreateInt(-inner_loop_lower_bound));
      _ofst++;
      indices.clear();
      index2.clear();
      indices.push_back(idx_offset);
      buff_arr_ref = ir->CreateArrayRef(rose_buffr,indices);
      write_ = buff_arr_ref;
      
      for (int i = 0; i < write_->n_dim(); i++)
        index2.push_back(write_->index(i));
      
      CG_outputRepr* write_var = ir->CreateArrayRefRepr(write_->symbol(), index2);
      
      //Let's add this statement
      if(!warmup_pipeline)
      {
        warmup_pipeline = ocg->CreateAssignment(0, write_var->clone(),sum_products_regs);
      }else{
        warmup_pipeline = ocg->StmtListAppend(warmup_pipeline->clone(), ocg->CreateAssignment(0, write_var->clone(),sum_products_regs));
      }
    }
  }
  
  
  init_pipeline.code = warmup_pipeline; 
  
  /**********************************************
   **********************************************/
  
  Relation init_xform(stmt[stmt_num].xform.n_out(), stmt[stmt_num].xform.n_out());
  Relation _tmpXform;
  F_And *_rt = init_xform.add_and();
  EQ_Handle eql;
  
  for (int i=1; i<=init_xform.n_out(); i++)
  {
    if(i != 2*depth -1 /*&& i != 2*depth+1*/)
    {
      eql = _rt->add_EQ();
      eql.update_coef(init_xform.output_var(i), 1);
      eql.update_coef(init_xform.input_var(i), -1);
      
    }
    //sets the nesting of the new for-loop
    else if (i == 2*depth-1)
    {
      
      eql = _rt->add_EQ();
      eql.update_coef(init_xform.output_var(i), 1);
      eql.update_coef(init_xform.input_var(i), -1);
      eql.update_const(-1);
    }
    /*
    //nesting of innermost step
    else if (i == 2*depth+1)
    {
    eql = _rt->add_EQ();
    eql.update_coef(init_xform.output_var(i), 1);
    eql.update_coef(init_xform.input_var(i), -1);
    eql.update_const(-1);
    }*/
  }
  
  
  //One similar line, relation to add to the inner most loop 2*depth+1
  Relation shift_depth(stmt[stmt_num].xform.n_out(), stmt[stmt_num].xform.n_out());
  _rt = shift_depth.add_and();
  
  for (int i=1; i<=shift_depth.n_out(); i++)
  {
    if(i != 2*depth+1)
    {
      eql = _rt->add_EQ();
      eql.update_coef(shift_depth.output_var(i), 1);
      eql.update_coef(shift_depth.input_var(i), -1);
      
    }
    //nesting of innermost step
    else if (i == 2*depth+1)
    {
      eql = _rt->add_EQ();
      eql.update_coef(shift_depth.output_var(i), 1);
      eql.update_coef(shift_depth.input_var(i), -1);
      eql.update_const(-1);
    }
  }
  
  //Also modify the iteration space of the new statement
  Relation map_newIS(init_pipeline.IS.n_set(), init_pipeline.IS.n_set());
  _rt = map_newIS.add_and();
  
  for (int i=1; i<=map_newIS.n_out(); i++)
  {
    if(i < depth )
    {
      eql = _rt->add_EQ();
      eql.update_coef(map_newIS.output_var(i), 1);
      eql.update_coef(map_newIS.input_var(i), -1);
      
    }
    else if (i == depth)
    {
      eql = _rt->add_EQ();
      eql.update_coef(map_newIS.output_var(i), 1);
      eql.update_coef(map_newIS.input_var(i), 0);
      eql.update_const(-1 * inner_loop_lower_bound );
      
    }
  }
  init_pipeline.IS = Composition (copy(map_newIS), copy(stmt[stmt_num].IS));
  //init_pipeline.IS =  copy(stmt[stmt_num].IS);
  init_pipeline.IS.simplify();
  
  //hard coding this for now, to change later...
  for(int tp=1; tp<=stmt[stmt_num].IS.n_set(); tp++)
    init_pipeline.IS.name_set_var(tp,loop_idxs_names[tp-1].c_str());
  
  //init_pipeline.IS.name_set_var(1, "k");
  //init_pipeline.IS.name_set_var(2, "j");
  //init_pipeline.IS.name_set_var(3, "i");
  init_pipeline.IS.setup_names();
  
  printf("iteration space of the initializing statement is \n"); init_pipeline.IS.print();
  printf("transforming the xform using the relation:\n"); init_xform.print();
  stmt[stmt_num].xform = Composition(copy(init_xform), copy(stmt[stmt_num].xform));
  
  
  //Create the new statement here:
  /**********************************************
   **********************************************/
  
  //I should also modify the iteration space of the 
  //statement. It should go from lower bound (L) to 
  //Upper bound (U) - 2 *_rafius. 
  //old: L<=IS<=U
  //new: L<=IS<=U-2*_radius
  
  Relation split_IS(stmt[stmt_num].IS.n_set());
  for(int tp=1; tp<=stmt[stmt_num].IS.n_set(); tp++)
    split_IS.name_set_var(tp,loop_idxs_names[tp-1].c_str());
  //split_IS.name_set_var(1, "k");
  //split_IS.name_set_var(2, "j");
  //split_IS.name_set_var(3, "i");
  
  _rt = split_IS.add_and();
  GEQ_Handle geql;
  geql = _rt->add_GEQ();
  geql.update_const(inner_loop_upper_bound-2*_radius);
  geql.update_coef(split_IS.set_var(depth),-1);
  split_IS.print();
  
  //stmt[stmt_num].IS = Intersection(copy(stmt[stmt_num].IS), copy(split_IS));
  
  //End modifying the IS
  
  CG_outputRepr * equilibrium_stmt = NULL;
  _ofst = 0;
  int r = _radius;
  
  //Pick the registers
  int _x =r;
  IR_roseScalarSymbol *rose_reg;
  IR_ScalarRef *rose_reg_ref;
  CG_outputRepr *reg_assignments=NULL;
  int _ctr =0;
  
  for (int z=0; z<=_radius; z++){
    for (int y=0; y<=z; y++){
      
      if(z==0 && y==0){
        
        CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
        CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
        CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x));
        
        std::vector<CG_outputRepr *> _idxs;
        _idxs.push_back(k_idx);
        _idxs.push_back(j_idx);
        _idxs.push_back(i_idx);
        
        CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
        rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
        rose_reg_ref = ir->CreateScalarRef(rose_reg);
        reg_assignments = ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone());
        _ctr++;
        
        
      }
      else if(y==0 && z!=0)
      {
        
        CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
        CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
        CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x));
        CG_outputRepr *k_idx_positive = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(z));//K+z
        CG_outputRepr *j_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(y));//j+y
        CG_outputRepr *k_idx_negative = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-z));//k-z
        CG_outputRepr *j_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-y));//j-y
        
        //These two are for reflection on diagonal
        CG_outputRepr *jz_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(z));//j+z
        CG_outputRepr *jz_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-z));//j-z
        
        //_in[k+z][j][x]
        std::vector<CG_outputRepr *> _idxs;
        _idxs.push_back(k_idx_positive->clone());
        _idxs.push_back(j_idx->clone());
        _idxs.push_back(i_idx->clone());
        
        CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
        
        
        //_in[k-z][j][x]::reflection on y-axis
        _idxs.clear();
        _idxs.push_back(k_idx_negative->clone());
        _idxs.push_back(j_idx->clone());
        _idxs.push_back(i_idx->clone());
        
        read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
        
        
        //_in[k][j+z][x]::reflection on y=z
        _idxs.clear();
        _idxs.push_back(k_idx->clone());
        _idxs.push_back(jz_idx_positive->clone());
        _idxs.push_back(i_idx->clone());
        
        read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
        
        
        //_in[k][j-z][x]::reflection on y=-z  
        _idxs.clear();
        _idxs.push_back(k_idx->clone());
        _idxs.push_back(jz_idx_negative->clone());
        _idxs.push_back(i_idx->clone());
        
        read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
        
        rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
        rose_reg_ref = ir->CreateScalarRef(rose_reg);
        reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                              ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
        _ctr++;
        
      }
      else if (z==y && z!=0 && y!=0) //points on the diagonal
      {
        CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
        CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
        CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x));
        CG_outputRepr *k_idx_positive = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(z));//K+z
        CG_outputRepr *j_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(y));//j+y
        CG_outputRepr *k_idx_negative = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-z));//k-z
        CG_outputRepr *j_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-y));//j-y
        
        //_in[k+z][j+y][x]
        std::vector<CG_outputRepr *> _idxs;
        _idxs.push_back(k_idx_positive->clone());
        _idxs.push_back(j_idx_positive->clone());
        _idxs.push_back(i_idx->clone());
        
        CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
        
        //_in[k-z][j+y][x]::reflection on y-axis
        _idxs.clear();
        _idxs.push_back(k_idx_negative->clone());
        _idxs.push_back(j_idx_positive->clone());
        _idxs.push_back(i_idx->clone());
        
        read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
        
        
        //_in[k-z][j-y][x]::reflection on y=-z
        _idxs.clear();
        _idxs.push_back(k_idx_negative->clone());
        _idxs.push_back(j_idx_negative->clone());
        _idxs.push_back(i_idx->clone());
        
        read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
        
        //_in[k+z][j-y][x]::reflection on z-axis
        _idxs.clear();
        _idxs.push_back(k_idx_positive->clone());
        _idxs.push_back(j_idx_negative->clone());
        _idxs.push_back(i_idx->clone());
        
        read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
        
        rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
        rose_reg_ref = ir->CreateScalarRef(rose_reg);
        reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                              ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
        _ctr++;
      }
    }
  }
  
  if(equilibrium_stmt == NULL) equilibrium_stmt = reg_assignments->clone();
  
  r=_radius;
  for(int n= r; n >= -_radius ; n--){
    
    //The plane we are working on is _in[K][J][r]
    //The plane of coefficients we use: coeff[K][J][n] 
    //The buffer we write to: ....TO FIGURE OUT....
    
    
    sum_products_regs = NULL;
    int _x =r;
    _ctr = 0;
    for (int z=0; z<=_radius; z++){
      for (int y=0; y<=z; y++){
        
        rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
        rose_reg_ref = ir->CreateScalarRef(rose_reg);
        IR_ScalarRef *scalar_cff_ref = ir->CreateScalarRef(new IR_roseScalarSymbol (ir, scalar_cffs_3d_array[z+_radius][y+_radius][n+1]));
        
        //CG_outputRepr* read_var = ocg->CreateTimes(rose_reg_ref->convert()->clone(),
        //      ocg->CreateDouble(stencil_coeff_3d[z+_radius][y+_radius][n+1]));
        
        CG_outputRepr* read_var = ocg->CreateTimes(rose_reg_ref->convert()->clone(),
                                                   scalar_cff_ref->convert()->clone());
        
        if (!sum_products_regs) 
          sum_products_regs = read_var;
        else  
          sum_products_regs = ocg->CreatePlus(sum_products_regs->clone(), read_var);
        _ctr++;
        
        
      }
    }
    
    //Create the buffer and its index
    rose_buffr = new IR_roseArraySymbol(ir, buff_syms[n+_radius]);
    CG_outputRepr *idx_offset = ocg->CreatePlus(buf_idx->clone(), ocg->CreateInt(_ofst));
    idx_offset = ocg->CreatePlus(idx_offset->clone(), ocg->CreateInt(-inner_loop_lower_bound));
    _ofst++;
    indices.clear();
    index2.clear();
    indices.push_back(idx_offset);
    buff_arr_ref = ir->CreateArrayRef(rose_buffr,indices);
    write_ = buff_arr_ref;
    
    for (int i = 0; i < write_->n_dim(); i++)
      index2.push_back(write_->index(i));
    
    CG_outputRepr* write_var = ir->CreateArrayRefRepr(write_->symbol(), index2);
    
    //Let's add this statement
    if(!equilibrium_stmt)
    {
      equilibrium_stmt = ocg->CreateAssignment(0, write_var->clone(),sum_products_regs->clone());
    }else{
      equilibrium_stmt = ocg->StmtListAppend(equilibrium_stmt->clone(), ocg->CreateAssignment(0, write_var->clone(),sum_products_regs->clone()));
    }
  }
  
  stmt[stmt_num].code = equilibrium_stmt; 
  
  /**********************************************
   **********************************************/
  
  temp_statement_buffer.push_back(init_pipeline);
  temp_statement_buffer.push_back(stmt[stmt_num]);
  
  //Statement _swap = stmt[stmt_num];
  //stmt[stmt_num] = init_pipeline;
  //stmt.push_back(_swap);
  
  
  
  /**********************************************
   Create a new statement which sums of the buffers
   which have partially evaluated sums stored in them
  **********************************************/
  
  Statement sum_buffers;
  sum_buffers.IS = copy(temp_statement_buffer[1].IS); 
  sum_buffers.xform = copy(temp_statement_buffer[1].xform);
  sum_buffers.loop_level = temp_statement_buffer[1].loop_level;
  sum_buffers.ir_stmt_node = temp_statement_buffer[1].ir_stmt_node;
  
  //1. Set its lexical order correctly by modifying the xform
  //We can acutally recycle the init_xform created earlier
  
  sum_buffers.xform = Composition(copy(init_xform),copy(temp_statement_buffer[1].xform));
  printf("the xform for the statement to add the buffers\n");
  sum_buffers.xform.print();
  
  //2. Let's build this statement
  //We sum buffers buff_0[i]+buff_1[i]+....
  //We write the result to output[k][j][i]
  init_rhs =NULL;
  for(int n=-_radius; n<=_radius; n++)
  {
    rose_buffr = new IR_roseArraySymbol(ir, buff_syms[n+_radius]);
    indices.clear();
    index2.clear();
    //indices.push_back(buf_idx->clone());
    indices.push_back(ocg->CreatePlus(buf_idx->clone(), ocg->CreateInt(-inner_loop_lower_bound)));
    buff_arr_ref = ir->CreateArrayRef(rose_buffr,indices);
    write_ = buff_arr_ref;
    
    for (int i = 0; i < write_->n_dim(); i++)
      index2.push_back(write_->index(i));
    CG_outputRepr* write_var = ir->CreateArrayRefRepr(write_->symbol(), index2);
    
    if(!init_rhs)
      init_rhs = write_var->clone();
    else
      init_rhs = ocg->CreatePlus(init_rhs->clone(), write_var->clone());
  }
  //The output array in the original stencil operation
  //int offset = depth - read_->n_dim();
  int offset = depth - in_dim;
  std::vector<CG_outputRepr *> temp_idxs;
  for(int i=offset; i<read_idxs.size(); i++)
    temp_idxs.push_back(read_idxs[i]->clone());
  //CG_outputRepr* write_var = ir->CreateArrayRefRepr(stencil_output->symbol(),read_idxs);
  CG_outputRepr* write_var = ir->CreateArrayRefRepr(stencil_output->symbol(),temp_idxs);
  sum_buffers.code = ocg->CreateAssignment(0, write_var->clone(),init_rhs);
  temp_statement_buffer.push_back(sum_buffers);
  
#if 0
  
  /**********************************************
   create the statements for the pipeline breakdown 
  **********************************************/
  
  
  
  _ofst=2*_radius-1;
  int _strt = inner_loop_upper_bound - (2*_radius -1);
  int _end  = inner_loop_upper_bound;
  
  std::vector<Statement> stmt_brkdown;
  for(int i=0; i<2*_radius; i++)
  {
    Statement brk;
    brk.IS = copy(temp_statement_buffer[1].IS); 
    brk.loop_level = temp_statement_buffer[1].loop_level;
    brk.ir_stmt_node = temp_statement_buffer[1].ir_stmt_node;
    brk.xform = Composition(copy(init_xform),copy(sum_buffers.xform)); //copying the sum_buffers xform, will transform it later
    stmt_brkdown.push_back(brk);
  }
  
  for(int _x=_strt; _x<=_end; _x++, _ofst--)
  {
    
    Relation map_newIS(stmt[stmt_num].IS.n_set(), stmt[stmt_num].IS.n_set());
    _rt = map_newIS.add_and();
    
    for (int i=1; i<=map_newIS.n_out(); i++)
    {
      if(i < depth )
      {
        eql = _rt->add_EQ();
        eql.update_coef(map_newIS.output_var(i), 1);
        eql.update_coef(map_newIS.input_var(i), -1);
        
      }
      else if (i == depth)
      {
        eql = _rt->add_EQ();
        eql.update_coef(map_newIS.output_var(i), 1);
        eql.update_coef(map_newIS.input_var(i), 0);
        eql.update_const(-1 * _x );
        
      }
    }
    
    int brk_idx = _x - _strt;
    stmt_brkdown[brk_idx].IS = Composition (copy(map_newIS), copy(stmt_brkdown[brk_idx].IS));
    stmt_brkdown[brk_idx].IS.simplify(); 
    for(int tp=1; tp<=stmt[stmt_num].IS.n_set(); tp++)
      stmt_brkdown[brk_idx].IS.name_set_var(tp,loop_idxs_names[tp-1].c_str());
    //hard coding this for now, to change later...
    //stmt_brkdown[brk_idx].IS.name_set_var(1, "k");
    //stmt_brkdown[brk_idx].IS.name_set_var(2, "j");
    //stmt_brkdown[brk_idx].IS.name_set_var(3, "i");
    stmt_brkdown[brk_idx].IS.setup_names();
    
    printf("the number of the statement to be added is :%d, and its IS is:\n", brk_idx+1);
    stmt_brkdown[brk_idx].IS.print();
    
    CG_outputRepr * brk_pipeline = NULL;
    brk_pipeline = NULL;
    
    //start here... again....
    IR_roseScalarSymbol *rose_reg;
    IR_ScalarRef *rose_reg_ref;
    CG_outputRepr *reg_assignments=NULL;
    int _ctr =0;
    
    for (int z=0; z<=_radius; z++){
      for (int y=0; y<=z; y++){
        
        if(z==0 && y==0){
          
          CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
          CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
          CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(1));
          
          std::vector<CG_outputRepr *> _idxs;
          _idxs.push_back(k_idx);
          _idxs.push_back(j_idx);
          _idxs.push_back(i_idx);
          
          CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone());
          _ctr++;
          
          
        }
        else if(y==0 && z!=0)
        {
          
          CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
          CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
          CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(1));
          CG_outputRepr *k_idx_positive = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(z));//K+z
          CG_outputRepr *j_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(y));//j+y
          CG_outputRepr *k_idx_negative = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-z));//k-z
          CG_outputRepr *j_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-y));//j-y
          
          //These two are for reflection on diagonal
          CG_outputRepr *jz_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(z));//j+z
          CG_outputRepr *jz_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-z));//j-z
          
          //_in[k+z][j][x]
          std::vector<CG_outputRepr *> _idxs;
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx->clone());
          _idxs.push_back(i_idx->clone());
          
          CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          
          
          //_in[k-z][j][x]::reflection on y-axis
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          //_in[k][j+z][x]::reflection on y=z
          _idxs.clear();
          _idxs.push_back(k_idx->clone());
          _idxs.push_back(jz_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          //_in[k][j-z][x]::reflection on y=-z  
          _idxs.clear();
          _idxs.push_back(k_idx->clone());
          _idxs.push_back(jz_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                                ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
          _ctr++;
          
        }
        else if (z==y && z!=0 && y!=0) //points on the diagonal
        {
          CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
          CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
          CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(1));
          CG_outputRepr *k_idx_positive = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(z));//K+z
          CG_outputRepr *j_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(y));//j+y
          CG_outputRepr *k_idx_negative = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-z));//k-z
          CG_outputRepr *j_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-y));//j-y
          
          //_in[k+z][j+y][x]
          std::vector<CG_outputRepr *> _idxs;
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          
          //_in[k-z][j+y][x]::reflection on y-axis
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          //_in[k-z][j-y][x]::reflection on y=-z
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          //_in[k+z][j-y][x]::reflection on z-axis
          _idxs.clear();
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                                ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
          _ctr++;
        }
      }
    }
    
    if(brk_pipeline == NULL) brk_pipeline = reg_assignments->clone();
    
    for(int n=0; n<=_ofst; n++)
    {
      sum_products_regs = NULL;
      _ctr = 0;
      
      for (int z=0; z<=_radius; z++){
        for (int y=0; y<=z; y++){
          
          rose_reg = new IR_roseScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          IR_ScalarRef *scalar_cff_ref = ir->CreateScalarRef(new IR_roseScalarSymbol (ir, 
                                                                                      scalar_cffs_3d_array[z+_radius][y+_radius][(_radius-n)+_radius]));
          
          //CG_outputRepr* read_var = ocg->CreateTimes(rose_reg_ref->convert()->clone(),
          //      ocg->CreateDouble(stencil_coeff_3d[z+_radius][y+_radius][(_radius-n)+_radius]));
          
          CG_outputRepr* read_var = ocg->CreateTimes(rose_reg_ref->convert()->clone(),
                                                     scalar_cff_ref->convert()->clone());
          
          if (!sum_products_regs) 
            sum_products_regs = read_var;
          else  
            sum_products_regs = ocg->CreatePlus(sum_products_regs->clone(), read_var);
          _ctr++;
          
          
        }
      }
      
      
      //Create the buffer and its index
      rose_buffr = new IR_roseArraySymbol(ir, buff_syms[(_radius-n)+_radius]);
      CG_outputRepr *idx_offset = ocg->CreatePlus(buf_idx->clone(), ocg->CreateInt(n));
      idx_offset = ocg->CreatePlus(idx_offset->clone(), ocg->CreateInt(-inner_loop_lower_bound));
      //_ofst++;
      indices.clear();
      index2.clear();
      indices.push_back(idx_offset);
      buff_arr_ref = ir->CreateArrayRef(rose_buffr,indices);
      write_ = buff_arr_ref;
      
      for (int i = 0; i < write_->n_dim(); i++)
        index2.push_back(write_->index(i));
      
      CG_outputRepr* write_var = ir->CreateArrayRefRepr(write_->symbol(), index2);
      
      //Let's add this statement
      if(!brk_pipeline)
      {
        //brk_pipeline = ocg->CreateAssignment(0, write_var->clone(),init_rhs);
      }else{
        brk_pipeline = ocg->StmtListAppend( brk_pipeline->clone(), ocg->CreateAssignment(0, write_var->clone(),sum_products_regs->clone()));
      }
    }
    
    brk_pipeline = ocg->StmtListAppend(brk_pipeline->clone(), sum_buffers.code->clone());
    stmt_brkdown[brk_idx].code = brk_pipeline;
    
  }
  
  //push these new statements for the pipeline breakdown
  //into the stmt
  for(int i=0; i<stmt_brkdown.size(); i++)
  {
    //stmt.push_back(stmt_brkdown[i]);
    temp_statement_buffer.push_back(stmt_brkdown[i]);
  }
  
  //The last part :: a for-loop to add up the buffers
  /*
    Statement sum_buffers_remainder; 
    sum_buffers_remainder.IS = copy(original_IS);
    sum_buffers_remainder.xform = Composition(copy(init_xform),copy(sum_buffers.xform));
    sum_buffers_remainder.xform = Composition(copy(shift_depth),copy(sum_buffers_remainder.xform));
    sum_buffers_remainder.loop_level = sum_buffers.loop_level;
    sum_buffers_remainder.ir_stmt_node = sum_buffers.ir_stmt_node;
    sum_buffers_remainder.code = sum_buffers.code->clone();
    
    Relation remainder_IS(stmt[stmt_num].IS.n_set());
    for(int tp=1; tp<=stmt[stmt_num].IS.n_set(); tp++)
    split_IS.name_set_var(tp,loop_idxs_names[tp-1].c_str());
    
    _rt = remainder_IS.add_and();
    GEQ_Handle geql1;
    geql1 = _rt->add_GEQ();
    geql1.update_const(-1* (inner_loop_upper_bound-(2*_radius-1)));
    geql1.update_coef(remainder_IS.set_var(depth),1);
    remainder_IS.print();
    
    sum_buffers_remainder.IS = Intersection(copy(sum_buffers_remainder.IS), copy(remainder_IS));
    sum_buffers_remainder.IS.print();
    temp_statement_buffer.push_back(sum_buffers_remainder);*/
  
#endif
  
  final_num_stmts = init_num_stmts + temp_statement_buffer.size() -1;
  
  
  //Let's modify the xforms for the statements following the original stmt[stmt_num]
  //None of this is nice....
  int stmts_added = temp_statement_buffer.size() -1;
  for(int i=stmt_num+1; i<stmt.size(); i++)
  {
    int factor_to_increar_lex_order = stmts_added+i;
    for(int t=0; t<factor_to_increar_lex_order; t++)
      stmt[i].xform = Composition(copy(init_xform), copy(stmt[i].xform));
  }
  
  
  //ugly manipulation of vector of statments, this should be improved
  std::vector<Statement> tp_stmt;
  for(int i=0; i<stmt_num; i++) tp_stmt.push_back(stmt[i]);
  for(int i=0; i<temp_statement_buffer.size(); i++) tp_stmt.push_back(temp_statement_buffer[i]);
  for(int i=stmt_num+1; i<stmt.size();i++) tp_stmt.push_back(stmt[i]);
  
  stmt = tp_stmt;
  
  
  /**********************************************
          Update the dependence graph
  **********************************************/
  DependenceGraph g(stmt[stmt_num].IS.n_set());
  
  for(int i=0; i<stmt.size(); i++) 
    g.insert();
  
  for (int i = 0; i < stmt.size(); i++)
    for (int j = i; j < stmt.size(); j++) {
      std::pair<std::vector<DependenceVector>,
                std::vector<DependenceVector> > dv = test_data_dependences(
                  ir, stmt[i].code, stmt[i].IS, stmt[j].code, stmt[j].IS,
                  freevar, index, stmt_nesting_level_[i],
                  stmt_nesting_level_[j]);
      
      for (int k = 0; k < dv.first.size(); k++) {
        
        if (is_dependence_valid_based_on_lex_order(i, j, dv.first[k],true))
          g.connect(i, j, dv.first[k]);
        else {
          g.connect(j, i, dv.first[k].reverse());
        }
      }
      for (int k = 0; k < dv.second.size(); k++)
        if (is_dependence_valid_based_on_lex_order(j, i, dv.second[k],false))
          g.connect(j, i, dv.second[k]);
        else {
          g.connect(i, j, dv.second[k].reverse());
        }
}
  
  dep = g;
  /***********************************************/

  call_counter++;
}


