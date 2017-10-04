


#include "loop.hh"
#include "omegatools.hh"
#include "ir_code.hh"
#include "chill_error.hh"

using namespace omega;

void Loop::stencilASEPadded(int stmt_num)   {
  
  debug_fprintf(stderr, "\nLoop::stencilASEPadded( stmt_num %d )   loop_stencil.cc\n", stmt_num); 
  //debug_fprintf(stderr, "stmt.size()  %d\n", stmt.size()); 
  //debug_fprintf(stderr, "uninterpreted  %d\n", uninterpreted_symbols.size()); 
  //for(int i=0; i<stmt_num; i++) debug_fprintf(stderr, "stmt %d  UNIN SYM map size %d\n", i,  uninterpreted_symbols[i].size()); 
  
  //First things first
  invalidateCodeGen();

  // find the stencil shape  
  find_stencil_shape( stmt_num ); 
  debug_fprintf(stderr, "found stencil shape\n\n"); 
  
  // this was a test, making the stencil NOT symmetric 
  //std::vector< chillAST_node* > acoeff;
  //acoeff.push_back( (chillAST_node *) (new chillAST_FloatingLiteral( 4.0f, NULL )));
  //stmt[stmt_num].statementStencil->coefficients.insert( stmt[stmt_num].statementStencil->coefficients.begin(), acoeff);
  //stmt[stmt_num].statementStencil->coefficients.pop_back(); 



  stmt[stmt_num].statementStencil->print(); fflush(stdout); 
  stmt[stmt_num].statementStencil->isSymmetric(); 
  
  // get chill_ast?  or do we keep Loop pure ?
  omega::CG_chillRepr *CR = (omega::CG_chillRepr *) stmt[stmt_num].code;
  chillAST_node *chillcode = CR->GetCode();
  //debug_fprintf(stderr, "chillcode (%s):\n", chillcode->getTypeString()); chillcode->print(); printf("\n(END)\n\n"); fflush(stdout);
  
  int depth = stmt[stmt_num].loop_level.size();
  //debug_fprintf(stderr, "depth %d\n", depth);
  
  LoopLevel _loop= stmt[stmt_num].loop_level[depth-1]; // innermost?
  
  int init_num_stmts = stmt.size();
  int final_num_stmts;
  
  
  // use Omega to get loop index names ?? 
  std::vector<std::string> loop_idxs_names;
  
  //debug_fprintf(stderr, "gathering loop indeces\n");
  std::vector<chillAST_VarDecl*> indeces; 
  chillcode->gatherLoopIndeces( indeces ); 
  //debug_fprintf(stderr, "\nchill version of loop indeces (%d of them):\n", indeces.size() ); 
  //for (int i=0; i<indeces.size(); i++) {
  //  indeces[i]->print(); printf(" %p\n", indeces[i]); fflush(stdout); 
  //} 
  //debug_fprintf(stderr, "\n"); 
  
  
  
  omega::Relation NewIS = getNewIS(stmt_num);  // omega 
  for(int i=0; i<depth; i++)
    {
      //printf("the omega loop index %d is %s\n", i, stmt[stmt_num].IS.set_var(i+1)->name().c_str());
      loop_idxs_names.push_back(stmt[stmt_num].IS.set_var(i+1)->name().c_str());
    }
  fflush(stdout);
  
  
  
  chillAST_ForStmt *innermost = chillcode->findContainingLoop();
  if (!innermost) { 
    debug_fprintf(stderr, "stencil code is not in any loops?\n");
    exit(-1);
  }
  

  //debug_fprintf(stderr, "\ninnermost loop is "); innermost->printControl(); printf("\n\n"); fflush(stdout); 
  
  int upper, lower;
  bool worked = innermost->upperBound( upper );
  if (!worked) { 
    debug_fprintf(stderr, "could not find upper bound\n"); 
  }
  worked = innermost->lowerBound( lower );
  if (!worked) { 
    debug_fprintf(stderr, "could not find lower bound bound\n"); 
  }
  
  //Dbg...
  printf("the lower bound of the inner loop is %d\n", lower);
  printf("the upper bound of the inner loop is %d\n", upper); 
  fflush(stdout); 
  
  int radius = stmt[stmt_num].statementStencil->radius();
  debug_fprintf(stderr, "radius of stencil is %d\n", radius); 
  
  int numbuffers = 1 + 2*radius;
  //debug_fprintf(stderr, "we'll allocate %d linear buffers\n", numbuffers);
  
  //Relation original_IS = copy(stmt[stmt_num].IS); // not used 
  
  int paddingSize = 2 * radius;
  int stencilBufferSize = 1 + paddingSize + upper - lower;
  //debug_fprintf(stderr, "padding %d    stencilBufferSize %d\n\n", paddingSize, stencilBufferSize);
  chillAST_NodeList arraypart;
  arraypart.push_back(new chillAST_IntegerLiteral(stencilBufferSize));

  // find the function we're modifying
  //easy, but probably risky ...   chillcode->findEnclosingFunction() 
  IR_chillCode *IR_RC = (IR_chillCode *)ir;  // use hidden info that this is IR_roseCode;
  chillAST_FunctionDecl *fd = IR_RC->chillfunc; // the function we're modifying
  
  
  // define the linear buffers buffer_1, buffer_2, etc
  std::vector<chillAST_VarDecl *> buff_syms; 
  for (int i=0; i<numbuffers; i++) { 
    char vname[128];
    sprintf(vname, "buffer_%d", i); // variable name 
    
    chillAST_VarDecl *vd = new chillAST_VarDecl( "double", "", vname, arraypart);
    //vd->print(0, stderr); debug_fprintf(stderr, ";\n");
    
    // add to function we're modifying
    fd->getBody()->insertChild(i, vd); // adds decl as the first staement in the function body
    fd->addDecl( vd ); // adds to symbol table
    
    buff_syms.push_back( vd ); 
  }
  
  // define the temps  (treg_1, treg_2, etc) 
  debug_fprintf(stderr, "Adding temporary registers\n");
  std::vector<chillAST_VarDecl *> treg_syms; 
  int num_temp_registers = (radius+1) * (radius+2) / 2;
  debug_fprintf(stderr, "\n%d temp registers\n", num_temp_registers );

  for (int i=0; i<num_temp_registers; i++) {     
    char vname[128];
    sprintf(vname, "treg_%d", i); // variable name 
    
    debug_fprintf(stderr, "%s\n", vname);
      

    chillAST_VarDecl *vd = new chillAST_VarDecl( "double", "", vname );
    //vd->print(0, stderr); debug_fprintf(stderr, ";\n");
    
    // add to function we're modifying
    fd->getBody()->insertChild(i, vd); // adds decl as the first staement in the function body
    fd->addDecl( vd ); // adds to symbol table
    
    treg_syms.push_back( vd ); 
  }
  
  stencilInfo *SI = stmt[stmt_num].statementStencil;
  chillAST_VarDecl *src = SI->srcArrayVariable;
  chillAST_VarDecl *dst = SI->dstArrayVariable;

  //Make a new Statement
  //Set its iteration space correctly                  // here 
  //Add it to the loop nest
  
  Statement init_pipeline;  // evaluate tregs
  init_pipeline.IS           = copy(stmt[stmt_num].IS); 
  init_pipeline.xform        = copy(stmt[stmt_num].xform);
  init_pipeline.loop_level   = stmt[stmt_num].loop_level;
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
    //debug_fprintf(stderr, "variable(?) %s\n", init_pipeline.IS.set_var(i)->name().c_str());
    read_idxs[i-1] = ir->builder()->CreateIdent(init_pipeline.IS.set_var(i)->name());
  }
  
  printf("\nthe array being written is:%s\n", stencil_output->symbol()->name().c_str());
  printf("its dimension is:%d\n\n", stencil_output->symbol()->n_dim());
  
  printf("the array being read is:%s\n", read_->symbol()->name().c_str());
  printf("its dimension is:%d\n\n", read_->symbol()->n_dim());
  fflush(stdout); 

  if (read_->symbol()->n_dim() != 3) { 
    printf("WARNING!!:: input array dimension read as:%d but setting it to:%d",read_->symbol()->n_dim(),3 );
  }
  fflush(stdout); 

  int in_dim = 3;
  
  CG_outputBuilder *ocg = ir->builder();
  fflush(stdout); 
  
  /**********************************************   A
   **********************************************/
  std::vector<CG_outputRepr *> indices;
  IR_chillArraySymbol *rose_buffr;
  CG_outputRepr *buf_idx = ir->builder()->CreateIdent(init_pipeline.IS.set_var(depth)->name());
  IR_ArrayRef * buff_arr_ref ;
  
  CG_outputRepr * warmup_pipeline = NULL;
  CG_outputRepr *init_rhs =NULL;
  CG_outputRepr *sum_products_regs =NULL;
  
  int _ofst;
  //Let's get the skeleton up for setting up the initialization code
  for (int r = -radius; r<radius; r++){
    
    //debug_fprintf(stderr, "\n"); 
    _ofst = 0;
    int _x =r;
    
    IR_chillScalarSymbol *rose_reg;
    IR_ScalarRef *rose_reg_ref;
    CG_outputRepr *reg_assignments=NULL;
    
    int _ctr =0;
    for (int z=0; z<=radius; z++){
      for (int y=0; y<=z; y++){
        //debug_fprintf(stderr, "z %d   y %d\n", z, y); 

        debug_fprintf(stderr, "r %d   z %d   y %d\n", r, z, y); 
        
        if(z==0 && y==0){  
          debug_fprintf(stderr, "\ncenter point\n"); 
          CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
          CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
          CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x)); // i+x
          
          std::vector<CG_outputRepr *> _idxs;
          _idxs.push_back(k_idx);
          _idxs.push_back(j_idx);
          _idxs.push_back(i_idx);
          
          CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]); // treg_syms is std::vector<SgVariableSymbol *>
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(), read_var->clone());
          
          if (r<0) 
            debug_fprintf(stderr, "%s = %s[k][j][i-%d]\n",  treg_syms[_ctr]->varname, read_->symbol()->name().c_str(), abs(r)); 
          else if (r == 0) 
            debug_fprintf(stderr, "%s = %s[k][j][i]\n",  treg_syms[_ctr]->varname, read_->symbol()->name().c_str()); 
          else if (r>0) 
            debug_fprintf(stderr, "%s = %s[k][j][i+%d]\n",  treg_syms[_ctr]->varname, read_->symbol()->name().c_str(), r); 
          
          _ctr++;
        }
        else if(y==0 && z!=0)
          {
            debug_fprintf(stderr, "\nabove center\n"); 
            CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
            CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
            CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x));
            CG_outputRepr *k_idx_positive = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(z));//k+z
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
            
            rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]);
            rose_reg_ref = ir->CreateScalarRef(rose_reg);
            reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                                  ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
            
            //if (r<0) 
            //  debug_fprintf(stderr, "%s = %s[k][j][i-%d] + ...\n",  treg_syms[_ctr]->varname, read_->symbol()->name().c_str(), abs(r)); 
            //else if (r == 0) 
            //  debug_fprintf(stderr, "%s = %s[k][j][i] + ...\n",  treg_syms[_ctr]->varname, read_->symbol()->name().c_str()); 
            //else if (r>0) 
            //  debug_fprintf(stderr, "%s = %s[k][j][i+%d] + ...\n",  treg_syms[_ctr]->varname, read_->symbol()->name().c_str(), r); 
            
            _ctr++;
            
          }
        else if (z==y && z!=0)  //  && y!=0) //points on the diagonal
          {
            debug_fprintf(stderr, "\non diagonal\n");
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
            
            rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]);
            rose_reg_ref = ir->CreateScalarRef(rose_reg);
            reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                                  ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
            
            if (r<0) 
              debug_fprintf(stderr, "%s = %s[k][j][i-%d] + ...\n",  treg_syms[_ctr]->varname, read_->symbol()->name().c_str(), abs(r)); 
            else if (r == 0) 
              debug_fprintf(stderr, "%s = %s[k][j][i] + ...\n",  treg_syms[_ctr]->varname, read_->symbol()->name().c_str()); 
            else if (r>0) 
              debug_fprintf(stderr, "%s = %s[k][j][i+%d] + ...\n",  treg_syms[_ctr]->varname, read_->symbol()->name().c_str(), r); 
            
            _ctr++;
          } // diagonal
        else if (z!=y && z!=0 && y!=0)        {
          debug_fprintf(stderr, "SHOULD NOT GET HERE\n"); 
          CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
          CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
          CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x));
          CG_outputRepr *k_idx_positive = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(z));//K+z
          CG_outputRepr *j_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(y));//j+y
          CG_outputRepr *k_idx_negative = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-z));//k-z
          CG_outputRepr *j_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-y));//j-y
          
          CG_outputRepr *k_idx_positive_y = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(y));//K+y
          CG_outputRepr *j_idx_positive_z = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(z));//j+z
          CG_outputRepr *k_idx_negative_y = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-y));//K-y
          CG_outputRepr *j_idx_negative_z = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-z));//J-z
          
          
          //_in[k+z][j+y][x]
          std::vector<CG_outputRepr *> _idxs;
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          
          //_in[k+y][j+z][x]
          _idxs.clear();
          _idxs.push_back(k_idx_positive_y->clone());
          _idxs.push_back(j_idx_positive_z->clone());
          _idxs.push_back(i_idx->clone());
          
          //read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          //_in[k-z][j+y][x]::reflection on y-axis
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          //_in[k-y][j+z][x]::reflection on y-axis
          _idxs.clear();
          _idxs.push_back(k_idx_negative_y->clone());
          _idxs.push_back(j_idx_positive_z->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          
          //_in[k-z][j-y][x]::reflection on y=-z
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          //_in[k-y][j-z][x]::reflection on y=-z
          _idxs.clear();
          _idxs.push_back(k_idx_negative_y->clone());
          _idxs.push_back(j_idx_negative_z->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          
          //_in[k+z][j-y][x]::reflection on z-axis
          _idxs.clear();
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          //_in[k+y][j-z][x]::reflection on z-axis
          _idxs.clear();
          _idxs.push_back(k_idx_positive_y->clone());
          _idxs.push_back(j_idx_negative_z->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                                ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
          _ctr++;
        } // not side, up, or diagonal 
        
      } // for y 
    } // for z 
    
    // create linear buffer terms  buffer_0, buffer_1, etc  
    
    debug_fprintf(stderr, "warmup_pipeline\n"); 
    if(warmup_pipeline == NULL) warmup_pipeline = reg_assignments->clone();
    else warmup_pipeline = ocg->StmtListAppend( warmup_pipeline->clone(),reg_assignments->clone());
    
    debug_fprintf(stderr, "\ncreating buffer_?[ X ] lines\n"); 
    for(int n = r; n >= -radius ; n--){
      debug_fprintf(stderr, "n %d\n", n); 

      //The plane we are working on is _in[K][J][r]
      //The plane of coefficients we use: coeff[K][J][n] 
      //The buffer we write to: ....TO FIGURE OUT....
      
      sum_products_regs =NULL;
      
      //debug_fprintf(stderr, "%s = ...\n", buff_syms[n+radius]->varname); 
      
      _ctr =0;
      
      //char *t = strdup("float");
      //char *nam = strdup("buh");
      //char *arr = strdup(""); 
      //chillAST_VarDecl *TEMP = new chillAST_VarDecl(t, nam, arr, (chillAST_node *)NULL); 
      
      for (int z=0; z<=radius; z++){
        for (int y=0; y<=z; y++){
          debug_fprintf(stderr, "z %d   y %d\n", z, y); 

          //debug_fprintf(stderr, "n %d   z %d   y %d\n", n, z, y ); 
          
          // get the coefficient ??  (as a chillAST_node)
          chillAST_node *coeff = SI->find_coefficient(n, y, z ); // wrong
          if (coeff) { 
            
            //coeff->print(0, stderr); debug_fprintf(stderr, "\n"); 
            //coeff->dump(); printf("\n\n"); fflush(stdout);
            
            rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]);
            rose_reg_ref = ir->CreateScalarRef(rose_reg);
            
            //IR_ScalarRef *scalar_cff_ref = ir->CreateScalarRef(new IR_roseScalarSymbol (ir, 
            //                                                                            TEMP)); // scalar_cffs_3d_array[z+radius][y+radius][n+1]));
            
            
            CG_outputRepr* read_var = ocg->CreateTimes(rose_reg_ref->convert()->clone(),
                                                       new CG_chillRepr( coeff ) ); // ?? 
            
            if (!sum_products_regs) 
              sum_products_regs = read_var;
            else  
              sum_products_regs = ocg->CreatePlus(sum_products_regs, read_var);
            
          }
          else debug_fprintf(stderr, "there was no coeff for  n %d   z %d   y %d\n", n, z, y ); 
          
          _ctr++;
        }
      }
      
      //Create the buffer and its index
      rose_buffr = new IR_chillArraySymbol(ir, buff_syms[n+radius]);
      CG_outputRepr *idx_offset = ocg->CreatePlus(buf_idx->clone(), ocg->CreateInt(_ofst));
      idx_offset = ocg->CreatePlus(idx_offset->clone(), ocg->CreateInt(-lower));
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
  
  /**********************************************   B
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
      //set's the nesting of the new for-loop
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
  
  for (int i=1; i<=map_newIS.n_out(); i++) {
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
        eql.update_const(-1 * lower );
        
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
  fflush(stdout);
  printf("transforming the xform using the relation:\n"); init_xform.print();
  fflush(stdout);

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
  geql.update_const(upper-2*radius);
  geql.update_coef(split_IS.set_var(depth),-1);
  split_IS.print();
  
  //stmt[stmt_num].IS = Intersection(copy(stmt[stmt_num].IS), copy(split_IS));
  
  //End modifying the IS
  
  CG_outputRepr * equilibrium_stmt = NULL;
  _ofst = 0;
  int r = radius;
  
  //Pick the registers
  int _x =r;
  IR_chillScalarSymbol *rose_reg;
  IR_ScalarRef *rose_reg_ref;
  CG_outputRepr *reg_assignments=NULL;
  int _ctr =0;
  
  for (int z=0; z<=radius; z++){
    for (int y=0; y<=z; y++){
      
      debug_fprintf(stderr, "z %d   y %d\n", z, y); 
      if(z==0 && y==0){
        
        CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
        CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
        CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x));
        
        std::vector<CG_outputRepr *> _idxs;
        _idxs.push_back(k_idx);
        _idxs.push_back(j_idx);
        _idxs.push_back(i_idx);
        
        CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
        rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]);
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
          
          rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]);
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
          
          rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                                ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
          _ctr++;
        }
      else if (z!=y && z!=0 && y!=0) //points on the diagonal
        {
          CG_outputRepr *k_idx = read_idxs[depth-3]->clone();//k
          CG_outputRepr *j_idx = read_idxs[depth-2]->clone();//j
          CG_outputRepr *i_idx = ocg->CreatePlus(read_idxs[depth-1]->clone(), ocg->CreateInt(_x));
          CG_outputRepr *k_idx_positive = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(z));//K+z
          CG_outputRepr *j_idx_positive = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(y));//j+y
          CG_outputRepr *k_idx_negative = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-z));//k-z
          CG_outputRepr *j_idx_negative = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-y));//j-y
          
          CG_outputRepr *k_idx_positive_y = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(y));//K+y
          CG_outputRepr *j_idx_positive_z = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(z));//j+z
          CG_outputRepr *k_idx_negative_y = ocg->CreatePlus(read_idxs[depth-3]->clone(), ocg->CreateInt(-y));//K-y
          CG_outputRepr *j_idx_negative_z = ocg->CreatePlus(read_idxs[depth-2]->clone(), ocg->CreateInt(-z));//J-z
          
          
          //_in[k+z][j+y][x]
          std::vector<CG_outputRepr *> _idxs;
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          CG_outputRepr* read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          
          //_in[k+y][j+z][x]
          _idxs.clear();
          _idxs.push_back(k_idx_positive_y->clone());
          _idxs.push_back(j_idx_positive_z->clone());
          _idxs.push_back(i_idx->clone());
          
          //read_var = ir->CreateArrayRefRepr(read_->symbol(),_idxs);
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          //_in[k-z][j+y][x]::reflection on y-axis
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx_positive->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          //_in[k-y][j+z][x]::reflection on y-axis
          _idxs.clear();
          _idxs.push_back(k_idx_negative_y->clone());
          _idxs.push_back(j_idx_positive_z->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          
          //_in[k-z][j-y][x]::reflection on y=-z
          _idxs.clear();
          _idxs.push_back(k_idx_negative->clone());
          _idxs.push_back(j_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          //_in[k-y][j-z][x]::reflection on y=-z
          _idxs.clear();
          _idxs.push_back(k_idx_negative_y->clone());
          _idxs.push_back(j_idx_negative_z->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          
          //_in[k+z][j-y][x]::reflection on z-axis
          _idxs.clear();
          _idxs.push_back(k_idx_positive->clone());
          _idxs.push_back(j_idx_negative->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          //_in[k+y][j-z][x]::reflection on z-axis
          _idxs.clear();
          _idxs.push_back(k_idx_positive_y->clone());
          _idxs.push_back(j_idx_negative_z->clone());
          _idxs.push_back(i_idx->clone());
          
          read_var = ocg->CreatePlus(read_var->clone(), ir->CreateArrayRefRepr(read_->symbol(), _idxs));
          
          
          rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          reg_assignments = ocg->StmtListAppend(reg_assignments->clone(),
                                                ocg->CreateAssignment(0, rose_reg_ref->convert()->clone(),read_var->clone()));
          _ctr++;
        }
    }
  }
  
  if(equilibrium_stmt == NULL) equilibrium_stmt = reg_assignments->clone();
  
  r=radius;
  for(int n= r; n >= -radius ; n--){
    debug_fprintf(stderr, "n %d\n", n); 

    //The plane we are working on is _in[K][J][r]
    //The plane of coefficients we use: coeff[K][J][n] 
    //The buffer we write to: ....TO FIGURE OUT....
    
    
    sum_products_regs = NULL;
    int _x =r;
    _ctr = 0;
    
    //char *t = strdup("float");
    //char *nam = strdup("buh");
    //char *arr = strdup(""); 
    //chillAST_VarDecl *TEMP = new chillAST_VarDecl(t, nam, arr, (chillAST_node *)NULL); 
    
    for (int z=0; z<=radius; z++){
      for (int y=0; y<=z; y++){
        debug_fprintf(stderr, "z %d   y %d\n", z, y); 

        // get the coefficient ??  (as a chillAST_node)
        chillAST_node *coeff = SI->find_coefficient( n, y, z );
        if (coeff) { 
          
          //coeff->print(0, stderr); debug_fprintf(stderr, "\n"); 
          //coeff->dump(); printf("\n\n"); fflush(stdout);
          
          
          rose_reg = new IR_chillScalarSymbol(ir, treg_syms[_ctr]);
          rose_reg_ref = ir->CreateScalarRef(rose_reg);
          //IR_ScalarRef *scalar_cff_ref = ir->CreateScalarRef(new IR_roseScalarSymbol (ir, scalar_cffs_3d_array[z+radius][y+radius][n+1]));
          
          // this assumes it's a variable 
          //IR_ScalarRef *scalar_cff_ref = ir->CreateScalarRef(new IR_roseScalarSymbol (ir, TEMP)); // scalar_cffs_3d_array[z+radius][y+radius][n+radius]));
          
          //CG_outputRepr* read_var = ocg->CreateTimes(rose_reg_ref->convert()->clone(),
          //      ocg->CreateDouble(stencil_coeff_3d[z+radius][y+radius][n+1]));
          
          CG_outputRepr* read_var = ocg->CreateTimes(rose_reg_ref->convert()->clone(),
                                                     new CG_chillRepr( coeff ) ); // ?? // needs CG_outputRepr
          //scalar_cff_ref->convert()->clone()); // needs CG_outputRepr
          
          if (!sum_products_regs) 
            sum_products_regs = read_var;
          else  
            sum_products_regs = ocg->CreatePlus(sum_products_regs->clone(), read_var);
        }
        else debug_fprintf(stderr, "there was no coeff for  n %d   z %d   y %d\n", n, z, y ); 
        _ctr++;
        
        
      }
    }
    
    //Create the buffer and it's index
    rose_buffr = new IR_chillArraySymbol(ir, buff_syms[n+radius]);
    CG_outputRepr *idx_offset = ocg->CreatePlus(buf_idx->clone(), ocg->CreateInt(_ofst));
    idx_offset = ocg->CreatePlus(idx_offset->clone(), ocg->CreateInt(-lower));
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
  
  std::vector<Statement> temp_statement_buffer;
  //debug_fprintf(stderr, "init_pipeline is temp_statement_buffer[%d]\n", temp_statement_buffer.size()); 
  temp_statement_buffer.push_back(init_pipeline);
  //debug_fprintf(stderr, "stmt[%d] is temp_statement_buffer[%d]\n", stmt_num, temp_statement_buffer.size()); 
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
  
  //1. Set it's lexical order correctly by modifying the xform
  //We can acutally recycle the init_xform created earlier
  
  sum_buffers.xform = Composition(copy(init_xform),copy(temp_statement_buffer[1].xform));
  //printf("the xform for the statement to add the buffers\n");
  //sum_buffers.xform.print();
  
  //2. Let's build this statement
  //We sum buffers buff_0[i]+buff_1[i]+....
  //We write the result to output[k][j][i]
  init_rhs =NULL;
  for(int n=-radius; n<=radius; n++)
    {
      rose_buffr = new IR_chillArraySymbol(ir, buff_syms[n+radius]);
      indices.clear();
      index2.clear();
      //indices.push_back(buf_idx->clone());
      indices.push_back(ocg->CreatePlus(buf_idx->clone(), ocg->CreateInt(-lower)));
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
  
  //debug_fprintf(stderr, "sum_buffers is temp_statement_buffer[%d]\n", temp_statement_buffer.size()); 
  temp_statement_buffer.push_back(sum_buffers);
  
  //############################################################################
  //############################################################################
  
  final_num_stmts = init_num_stmts + temp_statement_buffer.size() -1;
  
  
  //Let's modify the xform's for the statements following the original stmt[stmt_num]
  //None of this is nice....
  int stmts_added = temp_statement_buffer.size() -1;
  for(int i=stmt_num+1; i<stmt.size(); i++)
    {
      int factor_to_increar_lex_order = stmts_added+i;
      for(int t=0; t<factor_to_increar_lex_order; t++)
        stmt[i].xform = Composition(copy(init_xform), copy(stmt[i].xform));
    }
  
  
  //ugly manipulation of vector of statments, this should be improved
  // insert new statements in place of stmt[stmt_num]
  
  //debug_fprintf(stderr, "\nincreasing number of statements\ninitially, %d statements\n", stmt.size()); 
  std::vector<Statement> tp_stmt;
  // copy initial (stmt_num) statements
  //debug_fprintf(stderr, "copying initial %d statements\n", stmt_num); 
  for(int i=0; i<stmt_num; i++) {
    //debug_fprintf(stderr, "adding stmt[%d] as tp_stmt %d\n", i, tp_stmt.size()); 
    tp_stmt.push_back(stmt[i]);  
  }
  // insert statements in place of previous stmt[stmt_num]
  //debug_fprintf(stderr, "copying %d new statements\n", temp_statement_buffer.size()); 
  for(int i=0; i<temp_statement_buffer.size(); i++) {
    //debug_fprintf(stderr, "adding temp_statement_buffer[%d] as tp_stmt %d\n", i, tp_stmt.size()); 
    tp_stmt.push_back(temp_statement_buffer[i]);
  }
  
  int inserts = temp_statement_buffer.size()-1;  // keep original (all should be known) 
  // also have to insert uninterpreted symbols ...
  // is all are known, then we can just push back at the end
  // TODO make new empty map, insert in the correct place 
  for(int i=0; i<inserts; i++) {
    stmt_nesting_level_.push_back( stmt_nesting_level_[stmt_num] ); 
    uninterpreted_symbols.push_back( uninterpreted_symbols[stmt_num] ); 
    uninterpreted_symbols_stringrepr.push_back(uninterpreted_symbols_stringrepr[stmt_num] ); 
  }
  
  // copy remaining statements 
  //debug_fprintf(stderr, "copying %d remaining statements\n", stmt.size() - stmt_num); // ?? 
  for(int i=stmt_num+1; i<stmt.size();i++) {
    //debug_fprintf(stderr, "adding stmt[%d] as tp_stmt %d\n", i, tp_stmt.size()); 
    tp_stmt.push_back(stmt[i]);
  }
  stmt = tp_stmt;
  
  
  
  /**********************************************
          Update the dependence graph
  **********************************************/
  DependenceGraph g(stmt[stmt_num].IS.n_set());
  
  for(int i=0; i<stmt.size(); i++) 
    g.insert();
  
  //debug_fprintf(stderr, "\n\nstmt.size()  %d\n", stmt.size()); 
  //debug_fprintf(stderr, "uninterpreted  %d\n", uninterpreted_symbols.size()); 
  //debug_fprintf(stderr, "uninterpreted  %d\n", uninterpreted_symbols_stringrepr.size()); 
  //debug_fprintf(stderr, "nesting_level  %d\n\n", stmt_nesting_level_.size()); 
  //for (int i = 0; i < stmt_nesting_level_.size(); i++) debug_fprintf(stderr, "nesting %d\n", stmt_nesting_level_[i]); 
  
  for (int i = 0; i < stmt.size(); i++) { 
    for (int j = i; j < stmt.size(); j++) {
      //debug_fprintf(stderr, "i %d  j %d\n", i, j ); 
      
      
      std::pair<std::vector<DependenceVector>,
                std::vector<DependenceVector> > dv = test_data_dependences(this, ir,
                                                                           stmt[i].code, 
                                                                           stmt[i].IS, 
                                                                           stmt[j].code, 
                                                                           stmt[j].IS,
                                                                           freevar, 
                                                                           index, 
                                                                           stmt_nesting_level_[i],
                                                                           stmt_nesting_level_[j],
                                                                           uninterpreted_symbols[ i ],  // ??? 
                                                                           uninterpreted_symbols_stringrepr[ i ],unin_rel[i], dep_relation);
      // TODO dep_relation is out-dated: from Anand's
      
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
  }
  dep = g;
  /***********************************************/
  
  fflush(stdout); 
}
