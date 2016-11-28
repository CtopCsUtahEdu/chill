chillAST_node *LoopCuda::cudaize_codegen_v2() {
  debug_fprintf(stderr, "cudaize codegen V2\n");
  
  // unused?   CG_clangBuilder *ocg = dynamic_cast<CG_clangBuilder*>(ir->builder());
  //if (!ocg) return false;
  
  //protonu--adding an annote to track texture memory type
  /*  ANNOTE(k_cuda_texture_memory, "cuda texture memory", TRUE);
      int tex_mem_on = 0;
  */
  
  int tex_mem_on  = 0;
  int cons_mem_on = 0;
  
  debug_fprintf(stderr, "here goes nothing\n"); 
  
  
  CG_outputRepr* repr;
  
  
  std::set<std::string> uniqueRefs;
  std::set<std::string> uniqueWoRefs;
  std::vector<IR_ArrayRef *> ro_refs;
  std::vector<IR_ArrayRef *> wo_refs;
  std::vector<VarDefs> arrayVars;

  std::set<const chillAST_VarDecl *> pdSyms; // PD? Parameter definition(?) ....
  std::vector<chillAST_VarDecl *> parameterSymbols; 
  
  // the C code function will become the GPUside function

  // this is dumb. The only thing we've got is the position of the function in the file.
  // we remove the function body abd build a new one.
  // the parameters are not in the right order (probably)
  
  chillAST_FunctionDecl *origfunction =  function_that_contains_this_loop; 
  const char *fname = origfunction->functionName;
  int numparams = origfunction->parameters.size();
  //debug_fprintf(stderr, "func 0x%x has name 0x%x  %d parameters\n", func, fname, numparams); 




  // make a new function that will be the CPU side cuda code
  // it will take the name and parameters from the original C code 
  chillAST_node *p = origfunction->getParent();
  debug_fprintf(stderr, "parent of func is a %s with %d children\n", 
          p->getTypeString(), p->getNumChildren()); 
  chillAST_SourceFile *srcfile = origfunction->getSourceFile();
  debug_fprintf(stderr, "srcfile of func is %s\n", srcfile->SourceFileName );



  chillAST_FunctionDecl *CPUsidefunc = new chillAST_FunctionDecl(origfunction->returnType, fname,p);
  for (int i=0; i<numparams; i++) { 
    CPUsidefunc->addParameter( origfunction->parameters[i] ) ; 
  }
  chillAST_CompoundStmt *CPUfuncbody =  new chillAST_CompoundStmt; // so we can easily access
  CPUsidefunc->setBody( CPUfuncbody ); // but empty 
  //CPUsidefunc->setParent( origfunction->getParent() ); // unneeded


  debug_fprintf(stderr, "kernel name should be %s (?)\n", cu_kernel_name.c_str()); 
  chillAST_FunctionDecl *GPUKernel =  new  chillAST_FunctionDecl( origfunction->returnType /* ?? */,
                                                                  cu_kernel_name.c_str(), // fname, 
                                                                  p); 
  chillAST_CompoundStmt *GPUkernelbody =  new chillAST_CompoundStmt; // so we can easily access
  GPUKernel->setBody( GPUkernelbody ); // but empty 

  
  // change name of GPU side function 
  int which = p->findChild( origfunction ); 
  debug_fprintf(stderr, "func is child %d of srcfile\n", which);
  p->insertChild( which,  GPUKernel );
  p->insertChild( which,  CPUsidefunc );


  //which = p->findChild( CPUsidefunc ); 
  //debug_fprintf(stderr, "\nCPUsidefunc is now child %d of srcfile\n", which);
  //which = p->findChild( GPUKernel ); 
  //debug_fprintf(stderr, "GPUKernel is now child %d of srcfile\n", which);
  which = p->findChild( origfunction ); 
  //debug_fprintf(stderr, "original function  is now child %d of srcfile\n", which);
  //p->removeChild ( ) or similar to remove original 
  p->removeChild( which ); 


  char *part = strdup( srcfile->SourceFileName );
  char *dot = rindex( part, '.' );
  if  (dot) { 
    *dot = '\0';
  }

  // name is passed to cudaize, and stored in LoopCuda
  //char newname[800];
  //sprintf(newname, "%s_GPU\0", part);
  //debug_fprintf(stderr, "GPU side function will be %s\n", newname ); 
  //GPUKernel->setName( newname ); 

  GPUKernel->setFunctionGPU();   
  chillAST_CompoundStmt *kernelbody = new chillAST_CompoundStmt;
  GPUKernel->setBody( kernelbody );
  

  CPUsidefunc->print(); printf("\n\n"); fflush(stdout); 
  GPUKernel->print();   printf("\n\n"); fflush(stdout); 


  
  
  debug_fprintf(stderr, "this loop is in function %s\n", fname); 
  debug_fprintf(stderr, "function %s has %d parameters:\n",  fname, numparams ); 
  for (int i=0; i< function_that_contains_this_loop->parameters.size(); i++) { 
    debug_fprintf(stderr, "%d/%d  %s\n", i, numparams,  function_that_contains_this_loop->parameters[i]->varname); 
  }
  
  
  
  
  debug_fprintf(stderr, "%d statements\n", stmt.size()); 
  for (int j = 0; j < stmt.size(); j++) {
    debug_fprintf(stderr, "\nstmt j %d\n", j); 
    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[j].code);
    debug_fprintf(stderr, "%d array references in stmt j %d\n", refs.size(), j); 
    
    debug_fprintf(stderr, "\nabout to dump statement j %d\n", j); 
    CG_chillRepr * repr = (CG_chillRepr *) stmt[j].code;
    repr->dump(); 
    fflush(stdout); debug_fprintf(stderr, "\n\n\n\n");
    
    for (int i = 0; i < refs.size(); i++) {
      //const char *vname = static_cast<const char*>(refs[i]->name().c_str());
      char *vname = strdup( refs[i]->name().c_str() ) ; // just for printing 
      printf("ref i %d, ref var %s, write? %d\n", i, vname, refs[i]->is_write());
      fflush(stdout); 
      
      // at this point, GPUkernel has no parameters, we're creating them now.
      // look to see if the original function had these parameters
      chillAST_VarDecl *param = origfunction->hasParameterNamed( refs[i]->name().c_str() ); 
      if (!param) { 
        //debug_fprintf(stderr, "variable %s is NOT a parameter, it must be defined in the function body\n",vname); 
        continue;
      }
      
      //debug_fprintf(stderr, "%s is a parameter of original function\n", vname);
      debug_fprintf(stderr, "%s is a parameter\n", vname);
      
      // see if this ref is in uniqueRefs
      if (uniqueRefs.find(refs[i]->name()) == uniqueRefs.end()) {
        
        debug_fprintf(stderr, "adding variable %s to uniqueRefs\n", vname); 
        // if not, add it
        uniqueRefs.insert(refs[i]->name()); 
        
        // and if it's a write, add it to Unique Write (Only?) Refs as well
        if (refs[i]->is_write()) {
          debug_fprintf(stderr, "adding variable %s to unique WRITE Refs\n", vname); 
          uniqueWoRefs.insert(refs[i]->name()); // a set
          wo_refs.push_back(refs[i]);           // a vector of the same info? 
        } else { 
          ro_refs.push_back(refs[i]);
        }
      }
      
      
      if (refs[i]->is_write()
          && (uniqueWoRefs.find(refs[i]->name()) == uniqueWoRefs.end())) { // wasn't there before
        debug_fprintf(stderr, "adding variable %s to unique WRITE Refs even though we'd seen it as a read before\n", vname); 
        uniqueWoRefs.insert(refs[i]->name());
        wo_refs.push_back(refs[i]);
        //printf("adding %s to wo\n", static_cast<const char*>(refs[i]->name()));
      }
      
      // do a std::set manually
      bool inthere = false;
      for (int k=0; k<parameterSymbols.size(); k++) { 
        if (!strcmp( param->varname, parameterSymbols[k]->varname)) inthere = true;
      }
      if (!inthere) parameterSymbols.push_back( param ) ; 
      debug_fprintf(stderr, "parameterSymbols now has %d elements\n", parameterSymbols.size()); 

      //pdSyms.insert((const chillAST_VarDecl*) param);
      //debug_fprintf(stderr, "pdsyms now has %d elements\n", pdSyms.size()); 
    }
  } // for each stmt 
  
  debug_fprintf(stderr, "we read from %d parameter arrays, and write to %d parameter arrays\n", ro_refs.size(), wo_refs.size()); 
  printf("reading from array parameters ");
  for (int i = 0; i < ro_refs.size(); i++)
    printf("'%s' ", ro_refs[i]->name().c_str());
  printf("and writing to array parameters ");
  for (int i = 0; i < wo_refs.size(); i++)
    printf("'%s' ", wo_refs[i]->name().c_str());
  printf("\n"); fflush(stdout); 
  



  const char* gridName  = "dimGrid";       // hardcoded 
  const char* blockName = "dimBlock";
  
  //TODO: Could allow for array_dims_vars to be a mapping from array
  //references to to variable names that define their length.
  
  for (int i = 0; i < wo_refs.size(); i++) {
    std::string name = wo_refs[i]->name();
    
    debug_fprintf(stderr, "\nwritten parameter %d %s\n", i, name.c_str()); 
    
    char *tmpname = strdup( name.c_str() ); 
    // find the variable declaration in original 
    chillAST_VarDecl *param = origfunction->findParameterNamed( tmpname ); 
    if (!param) { 
      debug_fprintf(stderr, "loop_cuda_clang.cc can't find wo parameter named %s in function %s\n",tmpname,fname);
      exit(-1); 
    }
    //param->print(); printf("\n"); fflush(stdout); 
    
    VarDefs v; // scoping seems wrong/odd
    v.size_multi_dim = std::vector<int>();
    char buf[32];
    snprintf(buf, 32, "devO%dPtr", i + 1);
    v.name = buf;
    v.original_name = name; 
    
    v.tex_mapped  = false;
    v.cons_mapped = false;
    
    // find the underlying type of the array
    debug_fprintf(stderr, "finding underlying type of %s to make variable %s match\n",name.c_str(),buf);
    v.type = strdup(param->underlyingtype); // memory leak 
    //debug_fprintf(stderr, "v.type is %s\n", param->underlyingtype); 
    
    chillAST_node *so = new chillAST_Sizeof( v.type ); 
    //CG_chillRepr *thingsize = new omega::CG_chillRepr(  so );
    
    debug_fprintf(stderr, "\nloop_cuda_xxxx.cc  calculating size of output %s\n", buf ); 

    int numitems = 1;
    if (param->numdimensions < 1 || 
        param->arraysizes == NULL) { 
      //Lookup in array_dims (the cudaize call has this info for some variables?) 
      std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
      debug_fprintf(stderr, "it %s %d\n", (*it).first.c_str(), (*it).second);  
      numitems = (*it).second; 
    }
    else { 
      debug_fprintf(stderr, "numdimensions = %d\n", param->numdimensions);
      for (int i=0; i<param->numdimensions; i++) { 
        numitems *= param->arraysizes[i]; 
      }
    } 


    chillAST_IntegerLiteral *numthings = new chillAST_IntegerLiteral( numitems ); 
    
    debug_fprintf(stderr, "creating int mult size expression numitems %d x sizeof( %s )\n", numitems, v.type ); 
    
    // create a mult  
    v.size_expr = new chillAST_BinaryOperator( numthings, "*", so, NULL); 
    
    v.CPUside_param = param;
    v.in_data = 0;
    v.out_data = param;
    
    //Check for this variable in ro_refs and remove it at this point if it is both read and write
    std::vector<IR_ArrayRef *>::iterator it_;
    for (it_ = ro_refs.begin(); it_ != ro_refs.end(); it_++) {
      if ((*it_)->name() == wo_refs[i]->name()) {
        debug_fprintf(stderr, "found array ref for %s in ro, removing it from writes\n", (*it_)->name().c_str()); 
        break;
      }
    }
    if (it_ != ro_refs.end()) {
      v.in_data = param;           // ?? 
      ro_refs.erase(it_);
    }
    
    debug_fprintf(stderr, "adding written v to arrayVars\n\n"); 
    v.print(); 
    arrayVars.push_back(v);
  } //  wo_refs 
  
  
  
  
  
  
  
  
  for (int i = 0; i < ro_refs.size(); i++) {
    std::string name = ro_refs[i]->name();
    char *tmpname = strdup( name.c_str() ); 
    
    debug_fprintf(stderr, "\nread parameter %d %s \n", i, name.c_str()); 
    
    // find the variable declaration 
    chillAST_VarDecl *param = origfunction->findParameterNamed( tmpname ); 
    if (!param) { 
      debug_fprintf(stderr, "loop_cuda_clang.cc can't find ro parameter named %s in function %s\n",tmpname,fname);
      exit(-1);
    }
    
    VarDefs v; // scoping seems wrong/odd
    v.size_multi_dim = std::vector<int>();
    char buf[32];
    snprintf(buf, 32, "devI%dPtr", i + 1);
    v.name = buf;
    v.original_name = name; 
    v.tex_mapped = false;
    v.cons_mapped = false;


    // find the underlying type of the array
    debug_fprintf(stderr, "finding underlying type of %s to make variable %s match\n",name.c_str(),buf);
    v.type = strdup(param->underlyingtype); // memory leak 
    //debug_fprintf(stderr, "v.type is %s\n", param->underlyingtype); 
    chillAST_node *so = new chillAST_Sizeof( v.type ); 
    
#ifdef NOTYET
    //derick -- adding texture and constant mapping
    if ( texture != NULL) { 
      v.tex_mapped = (texture->is_array_mapped(name.c_str()))? true:false; //protonu-track tex mapped vars
    }
    if (v.tex_mapped){
      printf("this variable  %s is mapped to texture memory", name.c_str());
    }
    //derick -- this is commented out until constant memory is implemeted
    if ( constant_mem != NULL) { 
      v.cons_mapped = (constant_mem->is_array_mapped(name.c_str()))? true:false; //protonu-track tex mapped vars
    }
    if (v.cons_mapped){
      printf("this variable  %s is mapped to constant memory", name.c_str());
    }
#endif  // NOTYET
    
    //debug_fprintf(stderr, "\ncalculating size of input %s\n", buf );    
    //Size of the array = dim1 * dim2 * num bytes of our array type
    //If our input array is 2D (non-linearized), we want the actual
    //dimensions of the array (as it might be less than cu_n
    //CG_outputRepr* size;
    
    int numitems = 1;
    param->print(0, stderr); debug_fprintf(stderr, "\n");


    if (param->numdimensions < 1 || 
        param->arraysizes == NULL) { 
      //Lookup in array_dims (the cudaize call has this info for some variables?) 
      std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
      //debug_fprintf(stderr, "it %s %d\n", (*it).first.c_str(), (*it).second);  
      debug_fprintf(stderr, "LUA command says this variable %s should be size %d\n",  (*it).first.c_str(), (*it).second); 
      numitems = (*it).second; 

    }
    else { 
      debug_fprintf(stderr, "numdimensions = %d\n", param->numdimensions);
      for (int i=0; i<param->numdimensions; i++) { 
        numitems *= param->arraysizes[i]; 
      }
    } 




    chillAST_IntegerLiteral *numthings = new chillAST_IntegerLiteral( numitems ); 
    
    debug_fprintf(stderr, "creating int mult size expression numitems %d x sizeof( %s )\n", numitems, v.type ); 
    
    // create a mult  
    v.size_expr = new chillAST_BinaryOperator( numthings, "*", so, NULL); // 1024 * sizeof(float)  etc
    
    v.CPUside_param = param;
    v.in_data = param;
    v.out_data = 0;
    
    
    debug_fprintf(stderr, "adding input v to arrayVars\n\n"); 
    v.print(); 
    arrayVars.push_back(v);   
  } // end of READ refs
  
  
  debug_fprintf(stderr, "\n\nAdd our mallocs (and input array memcpys) %d arrayVars\n", arrayVars.size());
  //Add our mallocs (and input array memcpys)
  for (int i = 0; i < arrayVars.size(); i++) {

    //debug_fprintf(stderr, "0x%x\n", arrayVars[i].vardecl); 
    debug_fprintf(stderr, "arrayVar %d\nC side: %s    Kernel side %s\n", i, arrayVars[i].name.c_str(), arrayVars[i].original_name.c_str() ); 


    const char *kernelparamname = arrayVars[i].original_name.c_str(); 
    int pdsymoffset = -1;
    for (int j=0;j<parameterSymbols.size(); j++) { 
      if (!(strcmp( kernelparamname, parameterSymbols[j]->varname))) pdsymoffset = j;
    }
    if ( pdsymoffset == -1 ) { 
      // complain 
    }
    else { 
      // we will not know all array sizes for the kernel definition(??)
      chillAST_VarDecl *param =  (chillAST_VarDecl *)parameterSymbols[pdsymoffset]->clone(); 
      param->knownArraySizes = false; // ?? 
      
      //debug_fprintf(stderr, "adding %s to GPUKernel parameters\n", kernelparamname);
      GPUKernel->addParameter( param );
      
    }

    if(arrayVars[i].cons_mapped) {
      debug_fprintf(stderr, "arrayVar %d is cons mapped  (TODO) \n", i); 
      exit(-1); 
    }
    else { 
      debug_fprintf(stderr, "buildVariableDeclaration %s\n", arrayVars[i].name.c_str()); 
      // create a CHILL variable declaration and put it in the CPU side function

      char typ[128];
      sprintf(typ, "%s *", arrayVars[i].type); 

      chillAST_VarDecl *var = new chillAST_VarDecl( typ,
                                                    arrayVars[i].name.c_str(),
                                                    "", // TODO
                                                    NULL);
      // set the array info to match
      // store variable decl where we can get it easilly later
      arrayVars[i].vardecl = var;

      CPUfuncbody->insertChild(0, var );  // add the CPUside variable declaration 

      // do the CPU side cudaMalloc 
      chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( var, CPUfuncbody ); 
      chillAST_CStyleAddressOf *AO = new chillAST_CStyleAddressOf( DRE );
      chillAST_CStyleCastExpr *casttovoidptrptr = new chillAST_CStyleCastExpr( "void **", AO, NULL ); 
      chillAST_CudaMalloc *cmalloc = new chillAST_CudaMalloc( casttovoidptrptr, arrayVars[i].size_expr, NULL); 
      CPUfuncbody->addChild( cmalloc );

      debug_fprintf(stderr, "\ncudamalloc is:\n"); 
      cmalloc->print(); printf("\n"); fflush(stdout); 
      
      debug_fprintf(stderr, "\nnow the memcpy (for input variables only????)\n"); 
      if (arrayVars[i].in_data) {  // if it's input to the calculation, and we need to copy the data to the GPU
        //debug_fprintf(stderr, "it's an input to the calculation, so we need to copy the data to the GPU\n"); 
        
        // do the CPU side cudaMemcpy, CPU to GPU("device")
        //DRE = new chillAST_DeclRefExpr( var, CPUfuncbody ); 
        chillAST_CudaMemcpy *cmemcpy = new chillAST_CudaMemcpy( var, 
                                                                (chillAST_VarDecl*)(arrayVars[i].in_data), 
                                                                arrayVars[i].size_expr, "cudaMemcpyHostToDevice"); 
        CPUfuncbody->addChild( cmemcpy );
        
        //printf("\n"); cmemcpy->print(); printf("\n");fflush(stdout); 
      } // for input variables only (must be copied to GPU before the kernel call) 
      //else { 
      //  debug_fprintf(stderr, "it's not an input to the calculation, so no memcpy over to the GPU\n"); 
      //} 
    }  // not cons mapped 
  }   // for all arrayvars 


  debug_fprintf(stderr, "\nBuild dimGrid dim3 variables based on loop dimensions and ti/tj\n"); 
  //Build dimGrid dim3 variables based on loop dimensions and ti/tj
  char blockD1[120];
  char blockD2[120];
  int dim1 = 0;            // TODO 
  if (dim1) {
     debug_fprintf(stderr,"cu_tx, cu_ty    CASE NOT HANDLED\n"); 
     exit(-1); 
#ifdef NOTYET
   snprintf(blockD1, 120, "%s/%d",
             dim1->get_declaration()->get_name().getString().c_str(), cu_tx);
    snprintf(blockD2, 120, "%s/%d",
             dim2->get_declaration()->get_name().getString().c_str(), cu_ty);
#endif
  } else {
    debug_fprintf(stderr,"cu_bx, cu_by\n"); 
    snprintf(blockD1, 120, "%d", cu_bx);
    snprintf(blockD2, 120, "%d", cu_by);
    //snprintf(blockD1, 120, "%d/%d", cu_nx, cu_tx);
    //snprintf(blockD2, 120, "%d/%d", cu_ny, cu_ty);
  }
  debug_fprintf(stderr, "blockD1 '%s'\n", blockD1); 
  debug_fprintf(stderr, "blockD2 '%s'\n", blockD2); 
  
  chillAST_FunctionDecl *dimbuiltin = new chillAST_FunctionDecl( "dim3", "dim3" );
  dimbuiltin->setBuiltin();

  chillAST_CallExpr *CE1 = new chillAST_CallExpr( dimbuiltin, NULL );

  // create ARGS ro dim3. 
  debug_fprintf(stderr, "create ARGS to dim3\n"); 
  if (cu_bx && cu_by) {                                      // 2 constants
    debug_fprintf(stderr, "dim3 dimGrid %d %d\n", cu_bx, cu_by); 
    CE1->addArg( new chillAST_IntegerLiteral( cu_bx ));
    CE1->addArg( new chillAST_IntegerLiteral( cu_by ));
  }
  else if (cu_bx_repr && cu_by_repr) {                        // 2 expressions? 
    debug_fprintf(stderr, "dim3 dimGrid cu_bx_repr  cu_by_repr\n" ); 
    chillAST_node *code1  = cu_bx_repr-> GetCode();
    chillAST_node *code2  = cu_bx_repr-> GetCode();
    CE1->addArg( code1 ); 
    CE1->addArg( code2 ); 
  }
  else if (cu_bx_repr) {
    debug_fprintf(stderr, "dim3 dimGrid  cu_bx_repr 1\n");            // one expression, and a constant?
    cu_bx_repr->dump(); fflush(stdout); 
    chillAST_node *code  = cu_bx_repr-> GetCode();
    
    CE1->addArg( code ); 
    CE1->addArg( new chillAST_IntegerLiteral( cu_by ));

  }

  chillAST_VarDecl *dimgriddecl = new chillAST_VarDecl( "dim3", "dimGrid", "", NULL );
  dimgriddecl->setInit(CE1);
  CPUfuncbody->addChild( dimgriddecl ); 
  debug_fprintf(stderr, "appending DIMGRID repr to setup code\n\n");


  debug_fprintf(stderr, "\nDIMBLOCK\n"); 
  // DIMBLOCK 
  int bs1 = 32;
  int bs2 = 1;
  if (cu_tz > 1) { //  || cu_tz_repr) {
    debug_fprintf(stderr, "cu_tz\n"); 
    exit(-1); 
    
  }
  else { 
    debug_fprintf(stderr, "NOT cu_tz\n"); 
    if (cu_tx && cu_ty) { 
      debug_fprintf(stderr, "cu_tx && cu_ty\n"); 
      bs1 = cu_tx;
      bs2 = cu_ty; 
    }
    else if (cu_tx_repr && cu_ty_repr) { 
      debug_fprintf(stderr, "cu_tx && cu_ty REPR\n"); 
      exit(-1); 
    }
    
  }
  


  chillAST_CallExpr *CE2 = new chillAST_CallExpr( dimbuiltin, NULL );
  CE2->addArg( new chillAST_IntegerLiteral( bs1 ));
  CE2->addArg( new chillAST_IntegerLiteral( bs2 ));
  chillAST_VarDecl *dimblockdecl = new chillAST_VarDecl( "dim3", "dimBlock", "", NULL );
  dimblockdecl->setInit(CE2);
  
  CPUfuncbody->addChild( dimblockdecl ); 


  // kernel call 
  debug_fprintf(stderr, "KERNEL CALL\n"); 
  chillAST_CallExpr *kcall = new chillAST_CallExpr( GPUKernel,  CPUfuncbody);
  kcall->grid = dimgriddecl; 
    kcall->block =  dimblockdecl; 
  debug_fprintf(stderr, "kernel function parameters\n"); 
  for (int i = 0; i < arrayVars.size(); i++) { 
    //Throw in a type cast if our kernel takes 2D array notation
    //like (float(*) [1024])
    
    if (arrayVars[i].tex_mapped || arrayVars[i].cons_mapped) continue;

    chillAST_VarDecl *v = arrayVars[i].vardecl;
    chillAST_VarDecl *param = arrayVars[i].CPUside_param;

    //debug_fprintf(stderr, "numdimensions %d\n", param->numdimensions); 
    
    if (param->numdimensions > 1) { 
      debug_fprintf(stderr, "array Var %d %s is multidimensional\n",i, v->varname);
      //debug_fprintf(stderr, "underlying type %s\narraypart %s\n", v->underlyingtype, v->arraypart); 
      char line[128];
      sprintf(line, "%s (*)", v->underlyingtype ); 
      //debug_fprintf(stderr, "line '%s'\n", line);
      // we'll pass in a cast of the variable instead of just the variable.
      for (int i=1; i<param->numdimensions; i++) { 
        int l = strlen(line);
        //debug_fprintf(stderr, "l %d\n", l); 
        char *ptr = &line[l];
        //debug_fprintf(stderr, "[%d]", param->arraysizes[i]); 
        sprintf(ptr, "[%d]", param->arraysizes[i]); 
        //debug_fprintf(stderr, "i %d line '%s'\n", i, line);
        chillAST_CStyleCastExpr *CE = new chillAST_CStyleCastExpr( line, v, NULL );
        kcall->addArg( CE );
      }
      //int l = strlen(line);
      //char *ptr = line + l;
      //sprintf(ptr, ")");
      //debug_fprintf(stderr, "line '%s'\n", line); 
      
    }
    else { 
      debug_fprintf(stderr, "array Var %d %s is NOT multidimensional\n",i, v->varname);

      // we just need a decl ref expr inserted as the parameter/argument
      // when it prints, it will print just the array name
      chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( v, NULL);
      kcall->addArg( DRE );
    }
  }


   CPUfuncbody->addChild( kcall );           
  //kcall->addArg( 



  debug_fprintf(stderr, "\nfreeing Cuda variables\n"); 
  //cuda free variables
  for (int i = 0; i < arrayVars.size(); i++) {
    debug_fprintf(stderr, "arrayVar %d\n", i); 

    // Memcopy back if we have an output 
    if (arrayVars[i].out_data) {

      chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( arrayVars[i].vardecl, CPUfuncbody ); 
      chillAST_CudaMemcpy *cmemcpy = new chillAST_CudaMemcpy( (chillAST_VarDecl*)arrayVars[i].out_data, // wrong info
                                                              arrayVars[i].vardecl, 
                                                              arrayVars[i].size_expr, "cudaMemcpyDeviceToHost"); 
      CPUfuncbody->addChild( cmemcpy );
    }

    // CudaFree the variable
    chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( arrayVars[i].vardecl, CPUfuncbody ); 
    chillAST_CudaFree *cfree = new chillAST_CudaFree( arrayVars[i].vardecl, CPUfuncbody ); 
    CPUfuncbody->addChild( cfree );
    
  }
  //CPUsidefunc->print(); fflush(stdout); 
  //GPUKernel->print();   fflush(stdout); 




  debug_fprintf(stderr, "BUILD THE KERNEL\n"); 

  //Extract out kernel loop  (somewhat misnamed. This is NOT the body of the GPUKernel YET) 
  chillAST_node *kernelloop = getCode(  ); 
  debug_fprintf(stderr, "loop_cuda_clang.cc L1669 returned from getCode()\n");

  //debug_fprintf(stderr, "loop_cuda_clang.cc L1685  kernelloop =\n");
  //GPUKernel->getBody()->print(); fflush(stdout);
  //debug_fprintf(stderr, "\n\n"); 
         
  debug_fprintf(stderr, "loop_cuda_clang.cc L1685   kernelloop = \n");
  kernelloop->print(); 
  debug_fprintf(stderr, "\n\n"); 

  debug_fprintf(stderr, "%d arrayvars\n", arrayVars.size());  
  
  // this should just be sitting in a member of arrayVars
  std::map<std::string, chillAST_VarDecl*> loop_vars;
  for (int i = 0; i < arrayVars.size(); i++) {
    debug_fprintf(stderr, "arrayVars[%d]  name %s\n", i, arrayVars[i].original_name.c_str()); 
    //if (arrayVars[i].in_data)  debug_fprintf(stderr, "input ");
    //if (arrayVars[i].out_data)  debug_fprintf(stderr, "output ");
    //debug_fprintf(stderr, "\n");

    chillAST_VarDecl *d = GPUKernel->hasParameterNamed( arrayVars[i].original_name.c_str() ); 
    if (d) { 
      debug_fprintf(stderr, "inserting %s into loop_vars\n", arrayVars[i].original_name.c_str()); 
      loop_vars.insert(std::pair<std::string, chillAST_VarDecl*>(std::string(arrayVars[i].original_name), d));
    }
  }
  
  debug_fprintf(stderr, "\nfind variables used in the kernel (?)\n"); 

  // find all variables used in the function
  vector<chillAST_VarDecl*> decls;
  kernelloop->gatherVarDecls( decls );
  debug_fprintf(stderr, "%d variables in kernel\n", decls.size()); 
  for (int i=0; i<decls.size(); i++) { 
    debug_fprintf(stderr, "%s\n", decls[i]->varname); 
  }

  int nump = GPUKernel->parameters.size();
  debug_fprintf(stderr, "\n%d parameters to GPUKernel\n", nump); 
  for (int i=0; i<nump; i++) debug_fprintf(stderr, "parameter %s\n",  GPUKernel->parameters[i]->varname );
  debug_fprintf(stderr, "\n"); 

  

  //Figure out which loop variables will be our thread and block dimension variables
  debug_fprintf(stderr, "Figure out which loop variables will be our thread and block dimension variables\n"); 

  //Get our indexes  (threadIdx and blockIdx will replace some loops) 
  std::vector<const char*> indexes;

  if (cu_bx > 1 || cu_bx_repr) {
    indexes.push_back("bx");
    chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.x", "int", GPUKernel );
    chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *bxdecl = new chillAST_VarDecl( "int", "bx", "", GPUKernel );
    GPUKernel->addDecl( bxdecl );
    chillAST_DeclRefExpr *bx = new chillAST_DeclRefExpr( bxdecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( bx, "=", bid ); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(bxdecl); 
    kernelbody->addChild(assign); 
  }

  if (cu_by > 1 || cu_by_repr) {
    indexes.push_back("by");
    chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.y", "int", GPUKernel );
    chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *bydecl = new chillAST_VarDecl( "int", "by", "", GPUKernel );
    GPUKernel->addDecl( bydecl );
    chillAST_DeclRefExpr *by = new chillAST_DeclRefExpr( bydecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( by, "=", bid ); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(bydecl); 
    kernelbody->addChild(assign); 
  }  
  if (cu_tx_repr || cu_tx > 1) {
    //threadsPos = indexes.size();
    indexes.push_back("tx");
    chillAST_VarDecl *tiddecl = addBuiltin( "threadIdx.x", "int",     GPUKernel);
    chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( tiddecl ); 
    chillAST_VarDecl *txdecl = new chillAST_VarDecl( "int", "tx", "", GPUKernel);
    GPUKernel->addDecl( txdecl );
    chillAST_DeclRefExpr *tx = new chillAST_DeclRefExpr( txdecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( tx, "=", tid ); 
    assign->print(0, stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(txdecl); 
    kernelbody->addChild(assign); 
  }
  if (cu_ty_repr || cu_ty > 1) {
    indexes.push_back("ty");
    chillAST_VarDecl *biddecl = addBuiltin( "threadIdx.y", "int", GPUKernel );
    chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *tydecl = new chillAST_VarDecl( "int", "ty", "", GPUKernel );
    GPUKernel->addDecl( tydecl );
    chillAST_DeclRefExpr *ty = new chillAST_DeclRefExpr( tydecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( ty, "=", tid ); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(tydecl); 
    kernelbody->addChild(assign); 
  }
  if (cu_tz_repr || cu_tz > 1) {
    indexes.push_back("tz");
    chillAST_VarDecl *biddecl = addBuiltin( "threadIdx.z", "int", GPUKernel );
    chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( biddecl ); 
    chillAST_VarDecl *tzdecl = new chillAST_VarDecl( "int", "tz", "", GPUKernel );
    GPUKernel->addDecl( tzdecl );
    chillAST_DeclRefExpr *tz = new chillAST_DeclRefExpr( tzdecl ); 
    chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( tz, "=", tid ); 
    assign->print(0,stderr); debug_fprintf(stderr, "\n"); 

    kernelbody->addChild(tzdecl); 
    kernelbody->addChild(assign); 
  }


  debug_fprintf(stderr, "\n"); 
  for (int i = 0; i < indexes.size(); i++) {
    debug_fprintf(stderr, "indexes[%i] = '%s'\n", i, indexes[i] ); 
  }

  debug_fprintf(stderr, "\nbefore swapVarReferences(), code is\n{\n"); 
  kernelbody->print();

  debug_fprintf(stderr, "}\n\nswapVarReferences()\n"); 
  //swapVarReferences( kernelloop, GPUKernel );

  debug_fprintf(stderr, "\nafter swapVarReferences(), code is\n"); 
  kernelbody->print();
  debug_fprintf(stderr, "\n\n");
  
  
  debug_fprintf(stderr, "now replace indexes ... (and add syncs)\n"); 
  findReplacePreferedIdxs( kernelloop, GPUKernel );
  debug_fprintf(stderr, "DONE WITH replace indexes ... (and add syncs)\n"); 

  debug_fprintf(stderr, "\nswapped 2\nshould have syncs\nshould have indexes replaced by bx, tx, etc \n\n"); 
  kernelloop->print();

  // now remove loops that will be done by spreaking the loop count across cores
  // these are loops that have out indeces gathered above aas loop variables
  debug_fprintf(stderr, "removing loops for variables that will be determined by core index\n"); 
  chillAST_CompoundStmt *CS = new chillAST_CompoundStmt();
  
  CS->addChild( kernelloop ); // in case top level loop will go away
  //debug_fprintf(stderr, "arbitrary compoundstmt 0x%x to hold child kernelloop  0x%x\n", CS, kernelloop); 
  for (int i = 0; i < indexes.size(); i++) {
    debug_fprintf(stderr, "\nindexes[%i] = '%s'\n", i, indexes[i] ); 
    debug_fprintf(stderr, "forReduce()\n");
    
    kernelloop->loseLoopWithLoopVar( strdup(indexes[i]) ); 
  }


  


  debug_fprintf(stderr, "END cudaize codegen V2\n\n\n");
  debug_fprintf(stderr, "\nat end of cudaize_codegen_v2(), returning\n");
  CS->print(); 


  // variables in CS have not been added to GPUKernel.   fix that
  // should probably do this earlier/elsewhere
  vector<chillAST_VarDecl*> kerneldecls;
  vector<chillAST_VarDecl*> kerneldeclsused;
  GPUKernel->gatherVarDecls( kerneldecls );
  CS->gatherVarUsage( kerneldeclsused );




  debug_fprintf(stderr, "kernel defines %d variables\n", kerneldecls.size()); 
  for (int i=0; i<kerneldecls.size(); i++) { 
    chillAST_VarDecl *vd = kerneldecls[i]; 
    if (vd->isParmVarDecl()) { 
      vd->print(); 
      printf("  (parameter)");
      printf("\n"); fflush(stdout); 
    }
  }
  for (int i=0; i<kerneldecls.size(); i++) { 
    chillAST_VarDecl *vd = kerneldecls[i]; 
    if (vd->isBuiltin()) { 
      vd->print(); 
      printf("  (builtin)");
      printf("\n"); fflush(stdout); 
    }
  }
  for (int i=0; i<kerneldecls.size(); i++) { 
    chillAST_VarDecl *vd = kerneldecls[i]; 
    if ( (!vd->isParmVarDecl()) && (!vd->isBuiltin()) ) { 
      vd->print(); 
      printf("\n"); fflush(stdout); 
    }
  }
         printf("\n"); fflush(stdout); 


         debug_fprintf(stderr, "kernel uses    %d variables\n", kerneldeclsused.size()); 
       for (int i=0; i<kerneldeclsused.size(); i++) { 
         chillAST_VarDecl *vd = kerneldeclsused[i];
         debug_fprintf(stderr, "%2d %s\n", i, vd->varname); 
       }
         debug_fprintf(stderr, "\n\n");  



  int numdeclared = kerneldecls.size(); 
  for (int i=0; i<kerneldeclsused.size(); i++) { 
    chillAST_VarDecl *vd = kerneldeclsused[i];
    bool isdeclared = false;
    debug_fprintf(stderr, "%2d %s ", i, vd->varname); 
    if (vd->isBuiltin())     isdeclared = true;
    if (isdeclared) debug_fprintf(stderr, " (builtin)");
    else { 
      if (vd->isParmVarDecl()) isdeclared = true;
      if (isdeclared) debug_fprintf(stderr, " (param)");
    }
    for (int j=0; j<numdeclared; j++) { 
      if (kerneldeclsused[i] == kerneldecls[j] ) {
        isdeclared = true; 
        debug_fprintf(stderr, " (used %d is decl %d)", i, j); 
        break;
      }
    }
    debug_fprintf(stderr, "\n"); 

    if (!isdeclared) { 
      debug_fprintf(stderr, "declaration for %s needs to be added\n", vd->varname);
      GPUKernel->addChild( vd ); 
    }
  }  



  // take contents of CS and stuff it into GPUKernel, at the end after the declarations we might have just added 
  GPUKernel->addChild( CS ) ; // ?? could do each statement

  //debug_fprintf(stderr, "\nGPU side func is \n");
  //GPUKernel->print();

  return CS; 
} 
