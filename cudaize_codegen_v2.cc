
#include "cudaize_codegen_v3.cc"


chillAST_node *LoopCuda::cudaize_codegen_v2() {  // NOT WORKING ON THIS ONE NOW    ANAND'S  ??  
  debug_fprintf(stderr, "\nLoopCuda::cudaize codegen V2 (CHILL) Line 7 of cudaize_codegen_v2.cc\n");
  
  debug_fprintf(stderr, "here is the list of array_dims\n"); 
  for(std::map<std::string, int>::iterator it = array_dims.begin(); it != array_dims.end(); it++)  {
    debug_fprintf(stderr, "array_dims  '%s'  %d\n", it->first.c_str(), it->second);
  }
  debug_fprintf(stderr, "that is the list\n"); 

  CG_chillBuilder *ocg = dynamic_cast<CG_chillBuilder*>(ir->builder());
  if (!ocg) { 
    debug_fprintf(stderr, "no ocg?\n"); 
    return NULL;
  }
  
  
  
  // this routine takes a CPU only function and creates funcs that run the 
  // 'same' code on a GPU.  The result will have one CPU side function, 
  // and one or more GPU kernels.
  chillAST_FunctionDecl *origfunction =  function_that_contains_this_loop; 
  int numparams = origfunction->parameters.size();
  const char *fname = origfunction->functionName;
  chillAST_node *p = origfunction->getParent();
  int origfunction_location = p->findChild( origfunction );
  chillAST_SourceFile *srcfile = origfunction->getSourceFile();
  
  // make a new function that will be the CPU side cuda code
  // it will take the name and parameters from the original C code 
  debug_fprintf(stderr, "creating CPUsidefunc named %s\n", fname); 
  chillAST_FunctionDecl *CPUsidefunc = new chillAST_FunctionDecl(origfunction->returnType, fname, NULL); 

  for (int i=0; i<numparams; i++) { 
    CPUsidefunc->addParameter( origfunction->parameters[i] ) ; 
  }
  chillAST_CompoundStmt *CPUfuncbody =  CPUsidefunc->getBody(); 

  // copy the preprocessing statements from the original (??)
  int numpreproc =  origfunction->preprocessinginfo.size();
  debug_fprintf(stderr, "preprocessinginfo %d\n", numpreproc); 
  if (numpreproc != 0) { 
    debug_fprintf(stderr, "copying %d preprocessing statements from original %s to the new one that calls the GPU\n", numpreproc, fname); 
    for (int i=0; i<numpreproc; i++) { 
      CPUsidefunc->preprocessinginfo.push_back( origfunction->preprocessinginfo[i] );
    }
  }


  
  CPUbodySymtab = origfunction->getSymbolTable();   // local name 
  CPUsidefunc->setSymbolTable(  CPUbodySymtab );
  CPUparamSymtab = CPUsidefunc->getParameterSymbolTable(); // local name 
  
  
  // put this new cpu side function where the old one was in the sourcefile 
  int which = p->findChild( origfunction ); 
  int originalfunctionlocation = which; 
  p->insertChild( which,  CPUsidefunc );
  
  // remove original function  from the source   (bad idea ??) 
  which = p->findChild( origfunction ); 
  debug_fprintf(stderr, "original function  is now child %d of srcfile\n", which);
  p->removeChild( which );  // TODO do this last, because we look for it later
  
  //origfunction->print(0,stderr); debug_fprintf(stderr, "\n\n");

  debug_fprintf(stderr, "OK, made new (empty) CPU side function %s, removed original function\n",origfunction->functionName );
  p->print(); printf("\n\n\n"); fflush(stdout);
  
  debug_fprintf(stderr, "\nCPUSIDE func:\n"); 
  CPUsidefunc->print(); printf("\n\n"); fflush(stdout); 
  
  
  debug_fprintf(stderr, "this loop is in function %s\n", fname); 
  debug_fprintf(stderr, "function %s has %d parameters:\n",  fname, numparams ); 
  for (int i=0; i< function_that_contains_this_loop->parameters.size(); i++) { 
    debug_fprintf(stderr, "%d/%d  %s\n", i, numparams,  function_that_contains_this_loop->parameters[i]->varname); 
  }
  
  // create a builtin dim3() function    TODO do we need to check if it's already been created?
  chillAST_FunctionDecl *dimbuiltin = new chillAST_FunctionDecl( "dim3", "dim3" );
  dimbuiltin->setBuiltin();
  
  
  //sort cudaized according to lexical order into ordered cudaized statements
  debug_fprintf(stderr, "sort cudaized according to lexical order into ordered cudaized statements\n"); 
  
  debug_fprintf(stderr, "\n%d cudaized statements to be run on the GPU\n", cudaized.size()); 
  std::vector<int> sort_aid;
  std::set<int> all_cudaized_statements;
  
  std::vector<std::pair<std::set<int>, int> > ordered_cudaized_stmts;
  
  for (int i = 0; i < cudaized.size(); i++) {
    int what = get_const(stmt[*(cudaized[i].begin())].xform, 0, Output_Var); // 0 == outermost
    debug_fprintf(stderr, "sort_aid[%d] = %d\n", sort_aid.size(), what);
    debug_fprintf(stderr, "all_cudaized_statements.insert( %d, %d)\n", cudaized[i].begin(), cudaized[i].end());
    sort_aid.push_back( what ); 
    all_cudaized_statements.insert(cudaized[i].begin(), cudaized[i].end());
  }
  debug_fprintf(stderr, "%d in all_cudaized_statements\n\n", all_cudaized_statements.size()); 

  
  for (int i = 0; i < stmt.size(); i++) {
    if (all_cudaized_statements.find(i) == all_cudaized_statements.end()) {
      int j;
      for (j = 0; j < cudaized.size(); j++)
        if (get_const(stmt[i].xform, 0, Output_Var)
            == get_const(stmt[*(cudaized[j].begin())].xform, 0, Output_Var)) {
          cudaized[j].insert(i);
          break;
        }
      if (j == cudaized.size()
          && all_cudaized_statements.find(i)
          == all_cudaized_statements.end()) {
        sort_aid.push_back(get_const(stmt[i].xform, 0, Output_Var));
        int last = -1 + sort_aid.size();
        debug_fprintf(stderr, "at end, sort_aid[%d] = %d\n", last, sort_aid[last]);
      }
    }
  }
  
  debug_fprintf(stderr, "sorting ...\n"); 
  int num_sort_aid = sort_aid.size();
  debug_fprintf(stderr, "%d in sort_aid\n", num_sort_aid);
  std::sort(sort_aid.begin(), sort_aid.end());

  debug_fprintf(stderr, "after sort,\n");
  for (int i = 0; i < num_sort_aid; i++) {
    debug_fprintf(stderr, "sort_aid[%d] = %d\n", i, sort_aid[i]);
  }
  
  for (int i = 0; i < num_sort_aid; i++) {
    int start = i;
    
    while (i + 1 < num_sort_aid && sort_aid[i + 1] == sort_aid[i])
      i++;
    
    int j;
    for (j = 0; j < cudaized.size(); j++) { 
      if (get_const(stmt[*(cudaized[j].begin())].xform, 0, Output_Var)
          == sort_aid[start]) {
        ordered_cudaized_stmts.push_back(std::pair<std::set<int>, int>(cudaized[j], j));
        break;
      }
    }
    if (j == cudaized.size()) {
      std::set<int> temp;
      for (int j = 0; j < stmt.size(); j++) {
        
        if (all_cudaized_statements.find(j) == all_cudaized_statements.end()) { 
          if (sort_aid[start] == get_const(stmt[j].xform, 0, Output_Var)) {
            temp.insert(j);
          }
        }
      }
      ordered_cudaized_stmts.push_back(std::pair<std::set<int>, int>(temp, -1));
    }
  }
  
  debug_fprintf(stderr, "code after\n"); 
  
  
  ////////////////////////////////////////////////////////////////////////////
  
  
  debug_fprintf(stderr, "%d ordered_cudaized_stmts\n", ordered_cudaized_stmts.size() );
  for (int i=0; i<ordered_cudaized_stmts.size(); i++) { 
    debug_fprintf(stderr, "( ");
    for (std::set<int>::iterator j = ordered_cudaized_stmts[i].first.begin(); 
         j != ordered_cudaized_stmts[i].first.end(); j++) {
      debug_fprintf(stderr, "%d ", (*j) );
    } 
    debug_fprintf(stderr, ") (first)      %d (second)   ", ordered_cudaized_stmts[i].second);
    if (ordered_cudaized_stmts[i].second == -1) {
      debug_fprintf(stderr, "NOT ON GPU\n");
    }
    else {
      debug_fprintf(stderr, "    ON GPU\n"); // gee, documentation of this would have been nice. ya think?
    }
  }
  debug_fprintf(stderr, "\n");
  
  // find pointer-to-int, pointer-to-float?
  int num_ordered = ordered_cudaized_stmts.size(); 
  debug_fprintf(stderr, "looking for pointer-to-int, pointer-to-float in %d ordered_cudaized_stmts\n", num_ordered); 
  std::set<int> ptrs;
  for (int i = 0; i < num_ordered; i++) {
    debug_fprintf(stderr, "i %d\n", i); 
    for (std::set<int>::iterator it =
           ordered_cudaized_stmts[i].first.begin();
         it != ordered_cudaized_stmts[i].first.end(); it++) {
      
      std::vector<IR_PointerArrayRef *> ptrRefs = ir->FindPointerArrayRef(
                                                                         stmt[*it].code);
      
      debug_fprintf(stderr, "loop_cuda_XXXX.cc   *it %d,  %d pointer arrayrefs\n", *it, ptrRefs.size()); 
      for (int j = 0; j < ptrRefs.size(); j++)
        for (int k = 0; k < ptr_variables.size(); k++)
          if (ptrRefs[j]->name() == ptr_variables[k]->name())
            ptrs.insert(k);
    }
  } // i goes out of scope
  debug_fprintf(stderr, "found %d pointers to do mallocs for\n\n", ptrs.size()); 
  
  
  // for each pointer, build malloc( sizeof( int/float ) ) ??
  for (std::set<int>::iterator it = ptrs.begin(); it != ptrs.end(); it++) {
    if (ptr_variables[*it]->elem_type() == IR_CONSTANT_FLOAT) { 
      debug_fprintf(stderr, "pointer to float\n");
    }
    else if (ptr_variables[*it]->elem_type() == IR_CONSTANT_INT) {
      debug_fprintf(stderr, "pointer to INT\n");
    }
    else
      throw loop_error("Pointer type unidentified in cudaize_codegen_v2!");
    debug_fprintf(stderr, "TODO - DIDN'T ACTUALLY DO THE MALLOC\n");
  } 
  debug_fprintf(stderr, "done making mallocs?\n");
  
  
  
  
  //protonu--adding an annote to track texture memory type
  /*  ANNOTE(k_cuda_texture_memory, "cuda texture memory", TRUE);
      int tex_mem_on = 0;
  */
  
  int tex_mem_on  = 0;
  int cons_mem_on = 0;
  
  debug_fprintf(stderr, "\n\n*** here goes nothing\n"); 
  num_ordered = ordered_cudaized_stmts.size(); 
  debug_fprintf(stderr, "%d ordered cudaized stmts\n", num_ordered);
  
  std::vector< chillAST_VarDecl * > inspectorargs; // call from main to cpuside inspector has these arguments

  for (int CS = 0; CS < num_ordered; CS++) { // CS = cudaized statement 
    debug_fprintf(stderr, "HGN CS %d / %d\n", CS, num_ordered-1);
    
    CG_outputRepr* repr;
    std::vector<CudaIOVardef> arrayVars;
    std::vector<CudaIOVardef> localScopedVars;
    
    //std::vector<IR_ArrayRef *> ro_refs; // this one is not used ???
    //std::vector<IR_ArrayRef *> wo_refs;
    //std::set<std::string> uniqueRefs; // unused ?? 
    //std::set<std::string> uniqueWoRefs;
    
    std::set<const chillAST_VarDecl *> pdSyms;// PD? Parameter definition(?) ....
    std::vector<chillAST_VarDecl *> parameterSymbols; 
    
    
    chillAST_node *code_temp = getCode( 1234, ordered_cudaized_stmts[CS].first);
    
    

    if (code_temp != NULL) {
      
      //debug_fprintf(stderr, "\nHGN %d code_temp was NOT NULL\n", CS);
      //printf("\nloop_cuda_chill.cc L903 code_temp:\n"); code_temp->print(); printf("\n\n"); fflush(stdout);
      
      
      debug_fprintf(stderr, "set %d has size %d\n", CS,  ordered_cudaized_stmts[CS].first.size()); 
      
      // find first statement that HAS an inspector?
      std::set<int>::iterator i =
        ordered_cudaized_stmts[CS].first.begin();
      
      debug_fprintf(stderr, "first stmt i is %d\n", *i);
      
      for (; i != ordered_cudaized_stmts[CS].first.end(); i++) {
        if (stmt[*i].has_inspector) { 
          debug_fprintf (stderr, "stmt %d HAS an inspector\n", *i); 
          break;
        }
        else debug_fprintf (stderr, "stmt %d DOES NOT HAVE an inspector\n", *i); 
      }
      
      debug_fprintf(stderr, "i is %d   last is %d\n", *i, *ordered_cudaized_stmts[CS].first.end());
      
      if (i == ordered_cudaized_stmts[CS].first.end()) {  // there was no statement with an inspector
        debug_fprintf(stderr, "no statement had an inspector\n"); 
        // If not run on the gpu
        if ( ordered_cudaized_stmts[CS].second == -1 ) {
          setup_code = ocg->StmtListAppend(setup_code,  // becomes part of setup (after init)
                                           new CG_chillRepr(code_temp));
        }
      } 
      else { // there was an inspector 
        debug_fprintf(stderr, "there was an inspector\n"); 
        
        // create a function that becomes spmv_inspector(  )
        std::set<IR_ArrayRef *> outer_refs;
        for (std::set<int>::iterator j =
               ordered_cudaized_stmts[CS].first.begin();
             j != ordered_cudaized_stmts[CS].first.end(); j++) {
          
          if (stmt[*j].ir_stmt_node != NULL) {
            
            
            // find all loop/if init/cond/iteration in loops this is contained in
            debug_fprintf(stderr, "find all loop/if init/cond/iteration in loops this is contained in\n");
            std::vector<CG_outputRepr *> loop_refs =
              collect_loop_inductive_and_conditionals(stmt[*j].ir_stmt_node);
            debug_fprintf(stderr, "%d loop_refs\n\n", loop_refs.size()); 
            
            // find the array references in  the loop/if 
            for (int i = 0; i < loop_refs.size(); i++) {
              std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(loop_refs[i]);
              
              // make unique list of references (no repeated names) in outer_refs
              for (int l = 0; l < refs.size(); l++) {
                std::set<IR_ArrayRef *>::iterator k =
                  outer_refs.begin();
                for (; k != outer_refs.end(); k++)
                  if ((*k)->name() == refs[l]->name())
                    break;
                if (k == outer_refs.end()) { 
                  debug_fprintf(stderr, "a new array ref\n");
                  outer_refs.insert(refs[l]);
                }
              }
              
            }
            debug_fprintf(stderr, "\n%d non-repeated array name refs\nthey are:\n", outer_refs.size()); 
            for (std::set<IR_ArrayRef *>::iterator k =outer_refs.begin(); k != outer_refs.end(); k++) {
              debug_fprintf(stderr, "%s\n", (*k)->name().c_str());               
            }
            
          }
        }
        //--end
        
        
        char fname[100];
        sprintf(fname, "%s_inspector", (dynamic_cast<IR_chillCode*>(ir))->procedurename); 
        //std::string functionname = string(dynamic_cast<IR_chillCode*>(ir)->procedurename) + "_inspector";
        
        // all these are declared above as well??? 
        debug_fprintf(stderr, "\n\n*** building %s() AST   CS %d\n", fname, CS);
        //chillAST_FunctionDecl *origfunction = function_that_contains_this_loop;
        //chillAST_SourceFile   *srcfile = origfunction->getSourceFile();
        chillAST_FunctionDecl *inspectorFunc = new chillAST_FunctionDecl(strdup("void"), fname,  srcfile ); // this adds inspectorfunc to children of srcfile
 
        debug_fprintf(stderr, "original function was:\n");
        origfunction->print(); printf("\n\n"); fflush(stdout); 
        
        // create a compound statement function body, so we can add vardecls as needed

        debug_fprintf(stderr, "loop_cuda_chill.cc L991  code_temp %s\n", code_temp->getTypeString()); 
        //int *k = 0; int die = k[0];
        

        chillAST_CompoundStmt *inspectorbody;
        
        if (code_temp->isCompoundStmt()) {
          // can we just assign to CPUfunctionBody?  no
          inspectorbody =  (chillAST_CompoundStmt *)code_temp; 
        }
        else {  // old code was just one statement 
          inspectorbody =  new chillAST_CompoundStmt;
          inspectorbody->addChild( code_temp );
        }
        inspectorFunc->setBody( inspectorbody );  // this wlil ruin symbol tables 
        int numdefs = 0; // definitions inside inspectorbody
        
        
        // find which NONarray parameters are used in the code (?)
        code_temp->print(); printf("\n"); fflush(stdout);
        
        vector<chillAST_VarDecl*>  decls;
        code_temp->gatherVarUsage( decls );
        debug_fprintf(stderr, "%d vars used in code_temp\n\n", decls.size());
        for (int i=0; i<decls.size(); i++) { 
          printf("sym "); decls[i]->print(); printf(" \n"); fflush(stdout); 
        }
        printf("\n"); fflush(stdout); 
        
        // any of those that were parameters in the original code should 
        // be added as parameters in the new function
        // we can only check parameter NAMES? TODO  
        // actually, that is what we need!
        // the var in code temp is a clone, but we want to know 
        // about parameters with that NAME in the original (??? !!)
        for (int i=0; i<decls.size(); i++) { 
          //printf("%p  ",decls[i]);decls[i]->dump();printf("\n");fflush(stdout);
          // if (decls[i]->isAParameter) {  // this doesn't work, because we 
          // are dealing with a copy that has substitutions, and doesn't 
          // refer to the original vardecl (should it?)
          
          char *name = decls[i]->varname;
          debug_fprintf(stderr, "\nchecking for a parameter %d named %s in origfunction\n", i, name); 
          
          
          if (origfunction->hasParameterNamed( name )) { 
            debug_fprintf(stderr, "%s was a parameter in the original. adding it to inspectorFunc parameters\n", name);
            
            // this decl should have no parent ??
            // see chill_ast.cc addParameter. 
            //if (decls[i]->parent) { 
            //  debug_fprintf(stderr, "UHOH, this vardecl for %s I'm about to make a parameter of %s already had a parent??\n", name, fname); 
            //  exit(-1); 
            //} 
            
            // have to clone the vardecl and replace it in the code?
            chillAST_VarDecl *param = (chillAST_VarDecl *)decls[i]->clone();
            inspectorargs.push_back( decls[i] ); 
            inspectorFunc->addParameter( param ); 
            code_temp->replaceVarDecls( decls[i], param ); 
          }
          else { 
            debug_fprintf(stderr, "there was no parameter named %s in origfunction\n", name);  
            decls[i]->dump(); fflush(stdout);
            decls[i]->print(); printf("\n"); fflush(stdout);
            if (decls[i]->isAStruct()) { 
              debug_fprintf(stderr, "%s it's a struct\n", name); 
              
              // somehow, this is enough to mean that we need to pass it 
              //   in as a parameter?  TODO 
              // make a NEW vardecl, for the parameter
              chillAST_VarDecl *param = (chillAST_VarDecl *)decls[i]->clone();
              param->setStruct( true ); // clone should handle this ???

              inspectorargs.push_back( decls[i] ); 
              inspectorFunc->addParameter( param );
              param->setByReference( true );
              param->print(); printf("\n"); fflush(stdout);
              
              // swap out the vardecl in the declrefexp in the code?
              code_temp->replaceVarDecls( decls[i], param ); 
            }
            else { 
              // this will not be a parameter, we need to make the vardecl 
              // be inside the body of the function
              debug_fprintf(stderr, "adding VarDecl for %s inside inspectorbody\n", decls[i]->varname); 
              chillAST_VarDecl *vd = (chillAST_VarDecl *)decls[i]->clone();
              inspectorbody->insertChild( numdefs++, vd );
            }
          }
        }
        
        
        
        
        
        
        
        debug_fprintf(stderr, "\n\n*** processing %d outer_refs into parameters\n",outer_refs.size()); 
        for (std::set<IR_ArrayRef *>::iterator l = outer_refs.begin();
             l != outer_refs.end(); l++) {
          
          chillAST_ArraySubscriptExpr *ASE =((IR_chillArrayRef *)(*l))->chillASE;
          chillAST_VarDecl *vd = ASE->multibase();
          char *vname = vd->varname; 
          debug_fprintf(stderr, "vname %s\n", vname); 
          
          if ( chillAST_VarDecl *p = origfunction->hasParameterNamed( vname )) { 
            debug_fprintf(stderr, "%s was a parameter in the original. adding it\n", vname); 
            // we need to make a new version of this parameter. The old 
            // one had a size. this will just be a pointer. (??) TODO
            // for example, int index[494]  in the original becomes 
            // int *index in this function 
            chillAST_VarDecl *newparam = new chillAST_VarDecl( vd->vartype, "*", vd->varname);
            newparam->print(); printf("\n"); fflush(stdout);
            
            inspectorargs.push_back( p ); 
            inspectorFunc->addParameter( newparam );  
            newparam->print(); printf("\n"); fflush(stdout);
            newparam->dump(); printf("\n"); fflush(stdout);
            
            // substitute this parameter vardecl for the old one in the code
            code_temp->replaceVarDecls( vd, newparam ); 
            
          }
          
        }
        
        
        
        
        
        
        
        // TODO make sure vardecls are included ?
        debug_fprintf(stderr, "\nTHIS IS inspectorFunc\n");
        if (inspectorFunc->parent) {
          debug_fprintf(stderr, "IT HAS A PARENT of type %s\n", inspectorFunc->parent->getTypeString());
        }
        inspectorFunc->print(); printf("\n\n"); fflush(stdout); 
        
        debug_fprintf(stderr, "building call to %s\n", inspectorFunc->functionName); 
        chillAST_CallExpr *CE = new chillAST_CallExpr( inspectorFunc );
        
        debug_fprintf(stderr, "parameters will be\n");
        for (int i=0; i<inspectorargs.size(); i++) { 
          inspectorargs[i]->print(); printf("\n"); fflush(stdout); 
          CE->addArg( new chillAST_DeclRefExpr( inspectorargs[i] ));
        }
        printf("\n"); fflush(stdout);
        
        debug_fprintf(stderr, "adding inspectorfunc call to setup_code\n"); 
        setup_code = ocg->StmtListAppend(setup_code,
                                         new CG_chillRepr(CE));
        
        
      } // there was an inspector 
      
    } // code_temp not NULL
    else { 
      debug_fprintf(stderr, "HGN %d code_temp WAS NULL\n", CS); 
      exit(-1);
    }
    

    
    // still in int CS loop 
    debug_fprintf(stderr, "\n\n\n\n\n*** dimgrid dimblock\n");
    char gridName[20];
    char blockName[20];
    sprintf(gridName,  "dimGrid%i",  CS);
    sprintf(blockName, "dimBlock%i", CS);
    

    // still in int CS loop 2
    chillAST_FunctionDecl *GPUKernel = NULL;  // kernel for replacing ONE STATEMENT 

    if ( ordered_cudaized_stmts[CS].second != -1 ) {
      debug_fprintf(stderr, "\ncreating the kernel function\n"); 
      
      chillAST_FunctionDecl *origfunction =  function_that_contains_this_loop; 
      const char *fname = origfunction->functionName;
      int numparams = origfunction->parameters.size();
      debug_fprintf(stderr, "\n\noriginal func has name %s  %d parameters\n", fname, numparams); 
      
      chillAST_node *p = origfunction->getParent();
      debug_fprintf(stderr, "parent of orig is %p\n", p);
      if (!p)  {
        exit(-1); 
      }
      debug_fprintf(stderr, "parent of func is a %s with %d children\n", p->getTypeString(), p->getNumChildren()); 
      // defined above (twice!) chillAST_SourceFile *srcfile = origfunction->getSourceFile();
      debug_fprintf(stderr, "srcfile of func is %s\n", srcfile->SourceFileName );

      debug_fprintf(stderr, "offset %d\n", CS );
      debug_fprintf(stderr, "%d names\n", Vcu_kernel_name.size()); 
      debug_fprintf(stderr, "Vcu_kernel_name[offset %d] %p\n", CS, Vcu_kernel_name[CS]);
      if (CS >= Vcu_kernel_name.size()) {
        debug_fprintf(stderr, "%s line %d   would have died, dammit\n", __FILE__, __LINE__);

        for (int k=0; k<Vcu_kernel_name.size(); k++) {
          debug_fprintf(stderr, "kernel name %d/%d is %s\n", k, Vcu_kernel_name.size()-1, Vcu_kernel_name[k].c_str()); 
        }
        //die();
        //TODO: is this an error requiring an exception?
      }
      debug_fprintf(stderr, "\n\nkernel named %s\n", 
              //cu_kernel_name.c_str()
              Vcu_kernel_name[CS].c_str());

      GPUKernel =  new  chillAST_FunctionDecl( origfunction->returnType /* ?? */,
                                               //cu_kernel_name.c_str(), // fname, 
                                               Vcu_kernel_name[CS].c_str(),
                                               p);  // 
      GPUKernel->setFunctionGPU();   // this is something that runs on the GPU 
      

      std::vector< chillAST_VarDecl*>  GPUKernelParams; // 

      debug_fprintf(stderr, "ordered_cudaized_stmts[CS %d] WILL become a cuda kernel\n", CS); 
      
      // find the array refs in code_temp 
      //std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(code_temp); 
      //debug_fprintf(stderr, "%d array refs in CPUfunc\n", refs.size()); 
      //chillAST_node *code_temp
      std::vector<chillAST_ArraySubscriptExpr*> refs;  
      code_temp->gatherArrayRefs( refs, 0 );
      
      debug_fprintf(stderr, "%d array refs in CPUfunc\n", refs.size());
      //for (int i=0; i<refs.size(); i++) { 
      //  refs[i]->print(); printf("\n"); fflush(stdout); 
      //} 
      debug_fprintf(stderr, "code is:\n");
      //if (NULL == code_temp->parent) debug_fprintf(stderr, "code temp HAS NO PARENT\n");
      //code_temp->print(); printf("\n"); fflush(stdout);
      
      
      debug_fprintf(stderr, "\nchecking if var refs are a parameter\n"); 
      debug_fprintf(stderr, "\nthere are %d refs\n", refs.size()); 
      for (int i = 0; i < refs.size(); i++) {
        refs[i]->print(); printf("\n"); fflush(stdout); 
        debug_fprintf(stderr, "\nvar %d  ref\n", i);
        debug_fprintf(stderr, "ref %d   %s wo UNK\n", i, refs[i]->multibase()->varname); // , refs[i]->is_write());
      }
      
      fflush(stdout); 
      //If the array is not a parameter, then it's a local array and we
      //want to recreate it as a stack variable in the kernel as opposed to
      //passing it in.
      
      //  the thing that is being checked is (was?)  wrong
      
#if 1
      std::vector<IR_chillArrayRef *> ro_refs; // this one is used
      std::vector<IR_chillArrayRef *> wo_refs;
      std::vector<IR_chillArrayRef *> rw_refs;
#endif
      
      //std::vector<chillAST_ArraySubscriptExpr *> ro_refs;// try chillAST versions
      //std::vector<chillAST_ArraySubscriptExpr *> wo_refs;
      
      // this is stupid. Just make an array of character strings
      //std::set<std::string> uniqueRefs;
      //std::set<std::string> uniqueWoRefs;
      //what_t uniqueRefs;
      //what_t uniqueWoRefs;
      std::vector< char * >  uniqueRefs;
      
      // TODO 
      for (int i = 0; i < refs.size(); i++) {
        debug_fprintf(stderr, "\nvar %d  ref  in CS %d\n", i, CS); 
        
        //chillAST_VarDecl *vd = refs[i]->multibase(); // this returns a reference to i for c.i n
        chillAST_node *node = refs[i]->multibase(); // this returns a reference to c.i
        //debug_fprintf(stderr, "\nnode is a %s\n", node->getTypeString()); 
        
        string stringvar;
        if (node->isVarDecl()) { 
          chillAST_VarDecl *vd = (chillAST_VarDecl *) node ; // this returns a reference to c.i 
          stringvar = vd->varname; 
        }
        else if (node->isMemberExpr()) { 
          chillAST_MemberExpr *me = (chillAST_MemberExpr *) node;
          stringvar = me->stringRep();
        }
        
        const char *lookingfor = strdup( stringvar.c_str() ); 
        printf("%s wo \n", lookingfor); fflush(stdout); 
        
        //debug_fprintf(stderr, "\nL1205, CPUparamSymtab now has %d entries\nCPUparamSymtab\n", CPUparamSymtab->size());
        //printSymbolTable(CPUparamSymtab);
        //debug_fprintf(stderr, "L 1207, CPUbodySymtab now has %d entries\nCPUbodySymtab\n", CPUbodySymtab->size());
        //printSymbolTable(CPUbodySymtab);
        
        //for (int i = 0; i < kernel_parameters.size(); i++) { 
        //  debug_fprintf(stderr, "kernel parameter: %s\n", kernel_parameters[i].c_str()); 
        //} 
        
        
        // sure are doing this a LOT 
        chillAST_VarDecl *vd = symbolTableFindVariableNamed( CPUbodySymtab,  lookingfor );
        chillAST_VarDecl *p  = symbolTableFindVariableNamed( CPUparamSymtab, lookingfor );
        
        if (vd != NULL) { 
          debug_fprintf(stderr, "varname %s IS in CPUbodySymtab\n", lookingfor); 
          printSymbolTable(CPUbodySymtab);
          printSymbolTable(CPUparamSymtab);
          

          debug_fprintf(stderr, "ONLY check kernel_parameters for '%s' if varname is in CPUbodySymtab\n", lookingfor); 
          
          std::set<std::string>::iterator it;
          it = kernel_parameters.find(stringvar); 
          if ( it == kernel_parameters.end())  { 
            debug_fprintf(stderr, "varname %s is NOT in kernel_parameters. skipping\n", lookingfor); 
            
            continue;
          }
          else { 
            
            debug_fprintf(stderr, "umwut found '%s', which IS a kernel parameter\n", (*it).c_str()); 
            
            int numkp = kernel_parameters.size(); // WARNING, also kernel_params exists. different name!
            debug_fprintf(stderr, "%d in SET kernel_parameters\n", numkp); 
            for (std::set<std::string>::iterator k = kernel_parameters.begin(); k != kernel_parameters.end(); k++) {
              debug_fprintf(stderr, "'%s'\n", (*k).c_str());               
            }
            
          }
        }
        else {
          debug_fprintf(stderr, "\nvarname %s is NOT in CPUbodySymtab\n", lookingfor); 
          debug_fprintf(stderr, "\nbody Symtab is :\n"); printSymbolTable(CPUbodySymtab);
          debug_fprintf(stderr, "\nparam Symtab is :\n"); printSymbolTable(CPUparamSymtab);
          debug_fprintf(stderr, "\nCPUSide func is:\n"); CPUsidefunc->print(0,stderr); 
          debug_fprintf(stderr, "\nCPUfuncbody is:\n"); CPUfuncbody->print(0,stderr); 

          debug_fprintf(stderr, "\nDAMMIT\n\n"); 

        }
        
        
        debug_fprintf(stderr, "\nvar %d   %s\nwill be a GPU kernel parameter??\n", i, lookingfor);
        if (!vd) vd = p;

        if (!vd) { 
          debug_fprintf(stderr, "... but I can't find the decl to copy???\nloop_cuda_chill.cc  L1250\n");
          int *i=0;  int j = i[0]; 
          exit(-1); 
        }
        debug_fprintf(stderr, "need to copy/mimic "); vd->print(0, stderr); debug_fprintf(stderr, "\n");
        chillAST_VarDecl *newparam = (chillAST_VarDecl *) vd->clone() ; // copy 

        vd->print(0,stderr); debug_fprintf(stderr, "\n"); 

        debug_fprintf(stderr,"after cloning\n"); 
        printSymbolTable(CPUbodySymtab);


        //debug_fprintf(stderr, "newparam numdimensions %d\n", newparam->numdimensions );
        // TODO need to remove topmost size? 
        if (newparam->getArrayDimensions() > 0) {
          newparam->convertArrayToPointer(); // set first size now unknown
          //newparam->knownArraySizes = false;
          debug_fprintf(stderr, "[]");
          for (int i=1; i<newparam->numdimensions; i++) { 
            debug_fprintf(stderr, "[%d]", newparam->getArraySizeAsInt(i));
          }
          debug_fprintf(stderr, "\n"); 
        }
        newparam->print(0,stderr); debug_fprintf(stderr, "\n"); 

        printSymbolTable(CPUbodySymtab);
        printSymbolTable(CPUparamSymtab);
        
        debug_fprintf(stderr, "GPUKernel addparameter( %s )\n", newparam->varname); 
        GPUKernel->addParameter( newparam );  // TODO this parameter LOOKS like but it not the vardecl that the declrefexpr points to 

        printSymbolTable(CPUbodySymtab);
        printSymbolTable(CPUparamSymtab);

        // so ... add it now ??
        

        //vd->print(); printf("\n"); fflush(stdout); 
        //vd->dump(); printf("\n"); fflush(stdout); 
        //if (vd->isParmVarDecl()) { 
        //  debug_fprintf(stderr, "IS a parameter?\n");
        //  vd->dump(); printf("\n\n"); fflush(stdout); 
        //} 
        //if (!vd->isParmVarDecl()) { 
        //  debug_fprintf(stderr, "local array - kernel stack variable\n"); 
        
#if 1
        //debug_fprintf(stderr, "looking for %s in %d uniqueRefs\n",  stringvar.c_str(), uniqueRefs.size()); 
        int offset = charstarvectorindex(  stringvar.c_str(), uniqueRefs );
        if ( offset == -1 )  { // != -1uniqueRefs.find( stringvar.c_str() )  == uniqueRefs.end()) {
          // wasn't there 
          debug_fprintf(stderr, "adding variable %s to uniqueRefs\n",  stringvar.c_str()); 
          // if not, add it
          //uniqueRefs.insert(  stringvar.c_str() );
          uniqueRefs.push_back(  strdup( stringvar.c_str()) );  // leak 
          
          // remember, refs WAS  std::vector<chillAST_ArraySubscriptExpr*> refs;   WAS 
          //  std::vector<IR_chillArrayRef *> wo_refs;
          if (refs[i]-> imwrittento) {
            if(refs[i]->imreadfrom) {
              debug_fprintf(stderr, "adding variable %s to unique Read & Write Refs\n", stringvar.c_str());
              rw_refs.push_back(new IR_chillArrayRef(ir, refs[i], refs[i]->imwrittento));
            }
            else {
              debug_fprintf(stderr, "adding variable %s to unique Write Only Refs\n", stringvar.c_str() );
              wo_refs.push_back( new IR_chillArrayRef( ir, refs[i], refs[i]-> imwrittento /* true */ ) );
            }
          }
          else { // JUST read from 
            debug_fprintf(stderr, "adding variable %s to unique Read Only Refs\n", stringvar.c_str() ); // this is c.i
            debug_fprintf(stderr, "actually, adding "); refs[i]->print(); printf("\n"); fflush(stdout); 
            ro_refs.push_back( new IR_chillArrayRef( ir, refs[i], stringvar.c_str(), false )); // this is i
          }
        } // this is a new  reference 
        else debug_fprintf(stderr, "%s was already there?\n", stringvar.c_str()); 
#endif
        // NOT a parameter 
      } // for each ref  i 
      
      
      debug_fprintf(stderr, "NOW WE MAKE THE GPU SIDE CODE\n\n"); 
      // the original function has been removed, so the following fails 
      // int which = p->findChild( origfunction ); 
      // debug_fprintf(stderr, "func is child %d of srcfile\n", which);

      // GPUKernel was created with parent p, so it is already there
      p->insertChild(origfunction_location,  GPUKernel );
      debug_fprintf(stderr, "\n\nkernel named %s\n", GPUKernel->functionName); 
      
      
      
      
      
      // COMMENT NEEDED 
      debug_fprintf(stderr, "loop_cuda_chill.cc COMMENT NEEDED printing kernel parameters\n"); 
      for (std::set<std::string>::iterator i = kernel_parameters.begin();   
           i != kernel_parameters.end(); i++) {
        debug_fprintf(stderr, "kernel parameter %s\n", (*i).c_str()); 
      }

      // PRINT RO REFS 
      for (int j = 0; j < ro_refs.size(); j++)
        debug_fprintf(stderr, "ro_ref %d %s\n", j, ro_refs[j]->name().c_str());  // ro_refs is  std::vector<IR_chillArrayRef *>
      debug_fprintf(stderr, "\n\n"); 
      
      
      // COMMENT NEEDED 
      debug_fprintf(stderr, "COMMENT NEEDED FOR EACH KERNEL PARAMETER\n"); 
      for (std::set<std::string>::iterator i = kernel_parameters.begin();
           i != kernel_parameters.end(); i++) {
        
        std::string kp(*i); // TODO name below is exactly this
        debug_fprintf(stderr, "walking through kernel_parameters %s\n", kp.c_str()); 
        
        int j;
        for (j = 0; j < ro_refs.size(); j++)
          if (ro_refs[j]->name() == *i)  { 
            
            break;
          }
        if (j < ro_refs.size())
          continue;
        
        std::string name = *i;
        debug_fprintf(stderr, "perhaps adding parameter %s to GPU kernel parameters??? \n", name.c_str()); 
        
        // HERE  stmt_code code_temp
        debug_fprintf(stderr, "\n***find all scalar refs in code_temp\n"); 
        std::vector<chillAST_DeclRefExpr*> scalar_refs;
        code_temp->gatherScalarRefs( scalar_refs, 0 );
        
        debug_fprintf(stderr, "SCALAR REFS (not printed)\n"); 
        //for (int k = 0; k < scalar_refs.size(); k++) { 
        //  debug_fprintf(stderr, "scalar ref %d ", k);
        //  scalar_refs[k]->print(); printf("\n"); fflush(stdout);
        //} 

        bool found = false;
        chillAST_node *ref = NULL; 

        debug_fprintf(stderr, "looking for %s in scalar refs\n", name.c_str()); 
        chillAST_DeclRefExpr *dre = NULL;
        for (int k = 0; k < scalar_refs.size() && !found; k++) { 
          if ( name == scalar_refs[k]->declarationName ) { 
            ref = scalar_refs[k];
            
            found = true;
            break;
          }
        }      

        if (!found)  { 
          debug_fprintf(stderr, "we did NOT find the parameter %s in the scalar refs.  look for it in macros?\n", name.c_str()); 
          
          //file we're working on holds the macro definitions 
          int numMacros = srcfile->macrodefinitions.size();
          debug_fprintf(stderr, "there are %d macros\n", numMacros); 

          for (int i=0; i<numMacros && !found; i++) { 
            chillAST_MacroDefinition *macro =  srcfile->macrodefinitions[i];
            debug_fprintf(stderr, "macro %d, name '%s'   ", i, macro->macroName);
            macro->print();  printf("\n"); fflush(stdout); 

            char *blurb = macro->getRhsString();
            if (blurb == NULL) { debug_fprintf(stderr, "macro rhs NULL\n"); }
            else 
            {
              //debug_fprintf(stderr, "macro rhs "); macro->getBody()->print(); debug_fprintf(stderr, "\n");
              //debug_fprintf(stderr, "%p\n", blurb); 
              //debug_fprintf(stderr, "%s\n", blurb); 
              // TODO this will not work in most cases. 
              // It is comparing the RESULT of the macro with the parameter name
              // that will only work if the result does not depend on any macro parameter. 
              if (!strcmp(blurb, name.c_str())) { 
                found = true;
                debug_fprintf(stderr, "macro RHS matches????\n"); 
                debug_fprintf(stderr, "rhs is of type %s\n", macro->getBody()->getTypeString()); 
                // get decl ref expression?  (why?)
                ref = macro->getBody(); // rhs  TODO function name 
              }
            }
            
          } 
          debug_fprintf(stderr, "\n\n"); 

        }
        
        if (!found)  { debug_fprintf(stderr, "var_sym == NULL\n"); }
        else { debug_fprintf(stderr, "var_sym NOT == NULL\n"); }
      

        // UMWUT 
        if (found) {
          debug_fprintf(stderr, "checking name '%s' to see if it contains a dot  (TODO) \n",name.c_str()); 
          

          debug_fprintf(stderr, "make sure a symbol with this name is in the symbol table\n"); 
          
          debug_fprintf(stderr, "creating parameter that is address of value we want to pass to the GPU\n");
          debug_fprintf(stderr, "eg c_count = &c.count;\n");
          debug_fprintf(stderr, "%s line %d bailing\n\n", __FILE__, __LINE__);
          
          exit(-1); 
        } // end of UMWUT 
        else { 
          debug_fprintf(stderr, "var_sym was NULL, no clue what this is doing  %s\n", name.c_str()); 
        } // end of no clue 


      } // for each kernel parameter ???
      
      get_io_refs(ir, array_dims, refs, arrayVars);
#if 0
      // unclear where this came from. older version, I assume 
      debug_fprintf(stderr, "OK, first the %d OUTPUTS of the GPU code\n",  wo_refs.size() ); 
      for (int i = 0; i < wo_refs.size(); i++) {
        std::string name = wo_refs[i]->name(); // creates a symbol. ...
        debug_fprintf(stderr, "write only ref (output?) %s\n", name.c_str()); 
      }
      debug_fprintf(stderr, "\n");
      debug_fprintf(stderr, "original function  parameters are:\n");
      printSymbolTable( &(origfunction->parameters ));
#endif
      

#if 0
      for (int i = 0; i < wo_refs.size(); i++) {
        std::string name = wo_refs[i]->name();
        debug_fprintf(stderr, "output %s\n", name.c_str()); 
        
        char *tmpname = strdup( name.c_str() ); 
        // find the variable declaration in original
        // this seems to have no analog in Anand's code.  ????"
        chillAST_VarDecl *param = origfunction->findParameterNamed( tmpname ); 
        if (!param) { 
          debug_fprintf(stderr, "cudaize_codegen_v2.cc can't find wo parameter named %s in function %s\n",tmpname,fname);
          
          debug_fprintf(stderr, "the parameters are:\n");
          printSymbolTable( &(origfunction->parameters )); 

          //origfunction->print(0,stderr); debug_fprintf(stderr, "\n\n");
          exit(-1); 
        }
        //param->print(); printf("\n"); fflush(stdout); 
        
        CudaIOVardef v; // scoping seems wrong/odd
        v.size_multi_dim = std::vector<int>();
        char buf[32];
        snprintf(buf, 32, "devO%dPtr", i + 1);
        v.name = buf;
        v.original_name = name; 
        debug_fprintf(stderr, "we'll have an output variable called %s\n", buf);
        
        v.tex_mapped  = false;
        v.cons_mapped = false;
        
        // find the underlying type of the array
        //debug_fprintf(stderr, "finding underlying type of %s to make variable %s match\n",name.c_str(),buf);
        v.type = strdup(param->underlyingtype); // memory leak 
        //debug_fprintf(stderr, "v.type is %s\n", param->underlyingtype); 
        
        chillAST_node *so = new chillAST_Sizeof( v.type ); 
        //CG_chillRepr *thingsize = new omega::CG_chillRepr(  so );
        
        debug_fprintf(stderr, "\nloop_cuda_xxxx.cc  calculating size of output %s\n", buf ); 
        
        int numitems = 1;
        if (param->numdimensions < 1 || 
            param->getArrayDimensions() == 0) {
          debug_fprintf(stderr, "looking in array_dims numdimensions = %d\n", param->numdimensions);

          //Lookup in array_dims (the cudaize call has this info for some variables?) 
          std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
          if (it == array_dims.end()) { 
            debug_fprintf(stderr, "L955 Can't find %s in array_dims\n", name.c_str()); 
            numitems = 123456;
          }
          else { 
            debug_fprintf(stderr, "L955 it %s %d\n", (*it).first.c_str(), (*it).second);  
            numitems = (*it).second; 
          }
          
          numitems = (*it).second; 
        }
        else { 
          debug_fprintf(stderr, "numdimensions = %d\n", param->numdimensions);
          for (int i=0; i<param->numdimensions; i++) { 
            numitems *= param->getArraySizeAsInt(i);
          }
          debug_fprintf(stderr, "Detected Multi-dimensional array size of %d for %s \n", numitems, tmpname); 
        } 
        
        
        chillAST_IntegerLiteral *numthings = new chillAST_IntegerLiteral( numitems ); 
        
        debug_fprintf(stderr, "creating int mult size expression numitems %d x sizeof( %s )\n", numitems, v.type ); 
        





        debug_fprintf(stderr, "DERP, finally figured out the size expression for the output array ...  L983\n"); 
        
        // create a mult  
        v.size_expr = new chillAST_BinaryOperator( numthings, "*", so);
        debug_fprintf(stderr, "v.size_expr  "); v.size_expr->print(0,stderr); debug_fprintf(stderr, "\n");  
        
        v.CPUside_param = param;
        v.in_data = 0;
        v.out_data = param;
        
        //Check for this variable in ro_refs and remove it at this point if it is both read and write
        // why did we bother before ??? 
        std::vector<IR_chillArrayRef *>::iterator it_;
        for (it_ = ro_refs.begin(); it_ != ro_refs.end(); it_++) {
          if ((*it_)->chillASE->multibase()->varname == wo_refs[i]->name()) {
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
#else
      //get_io_array_refs(array_dims, wo_refs, io_dir::write_only, arrayVars);
#endif
      
      
#if 0
      for (int i = 0; i < ro_refs.size(); i++) {
        std::string name = ro_refs[i]->name();
        char *tmpname = strdup( name.c_str() ); 
        debug_fprintf(stderr, "\nread only parameter %d %s \n", i, name.c_str()); 
        
        //chillAST_VarDecl *param = origfunction->findParameterNamed( tmpname ); 
        //if (!param) { 
        //  debug_fprintf(stderr, "loop_cuda_chill.cc can't find ro parameter named %s in function %s\n",tmpname,fname);
        //  debug_fprintf(stderr, "the parameters are:\n");
        //  printSymbolTable( &(origfunction->parameters )); 
        //  exit(-1);
        //} 
        
        CudaIOVardef v; // scoping seems wrong/odd
        v.size_multi_dim = std::vector<int>();
        char buf[32];
        snprintf(buf, 32, "devI%dPtr", i + 1);
        v.name = buf;
        v.original_name = name; 
        v.tex_mapped = false;
        v.cons_mapped = false;
        
        bool isNOTAParameter = (kernel_parameters.find(name) == kernel_parameters.end());
        if (isNOTAParameter) { debug_fprintf(stderr, "%s is NOT a kernel parameter\n", name.c_str() );}
        else { debug_fprintf(stderr, "%s IS a parameter\n", name.c_str()); }
        
        
        // find the underlying type of the array
        debug_fprintf(stderr, "finding underlying type of %s to make variable %s match\n",name.c_str(),buf);
        // find the variable declaration
        chillAST_ArraySubscriptExpr* ASE = ro_refs[i]->chillASE;
        chillAST_VarDecl *base = ASE->multibase();
        v.type = strdup(base->underlyingtype); // memory leak 
        //debug_fprintf(stderr, "v.type is %s\n", base->underlyingtype); 
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
#endif        
        
        //debug_fprintf(stderr, "\ncalculating size of input %s\n", buf );    
        //Size of the array = dim1 * dim2 * num bytes of our array type
        //If our input array is 2D (non-linearized), we want the actual
        //dimensions of the array (as it might be less than cu_n
        //CG_outputRepr* size;
        
        int numitems = 1;
        base->print(0, stderr); debug_fprintf(stderr, "\n");
        
        
        if (base->numdimensions < 1 || 
            base->getArrayDimensions() == 0) {
          //Lookup in array_dims (the cudaize call has this info for some variables?) 
          std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
          if (it == array_dims.end()) { 
            debug_fprintf(stderr, "L1097 Can't find %s in array_dims\n", name.c_str()); 
            numitems = 123456;
          }
          else { 
            debug_fprintf(stderr, "L1097 it %s %d\n", (*it).first.c_str(), (*it).second);  
            numitems = (*it).second; 
          }
          //debug_fprintf(stderr, "it %s %d\n", (*it).first.c_str(), (*it).second);  
          debug_fprintf(stderr, "LUA command says this variable %s should be size %d\n",  (*it).first.c_str(), (*it).second); 
          numitems = (*it).second; 
          
        }
        else { 
          debug_fprintf(stderr, "numdimensions = %d\n", base->numdimensions);
          for (int i=0; i<base->numdimensions; i++) { 
            numitems *= base->getArraySizeAsInt(i);
          }
        } 
        
        
        
        
        chillAST_IntegerLiteral *numthings = new chillAST_IntegerLiteral( numitems ); 
        
        //debug_fprintf(stderr, "creating int mult size expression numitems %d x sizeof( %s )\n", numitems, v.type ); 
        debug_fprintf(stderr, "OK, finally figured out the size expression for the input array ...  L1117\n"); 
        
        
        // create a mult  
        v.size_expr = new chillAST_BinaryOperator( numthings, "*", so ); // 1024 * sizeof(float)  etc
        
        v.CPUside_param = base;
        v.in_data = base;
        v.out_data = 0;
        
        
        debug_fprintf(stderr, "adding input v to arrayVars\n\n"); 
        v.print(); 
        arrayVars.push_back(v);   
      } // end of READ refs
#else
      //get_io_array_refs(array_dims, ro_refs, io_dir::read_only, arrayVars);
#endif
      //get_io_array_refs(array_dims, rw_refs, io_dir::read_write, arrayVars);

      debug_fprintf(stderr, "done with READ ONLY\n\n"); 
      
      
      
      debug_fprintf(stderr, "done with ORDERED STATEMENTS\n"); 
      debug_fprintf(stderr, "WHAT IS THIS TEST\n"); 
      
      debug_fprintf(stderr, "body_symtab now has %d entries\n\nbody_symtab\n", CPUbodySymtab->size());
      printSymbolTable(CPUbodySymtab);
      
      debug_fprintf(stderr, "adding mallocs and array memcpys\n"); 
      debug_fprintf(stderr, "%d arrayVars\n", arrayVars.size()); 
      
      for (int i = 0; i < arrayVars.size(); i++) {
        chillAST_VarDecl *var = arrayVars[i].vardecl;
        char *aname = strdup(arrayVars[i].name.c_str() ); 
        debug_fprintf(stderr, "\narrayVar %d   %s\n", i, aname); 
        arrayVars[i].print();
        
        // there is no way these created arrays (pointers) could be in the CPU function, but check anyway?
        //chillAST_VarDecl *vd = symbolTableFindVariableNamed( CPUBOD->, aname );
        var = CPUsidefunc->findVariableNamed( aname );
                  
        if ( !var )  { // didn't have an actual vardecl. probably always the case
          // make one 
          //debug_fprintf(stderr, "buildVariableDeclaration %s\n", arrayVars[i].name.c_str()); 
          // create a CHILL variable declaration and put it in the CPU side function
          
          char typ[128];
          sprintf(typ, "%s *", arrayVars[i].type); 
          
          var = new chillAST_VarDecl( typ,
                                      "",
                                      arrayVars[i].name.c_str());

        debug_fprintf(stderr, "adding decl for %s to CPUsidefunc %s\n", var->varname, CPUsidefunc->functionName); 
          CPUsidefunc->prependStatement( var ); // adds the decl to body code
          CPUsidefunc->addDecl( var ); // also adds to and CHANGES symbol table 
        }
      
        
        // store variable decl where we can get it easily later
        debug_fprintf(stderr, "body_symtab ADDING %s L2952\n", var->varname); 
        arrayVars[i].vardecl = var;
        
        //debug_fprintf(stderr, "body_symtab had %d entries\n", CPUbodySymtab->size()); 
        //debug_fprintf(stderr, "func        had %d entries\n", CPUsidefunc->getSymbolTable()->size()); 
        

        //debug_fprintf(stderr, "func        had %d entries after addDecl()\n", CPUsidefunc->getSymbolTable()->size()); 
        
        
        CPUbodySymtab = CPUsidefunc->getSymbolTable(); // needed ot bodysym is not up to date TODO
        
        //debug_fprintf(stderr, "body_symtab now has %d entries\n", CPUbodySymtab->size()); 
        debug_fprintf(stderr, "body_symtab now has %d entries\n", CPUsidefunc->getSymbolTable()->size()); 
        
        
        
        // do the CPU side cudaMalloc 
        debug_fprintf(stderr, "cudaize_codegen_v2.cc L1184  building CUDAmalloc using %s    i %d\n", aname, i); 
        //arrayVars[i].size_expr->print(0,stderr); debug_fprintf(stderr, "\n");

        // wait, malloc?
        chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( var );
        chillAST_CStyleAddressOf *AO = new chillAST_CStyleAddressOf( DRE );
        chillAST_CStyleCastExpr *casttovoidptrptr = new chillAST_CStyleCastExpr( "void **", AO );
        chillAST_CudaMalloc *cmalloc = new chillAST_CudaMalloc( casttovoidptrptr, arrayVars[i].size_expr);
        
        debug_fprintf(stderr, "adding cudamalloc to 'setup code' for the loop\n"); 
        CPUfuncbody->addChild( cmalloc ); // TODO setup_code ?? 
        setup_code = ocg->StmtListAppend(setup_code,
                                         new CG_chillRepr(cmalloc));
        
        //debug_fprintf(stderr, "\ncudamalloc is:\n");
        //cmalloc->print(); printf("\n"); fflush(stdout); 
        
        if (arrayVars[i].in_data) {  
          //debug_fprintf(stderr, "\nnow the memcpy (for input variables only????)\n"); 
          // if it's input to the calculation, and we need to copy the data to the GPU
          debug_fprintf(stderr, "it's an input to the calculation, so we need to copy the data to the GPU\n"); 
          
          // do the CPU side cudaMemcpy, CPU to GPU("device")
          //DRE = new chillAST_DeclRefExpr( var, CPUfuncbody ); 
          chillAST_CudaMemcpy *cmemcpy = new chillAST_CudaMemcpy( var,
                                                                  arrayVars[i].in_data->as<chillAST_VarDecl>(),
                                                                  arrayVars[i].size_expr, "cudaMemcpyHostToDevice"); 
          
          debug_fprintf(stderr, "cudamemcpy is:\n");
          cmemcpy->print(0, stderr); debug_fprintf(stderr, "\n");
          
          debug_fprintf(stderr, "adding cudamemcpy to 'setup code' for the loop\n"); 
          CPUfuncbody->addChild( cmemcpy ); // TODO setup_code ?? 
          setup_code = ocg->StmtListAppend(setup_code,
                                           new CG_chillRepr(cmemcpy));
          //printf("\n"); cmemcpy->print(); printf("\n");fflush(stdout); 
        }
        
      }  // for each arrayVar
      
      //debug_fprintf(stderr, "perhaps passing scalars to the kernel function\n"); 
      // seemingly not ?
      
      debug_fprintf(stderr, "\nBuild dimGrid dim3 variables based on loop dimensions and ti/tj\n"); 
      //Build dimGrid dim3 variables based on loop dimensions and ti/tj
      debug_fprintf(stderr, "dim3 variables will be dimGrid%d and dimBlock%d ??\n", CS, CS); 
      
      
      debug_fprintf(stderr, "create ARGS for dim3 %s\n", gridName);     
      
      int what = ordered_cudaized_stmts[CS].second;
      
      if (VbxAst.size() == 0) { // fake a constant 1
        
      }

      if ( what >=  VbxAst.size() || what >=  VbyAst.size()) { 
        debug_fprintf(stderr, "what %d\n", what); 
        debug_fprintf(stderr, "Vbx size %d   Vby size %d\n", VbxAst.size(), VbyAst.size()); 
        
        debug_fprintf(stderr, "time to die\n"); 
        exit(-1); 
      }
      debug_fprintf(stderr, "creating dim3 decl of %s( ", gridName );
      VbxAst[what]->print(0,stderr); debug_fprintf(stderr, ", "); 
      
      VbyAst[what]->print(0,stderr); debug_fprintf(stderr, " )\n"); 
      
      chillAST_CallExpr *CE1 = new chillAST_CallExpr( new chillAST_DeclRefExpr(dimbuiltin) );
      CE1->addArg(VbxAst[what]);
      CE1->addArg(VbyAst[what]);
      chillAST_VarDecl *dimgriddecl = new chillAST_VarDecl( "dim3", "", gridName);
      dimgriddecl->setInit(CE1);
      
      CPUfuncbody->addChild( dimgriddecl );  // TODO remove ?  setup_code ?? 
      debug_fprintf(stderr, "adding dim3 dimGrid to setup code for the statement\n\n");
      setup_code = ocg->StmtListAppend(setup_code,  new CG_chillRepr( dimgriddecl ) ); 
      
      
      
      debug_fprintf(stderr, "\nOK, now %s\n", blockName); 
      chillAST_VarDecl *dimblockdecl = NULL; 
      if (VtzAst.size() > what &&  // there is one
          VtzAst[what]) {          // it is not null 
        
        // there is a 3rd tz to be used
        debug_fprintf(stderr, "tx, ty, and tz\n");
        chillAST_CallExpr *CE2 = new chillAST_CallExpr( new chillAST_DeclRefExpr(dimbuiltin) );
        CE2->addArg(VtxAst[what]);
        CE2->addArg(VtyAst[what]);
        CE2->addArg(VtzAst[what]);
        dimblockdecl = new chillAST_VarDecl( "dim3", "", blockName );
        dimblockdecl->setInit(CE2);
        
      }
      else if (VtyAst.size() > what &&
               VtyAst[what]) { // no tz
        debug_fprintf(stderr, "tx and ty\n");
        chillAST_CallExpr *CE2 = new chillAST_CallExpr( new chillAST_DeclRefExpr(dimbuiltin) );
        CE2->addArg(VtxAst[what]);
        CE2->addArg(VtyAst[what]);
        dimblockdecl = new chillAST_VarDecl( "dim3", "", blockName );
        dimblockdecl->setInit(CE2);
        dimblockdecl->print(0,stderr); debug_fprintf(stderr, "\n"); 
      }
      else {
        debug_fprintf(stderr, "tx only\n");
        chillAST_CallExpr *CE2 = new chillAST_CallExpr( new chillAST_DeclRefExpr(dimbuiltin) );
        CE2->addArg(VtxAst[what]);
        dimblockdecl = new chillAST_VarDecl( "dim3", "", blockName );
        dimblockdecl->setInit(CE2);
        dimblockdecl->print(0, stderr); debug_fprintf(stderr, "\n");
      }
      
      // Anand code has test for NULL dimblockdecl ... 
      CPUfuncbody->addChild( dimblockdecl );  // TODO remove ?  setup_code ?? 
      debug_fprintf(stderr, "adding dim3 %s to setup code for the statement\n\n", blockName);
      setup_code = ocg->StmtListAppend(setup_code,  new CG_chillRepr( dimblockdecl ) ); 
      
      
      debug_fprintf(stderr, "\nconfig?  ( the kernel call?? )\n"); 
      
      //debug_fprintf(stderr, "\nkernel named\n");GPUKernel->print(0,stderr); debug_fprintf(stderr, "\n"); 
      
      chillAST_CallExpr *kcall = new chillAST_CallExpr( new chillAST_DeclRefExpr(GPUKernel) );
      kcall->grid = dimgriddecl; 
      kcall->block =  dimblockdecl; 
      debug_fprintf(stderr, "kernel function parameters\n"); 
      for (int i = 0; i < arrayVars.size(); i++) { 
        //Throw in a type cast if our kernel takes 2D array notation
        //like (float(*) [1024])
        
        if (arrayVars[i].tex_mapped || arrayVars[i].cons_mapped) { 
          if (arrayVars[i].tex_mapped) debug_fprintf(stderr, "arrayVars[i].tex_mapped\n"); 
          if (arrayVars[i].cons_mapped) debug_fprintf(stderr, "arrayVars[i].cons_mapped\n"); 
          continue;
        }
        
        chillAST_VarDecl *v = arrayVars[i].vardecl;
        chillAST_VarDecl *param = arrayVars[i].CPUside_param;
        
        debug_fprintf(stderr, "param i %d,  numdimensions %d\n", i, param->numdimensions); 
        
        if (param->numdimensions > 1) { 
          debug_fprintf(stderr, "array Var %d %s is multidimensional\n",i, v->varname);
          //v->print(0, stderr); debug_fprintf(stderr, "\n");
          param->print(0, stderr); debug_fprintf(stderr, "\n\n"); 

          char line[128];
          sprintf(line, "%s (*)", v->underlyingtype ); 
          //debug_fprintf(stderr, "line '%s'\n", line);
          // we'll pass in a cast of the variable instead of just the variable.
          for (int i=1; i<param->numdimensions; i++) { 
            int l = strlen(line);
            //debug_fprintf(stderr, "l %d\n", l); 
            char *ptr = &line[l];
            //debug_fprintf(stderr, "[%d]", param->arraysizes[i]); 
            sprintf(ptr, "[%d]", param->getArraySizeAsInt(i));
            //debug_fprintf(stderr, "i %d line '%s'\n", i, line);
            chillAST_CStyleCastExpr *CE = new chillAST_CStyleCastExpr(line, new chillAST_DeclRefExpr(v));
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
          chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr(v);
          kcall->addArg( DRE );
        }
      }  // for each arrayVar
 

     
      debug_fprintf(stderr, "kernel call is "); kcall->print(0,stderr); debug_fprintf(stderr, "\n"); 
      CPUfuncbody->addChild( kcall );           
      
      debug_fprintf(stderr, "\nfreeing %d Cuda variables\n", arrayVars.size()); 
      //cuda free variables
      for (int i = 0; i < arrayVars.size(); i++) {
        debug_fprintf(stderr, "arrayVar %d\n", i); 
        
        // Memcopy back if we have an output 
        if (arrayVars[i].out_data) {
          debug_fprintf(stderr, "Memcopy back if we have an output\n"); 
          chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( arrayVars[i].vardecl );
          chillAST_CudaMemcpy *cmemcpy = new chillAST_CudaMemcpy( arrayVars[i].out_data->as<chillAST_VarDecl>(), // wrong info
                                                                  arrayVars[i].vardecl,
                                                                  arrayVars[i].size_expr, "cudaMemcpyDeviceToHost"); 
          CPUfuncbody->addChild( cmemcpy );
        }
        
        // CudaFree the variable
        chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( arrayVars[i].vardecl );
        chillAST_CudaFree *cfree = new chillAST_CudaFree( arrayVars[i].vardecl );
        CPUfuncbody->addChild( cfree );
        
      } 
      debug_fprintf(stderr, "\nDONE freeing %d Cuda variables\n", arrayVars.size()); 
      
      
      
      debug_fprintf(stderr, "loop_cuda_chill.cc BUILD THE KERNEL\n"); 
      chillAST_node *kernelbody = code_temp;  // wrong 
      GPUKernel->setBody( kernelbody ); 
      
      
      //Extract out the CPU loop.  (part of) this will become the GPU side code
      chillAST_node *CPUsideloop = getCode(  );  // CG_chillRepr getCode
      debug_fprintf(stderr, "loop_cuda_chill.cc L1669 returned from getCode()\n");
      
      debug_fprintf(stderr, "\n\n\nloop_cuda_chill.cc L1685   CPUsideloop = \n");
      CPUsideloop->print(); 
      debug_fprintf(stderr, "\n\n"); 
      
      debug_fprintf(stderr, "\nGPUKernel:\n"); 
      GPUKernel->print(0, stderr); debug_fprintf(stderr, "\n\n"); 
      
      debug_fprintf(stderr, "\ncode_temp:\n"); 
      code_temp->print(0, stderr); debug_fprintf(stderr, "\n\n"); 
      
      
      // At this point, we have the CPU-side code, in CPUsideloop
      // we have code_temp, which is what will become the body of the GPU kernel
      
      
      
      // we need to figure out which variables need to be added to the kernel
      
// first, remove / replace loop variables 
      chillAST_SymbolTable *st =  GPUKernel->getSymbolTable();
      if (!st) st = new chillAST_SymbolTable;
      GPUKernel->findLoopIndexesToReplace( st, false );

      debug_fprintf(stderr, "\nfind variables used in the kernel (?)\n"); 
      // find all variables used in the function
      vector<chillAST_VarDecl*> decls;
      GPUKernel->gatherVarUsage( decls );  // finds decls even if the actual decl is not in the ast but the reference to it is
      
      // attempt to get xxdiff to sync
      debug_fprintf(stderr, "\n                                                      ****** recursiveFindRefs()\n");
      
      debug_fprintf(stderr, "%d vars in syms USED in code_temp that need to be added to func_d \n\n", (int)decls.size()); 
      for (int i=0; i<decls.size(); i++) { 
        debug_fprintf(stderr, "%d   %s \n", i, decls[i]->varname); 
        decls[i]->print(0,stderr); debug_fprintf(stderr, "\n"); 
      }
      debug_fprintf(stderr, "\n\n"); 
      

      //int nump = GPUKernel->parameters.size();
      //debug_fprintf(stderr, "\n%d parameters to GPUKernel\n", nump); 
      //for (int i=0; i<nump; i++) debug_fprintf(stderr, "parameter %s\n",  GPUKernel->parameters[i]->varname );
      //debug_fprintf(stderr, "\n"); 
      
      
      
      //Figure out which loop variables will be our thread and block dimension variables
      debug_fprintf(stderr, "\nFigure out which loop variables will be our thread and block dimension variables\n"); 
      
      int beforecode = 0; // offset for var decls
      
      //Get our indexes  (threadIdx and blockIdx will replace some loops) 
      std::vector<const char*> indexes;
      
      //here, as we figure out that bx, by tx, etc will be loop variables, 
      //  we need to note that the thing they replace will NOT be needed. 
        
      if (VbxAst[what])  { // Vcu_bx[what] > 1 || cu_bx_repr[what]) {
        debug_fprintf(stderr, "adding bx to indexes\n"); 
        indexes.push_back("bx");
        
        // add definition of bx, and blockIdx.x to it in GPUKernel
        // int bx;
        chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.x", "int", GPUKernel );
        chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 
        chillAST_VarDecl *bxdecl;
        // see if bx is already defined in the Kernel
        bxdecl = GPUKernel->funcHasVariableNamed( "bx" );
        if (!bxdecl) { 
          debug_fprintf(stderr, "bx was NOT defined in GPUKernel before\n"); 
          GPUKernel->print(0,stderr); debug_fprintf(stderr, "\n\n"); 
          bxdecl= new chillAST_VarDecl( "int", "", "bx");
          GPUKernel->addDecl( bxdecl ); // to symbol table
          // if it was there, we shouldn't do this? 
          GPUKernel->prependStatement( bxdecl );
        }
        else debug_fprintf(stderr, "bx WAS defined in GPUKernel before\n"); 
        bxdecl->setInit( bid );  // add init

        //GPUKernel->addVariableToSymbolTable( bxdecl ); 


        // separate assign statement (if not using the vardecl init )
        //chillAST_DeclRefExpr *bx = new chillAST_DeclRefExpr( bxdecl ); 
        //chillAST_BinaryOperator *assign = new chillAST_BinaryOperator( bx, "=",bid); 
        //assign->print(0,stderr); debug_fprintf(stderr, "\n"); 
        //GPUKernel->addChild(assign); 

        // remove the 
      }
      
      
      if (VbyAst.size() > 0 && VbyAst[what]) { 
        debug_fprintf(stderr, "adding by to indexes\n");  // TODO wrong test 
        indexes.push_back("by");
        
        // add definition of by, and blockIdx.y to it in GPUKernel
        chillAST_VarDecl *biddecl = addBuiltin( "blockIdx.y", "int", GPUKernel);
        chillAST_DeclRefExpr *bid = new chillAST_DeclRefExpr( biddecl ); 

        chillAST_VarDecl *bydecl;
        // see if by is already defined in the Kernel
        bydecl = GPUKernel->funcHasVariableNamed( "by" ); 
        if (!bydecl) { 
          debug_fprintf(stderr, "by was NOT defined in GPUKernel before\n"); 
          bydecl= new chillAST_VarDecl( "int", "", "by");
          GPUKernel->addDecl( bydecl ); // to symbol table
          GPUKernel->prependStatement( bydecl );
        }
        else debug_fprintf(stderr, "by WAS defined in GPUKernel before\n"); 
        bydecl->setInit( bid ); // add init
        
        // separate assign statement (if not using the vardecl init )
        //chillAST_DeclRefExpr *by = new chillAST_DeclRefExpr( bydecl ); 
        //chillAST_BinaryOperator *assign = new chillAST_BinaryOperator(by,"=",bid); 
        //assign->print(0,stderr); debug_fprintf(stderr, "\n"); 
        //GPUKernel->addChild(assign); 
      }
      
      if (VtxAst.size() > 0 && VtxAst[what]) { 
        debug_fprintf(stderr, "adding tx to indexes\n"); 
        indexes.push_back("tx");
        chillAST_VarDecl *tiddecl = addBuiltin( "threadIdx.x", "int", GPUKernel);
        chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( tiddecl ); 

        chillAST_VarDecl *txdecl;
        // see if tx is already defined in the Kernel
        txdecl = GPUKernel->funcHasVariableNamed( "tx" ); 
        if (!txdecl) { 
          GPUKernel->print(0,stderr); 

          debug_fprintf(stderr, "tx was NOT defined in GPUKernel before\n"); 
          txdecl= new chillAST_VarDecl( "int", "", "tx");
          GPUKernel->addDecl( txdecl ); // to symbol table
          GPUKernel->prependStatement( txdecl );
        }
        else debug_fprintf(stderr, "tx WAS defined in GPUKernel before\n"); 
        txdecl->setInit( tid ); // add init

      }
      
      if (VtyAst.size() > 0 && VtyAst[what]) { 
        debug_fprintf(stderr, "adding ty to indexes\n"); 
        indexes.push_back("ty");
        chillAST_VarDecl *tiddecl = addBuiltin( "threadIdx.y", "int", GPUKernel);
        chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( tiddecl ); 

        chillAST_VarDecl *tydecl;
        // see if ty is already defined in the Kernel
        tydecl = GPUKernel->funcHasVariableNamed( "ty" ); 
        if (!tydecl) { 
          debug_fprintf(stderr, "ty was NOT defined in GPUKernel before\n"); 
          GPUKernel->print(0, stderr); 

          tydecl= new chillAST_VarDecl( "int", "", "ty");
          GPUKernel->addDecl( tydecl ); // to symbol table
          GPUKernel->prependStatement( tydecl );
        }
        else debug_fprintf(stderr, "ty WAS defined in GPUKernel before\n"); 
        tydecl->setInit( tid ); // add init
      }
      
      if (VtzAst.size() > 0 && VtzAst[what]) { 
        debug_fprintf(stderr, "adding tz to indexes\n"); 
        indexes.push_back("tz");
        chillAST_VarDecl *tiddecl = addBuiltin( "threadIdx.z", "int", GPUKernel);
        chillAST_DeclRefExpr *tid = new chillAST_DeclRefExpr( tiddecl ); 

        chillAST_VarDecl *tzdecl;
        // see if tz is already defined in the Kernel
        tzdecl = GPUKernel->funcHasVariableNamed( "tz" ); 
        if (!tzdecl) { 
          debug_fprintf(stderr, "tz was NOT defined in GPUKernel before\n"); 
          tzdecl= new chillAST_VarDecl( "int", "", "tz");
          GPUKernel->addDecl( tzdecl ); // to symbol table
          GPUKernel->prependStatement( tzdecl );
          //GPUKernel->insertChild(beforecode++, tzdecl);
        }
        else debug_fprintf(stderr, "tz WAS defined in GPUKernel before\n"); 
        tzdecl->setInit( tid ); // add init

      }
      
      
      
      debug_fprintf(stderr, "\n%d indexes\n", (int) indexes.size()); 
      for (int i = 0; i < indexes.size(); i++) {
        debug_fprintf(stderr, "indexes[%i] = '%s'\n", i, indexes[i] ); 
      }
      debug_fprintf(stderr, "\n"); 
      
      
      std::vector<chillAST_ArraySubscriptExpr*> kernelArrayRefs;
      code_temp->gatherArrayRefs( kernelArrayRefs, false );
      //debug_fprintf(stderr, "%d array refs in kernel\n",  kernelArrayRefs.size()); 
      

      // Look for arrayrefs used in kernel, to see if they are parameters to the GPU kernel (?)
      // and if it IS, check to see if it's a member of a struct accessed by a dot, and if it IS,
      // change it to a non-member expression  ??? 
      //
      debug_fprintf(stderr, "\nloop_cuda_chill.cc L2072  looking for %d arrayRefs\n", (int) kernelArrayRefs.size()); 
      for (int i = 0; i < kernelArrayRefs.size(); i++) { 
        //chillAST_node *vd = kernelArrayRefs[i]->multibase2(); 
        
        debug_fprintf(stderr, "ref %d = '", i);
        kernelArrayRefs[i]->multibase()->print(0,stderr); debug_fprintf(stderr, "'\n");
        //kernelArrayRefs[i]->print(0,stderr); debug_fprintf(stderr, "'\n");
        //kernelArrayRefs[i]->base->print(0,stderr); debug_fprintf(stderr, "'\n");
      }
      debug_fprintf(stderr, "in %d kernel_parameters\n", (int)kernel_parameters.size()); 
      for (std::set<std::string>::iterator i = kernel_parameters.begin();
           i != kernel_parameters.end(); i++) {
        debug_fprintf(stderr, "kernel parameter '%s'\n", (*i).c_str()); 
      }
      debug_fprintf(stderr, "\n"); 
      
      // Add remaining declarations
      for(auto v: decls) {
          if(!GPUKernel->funcHasVariableNamed(v->varname)) {
              GPUKernel->addDecl(v);
              GPUKernel->prependStatement(v);
          }
      }
      
      // TODO ... stuff comment above
      
      
      //debug_fprintf(stderr, "\nbefore swapVarReferences(), code is\n{\n"); 
      //kernelbody->print();
      //debug_fprintf(stderr, "}\n\nswapVarReferences()\n"); 
      //swapVarReferences( CPUsideloop, GPUKernel );
      //debug_fprintf(stderr, "\nafter swapVarReferences(), ");
      
      debug_fprintf(stderr, "kernel code is\n"); 
      GPUKernel->print();
      debug_fprintf(stderr, "\n\n");
      
      debug_fprintf(stderr, "now replace indexes ... (and add syncs)\n");
      findReplacePreferedIdxs( CPUsideloop, GPUKernel );
      debug_fprintf(stderr, "DONE WITH replace indexes ... (and add syncs)\n"); 
      
      debug_fprintf(stderr, "\nswapped 2\nshould have syncs\nshould have indexes replaced by bx, tx, etc \n"); 
      GPUKernel->print();
      
      
      
      
      //CPUsidefunc->print(0, stderr); debug_fprintf(stderr, "\n\n\n"); 
      
      GPUKernel->print(0, stderr); debug_fprintf(stderr, "\n\n\n");
      
      // now remove loops that will be done by spreading the loop count across cores
      // these are loops that have our indeces gathered above as loop variables
      debug_fprintf(stderr, "removing loops for variables that will be determined by core index\n"); 
      for (int i = 0; i < indexes.size(); i++) {
        debug_fprintf(stderr, "\nindexes[%i] = '%s'\n", i, indexes[i] ); 
        debug_fprintf(stderr, "forReduce()\n");
        
        GPUKernel->getBody()->loseLoopWithLoopVar( strdup(indexes[i]) ); 
      }
      
      
      GPUKernel->print(0, stderr); debug_fprintf(stderr, "\n\n\n");
    
    } // if THIS statement will become a kernel
    else 
    {
      debug_fprintf(stderr, "statement %d will NOT run on GPU\n", CS); 
    }


  } // for int CS  (each statement that COULD  become a kernel )
  
  debug_fprintf(stderr, "returning from cudaize_codegen_v2()\n"); 
  return NULL; 
}
