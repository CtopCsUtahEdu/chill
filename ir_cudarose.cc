/*****************************************************************************
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
   CHiLL's ROSE interface.

 Notes:
   Array supports mixed pointer and array type in a single declaration.

 History:
   2/2/2011 Created by Protonu Basu. 
*****************************************************************************/

#ifdef FRONTEND_ROSE 

#include <typeinfo>
#include "ir_cudarose.hh"
#include "loop.hh"

// these are very similar (the same?) 
#include "loop_cuda_chill.hh"


using namespace SageBuilder;
using namespace SageInterface;

IR_cudaroseCode::IR_cudaroseCode(const char *filename, const char* proc_name) :
  IR_roseCode(filename, proc_name, NULL) {
  
  debug_fprintf(stderr, "IR_cudaroseCode::IR_cudaroseCode()\n"); 
  //std::string file_suffix = StringUtility::fileNameSuffix(filename);
  char *fname = strdup(filename);
  char *f = fname;
  char *ptr = rindex(fname, '/');
  if (ptr) fname = ptr + 1;
  
  std::string orig_name(fname); 
  
  char *dot = index(fname, '.'); 
  if (dot) *dot = '\0';

  std::string naked_name( fname );
  //file->set_unparse_output_filename("rose_" + naked_name + ".cu");
  cudaFileToWrite = "rose_" + naked_name + ".cu";
  chillfunc->getSourceFile()->setFileToWrite( strdup( cudaFileToWrite.c_str())); 

  // these are from when there were Rose (Sg*) internals
  //gsym_ = root;
  //first_scope = firstScope;
  //parameter = symtab2_;
  //body = symtab3_;
  //defn = func->get_definition()->get_body();
  func_defn = chillfunc;  // func->get_definition();
  debug_fprintf(stderr, "IR_cudaroseCode::IR_cudaroseCode()  func_defn=%p\n", func_defn); 
}



IR_ArraySymbol *IR_cudaroseCode::CreateArraySymbol(const IR_Symbol *sym,
                                                   std::vector<omega::CG_outputRepr *> &size, 
                                                   int sharedAnnotation) {

  debug_fprintf(stderr, "\nCUDAROSECODE IR_cudaXXXXCode::CreateArraySymbol( sym = %s )\n", sym->name().c_str());
  debug_fprintf(stderr, "size.size() %d\n", size.size()); 

  static int rose_array_counter = 1;
  std::string s = std::string("_P") + omega::to_string(rose_array_counter++);
  debug_fprintf(stderr, "new array name is %s\n", s.c_str()); 
  
  if (typeid(*sym)  == typeid(IR_roseArraySymbol)) {
    debug_fprintf(stderr, "%s is an array\n",  sym->name().c_str());
    IR_roseArraySymbol *asym = (IR_roseArraySymbol *) sym;

    if (asym->base->isMemberExpr()) {
      debug_fprintf(stderr, "arraySymbol is a MemberExpr  "); asym->base->print(0,stderr); debug_fprintf(stderr, "\n"); 
    }

    chillAST_VarDecl *vd = asym->chillvd;  // ((const IR_roseArraySymbol *)sym)->chillvd;
    debug_fprintf(stderr, "vd is a %s\n", vd->getTypeString()); 
    vd->print(); printf("\n"); fflush(stdout); 


    debug_fprintf(stderr, "%s %s   %d dimensions    arraypart '%s'\n", vd->vartype, vd->varname, vd->numdimensions, vd->arraypart); 

    chillAST_VarDecl *newarray =  (chillAST_VarDecl *)vd->clone();
    newarray->varname = strdup( s.c_str() ); // 
    
    chillfunc->insertChild( 0, newarray);  // is this always the right function to add to? 
    chillfunc->addVariableToSymbolTable( newarray ); // always right? 

    char arraystring[128];
    char *aptr = arraystring;
    
    if (newarray->arraysizes) free(  newarray->arraysizes ); 
    newarray->arraysizes = (int *)malloc(size.size() * sizeof(int)); 

    for (int i=0; i<size.size(); i++) { 
      omega::CG_chillRepr *CR = (omega::CG_chillRepr *) size[i];
      chillAST_node *n = (CR->getChillCode()) [0];
      debug_fprintf(stderr, "size[%d] is a %s\n", i, n->getTypeString()); 
      n->print(0,stderr); debug_fprintf(stderr, "\n");

      int value = n->evalAsInt(); 
      debug_fprintf(stderr, "value is %d\n", value); 


      //chillAST_IntegerLiteral *IL = (chillAST_IntegerLiteral*) ((CR->getChillCode()) [0]);

      //printf("size[%d] (INTEGER LITERAL??)  '", i); IL->print(); printf("'\n"); fflush(stdout);
      newarray->arraysizes[i] = value; // this could die if new var will have MORE dimensions than the one we're copying

      sprintf(aptr, "[%d]",  value); 
      aptr += strlen(aptr);
    }
    debug_fprintf(stderr, "arraypart WAS %s  now %s\n", newarray->arraypart, arraystring); 
    newarray->arraypart = strdup(arraystring);
    newarray->numdimensions =  size.size(); 


    debug_fprintf(stderr, "newarray numdimensions %d\n", newarray->numdimensions); 
    newarray->varname = strdup(s.c_str()); 
    IR_roseArraySymbol *newsym = new IR_roseArraySymbol( asym->ir_, newarray );
    if (sharedAnnotation == 1) { 
      debug_fprintf(stderr, "%s is SHARED\n", newarray->varname );
      newarray->isShared = true; 
    }
    debug_fprintf(stderr, "done making a new array symbol\n"); 
    return newsym; 
  }
  debug_fprintf(stderr, "IR_cudaroseCode::CreateArraySymbol() but old symbol is not an array???\n"); 
  exit(-1);

  return NULL; // can't get to here 
}

bool IR_cudaroseCode::commit_loop(Loop *loop, int loop_num) {
  debug_fprintf(stderr, "IR_cudaROSECode::commit_loop()\n");

  if (loop == NULL)
    return true;
 
  //loop->printCode(); 
  //debug_fprintf(stderr, "loop->printCode done\n\n"); 
 
  LoopCuda *cu_loop = (LoopCuda *) loop;

  debug_fprintf(stderr, "IR_cudaxxxxCode::commit_loop() calling cu_loop->codegen()\n"); 
  chillAST_node * loopcode = cu_loop->codegen();
  debug_fprintf(stderr, "IR_cudaxxxxCode::commit_loop()   codegen DONE\n");

  if (!loopcode)
    return false;
  
  debug_fprintf(stderr, "loopcode is\n");
  loopcode->print(); fflush(stdout);
  debug_fprintf(stderr, "(END LOOPCODE)\n\n\n"); 
  

  // put "loopcode" into GPUside ?? easier in codegen
  if (NULL == entire_file_AST) { 
    debug_fprintf(stderr, "IR_cudaroseCode::commit_loop(),  entire_file_AST == NULL!\n");
    exit(-1); 
  }
  entire_file_AST->print();

  return true;
}

IR_cudaroseCode::~IR_cudaroseCode() {
}

#endif
