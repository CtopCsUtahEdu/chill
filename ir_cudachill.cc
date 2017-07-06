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

#include <typeinfo>
#include "ir_cudachill.hh"
#include "loop.hh"

// these are very similar (the same?) 
#include "loop_cuda_chill.hh"

IR_cudaChillCode::IR_cudaChillCode(chill::Parser *parser, const char *filename, const char* proc_name, const char* dest_name) :
  IR_chillCode(parser ,filename, proc_name, dest_name) {
  
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

  func_defn = chillfunc;  // func->get_definition();
}



IR_ArraySymbol *IR_cudaChillCode::CreateArraySymbol(const IR_Symbol *sym,
                                                   std::vector<omega::CG_outputRepr *> &size, 
                                                   int sharedAnnotation) {

  debug_fprintf(stderr, "\nCUDAROSECODE IR_cudaXXXXCode::CreateArraySymbol( sym = %s )\n", sym->name().c_str());
  debug_fprintf(stderr, "size.size() %d\n", size.size()); 

  static int rose_array_counter = 1;
  std::string s = std::string("_P") + omega::to_string(rose_array_counter++);
  debug_fprintf(stderr, "new array name is %s\n", s.c_str()); 
  
  if (typeid(*sym)  == typeid(IR_chillArraySymbol)) {
    debug_fprintf(stderr, "%s is an array\n",  sym->name().c_str());
    IR_chillArraySymbol *asym = (IR_chillArraySymbol *) sym;

    if (asym->base->isMemberExpr()) {
      debug_fprintf(stderr, "arraySymbol is a MemberExpr  "); asym->base->print(0,stderr); debug_fprintf(stderr, "\n"); 
    }

    chillAST_VarDecl *vd = asym->chillvd;  // ((const IR_roseArraySymbol *)sym)->chillvd;
    debug_fprintf(stderr, "vd is a %s\n", vd->getTypeString()); 
    vd->print(); printf("\n"); fflush(stdout); 


    //debug_fprintf(stderr, "%s %s   %d dimensions    arraypart '%s'\n", vd->vartype, vd->varname, vd->numdimensions, vd->arraypart);

    chillAST_NodeList arr;

    for (int i=0; i<size.size(); i++) { 
      omega::CG_chillRepr *CR = (omega::CG_chillRepr *) size[i];
      chillAST_node *n = (CR->getChillCode()) [0];

      //chillAST_IntegerLiteral *IL = (chillAST_IntegerLiteral*) ((CR->getChillCode()) [0]);

      //printf("size[%d] (INTEGER LITERAL??)  '", i); IL->print(); printf("'\n"); fflush(stdout);
      arr.push_back(n);
    }
    //debug_fprintf(stderr, "arraypart WAS %s  now %s\n", newarray->arraypart, arraystring);
    chillAST_VarDecl *newarray = new chillAST_VarDecl(vd->underlyingtype, "", s.c_str(), arr);
    chillfunc->getBody()->insertChild( 0, newarray);  // is this always the right function to add to?
    chillfunc->addVariableToSymbolTable( newarray ); // always right?

    debug_fprintf(stderr, "newarray numdimensions %d\n", newarray->numdimensions); 
    IR_chillArraySymbol *newsym = new IR_chillArraySymbol( asym->ir_, newarray );
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

bool IR_cudaChillCode::commit_loop(Loop *loop, int loop_num) {
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
