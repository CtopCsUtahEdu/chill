/*****************************************************************************
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
   CHiLL's CLANG interface.

 Notes:
   Array supports mixed pointer and array type in a single declaration.

 History:
   2/2/2011 Created by Protonu Basu. 
*****************************************************************************/

#include <typeinfo>
#include "ir_cudaclang.hh"
#include "loop.hh"
#include "loop_cuda_clang.hh"



IR_cudaclangCode::IR_cudaclangCode(const char *paramfilename, const char* proc_name) :
  IR_clangCode(paramfilename, strdup(proc_name)) {
  
  // filename and procedurename are internal to IR_clangcode and therefore part of IR_cudaclangcode
  fprintf(stderr, "IR_cudaxxxxCode::IR_cudaxxxxCode( %s, %s )\n", filename, procedurename); // proc_name );

  char *fname = strdup(paramfilename);
  char *f = fname;
  char *ptr = rindex(fname, '/');
  if (ptr) fname = ptr + 1;
  
  std::string orig_name(fname); 
  
  char *dot = index(fname, '.'); 
  if (dot) *dot = '\0';

  std::string naked_name( fname );
  cudaFileToWrite = "clang_" + naked_name + ".cu";
  //fprintf(stderr, "will write file %s\n",  cudaFileToWrite.c_str()); 
  chillfunc->getSourceFile()->setFileToWrite( strdup( cudaFileToWrite.c_str()) ); 
  func_defn = chillfunc;  
  
  fprintf(stderr, "IR_cudaxxxxCode::IR_cudaxxxxCode()  DONE\n"); 
}



IR_ArraySymbol *IR_cudaclangCode::CreateArraySymbol(const IR_Symbol *sym,
                                                    std::vector<omega::CG_outputRepr *> &size, 
                                                    int sharedAnnotation) {
  fprintf(stderr, "IR_cudaclangCode::CreateArraySymbol( sym = %s )\n", sym->name().c_str());
  fprintf(stderr, "size.size() %d\n", size.size()); 

  static int clang_array_counter = 1;
  std::string s = std::string("_P") + omega::to_string(clang_array_counter++);
  fprintf(stderr, "new array name is %s\n", s.c_str()); 

  if (typeid(*sym)  == typeid(IR_clangArraySymbol)) {
    //fprintf(stderr, "%s is an array\n",  sym->name().c_str());
    IR_clangArraySymbol *asym = (IR_clangArraySymbol *) sym;

    chillAST_VarDecl *vd = asym->chillvd;  // ((const IR_clangArraySymbol *)sym)->chillvd;
    vd->print(); printf("\n"); fflush(stdout); 
    fprintf(stderr, "%s %s   %d dimensions    arraypart '%s'\n", vd->vartype, vd->varname, vd->numdimensions, vd->arraypart); 

    chillAST_VarDecl *newarray =  (chillAST_VarDecl *)vd->clone();

    char arraystring[128];
    char *aptr = arraystring;
    for (int i=0; i<size.size(); i++) { 
      omega::CG_chillRepr *CR = (omega::CG_chillRepr *) size[i];
      chillAST_IntegerLiteral *IL = (chillAST_IntegerLiteral *) ( (CR->getChillCode()) [0]);
      printf("size[%d]  ", i); IL->print(); printf("\n"); fflush(stdout);
      newarray->arraysizes[i] = IL->value; // this could die if new var will have MORE dimensions than the one we're copying

      sprintf(aptr, "[%d]",  IL->value); 
      aptr += strlen(aptr);
    }
    fprintf(stderr, "arraypart WAS %s  now %s\n", newarray->arraypart, arraystring); 
    newarray->arraypart = strdup(arraystring);
    newarray->numdimensions =  size.size(); 


    fprintf(stderr, "newarray numdimensions %d\n", newarray->numdimensions); 
    newarray->varname = strdup(s.c_str()); 
    IR_clangArraySymbol *newsym = new IR_clangArraySymbol( asym->ir_, newarray, asym->offset_ );
    if (sharedAnnotation == 1) { 
      fprintf(stderr, "%s is SHARED\n", newarray->varname );
      newarray->isShared = true; 
    }
    fprintf(stderr, "done making a new array symbol\n"); 
    return newsym; 
  }
  fprintf(stderr, "IR_cudaclangCode::CreateArraySymbol() but old symbol is not an array???\n"); 
  exit(-1);

  IR_ArraySymbol *smb = new IR_clangArraySymbol(this, NULL);
  fprintf(stderr, "done making a new array symbol\n\n");
  return smb;
}



bool IR_cudaclangCode::commit_loop(Loop *loop, int loop_num) {
  if (loop == NULL)
    return true;
  
  fprintf(stderr, " IR_cudaxxxxCode::commit_loop()\n");
  loop->printCode(); 
  fprintf(stderr, "loop->printCode done\n\n"); 


  LoopCuda *cu_loop = (LoopCuda *) loop;
  chillAST_node * loopcode = cu_loop->codegen();
  if (!loopcode)
    return false;
  
  fprintf(stderr, "IR_cudaxxxxCode::commit_loop()   codegen DONE\n");
  fprintf(stderr, "loopcode is\n");
  loopcode->print(); fflush(stdout);
  fprintf(stderr, "(END LOOPCODE)\n\n\n"); 
  

  // put "loopcode" into GPUside ?? easier in codegen
  
  entire_file_AST->print();

  return NULL;
}

IR_cudaclangCode::~IR_cudaclangCode() {
}

