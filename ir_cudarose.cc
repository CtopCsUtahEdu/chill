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
  
  fprintf(stderr, "IR_cudaroseCode::IR_cudaroseCode()\n"); 
  //std::string file_suffix = StringUtility::fileNameSuffix(filename);


  std::string orig_name  = StringUtility::stripPathFromFileName(filename);
  std::string naked_name = StringUtility::stripFileSuffixFromFileName(
    orig_name);
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
  fprintf(stderr, "IR_cudaroseCode::IR_cudaroseCode()  func_defn=%p\n", func_defn); 
}



IR_ArraySymbol *IR_cudaroseCode::CreateArraySymbol(const IR_Symbol *sym,
                                                   std::vector<omega::CG_outputRepr *> &size, int sharedAnnotation) {

  //fprintf(stderr, "IR_cudaroseCode::CreateArraySymbol() gonna die\n");
  fprintf(stderr, "IR_cudaXXXXCode::CreateArraySymbol( sym = %s )\n", sym->name().c_str());
  fprintf(stderr, "size.size() %d\n", size.size()); 

  static int rose_array_counter = 1;
  std::string s = std::string("_P") + omega::to_string(rose_array_counter++);
  fprintf(stderr, "new array name is %s\n", s.c_str()); 

  if (typeid(*sym)  == typeid(IR_roseArraySymbol)) {
    //fprintf(stderr, "%s is an array\n",  sym->name().c_str());
    IR_roseArraySymbol *asym = (IR_roseArraySymbol *) sym;

    chillAST_VarDecl *vd = asym->chillvd;  // ((const IR_roseArraySymbol *)sym)->chillvd;
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
    IR_roseArraySymbol *newsym = new IR_roseArraySymbol( asym->ir_, newarray );
    if (sharedAnnotation == 1) { 
      fprintf(stderr, "%s is SHARED\n", newarray->varname );
      newarray->isShared = true; 
    }
    fprintf(stderr, "done making a new array symbol\n"); 
    return newsym; 
  }
  fprintf(stderr, "IR_cudaroseCode::CreateArraySymbol() but old symbol is not an array???\n"); 
  exit(-1);

  return NULL; // can't get to here 
}

bool IR_cudaroseCode::commit_loop(Loop *loop, int loop_num) {
  fprintf(stderr, "IR_cudaROSECode::commit_loop()\n");

  if (loop == NULL)
    return true;
 
  //loop->printCode(); 
  //fprintf(stderr, "loop->printCode done\n\n"); 
 
  LoopCuda *cu_loop = (LoopCuda *) loop;

  fprintf(stderr, "IR_cudaxxxxCode::commit_loop() calling cu_loop->codegen()\n"); 
  chillAST_node * loopcode = cu_loop->codegen();
  fprintf(stderr, "IR_cudaxxxxCode::commit_loop()   codegen DONE\n");

  if (!loopcode)
    return false;
  
  fprintf(stderr, "loopcode is\n");
  loopcode->print(); fflush(stdout);
  fprintf(stderr, "(END LOOPCODE)\n\n\n"); 
  

  // put "loopcode" into GPUside ?? easier in codegen
  if (NULL == entire_file_AST) { 
    fprintf(stderr, "IR_cudaroseCode::commit_loop(),  entire_file_AST == NULL!\n");
    exit(-1); 
  }
  entire_file_AST->print();

  return true;
}

IR_cudaroseCode::~IR_cudaroseCode() {
}

#endif
