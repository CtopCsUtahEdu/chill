

#include "chilldebug.h"
#include "chill_ast.hh"

int chillAST_node::chill_scalar_counter = 0;
int chillAST_node::chill_array_counter  = 1;


const char* Chill_AST_Node_Names[] = { 
  "Unknown AST node type",
  "SourceFile",
  "TypedefDecl",
  "VarDecl",
  //  "ParmVarDecl",  not used any more
  "FunctionDecl",
  "RecordDecl",
  "MacroDefinition", 
  "CompoundStmt",
  "ForStmt",
  "TernaryOperator",
  "BinaryOperator",
  "UnaryOperator",
  "ArraySubscriptExpr",
  "MemberExpr",
  "DeclRefExpr",
  "IntegerLiteral",
  "FloatingLiteral",
  "ImplicitCastExpr", // not sure we need this
  "ReturnStmt",
  "CallExpr",
  "DeclStmt", 
  "ParenExpr",
  "CStyleCastExpr",
  "CStyleAddressOf",
  "IfStmt",
  "SizeOf",
  "Malloc", 
  "Free",
  "NoOp",
// CUDA specific 
  "CudaMalloc",
  "CudaFree",
  "CudaMemcpy",
  "CudaKernelCall",
  "CudaSyncthreads",
  "fake1",
  "fake2", 
  "fake3"
};

char *parseUnderlyingType( char *sometype ) {
  int len = strlen(sometype);
  //fprintf(stderr, "parseUnderlyingType( %s )\n", sometype); 
  char *underlying = strdup(sometype); 
  char *p;
  char *start = underlying;

  // ugly.  we want to turn "float *" into "float" but "struct abc *" into struct abc.
  // there are probably many more cases. have an approved list?   TODO 
  if (strstr(underlying, "struct ")) start += 7;  // (length("struct "))
  //fprintf(stderr, "sometype '%s'   start '%s'\n", sometype, start); 
  if (p = index(start, ' ')) *p = '\0'; // end at first space     leak
  if (p = index(start, '[')) *p = '\0'; // leak
  if (p = index(start, '*')) *p = '\0'; // leak
  
  return underlying; 
}

void printSymbolTable( chillAST_SymbolTable *st ) { 
  //printf("%d entries\n", st->size()); 
  if (!st) return;
  for (int i=0; i<st->size(); i++) {  printf("%d  ", i ); (*st)[i]->printName(); printf("\n"); } 
  if (st->size() )printf("\n");
  fflush(stdout); 
}

void printSymbolTableMoreInfo( chillAST_SymbolTable *st ) { 
  //printf("%d entries\n", st->size()); 
  if (!st) return;
  for (int i=0; i<st->size(); i++) { printf("%d  ", i ); (*st)[i]->print(); printf("\n"); } 
  if (st->size() )printf("\n");
  fflush(stdout); 
}


bool symbolTableHasVariableNamed( chillAST_SymbolTable *table, const char *name ) {
  if (!table) return false; // ?? 
  int numvars = table->size();
  for (int i=0; i<numvars; i++) { 
    chillAST_VarDecl *vd = (*table)[i];
    if (!strcmp(name, vd->varname)) return true;  // need to check type? 
  }
  return false;
}



chillAST_VarDecl *symbolTableFindVariableNamed( chillAST_SymbolTable *table, const char *name ){  // fwd decl TODO too many similar named functions
  if (!table) return NULL; // ?? 
  int numvars = table->size();
  for (int i=0; i<numvars; i++) { 
    chillAST_VarDecl *vd = (*table)[i];
    if (!strcmp(name, vd->varname)) return vd;  // need to check type? 
  }
  return NULL;
}



char *ulhack( char *brackets ) // remove UL from numbers, MODIFIES the argument!
{
  //fprintf(stderr, "ulhack( \"%s\"  -> ", brackets); 
 // another hack. remove "UL" from integers 
  int len = strlen(brackets);
  for (int i=0; i< len-2; i++) {
    if (isdigit(brackets[i])) {
      if (brackets[i+1] == 'U' && brackets[i+2] == 'L') { 
        // remove 
        for (int j=i+3; j<len; j++) brackets[j-2] = brackets[j];
        len -=2;
        brackets[len] = '\0';
      }
    }
  }
  //fprintf(stderr, "\"%s\" )\n", brackets); 
  return brackets;
}


char *restricthack( char *typeinfo ) // remove __restrict__ , MODIFIES the argument!
{
  //if (!isRestrict( typeinfo )) return typeinfo;

  // there is a "__restrict__ " somewhere embedded. remove it.
  // duplicate work 
  string r( "__restrict__" );
  string t( typeinfo );
  size_t index = t.find( r );

  if (index == std::string::npos)  return typeinfo; 

  char *c = &( typeinfo[index] ); 
  char *after = c + 12;
  if (*after == ' ') after++;

  //fprintf(stderr, "after = '%s'\n", after); 

  while (*after != '\0') *c++ = *after++; 
  *c = '\0';

  return typeinfo; 

}





char *parseArrayParts( char *sometype ) {
  int len = strlen(sometype);
  char *arraypart = (char *)calloc(1 + strlen(sometype), sizeof(char));// leak

  int c = 0;
  for (int i=0; i<strlen(sometype); ) { 
    if ( sometype[i] == '*') arraypart[c++] = '*';
    if ( sometype[i] == '[') { 
      while (sometype[i] != ']') {
        arraypart[c++] = sometype[i++];
      }
      arraypart[c++] = ']'; 
    }
    i += 1;
  }
  ulhack(arraypart);
  restricthack( arraypart );

  //fprintf(stderr, "parseArrayParts( %s ) => %s\n", sometype, arraypart); 
  return arraypart;
}







char *splitTypeInfo( char *underlyingtype ) { // return the bracketed part of a type
  char *ap = ulhack(parseArrayParts( underlyingtype ));  // return this

  // now need to remove all that from the underlyingtype to get 
  char *arraypart = strdup("");  // leak
  if (index(underlyingtype, '[')) {
    // looks like an array 
    free(arraypart);
    char *start = index(underlyingtype, '['); // wrong.  can have int *buh[32]
    arraypart = strdup( start );
    if (*(start-1) == ' ') start--;  // hack 
    *start = '\0';

    // ugly. very leaky 
    strcpy( underlyingtype, parseUnderlyingType( underlyingtype )); 
    
    // ulhack( arraypart ); 
  }
  return ap;  // leak unless caller frees this 
}



bool isRestrict( const char *sometype ) { // does not modify sometype
  string r( "__restrict__" );
  string t( sometype );
  return (std::string::npos != t.find( r ) );
}



bool streq( const char *a, const char *b) { return !strcmp(a,b); };  // slightly less ugly // TODO enums

void chillindent( int howfar, FILE *fp ) { for (int i=0; i<howfar; i++) fprintf(fp, "  ");  }



  void chillAST_node::printPreprocBEFORE( int indent, FILE *fp ) { 
    for (int i=0; i< preprocessinginfo.size(); i++) { 
      if (preprocessinginfo[i]->position == CHILL_PREPROCESSING_LINEBEFORE || 
          preprocessinginfo[i]->position == CHILL_PREPROCESSING_IMMEDIATELYBEFORE) {
        fprintf(stderr, "before %d\n", preprocessinginfo[i]->position); 
        preprocessinginfo[i]->print(indent, fp); 
      }
    }
  }

  void chillAST_node::printPreprocAFTER( int indent, FILE *fp ) { 
    for (int i=0; i< preprocessinginfo.size(); i++) { 
      if (preprocessinginfo[i]->position == CHILL_PREPROCESSING_LINEAFTER || 
          preprocessinginfo[i]->position ==  CHILL_PREPROCESSING_TOTHERIGHT) { 
        fprintf(stderr, "after %d\n", preprocessinginfo[i]->position); 
        preprocessinginfo[i]->print(indent, fp); 
      }
    }
  }


chillAST_SourceFile::chillAST_SourceFile::chillAST_SourceFile() { 
  SourceFileName = strdup("No Source File");
  asttype = CHILLAST_NODETYPE_SOURCEFILE;
  parent = NULL; // top node
  metacomment = NULL;
  global_symbol_table = NULL;
  global_typedef_table = NULL;
  FileToWrite = NULL;
  frontend = strdup("unknown"); 
  isFromSourceFile = true;
  filename = NULL; 
};

chillAST_SourceFile::chillAST_SourceFile(const char *filename ) { 
  SourceFileName = strdup(filename); 
  asttype = CHILLAST_NODETYPE_SOURCEFILE;  
  parent = NULL; // top node
  metacomment = NULL;
  global_symbol_table = NULL;
  global_typedef_table = NULL;
  FileToWrite = NULL; 
  frontend = strdup("unknown"); 
  isFromSourceFile = true;
  filename = NULL; 
};

void chillAST_SourceFile::print( int indent, FILE *fp ) { 

  fflush(fp);
  fprintf(fp, "\n// this source derived from CHILL AST originally from file '%s' as parsed by frontend compiler %s\n\n", SourceFileName, frontend); 
  std::vector< char * > includedfiles; 
  int sofar = 0; 

  //fprintf(fp, "#define __rose_lt(x,y) ((x)<(y)?(x):(y))\n#define __rose_gt(x,y) ((x)>(y)?(x):(y))\n"); // help diff figure out what's going on

  int numchildren = children.size();
  fprintf(stderr, "// sourcefile has %d children\n", numchildren);
  fprintf(stderr, "they are\n");
  for (int i=0; i<numchildren; i++) {
    fprintf(stderr, "%s  ", children[i]->getTypeString());
    if (children[i]->isFunctionDecl()) { 
      fprintf(stderr, "%s  ", ((chillAST_FunctionDecl *)children[i])->functionName);
    }
    fprintf(stderr, "\n"); 
  } 

  for (int i=0; i<numchildren; i++) {
    //fprintf(fp, "\n// child %d of type %s:\n", i, children[i]->getTypeString()); 
    if (children[i]->isFromSourceFile) { 
      if (children[i]->isMacroDefinition()) fprintf(fp, "\n"); fflush(fp);
      children[i]->print( indent, fp );
      if (children[i]->isVarDecl()) fprintf(fp, ";\n"); fflush(fp);  // top level vardecl\n"); 
    }
    else { 

      // this should all go away 

#ifdef NOPE 
      if (children[i]->filename // not null and not empty string 
          //&&  0 != strlen(children[i]->filename)
          ) { // should not be necessary 
        //fprintf(fp, "// need an include for %s\n", children[i]->filename); 
        bool rddid = false;
        sofar = includedfiles.size(); 
        
        for (int j=0; j<sofar; j++) {
          //fprintf(stderr, "comparing %s and %s\n",  includedfiles[j], children[i]->filename ); 
          if (!strcmp( includedfiles[j], children[i]->filename) ) { // this file has already been included
            rddid = true;
            //fprintf(stderr, "already did that one\n"); 
          }
        }
        
        if (false == rddid) { // we need to include it now
          fprintf(fp, "#include \"%s\"\n", children[i]->filename);
          includedfiles.push_back(strdup(  children[i]->filename ));
        }
        //else { 
        //  fprintf(fp, "already did\n"); 
        //} 
      }
#endif // NOPE 


    }
  } 

  fflush(fp); 

  //fprintf(fp, "\n\n// functions??\n"); 
  //for (int i=0; i<functions.size(); i++) { 
  //  fprintf(fp, "\n\n"); functions[i]->print(0,fp); fflush(fp); 
  //} 
};
 



void chillAST_SourceFile::printToFile( char *filename ) {
  char fn[1024];

  if (NULL == filename)  {  // build up a filename using original name and frontend if known
    if (FileToWrite) { 
      strcpy( fn, FileToWrite ); 
    }
    else { 
      // input name with name of frontend compiler prepended
      if (frontend) sprintf(fn, "%s_%s\0", frontend, SourceFileName);
      else sprintf(fn, "UNKNOWNFRONTEND_%s\0", SourceFileName); // should never happen
    }
  }
  else strcpy( fn, filename );

  FILE *fp = fopen(fn, "w");
  if (!fp) { 
    fprintf(stderr, "can't open file '%s' for writing\n", fn);
    exit(-1);
  }
  
  //fprintf(fp, "\n\n");
  //dump(0, fp); 
  fprintf(fp, "\n\n");
  print(0, fp);
  
}



void chillAST_SourceFile::dump( int indent, FILE *fp ) { 
  fflush(fp); 
  fprintf(fp, "\n//CHILL AST originally from file '%s'\n", SourceFileName); 
  int numchildren = children.size();
  for (int i=0; i<numchildren; i++) {
    children[i]->dump( indent, fp );
  }
  fflush(fp); 
};



chillAST_MacroDefinition * chillAST_SourceFile::findMacro( const char *name ) {
  //fprintf(stderr, "chillAST_SourceFile::findMacro( %s )\n", name );
  
  int numMacros = macrodefinitions.size();
  for (int i=0; i<numMacros; i++) { 
    if (!strcmp( macrodefinitions[i]->macroName, name )) return macrodefinitions[i];
  }
  return NULL; // not found
}


chillAST_FunctionDecl * chillAST_SourceFile::findFunction( const char *name ) {
  //fprintf(stderr, "chillAST_SourceFile::findMacro( %s )\n", name );
  
  int numFuncs = functions.size();
  for (int i=0; i<numFuncs; i++) { 
    if ( !strcmp( functions[i]->functionName, name )) return functions[i];
  }
  return NULL;
}


chillAST_node *chillAST_SourceFile::findCall( const char *name ) {
  chillAST_MacroDefinition *macro = findMacro( name );
  if (macro) return (chillAST_node *)macro;
  chillAST_FunctionDecl *func =findFunction( name ); 
  return func;
}


chillAST_TypedefDecl::chillAST_TypedefDecl() { 
  underlyingtype = newtype = arraypart = NULL; 
  asttype = CHILLAST_NODETYPE_TYPEDEFDECL; 
  parent = NULL;
  metacomment = NULL;
  isStruct = isUnion = false;
  structname = NULL; 
  rd = NULL; 
  isFromSourceFile = true; // default 
  filename = NULL; 
};


chillAST_TypedefDecl::chillAST_TypedefDecl(char *t, char *nt, chillAST_node *par) { 
  //fprintf(stderr, "chillAST_TypedefDecl::chillAST_TypedefDecl( underlying type %s, newtype %s )\n", t, nt); 
  underlyingtype = strdup(t); 
  newtype = strdup(nt);
  arraypart = NULL; 
  asttype = CHILLAST_NODETYPE_TYPEDEFDECL;
  parent = NULL; 
  metacomment = NULL;
  isStruct = isUnion = false;
  structname = NULL; 
  rd = NULL; 
  isFromSourceFile = true; // default 
  filename = NULL; 
};


chillAST_TypedefDecl::chillAST_TypedefDecl(char *t, char *a, char *p, chillAST_node *par) { 
  underlyingtype = strdup(t); 
  //fprintf(stderr, "chillAST_TypedefDecl::chillAST_TypedefDecl( underlying type %s )\n", underlyingtype); 
  newtype = strdup(a);  // the new named type ??

  arraypart = strdup(p);  // array (p)art? 
  // splitarraypart(); // TODO 

  asttype = CHILLAST_NODETYPE_TYPEDEFDECL; 
  parent = par;
  metacomment = NULL;
  isStruct = isUnion = false;
  structname = NULL; 
  rd = NULL; 
  isFromSourceFile = true; // default 
  filename = NULL; 
};




void chillAST_TypedefDecl::print(  int indent,  FILE *fp ) {
  //fprintf(fp, "typedefdecl->print()\n"); 

  printPreprocBEFORE(indent, fp); 

  if (isStruct) { 
    fprintf(fp, "\n/* A typedef STRUCT */\n"); chillindent(indent, fp);
  }

  chillindent(indent, fp);
  fprintf(fp, "typedef "); fflush(fp); 
  
  if (rd) { 
    rd->print(indent, fp);   // needs to not print the ending semicolon ?? 
  }  
  
  else if (isStruct) {   
    fprintf(stderr, "/* no rd */\n"); 
    
    //fprintf(fp, "struct %s\n", structname);
    chillindent(indent, fp);
    fprintf(fp, "{\n");
    for (int i=0; i<subparts.size(); i++) { 
      //fprintf(fp, "a %s\n", subparts[i]->getTypeString()); 
      subparts[i]->print(indent+1, fp);
      fprintf(fp, ";\n");
    }
    fprintf(fp, "};\n");
  }
  else { 
    fprintf(fp, "/* NOT A STRUCT */ typedef %s  %s%s;\n",  underlyingtype, newtype, arraypart ); 
    dump(); printf("\n\n"); fflush(stdout); 
  }
  
    // then the newname 
  fprintf(fp, "%s;\n", newtype); 
  fflush(fp); 
  printPreprocAFTER(indent, fp); 

  return;
}


chillAST_VarDecl *chillAST_TypedefDecl::findSubpart( const char *name ) {
  //fprintf(stderr, "chillAST_TypedefDecl::findSubpart( %s )\n", name);
  //fprintf(stderr, "typedef %s  %s\n", structname, newtype); 

  if (rd) { // we have a record decl look there
    chillAST_VarDecl *sub = rd->findSubpart( name );
    //fprintf(stderr, "rd found subpart %p\n", sub); 
    return sub; 
  }

  // can this ever happen now ??? 
  int nsub = subparts.size();
  //fprintf(stderr, "%d subparts\n", nsub); 
  for (int i=0; i<nsub; i++) { 
    if ( !strcmp( name, subparts[i]->varname )) return subparts[i];
  }
  //fprintf(stderr, "subpart not found\n"); 

  
  return NULL; 
}


chillAST_RecordDecl * chillAST_TypedefDecl::getStructDef() { 
  if (rd) return rd;
  return NULL;  
}



chillAST_RecordDecl::chillAST_RecordDecl() { 
  asttype = CHILLAST_NODETYPE_RECORDDECL;
  name = strdup("unknown"); // ?? 
  originalname = NULL;      // ?? 
  isStruct = isUnion = false;
  parent = NULL; 
  isFromSourceFile = true; // default 
  filename = NULL; 
}

chillAST_RecordDecl::chillAST_RecordDecl( const char *nam, chillAST_node *p ) { 
  //fprintf(stderr, "chillAST_RecordDecl::chillAST_RecordDecl()\n");
  asttype = CHILLAST_NODETYPE_RECORDDECL;
  parent = p; 
  if (nam) name = strdup(nam);
  else name = strdup("unknown"); // ?? 
  originalname = NULL;      // ??   // make them do it manually?
  isStruct = isUnion = false;
  isFromSourceFile = true; // default 
  filename = NULL; 
}

chillAST_RecordDecl::chillAST_RecordDecl( const char *nam, const char *orig, chillAST_node *p ) { 
  //fprintf(stderr, "chillAST_RecordDecl::chillAST_RecordDecl( %s, (orig) )\n", nam); 
  asttype = CHILLAST_NODETYPE_RECORDDECL;
  parent = p; 

  if (nam) name = strdup(nam);
  else name = strdup("unknown"); // ?? 
  
  originalname = NULL;   
  if (orig) originalname = strdup(orig);
  
  isStruct = isUnion = false;
  isFromSourceFile = true; // default 
  filename = NULL; 
}



chillAST_VarDecl * chillAST_RecordDecl::findSubpart( const char *nam ){
  //fprintf(stderr, "chillAST_RecordDecl::findSubpart( %s )\n", nam);
  int nsub = subparts.size();
  //fprintf(stderr, "%d subparts\n", nsub);
  for (int i=0; i<nsub; i++) { 
    //fprintf(stderr, "comparing to '%s' to '%s'\n", nam, subparts[i]->varname);
    if ( !strcmp( nam, subparts[i]->varname )) return subparts[i];
  }
  fprintf(stderr, "chillAST_RecordDecl::findSubpart() couldn't find member NAMED %s in ", nam); print(); printf("\n\n"); fflush(stdout); 

  return NULL;   
}


chillAST_VarDecl * chillAST_RecordDecl::findSubpartByType( const char *typ ){
  //fprintf(stderr, "chillAST_RecordDecl::findSubpart( %s )\n", nam);
  int nsub = subparts.size();
  //fprintf(stderr, "%d subparts\n", nsub);
  for (int i=0; i<nsub; i++) { 
    //fprintf(stderr, "comparing '%s' to '%s'\n", typ, subparts[i]->vartype);
    if ( !strcmp( typ, subparts[i]->vartype )) return subparts[i];
  }
  //fprintf(stderr, "chillAST_RecordDecl::findSubpart() couldn't find member of TYPE %s in ", typ); print(); printf("\n\n"); fflush(stdout); 

  return NULL;   
}


void chillAST_RecordDecl::print( int indent,  FILE *fp ) {
  //fprintf(fp, "chillAST_RecordDecl::print()\n"); 
  printPreprocBEFORE(indent, fp); 

  chillindent(indent, fp);  
  if (isStruct) { 
    //fprintf(fp, "\n/* A Record Decl STRUCT */\n"); chillindent(indent, fp);
    fprintf(fp, "struct ");
    if ( strncmp( "unnamed", name, 7) ) fprintf(fp, "%s\n", name);
    
    chillindent(indent, fp);
    fprintf(fp, "{\n");
    for (int i=0; i<subparts.size(); i++) { 
      //fprintf(fp, "a %s\n", subparts[i]->getTypeString()); 
      subparts[i]->print(indent+1, fp);
      fprintf(fp, ";\n");
    }
    fprintf(fp, "} ");
    fprintf(fp, "\n");  // TODO need semicolon when defining struct. can't have it when part of a typedef. One of the following lines is correct in each case. 
    //fprintf(fp, ";\n");
  }
  else { 
    fprintf(fp, "/* UNKNOWN RECORDDECL print() */  ");
    exit(-1);
  }
  printPreprocAFTER(indent, fp); 
  fflush(fp); 
}


chillAST_SymbolTable * chillAST_RecordDecl::addVariableToSymbolTable( chillAST_VarDecl *vd ){ 
  // for now, just bail. or do we want the struct to have an actual symbol table?
  //fprintf(stderr, "chillAST_RecordDecl::addVariableToSymbolTable() ignoring struct member %s vardecl\n", vd->varname); 
  return NULL; // damn, I hope nothing uses this! 
}

void chillAST_RecordDecl::printStructure( int indent,  FILE *fp ) {
  chillindent(indent, fp);  
  if (isStruct) { 
    fprintf(fp, "struct { ", name);
    for (int i=0; i<subparts.size(); i++) { 
      subparts[i]->print( 0, fp); // ?? TODO indent level 
      fprintf(fp, "; ");
    }
    fprintf(fp, "} ");
  }
  else { 
    fprintf(fp, "/* UNKNOWN RECORDDECL printStructure() */  ");
    exit(-1);
  }
  fflush(fp); 
}



void chillAST_RecordDecl::dump( int indent,  FILE *fp ) {
  chillindent(indent, fp);  
  
}


chillAST_FunctionDecl::chillAST_FunctionDecl() { 
  functionName = strdup("YouScrewedUp"); 
  asttype = CHILLAST_NODETYPE_FUNCTIONDECL; 
  forwarddecl = externfunc = builtin = false; 
  uniquePtr = (void *) NULL;
  this->setFunctionCPU(); 
  parent = NULL;
  metacomment = NULL;
  symbol_table = NULL;   // eventually, pointing to body's symbol table
  typedef_table = NULL;
  body = new chillAST_CompoundStmt();
  isFromSourceFile = true; // default 
  filename = NULL; 
};


chillAST_FunctionDecl::chillAST_FunctionDecl(const char *rt, const char *fname, chillAST_node *par) { 
  returnType = strdup(rt);
  functionName = strdup(fname);
  this->setFunctionCPU(); 
  //fprintf(stderr, "functionName %s\n", functionName); 
  forwarddecl = externfunc = builtin = false; 

  asttype = CHILLAST_NODETYPE_FUNCTIONDECL; 
  parent = par;
  metacomment = NULL;
  if (par) par->getSourceFile()->addFunc( this );
  symbol_table = NULL;
  typedef_table = NULL;
  body = new chillAST_CompoundStmt();
  isFromSourceFile = true; // default 
  filename = NULL; 
};


chillAST_FunctionDecl::chillAST_FunctionDecl(const char *rt, const char *fname, chillAST_node *par, void *unique) { 
  returnType = strdup(rt);
  functionName = strdup(fname);
  this->setFunctionCPU(); 
  //fprintf(stderr, "functionName %s\n", functionName); 
  forwarddecl = externfunc = builtin = false; 

  body = NULL;
  asttype = CHILLAST_NODETYPE_FUNCTIONDECL; 
  uniquePtr = unique; // a quick way to check equivalence. DO NOT ACCESS THROUGH THIS 
  parent = par;
  metacomment = NULL;
  if (par) par->getSourceFile()->addFunc( this );
  symbol_table = NULL;
  typedef_table = NULL;
  isFromSourceFile = true; // default 
  filename = NULL; 
};


void chillAST_FunctionDecl::addParameter( chillAST_VarDecl *p) {
  //fprintf(stderr, "%s chillAST_FunctionDecl::addParameter( 0x%x  param %s)   total of %d parameters\n", functionName, p, p->varname, 1+parameters.size()); 

  if (symbolTableHasVariableNamed( &parameters, p->varname)) {
    fprintf(stderr, "chillAST_FunctionDecl::addParameter( %s ), parameter already exists?\n");
    // exit(-1); // ?? 
    return; // error? 
  }

  parameters.push_back(p);
  //addSymbolToTable( parameters, p ); 
  p->isAParameter = true;  
  
  p->setParent( this ); // ??  unclear TODO 
  //p->dump(); printf("\naddparameter done\n\n"); fflush(stdout); 
}

void chillAST_FunctionDecl::addDecl( chillAST_VarDecl *vd) { // to symbol table ONLY 
  //fprintf(stderr, "chillAST_FunctionDecl::addDecl( %s )\n", vd->varname);
  if (!body) {  
    //fprintf(stderr, "had no body\n"); 
    body = new chillAST_CompoundStmt();
    body->symbol_table = symbol_table;   // probably wrong if this ever does something
  }

  //fprintf(stderr, "before body->addvar(), func symbol table had %d entries\n", symbol_table->size()); 
  //fprintf(stderr, "before body->addvar(), body symbol table was %p\n", body->symbol_table); 
  //fprintf(stderr, "before body->addvar(), body symbol table had %d entries\n", body->symbol_table->size()); 
  //adds to body symbol table, and makes sure function has a copy. probably dumb
  symbol_table = body->addVariableToSymbolTable( vd ); 
  //fprintf(stderr, "after body->addvar(), func symbol table had %d entries\n", symbol_table->size()); 
}

chillAST_VarDecl *chillAST_FunctionDecl::hasParameterNamed( const char *name ) { 
  int numparams = parameters.size();
  for (int i=0; i<numparams; i++) { 
    if (!strcmp(name, parameters[i]->varname)) return parameters[i];  // need to check type? 
  }
  return NULL; 
}


// similar to symbolTableHasVariableNamed() but returns the variable definition
chillAST_VarDecl *chillAST_FunctionDecl::hasVariableNamed( const char *name ) {
  //fprintf(stderr, "chillAST_FunctionDecl::hasVariableNamed( %s )\n", name );

  // first check the parameters
  int numparams = parameters.size();
  for (int i=0; i<numparams; i++) { 
    chillAST_VarDecl *vd = parameters[i];
    if (!strcmp(name, vd->varname)) { 
      //fprintf(stderr, "yep, it's parameter %d\n", i); 
      return vd;  // need to check type? 
    }
  }
  //fprintf(stderr, "no parameter named %s\n", name); 

  if (!symbol_table) {
    //fprintf(stderr,"and no symbol_table, so no variable named %s\n", name);
    return NULL; // no symbol table so no variable by that name 
  }

  int numvars = symbol_table->size();
  //fprintf(stderr, "checking against %d variables\n", numvars); 
  for (int i=0; i<numvars; i++) { 
    chillAST_VarDecl *vd = (*symbol_table)[i];
    if (!strcmp(name, vd->varname)) {
      //fprintf(stderr, "yep, it's variable %d\n", i); 
      return vd;  // need to check type? 
    }
  }
  //fprintf(stderr, "not a parameter or variable named %s\n", name); 
  return NULL; 
}




void chillAST_FunctionDecl::setBody( chillAST_node * bod ) {
  //fprintf(stderr, "%s chillAST_FunctionDecl::setBody( 0x%x )   total of %d children\n", functionName, bod, 1+children.size()); 
  if (bod->isCompoundStmt())   body = (chillAST_CompoundStmt *)bod;
  else { 
    body = new chillAST_CompoundStmt();
    body->addChild( bod ); 
  }
  symbol_table = body->getSymbolTable(); 
  //addChild(bod);
  bod->setParent( this );  // well, ... 
}


void chillAST_FunctionDecl::insertChild(int i, chillAST_node* node) { 
  body->insertChild( i, node ); 
}

void chillAST_FunctionDecl::addChild(chillAST_node* node) { 
  body->addChild( node ); 
  node->parent = this; // this, or body?? 
}


void  chillAST_FunctionDecl::printParameterTypes( FILE *fp ) {  // also prints names
  fprintf(fp, "( "); 
  int numparameters = parameters.size(); 
  for (int i=0; i<numparameters; i++) {
    if (i!=0) fprintf(fp, ", "); 
    chillAST_VarDecl *p = parameters[i];
    p->print(0, fp); // note: no indent, as this is in the function parens
  }
  fprintf(fp, " )"); // end of input parameters

}




void chillAST_FunctionDecl::print(  int indent,  FILE *fp ) {
  //fprintf(fp, "\n// functiondecl %p    \n", this); 
  //chillindent(indent, fp); 
  //fprintf(fp, "//(functiondecl)  %d parameters\n", numparameters);

  printPreprocBEFORE(indent, fp); 

  fprintf(fp, "\n");
  chillindent(indent, fp); 

  if (externfunc)  fprintf(fp, "extern ");

  if (function_type == CHILL_FUNCTION_GPU) fprintf(fp, "__global__ "); 
  fprintf(fp, "%s %s",  returnType, functionName );
  printParameterTypes(fp); 


  
  // non-parameter variables  (now must have explicit vardecl in the body) 
  //int numvars = symbol_table.size();
  //for (int i=0; i<numvars; i++) { 
  //  symbol_table[i]->print(1,fp);
  //  fprintf(fp, ";\n"); 
  //} 

  // now the body 
  if (!(externfunc || forwarddecl)) { 
    if (body) { 
      fprintf(fp, "\n{\n"); 
      //chillindent(indent+1, fp); fprintf(fp, "//body\n"); fflush(fp); 
      body->print( indent+1, fp);
      fprintf(fp, "\n"); 
      //chillindent(indent+1, fp); fprintf(fp, "//END body\n"); fflush(fp); 
    
      // tidy up
      chillindent(indent, fp); 
      fprintf(fp, "}\n");
    } // if body 
    else { 
      fprintf(fp, "{}\n"); // should never happen, but not external and no body 
    }
  }
  else { // extern func or forward decl.   just end forward declaration 
    fprintf(fp, "; // fwd decl\n");
  }
  
  printPreprocAFTER(indent, fp); 

  fflush(fp); 
}
 


void chillAST_FunctionDecl::dump(  int indent,  FILE *fp ) {
  fprintf(fp, "\n"); 
  fprintf(fp, "// Function isFromSourceFile ");
  if (filename) fprintf(fp, "%s  ", filename); 
  if (isFromSourceFile) fprintf(fp, "true\n"); 
  else fprintf(fp, "false\n"); 
  chillindent(indent, fp); 
  fprintf(fp, "(FunctionDecl %s %s(",  returnType, functionName );
  
  int numparameters = parameters.size(); 
  for (int i=0; i<numparameters; i++) {
    if (i!=0) fprintf(fp, ", "); 
    chillAST_VarDecl *p = parameters[i];
    //fprintf(stderr, "param type %s  vartype %s\n", p->getTypeString(), p->vartype); 
    p->print(0, fp); // note: no indent, as this is in the function parens, ALSO print, not dump
  }
  fprintf(fp, ")\n"); // end of input parameters
  fflush(fp);
  
  // now the body - 
  if (body) body->dump( indent+1 , fp); 

  // tidy up
  chillindent(indent, fp); 
  fprintf(fp, ")\n");
  fflush(fp); 
}
 





void chillAST_FunctionDecl::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_FunctionDecl::gatherVarDecls()\n"); 
  //if (0 < children.size()) fprintf(stderr, "functiondecl has %d children\n", children.size()); 
  //fprintf(stderr, "functiondecl has %d parameters\n", numParameters());
  for (int i=0; i<numParameters(); i++) parameters[i]->gatherVarDecls( decls );   
  //fprintf(stderr, "after parms, %d decls\n", decls.size()); 
  for (int i=0; i<children.size(); i++) children[i]->gatherVarDecls( decls ); 
  //fprintf(stderr, "after children, %d decls\n", decls.size()); 
  body->gatherVarDecls( decls );  // todo, figure out if functiondecl has actual children
  //fprintf(stderr, "after body, %d decls\n", decls.size()); 
  //for (int d=0; d<decls.size(); d++) {
  //  decls[d]->print(0,stderr); fprintf(stderr, "\n"); 
  //} 
}


void chillAST_FunctionDecl::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //if (0 < children.size()) fprintf(stderr, "functiondecl has %d children\n", children.size()); 
  
  for (int i=0; i<numParameters(); i++) parameters[i]->gatherScalarVarDecls( decls );   
  for (int i=0; i<children.size(); i++) children[i]->gatherScalarVarDecls( decls ); 
  body->gatherScalarVarDecls( decls );  // todo, figure out if functiondecl has actual children
}


void chillAST_FunctionDecl::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //if (0 < children.size()) fprintf(stderr, "functiondecl has %d children\n", children.size()); 
  
  for (int i=0; i<numParameters(); i++) parameters[i]->gatherArrayVarDecls( decls );   
  for (int i=0; i<children.size(); i++) children[i]->gatherArrayVarDecls( decls ); 
  body->gatherArrayVarDecls( decls );  // todo, figure out if functiondecl has actual children
}


chillAST_VarDecl *chillAST_FunctionDecl::findArrayDecl( const char *name ) { 
  //fprintf(stderr, "chillAST_FunctionDecl::findArrayDecl( %s )\n", name );
  chillAST_VarDecl *p = hasParameterNamed( name ); 
  //if (p) fprintf(stderr, "function %s has parameter named %s\n", functionName, name );
  if (p && p->isArray()) return p;

  chillAST_VarDecl *v = hasVariableNamed ( name ); 
  //if (v) fprintf(stderr, "function %s has symbol table variable named %s\n", functionName, name );
  if (v && v->isArray()) return v;

  // declared variables that may not be in symbol table but probably should be
  vector<chillAST_VarDecl*> decls ;
  gatherArrayVarDecls( decls );
  for (int i=0; i<decls.size(); i++) { 
    chillAST_VarDecl *vd = decls[i]; 
    if (0 == strcmp(vd->varname, name ) && vd->isArray()) return vd;
  }

  //fprintf(stderr, "can't find array named %s in function %s \n", name, functionName); 
  return NULL; 
}


void chillAST_FunctionDecl::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<children.size(); i++) children[i]->gatherVarUsage( decls ); 
  body->gatherVarUsage( decls );  // todo, figure out if functiondecl has actual children
}


void chillAST_FunctionDecl::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  for (int i=0; i<children.size(); i++) children[i]->gatherDeclRefExprs( refs ); 
  body->gatherDeclRefExprs( refs );  // todo, figure out if functiondecl has actual children
}



void chillAST_FunctionDecl::cleanUpVarDecls() {  
  //fprintf(stderr, "\ncleanUpVarDecls() for function %s\n", functionName); 
  vector<chillAST_VarDecl*> used;
  vector<chillAST_VarDecl*> defined;
  vector<chillAST_VarDecl*> deletethese;

  gatherVarUsage( used ); 
  gatherVarDecls( defined ); 

  //fprintf(stderr, "\nvars used: \n"); 
  //for ( int i=0; i< used.size(); i++) { 
  //used[i]->print(0, stderr);  fprintf(stderr, "\n"); 
  //} 
  //fprintf(stderr, "\n"); 
  //fprintf(stderr, "\nvars defined: \n"); 
  //for ( int i=0; i< defined.size(); i++) { 
  //  defined[i]->print(0, stderr);  fprintf(stderr, "\n"); 
  //} 
  //fprintf(stderr, "\n"); 

  for ( int j=0; j < defined.size(); j++) { 
    //fprintf(stderr, "j %d  defined %s\n", j, defined[j]->varname); 
    bool definedandused = false;
    for ( int i=0; i < used.size(); i++) {
      if (used[i] == defined[j]) { 
        //fprintf(stderr, "i %d used %s\n", i, used[i]->varname); 
        //fprintf(stderr, "\n");
        definedandused = true;
        break;
      }
    }

    if (!definedandused) { 
      if ( defined[j]->isParmVarDecl() ) { 
        //fprintf(stderr, "we'd remove %s except that it's a parameter. Maybe someday\n", defined[j]->varname); 
      }
      else { 
        //fprintf(stderr, "we can probably remove the definition of %s\n", defined[j]->varname); 
        deletethese.push_back(  defined[j] ); 
      }
    }
  }


  //fprintf(stderr, "deleting %d vardecls\n", deletethese.size()); 
  for (int i=0; i<deletethese.size(); i++) { 
    //fprintf(stderr, "deleting varDecl %s\n",  deletethese[i]->varname); 
    chillAST_node *par =  deletethese[i]->parent; 
    par->removeChild( par->findChild( deletethese[i] )); 
  }


  //fprintf(stderr, "\n\nnow check for vars used but not defined\n"); 
  // now check for vars used but not defined?
  for ( int j=0; j < used.size(); j++) { 
    //fprintf(stderr, "%s is used\n", used[j]->varname); 
    bool definedandused = false;
    for ( int i=0; i < defined.size(); i++) {
      if (used[j] == defined[i]) { 
        //fprintf(stderr, "%s is defined\n", defined[i]->varname); 
        definedandused = true;
        break;
      }
    }
    if (!definedandused) { 
      //fprintf(stderr, "%s is used but not defined?\n", used[j]->varname); 
      // add it to the beginning of the function
      insertChild(0, used[j]); 
    }
  }
  
}

//void chillAST_FunctionDecl::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl ) {
//  for (int i=0; i<children.size(); i++) children[i]->replaceVarDecls( olddecl, newdecl ); 
//} 


bool chillAST_FunctionDecl::findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync ) { 
  if (body) body->findLoopIndexesToReplace( symtab, false ); 
  return false;
}



 chillAST_node *chillAST_FunctionDecl::constantFold() { 
   //fprintf(stderr, "chillAST_FunctionDecl::constantFold()\n");
   // parameters can't have constants?
   int numparameters = parameters.size(); 
   for (int i=0; i<numparameters; i++) {
     parameters[i]->constantFold();
   }
   if (body) body = (chillAST_CompoundStmt *)body->constantFold(); 
   return this;
 }


chillAST_MacroDefinition::chillAST_MacroDefinition() { 
  macroName = strdup("UNDEFINEDMACRO");
  asttype = CHILLAST_NODETYPE_MACRODEFINITION; 
  parent  = NULL;
  metacomment = NULL;
  symbol_table = NULL;
  //rhsideString = NULL;
  isFromSourceFile = true; // default 
  filename = NULL; 
};


chillAST_MacroDefinition::chillAST_MacroDefinition(const char *mname, chillAST_node *par) { 
  macroName = strdup(mname);
  asttype = CHILLAST_NODETYPE_MACRODEFINITION; 
  parent = par;
  metacomment = NULL;
  symbol_table = NULL;
  //rhsideString = NULL;

  if (par) par->getSourceFile()->addMacro( this );

  //fprintf(stderr, "chillAST_MacroDefinition::chillAST_MacroDefinition( %s, ", mname); 
  //if (par) fprintf(stderr, " parent NOT NULL);\n");
  //else fprintf(stderr, " parent NULL);\n");
  isFromSourceFile = true; // default 
  filename = NULL; 
};


chillAST_node* chillAST_MacroDefinition::clone() {

  // TODO ?? cloning a macro makes no sense
  return this;
#ifdef CONFUSED 

  //fprintf(stderr, "chillAST_MacroDefinition::clone() for %s\n", macroName); 
  chillAST_MacroDefinition *clo = new chillAST_MacroDefinition( macroName, parent); 
  for (int i=0; i<parameters.size(); i++) clo->addParameter( parameters[i] );
  clo->setBody( body->clone() );
  return clo; 
#endif 

}


void chillAST_MacroDefinition::setBody( chillAST_node * bod ) {
  //fprintf(stderr, "%s chillAST_MacroDefinition::setBody( 0x%x )   total of %d children\n", functionName, bod, 1+children.size()); 
  body = bod;
  bod->setParent( this );  // well, ... 
}


void chillAST_MacroDefinition::addParameter( chillAST_VarDecl *p) {
  //fprintf(stderr, "%s chillAST_MacroDefinition::addParameter( 0x%x )   total of %d children\n", functionName, p, 1+children.size()); 
  parameters.push_back(p);
  p->isAParameter = true; 
  p->setParent( this );

  addVariableToSymbolTable( p );
}


chillAST_VarDecl *chillAST_MacroDefinition::hasParameterNamed( const char *name ) { 
  int numparams = parameters.size();
  for (int i=0; i<numparams; i++) { 
    if (!strcmp(name, parameters[i]->varname)) return parameters[i];  // need to check type? 
  }
  return NULL; 
}


void chillAST_MacroDefinition::insertChild(int i, chillAST_node* node) { 
  body->insertChild( i, node ); 
}

void chillAST_MacroDefinition::addChild(chillAST_node* node) { 
  body->addChild( node ); 
  node->parent = this; // this, or body?? 
}


void chillAST_MacroDefinition::dump(  int indent,  FILE *fp ) {
  fprintf(fp, "\n"); 
  chillindent(indent, fp); 
  fprintf(fp, "(MacroDefinition %s(", macroName);
  for (int i=0; i<numParameters(); i++) { 
    fprintf(fp, "\n");
    chillindent(indent+1, fp);
    fprintf(fp, "(%s)", parameters[i]->varname); 
  }
  fprintf(fp, ")\n"); 
  body->dump( indent+1, fp);
  fprintf(fp, "\n"); 
  fflush(fp);
}


void chillAST_MacroDefinition::print(  int indent,  FILE *fp ) {  // UHOH   TODO 
  //fprintf(fp, "\n"); // ignore indentation
  //fprintf(stderr, "macro has %d parameters\n", numParameters()); 

  printPreprocBEFORE(indent, fp); 

  fprintf(fp, "#define %s", macroName);
  if (0 != numParameters()) { 
    fprintf(fp, "(");
    for (int i=0; i<numParameters(); i++) { 
      if (i) fprintf(fp, ",");
      fprintf(fp, "%s", parameters[i]->varname); 
    }
    fprintf(fp, ")  ");
  }

  if (body) body->print(0, fp); // TODO should force newlines out of multiline macros 
  fprintf(fp, "\n"); 
} 




chillAST_ForStmt::chillAST_ForStmt() {
  init = cond = incr = body = NULL;
  asttype = CHILLAST_NODETYPE_LOOP;  // breaking with tradition, this was CHILL_AST_FORSTMT
  conditionoperator = IR_COND_UNKNOWN;
  parent = NULL;
  metacomment = NULL;
  symbol_table = NULL;
  isFromSourceFile = true; // default 
  filename = NULL; 
}


chillAST_ForStmt::chillAST_ForStmt(  chillAST_node *ini, chillAST_node *con, chillAST_node *inc, chillAST_node *bod, chillAST_node *par) {
  parent = par; 
  metacomment = NULL;
  init = ini;
  cond = con;
  incr = inc;
  body = bod;
  init->setParent( this ); 
  cond->setParent( this ); 
  incr->setParent( this ); 
  
  //fprintf(stderr, "chillAST_ForStmt::chillAST_ForStmt() bod %p\n", bod); 

  if (body) body->setParent( this );  // not sure this should be legal 
  
  asttype = CHILLAST_NODETYPE_LOOP;  // breaking with tradition, this was CHILL_AST_FORSTMT  

  if (!cond->isBinaryOperator()) { 
    fprintf(stderr, "ForStmt conditional is of type %s. Expecting a BinaryOperator\n", cond->getTypeString());
    exit(-1); 
  }
  chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)cond;
  char *condstring = bo->op; 
  if (!strcmp(condstring, "<"))       conditionoperator = IR_COND_LT;
  else if (!strcmp(condstring, "<=")) conditionoperator = IR_COND_LE;
  else if (!strcmp(condstring, ">"))  conditionoperator = IR_COND_GT;
  else if (!strcmp(condstring, ">=")) conditionoperator = IR_COND_GE;
  else { 
    fprintf(stderr, "ForStmt, illegal/unhandled end condition \"%s\"\n", condstring);
    fprintf(stderr, "currently can only handle <, >, <=, >=\n");
    exit(1);
  }
  isFromSourceFile = true; // default 
  filename = NULL; 
}


bool chillAST_ForStmt::lowerBound( int &l ) { // l is an output (passed as reference)
  
  // above, cond must be a binaryoperator ... ??? 
  if (conditionoperator == IR_COND_LT || 
      conditionoperator == IR_COND_LE ) { 
    
    // lower bound is rhs of init 
    if (!init->isBinaryOperator()) { 
      fprintf(stderr, "chillAST_ForStmt::lowerBound() init is not a chillAST_BinaryOperator\n");
      exit(-1);
    }
    
    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)init;
    if (!init->isAssignmentOp()) { 
      fprintf(stderr, "chillAST_ForStmt::lowerBound() init is not an assignment chillAST_BinaryOperator\n");
      exit(-1);
    }
    
    //fprintf(stderr, "rhs "); bo->rhs->print(0,stderr);  fprintf(stderr, "   "); 
    l = bo->rhs->evalAsInt(); // float could be legal I suppose
    //fprintf(stderr, "   %d\n", l); 
    return true; 
  }
  else if (conditionoperator == IR_COND_GT || 
           conditionoperator == IR_COND_GE ) {  // decrementing 
    // lower bound is rhs of cond (not init)
    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)cond;
    l = bo->rhs->evalAsInt(); // float could be legal I suppose
    return true; 
  }
  
  // some case we don't handle ?? 
  fprintf(stderr, "chillAST_ForStmt::lowerBound() can't find lower bound of "); 
  print(0,stderr); 
  fprintf(stderr, "\n\n"); 
  return false;      // or exit ???
}


bool chillAST_ForStmt::upperBound( int &u ) { // u is an output (passed as reference)
  
  // above, cond must be a binaryoperator ... ??? 
  if (conditionoperator == IR_COND_GT || 
      conditionoperator == IR_COND_GE ) {  // decrementing 

    // upper bound is rhs of init 
    if (!init->isBinaryOperator()) { 
      fprintf(stderr, "chillAST_ForStmt::upperBound() init is not a chillAST_BinaryOperator\n");
      exit(-1);
    }

    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)init;
    if (!init->isAssignmentOp()) { 
      fprintf(stderr, "chillAST_ForStmt::upperBound() init is not an assignment chillAST_BinaryOperator\n");
      exit(-1);
    }

    u = bo->rhs->evalAsInt(); // float could be legal I suppose
    return true; 
  }
  else if (conditionoperator == IR_COND_LT || 
           conditionoperator == IR_COND_LE ) { 
    //fprintf(stderr, "upper bound is rhs of cond   ");
    // upper bound is rhs of cond (not init)
    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)cond;
    //bo->rhs->print(0,stderr);
    u = bo->rhs->evalAsInt(); // float could be legal I suppose

    if (conditionoperator == IR_COND_LT) u -= 1;  

    //fprintf(stderr, "    %d\n", u);
    return true; 
  }

  // some case we don't handle ?? 
  fprintf(stderr, "chillAST_ForStmt::upperBound() can't find upper bound of "); 
  print(0,stderr); 
  fprintf(stderr, "\n\n"); 
  return false;      // or exit ???
}




void chillAST_ForStmt::printControl( int in, FILE *fp ) {
  chillindent(in, fp);
  fprintf(fp, "for (");
  init->print(0, fp);
  fprintf(fp, "; ");
  cond->print(0, fp);
  fprintf(fp, "; ");
  incr->print(0, fp);
  fprintf(fp, ")");
  fflush(fp); 
}


void chillAST_ForStmt::print( int indent, FILE *fp ) {
  printPreprocBEFORE(indent, fp); 
  //fprintf(fp, "chillAST_ForStmt::print()\n"); 
  if (metacomment) { 
    chillindent(indent, fp);
    //for(int i=0; i<indent; i++) fprintf(fp, ".."); 
    fprintf(fp, "// %s\n", metacomment);
  }

  printControl(indent, fp);  // does not do a newline or bracket 
  fprintf(fp, " {\n"); 


  // I have no idea what made me do this next bit. 
  // A forstmt with compounds inside compounds ??? 
  // this should probably all go away 

  chillAST_node *b = body;
  //fprintf(fp, "b children %d\n", b->getNumChildren()); 
  //fprintf(fp, "body child 0 of type %s\n", b->children[0]->getTypeString()); 
  //fprintf(stderr, "forstmt body type %s\n", Chill_AST_Node_Names[b->asttype] ); 
  // deal with a tree of compound statements, in an ugly way. leave the ugliness
  while (1 == b->getNumChildren() && b->children[0]->isCompoundStmt()) { 
    b = b->children[0]; 
  }


  // this was to sometimes not enclose in a bracket. stupid. always enclose in a bracket.
  //if (1 == b->getNumChildren() && b->children[0]->isForStmt()) fprintf(fp, ") {\n" );
  //else if (1 == b->getNumChildren() ) fprintf(fp, ") { ?? \n" ); // to allow for() for( ) to not have open bracket?
  //else { 
    //fprintf(fp, ")\n");
    //chillindent(in, fp);
    //fprintf(fp, "{\n" );

    //fprintf(fp, ")");
  //}

  b->print(indent+1, fp );

  // I think this can't happen any more. body is always a compound statement 
  if (b->asttype ==  CHILLAST_NODETYPE_BINARYOPERATOR) { // a single assignment statement
    fprintf(fp, ";\n"); 
  }

  // always print brackets 

  //if ((1 == b->getNumChildren() && b->children[0]->isForStmt()) || 
  //    (1 != b->getNumChildren() )) {
  chillindent(indent, fp);
  fprintf(fp, "}\n" );  
  //}

  printPreprocAFTER(indent, fp); 
  fflush(fp); //
}

void chillAST_ForStmt::dump( int indent, FILE *fp ) {
  chillindent(indent, fp);
  fprintf(fp, "(ForStmt \n");

  init->dump(indent+1, fp);
  cond->dump(indent+1, fp);
  incr->dump(indent+1, fp);
  body->dump(indent+1, fp);

  chillindent(indent, fp);
  fprintf(fp, ")\n");
}
 
chillAST_node *chillAST_ForStmt::constantFold() { 
   init = init->constantFold(); 
   cond = cond->constantFold(); 
   incr = incr->constantFold(); 
   body = body->constantFold(); 
   return this; 
 }


 chillAST_node *chillAST_ForStmt::clone() { 
  chillAST_ForStmt *fs = new chillAST_ForStmt( init->clone(), cond->clone(), incr->clone(), body->clone(), parent); 
  fs->isFromSourceFile = isFromSourceFile;
  if (filename) fs->filename = strdup(filename);
  return fs;
 }

void chillAST_ForStmt::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_ForStmt::gatherVarDecls()\n"); 
  //fprintf(stderr, "chillAST_ForStmt::gatherVarDecls()  before %d\n", decls.size());
  // TODO clear a loop_var_decls variable and then walk it ? 
  init->gatherVarDecls( decls ); 
  cond->gatherVarDecls( decls ); 
  incr->gatherVarDecls( decls ); 
  body->gatherVarDecls( decls ); 
  //fprintf(stderr, "after %d\n", decls.size()); 
}

void chillAST_ForStmt::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_ForStmt::gatherScalarVarDecls()  before %d\n", decls.size());
  init->gatherScalarVarDecls( decls ); 
  cond->gatherScalarVarDecls( decls ); 
  incr->gatherScalarVarDecls( decls ); 
  body->gatherScalarVarDecls( decls ); 
}

void chillAST_ForStmt::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_ForStmt::gatherArrayVarDecls()  before %d\n", decls.size());
  init->gatherArrayVarDecls( decls ); 
  cond->gatherArrayVarDecls( decls ); 
  incr->gatherArrayVarDecls( decls ); 
  body->gatherArrayVarDecls( decls ); 
}

void chillAST_ForStmt::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) { 
  init->gatherArrayRefs( refs, 0 );  // 0 ??
  cond->gatherArrayRefs( refs, 0 );  // 0 ??
  incr->gatherArrayRefs( refs, 0 );  // 0 ??
  body->gatherArrayRefs( refs, 0 );  // 0 ??
}

void chillAST_ForStmt::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  init->gatherScalarRefs( refs, 0 );  // 0 ??
  cond->gatherScalarRefs( refs, 0 );  // 0 ??
  incr->gatherScalarRefs( refs, 0 );  // 0 ??
  body->gatherScalarRefs( refs, 0 );  // 0 ??
} 

void chillAST_ForStmt::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  init->gatherDeclRefExprs( refs ); 
  cond->gatherDeclRefExprs( refs ); 
  incr->gatherDeclRefExprs( refs ); 
  body->gatherDeclRefExprs( refs ); 
}



void chillAST_ForStmt::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  init->gatherVarUsage( decls ); 
  cond->gatherVarUsage( decls ); 
  incr->gatherVarUsage( decls ); 
  body->gatherVarUsage( decls ); 
}

void chillAST_ForStmt::gatherStatements(std::vector<chillAST_node*> &statements ){
  
  // for completeness, should do all 4. Maybe someday
  //init->gatherStatements( statements ); 
  //cond->gatherStatements( statements ); 
  //incr->gatherStatements( statements ); 
  body->gatherStatements( statements ); 
}



void chillAST_ForStmt::addSyncs() {
  //fprintf(stderr, "\nchillAST_ForStmt::addSyncs()  "); 
  //fprintf(stderr, "for (");
  //init->print(0, stderr);
  //fprintf(stderr, "; ");
  //cond->print(0, stderr);
  //fprintf(stderr, "; ");
  //incr->print(0, stderr);
  //fprintf(stderr, ")\n"); 
  
  if (!parent) { 
    fprintf(stderr, "uhoh, chillAST_ForStmt::addSyncs() ForStmt has no parent!\n");
    fprintf(stderr, "for (");
    init->print(0, stderr);
    fprintf(stderr, "; ");
    cond->print(0, stderr);
    fprintf(stderr, "; ");
    incr->print(0, stderr);
    fprintf(stderr, ")\n"); 

    return; // exit? 
  }

  if (parent->isCompoundStmt()) { 
    //fprintf(stderr, "ForStmt parent is CompoundStmt 0x%x\n", parent);
    vector<chillAST_node*> chillin = parent->getChildren();
    int numc = chillin.size();
    //fprintf(stderr, "ForStmt parent is CompoundStmt 0x%x with %d children\n", parent, numc);
    for (int i=0; i<numc; i++) { 
      if (this == parent->getChild(i)) { 
        //fprintf(stderr, "forstmt 0x%x is child %d of %d\n", this, i, numc); 
        chillAST_CudaSyncthreads *ST = new chillAST_CudaSyncthreads();
        parent->insertChild(i+1, ST);  // corrupts something ... 
        //fprintf(stderr, "Create a call to __syncthreads() 2\n"); 
        //parent->addChild(ST);  // wrong, but safer   still kills 
      }
    }

    chillin = parent->getChildren();
    int nowc = chillin.size();
    //fprintf(stderr, "old, new number of children = %d %d\n", numc, nowc); 
    
  }
  else { 
    fprintf(stderr, "chillAST_ForStmt::addSyncs() unhandled parent type %s\n", parent->getTypeString()); 
    exit(-1); 
  }

  //fprintf(stderr, "leaving addSyncs()\n"); 
}




void chillAST_ForStmt::removeSyncComment() { 
  //fprintf(stderr, "chillAST_ForStmt::removeSyncComment()\n"); 
  if (metacomment && strstr(metacomment, "~cuda~") && strstr(metacomment, "preferredIdx: ")) { 
    char *ptr =  strlen( "preferredIdx: " ) + strstr(metacomment, "preferredIdx: ");
    *ptr = '\0'; 
  }
}


bool chillAST_ForStmt::findLoopIndexesToReplace(chillAST_SymbolTable *symtab, bool forcesync ) { 
  fprintf(stderr, "\nchillAST_ForStmt::findLoopIndexesToReplace( force = %d )\n", forcesync); 
  //if (metacomment) fprintf(stderr, "metacomment '%s'\n", metacomment); 

  bool force = forcesync;
  bool didasync = false;
  if (forcesync) { 
    //fprintf(stderr, "calling addSyncs() because PREVIOUS ForStmt in a block had preferredIdx\n"); 
    addSyncs();
    didasync = true; 
  }
  
  //fprintf(stderr, "chillAST_ForStmt::findLoopIndexesToReplace()\n"); 
  if (metacomment && strstr(metacomment, "~cuda~") && strstr(metacomment, "preferredIdx: ")) { 
    //fprintf(stderr, "metacomment '%s'\n", metacomment); 
    
    char *copy = strdup(metacomment); 
    char *ptr  = strstr(copy, "preferredIdx: ");
    char *vname = ptr + strlen( "preferredIdx: " );
    char *space = strstr(vname, " "); // TODO index()
    if (space) { 
      //fprintf(stderr, "vname = '%s'\n", vname); 
      force = true; 
    }

    if ((!didasync) && force ) { 
      //fprintf(stderr, "calling addSyncs() because ForStmt metacomment had preferredIdx '%s'\n", vname); 
      addSyncs();
      removeSyncComment(); 
      didasync = true; 
    }

    if (space)   *space = '\0'; // if this is multiple words, grab the first one
    //fprintf(stderr, "vname = '%s'\n", vname); 
    
    //fprintf(stderr, "\nfor (");
    //init->print(0, stderr);
    //fprintf(stderr, "; ");
    //cond->print(0, stderr);
    //fprintf(stderr, "; ");
    //incr->print(0, stderr);
    //fprintf(stderr, ")    %s\n", metacomment );
    //fprintf(stderr, "prefer '%s'\n", vname );
    
    vector<chillAST_VarDecl*> decls;
    init->gatherVarLHSUsage( decls ); 
    //cond->gatherVarUsage( decls ); 
    //incr->gatherVarUsage( decls ); 
    //fprintf(stderr, "forstmt has %d vardecls in init, cond, inc\n", decls.size()); 

    if ( 1 != decls.size()) { 
      fprintf(stderr, "uhoh, preferred index in for statement, but multiple variables used\n");
      print(0,stderr);
      fprintf(stderr, "\nvariables are:\n"); 
      for (int i=0; i<decls.size(); i++) { 
        decls[i]->print(0,stderr); fprintf(stderr, "\n"); 
      }
      exit(0); 
    }
    chillAST_VarDecl* olddecl = decls[0];

    // RIGHT NOW, change all the references that this loop wants swapped out 
    // find vardecl for named preferred index.  it has to already exist
    fprintf(stderr, "RIGHT NOW, change all the references that this loop wants swapped out \n"); 
    int numsym = symtab->size(); 
    chillAST_VarDecl *newguy = NULL;
    fprintf(stderr, "%d symbols\n", numsym);
    for (int i=0; i<numsym; i++) { 
      fprintf(stderr, "sym %d is '%s'\n", i, (*symtab)[i]->varname);
      if (!strcmp(vname,  (*symtab)[i]->varname)) { 
        newguy = (*symtab)[i];
      }
    }
    if (!newguy) { 
      fprintf(stderr, "there is no defined variable %s\n", vname); 

      // make one ??  seems like this should never happen 
      newguy = new chillAST_VarDecl( olddecl->vartype, vname, ""/*?*/, NULL );
      // insert actual declaration in code location?   how?
    }


    // swap out old for new in init, cond, incr, body 
    if (newguy) { 
      fprintf(stderr, "\nwill replace %s with %s in init, cond, incr\n", olddecl->varname, newguy->varname); 
      fprintf(stderr, "was: for (");
      init->print(0, stderr);
      fprintf(stderr, "; ");
      cond->print(0, stderr);
      fprintf(stderr, "; ");
      incr->print(0, stderr);
      fprintf(stderr, ")\n");
      
      
      init->replaceVarDecls( olddecl, newguy ); 
      cond->replaceVarDecls( olddecl, newguy ); 
      incr->replaceVarDecls( olddecl, newguy ); 

      fprintf(stderr, " is: for (");
      init->print(0, stderr);
      fprintf(stderr, "; ");
      cond->print(0, stderr);
      fprintf(stderr, "; ");
      incr->print(0, stderr);
      fprintf(stderr, ")\n\n");

      fprintf(stderr,"recursing to body of type %s\n", body->getTypeString()); 
      body->replaceVarDecls( olddecl, newguy ); 

      fprintf(stderr, "\nafter recursing to body, this loop is   there should be no %s\n", olddecl->varname);
      print(0, stderr); fprintf(stderr, "\n"); 
      
    }
    
    //if (!space) // there was only one preferred
    //fprintf(stderr, "removing metacomment\n"); 
    metacomment = NULL; // memleak

  }

  // check for more loops.  We may have already swapped variables out in body (right above here)
  body->findLoopIndexesToReplace( symtab, false ) ; 

  return force; 
}

void chillAST_ForStmt::replaceChild( chillAST_node *old, chillAST_node *newchild ){
  //fprintf(stderr, "chillAST_ForStmt::replaceChild()  REALLY CALLING BODY->ReplaceCHILD\n"); 
  body->replaceChild( old, newchild );
}



void chillAST_ForStmt::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  // logic problem  if my loop var is olddecl! 

  //fprintf(stderr, "chillAST_ForStmt::replaceVarDecls( old %s,  new %s )\n", olddecl->varname, newdecl->varname);

  // this is called for inner loops!
  init->replaceVarDecls( olddecl, newdecl ); 
  cond->replaceVarDecls( olddecl, newdecl ); 
  incr->replaceVarDecls( olddecl, newdecl ); 
  body->replaceVarDecls( olddecl, newdecl ); 
}


void chillAST_ForStmt::gatherLoopIndeces( std::vector<chillAST_VarDecl*> &indeces ) { 
  //fprintf(stderr, "chillAST_ForStmt::gatherLoopIndeces()\nloop is:\n"); print(0,stderr); 

  vector<chillAST_VarDecl*> decls;
  init->gatherVarLHSUsage( decls ); 
  cond->gatherVarLHSUsage( decls ); 
  incr->gatherVarLHSUsage( decls ); 
  // note: NOT GOING INTO BODY OF THE LOOP

  int numdecls = decls.size(); 
  //fprintf(stderr, "gatherLoopIndeces(), %d lhs vardecls for this ForStmt\n", numdecls);

  for (int i=0; i<decls.size(); i++)  {
    //fprintf(stderr, "%s %p\n", decls[i]->varname, decls[i] );
    indeces.push_back( decls[i] ); 
  }
  
  // Don't forget to keep heading upwards!
  if (parent) {
    //fprintf(stderr, "loop %p has parent of type %s\n", this, parent->getTypeString()); 
    parent->gatherLoopIndeces( indeces );
  }
  //else fprintf(stderr, "this loop has no parent???\n");

}


void chillAST_ForStmt::gatherLoopVars(  std::vector<std::string> &loopvars ) {
  //fprintf(stderr, "gathering loop vars for loop   for (");
  //init->print(0, stderr);
  //fprintf(stderr, "; ");
  //cond->print(0, stderr);
  //fprintf(stderr, "; ");
  //incr->print(0, stderr);
  //fprintf(stderr, ")\n" );

  //init->dump(0, stderr); 


  vector<chillAST_VarDecl*> decls;
  init->gatherVarLHSUsage( decls ); 
  cond->gatherVarLHSUsage( decls ); 
  incr->gatherVarLHSUsage( decls ); 
  // note: NOT GOING INTO BODY OF THE LOOP
  
  for (int i=0; i<decls.size(); i++)  loopvars.push_back( strdup( decls[i]->varname )); 

}


void chillAST_ForStmt::loseLoopWithLoopVar( char *var ) { 

  //fprintf(stderr, "\nchillAST_ForStmt::loseLoopWithLoopVar(  %s )\n", var ); 

  // now recurse (could do first, I suppose) 
  // if you DON'T do this first, you may have already replaced yourself with this loop body
  // the body will no longer have this forstmt as parent, it will have the forstmt's parent as its parent
  //fprintf(stderr, "forstmt 0x%x, recursing loseLoop to body 0x%x of type %s with parent 0x%x of type %s\n", this, body,  body->getTypeString(), body->parent, body->parent->getTypeString()); 
  body->loseLoopWithLoopVar( var ) ; 




  // if *I* am a loop to be replaced, tell my parent to replace me with my loop body

  std::vector<std::string> loopvars;
  gatherLoopVars( loopvars );
  
  if (loopvars.size() != 1) { 
    fprintf(stderr, "uhoh, loop has more than a single loop var and trying to loseLoopWithLoopVar()\n");
    print(0,stderr);
    fprintf(stderr, "\nvariables are:\n"); 
    for (int i=0; i<loopvars.size(); i++) { 
      fprintf(stderr, "%s\n", loopvars[i].c_str()); 
    }
    
    exit(-1); 
  }
  
  //fprintf(stderr, "my loop var %s, looking for %s\n", loopvars[0].c_str(), var );
  if (!strcmp(var,  loopvars[0].c_str())) { 
    //fprintf(stderr, "OK, trying to lose myself!    for (");
    //init->print(0, stderr);
    //fprintf(stderr, "; ");
    //cond->print(0, stderr);
    //fprintf(stderr, "; ");
    //incr->print(0, stderr);
    //fprintf(stderr, ")\n" );   

    if (!parent) { 
      fprintf(stderr, "chillAST_ForStmt::loseLoopWithLoopVar()  I have no parent!\n");
      exit(-1);
    }

    vector<chillAST_VarDecl*> decls;
    init->gatherVarLHSUsage( decls );   // this can fail if init is outside the loop
    cond->gatherVarLHSUsage( decls ); 
    incr->gatherVarLHSUsage( decls ); 
    if (decls.size() > 1) { 
      fprintf(stderr, "chill_ast.cc multiple loop variables confuses me\n");
      exit(-1); 
    }
    chillAST_node *newstmt = body; 

    // ACTUALLY, if I am being replaced, and my loop conditional is a min (Ternary), then wrap my loop body in an if statement
    if (cond->isBinaryOperator()) { // what else could it be?
      chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *) cond;
      if (BO->rhs->isTernaryOperator()) { 

        chillAST_TernaryOperator *TO = (chillAST_TernaryOperator *)BO->rhs;
        chillAST_BinaryOperator *C =  (chillAST_BinaryOperator *)TO->condition;
        
        //fprintf(stderr, "loop condition RHS  is ternary\nCondition RHS");
        C->print(); printf("\n"); fflush(stdout); 
        chillAST_node *l = C->lhs;
        if (l->isParenExpr()) l = ((chillAST_ParenExpr *)l)->subexpr; 
        chillAST_node *r = C->rhs;
        if (r->isParenExpr()) r = ((chillAST_ParenExpr *)r)->subexpr; 

        //fprintf(stderr, "lhs is %s     rhs is %s\n", l->getTypeString(), r->getTypeString()); 
        
        chillAST_node *ifcondrhs = NULL; 
        if (!(l->isConstant())) ifcondrhs = l;
        else if (!(r->isConstant())) ifcondrhs = r;
        else { 
          // should never happen. 2 constants. infinite loop
          fprintf(stderr, "chill_ast.cc INIFNITE LOOP?\n"); 
          this->print(0,stderr); fprintf(stderr, "\n\n");
          exit(-1);
        }
        
        // wrap the loop body in an if
        chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( decls[0] ); 
        chillAST_BinaryOperator *ifcond = new chillAST_BinaryOperator( DRE, "<=", ifcondrhs ); 
        chillAST_IfStmt *ifstmt = new chillAST_IfStmt( ifcond, body, NULL, NULL ); 
        
        newstmt = ifstmt; 
      }
    }

    //fprintf(stderr, "forstmt 0x%x has parent 0x%x  of type %s\n", this, parent, parent->getTypeString()); 
    //fprintf(stderr, "forstmt will be replaced by\n");
    //newstmt->print(0,stderr); fprintf(stderr, "\n\n"); 

    parent->replaceChild( this, newstmt );
  }


}





chillAST_BinaryOperator::chillAST_BinaryOperator() {
  lhs = rhs = NULL;
  op = NULL;
  asttype = CHILLAST_NODETYPE_BINARYOPERATOR; 
  isFromSourceFile = true; // default 
  filename = NULL;
}


chillAST_BinaryOperator::chillAST_BinaryOperator(chillAST_node *l, const char *oper, chillAST_node *r, chillAST_node *par) { 
  //fprintf(stderr, "chillAST_BinaryOperator::chillAST_BinaryOperator( l %p  %s  r %p  par %p)\n", l, oper, r, par); 

  lhs = l;
  rhs = r;

  if (lhs) lhs->setParent( this );  
  if (rhs) rhs->setParent( this );  // may only have part of the lhs and rhs when binop is created
  op = strdup(oper);
  asttype = CHILLAST_NODETYPE_BINARYOPERATOR; 

  // if this writes to lhs and lhs type has an 'imwrittento' concept, set that up
  if (isAssignmentOp()) { 
    if (lhs && lhs->isArraySubscriptExpr()) {
      ((chillAST_ArraySubscriptExpr*)lhs)->imwrittento = true;
      //fprintf(stderr, "chillAST_BinaryOperator, op '=', lhs is an array reference  LVALUE\n"); 
    }
  }
  if (isAugmentedAssignmentOp()) {  // +=  etc 
    //fprintf(stderr, "isAugmentedAssignmentOp()  "); print(); fflush(stdout); 
    if (lhs && lhs->isArraySubscriptExpr()) { 
      //fprintf(stderr, "lhs is also read from  ");  lhs->print(); fflush(stdout); 
      ((chillAST_ArraySubscriptExpr*)lhs)->imreadfrom = true; // note will ALSO have imwrittento true
    }
  }

  isFromSourceFile = true; // default 
  filename = NULL;
}


int chillAST_BinaryOperator::evalAsInt() { 
  // very limited. allow +-*/ and integer literals ...
  if (isAssignmentOp()) return rhs->evalAsInt();  // ?? ignores/loses lhs info 

  if (!strcmp("+", op)) { 
    //fprintf(stderr, "chillAST_BinaryOperator::evalAsInt()   %d + %d\n", lhs->evalAsInt(), rhs->evalAsInt()); 
    return lhs->evalAsInt() + rhs->evalAsInt(); 
  }
  if (!strcmp("-", op)) return lhs->evalAsInt() - rhs->evalAsInt(); 
  if (!strcmp("*", op)) return lhs->evalAsInt() * rhs->evalAsInt(); 
  if (!strcmp("/", op)) return lhs->evalAsInt() / rhs->evalAsInt(); 
  
  fprintf(stderr, "chillAST_BinaryOperator::evalAsInt() unhandled op '%s'\n", op); 
  segfault(); 
}

chillAST_IntegerLiteral *chillAST_BinaryOperator::evalAsIntegerLiteral() { 
  return new chillAST_IntegerLiteral( evalAsInt() ); // ?? 
}

void chillAST_BinaryOperator::dump( int indent, FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "(BinaryOperator '%s'\n", op);

  if (lhs) lhs->dump(indent+1, fp); // lhs could be null
  else { chillindent(indent+1, fp); fprintf(fp, "(NULL)\n"); } 
  fflush(fp); 

  if (rhs) rhs->dump(indent+1, fp); // rhs could be null
  else { chillindent(indent+1, fp); fprintf(fp, "(NULL)\n"); } 
  fflush(fp); 

  chillindent(indent, fp); 
  fprintf(fp, ")\n");
  fflush(fp); 
}

void chillAST_BinaryOperator::print( int indent, FILE *fp ) {   // TODO this needparens logic is wrong
  printPreprocBEFORE(indent, fp); 

  chillindent( indent, fp ); 
  bool needparens = false;
  if (lhs) { 
    if (lhs->isImplicitCastExpr()) { 
      //  fprintf(stderr, "\nlhs 0x%x isImplicitCastExpr()\n", lhs);
      //  fprintf(stderr, "lhs subexpr 0x%x\n", ((chillAST_ImplicitCastExpr*)lhs)->subexpr);
      //  fprintf(stderr, "lhs subexpr type %s\n", ((chillAST_ImplicitCastExpr*)lhs)->subexpr->getTypeString());
      //   
      if (((chillAST_ImplicitCastExpr*)lhs)->subexpr->isNotLeaf()) needparens = true;
    } 
    else if (lhs->isNotLeaf())  { 
      if      (isMinusOp()     && lhs->isPlusOp())     needparens = false;
      else if (isPlusMinusOp() && lhs->isMultDivOp())  needparens = false;
      else needparens = true;
    }
  }

  //fprintf(stderr, "\n\nbinop    "); 
  //lhs->printonly(0,stderr); 
  //fprintf(stderr," %s ",op); 
  //rhs->printonly(0,stderr); 
  //fprintf(stderr,"\n"); 
  //fprintf(stderr, "op is %s   lhs %s   rhs %s\n", op, lhs->getTypeString(), rhs->getTypeString());
  //fprintf(stderr, "lhs "); lhs->printonly(0, stderr); fprintf(stderr, "    "); 
  //fprintf(stderr, "lhs needparens = %d\n", needparens); 


  if (needparens) fprintf(fp, "(");
  if (lhs) lhs->print( 0, fp );
  else fprintf(fp, "(NULL)"); 
  if (needparens) fprintf(fp, ")"); 

  fprintf( fp, " %s ", op);
  //fprintf(fp, "\n"); // TMP DELETEME DFL

  needparens = false;
  //fprintf(stderr, "binop rhs is of type %s\n", rhs->getTypeString()); 
  if (rhs) { 
    if (rhs->isImplicitCastExpr()) { 
      if (((chillAST_ImplicitCastExpr*)rhs)->subexpr->isNotLeaf()) needparens = true;
    } 
    //else if (rhs->isNotLeaf()) needparens = true; // too many parens. test too simple
    else if (rhs->isNotLeaf()) { 
      // really need the precedence ordering, and check relative of op and rhs op
      if      (isMinusOp() ) needparens = true;    // safer.  perhaps complicated thing on rhs of a minus
      else if (isPlusMinusOp() && rhs->isMultDivOp())  needparens = false;
      else needparens = true;
    }
  }
  //fprintf(stderr, "rhs "); rhs->printonly(0, stderr); fprintf(stderr, "    "); 
  //fprintf(stderr, "rhs needparens = %d\n\n", needparens); 
  //if (!needparens) fprintf(stderr, "rhs isNotLeaf() = %d\n", rhs->isNotLeaf()); 

  if (needparens) fprintf(fp, "(");
  if (rhs) rhs->print( 0, fp );
  else fprintf(fp, "(NULL)"); 
  if (needparens) fprintf(fp, ")"); 
  fflush(fp); 
  printPreprocAFTER(indent, fp); 

}

void chillAST_BinaryOperator::printonly( int indent, FILE *fp ) {

  lhs->printonly( indent, fp );
  fprintf( fp, " %s ", op);
  rhs->printonly( 0, fp );
  fflush(fp); 



}


class chillAST_node* chillAST_BinaryOperator::constantFold() { 
  //fprintf(stderr, "\nchillAST_BinaryOperator::constantFold()  ");
  //print(0,stderr); fprintf(stderr, "\n");

  lhs = lhs->constantFold();
  rhs = rhs->constantFold();
  
  chillAST_node *returnval = this;

  if (lhs->isConstant() && rhs->isConstant() ) { 
    //fprintf(stderr, "binop folding constants\n"); print(0,stderr); fprintf(stderr, "\n");

    if (streq(op, "+") || streq(op, "-") || streq(op, "*")) { 
      if (lhs->isIntegerLiteral() && rhs->isIntegerLiteral()) {
        chillAST_IntegerLiteral *l = (chillAST_IntegerLiteral *)lhs;
        chillAST_IntegerLiteral *r = (chillAST_IntegerLiteral *)rhs;
        chillAST_IntegerLiteral *I;
        
        if (streq(op, "+")) I = new chillAST_IntegerLiteral(l->value+r->value, parent);
        if (streq(op, "-")) I = new chillAST_IntegerLiteral(l->value-r->value, parent);
        if (streq(op, "*")) I = new chillAST_IntegerLiteral(l->value*r->value, parent);

        returnval = I;
        //fprintf(stderr, "%d %s %d becomes %d\n", l->value,op, r->value, I->value);
      }
      else { // at least one is a float

        // usually don't want to do this for floats or doubles
        // could probably check for special cases like 0.0/30.0  or X/X  or X/1.0 
#ifdef FOLDFLOATS
        float lval, rval;
        if (lhs->isIntegerLiteral()) { 
          lval = (float) ((chillAST_IntegerLiteral *)lhs)->value; 
        }
        else { 
          lval = ((chillAST_FloatingLiteral *)lhs)->value; 
        }

        if (rhs->isIntegerLiteral()) { 
          rval = (float) ((chillAST_IntegerLiteral *)rhs)->value; 
        }
        else { 
          rval = ((chillAST_FloatingLiteral *)rhs)->value; 
        }

        chillAST_FloatingLiteral *F;
        if (streq(op, "+")) F = new chillAST_FloatingLiteral(lval + rval, parent);
        if (streq(op, "-")) F = new chillAST_FloatingLiteral(lval - rval, parent);
        if (streq(op, "*")) F = new chillAST_FloatingLiteral(lval * rval, parent);

        returnval = F;
#endif

      }
    }
    //else fprintf(stderr, "can't fold op '%s' yet\n", op); 
  }

  //fprintf(stderr, "returning "); returnval->print(0,stderr); fprintf(stderr, "\n"); 
  return returnval;
}


class chillAST_node* chillAST_BinaryOperator::clone() { 
  //fprintf(stderr, "chillAST_BinaryOperator::clone() "); print(); printf("\n"); fflush(stdout); 

  chillAST_node* l = lhs->clone();
  chillAST_node* r = rhs->clone();
  chillAST_BinaryOperator *bo =  new chillAST_BinaryOperator( l, op, r, parent ); 
  l->setParent( bo );
  r->setParent( bo );
  bo->isFromSourceFile = isFromSourceFile;
  if (filename) bo->filename = strdup(filename); 
  return bo;
}

void chillAST_BinaryOperator::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*>  &refs, bool w ) {
  //fprintf(stderr, "chillAST_BinaryOperator::gatherArrayRefs()\n"); 
  //print(); fflush(stdout); fprintf(stderr, "\n"); 
  //if (isAugmentedAssignmentOp()) { 
  //  fprintf(stderr, "%s  is augmented assignment\n", op);
  //}

  //if (isAssignmentOp()) { 
  //  fprintf(stderr, "%s  is assignment\n", op);
  //}

  //if (isAugmentedAssignmentOp()) { // lhs is ALSO on the RHS, NOT as a write
  //  if (lhs->isArraySubscriptExpr()) { // probably some case where this fails
  //    ((chillAST_ArraySubscriptExpr *) lhs)->imreadfrom = true;  
  //    //lhs->&gatherArrayRefs( refs, 0 );
  //  }
  //} 

  //fprintf(stderr, "in chillAST_BinaryOperator::gatherArrayRefs(), %d &arrayrefs before\n", refs.size());  
  lhs->gatherArrayRefs( refs, isAssignmentOp() );
  //fprintf(stderr, "in chillAST_BinaryOperator::gatherArrayRefs(), %d &arrayrefs after lhs\n", refs.size());  
  rhs->gatherArrayRefs( refs, 0 );
  //fprintf(stderr, "in chillAST_BinaryOperator::gatherArrayRefs(), %d &refs\n", refs.size()); 
  
  //for (int i=0; i<refs.size(); i++) { 
  //  fprintf(stderr, "%s\n", (*refs)[i]->basedecl->varname); 
  //} 

}

void chillAST_BinaryOperator::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  lhs->gatherScalarRefs( refs, isAssignmentOp() );
  rhs->gatherScalarRefs( refs, 0 );
} 


void chillAST_BinaryOperator::replaceChild( chillAST_node *old, chillAST_node *newchild ) {
  //fprintf(stderr, "\nbinop::replaceChild( old 0x%x, new )    lhs 0x%x   rhd 0x%x\n", old, lhs, rhs);

  // will pointers match??
  if      (lhs == old) setLHS( newchild ); 
  else if (rhs == old) setRHS( newchild ); 
  
  // silently ignore? 
  //else { 
  //  fprintf(stderr, "\nERROR chillAST_BinaryOperator::replaceChild( old 0x%x, new )    lhs 0x%x   rhd 0x%x\n", old, lhs, rhs);
  //  fprintf(stderr, "old is not a child of this BinaryOperator\n");
  //  print();
  //  dump();
  //  exit(-1); 
  //} 
}



void chillAST_BinaryOperator::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_BinaryOperator::gatherVarDecls()\n"); 

  //fprintf(stderr, "chillAST_BinaryOperator::gatherVarDecls()  before %d\n", decls.size()); 
  //print(0,stderr); fprintf(stderr, "\n"); 
  //fprintf(stderr, "lhs is %s\n", lhs->getTypeString()); 
  if (lhs) lhs->gatherVarDecls( decls ); // 'if' to deal with partially formed
  if (rhs) rhs->gatherVarDecls( decls );
  //fprintf(stderr, "after %d\n", decls.size()); 
}


void chillAST_BinaryOperator::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_BinaryOperator::gatherScalarVarDecls()  before %d\n", decls.size()); 
  //fprintf(stderr, "lhs is %s\n", lhs->getTypeString()); 
  lhs->gatherScalarVarDecls( decls );
  rhs->gatherScalarVarDecls( decls );
  //fprintf(stderr, "after %d\n", decls.size()); 
}


void chillAST_BinaryOperator::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_BinaryOperator::gatherArrayVarDecls()  before %d\n", decls.size()); 
  //fprintf(stderr, "lhs is %s\n", lhs->getTypeString()); 
  lhs->gatherArrayVarDecls( decls );
  rhs->gatherArrayVarDecls( decls );
  //fprintf(stderr, "after %d\n", decls.size()); 
}



void chillAST_BinaryOperator::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  lhs->gatherDeclRefExprs( refs );
  rhs->gatherDeclRefExprs( refs );
}


void chillAST_BinaryOperator::gatherStatements(std::vector<chillAST_node*> &statements ){
  
  // what's legit?
  if (isAssignmentOp()) { 
    statements.push_back( this );
  }

}




void chillAST_BinaryOperator::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  lhs->gatherVarUsage( decls );
  rhs->gatherVarUsage( decls );
}

void chillAST_BinaryOperator::gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls ) {
  lhs->gatherVarUsage( decls );
}

 void chillAST_BinaryOperator::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
   //if (!strcmp(op, "<=")) { 
   //  fprintf(stderr, "chillAST_BinaryOperator::replaceVarDecls( old %s, new %s)\n", olddecl->varname, newdecl->varname );
   //  print(); printf("\n"); fflush(stdout); 
   //  fprintf(stderr, "binaryoperator, lhs is of type %s\n", lhs->getTypeString()); 
   //  fprintf(stderr, "binaryoperator, rhs is of type %s\n", rhs->getTypeString()); 
   //} 
   lhs->replaceVarDecls( olddecl, newdecl ); 
   rhs->replaceVarDecls( olddecl, newdecl ); 
   //if (!strcmp(op, "<=")) { 
   //  print(); printf("\n\n"); fflush(stdout); 
   //} 
 }


bool chillAST_BinaryOperator::isSameAs( chillAST_node *other ){
  if (!other->isBinaryOperator()) return false;
  chillAST_BinaryOperator *o = (chillAST_BinaryOperator *)other;
  if (strcmp(op, o->op))  return false; // different operators 
  return lhs->isSameAs( o->lhs ) && rhs->isSameAs( o->rhs ); // recurse
}




chillAST_TernaryOperator::chillAST_TernaryOperator() {
  op = strdup("?"); // the only one so far
  condition = lhs = rhs = NULL;
  asttype = CHILLAST_NODETYPE_TERNARYOPERATOR; 
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_TernaryOperator::chillAST_TernaryOperator(const char *oper, chillAST_node *c, chillAST_node *l, chillAST_node *r, chillAST_node *par) {
  
  op = strdup(oper);
  condition = c;        condition->setParent( this ); 
  lhs = l;              lhs->setParent( this ); 
  rhs = r;              rhs->setParent( this ); 
  asttype = CHILLAST_NODETYPE_TERNARYOPERATOR; 
  isFromSourceFile = true; // default 
  filename = NULL;
}

void chillAST_TernaryOperator::dump( int indent, FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "(TernaryOperator '%s'\n", op);
  condition->dump(indent+1, fp);
  lhs->dump(indent+1, fp);
  rhs->dump(indent+1, fp);
  chillindent(indent, fp); 
  fprintf(fp, ")\n");
  fflush(fp);
}

void chillAST_TernaryOperator::print( int indent, FILE *fp ) {
  printPreprocBEFORE(indent, fp); 
  chillindent(indent, fp);
  fprintf(fp, "(");
  condition->print(0,fp);
  fprintf(fp, "%s", op); 
  lhs->print(0,fp);
  fprintf(fp, ":");
  rhs->print(0,fp);
  fprintf(fp, ")");
  fflush(fp);
}

void chillAST_TernaryOperator::replaceChild( chillAST_node *old, chillAST_node *newchild ) {
  //fprintf(stderr, "\nbinop::replaceChild( old 0x%x, new )    lhs 0x%x   rhd 0x%x\n", old, lhs, rhs);

  // will pointers match??
  if      (lhs == old) setLHS( newchild ); 
  else if (rhs == old) setRHS( newchild ); 
  else if (condition == old) setCond( newchild );

  // silently ignore? 
  //else { 
  //}
}


void chillAST_TernaryOperator::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  condition->gatherVarDecls( decls );
  lhs->gatherVarDecls( decls );
  rhs->gatherVarDecls( decls );
}

void chillAST_TernaryOperator::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  condition->gatherScalarVarDecls( decls );
  lhs->gatherScalarVarDecls( decls );
  rhs->gatherScalarVarDecls( decls );
}


void chillAST_TernaryOperator::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  condition->gatherArrayVarDecls( decls );
  lhs->gatherArrayVarDecls( decls );
  rhs->gatherArrayVarDecls( decls );
}



void chillAST_TernaryOperator::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  condition->gatherDeclRefExprs( refs );
  lhs->gatherDeclRefExprs( refs );
  rhs->gatherDeclRefExprs( refs );
}



void chillAST_TernaryOperator::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  condition->gatherVarUsage( decls );
  lhs->gatherVarUsage( decls );
  rhs->gatherVarUsage( decls );
}

void chillAST_TernaryOperator::gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls ) {
  // this makes no sense for ternary ?? 
}

void chillAST_TernaryOperator::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
  condition->replaceVarDecls( olddecl, newdecl ); 
  lhs->replaceVarDecls( olddecl, newdecl ); 
  rhs->replaceVarDecls( olddecl, newdecl ); 
}

void chillAST_TernaryOperator::printonly( int indent, FILE *fp ) {
  fprintf(fp, "(");
  condition->printonly(0,fp);
  fprintf(fp, "%s", op); 
  lhs->printonly(0,fp);
  fprintf(fp, ":");
  rhs->printonly(0,fp);
  fprintf(fp, ")");
  fflush(fp);
}


class chillAST_node* chillAST_TernaryOperator::constantFold() { 
  condition = condition->constantFold();
  lhs = lhs->constantFold();
  rhs = rhs->constantFold();
  
  chillAST_node *returnval = this;

  if (condition->isConstant())  { 
    //fprintf(stderr, "ternop folding constants\n");
    //print(0,stderr);
    //fprintf(stderr, "\n");

    // assume op is "?"
    // TODO 
    /* 
    
    if (streq(op, "+") || streq(op, "-") || streq(op, "*")) { 
      if (lhs->isIntegerLiteral() && rhs->isIntegerLiteral()) {
        chillAST_IntegerLiteral *l = (chillAST_IntegerLiteral *)lhs;
        chillAST_IntegerLiteral *r = (chillAST_IntegerLiteral *)rhs;
        chillAST_IntegerLiteral *I;
        
        if (streq(op, "+")) I = new chillAST_IntegerLiteral(l->value+r->value, parent);
        if (streq(op, "-")) I = new chillAST_IntegerLiteral(l->value-r->value, parent);
        if (streq(op, "*")) I = new chillAST_IntegerLiteral(l->value*r->value, parent);

        returnval = I;
        //fprintf(stderr, "%d %s %d becomes %d\n", l->value,op, r->value, I->value);
      }
      else { // at least one is a float
        float lval, rval;
        if (lhs->isIntegerLiteral()) { 
          lval = (float) ((chillAST_IntegerLiteral *)lhs)->value; 
        }
        else { 
          lval = ((chillAST_FloatingLiteral *)lhs)->value; 
        }
        if (rhs->isIntegerLiteral()) { 
          rval = (float) ((chillAST_IntegerLiteral *)rhs)->value; 
        }
        else { 
          rval = ((chillAST_FloatingLiteral *)rhs)->value; 
        }

        chillAST_FloatingLiteral *F;
        if (streq(op, "+")) F = new chillAST_FloatingLiteral(lval + rval, parent);
        if (streq(op, "-")) F = new chillAST_FloatingLiteral(lval - rval, parent);
        if (streq(op, "*")) F = new chillAST_FloatingLiteral(lval * rval, parent);

        returnval = F;
      }
    }
    else fprintf(stderr, "can't fold op '%s' yet\n", op); 
    */
  }

  return returnval;
}

class chillAST_node* chillAST_TernaryOperator::clone() { 
  chillAST_node* c = condition->clone();
  chillAST_node* l = lhs->clone();
  chillAST_node* r = rhs->clone();
  chillAST_TernaryOperator *to =  new chillAST_TernaryOperator( op, l, r, parent ); 
  c->setParent( to ); 
  l->setParent( to );
  r->setParent( to );
  to->isFromSourceFile = isFromSourceFile;
  filename = NULL;
  return to;
}

void chillAST_TernaryOperator::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*>  &refs, bool w ) {
  condition->gatherArrayRefs( refs, isAssignmentOp() );
  lhs->gatherArrayRefs( refs, isAssignmentOp() );
  rhs->gatherArrayRefs( refs, 0 );
}

void chillAST_TernaryOperator::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  condition->gatherScalarRefs( refs, isAssignmentOp() );
  lhs->gatherScalarRefs( refs, isAssignmentOp() );
  rhs->gatherScalarRefs( refs, 0 );
} 







chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() { 
  //fprintf(stderr, "\n%p chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 0\n", this); 
  asttype = CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR; 
  base = index = NULL;
  basedecl = NULL; //fprintf(stderr, "setting basedecl NULL for ASE %p\n", this); 
  imwrittento = false; // ?? 
  imreadfrom  = false; // ?? 
  parent = NULL;
  metacomment = NULL;
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr()  NEED TO FAKE A LOCATION\n");
  isFromSourceFile = true; // default 
  filename = NULL;
  
}



chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, chillAST_node *par, void *unique ) { 

  //fprintf(stderr, "\nchillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 1\n"); 
  asttype = CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR; 
  bas->setParent( this );
  if (bas->isImplicitCastExpr()) base = ((chillAST_ImplicitCastExpr*)bas)->subexpr; // probably wrong
  else   base = bas;
  if (indx->isImplicitCastExpr()) index = ((chillAST_ImplicitCastExpr*)indx)->subexpr; // probably wrong
  else index = indx;
  base->setParent( this );
  index->setParent( this );
  imwrittento = false; // ?? 
  imreadfrom  = false; // ?? 
  uniquePtr = (void *) unique;
  //fprintf(stderr,"chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() original = 0x%x\n", uniquePtr); 
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 1 calling multibase()\n"); 
  basedecl = multibase();//fprintf(stderr, "%p  ASE 1 basedecl = %p\n",this,basedecl); 
  //basedecl->print(); printf("\n");
  //basedecl->dump(); printf("\n"); fflush(stdout); 
  //fprintf(stderr, "basedecl varname %s\n", basedecl->varname); 
  isFromSourceFile = true; // default 
  filename = NULL;
}



chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, bool writtento, chillAST_node *par, void  *unique ) {
  //fprintf(stderr, "\nchillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 2\n"); 

  asttype = CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR; 
  bas->setParent( this );
  if (bas->isImplicitCastExpr()) base = ((chillAST_ImplicitCastExpr*)bas)->subexpr; // probably wrong
  else base = bas;
    if (indx->isImplicitCastExpr()) index = ((chillAST_ImplicitCastExpr*)indx)->subexpr; // probably wrong
  else index = indx;
  base->setParent( this );
  index->setParent( this );
  imwrittento = writtento; // ?? 
  //fprintf(stderr, "ASE %p   imwrittento %d\n", this, imwrittento);
  imreadfrom  = false; // ??  

  uniquePtr = (void *) unique;
  //fprintf(stderr,"chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() original = 0x%x\n", uniquePtr); 

  basedecl = multibase(); 

  //fprintf(stderr, "%p  ASE 2 basedecl = %p\n", this, basedecl); 
  //printf("basedecl is  "); fflush(stdout); basedecl->print(); printf("\n"); fflush(stdout); 
  //basedecl->dump(); printf("\n"); fflush(stdout);
  //fprintf(stderr, "basedecl varname %s\n", basedecl->varname); 
  isFromSourceFile = true; // default 
  filename = NULL;
 }



chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr( chillAST_VarDecl *v, std::vector<chillAST_node *> indeces) {
  //fprintf(stderr, "\nchillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 4\n"); 
  //fprintf(stderr,"chillAST_ArraySubscriptExpr( chillAST_VarDecl *v, std::vector<int> indeces)\n");
  asttype = CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR; 
  int numindeces = indeces.size();
  //for (int i=0; i<numindeces; i++) { 
  //  printf("[");
  //  indeces[i]->print();
  //  printf("]");
  //} 
  //fflush(stdout); 
  //fprintf(stderr, "\n");
  
  chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( v->vartype, v->varname, v, NULL);
  basedecl = v; // ?? 
  //fprintf(stderr, "%p  ASE 3 basedecl = %p   ", this, basedecl); 
  //fprintf(stderr, "of type %s\n", basedecl->getTypeString()); 
  //basedecl->print(); printf("\n");
  //basedecl->dump(); printf("\n"); fflush(stdout); 
  //fprintf(stderr, "basedecl varname %s\n", basedecl->varname); 
  
  chillAST_ArraySubscriptExpr *rent = this; // parent for subnodes
  
  // these are on the top level ASE that we're creating here 
  base = (chillAST_node *) DRE;
  index = indeces[ numindeces-1];

  for (int i=numindeces-2; i>=0; i--) {
    
    chillAST_ArraySubscriptExpr *ASE = new  chillAST_ArraySubscriptExpr( DRE, indeces[i], rent, 0); 
    rent->base = ASE; // 
    rent = ASE;
  }
  
  imwrittento = false;
  imreadfrom = false; 
  //fprintf(stderr, "ASE is "); print(); printf("\n\n"); fflush(stdout); 
  isFromSourceFile = true; // default 
  filename = NULL;
}



chillAST_node *chillAST_node::getEnclosingStatement( int level ) {  // TODO do for subclasses?

  //fprintf(stderr, "chillAST_node::getEnclosingStatement( level %d ) node type %s\n", level, getTypeString());
  //print(); printf("\n"); fflush(stdout);

  // so far, user will ONLY call this directly on an array subscript expression
  if (isArraySubscriptExpr()) return parent->getEnclosingStatement( level+1); 

  if (level != 0) { 
    if (isBinaryOperator()    || 
        isUnaryOperator()     || 
        isTernaryOperator()   || 
        isReturnStmt()        ||
        isCallExpr()
        ) return this;
    

    // things that are not endpoints. recurse through parent 
    if (isMemberExpr())       return parent->getEnclosingStatement( level+1 ); 
    if (isImplicitCastExpr()) return parent->getEnclosingStatement( level+1 ); 
    if (isSizeof())           return parent->getEnclosingStatement( level+1 );  
    if (isCStyleCastExpr())   return parent->getEnclosingStatement( level+1 );  
    return NULL;
  }

  fprintf(stderr, "getEnclosingStatement() level %d type %s, returning NULL\n", level, getTypeString()); 
  segfault(); 

  return NULL;
}



void chillAST_ArraySubscriptExpr::gatherIndeces(std::vector<chillAST_node*>&ind) { 
  if (base->isArraySubscriptExpr()) ((chillAST_ArraySubscriptExpr *)base)->gatherIndeces( ind ); 
  ind.push_back( index );
}



void chillAST_ArraySubscriptExpr::dump( int indent, FILE *fp ) {
//  fprintf(stderr, "\n%p chillAST_ArraySubscriptExpr::dump()  basedecl %p\n", basedecl);
  
  char *local;
  if (basedecl && basedecl->vartype) {
    local = strdup( basedecl->vartype );
  }
  else { 
    fprintf(stderr, "%p chillAST_ArraySubscriptExpr::dump(), no basedecl ???\n",this);
    local = strdup("");
    //fprintf(stderr, "base is "); base->dump(); printf("\n"); base->print(); printf("\n"); fflush(stdout); 
    //print(); printf("\n"); fflush(stdout);
  }


  char *space = rindex(local, ' ');  // can't use index because it's a class member!
  if (space) *space = '\0';  // turn "float *" into "float"

  chillindent(indent, fp);
  //fprintf(fp, "(ArraySubscriptExpr '%s' ", local);
  if (basedecl)  { 
    //fprintf(stderr, " chillAST_ArraySubscriptExpr::dump() basedecl is of type %s\n",   basedecl->getTypeString()); 
    fprintf(fp, "(ArraySubscriptExpr (%s) '%s' ", basedecl->varname, local); 
  }
  else fprintf(stderr, " chillAST_ArraySubscriptExpr::dump() has no basedecl\n"); 
  free(local);

  if (imwrittento) { 
    if (imreadfrom) fprintf(fp, "lvalue AND rvalue\n"); 
    else            fprintf(fp, "lvalue\n");
  }
  else fprintf(fp, "rvalue\n"); 
  base->dump( indent+1, fp );
  index->dump(indent+1, fp); 

  chillindent(indent, fp);
  fprintf(fp, ")\n");
  fflush(fp); 
}



void chillAST_ArraySubscriptExpr::print( int indent, FILE *fp ) {
  base->print( indent, fp );
  fprintf(fp, "[");
  index->print(0, fp); 
  fprintf(fp, "]");
  fflush(fp); 
}
void chillAST_ArraySubscriptExpr::printonly( int indent, FILE *fp ) {
  base->printonly( indent, fp );
  fprintf(fp, "[");
  index->printonly(0, fp); 
  fprintf(fp, "]");
  fflush(fp); 
}


void chillAST_ArraySubscriptExpr::print( int indent, FILE *fp ) const {
  base->print( indent, fp );
  fprintf(fp, "[");
  index->print(0, fp); 
  fprintf(fp, "]");
  fflush(fp); 
};


chillAST_VarDecl *chillAST_ArraySubscriptExpr::multibase() { 
  // return the VARDECL of the thing the subscript is an index into
  //this should probably be a chillAST_node function instead of having all these ifs
  //print(); printf("\n"); fflush(stdout); 
  //base->print();  printf("\n"); fflush(stdout); 
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::multibase()  base of type %s\n", base->getTypeString()); 
  
  return base->multibase();  

  // this will be used to SET basedecl
  //basedecl = NULL; // do this so we don't confuse ourselves looking at uninitialized basedecl

  chillAST_node *b = base; 
  //fprintf(stderr, "base is of type %s\n", b->getTypeString());

  if (!b) return NULL; // just in case ??

  if (base->asttype == CHILLAST_NODETYPE_IMPLICITCASTEXPR) { // bad coding
    b = ((chillAST_ImplicitCastExpr*)b)->subexpr;
  }

  if (b->asttype == CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR) { // multidimensional array!
    // recurse
    return ((chillAST_ArraySubscriptExpr *)b)->multibase();
  }

  if (b->asttype == CHILLAST_NODETYPE_DECLREFEXPR) return(((chillAST_DeclRefExpr*)b)->getVarDecl());

  
  if (b->isBinaryOperator()) { 
    // presumably a dot or pointer ref that resolves to an array
    chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *) b;
    if ( strcmp(BO->op, ".") ) { 
      fprintf(stderr, "chillAST_ArraySubscriptExpr::multibase(), UNHANDLED case:\n");
      fprintf(stderr, "base is binary operator, of type %s\n", BO->op); 
      exit(-1);
    }

    chillAST_node *l = BO->lhs;
    chillAST_node *r = BO->rhs;
    printf("L %s\nR %s\n", l->getTypeString(), r->getTypeString()); 
    exit(-1); 

    return NULL; // TODO do checks?
  }

  if (b->isMemberExpr()) { 
    //c.i[c.count]    we want i member of inspector

    chillAST_MemberExpr *ME = (chillAST_MemberExpr *) b;
    //fprintf(stderr, "multibase() Member Expression "); ME->print(); printf("\n"); fflush(stdout); 

    chillAST_node *n = ME->base; //  WRONG   want the MEMBER
    //fprintf(stderr, "chillAST_ArraySubscriptExpr::multibase()  Member Expression base of type %s\n", n->getTypeString());
    //fprintf(stderr, "base is "); ME->base->dump(); 

    // NEED to be able to get lowest level recorddecl or typedef from this base

    fprintf(stderr, "chillast.cc, L2315, bailing??\n"); 
    exit(0); 

    if (!n->isDeclRefExpr()) { 
      fprintf(stderr, "MemberExpr member is not chillAST_DeclRefExpr\n");
      exit(-1);
    }
    chillAST_DeclRefExpr *DRE = (chillAST_DeclRefExpr *)n;
    n = DRE->decl;
    //fprintf(stderr, "DRE decl is of type %s\n", n->getTypeString()); 
    assert( n->isVarDecl() );
    chillAST_VarDecl *vd = (chillAST_VarDecl *) n;
    vd->print(); printf("\n"); fflush(stdout); 

    chillAST_TypedefDecl *tdd = vd->typedefinition; 
    chillAST_RecordDecl  *rd  = vd->vardef; 
    //fprintf(stderr, "tdd %p    rd %p\n", tdd, rd); 
    
    print(); printf("\n"); 
    dump();  printf("\n"); fflush(stdout);

    assert( tdd != NULL || rd != NULL );
    
    chillAST_VarDecl *sub;
    if (tdd) sub = tdd->findSubpart( ME->member ); 
    if (rd)  sub =  rd->findSubpart( ME->member ); 

    //fprintf(stderr, "subpart is "); sub->print(); printf("\n"); fflush(stdout); 
    
    return sub; // what if the sub is an array ??  TODO 
  }


  fprintf(stderr, "chillAST_ArraySubscriptExpr::multibase(), UNHANDLED case %s\n", 
          b->getTypeString()); 
  print(); printf("\n"); fflush(stdout);
  fprintf(stderr, "base is: "); b->print(); printf("\n"); fflush(stdout);
  segfault(); 
}


chillAST_node *chillAST_ArraySubscriptExpr::getIndex(int dim) {
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::getIndex( %d )\n", dim); 

  chillAST_node *b = base; 

  int depth = 0;
  std::vector<chillAST_node*> ind;
  chillAST_node *curindex = index;
  for (;;) { 
    if (b->asttype == CHILLAST_NODETYPE_IMPLICITCASTEXPR) b = ((chillAST_ImplicitCastExpr*)b)->subexpr;
    else if (b->asttype == CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR) {
      //fprintf(stderr, "base  "); b->print(); fprintf(stderr, "\n"); 
      //fprintf(stderr, "index "); curindex->print(); fprintf(stderr, "\n"); 
      ind.push_back(curindex);
      curindex = ((chillAST_ArraySubscriptExpr*)b)->index;
      b = ((chillAST_ArraySubscriptExpr*)b)->base; 
      depth++;
    }
    else { 
      //fprintf(stderr, "base  "); b->print(); fprintf(stderr, "\n"); 
      //fprintf(stderr, "index "); curindex->print(); fprintf(stderr, "\n"); 
      //fprintf(stderr, "stopping at base type %s\n", b->getTypeString());
      ind.push_back(curindex);
      break; 
    }
  }
  //fprintf(stderr, "depth %d\n", depth );
  //for (int i=0; i<ind.size(); i++) { ind[i]->print(); fprintf(stderr, "\n"); } 

  return ind[ depth - dim ]; 
  /* 
  if (dim == 0) return index; // single dimension 
  fprintf(stderr, "DIM NOT 0\n"); 
  // multidimension 
  chillAST_node *b = base; 
  if (base->asttype == CHILLAST_NODETYPE_IMPLICITCASTEXPR) { // bad coding
    b = ((chillAST_ImplicitCastExpr*)b)->subexpr;
  }
  if (b->asttype == CHILLAST_NODETYPE_IMPLICITCASTEXPR) { // bad coding
    b = ((chillAST_ImplicitCastExpr*)b)->subexpr;
  }
  
  b->print(); printf("\n"); fflush(stdout);
  if (b->asttype == CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR) {
    return ((chillAST_ArraySubscriptExpr *)b)->getIndex(dim-1);
  }

  fprintf(stderr, "chillAST_ArraySubscriptExpr::getIndex() failed\n");
  */ 
  exit(-1); 
}




class chillAST_node* chillAST_ArraySubscriptExpr::constantFold() { 
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::constantFold()\n"); 
  base  =  base->constantFold();
  index = index->constantFold();
  return this;
}

class chillAST_node* chillAST_ArraySubscriptExpr::clone() { 
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::clone() old imwrittento %d\n", imwrittento);
  //fprintf(stderr, "cloning "); print(); printf("\n"); fflush(stdout);
  //fprintf(stderr, "old base   "); base->print();  printf("\n"); fflush(stdout);
  //fprintf(stderr, "old base   "); base->dump();  printf("\n"); fflush(stdout);
  if (base->isDeclRefExpr()) { 
    chillAST_VarDecl *vd = (chillAST_VarDecl *)(((chillAST_DeclRefExpr *)base)->decl); 
    //fprintf(stderr, "old decl   "); vd->print();  printf("\n");fflush(stdout);
    //fprintf(stderr, "old decl   "); vd->dump();   printf("\n");fflush(stdout);
  }
  chillAST_node *b =  base->clone();
  //fprintf(stderr, "new base   "); b->print();  printf("\n"); fflush(stdout);
  //fprintf(stderr, "new base   "); b->dump();  printf("\n"); fflush(stdout);
  chillAST_node *i = index->clone();
  chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( b, i, imwrittento, parent, uniquePtr /* ?? */ ); 

  ASE->imreadfrom = false; // don't know this yet
  //ASE->imreadfrom = imreadfrom; // ?? 
  //if (ASE->imreadfrom) { 
  //  fprintf(stderr, "in chillAST_ArraySubscriptExpr::clone(), imreadfrom is being set. \n");
  //  ASE->print(); fflush(stdout); fprintf(stderr, "\n"); 
  //} 

  //fprintf(stderr, "cloned result "); ASE->print(); printf("\n\n\n"); fflush(stdout);
  //fprintf(stderr, "ASE clone()  this 0x%x    clone 0x%x\n", this, ASE); 
  ASE->isFromSourceFile = isFromSourceFile;
  if (filename) ASE->filename = strdup(filename); 
  return ASE;
}

void chillAST_ArraySubscriptExpr::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::gatherArrayRefs setting imwrittento %d for ", writtento); base->print(); printf("\n"); fflush(stdout); 

  //fprintf(stderr, "found an array subscript. &refs 0x%x   ", refs);
  if (!imwrittento) imwrittento = writtento;   // may be both written and not for += 
  fflush(stdout); 
  //fprintf(stderr, "refs[%d] = 0x%x  = ", refs.size(), this); print(); fflush(stdout);   printf("\n"); fflush(stdout); 


  index->gatherArrayRefs( refs, 0 ); // recurse first
  refs.push_back( this );
  //fprintf(stderr, " size now %d\n", refs.size()); 

}

void chillAST_ArraySubscriptExpr::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  index->gatherScalarRefs( refs, 0 ); 
} 

void chillAST_ArraySubscriptExpr::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::gatherVarDecls()\n"); 

  base->gatherVarDecls( decls );
  index->gatherVarDecls( decls );
}


void chillAST_ArraySubscriptExpr::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::gatherScalarVarDecls()\n");
  //fprintf(stderr, "base %s   index %s\n", base->getTypeString(), index->getTypeString()); 
  base->gatherScalarVarDecls( decls );
  index->gatherScalarVarDecls( decls );
}


void chillAST_ArraySubscriptExpr::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::gatherArrayVarDecls()\n");
  //fprintf(stderr, "base %s   index %s\n", base->getTypeString(), index->getTypeString()); 
  base->gatherArrayVarDecls( decls );
  index->gatherArrayVarDecls( decls );
}


void chillAST_ArraySubscriptExpr::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  base->gatherDeclRefExprs( refs );
  index->gatherDeclRefExprs( refs );
}


void chillAST_ArraySubscriptExpr::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  base->gatherVarUsage( decls );
  index->gatherVarUsage( decls );
}


void chillAST_ArraySubscriptExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  base->replaceVarDecls( olddecl, newdecl );
  index->replaceVarDecls( olddecl, newdecl );
}


void chillAST_ArraySubscriptExpr::replaceChild( chillAST_node *old, chillAST_node *newchild ) { 
  fprintf(stderr,"chillAST_ArraySubscriptExpr::replaceChild()\n"); 

  // arraysubscriptexpression doesn t really have children (should it?)
  // try index ???
  if (old == index) { 
    fprintf(stderr, "old is index\n");
    index = newchild;
    return;
  }
  
  // try base ??? unclear if this makes sense  TODO 
  if (old == base) { 
    fprintf(stderr, "old is base\n");
    base = newchild;
    return;
  }
  
  fprintf(stderr, "chillAST_ArraySubscriptExpr::replaceChild() old is not base or index\n"); 
  print(0,stderr); fprintf(stderr, "\nchild: ");
  if (!old) fprintf(stderr, "oldchild NULL!\n");
  old->print(0,stderr); fprintf(stderr, "\nnew: "); 
  newchild->print(0,stderr); fprintf(stderr, "\n"); 
  segfault(); // make easier for gdb
};


bool chillAST_ArraySubscriptExpr::operator!=( const chillAST_ArraySubscriptExpr &other) {
  bool opposite = *this == other;
  return !opposite;
}



bool chillAST_ArraySubscriptExpr::operator==( const chillAST_ArraySubscriptExpr &other) {
  //fprintf(stderr, "chillAST_ArraySubscriptExpr::operator==\n");
  //fprintf(stderr, "this->basedecl 0x%x     other.basedecl 0x%x\n", this->basedecl, other.basedecl);
  //this->basedecl->print(); printf("\n\n");
  //other.basedecl->print(); printf("\n"); fflush(stdout);

  //this->print(); printf(" 0x%x  == 0x%x ",this->uniquePtr, other.uniquePtr ); other.print(); printf(" ??  "); fflush(stdout); 
  //if ( this->uniquePtr == other.uniquePtr) fprintf(stderr, "t\n"); 
  //else fprintf(stderr, "f\n"); 
  return this->uniquePtr == other.uniquePtr; 
}





chillAST_MemberExpr::chillAST_MemberExpr() { 
  asttype = CHILLAST_NODETYPE_MEMBEREXPR; 
  base = NULL;
  member = NULL;
  parent = NULL;
  metacomment = NULL;
  exptype = CHILL_MEMBER_EXP_DOT; 
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_MemberExpr::chillAST_MemberExpr( chillAST_node *bas, const char *mem, chillAST_node *p, void *unique, CHILL_MEMBER_EXP_TYPE t ) { 
  asttype = CHILLAST_NODETYPE_MEMBEREXPR; 
  base = bas;
  base->setParent( this ); 
  member = strdup( mem );
  parent = p;
  metacomment = NULL;
  uniquePtr = unique;
  exptype = t;
  isFromSourceFile = true; // default 
  filename = NULL;

  return;  // ignore tests below ?? TODO ?? 


  // base needs to RESOLVE to a decl ref expr but may not BE one
  //   A.b . c   lhs is a binop or memberexpr

  if (bas->isBinaryOperator()) { 
    //fprintf(stderr, "checking binop to see if it resolved to a declrefexpr\n");
    // cheat for now or just remove the check below
    return; 
  }

  if (! ( bas->isDeclRefExpr() || bas->isArraySubscriptExpr() )) { 
    fprintf(stderr, "chillAST_MemberExpr::chillAST_MemberExpr(), base is of type %s\n", bas->getTypeString());
    fprintf(stderr, "chillAST_MemberExpr::chillAST_MemberExpr(), base is not DeclRefExpr\n");
    
    base->print();  printf(".%s\n", mem); fflush(stdout);  
    segfault(); 
  }
}


void chillAST_MemberExpr::dump( int indent, FILE *fp ) {
  chillindent(indent, fp);
  fprintf(fp, "(MemberExpr \n"); 

  base->dump( indent+1, fp );
  chillindent(indent+1, fp);
  if (exptype == CHILL_MEMBER_EXP_ARROW) fprintf(fp, "->");
  else fprintf(fp, "."); 
  
  fprintf(fp, "%s\n", member); 

  chillindent(indent, fp);
  fprintf(fp, ")\n");
}


void chillAST_MemberExpr::print( int indent, FILE *fp ) {
  base->print( indent, fp );
  if (exptype == CHILL_MEMBER_EXP_ARROW) fprintf(fp, "->");
  else fprintf(fp, "."); 
  fprintf(fp, "%s", member);
  fflush(fp); 
}


void chillAST_MemberExpr::printonly( int indent, FILE *fp ) {
  base->print( indent, fp );
  if (exptype == CHILL_MEMBER_EXP_ARROW) fprintf(fp, "->");
  else fprintf(fp, "."); 
  fprintf(fp, "%s", member);
  fflush(fp); 
}

char *chillAST_MemberExpr::getStringRep() { // char pointer to what we'd print
  if (base->isDeclRefExpr()) { // 
    chillAST_VarDecl *vd =  (chillAST_VarDecl *) ((chillAST_DeclRefExpr *)base)->decl;
    char *leak = (char *)malloc(128);
    if (exptype == CHILL_MEMBER_EXP_ARROW) sprintf(leak, "%s->%s", vd->varname, member);
    else sprintf(leak, "%s.%s", vd->varname, member);
    printstring = leak; 
    return leak;
  }


  // else 
    // TODO
    return strdup("chillAST_MemberExpr::getStringRep()hadanerror");
}


class chillAST_node* chillAST_MemberExpr::constantFold() { 
  base  =  base->constantFold();
  //member = member->constantFold();
  return this;
}

class chillAST_node* chillAST_MemberExpr::clone() { 
  chillAST_node *b =  base->clone();
  char *m = strdup( member ); // ?? 
  chillAST_MemberExpr *ME = new chillAST_MemberExpr( b, m, parent, uniquePtr /* ?? */ ); 
  ME->isFromSourceFile = isFromSourceFile;
  if (filename) ME->filename = strdup(filename); 
  return ME;
}

void chillAST_MemberExpr::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {
  base->gatherArrayRefs( refs, writtento );
}

void chillAST_MemberExpr::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  base->gatherScalarRefs( refs, writtento );
} 

void chillAST_MemberExpr::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  base->gatherVarDecls( decls );
}

void chillAST_MemberExpr::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  base->gatherScalarVarDecls( decls );
}


void chillAST_MemberExpr::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  base->gatherArrayVarDecls( decls );
}


void chillAST_MemberExpr::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  base->gatherDeclRefExprs( refs );
}


void chillAST_MemberExpr::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  base->gatherVarUsage( decls );
}


void chillAST_MemberExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  base->replaceVarDecls( olddecl, newdecl );
}

bool chillAST_MemberExpr::operator!=( const chillAST_MemberExpr &other) {
  bool opposite = *this == other;
  return !opposite;
}

bool chillAST_MemberExpr::operator==( const chillAST_MemberExpr &other) {
  return this->uniquePtr == other.uniquePtr; 
}


void chillAST_MemberExpr::replaceChild( chillAST_node *old, chillAST_node *newchild ) {
  //printf("\nMemberExpr::replaceChild(  )\n");
  //printf("old: "); 
  //old->print(); 
  //printf("\nnew: "); 
  //newchild->print(); 
  //printf("\n"); fflush(stdout); 

  // will pointers match??
  if (base == old) { 
    //fprintf(stderr, "old matches base of MemberExpr\n"); 
    base = newchild; 
  }
  else { 
    base->replaceChild( old, newchild ); 
  }
} 

chillAST_node  *chillAST_MemberExpr::multibase2() {  /*fprintf(stderr, "ME MB2\n" );*/ return (chillAST_node *)this; } 

chillAST_VarDecl* chillAST_MemberExpr::getUnderlyingVarDecl() { 
  fprintf(stderr, "chillAST_MemberExpr:getUnderlyingVarDecl()\n");
  print(); printf("\n"); fflush(stdout);
  exit(-1); 
  // find the member with the correct name
  
}




chillAST_VarDecl *chillAST_MemberExpr::multibase() {
  //c.i[c.count]    we want i member of c 
  //fprintf(stderr, "ME MB\n" ); 

  //fprintf(stderr, "chillAST_MemberExpr::multibase()\n");
  //print(); printf("\n"); fflush(stdout);
  //fprintf(stderr, "MemberExpr base is type %s,  member %s\n", base->getTypeString(), member);
  
  //chillAST_VarDecl *vd = base->getUnderlyingVarDecl(); // this is the only thing that ever calls this ??? 
  chillAST_VarDecl *vd = base->multibase(); // ?? 


  //fprintf(stderr, "vd "); vd->print(); printf("\n"); fflush(stdout);
    
  chillAST_RecordDecl *rd = vd->getStructDef();
  if (!rd) { 
    fprintf(stderr, "chillAST_MemberExpr::multibase() vardecl is not a struct??\n");
    fprintf(stderr, "vd "); vd->print(); printf("\n"); fflush(stdout);
    fprintf(stderr, "vd "); vd->dump();  printf("\n"); fflush(stdout);
    exit(-1);
  }

  // OK, we have the recorddecl that defines the structure
  // now find the member with the correct name
  chillAST_VarDecl *sub = rd->findSubpart( member );
  //fprintf(stderr, "sub %s:\n", member);
  if (!sub) { 
    fprintf(stderr, "can't find member %s in \n", member);
    rd->print(); 
  }
  //sub->print(); printf("\n");  fflush(stdout);
  //sub->dump() ; printf("\n");  fflush(stdout);

  return sub; 
  //find vardecl of member in def of base

  
}




chillAST_DeclRefExpr::chillAST_DeclRefExpr() { 
  asttype = CHILLAST_NODETYPE_DECLREFEXPR; 
  declarationType = strdup("UNKNOWN"); 
  declarationName = strdup("NONE");
  decl = NULL; 
  parent = NULL;
  metacomment = NULL;
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( const char *varname, chillAST_node *par ) { 
  asttype = CHILLAST_NODETYPE_DECLREFEXPR; 
  declarationType = strdup("UNKNOWN"); 
  declarationName = strdup(varname); 
  decl = NULL; 
  parent = par; 
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( const char *vartype, const char *varname, chillAST_node *par) {
  //fprintf(stderr, "DRE::DRE 0x%x   %s %s\n", this, vartype, varname ); 
  asttype = CHILLAST_NODETYPE_DECLREFEXPR; 
  declarationType = strdup(vartype); 
  declarationName = strdup(varname); 
  decl = NULL; 
  parent = par; 
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( const char *vartype, const char *varname, chillAST_node *d, chillAST_node *par ) {
  //fprintf(stderr, "DRE::DRE2 0x%x   %s %s  0x%x\n", this, vartype, varname, d ); 
  asttype = CHILLAST_NODETYPE_DECLREFEXPR; 
  declarationType = strdup(vartype); 
  declarationName = strdup(varname); 
  decl = d; 
  parent = par; 
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( chillAST_VarDecl *vd, chillAST_node *par ){ // variable def
  //fprintf(stderr, "DRE::DRE3 (VD)  0x%x   %s %s  0x%x\n", this, vd->vartype, vd->varname, vd ); 
  
  asttype = CHILLAST_NODETYPE_DECLREFEXPR; 
  declarationType = strdup(vd->vartype); 
  declarationName = strdup(vd->varname); 
  decl = vd; 
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}


chillAST_DeclRefExpr::chillAST_DeclRefExpr( chillAST_FunctionDecl *fd, chillAST_node *par ){ // function def 
  asttype = CHILLAST_NODETYPE_DECLREFEXPR; 
  declarationType = strdup(fd->returnType); 
  declarationName = strdup(fd->functionName); 
  decl = fd; 
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}



chillAST_DeclRefExpr *buildDeclRefExpr( chillAST_VarDecl  *vd) { 
  chillAST_DeclRefExpr *dre = new chillAST_DeclRefExpr( vd, NULL );
  
}

void chillAST_DeclRefExpr::print( int indent, FILE *fp) {
  chillindent(indent, fp);
  //fprintf(fp, "%s %s", declarationType, declarationName);  // this is printing  float *A 
  fprintf(fp, "%s", declarationName);  // this is printing  A 
  fflush(fp); 
}

void chillAST_DeclRefExpr::dump( int indent, FILE *fp) {
  chillindent(indent, fp);
  fprintf(fp, "(DeclRefExpr '%s' ", declarationType);  
  chillAST_VarDecl *vd = getVarDecl();
  if (vd) { 
    if (vd->isAParameter) fprintf(fp, "ParmVar  ");
    else fprintf(fp, "Var  ");
  }
  fprintf(fp, "'%s' ", declarationName);  // variable or function name 

  if (chillAST_FunctionDecl *fd = getFunctionDecl()) { 
    // print parameter types for functions 
    fd->printParameterTypes( fp );
  }
          
  fprintf(fp, ")\n"); 
  fflush(fp); 
}

class chillAST_node* chillAST_DeclRefExpr::constantFold() {  // can never do anything?
  return this;
}

class chillAST_node* chillAST_DeclRefExpr::clone() { 
  //fprintf(stderr, "chillAST_DeclRefExpr::clone()\n"); 
  chillAST_DeclRefExpr *DRE =  new chillAST_DeclRefExpr( declarationType, declarationName, decl, parent ); 
  DRE->isFromSourceFile = isFromSourceFile;
  if (filename) DRE->filename = strdup(filename); 
  return DRE;
}


void chillAST_DeclRefExpr::gatherVarDeclsMore( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_DeclRefExpr::gatherVarDeclsMore()\n"); 
  decl->gatherVarDeclsMore( decls ); 
}


void chillAST_DeclRefExpr::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_DeclRefExpr::gatherScalarVarDecls()\n"); 
  decl->gatherScalarVarDecls(decls); 
  //fprintf(stderr, "now %d scalar vardecls\n", decls.size()); 
}


void chillAST_DeclRefExpr::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_DeclRefExpr::gatherArrayVarDecls()\n"); 
  decl->gatherArrayVarDecls(decls); 
  //fprintf(stderr, "now %d Array vardecls\n", decls.size()); 
}


void chillAST_DeclRefExpr::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  refs.push_back(this); 
}

void chillAST_DeclRefExpr::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  refs.push_back(this); 
} 

void chillAST_DeclRefExpr::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_DeclRefExpr::gatherVarUsage()\n"); 
  for (int i=0; i<decls.size(); i++) { 
    if (decls[i] == decl) { 
      //fprintf(stderr, "decl was already there\n");
      return;
    }
    if (streq(declarationName, decls[i]->varname)) { 
      if (streq(declarationType, decls[i]->vartype)) { 
        //fprintf(stderr, "decl was already there\n");
        return;
      }
    }
  }
  chillAST_VarDecl *vd = getVarDecl();  // null for functiondecl
  if (vd) decls.push_back( vd ); 

}




void chillAST_DeclRefExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  //fprintf(stderr, "chillAST_DeclRefExpr::replaceVarDecls()\n"); 
  if (decl == olddecl) { 
    //fprintf(stderr, "replacing old %s with %s\n", olddecl->varname, newdecl->varname);
    //fprintf(stderr, "DRE was "); print(); 
    decl = newdecl; 
    declarationType = strdup(newdecl->vartype); 
    declarationName = strdup(newdecl->varname); 
    //fprintf(stderr, "\nDRE  is "); print(); fprintf(stderr, "\n\n"); 
  }
  else { 
    if (!strcmp(olddecl->varname, declarationName)) { 
      //fprintf(stderr, "uhoh, chillAST_DeclRefExpr::replaceVarDecls()\n"); 
      decl = newdecl; 
      declarationType = strdup(newdecl->vartype); 
      declarationName = strdup(newdecl->varname); 
    }
  }
}

chillAST_VarDecl *chillAST_ImplicitCastExpr::multibase() {
  return subexpr->multibase();
}


chillAST_VarDecl *chillAST_DeclRefExpr::multibase() {
  // presumably, this is being called because this DRE is the base of an ArraySubscriptExpr
  return getVarDecl();
}









void chillAST_VarDecl::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_VarDecl::gatherVarDecls()\n"); 
  for (int i=0; i<decls.size(); i++) { 
    if (decls[i] == this) { 
      //fprintf(stderr, "decl was already there\n");
      return;
    }
    if (streq(decls[i]->varname, varname)) { 
      if (streq(decls[i]->vartype, vartype)) { 
        //fprintf(stderr, "VarDecl (direct) decl was already there\n");
        return;
      }
    }
  }
  decls.push_back( this ); 
}


void chillAST_VarDecl::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_VarDecl::gatherScalarVarDecls(), %s numdimensions %d\n", varname, numdimensions); 

  if (numdimensions != 0) return; // not a scalar
  
  for (int i=0; i<decls.size(); i++) { 
    if (decls[i] == this) { 
      //fprintf(stderr, "decl was already there\n");
      return;
    }

    if (streq(decls[i]->varname, varname)) {      // wrong. scoping.  TODO
      if (streq(decls[i]->vartype, vartype)) { 
        //fprintf(stderr, "VarDecl (direct) decl was already there\n");
        return;
      }
    }
  }
  //fprintf(stderr, "adding vardecl for %s to decls\n", varname); 
  decls.push_back( this ); 
}


void chillAST_VarDecl::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_VarDecl::gatherScalarVarDecls(), %s numdimensions %d\n", varname, numdimensions); 

  if (numdimensions == 0) return; // not an array
  
  for (int i=0; i<decls.size(); i++) { 
    if (decls[i] == this) { 
      //fprintf(stderr, "decl was already there\n");
      return;
    }

    if (streq(decls[i]->varname, varname)) {      // wrong. scoping.  TODO
      if (streq(decls[i]->vartype, vartype)) { 
        //fprintf(stderr, "VarDecl (direct) decl was already there\n");
        return;
      }
    }
  }
  //fprintf(stderr, "adding vardecl for %s to decls\n", varname); 
  decls.push_back( this ); 
}



chillAST_node *chillAST_VarDecl::constantFold() {  return this; }

chillAST_node* chillAST_VarDecl::clone() {
  //fprintf(stderr, "\nchillAST_VarDecl::clone()  cloning vardecl for %s\n", varname); 
  //if (isAParameter) fprintf(stderr, "old vardecl IS a parameter\n");
  //else  fprintf(stderr, "old vardecl IS NOT a parameter\n");

  chillAST_VarDecl *vd  = new chillAST_VarDecl( vartype, strdup(varname), arraypart, parent); 
  vd->typedefinition = typedefinition;
  vd->vardef = vardef;

  vd->underlyingtype = strdup(underlyingtype); 

  vd->arraysizes = NULL;
  vd->numdimensions = numdimensions;

  if (arraypart != NULL && NULL!=arraysizes) {  // !strcmp(arraypart, "")) { 
    //fprintf(stderr, "in chillAST_VarDecl::clone(), cloning the array info\n");
    //fprintf(stderr, "numdimensions %d     arraysizes 0x%x\n", numdimensions, arraysizes) ;
    vd->numdimensions = numdimensions;

    if (arraysizes) { 
      vd->arraysizes = (int *)malloc( sizeof(int *) * numdimensions ); 
      for (int i=0; i< numdimensions; i++) { 
        //fprintf(stderr, "i %d\n", i); 
        vd->arraysizes[i] = arraysizes[i]; 
      }
    }
  }

  vd->knownArraySizes = this->knownArraySizes; 
  vd->isFromSourceFile = isFromSourceFile;
  if (filename) vd->filename = strdup(filename); 
  return vd;
}


void chillAST_VarDecl::splitarraypart() { 
  //fprintf(stderr, "chillAST_VarDecl::splitarraypart()  ");
  //fprintf(stderr, "%p  ", arraypart);
  //if (arraypart) fprintf(stderr, "%s", arraypart); 
  //fprintf(stderr, "\n");

  // split arraypart into  (leading??) asterisks and known sizes [1][2][3]
  if (!arraypart ||  // NULL 
      (arraypart && (*arraypart == '\0'))) { // or empty string

    // parts are both empty string
    if (arraypointerpart) free(arraypointerpart);
    arraypointerpart = strdup("");
    if (arraysetpart) free(arraysetpart);
    arraysetpart = strdup(""); 
    return;
  }

  // arraypart exists and is not empty
  int asteriskcount = 0;
  int fixedcount = 0;
  for ( int i=0; i<strlen(arraypart); i++) {
    if (arraypart[i] == '*') { 
      if (fixedcount) {
        fprintf(stderr, "illegal vardecl arraypart: '%s'\n", arraypart);
        segfault(); 
        exit(-1);
      }
      asteriskcount++;
    }
    else { // remainder is fixed? 
      fixedcount++; 
      // check for brackets and digits only?   TODO
    }
  }
  arraypointerpart = (char *) calloc( asteriskcount+1, sizeof(char));
  arraysetpart     = (char *) calloc( fixedcount+1,    sizeof(char));
  char *ptr = arraypart;
  for ( int i=0; i<asteriskcount; i++)  arraypointerpart[i] = *ptr++;
  for ( int i=0; i<fixedcount; i++)     arraysetpart[i]   = *ptr++;

  //fprintf(stderr, "%s = %s + %s\n", arraypart, arraypointerpart, arraysetpart); 
}






chillAST_IntegerLiteral::chillAST_IntegerLiteral(int val, chillAST_node *par){
  value = val; 
  asttype = CHILLAST_NODETYPE_INTEGERLITERAL;
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}

void chillAST_IntegerLiteral::print( int indent, FILE *fp) {
  chillindent(indent, fp);
  fprintf(fp, "%d", value);
  fflush(fp); 
}

void chillAST_IntegerLiteral::dump( int indent, FILE *fp) {
  chillindent(indent, fp);
  fprintf(fp, "(IntegerLiteral 'int' %d)\n", value);
  fflush(fp); 
}



class chillAST_node* chillAST_IntegerLiteral::constantFold() { return this; } // can never do anything


class chillAST_node* chillAST_IntegerLiteral::clone() { 
  
  chillAST_IntegerLiteral *IL = new  chillAST_IntegerLiteral( value, parent ); 
  IL->isFromSourceFile = isFromSourceFile; 
  if (filename) IL->filename = strdup(filename); 
  return IL; 

}
  
chillAST_FloatingLiteral::chillAST_FloatingLiteral(float val, chillAST_node *par){
  value = val; 
  precision = 1;
  float0double1 = 0; // which is live! 
  allthedigits = NULL; 
  asttype = CHILLAST_NODETYPE_FLOATINGLITERAL;
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(double val, chillAST_node *par){
  doublevalue = val; 
  precision = 2;
  float0double1 = 1; // which is live! 
  allthedigits = NULL; 
  asttype = CHILLAST_NODETYPE_FLOATINGLITERAL;
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(float val, int precis, chillAST_node *par){
  value = val; 
  precision = 1;
  float0double1 = 0; // which is live! 
  precision = precis; // 
  allthedigits = NULL; 
  asttype = CHILLAST_NODETYPE_FLOATINGLITERAL;
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(double val, int precis, chillAST_node *par){
  doublevalue = val; 
  float0double1 = 1; // which is live! 
  precision = precis; // 
  allthedigits = NULL; 
  asttype = CHILLAST_NODETYPE_FLOATINGLITERAL;
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(float val, const char *printthis, chillAST_node *par){
  value = val; 
  float0double1 = 0; // which is live! 
  precision = 1;
  allthedigits = NULL;
  if (printthis) allthedigits = strdup( printthis ); 
  //fprintf(stderr, "\nfloatingliteral allthedigits = '%s'\n", allthedigits); 
  asttype = CHILLAST_NODETYPE_FLOATINGLITERAL;
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(float val, int precis, const char *printthis, chillAST_node *par){
  value = val; 
  float0double1 = 0; // which is live! 
  precision = precis; // but value is a float??  TODO 
  allthedigits = NULL;
  if (printthis) { 
    //fprintf(stderr, "\nchillAST_FloatingLiteral constructor, printthis "); 
    //fprintf(stderr, "%p\n", printthis); 
    allthedigits = strdup( printthis ); 
  }
  //fprintf(stderr, "\nfloatingliteral allthedigits = '%s'\n", allthedigits); 
  asttype = CHILLAST_NODETYPE_FLOATINGLITERAL;
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}


chillAST_FloatingLiteral::chillAST_FloatingLiteral( chillAST_FloatingLiteral *old ) {
  //fprintf(stderr, "chillAST_FloatingLiteral::chillAST_FloatingLiteral( old ) allthedigits %p\n", old->allthedigits); 

  asttype = CHILLAST_NODETYPE_FLOATINGLITERAL;
  value          = old->value;
  doublevalue    = old->doublevalue; 
  float0double1  = old->float0double1;
  allthedigits = NULL;
  if (old->allthedigits) allthedigits = strdup(old->allthedigits); 
  precision      = old->precision;
  isFromSourceFile = true; // default 
  filename = NULL;
}



void chillAST_FloatingLiteral::print( int indent, FILE *fp) {
  chillindent(indent, fp);
  //fprintf(fp, "%f", value);
  // attempt to be more like rose output
  char output[1024]; // warning, hardcoded 

  if (allthedigits != NULL) {
    strcpy(output, allthedigits ); // if they have specified 100 digits of pi, give 'em 100 digits 
    //fprintf(stderr, "floatingliteral allthedigits = '%s'\n", allthedigits); 
  }
  else {
    if (float0double1 == 0)     sprintf(output, "%f", value);
    else sprintf(output, "%f", doublevalue);
    
    // next part to avoid printing 123.4560000000000000000000000000
    char *dot = index(output, '.');
    if (dot) { 
      char *end = output + strlen(output);
      char *onechar;
      char *lastnonzero = dot;
      for (onechar = output; onechar < end; onechar ++) { 
        if (*onechar != '0') lastnonzero = onechar;
      }
      
      if (lastnonzero == dot) 
        lastnonzero[2] = '\0';    // may be after end of string, but that should be OK
      else lastnonzero[1] = '\0'; // may be after end of string, but that should be OK
    }
  }
  if (precision == 1) { 
    int len = strlen(output);
    output[len] = 'f'; // explicit single precision
    output[len+1] = '\0';
  }

  fprintf(fp, "%s", output); 
  fflush(fp); 
}

void chillAST_FloatingLiteral::dump( int indent, FILE *fp) {
  chillindent(indent, fp);
  // 2x2 cases ???
  if (precision == 1) 
    fprintf(fp, "(FloatingLiteral 'float' "); 
  else fprintf(fp, "(FloatingLiteral 'double' "); 

  if (float0double1 == 0) fprintf(fp, "%f)\n", value);  // %f gives enough digits 
  else fprintf(fp, "%f)\n", doublevalue);  // %f gives enough digits 
  fflush(fp); 
}


chillAST_node* chillAST_FloatingLiteral::constantFold() { return this; }; // NOOP

chillAST_node* chillAST_FloatingLiteral::clone() { 
  //fprintf(stderr, "chillAST_FloatingLiteral::clone()  "); 
  //fprintf(stderr, "allthedigits %p \n", allthedigits); 
  chillAST_FloatingLiteral *newone = new  chillAST_FloatingLiteral( this ); 

  newone->isFromSourceFile = isFromSourceFile; 
  if (filename) newone->filename = strdup(filename); 
  //print(); printf("  "); newone->print(); printf("\n"); fflush(stdout); 
  return newone;
}
  
bool chillAST_FloatingLiteral::isSameAs( chillAST_node *other ){
  if (!other->isFloatingLiteral()) return false;
  chillAST_FloatingLiteral *o = (chillAST_FloatingLiteral *)other;
  // should we care about single vs double precision?
  if (float0double1 != o->float0double1) return false;
  if (float0double1 == 0) { 
    return value == o->value; // WARNING, comparing floats with ==
  }
  return doublevalue == o->doublevalue; // WARNING, comparing doubless with ==
} 





chillAST_UnaryOperator::chillAST_UnaryOperator( const char *oper, bool pre, chillAST_node *sub, chillAST_node *par ) { 
  op = strdup(oper);
  prefix = pre;
  subexpr = sub; 
  subexpr->setParent( this );
  asttype = CHILLAST_NODETYPE_UNARYOPERATOR; 
  parent = par; 
  isFromSourceFile = true; // default 
  filename = NULL;
}

void chillAST_UnaryOperator::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*>  &refs, bool w ) {
  subexpr->gatherArrayRefs( refs, isAssignmentOp()); // 
}



void chillAST_UnaryOperator::print( int indent, FILE *fp) {
  bool needparens = false;
  if (subexpr->isNotLeaf()) needparens = true; // may get more complicated

  chillindent( indent, fp); // will this ever be invoked?
  if (prefix) fprintf(fp, "%s", op); 
  if (needparens) fprintf(fp, "(");
  subexpr->print( 0, fp );
  if (needparens) fprintf(fp, ")"); 
  if (!prefix) fprintf(fp, "%s", op); 
  fflush(fp); 
}


void chillAST_UnaryOperator::dump( int indent, FILE *fp) {
  chillindent( indent, fp);
  fprintf(fp, "(UnaryOperator ");
  if (prefix) fprintf(fp, "prefix "); 
  else fprintf(fp, "postfix ");
  fprintf(fp, "%s\n", op);
  subexpr->dump(indent+1, fp); 

  chillindent( indent, fp);
  fprintf(fp, ")\n");
}


void chillAST_UnaryOperator::gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls ) {
  if ((!strcmp("++", op)) || (!strcmp("--", op))) {
    subexpr->gatherVarUsage( decls ); // do all unary modify the subexpr? (no, - ) 
  }
}



chillAST_node* chillAST_UnaryOperator::constantFold() { 
  //fprintf(stderr, "chillAST_UnaryOperator::constantFold() ");
  //print(); fprintf(stderr, "\n"); 

  subexpr = subexpr->constantFold();
  chillAST_node *returnval = this;
  if (subexpr->isConstant()) {
    //fprintf(stderr, "unary op folding constants\n");
    //print(0,stderr); fprintf(stderr, "\n");
    
    if (streq(op, "-")) { 
      if (subexpr->isIntegerLiteral()) {
        int intval = ((chillAST_IntegerLiteral*)subexpr)->value;
        chillAST_IntegerLiteral *I = new chillAST_IntegerLiteral( -intval, parent);
        returnval = I;
        //fprintf(stderr, "integer -%d becomes %d\n", intval, I->value);
      }
      else { 
        chillAST_FloatingLiteral *FL = (chillAST_FloatingLiteral*)subexpr;
        chillAST_FloatingLiteral *F = new chillAST_FloatingLiteral( FL ); // clone
        F->parent = FL->parent;

        F->value = -F->value;
        F->doublevalue = -F->doublevalue;
        
        F->print(); fprintf(stderr, "\n"); 
        
        returnval = F;
      }
    }
    else fprintf(stderr, "can't fold op '%s' yet\n", op); 
  }    
  return returnval;
}


class chillAST_node* chillAST_UnaryOperator::clone() { 
  chillAST_UnaryOperator *UO = new chillAST_UnaryOperator( op, prefix, subexpr->clone(), parent );
  UO->isFromSourceFile = isFromSourceFile; 
  if (filename) UO->filename = strdup(filename); 
  return UO; 
}


void chillAST_UnaryOperator::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarDecls( decls ); 
}


void chillAST_UnaryOperator::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherScalarVarDecls( decls ); 
}


void chillAST_UnaryOperator::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherArrayVarDecls( decls ); 
}


void chillAST_UnaryOperator::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  subexpr->gatherDeclRefExprs( refs ); 
}


void chillAST_UnaryOperator::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarUsage( decls ); 
}

 void chillAST_UnaryOperator::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
   subexpr->replaceVarDecls( olddecl, newdecl ); 
 }


int chillAST_UnaryOperator::evalAsInt() { 
  if (!strcmp("+", op)) return subexpr->evalAsInt();
  if (!strcmp("-", op)) return -subexpr->evalAsInt();
  if (!strcmp("++", op)) return 1 + subexpr->evalAsInt();
  if (!strcmp("--", op)) return subexpr->evalAsInt() - 1;

  fprintf(stderr, "chillAST_UnaryOperator::evalAsInt() unhandled op '%s'\n", op); 
  segfault(); 

}

bool chillAST_UnaryOperator::isSameAs( chillAST_node *other ){
  if (!other->isUnaryOperator()) return false;
  chillAST_UnaryOperator *o = (chillAST_UnaryOperator *)other;
  if (strcmp(op, o->op))  return false; // different operators 
  return subexpr->isSameAs( o->subexpr ); // recurse
}


chillAST_ImplicitCastExpr::chillAST_ImplicitCastExpr( chillAST_node *sub, chillAST_node *par ) {
  subexpr = sub;
  subexpr->setParent( this );
  asttype = CHILLAST_NODETYPE_IMPLICITCASTEXPR; 
  parent = par; 
  //fprintf(stderr, "ImplicitCastExpr 0x%x  has subexpr 0x%x", this, subexpr);
  //fprintf(stderr, " of type %s\n", subexpr->getTypeString()); 
  isFromSourceFile = true; // default 
  filename = NULL;
}

void chillAST_ImplicitCastExpr::print(  int indent, FILE *fp) {
  // No need to print anything, simply forward to the sub expression.
  subexpr->print( indent, fp );
  fflush(fp); 
};

void chillAST_ImplicitCastExpr::printonly(  int indent, FILE *fp) {
  // No need to print anything, simply forward to the sub expression.
  subexpr->printonly( indent, fp );
  fflush(fp); 
};

void chillAST_ImplicitCastExpr::replaceChild( chillAST_node *old, chillAST_node *newchild ){
  if (subexpr == old) { // should be the case for this to get called
    subexpr = newchild;
    subexpr->setParent( this );
    //old->parent = NULL;
    return;
  }

  fprintf(stderr, "chillAST_ImplicitCastExpr::replaceChild() called with bad 'old'\n");
  exit(-1);  // ?? 
}

class chillAST_node* chillAST_ImplicitCastExpr::constantFold() { 
  chillAST_node *child = subexpr->constantFold();
  child->setParent( parent ) ; // remove myself !! probably a bad idea. TODO 
  return child; 
}


class chillAST_node* chillAST_ImplicitCastExpr::clone() { 
  chillAST_ImplicitCastExpr *ICE = new chillAST_ImplicitCastExpr( subexpr->clone(), parent); 
  ICE->isFromSourceFile = isFromSourceFile; 
  if (filename) ICE->filename = strdup(filename); 
  return ICE; 
}


void chillAST_ImplicitCastExpr::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ) {
  subexpr->gatherArrayRefs( refs, w );
}

void chillAST_ImplicitCastExpr::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  subexpr->gatherScalarRefs( refs, writtento );
} 

void chillAST_ImplicitCastExpr::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarDecls( decls ); 
}


void chillAST_ImplicitCastExpr::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherScalarVarDecls( decls ); 
}


void chillAST_ImplicitCastExpr::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherArrayVarDecls( decls ); 
}


void chillAST_ImplicitCastExpr::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  subexpr->gatherDeclRefExprs( refs ); 
}


void chillAST_ImplicitCastExpr::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarUsage( decls ); 
}



chillAST_CStyleCastExpr::chillAST_CStyleCastExpr( const char *to, chillAST_node *sub, chillAST_node *par ) {

  //fprintf(stderr, "chillAST_CStyleCastExpr::chillAST_CStyleCastExpr( %s, ...)\n", to); 
  towhat = strdup(to);
  subexpr = sub;
  subexpr->setParent( this );
  asttype = CHILLAST_NODETYPE_CSTYLECASTEXPR; 
  parent = par; 
  //fprintf(stderr, "chillAST_CStyleCastExpr (%s)   sub 0x%x\n", towhat, sub ); 
  isFromSourceFile = true; // default 
  filename = NULL;
}

void chillAST_CStyleCastExpr::replaceChild( chillAST_node *old, chillAST_node *newchild ){
  if (subexpr == old) { // should be the case for this to get called
    subexpr = newchild;
    subexpr->setParent( this );
    //old->parent = NULL;
    return;
  }

  fprintf(stderr, "chillAST_CStyleCastExpr::replaceChild() called with bad 'old'\n");
  exit(-1);  // ?? 
}

 void chillAST_CStyleCastExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
   subexpr->replaceVarDecls( olddecl, newdecl);
 }

void chillAST_CStyleCastExpr::print(  int indent, FILE *fp) {
  //fprintf(stderr, "CStyleCastExpr::print()\n"); 
  chillindent(indent, fp); 

  // special cases? should probably walk the AST and change the literal itself
  if ( !strcmp("float", towhat)  && subexpr->isIntegerLiteral()) { // (float) 3 => 3.0f 
    subexpr->print( 0, fp ); fprintf(fp, ".0f");
  }
  else if ( !strcmp("double", towhat)  && subexpr->isIntegerLiteral()) { // (double) 3 => 3.0
    subexpr->print( 0, fp ); fprintf(fp, ".0");
  }
  else if ( !strcmp("float", towhat)  && subexpr->isFloatingLiteral()) { // (float) 3.0 => 3.0f 
    subexpr->print( 0, fp ); fprintf(fp, "f");
  }
  else { // general case 
    fprintf(fp, "((%s) ", towhat); 
    //fprintf(fp, "\ntowhat '%s'\n", towhat ); 
    
    if (subexpr->isVarDecl()) fprintf(fp, "%s", ((chillAST_VarDecl *)subexpr)->varname); 
    else subexpr->print( indent, fp );
    //fprintf(fp, "subexpr '%s' ", subexpr->getTypeString()); 
    fprintf(fp, ")"); 
  }
  fflush(fp); 
};


void chillAST_CStyleCastExpr::dump( int indent, FILE *fp) {
  chillindent(indent, fp);
  fprintf(fp, "(CStyleCastExpr  (%s) \n", towhat);
  subexpr->dump( indent+1, fp );
  chillindent(indent, fp);
  fprintf(fp, ")\n");
  fflush(fp); 
}

class chillAST_node* chillAST_CStyleCastExpr::constantFold() { 
  subexpr = subexpr->constantFold();
  return this; 
}


class chillAST_node* chillAST_CStyleCastExpr::clone() { 
  chillAST_CStyleCastExpr *CSCE = new chillAST_CStyleCastExpr( towhat, subexpr->clone(), parent ); 
  CSCE->isFromSourceFile = isFromSourceFile; 
  if (filename) CSCE->filename = strdup(filename); 
  return CSCE;
}

void chillAST_CStyleCastExpr::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ) {
  subexpr->gatherArrayRefs( refs, w );
}

void chillAST_CStyleCastExpr::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  subexpr->gatherScalarRefs( refs, writtento );
} 


void chillAST_CStyleCastExpr::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarDecls( decls ); 
}


void chillAST_CStyleCastExpr::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherScalarVarDecls( decls ); 
}


void chillAST_CStyleCastExpr::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherArrayVarDecls( decls ); 
}


void chillAST_CStyleCastExpr::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  subexpr->gatherDeclRefExprs( refs ); 
}


void chillAST_CStyleCastExpr::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarUsage( decls ); 
}




chillAST_CStyleAddressOf::chillAST_CStyleAddressOf( chillAST_node *sub, chillAST_node *par ) {
  subexpr = sub;
  subexpr->setParent( this );
  asttype = CHILLAST_NODETYPE_CSTYLEADDRESSOF; 
  parent = par; 
  //fprintf(stderr, "chillAST_CStyleCastExpr (%s)   sub 0x%x\n", towhat, sub ); 
  isFromSourceFile = true; // default 
  filename = NULL;
}

void chillAST_CStyleAddressOf::print(  int indent, FILE *fp) {
  //fprintf(stderr, "CStyleAddressOf::print()\n"); 
  chillindent(indent, fp); 
  fprintf(fp, "(&"); 
  subexpr->print( 0, fp );
  fprintf(fp, ")"); 
  fflush(fp); 
};

void chillAST_CStyleAddressOf::dump( int indent, FILE *fp) {
  chillindent(indent, fp);
  fprintf(fp, "(CStyleAddressOf \n");
  subexpr->print( indent+1, fp );
  chillindent(indent, fp);
  fprintf(fp, ")\n");
  fflush(fp); 
}

class chillAST_node* chillAST_CStyleAddressOf::constantFold() { 
  subexpr = subexpr->constantFold();
  return this; 
}

class chillAST_node* chillAST_CStyleAddressOf::clone() { 
  chillAST_CStyleAddressOf *CSAO = new chillAST_CStyleAddressOf( subexpr->clone(), parent ); 
  CSAO->isFromSourceFile = isFromSourceFile; 
  if (filename) CSAO->filename = strdup(filename); 
  return CSAO;
}

void chillAST_CStyleAddressOf::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ) {
  subexpr->gatherArrayRefs( refs, w );
}

void chillAST_CStyleAddressOf::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  subexpr->gatherScalarRefs( refs, writtento );
} 

void chillAST_CStyleAddressOf::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarDecls( decls ); 
}

void chillAST_CStyleAddressOf::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherScalarVarDecls( decls ); 
}


void chillAST_CStyleAddressOf::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherArrayVarDecls( decls ); 
}


void chillAST_CStyleAddressOf::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  subexpr->gatherDeclRefExprs( refs ); 
}


void chillAST_CStyleAddressOf::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarUsage( decls ); 
}




chillAST_Malloc::chillAST_Malloc(chillAST_node *size, chillAST_node *p) {
  thing = NULL;
  sizeexpr = size;  // probably a multiply like   sizeof(int) * 1024
  asttype = CHILLAST_NODETYPE_MALLOC;
  parent = p;
  isFromSourceFile = true; // default 
  filename = NULL;
};  

chillAST_Malloc::chillAST_Malloc(char *thething, chillAST_node *numthings, chillAST_node *p) {
  thing = strdup(thething);   // "int" or "float" or "struct widget"
  sizeexpr = numthings;  
  asttype = CHILLAST_NODETYPE_MALLOC;
  parent = p;
  isFromSourceFile = true; // default 
  filename = NULL;
};  

chillAST_node* chillAST_Malloc::constantFold() {
  sizeexpr->constantFold(); 
}

chillAST_node* chillAST_Malloc::clone() { 
  chillAST_Malloc *M = new chillAST_Malloc( thing, sizeexpr, parent); // the general version 
  M->isFromSourceFile = isFromSourceFile; 
  if (filename) M->filename = strdup(filename); 
  return M;
}; 

void chillAST_Malloc::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {
  sizeexpr->gatherArrayRefs( refs, writtento );
};


void chillAST_Malloc::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  sizeexpr->gatherScalarRefs( refs, writtento );
};

void chillAST_Malloc::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  sizeexpr->gatherVarDecls(decls); 
};

void chillAST_Malloc::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ){
  sizeexpr->gatherScalarVarDecls(decls); 
};

void chillAST_Malloc::gatherArrayVarDecls ( vector<chillAST_VarDecl*> &decls ) {
  sizeexpr->gatherArrayVarDecls(decls); 
};

void chillAST_Malloc::gatherVarUsage( vector<chillAST_VarDecl*> &decls ){
  sizeexpr->gatherVarUsage(decls); 
};



void chillAST_Malloc::print( int indent,  FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "malloc("); 
  
  if (thing) {
    fprintf(fp, " sizeof(%s) * ", thing );
  }
  sizeexpr->print(0,fp);
  fprintf(fp, ")");
  fflush(fp); 
};  


void chillAST_Malloc::dump(  int indent,  FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "(Malloc \n"); 
  sizeexpr->dump( indent+1, fp );
  chillindent(indent, fp); 
  fprintf(fp, ")\n");
  fflush(fp); 
};



chillAST_CudaMalloc::chillAST_CudaMalloc(chillAST_node *devmemptr, chillAST_node *size, chillAST_node *p) {
  devPtr = devmemptr; 
  sizeinbytes = size;  // probably a multiply like   sizeof(int) * 1024
  asttype = CHILLAST_NODETYPE_CUDAMALLOC;
  parent = p;
  isFromSourceFile = true; // default 
  filename = NULL;
};  

void chillAST_CudaMalloc::print(  int indent,  FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "cudaMalloc("); 
  devPtr->print( 0, fp );
  fprintf(fp, ",");
  sizeinbytes->print( 0, fp );
  fprintf(fp, ")");
  fflush(fp); 
};

void chillAST_CudaMalloc::dump(  int indent,  FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "(CudaMalloc \n"); 
  devPtr->dump( indent+1, fp );
  fprintf(fp, "\n");
  sizeinbytes->dump( indent+1, fp );
  fprintf(fp, ")\n");
  fflush(fp); 
};

class chillAST_node* chillAST_CudaMalloc::constantFold() { 
  devPtr = devPtr->constantFold();
  return this; 
}

class chillAST_node* chillAST_CudaMalloc::clone() { 
  chillAST_CudaMalloc *CM = new chillAST_CudaMalloc( devPtr->clone(), sizeinbytes->clone(), parent ); 
  CM->isFromSourceFile = isFromSourceFile; 
  if (filename) CM->filename = strdup(filename); 
  return CM; 
}

void chillAST_CudaMalloc::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ) {
  devPtr->gatherArrayRefs( refs, false );
  sizeinbytes->gatherArrayRefs( refs, false );
}

void chillAST_CudaMalloc::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  devPtr->gatherScalarRefs( refs, false );
  sizeinbytes->gatherScalarRefs( refs, false );
}

void chillAST_CudaMalloc::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  devPtr->gatherVarDecls( decls ); 
  sizeinbytes->gatherVarDecls( decls ); 
}


void chillAST_CudaMalloc::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  devPtr->gatherScalarVarDecls( decls ); 
  sizeinbytes->gatherScalarVarDecls( decls ); 
}



void chillAST_CudaMalloc::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  devPtr->gatherArrayVarDecls( decls ); 
  sizeinbytes->gatherArrayVarDecls( decls ); 
}



void chillAST_CudaMalloc::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  devPtr->gatherVarUsage( decls ); 
  sizeinbytes->gatherVarUsage( decls ); 
}



chillAST_CudaFree::chillAST_CudaFree(chillAST_VarDecl *var, chillAST_node *p) {
  variable = var; 
  parent = p;
  asttype = CHILLAST_NODETYPE_CUDAFREE;
  isFromSourceFile = true; // default 
  filename = NULL;
};  

void chillAST_CudaFree::print(  int indent,  FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "cudaFree(%s)", variable->varname); 
  fflush(fp); 
};

void chillAST_CudaFree::dump(  int indent,  FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "(CudaFree %s )\n", variable->varname); 
  fflush(fp); 
};

class chillAST_node* chillAST_CudaFree::constantFold() { 
  return this; 
}

class chillAST_node* chillAST_CudaFree::clone() { 
  chillAST_CudaFree *CF = new chillAST_CudaFree( variable, parent ); 
  CF->isFromSourceFile = isFromSourceFile; 
  if (filename) CF->filename = strdup(filename); 
  return CF; 
}

void chillAST_CudaFree::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ) {}
void chillAST_CudaFree::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {}

void chillAST_CudaFree::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  variable->gatherVarDecls( decls ); 
}


void chillAST_CudaFree::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  variable->gatherScalarVarDecls( decls ); 
}


void chillAST_CudaFree::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  variable->gatherArrayVarDecls( decls ); 
}



void chillAST_CudaFree::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  variable->gatherVarUsage( decls ); 
}









chillAST_CudaMemcpy::chillAST_CudaMemcpy(chillAST_VarDecl *d, chillAST_VarDecl *s, chillAST_node *siz, char *kind, chillAST_node *par) { 
  dest = d;
  src = s;
  //fprintf(stderr, "chillAST_CudaMemcpy::chillAST_CudaMemcpy( dest %s, src %s, ...)\n", d->varname, s->varname ); 
  size = siz;
  cudaMemcpyKind = kind;
  asttype = CHILLAST_NODETYPE_CUDAMEMCPY;
  isFromSourceFile = true; // default 
  filename = NULL;
  parent = par;
}; 

void chillAST_CudaMemcpy::print(  int indent,  FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "cudaMemcpy(%s,%s,", dest->varname, src->varname); 
  //dest->print( 0, fp );
  //fprintf(fp, ",");
  // src->print( 0, fp );  just want the src NAME, not name and array info 
  //fprintf(fp, ",");
  size->print( 0, fp );
  fprintf(fp, ",%s)", cudaMemcpyKind);
  fflush(fp); 
};

void chillAST_CudaMemcpy::dump(  int indent,  FILE *fp ) {
  chillindent(indent, fp); 
  fprintf(fp, "(CudaMemcpy \n"); 
  dest->dump( indent+1, fp );
  src->dump( indent+1, fp );
  size->dump( indent+1, fp );
  chillindent(indent+1, fp); 
  fprintf(fp, ",%s\n", cudaMemcpyKind);
  fflush(fp); 
};

class chillAST_node* chillAST_CudaMemcpy::constantFold() { 
  dest = (chillAST_VarDecl *)dest->constantFold();
  src  = (chillAST_VarDecl *)src->constantFold();
  size = size->constantFold();
  return this; 
}

class chillAST_node* chillAST_CudaMemcpy::clone() { 
  chillAST_CudaMemcpy *CMCPY = new chillAST_CudaMemcpy((chillAST_VarDecl *)(dest->clone()),(chillAST_VarDecl *)(src->clone()), size->clone(), strdup(cudaMemcpyKind), parent ); 
  CMCPY->isFromSourceFile = isFromSourceFile; 
  if (filename) CMCPY->filename = strdup(filename); 
  return CMCPY;
}

void chillAST_CudaMemcpy::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ) {
  dest->gatherArrayRefs( refs, false );
  src ->gatherArrayRefs( refs, false );
  size->gatherArrayRefs( refs, false );
}

void chillAST_CudaMemcpy::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  dest->gatherScalarRefs( refs, false );
  src ->gatherScalarRefs( refs, false );
  size->gatherScalarRefs( refs, false );
} 

void chillAST_CudaMemcpy::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  dest->gatherVarDecls( decls ); 
  src ->gatherVarDecls( decls ); 
  size->gatherVarDecls( decls ); 
}


void chillAST_CudaMemcpy::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  dest->gatherScalarVarDecls( decls ); 
  src ->gatherScalarVarDecls( decls ); 
  size->gatherScalarVarDecls( decls ); 
}


void chillAST_CudaMemcpy::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  dest->gatherArrayVarDecls( decls ); 
  src ->gatherArrayVarDecls( decls ); 
  size->gatherArrayVarDecls( decls ); 
}


void chillAST_CudaMemcpy::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  dest->gatherVarUsage( decls ); 
  src ->gatherVarUsage( decls ); 
  size->gatherVarUsage( decls ); 
}



chillAST_CudaSyncthreads::chillAST_CudaSyncthreads( chillAST_node *par) { 
  asttype = CHILLAST_NODETYPE_CUDASYNCTHREADS;
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
} 
 
 void chillAST_CudaSyncthreads::print( int indent,  FILE *fp ) {
   chillindent(indent, fp); 
   fprintf(fp, "__syncthreads()"); 
   fflush(fp); 
 }
 
 void chillAST_CudaSyncthreads::dump( int indent,  FILE *fp ) {
   chillindent(indent, fp); 
   fprintf(fp, "(syncthreads)\n"); 
   fflush(fp); 
 }
 









chillAST_ReturnStmt::chillAST_ReturnStmt( chillAST_node *retval, chillAST_node *par ) {
  asttype = CHILLAST_NODETYPE_RETURNSTMT; 
  returnvalue = retval;
  if (returnvalue) returnvalue->setParent( this );
  parent = par; 
  isFromSourceFile = true; // default 
  filename = NULL;
}


void chillAST_ReturnStmt::print( int indent, FILE *fp) {
  printPreprocBEFORE(indent, fp); 
  chillindent(indent, fp);
  if (returnvalue != NULL) {
    fprintf(fp, "return(");
    returnvalue->print( 0, fp );
    fprintf(fp, ")" ); // parent will add ";\n" ?? 
  }
  else { 
    fprintf(fp, "return");
  }
  fflush(fp); 
}


void chillAST_ReturnStmt::dump( int indent, FILE *fp) {
  chillindent(indent, fp);
  fprintf(fp, "(ReturnStmt");
  if (returnvalue) {
    fprintf(fp, "\n");
    returnvalue->dump(indent+1,fp);
    chillindent(indent, fp);
  }
  fprintf(fp, ")\n"); 
}


class chillAST_node* chillAST_ReturnStmt::constantFold() { 
  if (returnvalue) returnvalue = returnvalue->constantFold(); 
  return this;
}



class chillAST_node* chillAST_ReturnStmt::clone() { 
  chillAST_node *val = NULL; 
  if ( returnvalue ) val = returnvalue->clone();
  chillAST_ReturnStmt *RS = new chillAST_ReturnStmt( val, parent );
  RS->isFromSourceFile = isFromSourceFile; 
  if (filename) RS->filename = strdup(filename); 
  return RS;
}


void chillAST_ReturnStmt::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  if (returnvalue) returnvalue->gatherVarDecls( decls ); 
}


void chillAST_ReturnStmt::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  if (returnvalue) returnvalue->gatherScalarVarDecls( decls ); 
}


void chillAST_ReturnStmt::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  if (returnvalue) returnvalue->gatherArrayVarDecls( decls ); 
}



void chillAST_ReturnStmt::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  if (returnvalue) returnvalue->gatherDeclRefExprs( refs ); 
}



void chillAST_ReturnStmt::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  if (returnvalue) returnvalue->gatherVarUsage( decls ); 
}




chillAST_CallExpr::chillAST_CallExpr(chillAST_node *c, chillAST_node *par) { //, int numofargs, chillAST_node **theargs ) {
  
  //fprintf(stderr, "chillAST_CallExpr::chillAST_CallExpr  callee type %s\n", c->getTypeString()); 
  asttype = CHILLAST_NODETYPE_CALLEXPR;
  callee = c;
  //callee->setParent( this ); // ?? 
  numargs = 0;
  parent = par; 
  grid = block = NULL;
  isFromSourceFile = true; // default 
  filename = NULL;
}


void chillAST_CallExpr::addArg( chillAST_node *a ) {
  args.push_back( a );
  a->setParent( this );
  numargs += 1;
}


void chillAST_CallExpr::print(  int indent, FILE *fp) {
  printPreprocBEFORE(indent, fp); 
  chillindent(indent, fp);
  chillAST_FunctionDecl *FD = NULL;
  chillAST_MacroDefinition *MD = NULL;

  if (callee->isDeclRefExpr()) { 
    chillAST_DeclRefExpr *DRE = (chillAST_DeclRefExpr *) callee; 
    //fprintf(stderr, "DRE decl is 0x%x\n", DRE->decl); 
    if (!DRE->decl) { 
      // a macro? 
      fprintf(fp, "%s ", DRE->declarationName); 
      return; // ?? 
    }

    //fprintf(stderr, "DRE decl of type %s\n", DRE->decl->getTypeString()); 
    if ( (DRE->decl)->isFunctionDecl()) FD = (chillAST_FunctionDecl *)DRE->decl; 
    else { 
      fprintf(stderr, "chillAST_CallExpr::print() DRE decl of type %s\n", DRE->decl->getTypeString()); 
      exit(-1);
    }
  }
  else if (callee->isFunctionDecl()) FD = (chillAST_FunctionDecl *) callee;
  else if (callee->isMacroDefinition()) { 
    MD = (chillAST_MacroDefinition *) callee;
    fprintf(fp, "%s(", MD->macroName); 
  }
  else { 
    fprintf(stderr, "\nchillAST_CallExpr::print() callee of unhandled type %s\n", callee->getTypeString()); 
    callee->dump();
    exit(-1); 
  }
  
  if (FD) { 
    fprintf(fp, "%s", FD->functionName ); fflush(fp); 
    if (grid && block) {
      fprintf(fp, "<<<%s,%s>>>(", grid->varname, block->varname);    // a
    }
    else fprintf(fp, "(");
  }


  //callee->print( indent, fp);
  for (int i=0; i<args.size(); i++) { 
    if (i!=0) fprintf(fp, ", "); 
    args[i]->print(0, fp); 
  }
  fprintf(fp, ")");                                                //a
  fflush(fp); 
}

void chillAST_CallExpr::dump(  int indent, FILE *fp) {
  chillindent(indent, fp);
  fprintf(fp, "(CallExpr ");
  //fprintf(stderr, "callee type %s\n", callee->getTypeString()); 
  chillAST_FunctionDecl *fd = NULL;
  if (callee->isDeclRefExpr()) { // always?
    chillAST_DeclRefExpr *dre = (chillAST_DeclRefExpr *)callee;
    fd = dre->getFunctionDecl(); // if NULL, we've got a Vardecl instead
    if (fd) {
      //fd->print(); 
      fprintf(fp, "%s\n", fd->returnType);
    }

    callee->dump(indent+1, fp);
    if (fd) { 
      int numparams = fd->parameters.size();
      for (int i=0; i<numparams; i++) fd->parameters[i]->dump(indent+1, fp);
    }
  }
  chillindent(indent, fp);
  fprintf(fp, ")\n");
}

void chillAST_CallExpr::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {
  for (int i=0; i<args.size(); i++) { 
    args[i]->gatherArrayRefs( refs, writtento ); 
  }
}
void chillAST_CallExpr::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  for (int i=0; i<args.size(); i++) { 
    args[i]->gatherScalarRefs( refs, writtento ); 
  }
} 


void chillAST_CallExpr::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<args.size(); i++) { 
    args[i]->gatherVarDecls( decls ); 
  }
}


void chillAST_CallExpr::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<args.size(); i++) { 
    args[i]->gatherScalarVarDecls( decls ); 
  }
}


void chillAST_CallExpr::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<args.size(); i++) { 
    args[i]->gatherArrayVarDecls( decls ); 
  }
}


void chillAST_CallExpr::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  for (int i=0; i<args.size(); i++) { 
    args[i]->gatherDeclRefExprs( refs ); 
  }
}

void chillAST_CallExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  for (int i=0; i<args.size(); i++) args[i]->replaceVarDecls( olddecl, newdecl ); 
}

void chillAST_CallExpr::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<args.size(); i++) { 
    args[i]->gatherVarUsage( decls ); 
  }
}


chillAST_node* chillAST_CallExpr::constantFold() { 
  numargs = args.size(); // wrong place for this 
  for (int i=0; i<numargs; i++) { 
    args[i] = args[i]->constantFold(); 
  }
  return this; 
}

chillAST_node* chillAST_CallExpr::clone() { 
  //fprintf(stderr, "chillAST_CallExpr::clone()\n");
  //print(0, stderr); fprintf(stderr, "\n"); 

  chillAST_CallExpr *CE = new chillAST_CallExpr( callee->clone(), NULL );
  for (int i=0; i<args.size(); i++) CE->addArg( args[i]->clone() ); 
  CE->isFromSourceFile = isFromSourceFile; 
  if (filename) CE->filename = strdup(filename); 
  return CE; 
}




chillAST_VarDecl::chillAST_VarDecl() { 
  //fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl()  %p\n", this); 
  vartype = underlyingtype = varname = arraypart = arraypointerpart = arraysetpart = NULL;
  typedefinition = NULL; 

  //fprintf(stderr, "setting underlying type NULL\n" ); 
  init = NULL;
  numdimensions=0; arraysizes = NULL; 
  asttype = CHILLAST_NODETYPE_VARDECL;  // 
  parent = NULL;
  metacomment = NULL;

  vardef  = NULL;
  isStruct = false; 
  isAParameter = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // fprintf(stderr, "RDS = false\n"); 
  knownArraySizes = false;
  isFromSourceFile = true; // default 
  filename = NULL;
}; 



chillAST_VarDecl::chillAST_VarDecl( const char *t,  const char *n, const char *a, chillAST_node *par) { 
  //fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl( type %s, name %s, arraypart %s,  parent %p)  %p\n", t, n, a, par, this); 
  vartype   = strdup(t); 
  typedefinition = NULL;

  underlyingtype = parseUnderlyingType( vartype ); 
  //fprintf(stderr, "setting underlying type %s from %s\n",  underlyingtype, vartype ); 
  varname   = strdup(n); 
  arraypointerpart = arraysetpart = NULL;
  if (a) arraypart = strdup(a);
  else arraypart = strdup(""); 
  splitarraypart();

  init = NULL;
  numdimensions=0; arraysizes = NULL; 
  uniquePtr = NULL;
  asttype = CHILLAST_NODETYPE_VARDECL;
  parent = par;



  knownArraySizes = false; 
  //fprintf(stderr, "arraypart len %d\n", strlen(a)); 
  for (int i=0; i<strlen(a); i++) { 
    if (a[i] == '[') { numdimensions++; knownArraySizes = true; } 
    if (!knownArraySizes && a[i] == '*') numdimensions++;
  }
  
  vardef  = NULL;
  isStruct = false; 
  isAParameter = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // fprintf(stderr, "RDS = false\n"); 

  if (parent) { 
    //fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl(), adding to symbol table???\n"); 
    parent->addVariableToSymbolTable( this ); // should percolate up until something has a symbol table 
    
  }
  isFromSourceFile = true; // default 
  filename = NULL;
}; 



chillAST_VarDecl::chillAST_VarDecl( chillAST_RecordDecl *astruct, const char *nam, const char *array, chillAST_node *par) { 
  //fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl( %s  %p struct ", nam, this );
  const char *type = astruct->getName(); 
  //fprintf (stderr, "%s, name %s, arraypart %s parent ) %p\n", type, nam, array, this); // , par);

  vartype = strdup(type);

  // these always go together  ?? 
  vardef  = astruct;// pointer to the thing that says what is inside the struct
  isStruct = true;  // ?? wrong if it's a union  ?? TODO 
  //fprintf(stderr, "setting vardef of %s to %p\n", nam, vardef); 
  
  underlyingtype = parseUnderlyingType( vartype ); 
  //fprintf(stderr, "setting underlying type %s from %s\n",  underlyingtype, vartype ); 
  varname   = strdup(nam); 
  arraypart = strdup(array);
  arraypointerpart = arraysetpart = NULL;
  splitarraypart(); 

  init = NULL;
  numdimensions=0; arraysizes = NULL; 
  uniquePtr = NULL;
  asttype = CHILLAST_NODETYPE_VARDECL;
  parent = par;

  knownArraySizes = false; 
  //fprintf(stderr, "arraypart len %d\n", strlen(a)); 
  for (int i=0; i<strlen(array); i++) { 
    if (array[i] == '[') { numdimensions++; knownArraySizes = true; } 
    if (!knownArraySizes && array[i] == '*') numdimensions++;
  }
  
  isAParameter = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // fprintf(stderr, "RDS = false\n"); 
  typedefinition = NULL;

  //fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl( chillAST_RecordDecl *astruct, ...) MIGHT add struct to some symbol table\n"); 
  //if (parent) fprintf(stderr, "yep, adding it\n"); 

  if (parent) parent->addVariableToSymbolTable( this ); // should percolate up until something has a symbol table 

  isFromSourceFile = true; // default 
  filename = NULL;

}; 





chillAST_VarDecl::chillAST_VarDecl( chillAST_TypedefDecl *tdd,  const char *n, const char *a, chillAST_node *par) { 
  //fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl( %s  typedef ", n);
  const char *type = tdd->getStructName();
  //fprintf (stderr, "%s, name %s, arraypart %s parent ) %p\n", type, n, a,this); // , par);
  typedefinition = tdd;
  vartype   = strdup(type); 
  underlyingtype = parseUnderlyingType( vartype ); 
  //fprintf(stderr, "setting underlying type %s from %s\n",  underlyingtype, vartype ); 
  varname   = strdup(n); 
  arraypart = strdup(a);
  arraypointerpart = arraysetpart = NULL;
  splitarraypart(); 

  init = NULL;
  numdimensions=0; arraysizes = NULL; 
  uniquePtr = NULL;
  asttype = CHILLAST_NODETYPE_VARDECL;
  parent = par;

  knownArraySizes = false; 
  //fprintf(stderr, "arraypart len %d\n", strlen(a)); 
  for (int i=0; i<strlen(a); i++) { 
    if (a[i] == '[') { numdimensions++; knownArraySizes = true; } 
    if (!knownArraySizes && a[i] == '*') numdimensions++;
  }

  isStruct = tdd->isAStruct();
  
  vardef  = NULL;
  isAParameter = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // //fprintf(stderr, "RDS = false\n"); 
  if (parent) parent->addVariableToSymbolTable( this ); // should percolate up until something has a symbol table 
  isFromSourceFile = true; // default 
  filename = NULL;
}; 





chillAST_VarDecl::chillAST_VarDecl( const char *t,  const char *n, const char *a, void *ptr, chillAST_node *par) { 
  fprintf(stderr, "2chillAST_VarDecl::chillAST_VarDecl( type %s, name %s, arraypart '%s' ) %p\n", t, n, a, this); 
  //fprintf(stderr, "2chillAST_VarDecl::chillAST_VarDecl( type %s, name %s, arraypart %s, ptr 0x%x, parent 0x%x )\n", t, n, a, ptr, par); 


  vartype   = strdup(t); 
  typedefinition = NULL;
  underlyingtype = parseUnderlyingType( vartype ); 
  //fprintf(stderr, "setting underlying type %s from %s\n",  underlyingtype, vartype ); 
  varname   = strdup(n); 
  vardef = NULL;  // not a struct

  if (a) arraypart = strdup(a);
  else arraypart = strdup(""); // should catch this earlier
  arraypointerpart = arraysetpart = NULL;
  splitarraypart(); 

  init = NULL;
  numdimensions=0; arraysizes = NULL; 
  uniquePtr = ptr;
  asttype = CHILLAST_NODETYPE_VARDECL;
  parent = par;
  knownArraySizes = false; 

  //fprintf(stderr, "name arraypart len %d\n", strlen(a)); 
  //fprintf(stderr, "arraypart '%s'\n", arraypart); 
  for (int i=0; i<strlen(a); i++) { 
    if (a[i] == '[') { numdimensions++; knownArraySizes = true; } 
    if (!knownArraySizes && a[i] == '*') numdimensions++; // fails for  a[4000 * 4] 
  }  
  //if (0 == strlen(a) && numdimensions == 0) { 
  //  for (int i=0; i<strlen(t); i++) {   // handle float * x 
  //    if (t[i] == '[') numdimensions++;
  //    if (t[i] == '*') numdimensions++;
  //  }  
  //} 
  //fprintf(stderr, "2name %s numdimensions %d\n", n, numdimensions); 




  // this is from ir_clang.cc ConvertVarDecl(), that got executed AFTER the vardecl was constructed. dumb
  int numdim = 0;
  //knownArraySizes = true;
  //if (index(vartype, '*')) knownArraySizes = false;  // float *a;   for example
  //if (index(arraypart, '*'))  knownArraySizes = false;
  
  // note: vartype here, arraypart in next code..    is that right?
  if (index(vartype, '*')) { 
    for (int i = 0; i<strlen(vartype); i++) if (vartype[i] == '*') numdim++;
    //fprintf(stderr, "numd %d\n", numd);
    numdimensions = numdim; 
  }
  
  if (index(arraypart, '[')) {  // JUST [12][34][56]  no asterisks
    char *dupe = strdup(arraypart);

    int len = strlen(arraypart);
    for (int i=0; i<len; i++) if (dupe[i] == '[') numdim++;
    
    //fprintf(stderr, "numdim %d\n", numdim);
    
    numdimensions = numdim; 
    int *as =  (int *)malloc(sizeof(int *) * numdim );
    if (!as) { 
      fprintf(stderr, "can't malloc array sizes in ConvertVarDecl()\n");
      exit(-1);
    }
    arraysizes = as; // 'as' changed later!
    
    
    char *ptr = dupe;
    //fprintf(stderr, "dupe '%s'\n", ptr);
    while (ptr = index(ptr, '[')) {                   // this fails for float a[4000*4]
      ptr++;
      char *leak = strdup(ptr);
      char *close = index(leak, ']');
      if (close) *close = '\0'; 

      int l = strlen(leak);
      bool justdigits = true;
      bool justmath = true;
      for (int i=0; i<l; i++) { 
        char c = leak[i]; 
        if (!isdigit(c)) justdigits = false;
        if (!( isdigit(c) ||
               isblank(c) ||
               ((c == '+') || (c == '*')  || (c == '*')  || (c == '*')) || // math
               ((c == '(') || (c == ')')))
               ) { 
          //fprintf(stderr, " not justmath because '%c'\n", c); 
          justmath = false; 
        }
            
      }

      //fprintf(stderr, "tmp '%s'\n", leak);
      if (justdigits) { 
        int dim;
        sscanf(ptr, "%d", &dim);
        //fprintf(stderr, "dim %d\n", dim);
        *as++ = dim; 
      }
      else { 
        if (justmath) fprintf(stderr, "JUST MATH\n");
        fprintf(stderr, "need to evaluate %s, faking with hardcoded 16000\n", leak); 
        *as++ = 16000; // temp TODO DFL 
      }
      free (leak); 

      ptr =  index(ptr, ']');
      //fprintf(stderr, "bottom of loop, ptr = '%s'\n", ptr); 
    }
    free(dupe);
    //for (int i=0; i<numdim; i++) { 
    //  fprintf(stderr, "dimension %d = %d\n", i,  arraysizes[i]); 
    //} 
    
    //fprintf(stderr, "need to handle [] array to determine num dimensions\n");
    //exit(-1); 
  }
  
  isAParameter = false; 
  isStruct = false;
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // fprintf(stderr, "RDS = false\n"); 
  
  //print(); printf("\n"); fflush(stdout); 

  // currently this is bad, because a struct does not have a symbol table, so the 
  // members of a struct are passed up to the func or sourcefile. 
  if (parent) parent->addVariableToSymbolTable( this ); // should percolate up until something has a symbol table 


  //fprintf(stderr, "2chillAST_VarDecl::chillAST_VarDecl LEAVING\n"); 
  isFromSourceFile = true; // default 
  filename = NULL;
}; 


void chillAST_VarDecl::print( int indent, FILE *fp ) {
  printPreprocBEFORE(indent, fp); 

  //fprintf(fp, "VarDecl vartype '%s'    varname %s   ", vartype, varname); 
  //if (isAStruct()) fprintf(fp, "isAStruct()\n");
  //else  fprintf(fp, "NOT A Struct\n");

  // fprintf(fp, "\n");  fflush(fp); dump(0,fp); fflush(fp);  // debug

  chillindent(indent, fp);
  //fprintf(fp, "vardecl->print  vartype '%s'\n", vartype); 
  if (isDevice) fprintf(fp, "__device__ "); 
  if (isShared) fprintf(fp, "__shared__ "); 
  
  //if (isAStruct()) fprintf(fp, "/* isAStruct() */  ");
  //else fprintf(fp, "/* NOT A Struct() */  ");
  //if (vardef)      fprintf(fp, "/* vardef */  "); 
  //else  fprintf(fp, "/* NOT vardef */  "); 


  // this logic is probably wrong (what about pointer to struct? )
  if (isAStruct() && vardef) { // an unnamed  struct used only here ?? 
    //fprintf(stderr, "isAStruct() && vardef ?? \n");
    // print the internals of the struct and then the name 
    vardef->printStructure( 0, fp );
    fprintf(fp, "%s", varname ); 
    return;
  }
  
  
  if (typedefinition && typedefinition->isAStruct()) fprintf(fp, "struct "); 


  if (isAParameter) { 
    //fprintf(fp, "(param) nd %d", numdimensions ); 
    //dump(); 
    if (numdimensions > 0) {
      if (knownArraySizes) {  // just [12][34][56] 
        fprintf(fp, "%s ", vartype);
        if (byreference) fprintf(fp, "&");
        fprintf(fp, "%s", varname);
        for (int n=0; n< (numdimensions); n++) fprintf(fp, "[%d]", arraysizes[n]); 
      }
      else {  // some unknown array part    float *a;   or float (*)a[1234] 

        //fprintf(fp, "\nsome unknown\n"); 
        if (numdimensions == 1) { 
          //fprintf(fp, "\nnd1, vartype %s\n", vartype); 
          
          // TODO this if means I have probably made a mistake somewhere
          if (!index(vartype, '*')) fprintf(fp, "%s *%s",   vartype, varname ); // float *x
          else fprintf(fp, "%s%s", vartype, varname); // float *a; 
        }
        else { // more than one dimension 
          fprintf(fp, "%s (", vartype); 
          for (int n=0; n< (numdimensions-1); n++) fprintf(fp, "*");
          fprintf(fp, "%s)", varname);
          fprintf(fp, "[%d]", arraysizes[numdimensions-1]); 
        }
      }
    } // if numdimensions > 0
    else { // parameter float x
        fprintf(fp, "%s ", vartype);
        if (byreference) fprintf(fp, "&");
        fprintf(fp, "%s", varname);
    }
  } // end parameter 

  else { // NOT A PARAMETER
    //fprintf(fp, "NOT A PARAM ... vartype '%s'\n", vartype); 
    //if (isArray()) fprintf(stderr, "an array, numdimensions %d\n", numdimensions);
    //fprintf(stderr, "arraysizes %p\n", arraysizes);


  
  //if (isArray() && arraysizes == NULL) { 
  //    // we just know the number of dimensions but no sizes
  //    // int ***something
  //    fprintf(fp, "%s ", vartype);   // "int "
  //    for (int i=0; i<numdimensions; i++) fprintf(fp, "*"); // ***
  //    fprintf(fp, "%s", varname); // variable name 
  //  }
  //  else 

    fprintf(fp, "%s %s",   vartype, arraypointerpart); 
    if (isRestrict) fprintf(fp, " __restrict__ ");  // wrong place
    fprintf(fp, "%s%s", varname, arraysetpart );  
    if (init) { 
      fprintf(fp, " = ");  fflush(fp); 
      init->print(0, fp);
    }
  }
  fflush(fp); 
  //fprintf(stderr, "numdimensions %d    arraysizes address 0x%x\n",  numdimensions, arraysizes); 
  //if (!isAParameter) fprintf(fp, ";\n",   vartype, varname, arraypart );  
};




void chillAST_VarDecl::printName( int in, FILE *fp ) {
  chillindent(in, fp);
  fprintf(fp, "%s", varname);
};




void chillAST_VarDecl::dump( int indent, FILE *fp ) {
  chillindent(indent, fp);
  fprintf(fp, "(VarDecl \"'%s' '%s' '%s'\"  n_dim %d  )  ",  vartype, varname, arraypart, numdimensions);
  
  //fprintf(fp, "vardef %p\n", vardef);
  //if (vardef) fprintf(fp, "(typedef or struct!)\n"); 
  //fprintf(fp, "typedefinition %p\n", typedefinition);
  //if (isStruct) fprintf(fp, "isStruct\n"); 
  
  //if (isAParameter) fprintf(fp, "PARAMETER\n");
  //else fprintf(fp, "NOT PARAMETER\n");
  fflush(fp); 

  //segfault();  // see what called this 
};


chillAST_RecordDecl * chillAST_VarDecl::getStructDef() { 
  if (vardef) return vardef;
  if (typedefinition) return typedefinition->getStructDef();
  return NULL; 
}





chillAST_CompoundStmt::chillAST_CompoundStmt() {
  //fprintf(stderr, "chillAST_CompoundStmt::chillAST_CompoundStmt() %p\n", this); 
  asttype = CHILLAST_NODETYPE_COMPOUNDSTMT; 
  parent = NULL; 
  symbol_table = NULL;
  typedef_table = NULL;
  isFromSourceFile = true; // default 
  filename = NULL;
}; 


void  chillAST_CompoundStmt::print( int indent,  FILE *fp ) { 
  printPreprocBEFORE(indent, fp); 
  int numchildren = children.size();
  //fprintf(stderr, "NUMCHILDREN %d\n", numchildren); sleep(1); 
  for (int i=0; i<numchildren; i++) {
    children[i]->print(indent, fp);
    if (children[i]->asttype != CHILLAST_NODETYPE_FORSTMT  
        && children[i]->asttype != CHILLAST_NODETYPE_IFSTMT
        && children[i]->asttype != CHILLAST_NODETYPE_COMPOUNDSTMT
        //&& children[i]->asttype != CHILLAST_NODETYPE_VARDECL   // vardecl does its own ";\n"
        ) 
      {
        fprintf(fp, ";\n");  // probably wrong 
      }
  }
  fflush(fp); 
}

void chillAST_CompoundStmt::replaceChild( chillAST_node *old, chillAST_node *newchild ){
  //fprintf(stderr, "chillAST_CompoundStmt::replaceChild( old %s, new %s)\n", old->getTypeString(), newchild->getTypeString() ); 
   vector<chillAST_node*> dupe = children; 
   int numdupe = dupe.size();
  int any = 0; 
  
  for (int i=0; i<numdupe; i++) { 

    //fprintf(stderr, "\ni %d\n",i); 
    //for (int j=0; j<numdupe; j++) { 
    //  fprintf(stderr, "this 0x%x   children[%d/%d] = 0x%x type %s\n", this, j, children.size(), children[j], children[j]->getTypeString()); 
    //}


    if (dupe[i] == old) { 
      //fprintf(stderr, "replacing child %d of %d\n", i, numdupe); 
      //fprintf(stderr, "was \n"); print();
      children[i] = newchild;
      newchild->setParent( this );
      //fprintf(stderr, "is  \n");  print(); fprintf(stderr, "\n\n"); 
      // old->parent = NULL; 
      any = 1;
    }
  }

  if (!any) { 
    fprintf(stderr, "chillAST_CompoundStmt::replaceChild(), could not find old\n");
    exit(-1); 
  }
}


void chillAST_CompoundStmt::loseLoopWithLoopVar( char *var ) { 
  //fprintf(stderr, "chillAST_CompoundStmt::loseLoopWithLoopVar( %s )\n", var); 

  //fprintf(stderr, "CompoundStmt 0x%x has parent 0x%x  ", this, this->parent);
  //fprintf(stderr, "%s\n", parent->getTypeString()); 

  
  //fprintf(stderr, "CompoundStmt node has %d children\n", children.size()); 
  //fprintf(stderr, "before doing a damned thing, \n"); 
  //print();
  //dump(); fflush(stdout);
  //fprintf(stderr, "\n\n"); 

#ifdef DAMNED
  for (int j=0; j<children.size(); j++) { 
    fprintf(stderr, "j %d/%d  ", j, children.size()); 
    fprintf(stderr, "subnode %d 0x%x  ", j, children[j] );
    fprintf(stderr, "asttype %d  ", children[j]->asttype); 
    fprintf(stderr, "%s    ", children[j]->getTypeString());
    if (children[j]->isForStmt()) { 
      chillAST_ForStmt *FS = ((chillAST_ForStmt *)  children[j]); 
      fprintf(stderr, "for (");
      FS->init->print(0, stderr);
      fprintf(stderr, "; ");
      FS->cond->print(0, stderr);
      fprintf(stderr, "; ");
      FS->incr->print(0, stderr);
      fprintf(stderr, ")  with %d statements in body 0x%x\n",  FS->body->getNumChildren(), FS->body );   
    }
    else fprintf(stderr, "\n"); 
  }
#endif


  vector<chillAST_node*> dupe = children; // simple enough?
  for (int i=0; i<dupe.size(); i++) { 
    //for (int j=0; j<dupe.size(); j++) { 
    //  fprintf(stderr, "j %d/%d\n", j, dupe.size()); 
    //  fprintf(stderr, "subnode %d %s    ", j, children[j]->getTypeString());
    //  if (children[j]->isForStmt()) { 
    //    chillAST_ForStmt *FS = ((chillAST_ForStmt *)  children[j]); 
    //    fprintf(stderr, "for (");
    //     FS->init->print(0, stderr);
    //    fprintf(stderr, "; ");
    //    FS->cond->print(0, stderr);
    //    fprintf(stderr, "; ");
    //    FS->incr->print(0, stderr);
    //    fprintf(stderr, ")  with %d statements in body 0x%x\n",  FS->body->getNumChildren(), FS->body );   
    //} 
    //else fprintf(stderr, "\n"); 
    //}
    
    //fprintf(stderr, "CompoundStmt 0x%x recursing to child %d/%d\n", this, i, dupe.size()); 
    dupe[i]->loseLoopWithLoopVar( var );
  }
  //fprintf(stderr, "CompoundStmt node 0x%x done recursing\n", this ); 
}



void chillAST_CompoundStmt::dump(  int indent,  FILE *fp ) { 
  chillindent(indent, fp); 
  fprintf(fp, "(CompoundStmt \n" );
  int numchildren = children.size();

  //for (int i=0; i<numchildren; i++)  { 
  //  fprintf(fp, "%d %s 0x%x\n",  i, children[i]->getTypeString(), children[i]); 
  //} 
  //fprintf(fp, "\n"); 

  for (int i=0; i<numchildren; i++)  { 
    children[i]->dump(indent+1, fp);
  }
  chillindent(indent, fp); 
  fprintf(fp, ")\n"); 
};



chillAST_node*  chillAST_CompoundStmt::constantFold(){ 
  //fprintf(stderr, "chillAST_CompoundStmt::constantFold()\n"); 
  for (int i=0; i<children.size(); i++) children[i] =  children[i]->constantFold();
  return this;
}


chillAST_node*  chillAST_CompoundStmt::clone(){ 
  chillAST_CompoundStmt *cs = new chillAST_CompoundStmt();
  for (int i=0; i<children.size(); i++) cs->addChild( children[i]->clone() );
  cs->setParent( parent ); 
  cs->isFromSourceFile = isFromSourceFile; 
  if (filename) cs->filename = strdup(filename); 
  return cs;
}


void chillAST_CompoundStmt::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //fprintf(stderr, "chillAST_CompoundStmt::gatherVarDecls()\n"); 
  for (int i=0; i<children.size(); i++) children[i]->gatherVarDecls( decls ); 
}


void chillAST_CompoundStmt::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<children.size(); i++) children[i]->gatherScalarVarDecls( decls ); 
}


void chillAST_CompoundStmt::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<children.size(); i++) children[i]->gatherArrayVarDecls( decls ); 
}


void chillAST_CompoundStmt::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  for (int i=0; i<children.size(); i++) children[i]->gatherDeclRefExprs( refs ); 
}


void chillAST_CompoundStmt::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<children.size(); i++) children[i]->gatherVarUsage( decls ); 
}


void chillAST_CompoundStmt::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) { 
  for (int i=0; i<children.size(); i++) children[i]->gatherArrayRefs( refs, 0); 
}

void chillAST_CompoundStmt::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  for (int i=0; i<children.size(); i++) children[i]->gatherScalarRefs( refs, 0); 
} 
  
void chillAST_CompoundStmt::gatherStatements(std::vector<chillAST_node*> &statements ){
  for (int i=0; i<children.size(); i++) children[i]->gatherStatements( statements ); 
} 
 


void chillAST_CompoundStmt::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  for (int i=0; i<children.size(); i++) children[i]->replaceVarDecls( olddecl, newdecl ); 
}


bool chillAST_CompoundStmt::findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync ) { 

  // see how many elements we currently have
  int sofar = children.size(); 

  // make big enough to add a sync after each statement. wasteful. TODO
  // this prevents inserts happening at the forstmt::addSync() from causing a 
  // reallocation, which screwsup the loop below here 
  children.reserve( 2 * sofar );
  //fprintf(stderr, "sofar %d   reserved %d\n", sofar, 2*sofar); 

  bool force = false;
  for (int i=0; i<children.size(); i++) {  // children.size() to see it gain each time
    if (children.size() > sofar ) { 
      //fprintf(stderr, "HEY! CompoundStmt::findLoopIndexesToReplace() noticed that children increased from %d to %d\n", sofar, children.size()); 
      sofar = children.size(); 
    }

    //fprintf(stderr, "compound child %d of type %s force %d\n", i, children[i]->getTypeString(), force ); 
    bool thisforces = children[i]->findLoopIndexesToReplace( symtab, force );
    force = force || thisforces; // once set, always
  }

  return false; 

/* 
  vector<chillAST_node*> childrencopy;
  for (int i=0; i<children.size(); i++) childrencopy.push_back( children[i] ); 
  bool force = false;
  
  char *origtypes[64]; 
  int origsize = children.size(); 
  for (int i=0; i<children.size(); i++) { 
    fprintf(stderr, "ORIGINAL compound child %d of type %s\n", i, children[i]->getTypeString() ); 
    origtypes[i] = strdup( children[i]->getTypeString() ); 
    fprintf(stderr, "ORIGINAL compound child %d of type %s\n", i, children[i]->getTypeString() ); 
  }
    
  for (int i=0; i<childrencopy.size(); i++) { 
    fprintf(stderr, "compound child %d of type %s force %d\n", i, childrencopy[i]->getTypeString(), force ); 
    force = force || childrencopy[i]->findLoopIndexesToReplace( symtab, force ); // once set, always
  }

  fprintf(stderr, "\n"); 
  for (int i=0; i<origsize; i++) { 
    fprintf(stderr, "BEFORE compound child %d/%d of type %s\n",  i, origsize, origtypes[i]); 
  }
  for (int i=0; i<children.size(); i++) { 
    fprintf(stderr, "AFTER  compound child %d/%d of type %s\n", i, children.size(), children[i]->getTypeString() ); 
  }

  return false;
*/ 
}





chillAST_ParenExpr::chillAST_ParenExpr(  chillAST_node *sub, chillAST_node *par ){
  subexpr = sub;
  subexpr->setParent( this );
  asttype = CHILLAST_NODETYPE_PARENEXPR; 
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}

void chillAST_ParenExpr::print(  int indent,  FILE *fp ) { 
  //fprintf(stderr, "chillAST_ParenExpr::print()\n"); 
  chillindent(indent, fp); // hard to believe this will ever do anything
  fprintf(fp, "(" ); 
  subexpr->print( 0, fp ); 
  fprintf(fp, ")" ); 
  fflush(fp); 
}

void chillAST_ParenExpr::dump(  int indent,  FILE *fp ) { 
   chillindent(indent, fp);
   fprintf(fp, "(ParenExpr \n"); 
   subexpr->dump( indent+1, fp );
   chillindent(indent, fp);
   fprintf(fp, ")\n");
}


void chillAST_ParenExpr::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {
   subexpr->gatherArrayRefs( refs, writtento );
}

void chillAST_ParenExpr::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
   subexpr->gatherScalarRefs( refs, writtento );
} 



chillAST_node* chillAST_ParenExpr::constantFold() {
  subexpr = subexpr->constantFold();
  return this; 
}


chillAST_node* chillAST_ParenExpr::clone() {
  chillAST_ParenExpr *PE = new chillAST_ParenExpr( subexpr->clone(), NULL ); 
  PE->isFromSourceFile = isFromSourceFile; 
  if (filename) PE->filename = strdup(filename); 
  return PE; 
}

void chillAST_ParenExpr::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarDecls( decls ); 
}


void chillAST_ParenExpr::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherScalarVarDecls( decls ); 
}


void chillAST_ParenExpr::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherArrayVarDecls( decls ); 
}


void chillAST_ParenExpr::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  subexpr->gatherDeclRefExprs( refs ); 
}

void chillAST_ParenExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  subexpr->replaceVarDecls( olddecl, newdecl ); 
}

void chillAST_ParenExpr::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  subexpr->gatherVarUsage( decls ); 
}



chillAST_Sizeof::chillAST_Sizeof( char *athing, chillAST_node *par ){
  thing = strdup( athing ); // memory leak
  parent = par;
  isFromSourceFile = true; // default 
  filename = NULL;
}

void chillAST_Sizeof::print(  int indent,  FILE *fp ) { 
  //fprintf(stderr, "chillAST_Sizeof::print()\n"); 
  chillindent(indent, fp); // hard to believe this will ever do anything
  fprintf(fp, "sizeof(" ); 
  fprintf(fp, "%s)", thing ); 
  fflush(fp); 
}


void chillAST_Sizeof::dump(  int indent,  FILE *fp ) { 
   chillindent(indent, fp);
   fprintf(fp, "(Sizeof  %s )\n", thing); 
}

void chillAST_Sizeof::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {} 
void chillAST_Sizeof::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {}

chillAST_node* chillAST_Sizeof::constantFold() {
  return this; 
}

chillAST_node* chillAST_Sizeof::clone() {
  chillAST_Sizeof *SO = new chillAST_Sizeof( thing, NULL ); 
  SO->isFromSourceFile = isFromSourceFile; 
  if (filename) SO->filename = strdup(filename); 
  return SO; 
}

void chillAST_Sizeof::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {  // TODO 
}


void chillAST_Sizeof::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {  // TODO 
}


void chillAST_Sizeof::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {  // TODO 
}


void chillAST_Sizeof::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  // TODO 
}


void chillAST_Sizeof::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
}


void insertNewDeclAtLocationOfOldIfNeeded( chillAST_VarDecl *newdecl, chillAST_VarDecl *olddecl) {
  //fprintf(stderr, "insertNewDeclAtLocationOfOldIfNeeded( new 0x%x  old 0x%x\n", newdecl, olddecl );

  if (newdecl == NULL || olddecl == NULL) {
    fprintf(stderr, "chill_ast.cc insertNewDeclAtLocationOfOldIfNeeded() NULL decl\n");
    exit(-1);
  }

  if (newdecl == olddecl) return;

  newdecl->vartype = strdup(olddecl->vartype);

  chillAST_node *newparent = newdecl->parent;
  chillAST_node *oldparent = olddecl->parent;
  //fprintf(stderr, "newparent 0x%x   oldparent 0x%x\n", newparent, oldparent ); 
  if (newparent == oldparent) return;

  if (newparent != NULL) 
    //fprintf(stderr, "chill_ast.cc insertNewDeclAtLocationOfOldIfNeeded() new decl already has parent??  probably wrong\n");
  newdecl->parent = oldparent;  // will be true soon 

  // find actual location of old decl and insert new one there
  //fprintf(stderr, "oldparent is of type %s\n", oldparent->getTypeString()); // better be compoundstmt ??
  vector<chillAST_node*> children = oldparent->getChildren();
  
  int numchildren = children.size(); 
  //fprintf(stderr, "oldparent has %d children\n", numchildren); 
  
  if (numchildren == 0) {
    fprintf(stderr, "chill_ast.cc insertNewDeclAtLocationOfOldIfNeeded() impossible number of oldparent children (%d)\n", numchildren); 
    exit(-1);
  }

  bool newalreadythere = false;
  int index = -1;
  //fprintf(stderr, "olddecl is 0x%x\n", olddecl); 
  //fprintf(stderr, "I know of %d variables\n", numchildren);
  for (int i=0; i<numchildren; i++) { 
    chillAST_node *child = oldparent->getChild(i); 
    //fprintf(stderr, "child %d @ 0x%x is of type %s\n", i, child, child->getTypeString()); 
    if (children[i] == olddecl) { 
      index = i;
      //fprintf(stderr, "found old decl at index %d\n", index); 
    }
    if (children[i] == newdecl) {  
      newalreadythere = true; 
      //fprintf(stderr, "new already there @ index %d\n", i); 
    }
  }
  if (index == -1) { 
    fprintf(stderr, "chill_ast.cc insertNewDeclAtLocationOfOldIfNeeded() can't find old decl for %s\n", olddecl->varname);
    exit(-1);
  }

  if (!newalreadythere) oldparent->insertChild( index, newdecl );

}


void gatherVarDecls( vector<chillAST_node*> &code, vector<chillAST_VarDecl*> &decls) {
  //fprintf(stderr, "gatherVarDecls()\n");

  int numcode = code.size();
  //fprintf(stderr, "%d top level statements\n", numcode);
  for (int i=0; i<numcode; i++) {
    chillAST_node *statement = code[i];
    statement->gatherVarDecls( decls );
  }

}


void gatherVarUsage( vector<chillAST_node*> &code, vector<chillAST_VarDecl*> &decls) {
  //fprintf(stderr, "gatherVarUsage()\n");

  int numcode = code.size();
  //fprintf(stderr, "%d top level statements\n", numcode);
  for (int i=0; i<numcode; i++) {
    chillAST_node *statement = code[i];
    statement->gatherVarUsage( decls );
  }

}




chillAST_IfStmt::chillAST_IfStmt() { 
  cond     = thenpart = elsepart = NULL;
  asttype = CHILLAST_NODETYPE_IFSTMT; 
  isFromSourceFile = true; // default 
  filename = NULL;
}

chillAST_IfStmt::chillAST_IfStmt(chillAST_node *c, chillAST_node *t, chillAST_node *e, chillAST_node *p){
  cond = c;
  cond->setParent( this );
  thenpart = t;
  if (thenpart) thenpart->setParent( this ); 
  elsepart = e;
  if (elsepart) elsepart->setParent( this ); 
  parent = p;
  asttype = CHILLAST_NODETYPE_IFSTMT; 
  isFromSourceFile = true; // default 
  filename = NULL;
}

void chillAST_IfStmt::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  if (cond)         cond->gatherVarDecls( decls );
  if (thenpart) thenpart->gatherVarDecls( decls );
  if (elsepart) elsepart->gatherVarDecls( decls );
}


void chillAST_IfStmt::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  if (cond)         cond->gatherScalarVarDecls( decls );
  if (thenpart) thenpart->gatherScalarVarDecls( decls );
  if (elsepart) elsepart->gatherScalarVarDecls( decls );
}


void chillAST_IfStmt::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  if (cond)         cond->gatherArrayVarDecls( decls );
  if (thenpart) thenpart->gatherArrayVarDecls( decls );
  if (elsepart) elsepart->gatherArrayVarDecls( decls );
}


void chillAST_IfStmt::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  if (cond)         cond->gatherDeclRefExprs( refs );
  if (thenpart) thenpart->gatherDeclRefExprs( refs );
  if (elsepart) elsepart->gatherDeclRefExprs( refs );
}


void chillAST_IfStmt::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  if (cond)         cond->gatherVarUsage( decls );
  if (thenpart) thenpart->gatherVarUsage( decls );
  if (elsepart) elsepart->gatherVarUsage( decls );
}


void chillAST_IfStmt::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) { 
  cond->gatherArrayRefs( refs, 0 );  // 0 ??
  thenpart->gatherArrayRefs( refs, 0 );  // 0 ??
  if (elsepart) elsepart->gatherArrayRefs( refs, 0 );  // 0 ??
}

void chillAST_IfStmt::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  cond->gatherScalarRefs( refs, 0 );  // 0 ??
  thenpart->gatherScalarRefs( refs, 0 );  // 0 ??
  if (elsepart) elsepart->gatherScalarRefs( refs, 0 );  // 0 ??
} 


chillAST_node *chillAST_IfStmt::constantFold() { 
  if (cond) cond = cond->constantFold();
  if (thenpart) thenpart = thenpart->constantFold();
  if (elsepart) elsepart = elsepart->constantFold();
  return this; 
}

void chillAST_IfStmt::gatherStatements(std::vector<chillAST_node*> &statements ){

  //print(); printf("\n"); fflush(stdout); 
  thenpart->gatherStatements( statements );
  //fprintf(stderr, "ifstmt, after then, %d statements\n", statements.size()); 
  if (elsepart){ 
    //fprintf(stderr, "there is an elsepart of type %s\n", elsepart->getTypeString()); 
    elsepart->gatherStatements( statements );
  }
  //fprintf(stderr, "ifstmt, after else, %d statements\n", statements.size()); 
}



chillAST_node *chillAST_IfStmt::clone() { 
  chillAST_node *c, *t, *e; 
  c = t = e = NULL; 
  if (cond) c = cond->clone(); // has to be one, right? 
  if (thenpart) t = thenpart->clone();
  if (elsepart) e = elsepart->clone();

  chillAST_IfStmt *IS = new chillAST_IfStmt( c, t, e, parent); 
  IS->isFromSourceFile = isFromSourceFile;
  if (filename) IS->filename = strdup(filename); 
  return IS;
} 



void  chillAST_IfStmt::dump(  int indent,  FILE *fp ) { 
  chillindent(indent, fp);
  fprintf(fp, "(if ");
  fprintf(fp, "\n");

  cond->dump(indent+1, fp); 
  fprintf(fp, "\n");

  thenpart->dump(indent+1, fp); 
  fprintf(fp, "\n"); 

  if (elsepart) { 
    elsepart->dump(indent+1, fp); 
    fprintf(fp, "\n");
  }
  chillindent(indent, fp);
  fprintf(fp, ")\n");
}



void chillAST_IfStmt::print(int indent, FILE *fp ) { 
  printPreprocBEFORE(indent, fp); 
  chillindent(indent, fp);
  fprintf(fp, "if ("); 
  cond->print(0, fp); 
  
  bool needbracket = true; 
  if (thenpart->isBinaryOperator()) needbracket = false;
  if (thenpart->isCompoundStmt()) { // almost always true
    chillAST_CompoundStmt *CS = (chillAST_CompoundStmt*) thenpart;
    if (CS->children.size() == 1  && CS->children[0]->isBinaryOperator()) needbracket = false;
  }    
  
  if(needbracket)  fprintf(fp, ") {\n"); 
  else fprintf(fp, ")\n"); 
  
  thenpart->print(indent+1, fp); // end of line 
  
  if(needbracket)  {
    //fprintf(fp, "\n"); 
    chillindent(indent, fp);
    fprintf(fp, "}\n"); 
  }
  
  needbracket = true;
  if (elsepart) { 
    if (elsepart->isBinaryOperator()) needbracket = false;
    if (elsepart->isCompoundStmt()) { // almost always true
      chillAST_CompoundStmt *CS = (chillAST_CompoundStmt*) elsepart;
      
      if (CS->children.size() == 1  && CS->children[0]->isBinaryOperator()) needbracket = false;
      
    }    
    
    fprintf(fp, "\n"); 
    chillindent(indent, fp);
    
    if (needbracket) fprintf(fp, "else {\n"); 
    else fprintf(fp, "else\n"); 
    
    elsepart->print(indent+1, fp);
    
    if(needbracket)  {
      fprintf(fp, "\n"); 
      chillindent(indent, fp);
      fprintf(fp, "}\n"); 
    }
  }
  //else fprintf(fp, "else { /* NOTHING */ }"); 
}


 
bool chillAST_IfStmt::findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync ) { 
  thenpart->findLoopIndexesToReplace( symtab ); 
  elsepart->findLoopIndexesToReplace( symtab ); 
  return false; // ?? 
}

  

chillAST_node *lessthanmacro( chillAST_node *left, chillAST_node *right) { 

  chillAST_ParenExpr *lp1 = new chillAST_ParenExpr( left  );
  chillAST_ParenExpr *rp1 = new chillAST_ParenExpr( right );
  chillAST_BinaryOperator *cond = new chillAST_BinaryOperator( lp1, "<", rp1 );

  chillAST_ParenExpr *lp2 = new chillAST_ParenExpr( left  );
  chillAST_ParenExpr *rp2 = new chillAST_ParenExpr( right );
  
  chillAST_TernaryOperator *t = new chillAST_TernaryOperator("?", cond, lp2, rp2);
  
  return t; 
} 




// look for function declaration with a given name, in the tree with root "node"
void findFunctionDeclRecursive( chillAST_node *node, const char *procname, vector<chillAST_FunctionDecl*>& funcs )
{
  //fprintf(stderr, "findmanually()                CHILL AST node of type %s\n", node->getTypeString()); 
  
  if (node->isFunctionDecl()) { 
    char *name = ((chillAST_FunctionDecl *) node)->functionName; // compare name with desired name
    //fprintf(stderr, "node name 0x%x  ", name);
    //fprintf(stderr, "%s     procname ", name); 
    //fprintf(stderr, "0x%x  ", procname);
    //fprintf(stderr, "%s\n", procname); 
    if (!strcmp( name, procname)) {
      //fprintf(stderr, "found procedure %s\n", procname ); 
      funcs.push_back( (chillAST_FunctionDecl*) node );  // this is it 
      // quit recursing. probably not correct in some horrible case
      return; 
    }
    //else fprintf(stderr, "this is not the function we're looking for\n"); 
  }


  // this is where the children can be used effectively. 
  // we don't really care what kind of node we're at. We just check the node itself
  // and then its children is needed. 

  int numc = node->children.size();  
  //fprintf(stderr, "%d children\n", numc);

  for (int i=0; i<numc; i++) {
    //fprintf(stderr, "node of type %s is recursing to child %d\n",  node->getTypeString(), i); 
    findFunctionDeclRecursive( node->children[i], procname, funcs );
  }
  return; 
}


chillAST_FunctionDecl *findFunctionDecl( chillAST_node *node, const char *procname)
{
  vector<chillAST_FunctionDecl*> functions;
  findFunctionDeclRecursive( node, procname, functions );  

  if ( functions.size() == 0 ) { 
    fprintf(stderr, "could not find function named '%s'\n", procname);
    exit(-1);
  }
  
  if ( functions.size() > 1 ) { 
    fprintf(stderr, "oddly, found %d functions named '%s'\n", functions.size(), procname);
    fprintf(stderr, "I am unsure what to do\n"); 
    exit(-1);
  }
  
  //fprintf(stderr, "found the procedure named %s\n", procname); 
  return functions[0];
}


chillAST_SymbolTable *addSymbolToTable(  chillAST_SymbolTable *st, chillAST_VarDecl *vd ) // definition
{
  chillAST_SymbolTable *s = st;
  if (!s) s = new chillAST_SymbolTable; 
 
  int tablesize = s->size();
  
  for (int i=0; i<tablesize; i++) { 
    if ((*s)[i] == vd) { 
      //fprintf(stderr, "the exact same symbol, not just the same name, was already there\n"); 
      return s; // already there 
    }
  }

  for (int i=0; i<tablesize; i++) { 
    //fprintf(stderr, "name %s vs name %s\n", (*s)[i]->varname, vd->varname); 
    if (!strcmp( (*s)[i]->varname, vd->varname)) { 
      //fprintf(stderr, "symbol with the same name was already there\n"); 
      return s; // already there 
    }
  }

  //fprintf(stderr, "adding %s %s to a symbol table that didn't already have it\n", vd->vartype, vd->varname); 

  //printf("before:\n"); 
  //printSymbolTable( s ); fflush(stdout); 

  s->push_back(vd); // add it 

  //printf("after:\n"); 
  //printSymbolTable( s ); fflush(stdout); 
  return s;
}


chillAST_TypedefTable *addTypedefToTable(  chillAST_TypedefTable *tdt, chillAST_TypedefDecl *td )
{

  chillAST_TypedefTable *t = tdt;
  if (!t) t = new chillAST_TypedefTable;

  int tablesize = t->size();
  
  for (int i=0; i<tablesize; i++) {
    if ((*t)[i] == td) return t; // already there 
  }
  t->push_back(td); // add it 
  return t;
}


chillAST_NoOp::chillAST_NoOp( chillAST_node *p ) { 
  parent = p;   
  isFromSourceFile = true; // default 
  filename = NULL;
}; // so we have SOMETHING for NoOp in the cc file ???


chillAST_Preprocessing::chillAST_Preprocessing() { 
  position = CHILL_PREPROCESSING_POSITIONUNKNOWN;
  pptype   = CHILL_PREPROCESSING_TYPEUNKNOWN;
  blurb = strdup("");  // never use null. ignore the leak ??
}


 chillAST_Preprocessing::chillAST_Preprocessing(CHILL_PREPROCESSING_POSITION pos, 
                                                CHILL_PREPROCESSING_TYPE t, 
                                                char *text )
 {
   position = pos;
   pptype = t;
   blurb = strdup( text ); 
 }
  
void chillAST_Preprocessing::print( int indent, FILE *fp ) {  // probably very wrong
   if (pptype == CHILL_PREPROCESSING_LINEAFTER ) {
     fprintf(fp, "\n");
     chillindent(indent, fp);
   }
   if (pptype ==  CHILL_PREPROCESSING_LINEBEFORE) {  // ??? 
     //fprintf(fp, "\n");
     chillindent(indent, fp);
   }
   
   fprintf(fp, "%s", blurb); 

   if (pptype ==  CHILL_PREPROCESSING_TOTHERIGHT) {
      fprintf(fp, "\n");
   }


   if (pptype ==  CHILL_PREPROCESSING_LINEBEFORE) { 
     //fprintf(fp, "\n"); // comment seems to have \n at the end already
     //chillindent(indent, fp);
   }


   //if (pptype != CHILL_PREPROCESSING_IMMEDIATELYBEFORE && pptype != CHILL_PREPROCESSING_UNKNOWN) fprint(fp, "\n");
       
 }
