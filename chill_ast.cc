


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
  //debug_fprintf(stderr, "parseUnderlyingType( %s )\n", sometype); 
  char *underlying = strdup(sometype); 
  char *p;
  char *start = underlying;

  // ugly.  we want to turn "float *" into "float" but "struct abc *" into struct abc.
  // there are probably many more cases. have an approved list?   TODO 
  if (strstr(underlying, "struct ")) start += 7;  // (length("struct "))
  //debug_fprintf(stderr, "sometype '%s'   start '%s'\n", sometype, start); 
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

  // see if name has a dot or arrow (->) indicating that it is a structure/class
  const char *cdot = strstr( name, "." );
  const char *carrow = strstr(name, "->");  // initial 'c' for const - can't change those

  char *varname;
  char *subpart = NULL;

  if (cdot || carrow) { 
    debug_fprintf(stderr, "symbolTableFindVariableNamed(), name '%s' looks like a struct\n", name); 

    // so, look for the first part in the symbol table.
    // warning, this could be looking for a->b.c.d->e.f->g
    varname = strdup( name );

    char *dot   = strstr(varname, "." );
    char *arrow = strstr( varname, "->" );
    if (dot != NULL && arrow != NULL ) { // dot AND arrow, 
      debug_fprintf(stderr, "chillast.cc symbolTableFindVariableNamed(), name '%s' has both dot and arrow? TODO\n");
      exit(-1); 
    }
    else if (dot != NULL && !arrow) { // just dot(s).  dot points to the first one 
      //debug_fprintf(stderr, "name '%s' has dot(s)\n", varname);
      *dot = '\0'; // end string at the dot
      subpart = &(dot[1]);
      debug_fprintf(stderr, "will now look for a struct/class named %s that has member %s\n", varname, subpart);

    }
    else if (arrow != NULL && !dot) { // just arrow(s)  arrow points to the first one
      //debug_fprintf(stderr, "name '%s' has arrow(s)\n", varname);
      *arrow = '\0'; // end string at the arrow
      subpart = &(arrow[2]); 
      debug_fprintf(stderr, "will now look for a struct/class named %s that has member %s\n", varname, subpart);
    }
    else { // impossible 
      debug_fprintf(stderr, "chillast.cc symbolTableFindVariableNamed(), varname '%s', looks like a struct,  but I can't figure it out\n", varname);
      exit(-1); 
    }
  }
  else { 
    varname = strdup(name); 
  }

  int numvars = table->size();
  for (int i=0; i<numvars; i++) { 
    chillAST_VarDecl *vd = (*table)[i];
    if (!strcmp(varname, vd->varname)) { 
      debug_fprintf(stderr, "found variable named %s\n", varname);

      if (!subpart) return vd;  // need to check type? 

      // OK, we have a variable, which looks like a struct/class, and a subpart that is some member names
      //debug_fprintf(stderr, "but I don't know how to check if it has member %s\n", subpart); 
      
      char *dot   = strstr(subpart, "." );
      char *arrow = strstr(subpart, "->" );
      
      if (!dot && !arrow) { // whew, only one level of struct
        //debug_fprintf(stderr, "whew, only one level of struct\n"); 
        
        // make sure this variable definition is a struct
        if (vd->isAStruct()) { 
          //debug_fprintf(stderr, "%s is a struct of type %s\n", varname, vd->getTypeString()); 
          if (vd->isVarDecl()) { 
            chillAST_RecordDecl  *rd = vd->getStructDef(); 
            if (rd) { 
              //debug_fprintf(stderr, "has a recordDecl\n"); 
              
              chillAST_VarDecl *sp = rd->findSubpart( subpart );
              if (sp) { debug_fprintf(stderr, "found a struct member named %s\n", subpart); }
              else  { debug_fprintf(stderr, "DIDN'T FIND a struct member named %s\n", subpart); }
              return sp;  // return the subpart?? 
            }
            else { 
              debug_fprintf(stderr, "no recordDecl\n"); 
              exit(-1); 
            }
          }
          else { 
            debug_fprintf(stderr, "NOT a VarDecl???\n"); // impossible
          }
        }
        else { 
          debug_fprintf(stderr, "false alarm. %s is a variable, but doesn't have subparts\n", varname); 
          return NULL; // false alarm. a variable of the correct name exists, but is not a struct 
        }
      }
      
      debug_fprintf(stderr, "chillast.cc symbolTableFindVariableNamed(), name '%s'  can't figure out multiple levels of struct yet!\n"); 

      exit(-1); 
    }
  }
  return NULL;
}



char *ulhack( char *brackets ) // remove UL from numbers, MODIFIES the argument!
{
  //debug_fprintf(stderr, "ulhack( \"%s\"  -> ", brackets); 
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
  //debug_fprintf(stderr, "\"%s\" )\n", brackets); 
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

  //debug_fprintf(stderr, "after = '%s'\n", after); 

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

  //debug_fprintf(stderr, "parseArrayParts( %s ) => %s\n", sometype, arraypart); 
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



chillAST_VarDecl * chillAST_node::findVariableNamed( const char *name ) { // recursive
  if (hasSymbolTable()) { // look in my symbol table if I have one
    chillAST_VarDecl *vd = symbolTableFindVariableNamed( getSymbolTable(), name);
    if (vd) return vd; // found locally
  }
  if (!parent) return NULL; // no more recursion available
  // recurse upwards
  return parent->findVariableNamed( name ); 
}


chillAST_RecordDecl * chillAST_node::findRecordDeclNamed( const char *name ) { // recursive
  debug_fprintf(stderr, "%s::findRecordDeclNamed( %s )\n", getTypeString(), name); 
  // look in children
  int numchildren = children.size();
  debug_fprintf(stderr, "%d children\n", numchildren); 
  for (int i=0; i<numchildren; i++) {
    debug_fprintf(stderr, "child %d  %s\n", i, children[i]->getTypeString());
    if (children[i]->isRecordDecl()) {
      chillAST_RecordDecl *RD = (chillAST_RecordDecl *)children[i];
      debug_fprintf(stderr, "it is a recordDecl named '%s' vs '%s'\n", RD->getName(), name); 
      if (!strcmp( RD->getName(), name )) {
        debug_fprintf(stderr, "FOUND IT\n"); 
        return RD;
      }
    }
  }   
    
  if (!parent) return NULL; // no more recursion available
  // recurse upwards
  return parent->findRecordDeclNamed( name ); 
}


  void chillAST_node::printPreprocBEFORE( int indent, FILE *fp ) { 
    int numstmts = preprocessinginfo.size(); 
    //if (0 != numstmts) { 
    //  fprintf(fp, "chillAST_node::printPreprocBEFORE()  %d statements\n", numstmts); 
    //} 


    for (int i=0; i< numstmts; i++) { 
      //fprintf(fp, "stmt %d   %d\n", i, preprocessinginfo[i]->position); 
      if (preprocessinginfo[i]->position == CHILL_PREPROCESSING_LINEBEFORE || 
          preprocessinginfo[i]->position == CHILL_PREPROCESSING_IMMEDIATELYBEFORE) {
        //debug_fprintf(stderr, "before %d\n", preprocessinginfo[i]->position); 
        preprocessinginfo[i]->print(indent, fp); 
      }
    }
  }

  void chillAST_node::printPreprocAFTER( int indent, FILE *fp ) { 
    for (int i=0; i< preprocessinginfo.size(); i++) { 
      if (preprocessinginfo[i]->position == CHILL_PREPROCESSING_LINEAFTER || 
          preprocessinginfo[i]->position ==  CHILL_PREPROCESSING_TOTHERIGHT) { 
        //debug_fprintf(stderr, "after %d\n", preprocessinginfo[i]->position); 
        preprocessinginfo[i]->print(indent, fp); 
      }
    }
  }


chillAST_SourceFile::chillAST_SourceFile() {
    SourceFileName = strdup("No Source File");
    global_symbol_table = new chillAST_SymbolTable();
    global_typedef_table = new chillAST_TypedefTable();
    FileToWrite = NULL;
    frontend = strdup("unknown"); 
};

chillAST_SourceFile::chillAST_SourceFile(const char *filename ) { 
    SourceFileName = strdup(filename); 
    global_symbol_table = new chillAST_SymbolTable();
    global_typedef_table = new chillAST_TypedefTable();
    FileToWrite = NULL; 
    frontend = strdup("unknown"); 
};

chillAST_SourceFile::~chillAST_SourceFile() {
    delete this->global_symbol_table;
    delete this->global_typedef_table;
}

void chillAST_SourceFile::print( int indent, FILE *fp ) { 
  //debug_fprintf(stderr, "chillAST_SourceFile::print()\n"); 
  fflush(fp);
  fprintf(fp, "\n// this source derived from CHILL AST originally from file '%s' as parsed by frontend compiler %s\n\n", SourceFileName, frontend); 
  std::vector< char * > includedfiles; 
  int sofar = 0; 

  //fprintf(fp, "#define __rose_lt(x,y) ((x)<(y)?(x):(y))\n#define __rose_gt(x,y) ((x)>(y)?(x):(y))\n"); // help diff figure out what's going on

  int numchildren = children.size();
  //debug_fprintf(stderr, "// sourcefile has %d children\n", numchildren);
  //debug_fprintf(stderr, "they are\n");
  //for (int i=0; i<numchildren; i++) {
  //  debug_fprintf(stderr, "%s  ", children[i]->getTypeString());
  //  if (children[i]->isFunctionDecl()) {  
  //    debug_fprintf(stderr, "%s  ", ((chillAST_FunctionDecl *)children[i])->functionName);
  //  }
  //  debug_fprintf(stderr, "\n"); 
  //}  

  for (int i=0; i<numchildren; i++) {
    //fprintf(fp,  "\n// child %d of type %s:\n", i, children[i]->getTypeString());
    if (children[i]->isFromSourceFile) { 
      if (children[i]->isFunctionDecl()) { 
        debug_fprintf(stderr, "\nchild %d function %s\n",i,((chillAST_FunctionDecl *)children[i])->functionName); 
      } 
      //debug_fprintf(stderr, "child %d IS from source file\n", i); 
      //if (children[i]->isMacroDefinition()) fprintf(fp, "\n"); fflush(fp);
      children[i]->print( indent, fp );
      if (children[i]->isVarDecl()) fprintf(fp, ";\n"); fflush(fp);  // top level vardecl\n"); 
    }
    else { 
      //debug_fprintf(stderr, "child %d is not from source file\n", i); 
      // this should all go away 

#ifdef NOPE 
      if (children[i]->filename // not null and not empty string 
          //&&  0 != strlen(children[i]->filename)
          ) { // should not be necessary 
        //fprintf(fp, "// need an include for %s\n", children[i]->filename); 
        bool rddid = false;
        sofar = includedfiles.size(); 
        
        for (int j=0; j<sofar; j++) {
          //debug_fprintf(stderr, "comparing %s and %s\n",  includedfiles[j], children[i]->filename ); 
          if (!strcmp( includedfiles[j], children[i]->filename) ) { // this file has already been included
            rddid = true;
            //debug_fprintf(stderr, "already did that one\n"); 
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
    debug_fprintf(stderr, "can't open file '%s' for writing\n", fn);
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
  //debug_fprintf(stderr, "chillAST_SourceFile::findMacro( %s )\n", name );
  
  int numMacros = macrodefinitions.size();
  for (int i=0; i<numMacros; i++) { 
    if (!strcmp( macrodefinitions[i]->macroName, name )) return macrodefinitions[i];
  }
  return NULL; // not found
}


chillAST_FunctionDecl * chillAST_SourceFile::findFunction( const char *name ) {
  //debug_fprintf(stderr, "chillAST_SourceFile::findMacro( %s )\n", name );
  
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


chillAST_VarDecl * chillAST_SourceFile::findVariableNamed( const char *name ) { 
  debug_fprintf(stderr, "\SOURCEFILE SPECIAL %s  findVariableNamed( %s )\n", getTypeString(), name );   
  if (hasSymbolTable()) { // look in my symbol table if I have one
    debug_fprintf(stderr, "%s has a symbol table\n",  getTypeString()); 
    chillAST_VarDecl *vd = symbolTableFindVariableNamed( getSymbolTable(), name);
    if (vd) {
      debug_fprintf(stderr, "found it\n"); 
      return vd; // found locally
    }
    debug_fprintf(stderr, "%s has a symbol table but couldn't find %s\n", getTypeString(),  name ); 
  }

  debug_fprintf(stderr, "looking for %s in SourceFile global_symbol_table\n", name);
  chillAST_VarDecl *vd = symbolTableFindVariableNamed( global_symbol_table, name );
  if (vd) {
    debug_fprintf(stderr, "found it\n"); 
    return vd; // found locally
  }

  if (!parent) {
    debug_fprintf(stderr, "%s has no parent\n", getTypeString());
    return NULL; // no more recursion available
  }
  // recurse upwards
  //debug_fprintf(stderr, "recursing from %s up to parent %p\n", getTypeString(), parent);
  debug_fprintf(stderr, "recursing from %s up to parent\n", getTypeString());
  return parent->findVariableNamed( name ); 
}



chillAST_TypedefDecl::chillAST_TypedefDecl() { 
  underlyingtype = newtype = arraypart = NULL; 
  isStruct = isUnion = false;
  structname = NULL; 
  rd = NULL; 
};


chillAST_TypedefDecl::chillAST_TypedefDecl(char *t, char *nt, chillAST_node *par) { 
  //debug_fprintf(stderr, "chillAST_TypedefDecl::chillAST_TypedefDecl( underlying type %s, newtype %s )\n", t, nt); 
  underlyingtype = strdup(t); 
  newtype = strdup(nt);
  arraypart = NULL; 
  isStruct = isUnion = false;
  structname = NULL; 
  rd = NULL; 
};


chillAST_TypedefDecl::chillAST_TypedefDecl(char *t, char *a, char *p, chillAST_node *par) { 
  underlyingtype = strdup(t); 
  //debug_fprintf(stderr, "chillAST_TypedefDecl::chillAST_TypedefDecl( underlying type %s )\n", underlyingtype); 
  newtype = strdup(a);  // the new named type ??

  arraypart = strdup(p);  // array (p)art? 
  // splitarraypart(); // TODO 

  parent = par;
  isStruct = isUnion = false;
  structname = NULL; 
  rd = NULL; 
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
    debug_fprintf(stderr, "/* no rd */\n"); 
    
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
  //debug_fprintf(stderr, "chillAST_TypedefDecl::findSubpart( %s )\n", name);
  //debug_fprintf(stderr, "typedef %s  %s\n", structname, newtype); 

  if (rd) { // we have a record decl look there
    chillAST_VarDecl *sub = rd->findSubpart( name );
    //debug_fprintf(stderr, "rd found subpart %p\n", sub); 
    return sub; 
  }

  // can this ever happen now ??? 
  int nsub = subparts.size();
  //debug_fprintf(stderr, "%d subparts\n", nsub); 
  for (int i=0; i<nsub; i++) { 
    if ( !strcmp( name, subparts[i]->varname )) return subparts[i];
  }
  //debug_fprintf(stderr, "subpart not found\n"); 

  
  return NULL; 
}


chillAST_RecordDecl * chillAST_TypedefDecl::getStructDef() { 
  if (rd) return rd;
  return NULL;  
}



chillAST_RecordDecl::chillAST_RecordDecl() { 
  name = strdup("unknown"); // ??
  originalname = NULL;      // ?? 
  isStruct = isUnion = false;
}

chillAST_RecordDecl::chillAST_RecordDecl( const char *nam, chillAST_node *p ) { 
  //debug_fprintf(stderr, "chillAST_RecordDecl::chillAST_RecordDecl()\n");
  parent = p;
  if (nam) name = strdup(nam);
  else name = strdup("unknown"); // ?? 
  originalname = NULL;      // ??   // make them do it manually?
  isStruct = isUnion = false;
}

chillAST_RecordDecl::chillAST_RecordDecl( const char *nam, const char *orig, chillAST_node *p ) { 
  //debug_fprintf(stderr, "chillAST_RecordDecl::chillAST_RecordDecl( %s, (orig) )\n", nam); 
  parent = p;
  if (p) p->addChild( this );

  if (nam) name = strdup(nam);
  else name = strdup("unknown"); // ?? 
  
  originalname = NULL;   
  if (orig) originalname = strdup(orig);
  
  isStruct = isUnion = false;
}



chillAST_VarDecl * chillAST_RecordDecl::findSubpart( const char *nam ){
  //debug_fprintf(stderr, "chillAST_RecordDecl::findSubpart( %s )\n", nam);
  int nsub = subparts.size();
  //debug_fprintf(stderr, "%d subparts\n", nsub);
  for (int i=0; i<nsub; i++) { 
    //debug_fprintf(stderr, "comparing to '%s' to '%s'\n", nam, subparts[i]->varname);
    if ( !strcmp( nam, subparts[i]->varname )) return subparts[i];
  }
  debug_fprintf(stderr, "chillAST_RecordDecl::findSubpart() couldn't find member NAMED %s in ", nam); print(); printf("\n\n"); fflush(stdout); 

  return NULL;   
}


chillAST_VarDecl * chillAST_RecordDecl::findSubpartByType( const char *typ ){
  //debug_fprintf(stderr, "chillAST_RecordDecl::findSubpart( %s )\n", nam);
  int nsub = subparts.size();
  //debug_fprintf(stderr, "%d subparts\n", nsub);
  for (int i=0; i<nsub; i++) { 
    //debug_fprintf(stderr, "comparing '%s' to '%s'\n", typ, subparts[i]->vartype);
    if ( !strcmp( typ, subparts[i]->vartype )) return subparts[i];
  }
  //debug_fprintf(stderr, "chillAST_RecordDecl::findSubpart() couldn't find member of TYPE %s in ", typ); print(); printf("\n\n"); fflush(stdout); 

  return NULL;   
}


void chillAST_RecordDecl::print( int indent,  FILE *fp ) {
  //fprintf(fp, "chillAST_RecordDecl::print()\n"); 
  if (isUnnamed) return; 
  
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
  //debug_fprintf(stderr, "chillAST_RecordDecl::addVariableToSymbolTable() ignoring struct member %s vardecl\n", vd->varname); 
  return NULL; // damn, I hope nothing uses this! 
}

void chillAST_RecordDecl::printStructure( int indent,  FILE *fp ) {
  //debug_fprintf(stderr, "chillAST_RecordDecl::printStructure()\n"); 
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


chillAST_FunctionDecl::chillAST_FunctionDecl():body(this,0) {
  functionName = strdup("YouScrewedUp"); 
  forwarddecl = externfunc = builtin = false;
  uniquePtr = (void *) NULL;
  this->setFunctionCPU(); 
  //symbol_table = NULL;   // eventually, pointing to body's symbol table
  typedef_table = NULL;
  body = new chillAST_CompoundStmt();
};


chillAST_FunctionDecl::chillAST_FunctionDecl(const char *rt, const char *fname, chillAST_node *par):body(this,0) {
  returnType = strdup(rt);
  functionName = strdup(fname);
  this->setFunctionCPU(); 
  //debug_fprintf(stderr, "functionName %s\n", functionName); 
  forwarddecl = externfunc = builtin = false; 

  parent = par;
  if (par) par->getSourceFile()->addFunc( this );
  // symbol_table = NULL; //use body's instead
  typedef_table = NULL;
  body = new chillAST_CompoundStmt();
};


chillAST_FunctionDecl::chillAST_FunctionDecl(const char *rt, const char *fname, chillAST_node *par, void *unique)
    :body(this,0) {
  returnType = strdup(rt);
  functionName = strdup(fname);
  this->setFunctionCPU(); 
  //debug_fprintf(stderr, "functionName %s\n", functionName); 
  forwarddecl = externfunc = builtin = false; 

  body = new chillAST_CompoundStmt();
  uniquePtr = unique; // a quick way to check equivalence. DO NOT ACCESS THROUGH THIS
  parent = par;
  if (par) par->getSourceFile()->addFunc( this );
  //symbol_table = NULL; // use body's 
  typedef_table = NULL;
};


void chillAST_FunctionDecl::addParameter( chillAST_VarDecl *p) {
  debug_fprintf(stderr, "%s chillAST_FunctionDecl::addParameter( 0x%x  param %s)   total of %d parameters\n", functionName, p, p->varname, 1+parameters.size()); 

  if (symbolTableHasVariableNamed( &parameters, p->varname)) { // NOT recursive. just in FunctionDecl
    debug_fprintf(stderr, "chillAST_FunctionDecl::addParameter( %s ), parameter already exists?\n", p->varname);
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
  debug_fprintf(stderr, "chillAST_FunctionDecl::addDecl( %s )\n", vd->varname);
  if (!body) {  
    //debug_fprintf(stderr, "had no body\n"); 
    body = new chillAST_CompoundStmt();
    
    //body->symbol_table = symbol_table;   // probably wrong if this ever does something
  }

  //debug_fprintf(stderr, "before body->addvar(), func symbol table had %d entries\n", symbol_table->size()); 
  //debug_fprintf(stderr, "before body->addvar(), body symbol table was %p\n", body->symbol_table); 
  //debug_fprintf(stderr, "before body->addvar(), body symbol table had %d entries\n", body->symbol_table->size()); 
  //adds to body symbol table, and makes sure function has a copy. probably dumb
  body->symbol_table = body->addVariableToSymbolTable( vd ); 
  //debug_fprintf(stderr, "after body->addvar(), func symbol table had %d entries\n", symbol_table->size()); 
}

chillAST_VarDecl *chillAST_FunctionDecl::hasParameterNamed( const char *name ) { 
  int numparams = parameters.size();
  for (int i=0; i<numparams; i++) { 
    if (!strcmp(name, parameters[i]->varname)) return parameters[i];  // need to check type? 
  }
  return NULL; 
}


// similar to symbolTableHasVariableNamed() but returns the variable definition
chillAST_VarDecl *chillAST_FunctionDecl::funcHasVariableNamed( const char *name ) { // NOT recursive
  //debug_fprintf(stderr, "chillAST_FunctionDecl::funcHasVariableNamed( %s )\n", name );

  // first check the parameters
  int numparams = parameters.size();
  for (int i=0; i<numparams; i++) { 
    chillAST_VarDecl *vd = parameters[i];
    if (!strcmp(name, vd->varname)) { 
      //debug_fprintf(stderr, "yep, it's parameter %d\n", i); 
      return vd;  // need to check type? 
    }
  }
  //debug_fprintf(stderr, "no parameter named %s\n", name); 

  chillAST_SymbolTable *st = getSymbolTable(); 
  if (!st) {
    debug_fprintf(stderr,"and no symbol_table, so no variable named %s\n", name);
    return NULL; // no symbol table so no variable by that name 
  }

  
  int numvars =  st->size();
  //debug_fprintf(stderr, "checking against %d variables\n", numvars); 
  for (int i=0; i<numvars; i++) { 
    chillAST_VarDecl *vd = (*st)[i];
    //debug_fprintf(stderr, "comparing '%s' to '%s'\n", name, vd->varname); 
    if (!strcmp(name, vd->varname)) {
      //debug_fprintf(stderr, "yep, it's variable %d\n", i); 
      debug_fprintf(stderr, "%s was already defined in the function body\n", vd->varname); 
      return vd;  // need to check type? 
    }
  }
  debug_fprintf(stderr, "not a parameter or variable named %s\n", name); 
  return NULL; 
}




void chillAST_FunctionDecl::setBody( chillAST_node * bod ) {
  //debug_fprintf(stderr, "%s chillAST_FunctionDecl::setBody( 0x%x )   total of %d children\n", functionName, bod, 1+children.size()); 
  if (bod->isCompoundStmt())   body = (chillAST_CompoundStmt *)bod;
  else { 
    body = new chillAST_CompoundStmt();
    body->addChild( bod ); 
  }
  //symbol_table = body->getSymbolTable(); 
  //addChild(bod);
  bod->setParent( this );  // well, ... 
}

void  chillAST_FunctionDecl::printParameterTypes( FILE *fp ) {  // also prints names
  //debug_fprintf(stderr, "\n\n%s chillAST_FunctionDecl::printParameterTypes()\n", functionName); 
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
  fprintf(fp, "// isFromSourceFile ");
  if (filename) fprintf(fp, "%s  ", filename); 
  if (isFromSourceFile) fprintf(fp, "true\n"); 
  else fprintf(fp, "false\n"); 
  chillindent(indent, fp); 
  fprintf(fp, "(FunctionDecl %s %s(",  returnType, functionName );
  
  int numparameters = parameters.size(); 
  for (int i=0; i<numparameters; i++) {
    if (i!=0) fprintf(fp, ", "); 
    chillAST_VarDecl *p = parameters[i];
    //debug_fprintf(stderr, "param type %s  vartype %s\n", p->getTypeString(), p->vartype); 
    p->print(0, fp); // note: no indent, as this is in the function parens, ALSO print, not dump
  }
  fprintf(fp, ")\n"); // end of input parameters
  
  // now the body - 
  if (body) body->dump( indent+1 , fp); 

  // tidy up
  chillindent(indent, fp); 
  fprintf(fp, ")\n");
  fflush(fp); 
}
 

void chillAST_FunctionDecl::gatherVarDecls( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<numParameters(); i++) parameters[i]->gatherVarDecls( decls );
  //for (int i=0; i<children.size(); i++) children[i]->gatherVarDecls( decls );
  body->gatherVarDecls( decls );
}


void chillAST_FunctionDecl::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<numParameters(); i++) parameters[i]->gatherScalarVarDecls( decls );
  //for (int i=0; i<children.size(); i++) children[i]->gatherScalarVarDecls( decls );
  body->gatherScalarVarDecls( decls );
}


void chillAST_FunctionDecl::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  for (int i=0; i<numParameters(); i++) parameters[i]->gatherArrayVarDecls( decls );
  //for (int i=0; i<children.size(); i++) children[i]->gatherArrayVarDecls( decls );
  body->gatherArrayVarDecls( decls );
}


chillAST_VarDecl *chillAST_FunctionDecl::findArrayDecl( const char *name ) { 
  //debug_fprintf(stderr, "chillAST_FunctionDecl::findArrayDecl( %s )\n", name );
  chillAST_VarDecl *p = hasParameterNamed( name ); 
  //if (p) debug_fprintf(stderr, "function %s has parameter named %s\n", functionName, name );
  if (p && p->isArray()) return p;

  chillAST_VarDecl *v = funcHasVariableNamed ( name ); 
  //if (v) debug_fprintf(stderr, "function %s has symbol table variable named %s\n", functionName, name );
  if (v && v->isArray()) return v;

  // declared variables that may not be in symbol table but probably should be
  vector<chillAST_VarDecl*> decls ;
  gatherArrayVarDecls( decls );
  for (int i=0; i<decls.size(); i++) { 
    chillAST_VarDecl *vd = decls[i]; 
    if (0 == strcmp(vd->varname, name ) && vd->isArray()) return vd;
  }

  //debug_fprintf(stderr, "can't find array named %s in function %s \n", name, functionName); 
  return NULL; 
}

void chillAST_FunctionDecl::cleanUpVarDecls() {  
  //debug_fprintf(stderr, "\ncleanUpVarDecls() for function %s\n", functionName); 
  vector<chillAST_VarDecl*> used;
  vector<chillAST_VarDecl*> defined;
  vector<chillAST_VarDecl*> deletethese;

  gatherVarUsage( used ); 
  gatherVarDecls( defined ); 

  //debug_fprintf(stderr, "\nvars used: \n"); 
  //for ( int i=0; i< used.size(); i++) { 
  //used[i]->print(0, stderr);  debug_fprintf(stderr, "\n"); 
  //} 
  //debug_fprintf(stderr, "\n"); 
  //debug_fprintf(stderr, "\nvars defined: \n"); 
  //for ( int i=0; i< defined.size(); i++) { 
  //  defined[i]->print(0, stderr);  debug_fprintf(stderr, "\n"); 
  //} 
  //debug_fprintf(stderr, "\n"); 

  for ( int j=0; j < defined.size(); j++) { 
    //debug_fprintf(stderr, "j %d  defined %s\n", j, defined[j]->varname); 
    bool definedandused = false;
    for ( int i=0; i < used.size(); i++) {
      if (used[i] == defined[j]) { 
        //debug_fprintf(stderr, "i %d used %s\n", i, used[i]->varname); 
        //debug_fprintf(stderr, "\n");
        definedandused = true;
        break;
      }
    }

    if (!definedandused) { 
      if ( defined[j]->isParmVarDecl() ) { 
        //debug_fprintf(stderr, "we'd remove %s except that it's a parameter. Maybe someday\n", defined[j]->varname); 
      }
      else { 
        //debug_fprintf(stderr, "we can probably remove the definition of %s\n", defined[j]->varname); 
        deletethese.push_back(  defined[j] ); 
      }
    }
  }


  //debug_fprintf(stderr, "deleting %d vardecls\n", deletethese.size()); 
  for (int i=0; i<deletethese.size(); i++) { 
    //debug_fprintf(stderr, "deleting varDecl %s\n",  deletethese[i]->varname); 
    chillAST_node *par =  deletethese[i]->parent; 
    par->removeChild( par->findChild( deletethese[i] )); 
  }


  //debug_fprintf(stderr, "\n\nnow check for vars used but not defined\n"); 
  // now check for vars used but not defined?
  for ( int j=0; j < used.size(); j++) { 
    //debug_fprintf(stderr, "%s is used\n", used[j]->varname); 
    bool definedandused = false;
    for ( int i=0; i < defined.size(); i++) {
      if (used[j] == defined[i]) { 
        //debug_fprintf(stderr, "%s is defined\n", defined[i]->varname); 
        definedandused = true;
        break;
      }
    }
    if (!definedandused) { 
      //debug_fprintf(stderr, "%s is used but not defined?\n", used[j]->varname); 
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
   //debug_fprintf(stderr, "chillAST_FunctionDecl::constantFold()\n");
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
  rhsString = NULL;
  symbol_table = NULL;
  //rhsideString = NULL;
};


chillAST_MacroDefinition::chillAST_MacroDefinition(const char *mname, chillAST_node *par) { 
  macroName = strdup(mname);
  rhsString = NULL;
  parent = par;
  symbol_table = NULL;
  //rhsideString = NULL;

  if (par) par->getSourceFile()->addMacro( this );

  //debug_fprintf(stderr, "chillAST_MacroDefinition::chillAST_MacroDefinition( %s, ", mname); 
  //if (par) debug_fprintf(stderr, " parent NOT NULL);\n");
  //else debug_fprintf(stderr, " parent NULL);\n");
};


chillAST_MacroDefinition::chillAST_MacroDefinition(const char *mname, const char *rhs, chillAST_node *par) { 
  macroName = strdup(mname);
  rhsString = strdup(rhs);
  parent = par;
  symbol_table = NULL;

  if (par) par->getSourceFile()->addMacro( this );

  //debug_fprintf(stderr, "chillAST_MacroDefinition::chillAST_MacroDefinition( %s, ", mname); 
  //if (par) debug_fprintf(stderr, " parent NOT NULL);\n");
  //else debug_fprintf(stderr, " parent NULL);\n");
};


chillAST_node* chillAST_MacroDefinition::clone() {

  // TODO ?? cloning a macro makes no sense
  return this;
#ifdef CONFUSED 

  //debug_fprintf(stderr, "chillAST_MacroDefinition::clone() for %s\n", macroName); 
  chillAST_MacroDefinition *clo = new chillAST_MacroDefinition( macroName, parent); 
  for (int i=0; i<parameters.size(); i++) clo->addParameter( parameters[i] );
  clo->setBody( body->clone() );
  return clo; 
#endif 

}


void chillAST_MacroDefinition::setBody( chillAST_node * bod ) {
  debug_fprintf(stderr, "%s chillAST_MacroDefinition::setBody( 0x%x )\n", macroName, bod); 
  body = bod;
  debug_fprintf(stderr, "body is:\n"); body->print(0,stderr); debug_fprintf(stderr, "\n\n");
  rhsString = body->stringRep(); 
  bod->setParent( this );  // well, ... 
}


void chillAST_MacroDefinition::addParameter( chillAST_VarDecl *p) {
  //debug_fprintf(stderr, "%s chillAST_MacroDefinition::addParameter( 0x%x )   total of %d children\n", functionName, p, 1+children.size()); 
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
  if (rhsString) fprintf(fp, " (aka %s)"); 
  fprintf(fp, "\n"); 
  fflush(fp);
}


void chillAST_MacroDefinition::print(  int indent,  FILE *fp ) {  // UHOH   TODO 
  //fprintf(fp, "\n"); // ignore indentation
  //debug_fprintf(stderr, "macro has %d parameters\n", numParameters()); 

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




chillAST_ForStmt::chillAST_ForStmt():init(this, 0),cond(this,1),incr(this,2),body(this,3) {
  init = cond = incr = NULL;
  body = new chillAST_CompoundStmt();

  conditionoperator = IR_COND_UNKNOWN;
  symbol_table = NULL;
}


chillAST_ForStmt::chillAST_ForStmt(  chillAST_node *ini, chillAST_node *con, chillAST_node *inc, chillAST_node *bod, chillAST_node *par):chillAST_ForStmt() {
  parent = par; 
  init = ini;
  cond = con;
  incr = inc;
  body = bod;

  if (!cond->isBinaryOperator()) {
    debug_fprintf(stderr, "ForStmt conditional is of type %s. Expecting a BinaryOperator\n", cond->getTypeString());
    exit(-1); 
  }
  chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)cond();
  char *condstring = bo->op; 
  if (!strcmp(condstring, "<"))       conditionoperator = IR_COND_LT;
  else if (!strcmp(condstring, "<=")) conditionoperator = IR_COND_LE;
  else if (!strcmp(condstring, ">"))  conditionoperator = IR_COND_GT;
  else if (!strcmp(condstring, ">=")) conditionoperator = IR_COND_GE;
  else { 
    debug_fprintf(stderr, "ForStmt, illegal/unhandled end condition \"%s\"\n", condstring);
    debug_fprintf(stderr, "currently can only handle <, >, <=, >=\n");
    exit(1);
  }
}


bool chillAST_ForStmt::lowerBound( int &l ) { // l is an output (passed as reference)
  
  // above, cond must be a binaryoperator ... ??? 
  if (conditionoperator == IR_COND_LT || 
      conditionoperator == IR_COND_LE ) { 
    
    // lower bound is rhs of init 
    if (!init->isBinaryOperator()) { 
      debug_fprintf(stderr, "chillAST_ForStmt::lowerBound() init is not a chillAST_BinaryOperator\n");
      exit(-1);
    }
    
    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)init();
    if (!init->isAssignmentOp()) { 
      debug_fprintf(stderr, "chillAST_ForStmt::lowerBound() init is not an assignment chillAST_BinaryOperator\n");
      exit(-1);
    }
    
    //debug_fprintf(stderr, "rhs "); bo->rhs->print(0,stderr);  debug_fprintf(stderr, "   "); 
    l = bo->rhs->evalAsInt(); // float could be legal I suppose
    //debug_fprintf(stderr, "   %d\n", l); 
    return true; 
  }
  else if (conditionoperator == IR_COND_GT || 
           conditionoperator == IR_COND_GE ) {  // decrementing 
    // lower bound is rhs of cond (not init)
    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)cond();
    l = bo->rhs->evalAsInt(); // float could be legal I suppose
    return true; 
  }
  
  // some case we don't handle ?? 
  debug_fprintf(stderr, "chillAST_ForStmt::lowerBound() can't find lower bound of "); 
  print(0,stderr); 
  debug_fprintf(stderr, "\n\n"); 
  return false;      // or exit ???
}


bool chillAST_ForStmt::upperBound( int &u ) { // u is an output (passed as reference)
  
  // above, cond must be a binaryoperator ... ??? 
  if (conditionoperator == IR_COND_GT || 
      conditionoperator == IR_COND_GE ) {  // decrementing 

    // upper bound is rhs of init 
    if (!init->isBinaryOperator()) { 
      debug_fprintf(stderr, "chillAST_ForStmt::upperBound() init is not a chillAST_BinaryOperator\n");
      exit(-1);
    }

    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)init();
    if (!init->isAssignmentOp()) { 
      debug_fprintf(stderr, "chillAST_ForStmt::upperBound() init is not an assignment chillAST_BinaryOperator\n");
      exit(-1);
    }

    u = bo->rhs->evalAsInt(); // float could be legal I suppose
    return true; 
  }
  else if (conditionoperator == IR_COND_LT || 
           conditionoperator == IR_COND_LE ) { 
    //debug_fprintf(stderr, "upper bound is rhs of cond   ");
    // upper bound is rhs of cond (not init)
    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)cond();
    //bo->rhs->print(0,stderr);
    u = bo->rhs->evalAsInt(); // float could be legal I suppose

    if (conditionoperator == IR_COND_LT) u -= 1;  

    //debug_fprintf(stderr, "    %d\n", u);
    return true; 
  }

  // some case we don't handle ?? 
  debug_fprintf(stderr, "chillAST_ForStmt::upperBound() can't find upper bound of "); 
  print(0,stderr); 
  debug_fprintf(stderr, "\n\n"); 
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
  //debug_fprintf(stderr, "forstmt body type %s\n", Chill_AST_Node_Names[b->asttype] ); 
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
  if (b->getType() ==  CHILLAST_NODETYPE_BINARYOPERATOR) { // a single assignment statement
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

void chillAST_ForStmt::gatherStatements(std::vector<chillAST_node*> &statements ){
  
  // for completeness, should do all 4. Maybe someday
  //init->gatherStatements( statements ); 
  //cond->gatherStatements( statements ); 
  //incr->gatherStatements( statements ); 
  body->gatherStatements( statements ); 
}



void chillAST_ForStmt::addSyncs() {
  //debug_fprintf(stderr, "\nchillAST_ForStmt::addSyncs()  "); 
  //debug_fprintf(stderr, "for (");
  //init->print(0, stderr);
  //debug_fprintf(stderr, "; ");
  //cond->print(0, stderr);
  //debug_fprintf(stderr, "; ");
  //incr->print(0, stderr);
  //debug_fprintf(stderr, ")\n"); 
  
  if (!parent) { 
    debug_fprintf(stderr, "uhoh, chillAST_ForStmt::addSyncs() ForStmt has no parent!\n");
    debug_fprintf(stderr, "for (");
    init->print(0, stderr);
    debug_fprintf(stderr, "; ");
    cond->print(0, stderr);
    debug_fprintf(stderr, "; ");
    incr->print(0, stderr);
    debug_fprintf(stderr, ")\n"); 

    return; // exit? 
  }

  if (parent->isCompoundStmt()) { 
    //debug_fprintf(stderr, "ForStmt parent is CompoundStmt 0x%x\n", parent);
    vector<chillAST_node*> chillin = parent->getChildren();
    int numc = chillin.size();
    //debug_fprintf(stderr, "ForStmt parent is CompoundStmt 0x%x with %d children\n", parent, numc);
    for (int i=0; i<numc; i++) { 
      if (this == parent->getChild(i)) { 
        //debug_fprintf(stderr, "forstmt 0x%x is child %d of %d\n", this, i, numc); 
        chillAST_CudaSyncthreads *ST = new chillAST_CudaSyncthreads();
        parent->insertChild(i+1, ST);  // corrupts something ... 
        //debug_fprintf(stderr, "Create a call to __syncthreads() 2\n"); 
        //parent->addChild(ST);  // wrong, but safer   still kills 
      }
    }

    chillin = parent->getChildren();
    int nowc = chillin.size();
    //debug_fprintf(stderr, "old, new number of children = %d %d\n", numc, nowc); 
    
  }
  else { 
    debug_fprintf(stderr, "chillAST_ForStmt::addSyncs() unhandled parent type %s\n", parent->getTypeString()); 
    exit(-1); 
  }

  //debug_fprintf(stderr, "leaving addSyncs()\n"); 
}




void chillAST_ForStmt::removeSyncComment() { 
  //debug_fprintf(stderr, "chillAST_ForStmt::removeSyncComment()\n"); 
  if (metacomment && strstr(metacomment, "~cuda~") && strstr(metacomment, "preferredIdx: ")) { 
    char *ptr =  strlen( "preferredIdx: " ) + strstr(metacomment, "preferredIdx: ");
    *ptr = '\0'; 
  }
}


bool chillAST_ForStmt::findLoopIndexesToReplace(chillAST_SymbolTable *symtab, bool forcesync ) { 
  debug_fprintf(stderr, "\nchillAST_ForStmt::findLoopIndexesToReplace( force = %d )\n", forcesync); 
  //if (metacomment) debug_fprintf(stderr, "metacomment '%s'\n", metacomment); 

  bool force = forcesync;
  bool didasync = false;
  if (forcesync) { 
    //debug_fprintf(stderr, "calling addSyncs() because PREVIOUS ForStmt in a block had preferredIdx\n"); 
    addSyncs();
    didasync = true; 
  }
  
  //debug_fprintf(stderr, "chillAST_ForStmt::findLoopIndexesToReplace()\n"); 
  if (metacomment && strstr(metacomment, "~cuda~") && strstr(metacomment, "preferredIdx: ")) { 
    //debug_fprintf(stderr, "metacomment '%s'\n", metacomment); 
    
    char *copy = strdup(metacomment); 
    char *ptr  = strstr(copy, "preferredIdx: ");
    char *vname = ptr + strlen( "preferredIdx: " );
    char *space = strstr(vname, " "); // TODO index()
    if (space) { 
      //debug_fprintf(stderr, "vname = '%s'\n", vname); 
      force = true; 
    }

    if ((!didasync) && force ) { 
      //debug_fprintf(stderr, "calling addSyncs() because ForStmt metacomment had preferredIdx '%s'\n", vname); 
      addSyncs();
      removeSyncComment(); 
      didasync = true; 
    }

    if (space)   *space = '\0'; // if this is multiple words, grab the first one
    //debug_fprintf(stderr, "vname = '%s'\n", vname); 
    
    //debug_fprintf(stderr, "\nfor (");
    //init->print(0, stderr);
    //debug_fprintf(stderr, "; ");
    //cond->print(0, stderr);
    //debug_fprintf(stderr, "; ");
    //incr->print(0, stderr);
    //debug_fprintf(stderr, ")    %s\n", metacomment );
    //debug_fprintf(stderr, "prefer '%s'\n", vname );
    
    vector<chillAST_VarDecl*> decls;
    init->gatherVarLHSUsage( decls ); 
    //cond->gatherVarUsage( decls ); 
    //incr->gatherVarUsage( decls ); 
    //debug_fprintf(stderr, "forstmt has %d vardecls in init, cond, inc\n", decls.size()); 

    if ( 1 != decls.size()) { 
      debug_fprintf(stderr, "uhoh, preferred index in for statement, but multiple variables used\n");
      print(0,stderr);
      debug_fprintf(stderr, "\nvariables are:\n"); 
      for (int i=0; i<decls.size(); i++) { 
        decls[i]->print(0,stderr); debug_fprintf(stderr, "\n"); 
      }
      exit(0); 
    }
    chillAST_VarDecl* olddecl = decls[0];

    // RIGHT NOW, change all the references that this loop wants swapped out 
    // find vardecl for named preferred index.  it has to already exist
    debug_fprintf(stderr, "RIGHT NOW, change all the references that this loop wants swapped out \n"); 

    chillAST_VarDecl *newguy = findVariableNamed( vname ); // recursive
    if (!newguy) { 
      debug_fprintf(stderr, "there was no variable named %s anywhere I could find\n", vname);
    }

    // wrong - this only looks at variables defined in the forstmt, not 
    // in parents of the forstmt
    //int numsym = symtab->size(); 
    //debug_fprintf(stderr, "%d symbols\n", numsym);
    //for (int i=0; i<numsym; i++) { 
    //  debug_fprintf(stderr, "sym %d is '%s'\n", i, (*symtab)[i]->varname);
    //  if (!strcmp(vname,  (*symtab)[i]->varname)) { 
    //    newguy = (*symtab)[i];
    // }
    //}
    if (!newguy) { 
      debug_fprintf(stderr, "chillAST_ForStmt::findLoopIndexesToReplace() there is no defined variable %s\n", vname); 

      // make one ??  seems like this should never happen 
      newguy = new chillAST_VarDecl( olddecl->vartype, vname, ""/*?*/, NULL );
      // insert actual declaration in code location?   how?

      // find parent of the ForStmt?
      // find parent^n of the ForStmt that is not a Forstmt?
      // find parent^n of the Forstmt that is a FunctionDecl?
      chillAST_node *contain = findContainingNonLoop();
      if (contain == NULL) { 
        debug_fprintf(stderr, "nothing but loops all the way up?\n");
        exit(0);
      }
      debug_fprintf(stderr, "containing non-loop is a %s\n", contain->getTypeString()); 

      contain->print(0,stderr);
      contain->insertChild( 0, newguy ); // ugly order TODO
      contain->addVariableToSymbolTable( newguy ); // adds to first enclosing symbolTable
      
      if (!  symbolTableHasVariableNamed( contain->getSymbolTable(), vname )) { 
        debug_fprintf(stderr, "container doesn't have a var names %s afterwards???\n", vname); 
        exit(-1); 
      }
    }


    // swap out old for new in init, cond, incr, body 
    if (newguy) { 
      debug_fprintf(stderr, "\nwill replace %s with %s in init, cond, incr\n", olddecl->varname, newguy->varname); 
      debug_fprintf(stderr, "was: for (");
      init->print(0, stderr);
      debug_fprintf(stderr, "; ");
      cond->print(0, stderr);
      debug_fprintf(stderr, "; ");
      incr->print(0, stderr);
      debug_fprintf(stderr, ")\n");
      
      
      init->replaceVarDecls( olddecl, newguy ); 
      cond->replaceVarDecls( olddecl, newguy ); 
      incr->replaceVarDecls( olddecl, newguy ); 

      debug_fprintf(stderr, " is: for (");
      init->print(0, stderr);
      debug_fprintf(stderr, "; ");
      cond->print(0, stderr);
      debug_fprintf(stderr, "; ");
      incr->print(0, stderr);
      debug_fprintf(stderr, ")\n\n");

      debug_fprintf(stderr,"recursing to ForStmt body of type %s\n", body->getTypeString()); 
      body->replaceVarDecls( olddecl, newguy ); 

      debug_fprintf(stderr, "\nafter recursing to body, this loop is   (there should be no %s)\n", olddecl->varname);
      print(0, stderr); debug_fprintf(stderr, "\n"); 
      
    }
    
    //if (!space) // there was only one preferred
    //debug_fprintf(stderr, "removing metacomment\n"); 
    metacomment = NULL; // memleak

  }

  // check for more loops.  We may have already swapped variables out in body (right above here)
  body->findLoopIndexesToReplace( symtab, false ) ; 

  return force; 
}

void chillAST_ForStmt::replaceChild( chillAST_node *old, chillAST_node *newchild ){
  //debug_fprintf(stderr, "chillAST_ForStmt::replaceChild()  REALLY CALLING BODY->ReplaceCHILD\n"); 
  body->replaceChild( old, newchild );
}



void chillAST_ForStmt::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  // logic problem  if my loop var is olddecl! 

  //debug_fprintf(stderr, "chillAST_ForStmt::replaceVarDecls( old %s,  new %s )\n", olddecl->varname, newdecl->varname);

  // this is called for inner loops!
  init->replaceVarDecls( olddecl, newdecl ); 
  cond->replaceVarDecls( olddecl, newdecl ); 
  incr->replaceVarDecls( olddecl, newdecl ); 
  body->replaceVarDecls( olddecl, newdecl ); 
}


void chillAST_ForStmt::gatherLoopIndeces( std::vector<chillAST_VarDecl*> &indeces ) { 
  //debug_fprintf(stderr, "chillAST_ForStmt::gatherLoopIndeces()\nloop is:\n"); print(0,stderr); 

  vector<chillAST_VarDecl*> decls;
  init->gatherVarLHSUsage( decls ); 
  cond->gatherVarLHSUsage( decls ); 
  incr->gatherVarLHSUsage( decls ); 
  // note: NOT GOING INTO BODY OF THE LOOP

  int numdecls = decls.size(); 
  //debug_fprintf(stderr, "gatherLoopIndeces(), %d lhs vardecls for this ForStmt\n", numdecls);

  for (int i=0; i<decls.size(); i++)  {
    //debug_fprintf(stderr, "%s %p\n", decls[i]->varname, decls[i] );
    indeces.push_back( decls[i] ); 
  }
  
  // Don't forget to keep heading upwards!
  if (parent) {
    //debug_fprintf(stderr, "loop %p has parent of type %s\n", this, parent->getTypeString()); 
    parent->gatherLoopIndeces( indeces );
  }
  //else debug_fprintf(stderr, "this loop has no parent???\n");

}


void chillAST_ForStmt::gatherLoopVars(  std::vector<std::string> &loopvars ) {
  //debug_fprintf(stderr, "gathering loop vars for loop   for (");
  //init->print(0, stderr);
  //debug_fprintf(stderr, "; ");
  //cond->print(0, stderr);
  //debug_fprintf(stderr, "; ");
  //incr->print(0, stderr);
  //debug_fprintf(stderr, ")\n" );

  //init->dump(0, stderr); 


  vector<chillAST_VarDecl*> decls;
  init->gatherVarLHSUsage( decls ); 
  cond->gatherVarLHSUsage( decls ); 
  incr->gatherVarLHSUsage( decls ); 
  // note: NOT GOING INTO BODY OF THE LOOP
  
  for (int i=0; i<decls.size(); i++)  loopvars.push_back( strdup( decls[i]->varname )); 

}


void chillAST_ForStmt::loseLoopWithLoopVar( char *var ) { 

  //debug_fprintf(stderr, "\nchillAST_ForStmt::loseLoopWithLoopVar(  %s )\n", var ); 

  // now recurse (could do first, I suppose) 
  // if you DON'T do this first, you may have already replaced yourself with this loop body
  // the body will no longer have this forstmt as parent, it will have the forstmt's parent as its parent
  //debug_fprintf(stderr, "forstmt 0x%x, recursing loseLoop to body 0x%x of type %s with parent 0x%x of type %s\n", this, body,  body->getTypeString(), body->parent, body->parent->getTypeString()); 
  body->loseLoopWithLoopVar( var ) ; 




  // if *I* am a loop to be replaced, tell my parent to replace me with my loop body

  std::vector<std::string> loopvars;
  gatherLoopVars( loopvars );
  
  if (loopvars.size() != 1) { 
    debug_fprintf(stderr, "uhoh, loop has more than a single loop var and trying to loseLoopWithLoopVar()\n");
    print(0,stderr);
    debug_fprintf(stderr, "\nvariables are:\n"); 
    for (int i=0; i<loopvars.size(); i++) { 
      debug_fprintf(stderr, "%s\n", loopvars[i].c_str()); 
    }
    
    exit(-1); 
  }
  
  //debug_fprintf(stderr, "my loop var %s, looking for %s\n", loopvars[0].c_str(), var );
  if (!strcmp(var,  loopvars[0].c_str())) { 
    //debug_fprintf(stderr, "OK, trying to lose myself!    for (");
    //init->print(0, stderr);
    //debug_fprintf(stderr, "; ");
    //cond->print(0, stderr);
    //debug_fprintf(stderr, "; ");
    //incr->print(0, stderr);
    //debug_fprintf(stderr, ")\n" );   

    if (!parent) { 
      debug_fprintf(stderr, "chillAST_ForStmt::loseLoopWithLoopVar()  I have no parent!\n");
      exit(-1);
    }

    vector<chillAST_VarDecl*> decls;
    init->gatherVarLHSUsage( decls );   // this can fail if init is outside the loop
    cond->gatherVarLHSUsage( decls ); 
    incr->gatherVarLHSUsage( decls ); 
    if (decls.size() > 1) { 
      debug_fprintf(stderr, "chill_ast.cc multiple loop variables confuses me\n");
      exit(-1); 
    }
    chillAST_node *newstmt = body; 

    // ACTUALLY, if I am being replaced, and my loop conditional is a min (Ternary), then wrap my loop body in an if statement
    if (cond->isBinaryOperator()) { // what else could it be?
      chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *) cond();
      if (BO->rhs->isTernaryOperator()) { 

        chillAST_TernaryOperator *TO = (chillAST_TernaryOperator *)BO->rhs();
        chillAST_BinaryOperator *C =  (chillAST_BinaryOperator *)TO->condition();
        
        //debug_fprintf(stderr, "loop condition RHS  is ternary\nCondition RHS");
        C->print(); printf("\n"); fflush(stdout); 
        chillAST_node *l = C->lhs;
        if (l->isParenExpr()) l = ((chillAST_ParenExpr *)l)->subexpr; 
        chillAST_node *r = C->rhs;
        if (r->isParenExpr()) r = ((chillAST_ParenExpr *)r)->subexpr; 

        //debug_fprintf(stderr, "lhs is %s     rhs is %s\n", l->getTypeString(), r->getTypeString()); 
        
        chillAST_node *ifcondrhs = NULL; 
        if (!(l->isConstant())) ifcondrhs = l;
        else if (!(r->isConstant())) ifcondrhs = r;
        else { 
          // should never happen. 2 constants. infinite loop
          debug_fprintf(stderr, "chill_ast.cc INIFNITE LOOP?\n"); 
          this->print(0,stderr); debug_fprintf(stderr, "\n\n");
          exit(-1);
        }
        
        // wrap the loop body in an if
        chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( decls[0] ); 
        chillAST_BinaryOperator *ifcond = new chillAST_BinaryOperator( DRE, "<=", ifcondrhs ); 
        chillAST_IfStmt *ifstmt = new chillAST_IfStmt( ifcond, body, NULL, NULL ); 
        
        newstmt = ifstmt; 
      }
    }

    //debug_fprintf(stderr, "forstmt 0x%x has parent 0x%x  of type %s\n", this, parent, parent->getTypeString()); 
    //debug_fprintf(stderr, "forstmt will be replaced by\n");
    //newstmt->print(0,stderr); debug_fprintf(stderr, "\n\n"); 

    parent->replaceChild( this, newstmt );
  }


}





chillAST_BinaryOperator::chillAST_BinaryOperator():lhs(this,0),rhs(this,1) {
  //debug_fprintf(stderr, "chillAST_BinaryOperator::chillAST_BinaryOperator()  %p\n", this);
  op = NULL;
}


chillAST_BinaryOperator::chillAST_BinaryOperator(chillAST_node *l, const char *oper, chillAST_node *r, chillAST_node *par):chillAST_BinaryOperator() {
  //debug_fprintf(stderr, "chillAST_BinaryOperator::chillAST_BinaryOperator( l %p  %s  r %p  par %p)  %p\n", l, oper, r, par, this); 

  //if (l && r ) { 
  //  debug_fprintf(stderr, "("); l->print(0,stderr); debug_fprintf(stderr, ") %s (", oper); r->print(0,stderr); debug_fprintf(stderr, ")\n\n"); 
  //} 

  lhs = l;
  rhs = r;
  parent = par;

  if (lhs) lhs->setParent( this );  
  if (rhs) rhs->setParent( this );  // may only have part of the lhs and rhs when binop is created
  op = strdup(oper);

  // if this writes to lhs and lhs type has an 'imwrittento' concept, set that up
  if (isAssignmentOp()) { 
    if (lhs && lhs->isArraySubscriptExpr()) {
      ((chillAST_ArraySubscriptExpr*)lhs())->imwrittento = true;
      //debug_fprintf(stderr, "chillAST_BinaryOperator, op '=', lhs is an array reference  LVALUE\n"); 
    }
  }
  if (isAugmentedAssignmentOp()) {  // +=  etc 
    //debug_fprintf(stderr, "isAugmentedAssignmentOp()  "); print(); fflush(stdout); 
    if (lhs && lhs->isArraySubscriptExpr()) { 
      //debug_fprintf(stderr, "lhs is also read from  ");  lhs->print(); fflush(stdout); 
      ((chillAST_ArraySubscriptExpr*)lhs())->imreadfrom = true; // note will ALSO have imwrittento true
    }
  }
}


int chillAST_BinaryOperator::evalAsInt() { 
  // very limited. allow +-*/ and integer literals ...
  if (isAssignmentOp()) return rhs->evalAsInt();  // ?? ignores/loses lhs info 

  if (!strcmp("+", op)) { 
    //debug_fprintf(stderr, "chillAST_BinaryOperator::evalAsInt()   %d + %d\n", lhs->evalAsInt(), rhs->evalAsInt()); 
    return lhs->evalAsInt() + rhs->evalAsInt(); 
  }
  if (!strcmp("-", op)) return lhs->evalAsInt() - rhs->evalAsInt(); 
  if (!strcmp("*", op)) return lhs->evalAsInt() * rhs->evalAsInt(); 
  if (!strcmp("/", op)) return lhs->evalAsInt() / rhs->evalAsInt(); 
  
  debug_fprintf(stderr, "chillAST_BinaryOperator::evalAsInt() unhandled op '%s'\n", op); 
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
      //  debug_fprintf(stderr, "\nlhs 0x%x isImplicitCastExpr()\n", lhs);
      //  debug_fprintf(stderr, "lhs subexpr 0x%x\n", ((chillAST_ImplicitCastExpr*)lhs)->subexpr);
      //  debug_fprintf(stderr, "lhs subexpr type %s\n", ((chillAST_ImplicitCastExpr*)lhs)->subexpr->getTypeString());
      //   
      if (((chillAST_ImplicitCastExpr*)lhs())->subexpr->isNotLeaf()) needparens = true;
    } 
    else if (lhs->isNotLeaf())  { 
      if      (isMinusOp()     && lhs->isPlusOp())     needparens = false;
      else if (isPlusMinusOp() && lhs->isMultDivOp())  needparens = false;
      else needparens = true;
    }
  }

  //debug_fprintf(stderr, "\n\nbinop    "); 
  //lhs->printonly(0,stderr); 
  //debug_fprintf(stderr," %s ",op); 
  //rhs->printonly(0,stderr); 
  //debug_fprintf(stderr,"\n"); 
  //debug_fprintf(stderr, "op is %s   lhs %s   rhs %s\n", op, lhs->getTypeString(), rhs->getTypeString());
  //debug_fprintf(stderr, "lhs "); lhs->printonly(0, stderr); debug_fprintf(stderr, "    "); 
  //debug_fprintf(stderr, "lhs needparens = %d\n", needparens); 


  if (needparens) fprintf(fp, "(");
  if (lhs) lhs->print( 0, fp );
  else fprintf(fp, "(NULL)"); 
  if (needparens) fprintf(fp, ")"); 

  fprintf( fp, " %s ", op);

  needparens = false;
  //debug_fprintf(stderr, "binop rhs is of type %s\n", rhs->getTypeString()); 
  if (rhs) { 
    if (rhs->isImplicitCastExpr()) { 
      if (((chillAST_ImplicitCastExpr*)rhs())->subexpr->isNotLeaf()) needparens = true;
    } 
    //else if (rhs->isNotLeaf()) needparens = true; // too many parens. test too simple
    else if (rhs->isNotLeaf()) { 
      // really need the precedence ordering, and check relative of op and rhs op
      if      (isMinusOp() ) needparens = true;    // safer.  perhaps complicated thing on rhs of a minus
      else if (isPlusMinusOp() && rhs->isMultDivOp())  needparens = false;
      else needparens = true;
    }
  }
  //debug_fprintf(stderr, "rhs "); rhs->printonly(0, stderr); debug_fprintf(stderr, "    "); 
  //debug_fprintf(stderr, "rhs needparens = %d\n\n", needparens); 
  //if (!needparens) debug_fprintf(stderr, "rhs isNotLeaf() = %d\n", rhs->isNotLeaf()); 

  if (needparens) fprintf(fp, "(");
  if (rhs) rhs->print( 0, fp );
  else fprintf(fp, "(NULL)"); 
  if (needparens) fprintf(fp, ")"); 
  fflush(fp); 
  printPreprocAFTER(indent, fp); 

}


char *chillAST_BinaryOperator::stringRep(int indent ) { 
  std::string s = string( lhs->stringRep() ) + " " + op + " " +  string(lhs->stringRep() );
  return strdup( s.c_str() ); 
}



void chillAST_BinaryOperator::printonly( int indent, FILE *fp ) {

  lhs->printonly( indent, fp );
  fprintf( fp, " %s ", op);
  rhs->printonly( 0, fp );
  fflush(fp); 



}


class chillAST_node* chillAST_BinaryOperator::constantFold() { 
  //debug_fprintf(stderr, "\nchillAST_BinaryOperator::constantFold()  ");
  //print(0,stderr); debug_fprintf(stderr, "\n");

  lhs = lhs->constantFold();
  rhs = rhs->constantFold();
  
  chillAST_node *returnval = this;

  if (lhs->isConstant() && rhs->isConstant() ) { 
    //debug_fprintf(stderr, "binop folding constants\n"); print(0,stderr); debug_fprintf(stderr, "\n");

    if (streq(op, "+") || streq(op, "-") || streq(op, "*")) { 
      if (lhs->isIntegerLiteral() && rhs->isIntegerLiteral()) {
        chillAST_IntegerLiteral *l = (chillAST_IntegerLiteral *)lhs();
        chillAST_IntegerLiteral *r = (chillAST_IntegerLiteral *)rhs();
        chillAST_IntegerLiteral *I;
        
        if (streq(op, "+")) I = new chillAST_IntegerLiteral(l->value+r->value, parent);
        if (streq(op, "-")) I = new chillAST_IntegerLiteral(l->value-r->value, parent);
        if (streq(op, "*")) I = new chillAST_IntegerLiteral(l->value*r->value, parent);

        returnval = I;
        //debug_fprintf(stderr, "%d %s %d becomes %d\n", l->value,op, r->value, I->value);
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
    //else debug_fprintf(stderr, "can't fold op '%s' yet\n", op); 
  }

  //debug_fprintf(stderr, "returning "); returnval->print(0,stderr); debug_fprintf(stderr, "\n"); 
  return returnval;
}


class chillAST_node* chillAST_BinaryOperator::clone() { 
  //debug_fprintf(stderr, "chillAST_BinaryOperator::clone() "); print(); printf("\n"); fflush(stdout); 

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
  //debug_fprintf(stderr, "chillAST_BinaryOperator::gatherArrayRefs()\n"); 
  //print(); fflush(stdout); debug_fprintf(stderr, "\n"); 
  //if (isAugmentedAssignmentOp()) { 
  //  debug_fprintf(stderr, "%s  is augmented assignment\n", op);
  //}

  //if (isAssignmentOp()) { 
  //  debug_fprintf(stderr, "%s  is assignment\n", op);
  //}

  //if (isAugmentedAssignmentOp()) { // lhs is ALSO on the RHS, NOT as a write
  //  if (lhs->isArraySubscriptExpr()) { // probably some case where this fails
  //    ((chillAST_ArraySubscriptExpr *) lhs)->imreadfrom = true;  
  //    //lhs->&gatherArrayRefs( refs, 0 );
  //  }
  //} 

  //debug_fprintf(stderr, "in chillAST_BinaryOperator::gatherArrayRefs(), %d &arrayrefs before\n", refs.size());  
  lhs->gatherArrayRefs( refs, isAssignmentOp() );
  //debug_fprintf(stderr, "in chillAST_BinaryOperator::gatherArrayRefs(), %d &arrayrefs after lhs\n", refs.size());  
  rhs->gatherArrayRefs( refs, 0 );
  //debug_fprintf(stderr, "in chillAST_BinaryOperator::gatherArrayRefs(), %d &refs\n", refs.size()); 
  
  //for (int i=0; i<refs.size(); i++) { 
  //  debug_fprintf(stderr, "%s\n", (*refs)[i]->basedecl->varname); 
  //} 

}

void chillAST_BinaryOperator::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  lhs->gatherScalarRefs( refs, isAssignmentOp() );
  rhs->gatherScalarRefs( refs, 0 );
} 


void chillAST_BinaryOperator::replaceChild( chillAST_node *old, chillAST_node *newchild ) {
  //debug_fprintf(stderr, "\nbinop::replaceChild( old 0x%x, new )    lhs 0x%x   rhd 0x%x\n", old, lhs, rhs);

  // will pointers match??
  if      (lhs == old) setLHS( newchild ); 
  else if (rhs == old) setRHS( newchild ); 
  
  // silently ignore? 
  //else { 
  //  debug_fprintf(stderr, "\nERROR chillAST_BinaryOperator::replaceChild( old 0x%x, new )    lhs 0x%x   rhd 0x%x\n", old, lhs, rhs);
  //  debug_fprintf(stderr, "old is not a child of this BinaryOperator\n");
  //  print();
  //  dump();
  //  exit(-1); 
  //} 
}

void chillAST_BinaryOperator::gatherStatements(std::vector<chillAST_node*> &statements ){
  
  // what's legit?
  if (isAssignmentOp()) { 
    statements.push_back( this );
  }

}

void chillAST_BinaryOperator::gatherVarLHSUsage( vector<chillAST_VarDecl*> &decls ) {
  lhs->gatherVarUsage( decls );
}

 void chillAST_BinaryOperator::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
   //if (!strcmp(op, "<=")) { 
   //  debug_fprintf(stderr, "chillAST_BinaryOperator::replaceVarDecls( old %s, new %s)\n", olddecl->varname, newdecl->varname );
   //  print(); printf("\n"); fflush(stdout); 
   //  debug_fprintf(stderr, "binaryoperator, lhs is of type %s\n", lhs->getTypeString()); 
   //  debug_fprintf(stderr, "binaryoperator, rhs is of type %s\n", rhs->getTypeString()); 
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




chillAST_TernaryOperator::chillAST_TernaryOperator():condition(this,0),lhs(this,1),rhs(this,2) {
  op = strdup("?"); // the only one so far
}

chillAST_TernaryOperator::chillAST_TernaryOperator(const char *oper, chillAST_node *c, chillAST_node *l, chillAST_node *r, chillAST_node *par):chillAST_TernaryOperator() {
  op = strdup(oper);
  condition = c;
  lhs = l;
  rhs = r;
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
  //debug_fprintf(stderr, "\nbinop::replaceChild( old 0x%x, new )    lhs 0x%x   rhd 0x%x\n", old, lhs, rhs);

  // will pointers match??
  if      (lhs == old) setLHS( newchild ); 
  else if (rhs == old) setRHS( newchild ); 
  else if (condition == old) setCond( newchild );

  // silently ignore? 
  //else { 
  //}
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
    //debug_fprintf(stderr, "ternop folding constants\n");
    //print(0,stderr);
    //debug_fprintf(stderr, "\n");

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
        //debug_fprintf(stderr, "%d %s %d becomes %d\n", l->value,op, r->value, I->value);
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
    else debug_fprintf(stderr, "can't fold op '%s' yet\n", op); 
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







chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr():base(this,0),index(this,1) {
  base = index = NULL;
  basedecl = NULL; //debug_fprintf(stderr, "setting basedecl NULL for ASE %p\n", this); 
  imwrittento = false; // ?? 
  imreadfrom  = false; // ?? 
}



chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, chillAST_node *par, void *unique ):chillAST_ArraySubscriptExpr() {

  //debug_fprintf(stderr, "\nchillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 1\n");
  //debug_fprintf(stderr, "ASE index %p ", indx); indx->print(0,stderr); debug_fprintf(stderr, "\n"); 
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
  //debug_fprintf(stderr,"chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() original = 0x%x\n", uniquePtr); 
  //debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 1 calling multibase()\n"); 
  basedecl = multibase();//debug_fprintf(stderr, "%p  ASE 1 basedecl = %p\n",this,basedecl); 
  //basedecl->print(); printf("\n");
  //basedecl->dump(); printf("\n"); fflush(stdout); 
  //debug_fprintf(stderr, "basedecl varname %s\n", basedecl->varname); 

  //debug_fprintf(stderr, "\nASE %p   parent %p  ", this, parent); print(0,stderr); debug_fprintf(stderr, "\n\n"); 
}



chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, bool writtento, chillAST_node *par, void  *unique ):chillAST_ArraySubscriptExpr() {
  //debug_fprintf(stderr, "\nchillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 2  parent %p\n", par ); 
  //debug_fprintf(stderr, "ASE %p   index %p ", this, indx); indx->print(0,stderr); debug_fprintf(stderr, "\n"); 
  
  bas->setParent( this );
  if (bas->isImplicitCastExpr()) base = ((chillAST_ImplicitCastExpr*)bas)->subexpr; // probably wrong
  else base = bas;

  if (indx->isImplicitCastExpr()) index = ((chillAST_ImplicitCastExpr*)indx)->subexpr; // probably wrong
  else index = indx;
  
  //debug_fprintf(stderr, "setting parent of base  %p to %p\n", base, this);
  //debug_fprintf(stderr, "setting parent of index %p to %p\n", index, this);
  base->setParent( this );
  index->setParent( this ); 
  
  imwrittento = writtento; // ?? 
  //debug_fprintf(stderr, "ASE %p   imwrittento %d\n", this, imwrittento);
  imreadfrom  = false; // ??  

  uniquePtr = (void *) unique;
  //debug_fprintf(stderr,"chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() original = 0x%x\n", uniquePtr); 

  basedecl = multibase(); 

  //debug_fprintf(stderr, "%p  ASE 2 basedecl = %p\n", this, basedecl); 
  //printf("basedecl is  "); fflush(stdout); basedecl->print(); printf("\n"); fflush(stdout); 
  //basedecl->dump(); printf("\n"); fflush(stdout);
  //debug_fprintf(stderr, "basedecl varname %s\n", basedecl->varname); 

  //debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 2 DONE\n");
  //print(0,stderr); debug_fprintf(stderr, "\n\n"); 
  //debug_fprintf(stderr, "\nASE %p   parent %p  ", this, parent); print(0,stderr); debug_fprintf(stderr, "\n\n"); 
 }



chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr( chillAST_VarDecl *v, std::vector<chillAST_node *> indeces,  chillAST_node *par):chillAST_ArraySubscriptExpr() {
  //debug_fprintf(stderr, "\nchillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr() 4\n"); 
  //debug_fprintf(stderr,"chillAST_ArraySubscriptExpr( chillAST_VarDecl *v, std::vector<int> indeces)\n");
  parent = par;
  //if (parent == NULL) { 
  //  debug_fprintf(stderr, "dammit.  ASE %p has no parent\n", this); 
  //} 


  int numindeces = indeces.size();
  for (int i=0; i<numindeces; i++) { 
    debug_fprintf(stderr, "ASE index %d  ", i); indeces[i]->print(0,stderr); debug_fprintf(stderr, "\n"); 
  //  printf("[");
  //  indeces[i]->print();
  //  printf("]");
  } 
  //fflush(stdout); 
  //debug_fprintf(stderr, "\n");
  
  chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( v->vartype, v->varname, v, NULL);
  basedecl = v; // ?? 
  //debug_fprintf(stderr, "%p  ASE 3 basedecl = %p   ", this, basedecl); 
  //debug_fprintf(stderr, "of type %s\n", basedecl->getTypeString()); 
  //basedecl->print(); printf("\n");
  //basedecl->dump(); printf("\n"); fflush(stdout); 
  //debug_fprintf(stderr, "basedecl varname %s\n", basedecl->varname); 
  
  chillAST_ArraySubscriptExpr *rent = this; // parent for subnodes
  
  // these are on the top level ASE that we're creating here 
  base = (chillAST_node *) DRE;
  index = indeces[ numindeces-1];

  base->setParent( this );
  index->setParent(this); 

  for (int i=numindeces-2; i>=0; i--) {
    
    chillAST_ArraySubscriptExpr *ASE = new  chillAST_ArraySubscriptExpr( DRE, indeces[i], rent, 0); 
    rent->base = ASE; // 
    rent = ASE;
  }
  
  imwrittento = false;
  imreadfrom = false; 
  //debug_fprintf(stderr, "ASE is "); print(); printf("\n\n"); fflush(stdout); 

  //debug_fprintf(stderr, "\nASE %p   parent %p  ", this, parent); print(0,stderr); debug_fprintf(stderr, "\n\n"); 
}



chillAST_node *chillAST_node::getEnclosingStatement( int level ) {  // TODO do for subclasses?

  //debug_fprintf(stderr, "chillAST_node::getEnclosingStatement( level %d ) node type %s\n", level, getTypeString());
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

  debug_fprintf(stderr, "getEnclosingStatement() level %d type %s, returning NULL\n", level, getTypeString()); 
  segfault(); 

  return NULL;
}



void chillAST_ArraySubscriptExpr::gatherIndeces(std::vector<chillAST_node*>&ind) { 
  if (base->isArraySubscriptExpr()) ((chillAST_ArraySubscriptExpr *)base())->gatherIndeces( ind );
  ind.push_back( index );
}



void chillAST_ArraySubscriptExpr::dump( int indent, FILE *fp ) {
//  debug_fprintf(stderr, "\n%p chillAST_ArraySubscriptExpr::dump()  basedecl %p\n", basedecl);
  
  char *local;
  if (basedecl && basedecl->vartype) {
    local = strdup( basedecl->vartype );
  }
  else { 
    debug_fprintf(stderr, "%p chillAST_ArraySubscriptExpr::dump(), no basedecl ???\n",this);
    local = strdup("");
    //debug_fprintf(stderr, "base is "); base->dump(); printf("\n"); base->print(); printf("\n"); fflush(stdout); 
    //print(); printf("\n"); fflush(stdout);
  }


  char *space = rindex(local, ' ');  // can't use index because it's a class member!
  if (space) *space = '\0';  // turn "float *" into "float"

  chillindent(indent, fp);
  //fprintf(fp, "(ArraySubscriptExpr '%s' ", local);
  if (basedecl)  { 
    //debug_fprintf(stderr, " chillAST_ArraySubscriptExpr::dump() basedecl is of type %s\n",   basedecl->getTypeString()); 
    fprintf(fp, "(ArraySubscriptExpr (%s) '%s' ", basedecl->varname, local); 
  }
  else debug_fprintf(stderr, " chillAST_ArraySubscriptExpr::dump() has no basedecl\n"); 
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

char *chillAST_ArraySubscriptExpr::stringRep(int indent ) { 
  debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::stringRep\n"); 

  char *blurb;
  char *b = base->stringRep(0); 
  char *i = index->stringRep(0); 
  // combine.  shoudl be using strings. much cleaner  TODO
  std::string s = string(b) + "[" + string(i) + "]";
  debug_fprintf(stderr, "ASE stringrep %s\n", s.c_str()); 
  return strdup( s.c_str()); 
  

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
  //debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::multibase()  base of type %s\n", base->getTypeString()); 
  
  return base->multibase();  

  // this will be used to SET basedecl
  //basedecl = NULL; // do this so we don't confuse ourselves looking at uninitialized basedecl

  chillAST_node *b = base; 
  //debug_fprintf(stderr, "base is of type %s\n", b->getTypeString());

  if (!b) return NULL; // just in case ??

  if (base->getType() == CHILLAST_NODETYPE_IMPLICITCASTEXPR) { // bad coding
    b = ((chillAST_ImplicitCastExpr*)b)->subexpr;
  }

  if (b->getType() == CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR) { // multidimensional array!
    // recurse
    return ((chillAST_ArraySubscriptExpr *)b)->multibase();
  }

  if (b->getType() == CHILLAST_NODETYPE_DECLREFEXPR) return(((chillAST_DeclRefExpr*)b)->getVarDecl());

  
  if (b->isBinaryOperator()) { 
    // presumably a dot or pointer ref that resolves to an array
    chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *) b;
    if ( strcmp(BO->op, ".") ) { 
      debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::multibase(), UNHANDLED case:\n");
      debug_fprintf(stderr, "base is binary operator, of type %s\n", BO->op); 
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
    //debug_fprintf(stderr, "multibase() Member Expression "); ME->print(); printf("\n"); fflush(stdout); 

    chillAST_node *n = ME->base; //  WRONG   want the MEMBER
    //debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::multibase()  Member Expression base of type %s\n", n->getTypeString());
    //debug_fprintf(stderr, "base is "); ME->base->dump(); 

    // NEED to be able to get lowest level recorddecl or typedef from this base

    debug_fprintf(stderr, "chillast.cc, L2315, bailing??\n"); 
    exit(0); 

    if (!n->isDeclRefExpr()) { 
      debug_fprintf(stderr, "MemberExpr member is not chillAST_DeclRefExpr\n");
      exit(-1);
    }
    chillAST_DeclRefExpr *DRE = (chillAST_DeclRefExpr *)n;
    n = DRE->decl;
    //debug_fprintf(stderr, "DRE decl is of type %s\n", n->getTypeString()); 
    assert( n->isVarDecl() );
    chillAST_VarDecl *vd = (chillAST_VarDecl *) n;
    vd->print(); printf("\n"); fflush(stdout); 

    chillAST_TypedefDecl *tdd = vd->typedefinition; 
    chillAST_RecordDecl  *rd  = vd->vardef; 
    //debug_fprintf(stderr, "tdd %p    rd %p\n", tdd, rd); 
    
    print(); printf("\n"); 
    dump();  printf("\n"); fflush(stdout);

    assert( tdd != NULL || rd != NULL );
    
    chillAST_VarDecl *sub;
    if (tdd) sub = tdd->findSubpart( ME->member ); 
    if (rd)  sub =  rd->findSubpart( ME->member ); 

    //debug_fprintf(stderr, "subpart is "); sub->print(); printf("\n"); fflush(stdout); 
    
    return sub; // what if the sub is an array ??  TODO 
  }


  debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::multibase(), UNHANDLED case %s\n", 
          b->getTypeString()); 
  print(); printf("\n"); fflush(stdout);
  debug_fprintf(stderr, "base is: "); b->print(); printf("\n"); fflush(stdout);
  segfault(); 
}


chillAST_node *chillAST_ArraySubscriptExpr::getIndex(int dim) {
  //debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::getIndex( %d )\n", dim); 

  chillAST_node *b = base; 

  int depth = 0;
  std::vector<chillAST_node*> ind;
  chillAST_node *curindex = index;
  for (;;) { 
    if (b->getType() == CHILLAST_NODETYPE_IMPLICITCASTEXPR) b = ((chillAST_ImplicitCastExpr*)b)->subexpr;
    else if (b->getType() == CHILLAST_NODETYPE_ARRAYSUBSCRIPTEXPR) {
      //debug_fprintf(stderr, "base  "); b->print(); debug_fprintf(stderr, "\n"); 
      //debug_fprintf(stderr, "index "); curindex->print(); debug_fprintf(stderr, "\n"); 
      ind.push_back(curindex);
      curindex = ((chillAST_ArraySubscriptExpr*)b)->index;
      b = ((chillAST_ArraySubscriptExpr*)b)->base; 
      depth++;
    }
    else { 
      //debug_fprintf(stderr, "base  "); b->print(); debug_fprintf(stderr, "\n"); 
      //debug_fprintf(stderr, "index "); curindex->print(); debug_fprintf(stderr, "\n"); 
      //debug_fprintf(stderr, "stopping at base type %s\n", b->getTypeString());
      ind.push_back(curindex);
      break; 
    }
  }
  //debug_fprintf(stderr, "depth %d\n", depth );
  //for (int i=0; i<ind.size(); i++) { ind[i]->print(); debug_fprintf(stderr, "\n"); } 

  return ind[ depth - dim ]; 
  /* 
  if (dim == 0) return index; // single dimension 
  debug_fprintf(stderr, "DIM NOT 0\n"); 
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

  debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::getIndex() failed\n");
  */ 
  exit(-1); 
}




class chillAST_node* chillAST_ArraySubscriptExpr::constantFold() { 
  //debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::constantFold()\n"); 
  base  =  base->constantFold();
  index = index->constantFold();
  return this;
}

class chillAST_node* chillAST_ArraySubscriptExpr::clone() { 
  //debug_fprintf(stderr,"chillAST_ArraySubscriptExpr::clone() old imwrittento %d\n", imwrittento);
  //debug_fprintf(stderr, "cloning ASE %p ", this); print(0,stderr); printf(" with parent %p\n", parent); fflush(stdout);
  //debug_fprintf(stderr, "base %p  base->parent %p     index %p  index->parent %p\n", base, base->parent, index, index->parent);  

  //debug_fprintf(stderr, "old base   "); base->print();  printf("\n"); fflush(stdout);
  //debug_fprintf(stderr, "old base   "); base->dump();  printf("\n"); fflush(stdout);
  if (base->isDeclRefExpr()) { 
    chillAST_VarDecl *vd = (chillAST_VarDecl *)(((chillAST_DeclRefExpr *)base())->decl);
    //debug_fprintf(stderr, "old decl   "); vd->print();  printf("\n");fflush(stdout);
    //debug_fprintf(stderr, "old decl   "); vd->dump();   printf("\n");fflush(stdout);
  }
  chillAST_node *b =  base->clone();
  //debug_fprintf(stderr, "new base   "); b->print();  printf("\n"); fflush(stdout);
  //debug_fprintf(stderr, "new base   "); b->dump();  printf("\n"); fflush(stdout);

  chillAST_node *i = index->clone();
  //debug_fprintf(stderr, "new index  "); i->print();  printf("\n"); fflush(stdout);

  
  //if (!index->parent) { 
  //  debug_fprintf(stderr, "ASE %p SOURCE OF CLONE INDEX %p of type %s HAS NO PARENT\n", this, index, index->getTypeString());
  //  debug_fprintf(stderr, "ASE SOURCE IS  "); print(0,stderr); debug_fprintf(stderr, "\n\n");
  //} 
  //debug_fprintf(stderr, "cloning AST %p, after cloning base and index, creating a new ASE\n", this); 
  chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( b, i, imwrittento, parent, uniquePtr /* ?? */ ); 
  //debug_fprintf(stderr, "cloned AST will be %p with parent %p and base %p  index %p\n", ASE, parent, b, i); 

  ASE->imreadfrom = false; // don't know this yet
  //ASE->imreadfrom = imreadfrom; // ?? 
  //if (ASE->imreadfrom) { 
  //  debug_fprintf(stderr, "in chillAST_ArraySubscriptExpr::clone(), imreadfrom is being set. \n");
  //  ASE->print(); fflush(stdout); debug_fprintf(stderr, "\n"); 
  //} 

  //debug_fprintf(stderr, "cloned result "); ASE->print(); printf("\n\n\n"); fflush(stdout);
  //debug_fprintf(stderr, "ASE clone()  this 0x%x    clone 0x%x\n", this, ASE); 
  ASE->isFromSourceFile = isFromSourceFile;
  if (filename) ASE->filename = strdup(filename); 
  return ASE;
}

void chillAST_ArraySubscriptExpr::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {
  //debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::gatherArrayRefs setting imwrittento %d for ", writtento); 
//debug_fprintf(stderr, "%s ", base->getTypeString()); 
//base->print(); printf("\n"); fflush(stdout); 

  //debug_fprintf(stderr, "found an array subscript. &refs 0x%x   ", refs);
  if (!imwrittento) imwrittento = writtento;   // may be both written and not for += 
  fflush(stdout); 

  //debug_fprintf(stderr, "recursing on index ");  index->print(0,stderr); debug_fprintf(stderr, "\n");
  index->gatherArrayRefs( refs, 0 ); // recurse first
  //debug_fprintf(stderr, "adding this "); print(0,stderr); debug_fprintf(stderr, "\n"); 
  //debug_fprintf(stderr, "refs[%d] = 0x%x  = ", refs.size(), this); print(); fflush(stdout);   
  refs.push_back( this );

  //debug_fprintf(stderr, " size now %d\n", refs.size()); 

}

void chillAST_ArraySubscriptExpr::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  index->gatherScalarRefs( refs, 0 ); 
} 

void chillAST_ArraySubscriptExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  base->replaceVarDecls( olddecl, newdecl );
  index->replaceVarDecls( olddecl, newdecl );
}


void chillAST_ArraySubscriptExpr::replaceChild( chillAST_node *old, chillAST_node *newchild ) { 
  //debug_fprintf(stderr,"chillAST_ArraySubscriptExpr::replaceChild()\n"); 

  // arraysubscriptexpression doesn t really have children (should it?)
  // try index ???
  if (old == index) { 
    //debug_fprintf(stderr, "old is index\n");
    index = newchild;
    index->parent = this;
    return;
  }
  
  // try base ??? unclear if this makes sense  TODO 
  if (old == base) { 
    //debug_fprintf(stderr, "old is base\n");
    base = newchild;
    base->parent = this;
    return;
  }
  
  debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::replaceChild() old is not base or index\n"); 
  print(0,stderr); debug_fprintf(stderr, "\nchild: ");
  if (!old) debug_fprintf(stderr, "oldchild NULL!\n");
  old->print(0,stderr); debug_fprintf(stderr, "\nnew: "); 
  newchild->print(0,stderr); debug_fprintf(stderr, "\n"); 
  segfault(); // make easier for gdb
};


bool chillAST_ArraySubscriptExpr::operator!=( const chillAST_ArraySubscriptExpr &other) {
  bool opposite = *this == other;
  return !opposite;
}



bool chillAST_ArraySubscriptExpr::operator==( const chillAST_ArraySubscriptExpr &other) {
  //debug_fprintf(stderr, "chillAST_ArraySubscriptExpr::operator==\n");
  //debug_fprintf(stderr, "this->basedecl 0x%x     other.basedecl 0x%x\n", this->basedecl, other.basedecl);
  //this->basedecl->print(); printf("\n\n");
  //other.basedecl->print(); printf("\n"); fflush(stdout);

  //this->print(); printf(" 0x%x  == 0x%x ",this->uniquePtr, other.uniquePtr ); other.print(); printf(" ??  "); fflush(stdout); 
  //if ( this->uniquePtr == other.uniquePtr) debug_fprintf(stderr, "t\n"); 
  //else debug_fprintf(stderr, "f\n"); 
  return this->uniquePtr == other.uniquePtr; 
}





chillAST_MemberExpr::chillAST_MemberExpr() { 
  base = NULL;
  member = NULL;
  exptype = CHILL_MEMBER_EXP_DOT;
}

chillAST_MemberExpr::chillAST_MemberExpr( chillAST_node *bas, const char *mem, chillAST_node *p, void *unique, CHILL_MEMBER_EXP_TYPE t ) { 
  base = bas;
  if (base)   base->setParent( this ); 
  if (mem)    member = strdup( mem );
  parent = p;
  uniquePtr = unique;
  exptype = t;

  return;  // ignore tests below ?? TODO ?? 


  // base needs to RESOLVE to a decl ref expr but may not BE one
  //   A.b . c   lhs is a binop or memberexpr

  if (bas->isBinaryOperator()) { 
    //debug_fprintf(stderr, "checking binop to see if it resolved to a declrefexpr\n");
    // cheat for now or just remove the check below
    return; 
  }

  if (! ( bas->isDeclRefExpr() || bas->isArraySubscriptExpr() )) { 
    debug_fprintf(stderr, "chillAST_MemberExpr::chillAST_MemberExpr(), base is of type %s\n", bas->getTypeString());
    debug_fprintf(stderr, "chillAST_MemberExpr::chillAST_MemberExpr(), base is not DeclRefExpr\n");
    
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
  if (base) base->print( indent, fp );
  else {
    chillindent( indent, fp );
    fprintf(fp, "(NULL)"); 
  }
  if (exptype == CHILL_MEMBER_EXP_ARROW) fprintf(fp, "->");
  else fprintf(fp, "."); 
  if (member) fprintf(fp, "%s", member);
  else fprintf(fp, "(NULL)"); 
  fflush(fp); 
}


void chillAST_MemberExpr::printonly( int indent, FILE *fp ) {
  base->print( indent, fp );
  if (exptype == CHILL_MEMBER_EXP_ARROW) fprintf(fp, "->");
  else fprintf(fp, "."); 
  fprintf(fp, "%s", member);
  fflush(fp); 
}

char *chillAST_MemberExpr::stringRep( int indent ) { // char pointer to what we'd print
  debug_fprintf(stderr, "*chillAST_MemberExpr::stringRep()\n"); 
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
  debug_fprintf(stderr, "chillAST_MemberExpr::gatherArrayRefs()   "); print(0,stderr); debug_fprintf(stderr, "\n"); 
  debug_fprintf(stderr, "base of of type %s\n", base->getTypeString()); 
  base->gatherArrayRefs( refs, writtento ); // 
  
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
    //debug_fprintf(stderr, "old matches base of MemberExpr\n"); 
    base = newchild; 
  }
  else { 
    base->replaceChild( old, newchild ); 
  }
} 

chillAST_node  *chillAST_MemberExpr::multibase2() {  /*debug_fprintf(stderr, "ME MB2\n" );*/ return (chillAST_node *)this; } 

chillAST_VarDecl* chillAST_MemberExpr::getUnderlyingVarDecl() { 
  debug_fprintf(stderr, "chillAST_MemberExpr:getUnderlyingVarDecl()\n");
  print(); printf("\n"); fflush(stdout);
  exit(-1); 
  // find the member with the correct name
  
}




chillAST_VarDecl *chillAST_MemberExpr::multibase() {
  //c.i[c.count]    we want i member of c 
  //debug_fprintf(stderr, "ME MB\n" ); 

  //debug_fprintf(stderr, "chillAST_MemberExpr::multibase()\n");
  //print(); printf("\n"); fflush(stdout);
  //debug_fprintf(stderr, "MemberExpr base is type %s,  member %s\n", base->getTypeString(), member);
  
  //chillAST_VarDecl *vd = base->getUnderlyingVarDecl(); // this is the only thing that ever calls this ??? 
  chillAST_VarDecl *vd = base->multibase(); // ?? 


  //debug_fprintf(stderr, "vd "); vd->print(); printf("\n"); fflush(stdout);
    
  chillAST_RecordDecl *rd = vd->getStructDef();
  if (!rd) { 
    debug_fprintf(stderr, "chillAST_MemberExpr::multibase() vardecl is not a struct??\n");
    debug_fprintf(stderr, "vd "); vd->print(); printf("\n"); fflush(stdout);
    debug_fprintf(stderr, "vd "); vd->dump();  printf("\n"); fflush(stdout);
    exit(-1);
  }

  // OK, we have the recorddecl that defines the structure
  // now find the member with the correct name
  chillAST_VarDecl *sub = rd->findSubpart( member );
  //debug_fprintf(stderr, "sub %s:\n", member);
  if (!sub) { 
    debug_fprintf(stderr, "can't find member %s in \n", member);
    rd->print(); 
  }
  //sub->print(); printf("\n");  fflush(stdout);
  //sub->dump() ; printf("\n");  fflush(stdout);

  return sub; 
  //find vardecl of member in def of base

  
}




chillAST_DeclRefExpr::chillAST_DeclRefExpr() { 
  declarationType = strdup("UNKNOWN");
  declarationName = strdup("NONE");
  decl = NULL; 
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( const char *varname, chillAST_node *par ) { 
  declarationType = strdup("UNKNOWN");
  declarationName = strdup(varname); 
  decl = NULL; 
  parent = par; 
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( const char *vartype, const char *varname, chillAST_node *par) {
  //debug_fprintf(stderr, "DRE::DRE 0x%x   %s %s\n", this, vartype, varname ); 
  declarationType = strdup(vartype);
  declarationName = strdup(varname); 
  decl = NULL; 
  parent = par; 
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( const char *vartype, const char *varname, chillAST_node *d, chillAST_node *par ) {
  //debug_fprintf(stderr, "DRE::DRE2 0x%x   %s %s  0x%x\n", this, vartype, varname, d ); 
  declarationType = strdup(vartype);
  declarationName = strdup(varname); 
  decl = d; 
  parent = par; 
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( chillAST_VarDecl *vd, chillAST_node *par ){ // variable def
  //debug_fprintf(stderr, "DRE::DRE3 (VD)  0x%x   %s %s  0x%x\n", this, vd->vartype, vd->varname, vd ); 
  
  declarationType = strdup(vd->vartype);
  declarationName = strdup(vd->varname); 
  decl = vd; 
  parent = par;
}


chillAST_DeclRefExpr::chillAST_DeclRefExpr( chillAST_FunctionDecl *fd, chillAST_node *par ){ // function def 
  declarationType = strdup(fd->returnType);
  declarationName = strdup(fd->functionName); 
  decl = fd; 
  parent = par;
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


char *chillAST_DeclRefExpr::stringRep( int indent ) { 
  return strdup( declarationName ); 
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
  //debug_fprintf(stderr, "chillAST_DeclRefExpr::clone()\n"); 
  chillAST_DeclRefExpr *DRE =  new chillAST_DeclRefExpr( declarationType, declarationName, decl, parent ); 
  DRE->isFromSourceFile = isFromSourceFile;
  if (filename) DRE->filename = strdup(filename); 
  return DRE;
}


void chillAST_DeclRefExpr::gatherVarDeclsMore( vector<chillAST_VarDecl*> &decls ) {
  //debug_fprintf(stderr, "chillAST_DeclRefExpr::gatherVarDeclsMore()\n"); 
  decl->gatherVarDeclsMore( decls ); 
}


void chillAST_DeclRefExpr::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //debug_fprintf(stderr, "chillAST_DeclRefExpr::gatherScalarVarDecls()\n"); 
  decl->gatherScalarVarDecls(decls); 
  //debug_fprintf(stderr, "now %d scalar vardecls\n", decls.size()); 
}


void chillAST_DeclRefExpr::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //debug_fprintf(stderr, "chillAST_DeclRefExpr::gatherArrayVarDecls()\n"); 
  decl->gatherArrayVarDecls(decls); 
  //debug_fprintf(stderr, "now %d Array vardecls\n", decls.size()); 
}


void chillAST_DeclRefExpr::gatherDeclRefExprs( vector<chillAST_DeclRefExpr *>&refs ) {
  refs.push_back(this); 
}

void chillAST_DeclRefExpr::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  refs.push_back(this); 
} 

void chillAST_DeclRefExpr::gatherVarUsage( vector<chillAST_VarDecl*> &decls ) {
  //debug_fprintf(stderr, "chillAST_DeclRefExpr::gatherVarUsage()\n"); 
  for (int i=0; i<decls.size(); i++) { 
    if (decls[i] == decl) { 
      //debug_fprintf(stderr, "decl was already there\n");
      return;
    }
    if (streq(declarationName, decls[i]->varname)) { 
      if (streq(declarationType, decls[i]->vartype)) { 
        //debug_fprintf(stderr, "decl was already there\n");
        return;
      }
    }
  }
  chillAST_VarDecl *vd = getVarDecl();  // null for functiondecl
  if (vd) decls.push_back( vd ); 

}




void chillAST_DeclRefExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  //debug_fprintf(stderr, "chillAST_DeclRefExpr::replaceVarDecls()\n"); 
  if (decl == olddecl) { 
    //debug_fprintf(stderr, "replacing old %s with %s\n", olddecl->varname, newdecl->varname);
    //debug_fprintf(stderr, "DRE was "); print(); 
    decl = newdecl; 
    declarationType = strdup(newdecl->vartype); 
    declarationName = strdup(newdecl->varname); 
    //debug_fprintf(stderr, "\nDRE  is "); print(); debug_fprintf(stderr, "\n\n"); 
  }
  else { 
    if (!strcmp(olddecl->varname, declarationName)) { 
      //debug_fprintf(stderr, "uhoh, chillAST_DeclRefExpr::replaceVarDecls()\n"); 
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
  //debug_fprintf(stderr, "chillAST_VarDecl::gatherVarDecls()\n"); 
  for (int i=0; i<decls.size(); i++) { 
    if (decls[i] == this) { 
      //debug_fprintf(stderr, "decl was already there\n");
      return;
    }
    if (streq(decls[i]->varname, varname)) { 
      if (streq(decls[i]->vartype, vartype)) { 
        //debug_fprintf(stderr, "VarDecl (direct) decl was already there\n");
        return;
      }
    }
  }
  decls.push_back( this ); 
}


void chillAST_VarDecl::gatherScalarVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //debug_fprintf(stderr, "chillAST_VarDecl::gatherScalarVarDecls(), %s numdimensions %d\n", varname, numdimensions); 

  if (numdimensions != 0) return; // not a scalar
  
  for (int i=0; i<decls.size(); i++) { 
    if (decls[i] == this) { 
      //debug_fprintf(stderr, "decl was already there\n");
      return;
    }

    if (streq(decls[i]->varname, varname)) {      // wrong. scoping.  TODO
      if (streq(decls[i]->vartype, vartype)) { 
        //debug_fprintf(stderr, "VarDecl (direct) decl was already there\n");
        return;
      }
    }
  }
  //debug_fprintf(stderr, "adding vardecl for %s to decls\n", varname); 
  decls.push_back( this ); 
}


void chillAST_VarDecl::gatherArrayVarDecls( vector<chillAST_VarDecl*> &decls ) {
  //debug_fprintf(stderr, "chillAST_VarDecl::gatherScalarVarDecls(), %s numdimensions %d\n", varname, numdimensions); 

  if (numdimensions == 0) return; // not an array
  
  for (int i=0; i<decls.size(); i++) { 
    if (decls[i] == this) { 
      //debug_fprintf(stderr, "decl was already there\n");
      return;
    }

    if (streq(decls[i]->varname, varname)) {      // wrong. scoping.  TODO
      if (streq(decls[i]->vartype, vartype)) { 
        //debug_fprintf(stderr, "VarDecl (direct) decl was already there\n");
        return;
      }
    }
  }
  //debug_fprintf(stderr, "adding vardecl for %s to decls\n", varname); 
  decls.push_back( this ); 
}



chillAST_node *chillAST_VarDecl::constantFold() {  return this; }

chillAST_node* chillAST_VarDecl::clone() {
  //debug_fprintf(stderr, "\nchillAST_VarDecl::clone()  cloning vardecl for %s\n", varname); 
  //if (isAParameter) debug_fprintf(stderr, "old vardecl IS a parameter\n");
  //else  debug_fprintf(stderr, "old vardecl IS NOT a parameter\n");

  chillAST_VarDecl *vd  = new chillAST_VarDecl( vartype, strdup(varname), arraypart, NULL);  // NULL so we don't add the variable AGAIN to the (presumably) function 
  
  vd->typedefinition = typedefinition;
  vd->vardef = vardef; // perhaps should not do this     TODO 

  vd->underlyingtype = strdup(underlyingtype); 

  vd->arraysizes = NULL;
  vd->knownArraySizes = knownArraySizes; 
  vd->numdimensions = numdimensions;
  vd->arraypointerpart = NULL;

  if (arraypart != NULL && NULL!=arraysizes) {  // !strcmp(arraypart, "")) { 
    //debug_fprintf(stderr, "in chillAST_VarDecl::clone(), cloning the array info\n");
    //debug_fprintf(stderr, "numdimensions %d     arraysizes 0x%x\n", numdimensions, arraysizes) ;
    vd->numdimensions = numdimensions;

    if (arraysizes) { 
      vd->arraysizes = (int *)malloc( sizeof(int *) * numdimensions ); 
      for (int i=0; i< numdimensions; i++) { 
        //debug_fprintf(stderr, "i %d\n", i); 
        vd->arraysizes[i] = arraysizes[i]; 
      }
    }
  }

  if ( arraypointerpart ) { 
    //debug_fprintf(stderr, "copying arraypointerpart\n"); 
    vd->arraypointerpart = strdup( arraypointerpart);
  }

  vd->isStruct = this->isStruct; 
  //vd->insideAStruct =  this->insideAStruct; 

  //if (vd->isStruct)  debug_fprintf(stderr, "vardecl::clone()  %s is a struct\n", varname); 
  //else debug_fprintf(stderr, "vardecl::clone()  %s is NOT a struct\n", varname); 


  vd->knownArraySizes = this->knownArraySizes; 
  vd->isFromSourceFile = isFromSourceFile;
  if (filename) vd->filename = strdup(filename); 
  return vd;
}


void chillAST_VarDecl::splitarraypart() { 
  //debug_fprintf(stderr, "chillAST_VarDecl::splitarraypart()  ");
  //debug_fprintf(stderr, "%p  ", arraypart);
  //if (arraypart) debug_fprintf(stderr, "%s", arraypart); 
  //debug_fprintf(stderr, "\n");

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
    if (this->arraypart[i] == '*') { 
      if (!fixedcount) {
        asteriskcount++;
      }
    }
    else { // remainder is fixed? 
      fixedcount++; 
      // check for brackets and digits only?   TODO
    }
  }
  this->arraypointerpart = (char *) calloc( asteriskcount+1, sizeof(char));
  this->arraysetpart     = (char *) calloc( fixedcount+1,    sizeof(char));
  
  char *ptr = arraypart;
  for ( int i=0; i<asteriskcount; i++)  arraypointerpart[i] = *ptr++;
  for ( int i=0; i<fixedcount; i++)     arraysetpart[i]   = *ptr++;
}






chillAST_IntegerLiteral::chillAST_IntegerLiteral(int val, chillAST_node *par){
  value = val; 
  parent = par;
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
  parent = par;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(double val, chillAST_node *par){
  doublevalue = val; 
  precision = 2;
  float0double1 = 1; // which is live! 
  allthedigits = NULL; 
  parent = par;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(float val, int precis, chillAST_node *par){
  value = val; 
  precision = 1;
  float0double1 = 0; // which is live! 
  precision = precis; // 
  allthedigits = NULL; 
  parent = par;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(double val, int precis, chillAST_node *par){
  doublevalue = val; 
  float0double1 = 1; // which is live! 
  precision = precis; // 
  allthedigits = NULL; 
  parent = par;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(float val, const char *printthis, chillAST_node *par){
  value = val; 
  float0double1 = 0; // which is live! 
  precision = 1;
  allthedigits = NULL;
  if (printthis) allthedigits = strdup( printthis ); 
  //debug_fprintf(stderr, "\nfloatingliteral allthedigits = '%s'\n", allthedigits); 
  parent = par;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(float val, int precis, const char *printthis, chillAST_node *par){
  value = val; 
  float0double1 = 0; // which is live! 
  precision = precis; // but value is a float??  TODO 
  allthedigits = NULL;
  if (printthis) { 
    //debug_fprintf(stderr, "\nchillAST_FloatingLiteral constructor, printthis "); 
    //debug_fprintf(stderr, "%p\n", printthis); 
    allthedigits = strdup( printthis ); 
  }
  //debug_fprintf(stderr, "\nfloatingliteral allthedigits = '%s'\n", allthedigits); 
  parent = par;
}


chillAST_FloatingLiteral::chillAST_FloatingLiteral( chillAST_FloatingLiteral *old ) {
  //debug_fprintf(stderr, "chillAST_FloatingLiteral::chillAST_FloatingLiteral( old ) allthedigits %p\n", old->allthedigits); 

  value          = old->value;
  doublevalue    = old->doublevalue; 
  float0double1  = old->float0double1;
  allthedigits = NULL;
  if (old->allthedigits) allthedigits = strdup(old->allthedigits); 
  precision      = old->precision;
}



void chillAST_FloatingLiteral::print( int indent, FILE *fp) {
  chillindent(indent, fp);
  //fprintf(fp, "%f", value);
  // attempt to be more like rose output
  char output[1024]; // warning, hardcoded 

  if (allthedigits != NULL) {
    strcpy(output, allthedigits ); // if they have specified 100 digits of pi, give 'em 100 digits 
    //debug_fprintf(stderr, "floatingliteral allthedigits = '%s'\n", allthedigits); 
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
  //debug_fprintf(stderr, "chillAST_FloatingLiteral::clone()  "); 
  //debug_fprintf(stderr, "allthedigits %p \n", allthedigits); 
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





chillAST_UnaryOperator::chillAST_UnaryOperator( const char *oper, bool pre, chillAST_node *sub, chillAST_node *par ):subexpr(this,0) {
  op = strdup(oper);
  prefix = pre;
  subexpr = sub; 
  subexpr->setParent( this );
  parent = par;
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
  //debug_fprintf(stderr, "chillAST_UnaryOperator::constantFold() ");
  //print(); debug_fprintf(stderr, "\n"); 

  subexpr = subexpr->constantFold();
  chillAST_node *returnval = this;
  if (subexpr->isConstant()) {
    //debug_fprintf(stderr, "unary op folding constants\n");
    //print(0,stderr); debug_fprintf(stderr, "\n");
    
    if (streq(op, "-")) { 
      if (subexpr->isIntegerLiteral()) {
        int intval = ((chillAST_IntegerLiteral*)subexpr())->value;
        chillAST_IntegerLiteral *I = new chillAST_IntegerLiteral( -intval, parent);
        returnval = I;
        //debug_fprintf(stderr, "integer -%d becomes %d\n", intval, I->value);
      }
      else { 
        chillAST_FloatingLiteral *FL = (chillAST_FloatingLiteral*)subexpr();
        chillAST_FloatingLiteral *F = new chillAST_FloatingLiteral( FL ); // clone
        F->parent = FL->parent;

        F->value = -F->value;
        F->doublevalue = -F->doublevalue;
        
        F->print(); debug_fprintf(stderr, "\n"); 
        
        returnval = F;
      }
    }
    else debug_fprintf(stderr, "can't fold op '%s' yet\n", op); 
  }    
  return returnval;
}


class chillAST_node* chillAST_UnaryOperator::clone() { 
  chillAST_UnaryOperator *UO = new chillAST_UnaryOperator( op, prefix, subexpr->clone(), parent );
  UO->isFromSourceFile = isFromSourceFile; 
  if (filename) UO->filename = strdup(filename); 
  return UO; 
}

 void chillAST_UnaryOperator::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
   subexpr->replaceVarDecls( olddecl, newdecl ); 
 }


int chillAST_UnaryOperator::evalAsInt() { 
  if (!strcmp("+", op)) return subexpr->evalAsInt();
  if (!strcmp("-", op)) return -subexpr->evalAsInt();
  if (!strcmp("++", op)) return 1 + subexpr->evalAsInt();
  if (!strcmp("--", op)) return subexpr->evalAsInt() - 1;

  debug_fprintf(stderr, "chillAST_UnaryOperator::evalAsInt() unhandled op '%s'\n", op); 
  segfault(); 

}

bool chillAST_UnaryOperator::isSameAs( chillAST_node *other ){
  if (!other->isUnaryOperator()) return false;
  chillAST_UnaryOperator *o = (chillAST_UnaryOperator *)other;
  if (strcmp(op, o->op))  return false; // different operators 
  return subexpr->isSameAs( o->subexpr ); // recurse
}


chillAST_ImplicitCastExpr::chillAST_ImplicitCastExpr( chillAST_node *sub, chillAST_node *par ):subexpr(this,0) {
  subexpr = sub;
  subexpr->setParent( this );
  parent = par;
  //debug_fprintf(stderr, "ImplicitCastExpr 0x%x  has subexpr 0x%x", this, subexpr);
  //debug_fprintf(stderr, " of type %s\n", subexpr->getTypeString()); 
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

  debug_fprintf(stderr, "chillAST_ImplicitCastExpr::replaceChild() called with bad 'old'\n");
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

chillAST_CStyleCastExpr::chillAST_CStyleCastExpr( const char *to, chillAST_node *sub, chillAST_node *par ):subexpr(this,0) {

  //debug_fprintf(stderr, "chillAST_CStyleCastExpr::chillAST_CStyleCastExpr( %s, ...)\n", to); 
  towhat = strdup(to);
  subexpr = sub;
  if (subexpr) subexpr->setParent( this );
  parent = par;
  //debug_fprintf(stderr, "chillAST_CStyleCastExpr (%s)   sub 0x%x\n", towhat, sub ); 
}

void chillAST_CStyleCastExpr::replaceChild( chillAST_node *old, chillAST_node *newchild ){
  if (subexpr == old) { // should be the case for this to get called
    subexpr = newchild;
    subexpr->setParent( this );
    //old->parent = NULL;
    return;
  }

  debug_fprintf(stderr, "chillAST_CStyleCastExpr::replaceChild() called with bad 'old'\n");
  exit(-1);  // ?? 
}

 void chillAST_CStyleCastExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
   subexpr->replaceVarDecls( olddecl, newdecl);
 }

void chillAST_CStyleCastExpr::print(  int indent, FILE *fp) {
  //debug_fprintf(stderr, "CStyleCastExpr::print()\n"); 
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
    
    if (subexpr->isVarDecl()) fprintf(fp, "%s", ((chillAST_VarDecl *)subexpr())->varname);
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

chillAST_CStyleAddressOf::chillAST_CStyleAddressOf( chillAST_node *sub, chillAST_node *par ):subexpr(this,0) {
  subexpr = sub;
  subexpr->setParent( this );
  parent = par;
  //debug_fprintf(stderr, "chillAST_CStyleCastExpr (%s)   sub 0x%x\n", towhat, sub ); 
}

void chillAST_CStyleAddressOf::print(  int indent, FILE *fp) {
  //debug_fprintf(stderr, "CStyleAddressOf::print()\n"); 
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

chillAST_Malloc::chillAST_Malloc(chillAST_node *size, chillAST_node *p):sizeexpr(this,0) {
  thing = NULL;
  sizeexpr = size;  // probably a multiply like   sizeof(int) * 1024
  parent = p;
};

chillAST_Malloc::chillAST_Malloc(char *thething, chillAST_node *numthings, chillAST_node *p):sizeexpr(this,0) {
  thing = strdup(thething);   // "int" or "float" or "struct widget"
  sizeexpr = numthings;  
  parent = p;
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



chillAST_CudaMalloc::chillAST_CudaMalloc(chillAST_node *devmemptr, chillAST_node *size, chillAST_node *p):devPtr(this,0), sizeinbytes(this,1) {
  devPtr = devmemptr; 
  sizeinbytes = size;  // probably a multiply like   sizeof(int) * 1024
  parent = p;
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

chillAST_CudaFree::chillAST_CudaFree(chillAST_VarDecl *var, chillAST_node *p):variable(this,0) {
  variable = var; 
  parent = p;
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

chillAST_CudaMemcpy::chillAST_CudaMemcpy(chillAST_VarDecl *d, chillAST_VarDecl *s, chillAST_node *siz, char *kind, chillAST_node *par):dest(this,0),src(this,1),size(this,2) {
  dest = d;
  src = s;
  //debug_fprintf(stderr, "chillAST_CudaMemcpy::chillAST_CudaMemcpy( dest %s, src %s, ...)\n", d->varname, s->varname ); 
  size = siz;
  cudaMemcpyKind = kind;
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

chillAST_CudaSyncthreads::chillAST_CudaSyncthreads( chillAST_node *par) { 
  parent = par;
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
 









chillAST_ReturnStmt::chillAST_ReturnStmt( chillAST_node *retval, chillAST_node *par ):returnvalue(this,0) {
  returnvalue = retval;
  if (returnvalue) returnvalue->setParent( this );
  parent = par; 
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

chillAST_CallExpr::chillAST_CallExpr(chillAST_node *c, chillAST_node *par) { //, int numofargs, chillAST_node **theargs ) {
  
  //debug_fprintf(stderr, "chillAST_CallExpr::chillAST_CallExpr  callee type %s\n", c->getTypeString()); 
  callee = c;
  //callee->setParent( this ); // ?? 
  numargs = 0;
  parent = par; 
  grid = block = NULL;
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
    //debug_fprintf(stderr, "DRE decl is 0x%x\n", DRE->decl); 
    if (!DRE->decl) { 
      // a macro? 
      fprintf(fp, "%s ", DRE->declarationName); 
      return; // ?? 
    }

    //debug_fprintf(stderr, "DRE decl of type %s\n", DRE->decl->getTypeString()); 
    if ( (DRE->decl)->isFunctionDecl()) FD = (chillAST_FunctionDecl *)DRE->decl; 
    else { 
      debug_fprintf(stderr, "chillAST_CallExpr::print() DRE decl of type %s\n", DRE->decl->getTypeString()); 
      exit(-1);
    }
  }
  else if (callee->isFunctionDecl()) FD = (chillAST_FunctionDecl *) callee;
  else if (callee->isMacroDefinition()) { 
    MD = (chillAST_MacroDefinition *) callee;
    fprintf(fp, "%s(", MD->macroName); 
  }
  else { 
    debug_fprintf(stderr, "\nchillAST_CallExpr::print() callee of unhandled type %s\n", callee->getTypeString()); 
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
  //debug_fprintf(stderr, "callee type %s\n", callee->getTypeString()); 
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
  //debug_fprintf(stderr, "chillAST_CallExpr::clone()\n");
  //print(0, stderr); debug_fprintf(stderr, "\n"); 

  chillAST_CallExpr *CE = new chillAST_CallExpr( callee->clone(), NULL );
  for (int i=0; i<args.size(); i++) CE->addArg( args[i]->clone() ); 
  CE->isFromSourceFile = isFromSourceFile; 
  if (filename) CE->filename = strdup(filename); 
  return CE; 
}




chillAST_VarDecl::chillAST_VarDecl() { 
  //debug_fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl()  %p\n", this); 
  vartype = underlyingtype = varname = arraypart = arraypointerpart = arraysetpart = NULL;
  typedefinition = NULL; 

  //debug_fprintf(stderr, "setting underlying type NULL\n" ); 
  init = NULL;
  numdimensions=0; arraysizes = NULL; 

  vardef  = NULL;
  isStruct = false; 
  //insideAStruct = false; 
  isAParameter = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // debug_fprintf(stderr, "RDS = false\n"); 
  knownArraySizes = false;
};



chillAST_VarDecl::chillAST_VarDecl( const char *t,  const char *n, const char *a, chillAST_node *par) { 
  //debug_fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl( type %s, name %s, arraypart %s,  parent %p)  %p\n", t, n, a, par, this); 
  vartype   = strdup(t); 
  typedefinition = NULL;

  underlyingtype = parseUnderlyingType( vartype ); 
  //debug_fprintf(stderr, "setting underlying type %s from %s\n",  underlyingtype, vartype ); 
  varname   = strdup(n); 
  arraypointerpart = arraysetpart = NULL;
  if (a) arraypart = strdup(a);
  else arraypart = strdup(""); 
  splitarraypart();

  init = NULL;
  numdimensions=0; arraysizes = NULL; 
  uniquePtr = NULL;
  parent = par;



  knownArraySizes = false; 
  //debug_fprintf(stderr, "arraypart len %d\n", strlen(a)); 
  for (int i=0; i<strlen(a); i++) { 
    if (a[i] == '[') { numdimensions++; knownArraySizes = true; } 
    if (!knownArraySizes && a[i] == '*') numdimensions++;
  }
  
  vardef  = NULL;
  isStruct = false; 
  //insideAStruct = false; 
  isAParameter = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // debug_fprintf(stderr, "RDS = false\n"); 

  if (parent) { 
    //debug_fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl( %s ), adding to symbol table???\n", varname); 
    parent->addVariableToSymbolTable( this ); // should percolate up until something has a symbol table 
    
  }
};



chillAST_VarDecl::chillAST_VarDecl( chillAST_RecordDecl *astruct, const char *nam, const char *array, chillAST_node *par) { 
  //debug_fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl( %s  %p struct ", nam, this );
  const char *type = astruct->getName(); 
  //debug_fprintf(stderr, "%s, name %s, arraypart %s parent ) %p\n", type, nam, array, this); // , par);

  vartype = strdup(type);

  // these always go together  ?? 
  vardef  = astruct;// pointer to the thing that says what is inside the struct
  isStruct = true;  // ?? wrong if it's a union  ?? TODO 
  //insideAStruct = false; 
  //debug_fprintf(stderr, "setting vardef of %s to %p\n", nam, vardef); 
  
  underlyingtype = parseUnderlyingType( vartype ); 
  //debug_fprintf(stderr, "setting underlying type %s from %s\n",  underlyingtype, vartype ); 
  varname   = strdup(nam); 
  arraypart = strdup(array);
  arraypointerpart = arraysetpart = NULL;
  splitarraypart(); 

  init = NULL;
  numdimensions=0; arraysizes = NULL; 
  uniquePtr = NULL;
  parent = par;

  knownArraySizes = false; 
  //debug_fprintf(stderr, "arraypart len %d\n", strlen(a)); 
  for (int i=0; i<strlen(array); i++) { 
    if (array[i] == '[') { numdimensions++; knownArraySizes = true; } 
    if (!knownArraySizes && array[i] == '*') numdimensions++;
  }
  
  isAParameter = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // debug_fprintf(stderr, "RDS = false\n"); 
  typedefinition = NULL;

  //debug_fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl( chillAST_RecordDecl *astruct, ...) MIGHT add struct to some symbol table\n"); 
  //if (parent) debug_fprintf(stderr, "yep, adding it\n"); 

  if (parent) parent->addVariableToSymbolTable( this ); // should percolate up until something has a symbol table 
};





chillAST_VarDecl::chillAST_VarDecl( chillAST_TypedefDecl *tdd,  const char *n, const char *a, chillAST_node *par) { 
  //debug_fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl( %s  typedef ", n);
  const char *type = tdd->getStructName();
  //fprintf (stderr, "%s, name %s, arraypart %s parent ) %p\n", type, n, a,this); // , par);
  typedefinition = tdd;
  vartype   = strdup(type); 
  underlyingtype = parseUnderlyingType( vartype ); 
  //debug_fprintf(stderr, "setting underlying type %s from %s\n",  underlyingtype, vartype ); 
  varname   = strdup(n); 
  arraypart = strdup(a);
  arraypointerpart = arraysetpart = NULL;
  splitarraypart(); 

  init = NULL;
  numdimensions=0; arraysizes = NULL; 
  uniquePtr = NULL;
  parent = par;

  knownArraySizes = false; 
  //debug_fprintf(stderr, "arraypart len %d\n", strlen(a)); 
  for (int i=0; i<strlen(a); i++) { 
    if (a[i] == '[') { numdimensions++; knownArraySizes = true; } 
    if (!knownArraySizes && a[i] == '*') numdimensions++;
  }

  isStruct = tdd->isAStruct();
  //insideAStruct = false; 
  
  vardef  = NULL;
  isAParameter = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // //debug_fprintf(stderr, "RDS = false\n"); 
  if (parent) parent->addVariableToSymbolTable( this ); // should percolate up until something has a symbol table 
};





chillAST_VarDecl::chillAST_VarDecl( const char *t,  const char *n, const char *a, void *ptr, chillAST_node *par) { 
  debug_fprintf(stderr, "2chillAST_VarDecl::chillAST_VarDecl( type %s, name %s, arraypart '%s' ) %p\n", t, n, a, this); 
  //debug_fprintf(stderr, "2chillAST_VarDecl::chillAST_VarDecl( type %s, name %s, arraypart %s, ptr 0x%x, parent 0x%x )\n", t, n, a, ptr, par); 


  vartype   = strdup(t); 
  typedefinition = NULL;
  underlyingtype = parseUnderlyingType( vartype ); 
  //debug_fprintf(stderr, "setting underlying type %s from %s\n",  underlyingtype, vartype ); 
  varname   = strdup(n); 
  vardef = NULL;  // not a struct
  isStruct = false;
  isAParameter = false; 

  if (a) arraypart = strdup(a);
  else arraypart = strdup(""); // should catch this earlier
  arraypointerpart = arraysetpart = NULL;
  splitarraypart(); 

  init = NULL;
  numdimensions=0; arraysizes = NULL; 
  uniquePtr = ptr;
  parent = par;
  knownArraySizes = false; 

  if (par) par->addChild(this); // ??
  
  //debug_fprintf(stderr, "name arraypart len %d\n", strlen(a)); 
  //debug_fprintf(stderr, "arraypart '%s'\n", arraypart); 
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
  //debug_fprintf(stderr, "2name %s numdimensions %d\n", n, numdimensions); 




  // this is from ir_clang.cc ConvertVarDecl(), that got executed AFTER the vardecl was constructed. dumb
  int numdim = 0;
  //knownArraySizes = true;
  //if (index(vartype, '*')) knownArraySizes = false;  // float *a;   for example
  //if (index(arraypart, '*'))  knownArraySizes = false;
  
  // note: vartype here, arraypart in next code..    is that right?
  if (index(vartype, '*')) { 
    for (int i = 0; i<strlen(vartype); i++) if (vartype[i] == '*') numdim++;
    //debug_fprintf(stderr, "numd %d\n", numd);
    numdimensions = numdim; 
  }
  
  if (index(arraypart, '[')) {  // JUST [12][34][56]  no asterisks
    char *dupe = strdup(arraypart);

    int len = strlen(arraypart);
    for (int i=0; i<len; i++) if (dupe[i] == '[') numdim++;
    
    //debug_fprintf(stderr, "numdim %d\n", numdim);
    
    numdimensions = numdim; 
    int *as =  (int *)malloc(sizeof(int *) * numdim );
    if (!as) { 
      debug_fprintf(stderr, "can't malloc array sizes in ConvertVarDecl()\n");
      exit(-1);
    }
    arraysizes = as; // 'as' changed later!
    
    
    char *ptr = dupe;
    //debug_fprintf(stderr, "dupe '%s'\n", ptr);
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
          //debug_fprintf(stderr, " not justmath because '%c'\n", c); 
          justmath = false; 
        }
            
      }

      //debug_fprintf(stderr, "tmp '%s'\n", leak);
      if (justdigits) { 
        int dim;
        sscanf(ptr, "%d", &dim);
        //debug_fprintf(stderr, "dim %d\n", dim);
        *as++ = dim; 
      }
      else { 
        if (justmath) debug_fprintf(stderr, "JUST MATH\n");
        debug_fprintf(stderr, "need to evaluate %s, faking with hardcoded 16000\n", leak); 
        *as++ = 16000; // temp TODO DFL 
      }
      free (leak); 

      ptr =  index(ptr, ']');
      //debug_fprintf(stderr, "bottom of loop, ptr = '%s'\n", ptr); 
    }
    free(dupe);
    //for (int i=0; i<numdim; i++) { 
    //  debug_fprintf(stderr, "dimension %d = %d\n", i,  arraysizes[i]); 
    //} 
    
    //debug_fprintf(stderr, "need to handle [] array to determine num dimensions\n");
    //exit(-1); 
  }
  
  
  //insideAStruct = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // debug_fprintf(stderr, "RDS = false\n"); 
  
  //print(); printf("\n"); fflush(stdout); 

  // currently this is bad, because a struct does not have a symbol table, so the 
  // members of a struct are passed up to the func or sourcefile. 
  if (parent) parent->addVariableToSymbolTable( this ); // should percolate up until something has a symbol table 


  //debug_fprintf(stderr, "2chillAST_VarDecl::chillAST_VarDecl LEAVING\n"); 
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


  //debug_fprintf(stderr, "chillAST_VarDecl::print()  %s\n", varname ); 
  //if (isParmVarDecl()) debug_fprintf(stderr, "%s is a parameter\n", varname); 
  //if (isAStruct()) debug_fprintf(stderr, "%s is a struct\n", varname); 
  //else debug_fprintf(stderr, "%s is NOT a struct\n", varname); 
  //if (!parent) debug_fprintf(stderr, "VARDECL HAS NO PARENT\n");
  //else debug_fprintf(stderr, "parent of %s is type %s\n", varname, parent->getTypeString()); 

  // this logic is probably wrong (what about pointer to struct? )
  if ((!isAParameter) && isAStruct() && vardef) { // an unnamed  struct used only here ?? 
    //debug_fprintf(stderr, "isAStruct() && vardef ?? \n");
    // print the internals of the struct and then the name 
    vardef->printStructure( 0, fp );
    fprintf(fp, "%s", varname ); 
    return;
  }
  
  // ugly logic TODO 
  if (typedefinition && typedefinition->isAStruct()) fprintf(fp, "struct "); 

  if (isAParameter) { 
    //if (isAStruct())  fprintf(fp, "struct "); 
    //fprintf(fp, "(param) nd %d", numdimensions ); 
    //dump(); 
    if (numdimensions > 0) {
      if (knownArraySizes) {  // just [12][34][56] 
        fprintf(fp, "%s ", vartype);
        if (byreference) fprintf(fp, "&");
        fprintf(fp, "%s", varname);
        for (int n=0; n< (numdimensions); n++) fprintf(fp, "[%d]", arraysizes[n]); 
      }
      else {  // some unknown array part    float *a;  or float **a;  or float (*)a[1234] 

        //fprintf(fp, "\nsome unknown\n"); 
        if (numdimensions == 1) { 
          //fprintf(fp, "\nnd1, vartype %s\n", vartype); 
          
          // TODO this if means I have probably made a mistake somewhere
          if (!index(vartype, '*')) fprintf(fp, "%s *%s",   vartype, varname ); // float *x
          else fprintf(fp, "%s%s", vartype, varname); // float *a; 
        }
        else { // more than one dimension 

          if ( !strcmp("", arraysetpart) ) { // no known dimensions   float ***a;
            fprintf(fp, "%s %s%s", vartype, arraypointerpart, varname); 
          }
          else if ( !strcmp("", arraypointerpart)) { // ALL known  float a[2][7];
            fprintf(fp, "%s %s", vartype, varname); 
            for (int n=0; n< numdimensions; n++)  fprintf(fp, "[%d]", arraysizes[n]);
          }
          else { //   float (*)a[1234] 
            // this seems really wrong 
            //     float (*)a[1234] 
          fprintf(fp, "%s (", vartype); 
          for (int n=0; n< (numdimensions-1); n++) fprintf(fp, "*");
          fprintf(fp, "%s)", varname);
          fprintf(fp, "[%d]", arraysizes[numdimensions-1]); 
        }
          
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
    //if (isArray()) debug_fprintf(stderr, "an array, numdimensions %d\n", numdimensions);
    //debug_fprintf(stderr, "arraysizes %p\n", arraysizes);


  
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
  //debug_fprintf(stderr, "chillAST_CompoundStmt::chillAST_CompoundStmt() %p\n", this); 
  symbol_table = new chillAST_SymbolTable;
  typedef_table = NULL;
};


void  chillAST_CompoundStmt::print( int indent,  FILE *fp ) { 
  printPreprocBEFORE(indent, fp); 
  int numchildren = children.size();
  //debug_fprintf(stderr, "NUMCHILDREN %d\n", numchildren); sleep(1); 
  for (int i=0; i<numchildren; i++) {
    children[i]->print(indent, fp);
    if (children[i]->getType() != CHILLAST_NODETYPE_FORSTMT
        && children[i]->getType() != CHILLAST_NODETYPE_IFSTMT
        && children[i]->getType() != CHILLAST_NODETYPE_COMPOUNDSTMT
        //&& children[i]->asttype != CHILLAST_NODETYPE_VARDECL   // vardecl does its own ";\n"
        ) 
      {
        fprintf(fp, ";\n");  // probably wrong 
      }
  }
  fflush(fp); 
}

void chillAST_CompoundStmt::replaceChild( chillAST_node *old, chillAST_node *newchild ){
  //debug_fprintf(stderr, "chillAST_CompoundStmt::replaceChild( old %s, new %s)\n", old->getTypeString(), newchild->getTypeString() ); 
   vector<chillAST_node*> dupe = children; 
   int numdupe = dupe.size();
  int any = 0; 
  
  for (int i=0; i<numdupe; i++) { 

    //debug_fprintf(stderr, "\ni %d\n",i); 
    //for (int j=0; j<numdupe; j++) { 
    //  debug_fprintf(stderr, "this 0x%x   children[%d/%d] = 0x%x type %s\n", this, j, children.size(), children[j], children[j]->getTypeString()); 
    //}


    if (dupe[i] == old) { 
      //debug_fprintf(stderr, "replacing child %d of %d\n", i, numdupe); 
      //debug_fprintf(stderr, "was \n"); print();
      children[i] = newchild;
      newchild->setParent( this );
      //debug_fprintf(stderr, "is  \n");  print(); debug_fprintf(stderr, "\n\n"); 
      // old->parent = NULL; 
      any = 1;
    }
  }

  if (!any) { 
    debug_fprintf(stderr, "chillAST_CompoundStmt::replaceChild(), could not find old\n");
    exit(-1); 
  }
}


void chillAST_CompoundStmt::loseLoopWithLoopVar( char *var ) { 
  //debug_fprintf(stderr, "chillAST_CompoundStmt::loseLoopWithLoopVar( %s )\n", var); 

  //debug_fprintf(stderr, "CompoundStmt 0x%x has parent 0x%x  ", this, this->parent);
  //debug_fprintf(stderr, "%s\n", parent->getTypeString()); 

  
  //debug_fprintf(stderr, "CompoundStmt node has %d children\n", children.size()); 
  //debug_fprintf(stderr, "before doing a damned thing, \n"); 
  //print();
  //dump(); fflush(stdout);
  //debug_fprintf(stderr, "\n\n"); 

#ifdef DAMNED
  for (int j=0; j<children.size(); j++) { 
    debug_fprintf(stderr, "j %d/%d  ", j, children.size()); 
    debug_fprintf(stderr, "subnode %d 0x%x  ", j, children[j] );
    debug_fprintf(stderr, "asttype %d  ", children[j]->asttype); 
    debug_fprintf(stderr, "%s    ", children[j]->getTypeString());
    if (children[j]->isForStmt()) { 
      chillAST_ForStmt *FS = ((chillAST_ForStmt *)  children[j]); 
      debug_fprintf(stderr, "for (");
      FS->init->print(0, stderr);
      debug_fprintf(stderr, "; ");
      FS->cond->print(0, stderr);
      debug_fprintf(stderr, "; ");
      FS->incr->print(0, stderr);
      debug_fprintf(stderr, ")  with %d statements in body 0x%x\n",  FS->body->getNumChildren(), FS->body );   
    }
    else debug_fprintf(stderr, "\n"); 
  }
#endif


  vector<chillAST_node*> dupe = children; // simple enough?
  for (int i=0; i<dupe.size(); i++) { 
    //for (int j=0; j<dupe.size(); j++) { 
    //  debug_fprintf(stderr, "j %d/%d\n", j, dupe.size()); 
    //  debug_fprintf(stderr, "subnode %d %s    ", j, children[j]->getTypeString());
    //  if (children[j]->isForStmt()) { 
    //    chillAST_ForStmt *FS = ((chillAST_ForStmt *)  children[j]); 
    //    debug_fprintf(stderr, "for (");
    //     FS->init->print(0, stderr);
    //    debug_fprintf(stderr, "; ");
    //    FS->cond->print(0, stderr);
    //    debug_fprintf(stderr, "; ");
    //    FS->incr->print(0, stderr);
    //    debug_fprintf(stderr, ")  with %d statements in body 0x%x\n",  FS->body->getNumChildren(), FS->body );   
    //} 
    //else debug_fprintf(stderr, "\n"); 
    //}
    
    //debug_fprintf(stderr, "CompoundStmt 0x%x recursing to child %d/%d\n", this, i, dupe.size()); 
    dupe[i]->loseLoopWithLoopVar( var );
  }
  //debug_fprintf(stderr, "CompoundStmt node 0x%x done recursing\n", this ); 
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
    fprintf(fp, "\n"); // ???
    fflush(fp);
  }
  chillindent(indent, fp); 
  fprintf(fp, ")\n"); 
};



chillAST_node*  chillAST_CompoundStmt::constantFold(){ 
  //debug_fprintf(stderr, "chillAST_CompoundStmt::constantFold()\n"); 
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
  //debug_fprintf(stderr, "sofar %d   reserved %d\n", sofar, 2*sofar); 

  bool force = false;
  for (int i=0; i<children.size(); i++) {  // children.size() to see it gain each time
    if (children.size() > sofar ) { 
      //debug_fprintf(stderr, "HEY! CompoundStmt::findLoopIndexesToReplace() noticed that children increased from %d to %d\n", sofar, children.size()); 
      sofar = children.size(); 
    }

    //debug_fprintf(stderr, "compound child %d of type %s force %d\n", i, children[i]->getTypeString(), force ); 
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
    debug_fprintf(stderr, "ORIGINAL compound child %d of type %s\n", i, children[i]->getTypeString() ); 
    origtypes[i] = strdup( children[i]->getTypeString() ); 
    debug_fprintf(stderr, "ORIGINAL compound child %d of type %s\n", i, children[i]->getTypeString() ); 
  }
    
  for (int i=0; i<childrencopy.size(); i++) { 
    debug_fprintf(stderr, "compound child %d of type %s force %d\n", i, childrencopy[i]->getTypeString(), force ); 
    force = force || childrencopy[i]->findLoopIndexesToReplace( symtab, force ); // once set, always
  }

  debug_fprintf(stderr, "\n"); 
  for (int i=0; i<origsize; i++) { 
    debug_fprintf(stderr, "BEFORE compound child %d/%d of type %s\n",  i, origsize, origtypes[i]); 
  }
  for (int i=0; i<children.size(); i++) { 
    debug_fprintf(stderr, "AFTER  compound child %d/%d of type %s\n", i, children.size(), children[i]->getTypeString() ); 
  }

  return false;
*/ 
}





chillAST_ParenExpr::chillAST_ParenExpr(  chillAST_node *sub, chillAST_node *par ):subexpr(this, 0){
  subexpr = sub;
  subexpr->setParent( this );
  parent = par;
}

void chillAST_ParenExpr::print(  int indent,  FILE *fp ) { 
  //debug_fprintf(stderr, "chillAST_ParenExpr::print()\n"); 
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

void chillAST_ParenExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  subexpr->replaceVarDecls( olddecl, newdecl ); 
}

chillAST_Sizeof::chillAST_Sizeof( char *athing, chillAST_node *par ){
  thing = strdup( athing ); // memory leak
  parent = par;
}

void chillAST_Sizeof::print(  int indent,  FILE *fp ) { 
  //debug_fprintf(stderr, "chillAST_Sizeof::print()\n"); 
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

void insertNewDeclAtLocationOfOldIfNeeded( chillAST_VarDecl *newdecl, chillAST_VarDecl *olddecl) {
  //debug_fprintf(stderr, "insertNewDeclAtLocationOfOldIfNeeded( new 0x%x  old 0x%x\n", newdecl, olddecl );

  if (newdecl == NULL || olddecl == NULL) {
    debug_fprintf(stderr, "chill_ast.cc insertNewDeclAtLocationOfOldIfNeeded() NULL decl\n");
    exit(-1);
  }

  if (newdecl == olddecl) return;

  newdecl->vartype = strdup(olddecl->vartype);

  chillAST_node *newparent = newdecl->parent;
  chillAST_node *oldparent = olddecl->parent;
  //debug_fprintf(stderr, "newparent 0x%x   oldparent 0x%x\n", newparent, oldparent ); 
  if (newparent == oldparent) return;

  if (newparent != NULL) 
    //debug_fprintf(stderr, "chill_ast.cc insertNewDeclAtLocationOfOldIfNeeded() new decl already has parent??  probably wrong\n");
  newdecl->parent = oldparent;  // will be true soon 

  // find actual location of old decl and insert new one there
  //debug_fprintf(stderr, "oldparent is of type %s\n", oldparent->getTypeString()); // better be compoundstmt ??
  vector<chillAST_node*> children = oldparent->getChildren();
  
  int numchildren = children.size(); 
  //debug_fprintf(stderr, "oldparent has %d children\n", numchildren); 
  
  if (numchildren == 0) {
    debug_fprintf(stderr, "chill_ast.cc insertNewDeclAtLocationOfOldIfNeeded() impossible number of oldparent children (%d)\n", numchildren); 
    exit(-1);
  }

  bool newalreadythere = false;
  int index = -1;
  //debug_fprintf(stderr, "olddecl is 0x%x\n", olddecl); 
  //debug_fprintf(stderr, "I know of %d variables\n", numchildren);
  for (int i=0; i<numchildren; i++) { 
    chillAST_node *child = oldparent->getChild(i); 
    //debug_fprintf(stderr, "child %d @ 0x%x is of type %s\n", i, child, child->getTypeString()); 
    if (children[i] == olddecl) { 
      index = i;
      //debug_fprintf(stderr, "found old decl at index %d\n", index); 
    }
    if (children[i] == newdecl) {  
      newalreadythere = true; 
      //debug_fprintf(stderr, "new already there @ index %d\n", i); 
    }
  }
  if (index == -1) { 
    debug_fprintf(stderr, "chill_ast.cc insertNewDeclAtLocationOfOldIfNeeded() can't find old decl for %s\n", olddecl->varname);
    exit(-1);
  }

  if (!newalreadythere) oldparent->insertChild( index, newdecl );

}


void gatherVarDecls( vector<chillAST_node*> &code, vector<chillAST_VarDecl*> &decls) {
  //debug_fprintf(stderr, "gatherVarDecls()\n");

  int numcode = code.size();
  //debug_fprintf(stderr, "%d top level statements\n", numcode);
  for (int i=0; i<numcode; i++) {
    chillAST_node *statement = code[i];
    statement->gatherVarDecls( decls );
  }

}


void gatherVarUsage( vector<chillAST_node*> &code, vector<chillAST_VarDecl*> &decls) {
  //debug_fprintf(stderr, "gatherVarUsage()\n");

  int numcode = code.size();
  //debug_fprintf(stderr, "%d top level statements\n", numcode);
  for (int i=0; i<numcode; i++) {
    chillAST_node *statement = code[i];
    statement->gatherVarUsage( decls );
  }

}




chillAST_IfStmt::chillAST_IfStmt():cond(this,0),thenpart(this,1),elsepart(this,2) {
}

chillAST_IfStmt::chillAST_IfStmt(chillAST_node *c, chillAST_node *t, chillAST_node *e, chillAST_node *p):chillAST_IfStmt(){
  cond = c;
  thenpart = t;
  elsepart = e;
  parent = p;
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
  //debug_fprintf(stderr, "ifstmt, after then, %d statements\n", statements.size()); 
  if (elsepart){ 
    //debug_fprintf(stderr, "there is an elsepart of type %s\n", elsepart->getTypeString()); 
    elsepart->gatherStatements( statements );
  }
  //debug_fprintf(stderr, "ifstmt, after else, %d statements\n", statements.size()); 
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
  if (cond) cond->print(0, fp);
  else fprintf(fp, "(NULL cond)"); 
  
  bool needbracket = true; 
  if (thenpart) { 
  if (thenpart->isBinaryOperator()) needbracket = false;
  if (thenpart->isCompoundStmt()) { // almost always true
    chillAST_CompoundStmt *CS = (chillAST_CompoundStmt*) thenpart();
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
  }
  else fprintf(fp, "(NULL thenpart)");

  
  needbracket = true;
  if (elsepart) { 
    if (elsepart->isBinaryOperator()) needbracket = false;
    if (elsepart->isCompoundStmt()) { // almost always true
      chillAST_CompoundStmt *CS = (chillAST_CompoundStmt*) elsepart();
      
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
  //debug_fprintf(stderr, "findmanually()                CHILL AST node of type %s\n", node->getTypeString()); 
  
  if (node->isFunctionDecl()) { 
    char *name = ((chillAST_FunctionDecl *) node)->functionName; // compare name with desired name
    //debug_fprintf(stderr, "node name 0x%x  ", name);
    //debug_fprintf(stderr, "%s     procname ", name); 
    //debug_fprintf(stderr, "0x%x  ", procname);
    //debug_fprintf(stderr, "%s\n", procname); 
    if (!strcmp( name, procname)) {
      //debug_fprintf(stderr, "found procedure %s\n", procname ); 
      funcs.push_back( (chillAST_FunctionDecl*) node );  // this is it 
      // quit recursing. probably not correct in some horrible case
      return; 
    }
    //else debug_fprintf(stderr, "this is not the function we're looking for\n"); 
  }


  // this is where the children can be used effectively. 
  // we don't really care what kind of node we're at. We just check the node itself
  // and then its children is needed. 

  int numc = node->children.size();  
  debug_fprintf(stderr, "(top)node has %d children\n", numc);

  for (int i=0; i<numc; i++) {
    if (node->isSourceFile()) { 
      debug_fprintf(stderr, "node of type %s is recursing to child %d of type %s\n",  node->getTypeString(), i,  node->children[i]->getTypeString()); 
      if (node->children[i]->isFunctionDecl()) { 
        chillAST_FunctionDecl *fd = (chillAST_FunctionDecl*) node->children[i];
        debug_fprintf(stderr, "child %d is functiondecl %s\n", i, fd->functionName); 
      }
    }
    findFunctionDeclRecursive( node->children[i], procname, funcs );
    
  }
  return; 
}


chillAST_FunctionDecl *findFunctionDecl( chillAST_node *node, const char *procname)
{
  vector<chillAST_FunctionDecl*> functions;
  findFunctionDeclRecursive( node, procname, functions );  

  if ( functions.size() == 0 ) { 
    debug_fprintf(stderr, "could not find function named '%s'\n", procname);
    exit(-1);
  }
  
  if ( functions.size() > 1 ) { 
    debug_fprintf(stderr, "oddly, found %d functions named '%s'\n", functions.size(), procname);
    debug_fprintf(stderr, "I am unsure what to do\n"); 

    for (int f = 0; f < functions.size(); f++) { 
      debug_fprintf(stderr, "function %d  %p   %s\n", f, functions[f], functions[f]->functionName); 
    }
    exit(-1);
  }
  
  //debug_fprintf(stderr, "found the procedure named %s\n", procname); 
  return functions[0];
}


chillAST_SymbolTable *addSymbolToTable(  chillAST_SymbolTable *st, chillAST_VarDecl *vd ) // definition
{
  chillAST_SymbolTable *s = st;
  if (!s) s = new chillAST_SymbolTable; 
 
  int tablesize = s->size();
  
  for (int i=0; i<tablesize; i++) { 
    if ((*s)[i] == vd) { 
      //debug_fprintf(stderr, "the exact same symbol, not just the same name, was already there\n"); 
      return s; // already there 
    }
  }

  for (int i=0; i<tablesize; i++) { 
    //debug_fprintf(stderr, "name %s vs name %s\n", (*s)[i]->varname, vd->varname); 
    if (!strcmp( (*s)[i]->varname, vd->varname)) { 
      //debug_fprintf(stderr, "symbol with the same name was already there\n"); 
      return s; // already there 
    }
  }
  s->push_back(vd); // add it
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
