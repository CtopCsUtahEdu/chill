


#include "chill_ast.hh"
#include "printer/dump.h"
#include "printer/cfamily.h"
#include <fstream>
#include <cstring>

int chillAST_node::chill_scalar_counter = 0;
int chillAST_node::chill_array_counter  = 1;

#ifdef chillast_nodetype
#error "chillast_nodetype already defined"
#else
#define chillast_nodetype(n, s)                   s,
#define chillast_nodetype_alias(a, b)             /* nothing */
#endif

const char* Chill_AST_Node_Names[] = { 
  "Unknown AST node type",
#include "chill_ast.def"
};

#undef chillast_nodetype
#undef chillast_nodetype_alias

char *parseUnderlyingType( char *sometype ) {
    int len = strlen(sometype);
  char *underlying = strdup(sometype);
  char *p;

  p = &underlying[len - 1];

  while (p > underlying)
    if (*p == ' ' || *p == '*')
      --p;
    else if (*p == ']') {
      while (*p != '[') --p;
      --p;
    }
    else break;

  *(p + 1) = '\0';

  return underlying;
}

void printSymbolTable( chillAST_SymbolTable *st ) {
  if (!st) return;
  for (int i=0; i<st->size(); i++) {  printf("%d  %s", i , (*st)[i]->varname); printf("\n"); }
  if (st->size() )printf("\n");
  fflush(stdout);
}

void printSymbolTableMoreInfo( chillAST_SymbolTable *st ) {
  if (!st) return;
  for (int i=0; i<st->size(); i++) {  printf("%d  %s", i , (*st)[i]->stringRep().c_str()); printf("\n"); }
  if (st->size() )printf("\n");
  fflush(stdout);
}

chillAST_VarDecl* symbolTableFindVariableNamed( chillAST_SymbolTable *table, const char *name ) {
  if (!table) return nullptr; // ??
  for (auto vd: *table)
    if (!strcmp(name, vd->varname)) return vd;
  return nullptr;
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

char *splitTypeInfo( char *underlyingtype ) {
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

    strcpy( underlyingtype, underlyingtype);
  }
  return arraypart;  // leak unless caller frees this
}

bool isRestrict( const char *sometype ) { // does not modify sometype
  string r( "__restrict__" );
  string t( sometype );
  return (std::string::npos != t.find( r ) );
}



bool streq( const char *a, const char *b) { return !strcmp(a,b); };  // slightly less ugly // TODO enums

void chillindent( int howfar, FILE *fp ) { for (int i=0; i<howfar; i++) fprintf(fp, "  ");  }


void chillAST_node::print( int indent,  std::ostream& o ) {
  std::string ind = "";
  while (indent-- > 0)
    ind += "  ";
  chill::printer::CFamily c;
  c.print(this, ind, o);
}
void chillAST_node::dump( int indent,  std::ostream& o ) {
  std::string ind = "";
  while (indent-- > 0)
    ind += "  ";
  chill::printer::Dump c;
  c.print(this, ind, o);
}

std::string chillAST_node::stringRep(int indent) {
  std::string ind = "";
  while (indent-- > 0)
    ind += "  ";
  chill::printer::CFamily c;
  return c.print(this, ind);
}

chillAST_VarDecl * chillAST_node::findVariableNamed( const char *name ) { // recursive
  if (hasSymbolTable()) { // look in my symbol table if I have one
    chillAST_VarDecl *vd = symbolTableFindVariableNamed( getSymbolTable(), name);
    if (vd) return vd; // found locally
  }
  if (!parent) return nullptr; // no more recursion available
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
      auto RD = (chillAST_RecordDecl *)children[i];
      debug_fprintf(stderr, "it is a recordDecl named '%s' vs '%s'\n", RD->getName(), name); 
      if (!strcmp( RD->getName(), name )) {
        debug_fprintf(stderr, "FOUND IT\n"); 
        return RD;
      }
    }
  }   
    
  if (!parent) return nullptr; // no more recursion available
  // recurse upwards
  return parent->findRecordDeclNamed( name ); 
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

void chillAST_SourceFile::printToFile( char *filename ) {
  std::string fn;

  if (filename)
    fn = filename;
  else {
    // build up a filename using original name and frontend if known
    if (FileToWrite)
      fn = FileToWrite;
    else {
      // input name with name of frontend compiler prepended
      if (frontend)
        fn = std::string(frontend) + "_" + SourceFileName;
      else
        fn = std::string("UNKNOWNFRONTEND_") + SourceFileName; // should never happen
    }
  }

  std::ofstream fp(fn.c_str());
  
  print(0, fp);
}


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


chillAST_TypedefDecl::chillAST_TypedefDecl(const char *t, const char *nt):chillAST_TypedefDecl() {
  //debug_fprintf(stderr, "chillAST_TypedefDecl::chillAST_TypedefDecl( underlying type %s, newtype %s )\n", t, nt); 
  underlyingtype = strdup(t); 
  newtype = strdup(nt);
};


chillAST_TypedefDecl::chillAST_TypedefDecl(const char *t, const char *a, const char *p):chillAST_TypedefDecl() {
  underlyingtype = strdup(t); 
  newtype = strdup(a);  // the new named type ?

  arraypart = strdup(p);  // array (p)art? 
  // splitarraypart(); // TODO
};

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



void chillAST_VarDecl::loseLoopWithLoopVar(char* var) {
    /* do nothing */
}



chillAST_RecordDecl::chillAST_RecordDecl() { 
  name = NULL; // ??
  originalname = NULL;      // ?? 
  isStruct = isUnion = false;
}

chillAST_RecordDecl::chillAST_RecordDecl( const char *nam):chillAST_RecordDecl() {
  if (nam) name = strdup(nam);
  else name = strdup("unknown"); // ?? 
  originalname = NULL;      // ??   // make them do it manually?
}

chillAST_RecordDecl::chillAST_RecordDecl( const char *nam, const char *orig):chillAST_RecordDecl() {
  //debug_fprintf(stderr, "chillAST_RecordDecl::chillAST_RecordDecl( %s, (orig) )\n", nam); 
  if (nam) name = strdup(nam);
  else name = strdup("unknown"); // ?? 
  
  originalname = NULL;   
  if (orig) originalname = strdup(orig);
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

chillAST_FunctionDecl::chillAST_FunctionDecl():body(this,0) {
  functionName = strdup("YouScrewedUp"); 
  forwarddecl = externfunc = builtin = false;
  uniquePtr = (void *) NULL;
  this->setFunctionCPU(); 
  //symbol_table = NULL;   // eventually, pointing to body's symbol table
  typedef_table = NULL;
  body = new chillAST_CompoundStmt();
};


chillAST_FunctionDecl::chillAST_FunctionDecl(const char *rt, const char *fname ):body(this,0) {
  returnType = strdup(rt);
  functionName = strdup(fname);
  this->setFunctionCPU(); 
  //debug_fprintf(stderr, "functionName %s\n", functionName); 
  forwarddecl = externfunc = builtin = false; 

  // symbol_table = NULL; //use body's instead
  typedef_table = NULL;
  body = new chillAST_CompoundStmt();
};


chillAST_FunctionDecl::chillAST_FunctionDecl(const char *rt, const char *fname , void *unique):body(this,0) {
  returnType = strdup(rt);
  functionName = strdup(fname);
  this->setFunctionCPU(); 
  //debug_fprintf(stderr, "functionName %s\n", functionName); 
  forwarddecl = externfunc = builtin = false; 

  body = new chillAST_CompoundStmt();
  uniquePtr = unique; // a quick way to check equivalence. DO NOT ACCESS THROUGH THIS
  //symbol_table = NULL; // use body's
  typedef_table = NULL;
};


void chillAST_FunctionDecl::addParameter( chillAST_VarDecl *p) {
  debug_fprintf(stderr, "%s chillAST_FunctionDecl::addParameter( 0x%x  param %s)   total of %d parameters\n", functionName, p, p->varname, 1+parameters.size()); 

  if (symbolTableFindVariableNamed( &parameters, p->varname)) { // NOT recursive. just in FunctionDecl
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
      body->insertChild(0, used[j]);
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
  macroName = NULL;
  rhsString = NULL;
  symbol_table = NULL;
  //rhsideString = NULL;
};


chillAST_MacroDefinition::chillAST_MacroDefinition(const char *mname):chillAST_MacroDefinition() {
  macroName = strdup(mname);

  //TODO getSourceFile()->addMacro( this );
};


chillAST_MacroDefinition::chillAST_MacroDefinition(const char *mname, const char *rhs) {
  macroName = strdup(mname);
  rhsString = strdup(rhs);
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
  rhsString = strdup(body->stringRep().c_str());
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

chillAST_ForStmt::chillAST_ForStmt():init(this, 0),cond(this,1),incr(this,2),body(this,3) {
  init = cond = incr = NULL;
  body = new chillAST_CompoundStmt();

  conditionoperator = IR_COND_UNKNOWN;
  symbol_table = NULL;
  pragma = NULL;
}


chillAST_ForStmt::chillAST_ForStmt(  chillAST_node *ini, chillAST_node *con, chillAST_node *inc, chillAST_node *bod):chillAST_ForStmt() {
  init = ini;
  cond = con;
  incr = inc;
  body = bod;

  if (!cond->isBinaryOperator()) {
    debug_fprintf(stderr, "ForStmt conditional is of type %s. Expecting a BinaryOperator\n", cond->getTypeString());
    exit(-1); 
  }
  chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)cond;
  char *condstring = bo->op;
  if (!strcmp(condstring, "<"))       conditionoperator = IR_COND_LT;
  else if (!strcmp(condstring, "<=")) conditionoperator = IR_COND_LE;
  else if (!strcmp(condstring, ">"))  conditionoperator = IR_COND_GT;
  else if (!strcmp(condstring, ">=")) conditionoperator = IR_COND_GE;
  else {
    // TODO this is wrong, unhandled will be marked and treated as a block in IR_chill*
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
    
    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)init;
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
    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)cond;
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

    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)init;
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
    chillAST_BinaryOperator *bo = (chillAST_BinaryOperator *)cond;
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


chillAST_node *chillAST_ForStmt::constantFold() { 
   init = init->constantFold(); 
   cond = cond->constantFold(); 
   incr = incr->constantFold(); 
   body = body->constantFold(); 
   return this; 
 }


chillAST_node *chillAST_ForStmt::clone() {
  chillAST_ForStmt *fs = new chillAST_ForStmt( init->clone(), cond->clone(), incr->clone(), body->clone());
  fs->parent = parent;
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
      newguy = new chillAST_VarDecl( olddecl->vartype, "", vname );
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
      contain->prependStatement( newguy );
      contain->addVariableToSymbolTable( newguy ); // adds to first enclosing symbolTable
      
      if (!  symbolTableFindVariableNamed( contain->getSymbolTable(), vname )) {
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
  // now recurse (could do first, I suppose)
  // if you DON'T do this first, you may have already replaced yourself with this loop body
  // the body will no longer have this forstmt as parent, it will have the forstmt's parent as its parent
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
  
  if (!strcmp(var,  loopvars[0].c_str())) {
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
      chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *) cond;
      if (BO->rhs->isTernaryOperator()) { 

        chillAST_TernaryOperator *TO = (chillAST_TernaryOperator *)BO->rhs;
        chillAST_BinaryOperator *C =  (chillAST_BinaryOperator *)TO->condition;
        
        C->print(); printf("\n"); fflush(stdout);
        chillAST_node *l = C->lhs;
        if (l->isParenExpr()) l = ((chillAST_ParenExpr *)l)->subexpr; 
        chillAST_node *r = C->rhs;
        if (r->isParenExpr()) r = ((chillAST_ParenExpr *)r)->subexpr; 

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
        chillAST_IfStmt *ifstmt = new chillAST_IfStmt( ifcond, body, NULL );
        
        newstmt = ifstmt; 
      }
    }

    parent->replaceChild( this, newstmt );
  }


}

chillAST_WhileStmt::chillAST_WhileStmt(chillAST_node *cond, chillAST_node *body):chillAST_WhileStmt() {
  this->cond = cond;
  this->body = body;
}

chillAST_node* chillAST_WhileStmt::clone() {
  chillAST_node* c = cond->clone();
  chillAST_node* b = body->clone();
  chillAST_WhileStmt *ws =  new chillAST_WhileStmt( c, b );
  ws->isFromSourceFile = isFromSourceFile;
  if (filename) ws->filename = strdup(filename);
  return ws;
}

void chillAST_WhileStmt::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*>  &refs, bool w ) {
  cond->gatherArrayRefs( refs, false );
  body->gatherArrayRefs( refs, false );
}

void chillAST_WhileStmt::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {
  cond->gatherScalarRefs( refs, false );
  body->gatherScalarRefs( refs, false );
}


chillAST_BinaryOperator::chillAST_BinaryOperator():lhs(this,0),rhs(this,1) {
  //debug_fprintf(stderr, "chillAST_BinaryOperator::chillAST_BinaryOperator()  %p\n", this);
  op = NULL;
}


chillAST_BinaryOperator::chillAST_BinaryOperator(chillAST_node *l, const char *oper, chillAST_node *r):chillAST_BinaryOperator() {
  lhs = l;
  rhs = r;
  op = strdup(oper);
  // if this writes to lhs and lhs type has an 'imwrittento' concept, set that up
  // TODO move it to canonicalization pass
  if (isAssignmentOp()) { 
    if (lhs && lhs->isArraySubscriptExpr())
      ((chillAST_ArraySubscriptExpr*)lhs)->imwrittento = true;
  }
  if (isAugmentedAssignmentOp()) {  // +=  etc 
    //debug_fprintf(stderr, "isAugmentedAssignmentOp()  "); print(); fflush(stdout); 
    if (lhs && lhs->isArraySubscriptExpr()) { 
      //debug_fprintf(stderr, "lhs is also read from  ");  lhs->print(); fflush(stdout); 
      ((chillAST_ArraySubscriptExpr*)lhs)->imreadfrom = true; // note will ALSO have imwrittento true
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

  throw std::runtime_error(std::string("chillAST_BinaryOperator::evalAsInt() unhandled op ") + op);
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
        chillAST_IntegerLiteral *l = (chillAST_IntegerLiteral *)lhs;
        chillAST_IntegerLiteral *r = (chillAST_IntegerLiteral *)rhs;
        chillAST_IntegerLiteral *I;
        
        if (streq(op, "+")) I = new chillAST_IntegerLiteral(l->value+r->value);
        if (streq(op, "-")) I = new chillAST_IntegerLiteral(l->value-r->value);
        if (streq(op, "*")) I = new chillAST_IntegerLiteral(l->value*r->value);

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
  chillAST_BinaryOperator *bo =  new chillAST_BinaryOperator( l, op, r );
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

void chillAST_BinaryOperator::gatherStatements(std::vector<chillAST_node*> &statements ){
  
  // what's legit? TODO No
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

chillAST_TernaryOperator::chillAST_TernaryOperator(const char *oper, chillAST_node *c, chillAST_node *l, chillAST_node *r):chillAST_TernaryOperator() {
  op = strdup(oper);
  condition = c;
  lhs = l;
  rhs = r;
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
  chillAST_TernaryOperator *to =  new chillAST_TernaryOperator( op, c, l, r );
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



chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, void *unique ):chillAST_ArraySubscriptExpr() {
  if (bas->isImplicitCastExpr()) base = ((chillAST_ImplicitCastExpr*)bas)->subexpr; // probably wrong
  else   base = bas;
  if (indx->isImplicitCastExpr()) index = ((chillAST_ImplicitCastExpr*)indx)->subexpr; // probably wrong
  else index = indx;
  uniquePtr = unique;
  //! basedecl = multibase();//debug_fprintf(stderr, "%p  ASE 1 basedecl = %p\n",this,basedecl);
}



chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr( chillAST_node *bas, chillAST_node *indx, bool writtento, void  *unique ):chillAST_ArraySubscriptExpr() {
  if (bas->isImplicitCastExpr()) base = ((chillAST_ImplicitCastExpr*)bas)->subexpr; // probably wrong
  else base = bas;
  if (indx->isImplicitCastExpr()) index = ((chillAST_ImplicitCastExpr*)indx)->subexpr; // probably wrong
  else index = indx;
  imwrittento = writtento; // ??
  uniquePtr = unique;
  //! basedecl = multibase();
 }



chillAST_ArraySubscriptExpr::chillAST_ArraySubscriptExpr( chillAST_VarDecl *v, std::vector<chillAST_node *> indeces,  chillAST_node *par):chillAST_ArraySubscriptExpr() {
  parent = par;

  int numindeces = indeces.size();

  chillAST_DeclRefExpr *DRE = new chillAST_DeclRefExpr( v->vartype, v->varname, v);
  basedecl = v; // ?? 

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



chillAST_node *chillAST_node::getEnclosingStatement() {
  if (!parent)
    return this;
  if (parent->isForStmt()      ||
      parent->isIfStmt()       ||
      parent->isVarDecl()      ||
      parent->isWhileStmt()    ||
      parent->isCompoundStmt() ||
      parent->isSourceFile()   ||
      parent->isFunctionDecl())
    return this;
  return parent->getEnclosingStatement();
}



void chillAST_ArraySubscriptExpr::gatherIndeces(std::vector<chillAST_node*>&ind) { 
  if (base->isArraySubscriptExpr()) ((chillAST_ArraySubscriptExpr *)base)->gatherIndeces( ind );
  ind.push_back( index );
}

chillAST_VarDecl *chillAST_ArraySubscriptExpr::multibase() {
  if (basedecl) return basedecl;
  return basedecl = base->multibase();
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
    chillAST_VarDecl *vd = (chillAST_VarDecl *)(((chillAST_DeclRefExpr *)base)->decl);
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
  chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( b, i, imwrittento, uniquePtr /* ?? */ );
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
  
  print(0,stderr); debug_fprintf(stderr, "\nchild: ");
  if (!old) debug_fprintf(stderr, "oldchild NULL!\n");
  old->print(0,stderr); debug_fprintf(stderr, "\nnew: "); 
  newchild->print(0,stderr); debug_fprintf(stderr, "\n");
  throw std::runtime_error("chillAST_ArraySubscriptExpr::replaceChild() old is not base or index");
};


bool chillAST_ArraySubscriptExpr::operator!=( const chillAST_ArraySubscriptExpr &other) {
  bool opposite = *this == other;
  return !opposite;
}



bool chillAST_ArraySubscriptExpr::operator==( const chillAST_ArraySubscriptExpr &other) {
  return this->uniquePtr == other.uniquePtr;
}





chillAST_MemberExpr::chillAST_MemberExpr():base(this,0) {
  basedecl = NULL;
  member = NULL;
  exptype = CHILL_MEMBER_EXP_DOT;
}

chillAST_MemberExpr::chillAST_MemberExpr( chillAST_node *bas, const char *mem, void *unique, CHILL_MEMBER_EXP_TYPE t ):chillAST_MemberExpr() {
  base = bas;
  if (mem)    member = strdup( mem );
  uniquePtr = unique;
  exptype = t;

  return;
}

class chillAST_node* chillAST_MemberExpr::constantFold() {
  base  =  base->constantFold();
  //member = member->constantFold();
  return this;
}

class chillAST_node* chillAST_MemberExpr::clone() { 
  chillAST_node *b =  base->clone();
  char *m = strdup( member ); // ?? 
  chillAST_MemberExpr *ME = new chillAST_MemberExpr( b, m, uniquePtr /* ?? */ );
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

chillAST_VarDecl* chillAST_MemberExpr::getUnderlyingVarDecl() {
  debug_fprintf(stderr, "chillAST_MemberExpr:getUnderlyingVarDecl()\n");
  print(); printf("\n"); fflush(stdout);
  exit(-1); 
  // find the member with the correct name
  
}




chillAST_VarDecl *chillAST_MemberExpr::multibase() {
  if (basedecl) return basedecl;
  chillAST_VarDecl *vd = base->multibase(); // ??

  chillAST_RecordDecl *rd = vd->getStructDef();
  if (!rd)
    throw std::runtime_error("chillAST_MemberExpr::multibase() vardecl is not a struct??");

  // OK, we have the recorddecl that defines the structure
  // now find the member with the correct name
  basedecl = rd->findSubpart( member );

  if (!basedecl) {
    debug_fprintf(stderr, "can't find member %s in \n", member);
    rd->print();
  }

  return basedecl;
  //find vardecl of member in def of base
}




chillAST_DeclRefExpr::chillAST_DeclRefExpr() { 
  declarationType = NULL;
  declarationName = NULL;
  decl = NULL;
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( const char *varname): chillAST_DeclRefExpr() {
  declarationName = strdup(varname);
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( const char *vartype, const char *varname): chillAST_DeclRefExpr() {
  declarationType = strdup(vartype);
  declarationName = strdup(varname); 
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( const char *vartype, const char *varname, chillAST_node *d) {
  //debug_fprintf(stderr, "DRE::DRE2 0x%x   %s %s  0x%x\n", this, vartype, varname, d );
  declarationType = vartype? strdup(vartype): nullptr;
  declarationName = varname? strdup(varname): nullptr;
  decl = d; 
}

chillAST_DeclRefExpr::chillAST_DeclRefExpr( chillAST_node *d){ // variable def
  if (d->isVarDecl()) {
    auto vd = dynamic_cast<chillAST_VarDecl*>(d);
    declarationType = strdup(vd->vartype);
    declarationName = strdup(vd->varname);
    decl = vd;
  } else if (d->isFunctionDecl()) {
    auto fd = dynamic_cast<chillAST_FunctionDecl*>(d);
    declarationType = strdup(fd->returnType);
    declarationName = strdup(fd->functionName);
    decl = fd;
  } else if (d->isMacroDefinition()) {
    auto md = dynamic_cast<chillAST_MacroDefinition*>(d);
    declarationName = strdup(md->macroName);
    declarationType = nullptr;
    decl = md;
  } else throw std::runtime_error("Illegal DeclRefExpr");
}

class chillAST_node* chillAST_DeclRefExpr::constantFold() {  // can never do anything?
  return this;
}

class chillAST_node* chillAST_DeclRefExpr::clone() { 
  //debug_fprintf(stderr, "chillAST_DeclRefExpr::clone()\n"); 
  chillAST_DeclRefExpr *DRE =  new chillAST_DeclRefExpr( declarationType, declarationName, decl);
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


void chillAST_VarDecl::convertArrayToPointer() {

    if (numdimensions == 0) {
        // not an array
        // TODO: this is an error
    }

    // Array dimensions sizes are stored in the nodes children
    //   so kick the first one out
    this->removeChild(0);

    auto ln = strlen(this->arraypointerpart);
    char* new_arraypointerpart = (char*) malloc(sizeof(char) * ln + 2);
    memcpy(new_arraypointerpart, this->arraypointerpart, ln);
    new_arraypointerpart[ln    ] = '*';
    new_arraypointerpart[ln + 1] = '\0';
    free(this->arraypointerpart);
    this->arraypointerpart = new_arraypointerpart;
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



chillAST_node *chillAST_VarDecl::constantFold() {
  if (init)
    init = init->constantFold();
  return this;
}

chillAST_node* chillAST_VarDecl::clone() {
  //debug_fprintf(stderr, "\nchillAST_VarDecl::clone()  cloning vardecl for %s\n", varname); 
  //if (isAParameter) debug_fprintf(stderr, "old vardecl IS a parameter\n");
  //else  debug_fprintf(stderr, "old vardecl IS NOT a parameter\n");

  chillAST_VarDecl *vd  = new chillAST_VarDecl( vartype, arraypointerpart, strdup(varname), children, NULL);  // NULL so we don't add the variable AGAIN to the (presumably) function
  
  vd->typedefinition = typedefinition;
  vd->vardef = vardef; // perhaps should not do this     TODO 

  vd->underlyingtype = strdup(underlyingtype); 

  vd->numdimensions = numdimensions;
  vd->arraypointerpart = NULL;

  if ( arraypointerpart ) { 
    //debug_fprintf(stderr, "copying arraypointerpart\n"); 
    vd->arraypointerpart = strdup( arraypointerpart);
  }

  vd->isStruct = this->isStruct;
  vd->isAParameter = this->isAParameter;

  vd->isFromSourceFile = isFromSourceFile;
  if (filename) vd->filename = strdup(filename); 
  return vd;
}

chillAST_IntegerLiteral::chillAST_IntegerLiteral(int val){
  value = val; 
}

class chillAST_node* chillAST_IntegerLiteral::constantFold() { return this; } // can never do anything


class chillAST_node* chillAST_IntegerLiteral::clone() { 
  
  chillAST_IntegerLiteral *IL = new  chillAST_IntegerLiteral( value );
  IL->isFromSourceFile = isFromSourceFile; 
  if (filename) IL->filename = strdup(filename); 
  return IL; 

}
  
chillAST_FloatingLiteral::chillAST_FloatingLiteral(float val){
  value = val; 
  precision = 1;
  allthedigits = NULL;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(double val){
  value = val;
  precision = 2;
  allthedigits = NULL;
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(double val, int precis){
  value = val; 
  precision = precis;
  allthedigits = NULL; 
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(double val, const char *printthis){
  value = val; 
  precision = 2;
  allthedigits = NULL;
  if (printthis) allthedigits = strdup( printthis );
}

chillAST_FloatingLiteral::chillAST_FloatingLiteral(double val, int precis, const char *printthis){
  value = val; 
  precision = precis;
  allthedigits = NULL;
  if (printthis)
    allthedigits = strdup( printthis );
}


chillAST_FloatingLiteral::chillAST_FloatingLiteral( chillAST_FloatingLiteral *old ) {
  //debug_fprintf(stderr, "chillAST_FloatingLiteral::chillAST_FloatingLiteral( old ) allthedigits %p\n", old->allthedigits); 

  value          = old->value;
  allthedigits = NULL;
  if (old->allthedigits) allthedigits = strdup(old->allthedigits); 
  precision      = old->precision;
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
  return value == o->value; // WARNING, comparing floats with ==
}





chillAST_UnaryOperator::chillAST_UnaryOperator( const char *oper, bool pre, chillAST_node *sub):subexpr(this,0) {
  op = strdup(oper);
  prefix = pre;
  subexpr = sub; 
}

void chillAST_UnaryOperator::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*>  &refs, bool w ) {
  subexpr->gatherArrayRefs( refs, isAssignmentOp()); // 
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
        int intval = ((chillAST_IntegerLiteral*)subexpr)->value;
        chillAST_IntegerLiteral *I = new chillAST_IntegerLiteral( -intval );
        returnval = I;
        //debug_fprintf(stderr, "integer -%d becomes %d\n", intval, I->value);
      }
      else { 
        chillAST_FloatingLiteral *FL = (chillAST_FloatingLiteral*)subexpr;
        chillAST_FloatingLiteral *F = new chillAST_FloatingLiteral( FL ); // clone
        F->value = -F->value;
        F->allthedigits = NULL;
        returnval = F;
      }
    }
    else debug_fprintf(stderr, "can't fold op '%s' yet\n", op); 
  }    
  return returnval;
}


class chillAST_node* chillAST_UnaryOperator::clone() { 
  chillAST_UnaryOperator *UO = new chillAST_UnaryOperator( op, prefix, subexpr );
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
  if (!strcmp("++", op)) return (prefix?1:0) + subexpr->evalAsInt();
  if (!strcmp("--", op)) return subexpr->evalAsInt() - (prefix?1:0);

  throw std::runtime_error(std::string("chillAST_UnaryOperator::evalAsInt() unhandled op ") + op);
}

bool chillAST_UnaryOperator::isSameAs( chillAST_node *other ){
  if (!other->isUnaryOperator()) return false;
  chillAST_UnaryOperator *o = (chillAST_UnaryOperator *)other;
  if (strcmp(op, o->op))  return false; // different operators 
  return subexpr->isSameAs( o->subexpr ); // recurse
}


chillAST_ImplicitCastExpr::chillAST_ImplicitCastExpr( chillAST_node *sub):subexpr(this,0) {
  subexpr = sub;
}

class chillAST_node* chillAST_ImplicitCastExpr::constantFold() {
  chillAST_node *child = subexpr->constantFold();
  child->setParent( parent ) ; // remove myself !! probably a bad idea. TODO 
  return child; 
}


class chillAST_node* chillAST_ImplicitCastExpr::clone() { 
  chillAST_ImplicitCastExpr *ICE = new chillAST_ImplicitCastExpr( subexpr->clone() );
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

chillAST_CStyleCastExpr::chillAST_CStyleCastExpr( const char *to, chillAST_node *sub):subexpr(this,0) {
  towhat = strdup(to);
  subexpr = sub;
}

 void chillAST_CStyleCastExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl) { 
   subexpr->replaceVarDecls( olddecl, newdecl);
 }

class chillAST_node* chillAST_CStyleCastExpr::constantFold() { 
  subexpr = subexpr->constantFold();
  int prec = 0;
  if (!strcmp("float", towhat))
    prec = 1;
  else if (!strcmp("double", towhat))
    prec = 2;
  if (subexpr->isConstant() && prec > 0) {
    double val;
    if (subexpr->isIntegerLiteral())
      val = ((chillAST_IntegerLiteral*)subexpr)->value;
    if (subexpr->isFloatingLiteral())
      val = ((chillAST_FloatingLiteral*)subexpr)->value;
    return new chillAST_FloatingLiteral(val,prec);
  }
  if (!strcmp("long", towhat) || !strcmp("int", towhat))
    if (subexpr->isIntegerLiteral())
      return subexpr;
  return this;
}


class chillAST_node* chillAST_CStyleCastExpr::clone() { 
  chillAST_CStyleCastExpr *CSCE = new chillAST_CStyleCastExpr( towhat, subexpr->clone() );
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

chillAST_CStyleAddressOf::chillAST_CStyleAddressOf( chillAST_node *sub):subexpr(this,0) {
  subexpr = sub;
}

class chillAST_node* chillAST_CStyleAddressOf::constantFold() { 
  subexpr = subexpr->constantFold();
  return this; 
}

class chillAST_node* chillAST_CStyleAddressOf::clone() { 
  chillAST_CStyleAddressOf *CSAO = new chillAST_CStyleAddressOf( subexpr->clone() );
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

chillAST_Malloc::chillAST_Malloc(chillAST_node *size):sizeexpr(this,0) {
  thing = NULL;
  sizeexpr = size;  // probably a multiply like   sizeof(int) * 1024
};

chillAST_Malloc::chillAST_Malloc(char *thething, chillAST_node *numthings):sizeexpr(this,0) {
  thing = strdup(thething);   // "int" or "float" or "struct widget"
  sizeexpr = numthings;  
};

chillAST_node* chillAST_Malloc::constantFold() {
  sizeexpr->constantFold(); 
}

chillAST_node* chillAST_Malloc::clone() { 
  chillAST_Malloc *M = new chillAST_Malloc( thing, sizeexpr ); // the general version
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


chillAST_CudaMalloc::chillAST_CudaMalloc(chillAST_node *devmemptr, chillAST_node *size):devPtr(this,0), sizeinbytes(this,1) {
  devPtr = devmemptr; 
  sizeinbytes = size;  // probably a multiply like   sizeof(int) * 1024
};

class chillAST_node* chillAST_CudaMalloc::constantFold() { 
  devPtr = devPtr->constantFold();
  return this; 
}

class chillAST_node* chillAST_CudaMalloc::clone() { 
  chillAST_CudaMalloc *CM = new chillAST_CudaMalloc( devPtr->clone(), sizeinbytes->clone() );
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

chillAST_CudaFree::chillAST_CudaFree(chillAST_VarDecl *var):variable(this,0) {
  variable = var; 
};

class chillAST_node* chillAST_CudaFree::constantFold() { 
  return this; 
}

class chillAST_node* chillAST_CudaFree::clone() { 
  chillAST_CudaFree *CF = new chillAST_CudaFree( variable );
  CF->isFromSourceFile = isFromSourceFile; 
  if (filename) CF->filename = strdup(filename); 
  return CF; 
}

void chillAST_CudaFree::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool w ) {}
void chillAST_CudaFree::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {}

chillAST_CudaMemcpy::chillAST_CudaMemcpy(chillAST_VarDecl *d, chillAST_VarDecl *s, chillAST_node *siz, char *kind):dest(this,0),src(this,1),size(this,2) {
  dest = d;
  src = s;
  size = siz;
  cudaMemcpyKind = kind;
};

class chillAST_node* chillAST_CudaMemcpy::constantFold() {
  dest = (chillAST_VarDecl *)dest->constantFold();
  src  = (chillAST_VarDecl *)src->constantFold();
  size = size->constantFold();
  return this; 
}

class chillAST_node* chillAST_CudaMemcpy::clone() { 
  chillAST_CudaMemcpy *CMCPY = new chillAST_CudaMemcpy((chillAST_VarDecl *)(dest->clone()),(chillAST_VarDecl *)(src->clone()), size->clone(), strdup(cudaMemcpyKind) );
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


chillAST_ReturnStmt::chillAST_ReturnStmt( chillAST_node *retval):returnvalue(this,0) {
  returnvalue = retval;
}

class chillAST_node* chillAST_ReturnStmt::constantFold() { 
  if (returnvalue) returnvalue = returnvalue->constantFold(); 
  return this;
}



class chillAST_node* chillAST_ReturnStmt::clone() { 
  chillAST_node *val = NULL; 
  if ( returnvalue ) val = returnvalue->clone();
  chillAST_ReturnStmt *RS = new chillAST_ReturnStmt( val );
  RS->isFromSourceFile = isFromSourceFile; 
  if (filename) RS->filename = strdup(filename); 
  return RS;
}

chillAST_CallExpr::chillAST_CallExpr(chillAST_node *c):callee(this,0) { //, int numofargs, chillAST_node **theargs ) {
  callee = c;
  numargs = 0;
  grid = block = NULL;
}


void chillAST_CallExpr::addArg( chillAST_node *a ) {
  args.push_back( a );
  a->setParent( this );
  numargs += 1;
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

  chillAST_CallExpr *CE = new chillAST_CallExpr( callee->clone() );
  for (int i=0; i<args.size(); i++) CE->addArg( args[i]->clone() ); 
  CE->isFromSourceFile = isFromSourceFile; 
  if (filename) CE->filename = strdup(filename); 
  return CE; 
}




chillAST_VarDecl::chillAST_VarDecl() { 
  //debug_fprintf(stderr, "chillAST_VarDecl::chillAST_VarDecl()  %p\n", this); 
  vartype = underlyingtype = varname = arraypointerpart = arraysetpart = NULL;
  typedefinition = NULL; 

  //debug_fprintf(stderr, "setting underlying type NULL\n" ); 
  init = NULL;
  numdimensions=0;

  uniquePtr = NULL;
  vardef  = NULL;
  isStruct = false; 
  //insideAStruct = false; 
  isAParameter = false; 
  byreference = false;
  isABuiltin = false; 
  isRestrict = isDevice = isShared = false; // debug_fprintf(stderr, "RDS = false\n"); 
};

chillAST_VarDecl::chillAST_VarDecl( chillAST_RecordDecl *astruct, const char *ap, const char *nam, chillAST_NodeList arraypart):chillAST_VarDecl(astruct->getName(), ap, nam, arraypart, NULL) {
  vardef  = astruct;// pointer to the thing that says what is inside the struct
};

chillAST_VarDecl::chillAST_VarDecl( chillAST_TypedefDecl *tdd, const char *ap, const char *n, chillAST_NodeList arraypart):chillAST_VarDecl(tdd->getStructName(), ap, n, arraypart, NULL) {
  typedefinition = tdd;
  isStruct = tdd->isAStruct();
};

chillAST_VarDecl::chillAST_VarDecl( const char *t, const char *ap, const char *n, chillAST_NodeList arraypart,  void *ptr):chillAST_VarDecl() {
  vartype   = strdup(t);
  underlyingtype = parseUnderlyingType( vartype );
  varname   = strdup(n);
  uniquePtr = ptr;
  arraypointerpart = strdup(ap);

  for (auto i = arraypart.begin(); i!= arraypart.end(); ++i)
    addChild(*i);

  // This should not have any array part as parsed by the front end
  numdimensions = arraypart.size();
  const char * p = ap;
  while (*p) {
    if (*p=='*')
      numdimensions++;
    ++p;
  }
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


void chillAST_CompoundStmt::replaceChild( chillAST_node *old, chillAST_node *newchild ){

    for (int i=0; i<this->children.size(); i++) {
        if(this->getChild(i) == old) {
            this->setChild(i, newchild);
        }
        else {
            this->getChild(i)->replaceChild(old, newchild);
        }
    }
}


void chillAST_CompoundStmt::loseLoopWithLoopVar( char *var ) {
  for (int i=0; i<children.size(); i++) {
    children[i]->loseLoopWithLoopVar( var );
  }
}



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





chillAST_ParenExpr::chillAST_ParenExpr(  chillAST_node *sub ):subexpr(this, 0){
  subexpr = sub;
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
  chillAST_ParenExpr *PE = new chillAST_ParenExpr( subexpr );
  PE->isFromSourceFile = isFromSourceFile; 
  if (filename) PE->filename = strdup(filename); 
  return PE; 
}

void chillAST_ParenExpr::replaceVarDecls( chillAST_VarDecl *olddecl, chillAST_VarDecl *newdecl){
  subexpr->replaceVarDecls( olddecl, newdecl ); 
}

chillAST_Sizeof::chillAST_Sizeof( char *athing ){
  thing = strdup( athing ); // memory leak
}

void chillAST_Sizeof::gatherArrayRefs( std::vector<chillAST_ArraySubscriptExpr*> &refs, bool writtento ) {} 
void chillAST_Sizeof::gatherScalarRefs( std::vector<chillAST_DeclRefExpr*> &refs, bool writtento ) {}

chillAST_node* chillAST_Sizeof::constantFold() {
  return this; 
}

chillAST_node* chillAST_Sizeof::clone() {
  chillAST_Sizeof *SO = new chillAST_Sizeof( thing );
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

chillAST_IfStmt::chillAST_IfStmt(chillAST_node *c, chillAST_node *t, chillAST_node *e ):chillAST_IfStmt(){
  cond = c;
  thenpart = t;
  elsepart = e;
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

void chillAST_IfStmt::loseLoopWithLoopVar(char* var) {
    thenpart->loseLoopWithLoopVar(var);
    if(elsepart) {
        elsepart->loseLoopWithLoopVar(var);
    }
}


chillAST_node *chillAST_IfStmt::clone() { 
  chillAST_node *c, *t, *e; 
  c = t = e = NULL; 
  if (cond) c = cond->clone(); // has to be one, right? 
  if (thenpart) t = thenpart->clone();
  if (elsepart) e = elsepart->clone();

  chillAST_IfStmt *IS = new chillAST_IfStmt( c, t, e );
  IS->isFromSourceFile = isFromSourceFile;
  if (filename) IS->filename = strdup(filename); 
  return IS;
} 


bool chillAST_IfStmt::findLoopIndexesToReplace(  chillAST_SymbolTable *symtab, bool forcesync ) { 
  thenpart->findLoopIndexesToReplace( symtab );
  if(elsepart) {
      elsepart->findLoopIndexesToReplace( symtab );
  }
  return false; // ?? 
}

  

chillAST_node *minmaxTernary(const char * op, chillAST_node *left, chillAST_node *right) {

  chillAST_node *lp1 = left -> clone();
  chillAST_node *rp1 = right -> clone();
  chillAST_BinaryOperator *cond = new chillAST_BinaryOperator( lp1, op, rp1 );
  chillAST_node *lp2 = left -> clone();
  chillAST_node *rp2 = right -> clone();
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

  if ( functions.size() == 0 )
    throw std::runtime_error(std::string("could not find function named ") + procname);

  if ( functions.size() > 1 )
    throw std::runtime_error(std::string("Multiple function named ") + procname);

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

