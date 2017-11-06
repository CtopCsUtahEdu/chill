/*****************************************************************************
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
 CHiLL's rose interface.

 Notes:
 Array supports mixed pointer and array type in a single declaration.

 History:
 02/23/2009 Created by Chun Chen.
*****************************************************************************/

#include "parser/rose.h"
#include <rose.h>
#include <sageBuilder.h>
#include <AstInterface_ROSE.h>
#include "scanner/definitionLinker.h"
#include "scanner/sanityCheck.h"

using namespace SageBuilder;
using namespace SageInterface;
using namespace chill;

// forward declarations
chillAST_node * ConvertRoseFile(  SgGlobal *sg, const char *filename ); // the entire file
chillAST_node * ConvertRoseFunctionDecl( SgFunctionDeclaration *D );
chillAST_node * ConvertRoseParamVarDecl( SgInitializedName *vardecl );
chillAST_node * ConvertRoseFunctionRefExp( SgFunctionRefExp *FRE );
chillAST_node * ConvertRoseInitName( SgInitializedName *vardecl );
chillAST_node * ConvertRoseVarDecl( SgVariableDeclaration *vardecl ); // stupid name TODO
chillAST_node * ConvertRoseForStatement( SgForStatement *forstatement );
chillAST_node * ConvertRoseWhileStatement( SgWhileStmt *whilestmt );
chillAST_node * ConvertRoseExprStatement( SgExprStatement *exprstatement );
chillAST_node * ConvertRoseBinaryOp( SgBinaryOp *binaryop );
chillAST_node * ConvertRoseMemberExpr( SgBinaryOp *binaryop); // binop! a.b
chillAST_node * ConvertRoseArrowExp  ( SgBinaryOp *binaryop); // binop! a->b
char *          ConvertRoseMember( SgVarRefExp* memb ); // TODO
chillAST_node * ConvertRoseUnaryOp( SgUnaryOp *unaryop );
chillAST_node * ConvertRoseVarRefExp( SgVarRefExp *varrefexp );
chillAST_node * ConvertRoseIntVal( SgIntVal *riseintval );
chillAST_node * ConvertRoseFloatVal( SgFloatVal *rosefloatval );
chillAST_node * ConvertRoseDoubleVal( SgDoubleVal *rosecdoubleval );
chillAST_node * ConvertRoseBasicBlock( SgBasicBlock *bb );
chillAST_node * ConvertRoseFunctionCallExp( SgFunctionCallExp* );
chillAST_node * ConvertRoseReturnStmt( SgReturnStmt *rs );
chillAST_node * ConvertRoseArrayRefExp( SgPntrArrRefExp *roseARE );
chillAST_node * ConvertRoseCastExp( SgCastExp *roseCE );
chillAST_node * ConvertRoseAssignInitializer( SgAssignInitializer *roseAI );
// TODO
chillAST_node * ConvertRoseStructDefinition( SgClassDefinition *def );
chillAST_node * ConvertRoseStructDeclaration( SgClassDeclaration *dec );


chillAST_node * ConvertRoseIfStmt( SgIfStmt *ifstatement );

chillAST_node * ConvertRoseTypeDefDecl( SgTypedefDeclaration *TDD );

chillAST_node * ConvertRoseGenericAST( SgNode *n );

// a global variable. it's OK.
SgProject *OneAndOnlySageProject;  // a global

// more globals. These may be dumb. TODO

//vector< chillAST_  >  // named struct types ??



// TODO move to ir_rose.hh
SgNode *toplevel;



// temp
void die() { debug_fprintf(stderr, "\ndie()\n"); int *i=0; int j=i[0]; }



void ConvertRosePreprocessing(  SgNode *sg, chillAST_node *n ) { // add preprocessing, attached to sg, to n
  SgLocatedNode *locatedNode = isSgLocatedNode (sg);
  if (locatedNode != NULL) {

    Sg_File_Info *FI = locatedNode->get_file_info();
    std::string nodefile; // file this node came from
    //nodefile =  FI->get_raw_filename();
    nodefile =  FI->get_filenameString () ;

    if (!strstr( nodefile.c_str(), "rose_edg_required_macros")) {  // ignore this file
      //debug_fprintf(stderr, "\nfound a located node from raw file %s  line %d   col %d\n",
      //        nodefile.c_str(), FI->get_line(), FI->get_col());
    }

    AttachedPreprocessingInfoType *comments = locatedNode->getAttachedPreprocessingInfo ();

    if (comments != NULL) {
      debug_fprintf(stderr, "\nhey! comments! on a %s\n", n->getTypeString());

      AttachedPreprocessingInfoType::iterator i;
      int counter = 0;
      for (i = comments->begin (); i != comments->end (); i++) counter++;
      debug_fprintf(stderr, "%d preprocessing info\n\n", counter);

      counter = 0;
      for (i = comments->begin (); i != comments->end (); i++){
        // this logic seems REALLY WRONG
        CHILL_PREPROCESSING_POSITION p = CHILL_PREPROCESSING_POSITIONUNKNOWN;
        if ((*i)->getRelativePosition () == PreprocessingInfo::inside) {
          // ??? t =
        }
        else if ((*i)->getRelativePosition () == PreprocessingInfo::before) {
          p =  CHILL_PREPROCESSING_LINEBEFORE;
        }
        else if ((*i)->getRelativePosition () == PreprocessingInfo::after) {
          p =  CHILL_PREPROCESSING_TOTHERIGHT;
        }

        char *blurb = strdup( (*i)->getString().c_str() );
        chillAST_node *pre;

        string preproctype = string( PreprocessingInfo::directiveTypeName((*i)->getTypeOfDirective ()). c_str () );

        char *thing = strdup( (*i)->getString ().c_str ());
        if (preproctype == "CplusplusStyleComment") {
          debug_fprintf(stderr, "a comment  %s\n", thing);
          chillAST_Preprocessing *pp = new chillAST_Preprocessing( p, CHILL_PREPROCESSING_COMMENT, blurb);
          n-> preprocessinginfo.push_back( pp);
        }
        if (preproctype == "CpreprocessorDefineDeclaration") {
          debug_fprintf(stderr, "a #define %s\n", thing);
          chillAST_Preprocessing *pp = new chillAST_Preprocessing( p, CHILL_PREPROCESSING_POUNDDEFINE, blurb);
          n-> preprocessinginfo.push_back( pp);
        }
        if (preproctype == "CpreprocessorIncludeDeclaration") {
          debug_fprintf(stderr, "a #include %s\n", thing);
          chillAST_Preprocessing *pp = new chillAST_Preprocessing( p, CHILL_PREPROCESSING_POUNDINCLUDE, blurb);
          n-> preprocessinginfo.push_back( pp);
        }


      } // for each comment attached to a node
    } // comments != NULL
  } // located node
} // ConvertRosePreprocessing



chillAST_node * ConvertRoseFunctionRefExp( SgFunctionRefExp *FRE ) {
  SgFunctionDeclaration *fdecl = FRE->getAssociatedFunctionDeclaration();
  SgFunctionDeclaration *defining_fdecl = (SgFunctionDeclaration *) fdecl->get_definingDeclaration(); // this fails for builtins?


  if (defining_fdecl) {
    fdecl = defining_fdecl;
    debug_fprintf(stderr, "symbol->get_definingDeclaration() %p  %s\n", fdecl, fdecl->get_name().str() );
  }
  else {
    debug_fprintf(stderr, "symbol->get_definingDeclaration() NULL  a builtin?\n");
  }

  const char *name = strdup(fdecl->get_name().str());

  // fdecl should match the uniquePtr for some function definition we've seen already (todo builtins?)
  chillAST_FunctionDecl *chillfd = NULL;
  int numfuncs = FunctionDeclarations.size();
  debug_fprintf(stderr, "there are %d functions to compare to\n", numfuncs);
  for (int i=0; i<numfuncs; i++) {
    SgFunctionDeclaration *fd = (SgFunctionDeclaration *) FunctionDeclarations[i]->uniquePtr;

    debug_fprintf(stderr, "func %2d unique %p %s  vs fdecl %p %s\n", i, fd, fd->get_name().str(), fdecl, name );
    if (fd == fdecl) {
      chillfd = FunctionDeclarations[i];
      debug_fprintf(stderr, "found it at functiondeclaration %d of %d\n", i, numfuncs);
    }
    //else  { // temp compare names until I can figure out why the nodes are not the same
    //  if (!strcmp( fd->get_name().str(), name )) {
    //    debug_fprintf(stderr, "\nWARNING: HACK TO FIND FUNCTIONDECL TRIGGERED. name matched but not node address\n\n");
    //    chillfd = FunctionDeclarations[i];
    //  }
    //}
  }
  if (chillfd == NULL) { debug_fprintf(stderr, "couldn't find function definition in the locally defined list of functions\n"); exit(-1); }

  // make a DeclRefExpr from the function definition
  chillAST_DeclRefExpr *DRE = new  chillAST_DeclRefExpr( chillfd );

  return DRE;
}


chillAST_node * ConvertRoseFile(  SgNode *sg, const char *filename )// the entire file
{
  debug_fprintf(stderr, "ConvertRoseFile(  SgGlobal *sg, filename %s );\n", filename);

  chillAST_SourceFile * topnode = new chillAST_SourceFile( filename  );  // empty

  std::string sourcefile = "";

  ConvertRosePreprocessing( sg, topnode );  // handle preprocessing attached to the rose file node


  SgLocatedNode *locatedNode = isSgLocatedNode (sg);
  if (locatedNode != NULL) {
    debug_fprintf(stderr, "TOPMOST located node\n");
    Sg_File_Info *FI = locatedNode->get_file_info();
    sourcefile =  FI->get_raw_filename();
    sourcefile =  FI->get_filename();
    debug_fprintf(stderr, "sourcefile is '%s'\n", sourcefile.c_str());
  }



  topnode->setFrontend("rose");
  topnode->chill_array_counter  = 1;
  topnode->chill_scalar_counter = 0;

  std::vector<std::pair< SgNode *, std::string > > topnodes = sg->returnDataMemberPointers();
  int numtop = topnodes.size();
  debug_fprintf(stderr, "%d top nodes\n", topnodes.size());
  for (int i=0; i<numtop; i++) {
    SgNode *n    = topnodes[i].first;
    string blurb = topnodes[i].second;

    // we want to ignore all the builtins
    //debug_fprintf(stderr, "%3d/%d   %p   %s    ", i, numtop, n, blurb.c_str());
    //debug_fprintf(stderr, "node %s\n", n->class_name().c_str());

    std::string nodefile = ""; // file this node came from
    locatedNode = NULL;

    if (n && (locatedNode = isSgLocatedNode (n))) { // purposeful assignment not equality check
      Sg_File_Info *FI = locatedNode->get_file_info();
      nodefile =  FI->get_filenameString () ;
      if (!strstr( nodefile.c_str(), "rose_edg_required_macros")) {  // denugging, but ignore this file
        //if (nodefile != sourcefile) {
        //debug_fprintf(stderr, "\nfound a located node from raw file %s  line %d   col %d\n",
        //        nodefile.c_str(), FI->get_line(), FI->get_col());
      }
    }

    // not sure what these are; ignoring except for debugging info
    if ( n == NULL ) {
      debug_fprintf(stderr, "topnode %d of %d, first == NULL??  blurb %s\n", i, numtop, blurb.c_str());
    }
    else {  // n exists. deal with each type of thing

      if ( isSg_File_Info(n) ) {
        Sg_File_Info *FI = (Sg_File_Info *) n;

        debug_fprintf(stderr, "top node %d/%d    Sg_File_Info\n", i, numtop);
        string fname = FI->get_filenameString () ;
        debug_fprintf(stderr, "file %s  line %d   col %d\n", fname.c_str(), FI->get_line(), FI->get_col());
      }
      else if ( isSgSymbolTable(n) ) {
        SgSymbolTable *ST = (SgSymbolTable *) n;
        debug_fprintf(stderr, "top node %d/%d    SgSymbolTable  (IGNORING)\n", i, numtop);
        //ST->print(); fflush(stdout);
      }
      else if ( isSgFunctionDeclaration(n) ) {
        //debug_fprintf(stderr, "it's a function DECLARATION\n");
        SgFunctionDeclaration *fd = (SgFunctionDeclaration *)n;
        const char *name = strdup( fd->get_name().str()) ;
        //const char *name = fd->get_name().str() ;
        //debug_fprintf(stderr, "name %p\n", name );

        if (strncmp("__builtin", name, 9) &&//if name DOESN'T start with __builtin
            strcmp("__sync_lock_test_and_set", name) &&
            strcmp("__sync_lock_release", name)
          )   // ignore builtins.  I can't find a better way to test
        {
          debug_fprintf(stderr, "\nfunctiondecl                     %s blurb %s\n", name, blurb.c_str());
          bool samefile =  (nodefile == sourcefile);
          debug_fprintf(stderr, "nodefile   %s\nsourcefile %s\n", nodefile.c_str(), sourcefile.c_str());
          if (samefile) debug_fprintf(stderr, "SAME FILE\n");
          else  debug_fprintf(stderr, "NOT THE SAME FILE\n");

          //debug_fprintf(stderr, "\n\n%3d   %p   %s    ", i, n, blurb.c_str());
          //debug_fprintf(stderr, "node %s\n", n->class_name().c_str());
          //debug_fprintf(stderr, "adding function decl %s because it is not a builtin\n", name);
          //debug_fprintf(stderr, "topnode has %d children\n", topnode->getNumChildren());

          chillAST_node *node =  ConvertRoseFunctionDecl(fd);
          topnode->addChild(node);
          node ->isFromSourceFile = samefile;
          node->filename = strdup(nodefile.c_str());

          //debug_fprintf(stderr, "ir_rose.cc adding function %s as child of topnode\n\n", name);
          //topnode->addChild( node ); // this is done in the convert
          //debug_fprintf(stderr, "topnode now has %d children\n", topnode->getNumChildren());
        }
        else {
          //debug_fprintf(stderr, "ignoring %s\n", name);
        }
      }
      else if (isSgVariableDeclaration(n)) {
        debug_fprintf(stderr, "\n\nA TOP LEVEL GLOBAL VARIABLE\n");
        debug_fprintf(stderr, "\n%3d   %p   %s  \n", i, n, blurb.c_str());
        //debug_fprintf(stderr, "node %s\n", n->class_name().c_str());
        SgVariableDeclaration *rosevd = (SgVariableDeclaration *)n;
        chillAST_node *vd = ConvertRoseVarDecl( rosevd );

        vd->isFromSourceFile = (nodefile == sourcefile);
        vd->filename = strdup(nodefile.c_str());
        debug_fprintf(stderr, "global "); vd->print(); printf("\n"); fflush(stdout);
        if (vd->parent) {
          debug_fprintf(stderr, "global has parent of type %s\n", vd->parent->getTypeString());
          if (vd->parent->isSourceFile()) { // ?? other cases  TODO
            debug_fprintf(stderr, "adding global variable as child of sourcefile\n");
            vd->parent->addChild(vd); // seems wrong

          }
        }
        else debug_fprintf(stderr, "global has parent no parent\n");
        topnode->addChild( vd );
      }

      else if (isSgClassDeclaration(n)) {
        debug_fprintf(stderr, "\n\nA TOP LEVEL CLASS OR STRUCT DEFINITION\n");
        debug_fprintf(stderr, "\n%3d   %p   %s  \n", i, n, blurb.c_str());
        //debug_fprintf(stderr, "node %s\n", n->class_name().c_str());

        SgClassDeclaration *SD = (SgClassDeclaration *)n;
        SgClassDeclaration::class_types class_type = SD->get_class_type();
        if (class_type == SgClassDeclaration::e_struct) {
          //debug_fprintf(stderr, "a struct\n");

          // what we really want is the class DEFINTION, not the DECLARATION
          SgClassDefinition *def = SD->get_definition();
          if (def == NULL) continue;
          chillAST_node *structdef = ConvertRoseStructDefinition( def ); // really a recorddecl
          structdef->isFromSourceFile = (nodefile == sourcefile);
          structdef->filename = strdup(nodefile.c_str());
          //debug_fprintf(stderr, "struct is %p\n", structdef);

          chillAST_RecordDecl *RD = ( chillAST_RecordDecl *) structdef;

          // figure out what file the struct decl is in

          SgLocatedNode *locatedNode = isSgLocatedNode (SD);
          if (locatedNode != NULL) {
            debug_fprintf(stderr, "know the location of the struct definition\n");
            Sg_File_Info *FI = locatedNode->get_file_info();
            std::string nodefile =  FI->get_filename();
            debug_fprintf(stderr, "nodefile is '%s'\n", nodefile.c_str());

            if (nodefile == topnode->SourceFileName) {

              debug_fprintf(stderr, "adding struct DEFINITION %s to children of topnode\n", RD->getName());
              topnode->addChild( structdef );  // adds the STRUCT but not the individual members
              // TODO this should only happen if the class declaration is in the same file as topnode (a source file)
              // this is done above debug_fprintf(stderr, "*** we need to add struct definition to globals ***\n");
            }
          }
          else debug_fprintf(stderr, "DON'T know the location of the struct definition\n");
        }
        else
          debug_fprintf(stderr, "unhandled top node SgClassDeclaration that is not a struct!\n");

      }
      //else {
      //   if (strncmp("__builtin", name, 9)) debug_fprintf(stderr, "decl %s is a forward declaration or a builtin\n", name );
      //}
      //}
      else if (isSgTypedefDeclaration(n)) {  // sometimes structs are this

        debug_fprintf(stderr, "\nTYPEDEF\n\n");
        debug_fprintf(stderr, "\n\n%3d   %p   %s    ", i, n, blurb.c_str());
        debug_fprintf(stderr, "node %s\n", n->class_name().c_str());

        //debug_fprintf(stderr, "\nsometimes structs are this calling  ConvertRoseTypeDefDecl\n");
        SgTypedefDeclaration *TDD = (SgTypedefDeclaration *) n;
        chillAST_TypedefDecl *td = (chillAST_TypedefDecl *)ConvertRoseTypeDefDecl( TDD );
        td->isFromSourceFile = (nodefile == sourcefile);
        td->filename = strdup(nodefile.c_str());

        topnode->addChild( td );
        topnode->addTypedefToTypedefTable( td );
        //debug_fprintf(stderr, "done with SgTypedefDeclaration\n");
      }
      else {
        //debug_fprintf(stderr, "\n\n%3d   %p   %s    ", i, n, blurb.c_str());
        //debug_fprintf(stderr, "node %s\n", n->class_name().c_str());
        debug_fprintf(stderr, "unhandled top node %d/%d of type %s\n", i, numtop, n->class_name().c_str());
      }
    } // non-NULL node
  } // for each top level node


  return topnode;
}




// shorten really ugly unnamed struct names, or just strdup to turn const char * to char *
char *shortenRoseUnnamedName( const char *origname ) {  // usually a TYPE not a name ...

  // this used to be something like "
  if (origname == NULL) return NULL; // ??

  int l = strlen(origname);
  //debug_fprintf(stderr, "\nshortenRoseUnnamedName( origname %d characters ) origname was '%s'\n", l, origname);

  if ( l > 25 ) {
    //debug_fprintf(stderr, "long (variable type) name is '%s'\n", origname );
    if ( (!strncmp(        "__unnamed_class", origname, 15)) ||
         (!strncmp( "struct __unnamed_class", origname, 22)) ||
         (!strncmp( "__anonymous_", origname, 13))) {
      //debug_fprintf(stderr, "an unnamed struct with %d characters in the name!\n", strlen(origname));

      // this was the beginning of dealing with unnamed struct names inside unnamed structs, but that seems not to be needed
      //string o( origname );
      //std::size_t  first = o.find( string("__unnamed_class"));
      //debug_fprintf(stderr, "first %d\n", first);
      //
      //string rest = o.substr( first + 5 );
      //debug_fprintf(stderr, "rest %s\n", rest.c_str());
      //std::size_t next = rest.find( string("__unnamed_class"));
      //debug_fprintf(stderr, "next %d\n", next);


      bool startsWithStruct = 0 == strncmp( "struct ", origname, 7);

      string buh( origname );
      string underlinel( "_L" );
      std::size_t found = buh.find( underlinel );
      if (found!=std::string::npos) {
        //debug_fprintf(stderr, "it has _L at %d\n", found);
        int linenumber;
        sscanf( &origname[2 + found], "%d", &linenumber );
        //debug_fprintf(stderr, "line number %d\n", linenumber);
        char newname[128];
        if (startsWithStruct)
          sprintf(newname, "struct unnamedStructAtLine%d\0", linenumber);
        else
          sprintf(newname, "unnamedStructAtLine%d\0", linenumber);
        char *shortname = strdup(newname);
        //debug_fprintf(stderr, "shortened name is %s\n\n", shortname);
        return shortname;
      }
    }
  }
  //debug_fprintf(stderr, "unable to shorten '%s'\n", origname);
  return strdup(origname); // unable to shorten but still have to copy
}





char * shortenRoseStructMemberName( const char *oldname ) {
  char *temp = strdup(oldname);
  //debug_fprintf(stderr, "shortenRoseStructMemberName( '%s' )\n", oldname);
  if (rindex(oldname, ':')) {
    int i = rindex(oldname, ':') - oldname;
    //debug_fprintf(stderr, "last part i=%d   '%s'\n", i, &(temp[i]));
    if (oldname[i-1] == ':') {
      char *shorter = strdup(&oldname[i+1]); // starting after ::
      free(temp);
      return shorter;
    }
  }

  return temp;
}


chillAST_node * ConvertRoseFunctionDecl( SgFunctionDeclaration *fd )
{
  const char *functionname = strdup( fd->get_name().str());
  debug_fprintf(stderr, "\nConvertRoseFunctionDecl( %s )\n", functionname);


  // need return type
  SgType *rt = fd->get_orig_return_type();
  string temp = rt->unparseToString(); // so it stays in scope !!
  const char *returntype = temp.c_str();

  chillAST_FunctionDecl *chillFD = new chillAST_FunctionDecl( returntype,  functionname, (void *)fd /* unique */ );
  ConvertRosePreprocessing( fd, chillFD);  // before doing the function decl itself?

  // add parameters
  std::vector<SgInitializedName*> args = fd->get_args();
  int numargs =  args.size();
  for (int i=0; i<numargs; i++) {
    chillAST_VarDecl *chillPVD = (chillAST_VarDecl *)ConvertRoseParamVarDecl( args[i] );
    chillFD->addParameter(chillPVD);
    // already done inside ConvertRoseParamVarDecl   VariableDeclarations.push_back(chillPVD);  // global?
  }

  // add body IF THERE IS ONE
  SgFunctionDefinition *funcdef = fd->get_definition();
  if (funcdef)  {
    SgBasicBlock *bodybasicblock = funcdef->get_body();
    debug_fprintf(stderr, "got body\n");

    std::vector<SgStatement* > statements = bodybasicblock->get_statements();
    int num_statements = statements.size();
    debug_fprintf(stderr, "%d statements in FunctionDecl body\n", num_statements);

    // create a compound statement for the function body, to hold the rest of the statements
    chillAST_CompoundStmt *chillCS = new chillAST_CompoundStmt;
    chillCS->setParent( chillFD );
    debug_fprintf(stderr, "chillCS is %p\n", chillCS);

    for (int i=0; i<num_statements; i++) {
      SgStatement* statement = statements[i];
      debug_fprintf(stderr, "\nstatement %d     %s\n", i, statement->unparseToString().c_str());
      //debug_fprintf(stderr,"calling ConvertRoseGenericAST with parent %p\n",chillCS);
      debug_fprintf(stderr, "calling ConvertRoseGenericAST\n", chillCS);
      chillAST_node *n =  ConvertRoseGenericAST( statement );
      if (n) {
        chillCS->addChild( n );
      }

    }
    chillFD->setBody ( chillCS );
  }
  else {
    //debug_fprintf(stderr, "function %s is a forward declaration or external\n", functionname);

    if(fd->get_externBrace()) {
      chillFD->setExtern();
    }
    chillFD->setForward();
  }

  FunctionDeclarations.push_back(chillFD);

  return chillFD;
}

// todo initname for vardecl ???
chillAST_node * ConvertRoseParamVarDecl( SgInitializedName *vardecl )
{
  //debug_fprintf(stderr, "ConvertRoseParamVarDecl()   ");
  chillAST_VarDecl *chillVD = (chillAST_VarDecl *) ConvertRoseInitName( vardecl );
  chillVD->isAParameter = true;
  debug_fprintf(stderr, "new parameter:\n");
  //chillVD->dump(); printf("\n"); fflush(stdout); // dump in ConvertRoseInitName

  return chillVD;
}


chillAST_NodeList ConvertSgArrayType( SgType* Typ ) {
  chillAST_NodeList ap;


  return ap;
}

chillAST_node * ConvertRoseInitName( SgInitializedName *initname ) // TODO probably wrong
{
  debug_fprintf(stderr, "\n\n***  ConvertRoseInitName()  %s\n", initname->unparseToString().c_str());
  debug_fprintf(stderr, "initname %s\n", initname->unparseToString().c_str());

  int numattr = initname->numberOfAttributes();


  char *varname = shortenRoseStructMemberName( initname->unparseToString().c_str() );
  debug_fprintf(stderr, "varname '%s'\n", varname);

  SgType *typ = initname->get_type();
  // !! if typ->unparseToString()->c_str(), the string and therefore the pointer to char are freed before the next statement !
  chillAST_NodeList arr;

  while (isSgArrayType(typ) || isSgPointerType(typ)) {
    if (isSgArrayType(typ)) {
      SgArrayType *AT = (SgArrayType *) typ;
      SgExpression *indexExp = AT->get_index();
      arr.push_back(ConvertRoseGenericAST(indexExp));
      typ = AT->get_base_type();
    } else {
      SgPointerType *AT = (SgPointerType *) typ;
      arr.push_back(new chillAST_NULL());
      typ = AT->get_base_type();
    }
  }
  bool isRef = false;
  if (isSgReferenceType(typ)) {
    isRef = true;
    typ = ((SgReferenceType*)typ)->get_base_type();
  }

  string really = typ->unparseToString();
  const char *otype =   really.c_str();
  debug_fprintf(stderr, "original vartype 0x%x '%s'\n", otype, otype);

  bool restricted = isRestrict( otype );

  char *vartype = parseUnderlyingType(restricthack( shortenRoseUnnamedName( otype )));
  char *arraypart;
  arraypart =  parseArrayParts( strdup(otype) );
  debug_fprintf(stderr, "HACK vartype %s arraypart %s\n", vartype, arraypart);

  // need underlying type to pass to constructor?  double and arraypart **, not type double ** and arraypart **

  SgDeclarationStatement *dec = initname->get_declaration();
  SgDeclarationStatement *defdec = dec->get_definingDeclaration();


  // figure out if this is some non-standard typedef'd type
  char *bracket = index(vartype, '{');
  if (bracket) {   // remove extra for structs
    *bracket = '\0';
    if (*(bracket-1) == ' ')  *(bracket-1) = '\0';
  }

  chillAST_VarDecl * chillVD = new chillAST_VarDecl( vartype, arraypart,  varname, arr,(void *)initname );
  if ( !strncmp(vartype, "struct ", 7) ) {
    chillVD->setStruct(true);
  }

  chillVD->isRestrict = restricted; // TODO nicer way
  chillVD->uniquePtr = defdec;
  chillVD->byreference = isRef;

  debug_fprintf(stderr, "ConvertRoseInitName()  storing variable declaration '%s' with unique value %p\n", varname,  chillVD->uniquePtr );
  // store this away for declrefexpr that references it!
  VariableDeclarations.push_back(chillVD);
  //debug_fprintf(stderr, "ConvertRoseInitName() END\n");

  // check for an initializer    int i = 0;
  SgInitializer * initptr = initname->get_initptr();
  if (initptr) {
    debug_fprintf(stderr, "%s gets initialized\n", chillVD->varname);
    chillAST_node *init = ConvertRoseGenericAST( initptr );  // NULL);
    chillVD->setInit( init );

  }
  //chillVD->dump(); printf("\n"); fflush(stdout);
  return chillVD;
}



char *fixUnnamedStructType( char *otype, char *entiredecl ) // deal with unnamed struct messiness
{
  char *result = otype; // default is to not change anything

  // see if otype looks like an unnamed struct
  if ( 0 == strncmp(otype, "struct", 6)) { // it's a struct
    // find the first non-space character starting at position 8
    int l = strlen(otype);
    //debug_fprintf(stderr, "%d chars in '%s'\n", l, otype);
    for (int i=6; i<l; i++) {
      char c = otype[i];
      //debug_fprintf(stderr, "char %d is '%c'\n", i, c);
      if (c != ' ') {
        if (c == '{') {
          // first nonblank is open bracket, it's an unnamed struct
          //debug_fprintf(stderr, "it's an unnamed struct!\n");

          //debug_fprintf(stderr, "want to get the type from '%s'\n", entiredecl);
          char *decl = strdup(entiredecl);
          if (strncmp(decl, "struct ", 7)) { // make sure entiredecl looks like "struct something"
            debug_fprintf(stderr, "ir_rose.ccERROR, trying to get name of an unnamed struct from '%s'\n", decl);
            exit(-1);
          }

          char *bettertype = decl;
          char *p = bettertype + 6;

          // handle possible lots of spaces (should never happen)
          //debug_fprintf(stderr, "bettertype '%s'\n", bettertype);
          l = strlen(bettertype);
          for (int j=6; j<l; j++) {
            if (*p == ' ') { // ignore initial spaces after "struct"
              p++;
            }
            else break; // not a space, we should be pointing at the start of the name
          }

          // find the name. end name at first space if any

          l = strlen(p); // how many chars we haven't looked at yet  (off by one?)
          for (int j=0; j<l; j++) {
            if (*p != ' ') { // include non spaces
              p++;
            }
            else {  // a space, end the name
              *p = '\0';
              break;
            }
          }
          debug_fprintf(stderr, "unnamed struct '%s'\n", bettertype);
          result = bettertype;

          break;
        }
        else {
          // first nonblank looks like a struct name - leave the type alone
          break;
        }
      }
    }
  }
  return result;
}




chillAST_node * ConvertRoseVarDecl( SgVariableDeclaration *vardecl )
{
  debug_fprintf(stderr, "\nConvertRoseVarDecl() \n");

  SgDeclarationStatement *defdecl = vardecl->get_definingDeclaration(); // unique
  debug_fprintf(stderr, "defdecl %p\n", defdecl);
  if (defdecl == NULL) defdecl = vardecl;

  std::vector<SgInitializedName* > names = vardecl->get_variables();
  if (1 != names.size()) debug_fprintf(stderr, "%d initialized names\n", names.size());

  char *entiredecl = strdup( vardecl->unparseToString().c_str());

  if ( names.size() > 1 ) {
    debug_fprintf(stderr, "ConvertRoseVarDecl()  %s\n", entiredecl);
    debug_fprintf(stderr, "too many decls in a decl!\n");
    exit(-1);
  }

  // first, get the type. this may be a really ugly thing for an unnamed struct (not with more recent rose)
  SgInitializedName* iname =  names[0];
  chillAST_VarDecl* vd = (chillAST_VarDecl*)ConvertRoseInitName(iname);

  // this if handles structs  DEFINITIONS (and typedefs?)
  // things like
  //  struct { int i} b;   // this is an unnamed struct. the vardecl for b will have a BaseTypeDefiningDeclaration
  //
  //  struct a { int i; } b;  // this will ALSO have a BaseTypeDefiningDeclaration, for struct a

  chillAST_RecordDecl *RD = NULL;
  SgDeclarationStatement *defining = NULL;

  if (vardecl->get_variableDeclarationContainsBaseTypeDefiningDeclaration()) {
    //struct {
    //  struct { a,b,c} d,e;
    //}
    // d willhave a defining decl.  e will not


    debug_fprintf(stderr, "in ConvertRoseVarDecl(), there is a defining declaration  (a struct or typedef?)\n");
    SgDeclarationStatement *DS = vardecl->get_baseTypeDefiningDeclaration();
    //debug_fprintf(stderr, "DS type %s\n", roseGlobalVariantNameList[DS->variantT()]);

    if (SgClassDeclaration *CD = isSgClassDeclaration(DS)) {
      debug_fprintf(stderr, "it's a ClassDeclaration\n");
      SgClassDeclaration::class_types class_type = CD->get_class_type();
      if (class_type == SgClassDeclaration::e_struct) {
        debug_fprintf(stderr, "it's a ClassDeclaration of a struct\n");
        // RD should be the RecordDecl that says what's in the struct
        RD = (chillAST_RecordDecl *) ConvertRoseStructDeclaration( CD );
        // do we need to remember this struct type somewhere in case there are more of them?

      }
    }
  }


  if (RD) {
    debug_fprintf(stderr, "we know what the struct definition looks like because it was part of this vardecl\n");

    vd->vardef = RD;
    //parent->addChild( vd ); // ??
    vd->setStruct( true );
    vd->uniquePtr = defdecl; // ??
    debug_fprintf(stderr, "dammit setting %s uniquePtr to %p the SgVariableDeclaration that defined it?\n", vd->varname, vd->uniquePtr);

    debug_fprintf(stderr, "STORING vardecl %s in global VariableDeclarations %d\n", vd->varname, VariableDeclarations.size());
  }

  debug_fprintf(stderr, "ConvertRoseVarDecl() storing variable declaration '%s' with unique value %p from  "
      "SgInitializedName\n", entiredecl,  vd->uniquePtr );


  // store this away for declrefexpr that references it!
  // since we called ConvertRoseInitName() which added it already, don't do that again.
  //VariableDeclarations.push_back(chillVD);
  return vd;
}


chillAST_node * ConvertRoseForStatement( SgForStatement *fs )
{

  debug_fprintf(stderr, "\nConvertRoseForStatement()  parent %p\n");
  debug_fprintf(stderr, "%s\n", fs->unparseToString().c_str());
  std::vector<SgStatement* >inits  = fs->get_init_stmt();  // these 2 seem to be equivalent.

  if (1 < inits.size())
    throw std::runtime_error("ConvertRoseForStatement (ir_rose.cc) more than a single statement in the init, not handled\n");
  SgStatement *roseinit = inits[0];

  SgExpression *rosecond  = fs->get_test_expr();
  SgStatement  *rosecond2 = fs->get_test();

  SgExpression *roseincr  = fs->get_increment();
  SgStatement  *rosebody  = fs->get_loop_body();

  // create the 4 components of a for statement
  chillAST_node *init = ConvertRoseGenericAST( roseinit );
  chillAST_node *cond = ConvertRoseGenericAST( rosecond );
  chillAST_node *incr = ConvertRoseGenericAST( roseincr );
  chillAST_node *body = ConvertRoseGenericAST( rosebody );

  // Body is a compound statements - codeblock
  if (!body->isCompoundStmt()) {
    chillAST_CompoundStmt *cs = new chillAST_CompoundStmt();
    cs->addChild( body );
    body = cs;
  }
  chillAST_ForStmt *chill_loop = new  chillAST_ForStmt( init, cond, incr, body);
  return chill_loop;
}

chillAST_node * ConvertRoseWhileStmt( SgWhileStmt *whilestmt ) {
  SgStatement *rosecond  = whilestmt->get_condition();
  SgStatement *rosebody  = whilestmt->get_body();

  chillAST_node *cond = ConvertRoseGenericAST( rosecond );
  chillAST_node *body = ConvertRoseGenericAST( rosebody );

  if (!body->isCompoundStmt()) {
    chillAST_CompoundStmt *cs = new chillAST_CompoundStmt();
    cs->addChild(body);
    body = cs;
  }
  chillAST_WhileStmt *chill_loop = new chillAST_WhileStmt(cond, body);
  return chill_loop;
}

chillAST_node * ConvertRoseConditionalExp( SgConditionalExp *condexp ) {
  SgExpression *rosecond  = condexp->get_conditional_exp();
  SgExpression *rosetrue  = condexp->get_true_exp();
  SgExpression *rosefalse  = condexp->get_false_exp();

  chillAST_node *cond = ConvertRoseGenericAST( rosecond );
  chillAST_node *trueexpr = ConvertRoseGenericAST( rosetrue );
  chillAST_node *falseexpr = ConvertRoseGenericAST( rosefalse );

  chillAST_TernaryOperator *chill_ternary = new chillAST_TernaryOperator("?", cond, trueexpr, falseexpr);
  return chill_ternary;
}

chillAST_node * ConvertRoseExprStatement( SgExprStatement *exprstatement )
{
  chillAST_node *ret = NULL;
  debug_fprintf(stderr, "ConvertRoseExprStatement() exprstatement %s\n", exprstatement->unparseToString().c_str());
  SgExpression *expr = exprstatement->get_expression();
  //if (isSgExprListExp(expr)) debug_fprintf(stderr, "ExprListExpr\n");
  //if (isSgCommaOpExp(expr))  debug_fprintf(stderr, "commaop expr\n"); // a special kind of Binary op
  if (isSgBinaryOp(expr)) ret = ConvertRoseBinaryOp( (SgBinaryOp *) expr );
  else if ( isSgUnaryOp(expr)     ) ret = ConvertRoseUnaryOp ((SgUnaryOp *)expr );
  else if ( isSgIntVal(expr)     ) ret = ConvertRoseIntVal   ((SgIntVal *)expr );
  else if ( isSgFloatVal(expr)   ) ret = ConvertRoseFloatVal   ((SgFloatVal *)expr );
  else if ( isSgDoubleVal(expr)  ) ret = ConvertRoseDoubleVal ((SgDoubleVal *)expr );
  else if ( isSgFunctionCallExp(expr)  ) ret = ConvertRoseFunctionCallExp ((SgFunctionCallExp *)expr );


  else {
    debug_fprintf(stderr, "SgExprStatement of unhandled type %s\n", expr->class_name().c_str() );

    debug_fprintf(stderr, "%s\n", expr->unparseToString().c_str());
    std::vector<std::pair< SgNode *, std::string > > subnodes = expr->returnDataMemberPointers();
    debug_fprintf(stderr, "%d parts\n", subnodes.size());
    for (int i=0; i<subnodes.size(); i++) {
      SgNode *part =  subnodes[i].first;
      debug_fprintf(stderr, "part %d %p\n", i, part);
      std::string str =  subnodes[i].second;
      if ( part ) {
        debug_fprintf(stderr, "part %d %s\n", i, part->unparseToString().c_str());
        debug_fprintf(stderr, "part %d class %s\n", i,part->class_name().c_str());
      }
      debug_fprintf(stderr, "\n");
    }
    exit(-1);
  }

  return ret;
}

// V_SgNumVariants

const char * binop_string( VariantT typ ) {
  switch (typ) {
  case V_SgAddOp: return "+";
  case V_SgAndOp: return "&&";
    //case V_SgArrowStarOp: op = '';
  case V_SgAssignOp: return "=";
  case V_SgAndAssignOp:   return "&=";  // not in docs ?? &&=
  case V_SgDivAssignOp:  return "/=";  // not in docs
    //case V_SgExponentiationAssignOp:  return "/=";  // not in docs
    //case V_SgIorAssignOp:  return "/=";  // not in docs
    //case V_SgLshiftAssignOp:  return "/=";  // not in docs
  case V_SgMinusAssignOp: return "-=";  // not in docs
  case V_SgModAssignOp: return "%=";  // not in docs
  case V_SgMultAssignOp: return "*=";  // not in docs
  case V_SgPlusAssignOp:  return "+=";  // not in docs
    //case V_SgRshiftAssignOp:  return "/=";  // not in docs
    //case V_SgJavaUnsignedRshiftAssignOp: return ">>=";  // not in docs
    //case V_SgXorAssignOp:  return "+=";  // not in docs


  case V_SgBitAndOp: return "&";
  case V_SgBitOrOp: return "|";
    //case V_SgBitXorOp: return "";
  case V_SgCommaOpExp: return ",";
    //case V_SgCompoundAssignOp: return "";
    //case V_SgConcatenationOp: return "";
  case V_SgDivideOp: return "/";
    //case V_SgDotOp: return "";
    //case V_SgDotStarOp: return "";
  case V_SgEqualityOp: return "==";
  case V_SgExponentiationOp: return "^";
  case V_SgGreaterOrEqualOp: return ">=";
  case V_SgGreaterThanOp: return ">";
    //case V_SgIntegerDivideOp: return "";
    //case V_SgIsNotOp: return "";
    //case V_SgIsOp: return "";
    //case V_SgJavaUnsignedRshiftOp: return "";
  case V_SgLessOrEqualOp: return "<=";
  case V_SgLessThanOp: return "<";
  case V_SgLshiftOp: return "<<";
    //case V_SgMembershipOp: return "";
  case V_SgModOp: return "%";
  case V_SgMultiplyOp: return "*";
    //case V_SgNonMembershipOp: return "";
  case V_SgNotEqualOp: return "!=";
  case V_SgOrOp: return "|";
  case V_SgPntrArrRefExp: return "[]";  // can't really be used except as special case ??
    //case V_SgPointerAssignOp: return "";
  case V_SgRshiftOp: return ">>";
    //case V_SgScopeOp: return "";
  case V_SgSubtractOp: return "-";
    //case V_SgUserDefinedBinaryOp: return "";
  case V_SgDotExp: return ".";
  case V_SgArrowExp: return "->";

  default:
    debug_fprintf(stderr, "unknown Rose BinaryOp string, type %d = %s\n",  typ, roseGlobalVariantNameList[ typ ] );
    int *i = 0;
    int j = i[0]; // segfault
    exit(-1);
  }

}


const char * unaryop_string( VariantT typ ) {
  switch (typ) {
  case V_SgAddressOfOp:  return "&";
    //case V_SgBitComplementOp:  return "";
  case V_SgCastExp:  //debug_fprintf(stderr, "sgcastexp\n");
    return "??";    //                         ??
    //case V_SgConjugateOp:  return "";
    //case V_SgExpressionRootOp:  return "";
    //case V_SgImagPartOp:  return "";
  case V_SgMinusMinusOp:  return "--";
  case V_SgMinusOp:  return "-";
  case V_SgNotOp:  return "!";
  case V_SgPlusPlusOp:  return "++";
  case V_SgPointerDerefExp:  return "*";
    //case V_SgRealPartOp:  return "";
    //case V_SgThrowOp:  return "";
    //case V_SgUnaryAddOp:  return "";
    //case V_SgUserDefinedUnaryOp:  return "";
  case V_SgUnaryAddOp: return "+";

  default:
    debug_fprintf(stderr, "unknown Rose UnaryOp string, type %d = %s\n",  typ, roseGlobalVariantNameList[ typ ] );
    exit(-1);
  }

}






chillAST_node * ConvertRoseBinaryOp( SgBinaryOp *rose_binop )
{
  debug_fprintf(stderr, "\nConvertRoseBinaryOp()\n");
  debug_fprintf(stderr, "%s\n", rose_binop->unparseToString().c_str());

  VariantT typ = rose_binop->variantT();

  const char *op = binop_string( typ );
  debug_fprintf(stderr, "op is %s\n", op );
  debug_fprintf(stderr, "LHS is %s\n", rose_binop->get_lhs_operand_i()->unparseToString().c_str());
  debug_fprintf(stderr, "RHS is %s\n", rose_binop->get_rhs_operand_i()->unparseToString().c_str());

  // special case. rose says member func is a binop
  if ( !strcmp(op, ".") )
    return ConvertRoseMemberExpr( rose_binop );

  // Rose encodes Array Subscript Expression as a binary operator array '[]' index
  // make that a chill ArraySubscriptExpr
  if (isSgPntrArrRefExp(rose_binop)) return ConvertRoseArrayRefExp( (SgPntrArrRefExp *)rose_binop );

  // when this is a . (member) operation, it would be nice for the rhs
  // to know that, so that it could know to look for the reference variable
  // in the struct definitions  TODO

  debug_fprintf(stderr, "ir_rose.cc L1357, making a binop with no LHS no RHS but a parent\n");

  chillAST_node *l = ConvertRoseGenericAST( rose_binop->get_lhs_operand_i() );

  chillAST_node *r = ConvertRoseGenericAST( rose_binop->get_rhs_operand_i());

  chillAST_BinaryOperator *chill_binop = new chillAST_BinaryOperator(l, op, r);

  return chill_binop;
}



chillAST_node * ConvertRoseMemberExpr( SgBinaryOp *rose_binop ) // rose member exp is a binop
{
  debug_fprintf(stderr, "\nConvertXXXXMemberExp()\n");

  VariantT typ = rose_binop->variantT();

  const char *op = binop_string( typ );
  debug_fprintf(stderr, "op is %s\n", op);
  debug_fprintf(stderr, "LHS is %s\n", rose_binop->get_lhs_operand_i()->unparseToString().c_str());
  debug_fprintf(stderr, "RHS is %s\n", rose_binop->get_rhs_operand_i()->unparseToString().c_str());

  if ( strcmp(op, ".") )
    throw std::runtime_error("Member expression is NOT a binop with dot as the operation?\n");

  chillAST_node *base   = ConvertRoseGenericAST( rose_binop->get_lhs_operand_i() );
  char *member = ConvertRoseMember( (SgVarRefExp*)(rose_binop->get_rhs_operand_i()) );

  return new chillAST_MemberExpr( base, member, NULL );
}



chillAST_node * ConvertRoseArrowExp( SgBinaryOp *rose_binop ) // rose arrow (member) exp is a binop
{
  VariantT typ = rose_binop->variantT();
  debug_fprintf(stderr, "ConvertRoseMemberExp()  AST Node is %d %s\n", typ, roseGlobalVariantNameList[ typ ] );

  const char *op = binop_string( typ );
  // special case. rose says member func is a binop
  if ( strcmp(op, "->") )
    throw std::runtime_error("this member expression is NOT a binop with arrow as the operation?\n");

  typ = rose_binop->get_rhs_operand_i()->variantT();
  if (strcmp( "SgVarRefExp", roseGlobalVariantNameList[ typ ]))
    throw std::runtime_error("rhs of binop arrow expression does not seem right\n");

  chillAST_node *base = ConvertRoseGenericAST( rose_binop->get_lhs_operand_i());
  char * member = ConvertRoseMember( (SgVarRefExp*)(rose_binop->get_rhs_operand_i()));

  return new chillAST_MemberExpr( base, member, NULL, CHILL_MEMBER_EXP_ARROW);
}





char * ConvertRoseMember( SgVarRefExp* memb ) // the member itself
{
  char *member = strdup(memb->unparseToString().c_str());
  return member;
}








chillAST_node * ConvertRoseUnaryOp( SgUnaryOp *rose_unaryop )
{
  VariantT typ = rose_unaryop->variantT();

  if (isSgCastExp(rose_unaryop)) return ConvertRoseCastExp( (SgCastExp *)rose_unaryop );

  const char *op = unaryop_string( typ );
  // prefix/postfix
  // rose docs say there is no "unknown"
  // SgUnaryOp::Sgop_mode   SgUnaryOp::prefix  SgUnaryOp::postfix
  bool pre = (SgUnaryOp::prefix == rose_unaryop->get_mode());
  chillAST_node *sub = ConvertRoseGenericAST( rose_unaryop->get_operand() );

  return new chillAST_UnaryOperator( op, pre, sub );
}



chillAST_node * ConvertRoseVarRefExp( SgVarRefExp *rose_varrefexp )
{
  // this is equivalent to chill declrefexpr ??  but we always know it's a variable
  char *varname = strdup(rose_varrefexp->unparseToString().c_str());

  SgVariableSymbol  *sym = rose_varrefexp->get_symbol();
  SgInitializedName *de = sym->get_declaration();

  // ugliness to remove "UL" from array sizes
  char *ugly = strdup( sym->get_type()->unparseToString().c_str() );
  ulhack(ugly);
  const char *typ = strdup(ugly);
  char *underlying = parseUnderlyingType( strdup(typ) );

  chillAST_DeclRefExpr * chillDRE = new chillAST_DeclRefExpr(typ,  varname );
  return chillDRE;
}




chillAST_node * ConvertRoseIntVal( SgIntVal *roseintval )
{
  int val = roseintval->get_value();
  chillAST_IntegerLiteral  *chillIL = new chillAST_IntegerLiteral( val );
  return chillIL;
}


chillAST_node * ConvertRoseFloatVal( SgFloatVal *rosefloatval )
{
  float val = rosefloatval->get_value();

  // TODO see if we can find the text version, in case they entered 87 digits of pi
  // the clang version does this
  chillAST_FloatingLiteral  *chillFL = new chillAST_FloatingLiteral( val );
  return chillFL;
}




chillAST_node * ConvertRoseDoubleVal( SgDoubleVal *rosedoubleval ) // AST loses precision, stores only float ???
{
  double val = rosedoubleval->get_value();
  chillAST_FloatingLiteral  *chillFL = new chillAST_FloatingLiteral( val );
  return chillFL;
}



chillAST_node * ConvertRoseBasicBlock( SgBasicBlock *bb )
{
  // for now, just a compound statement.  probably need potential for scoping associated with the block
  std::vector<SgStatement* > statements = bb->get_statements();
  int numchildren = statements.size();

  // make an empty CHILL compound statement
  chillAST_CompoundStmt *chillCS = new chillAST_CompoundStmt;

  for (int i=0; i<numchildren; i++) {
    SgStatement *child = statements[i];
    chillAST_node *n =  ConvertRoseGenericAST( (SgNode *)child );

    // comment from CLANG version.  TODO
    // usually n will be a statement. We just add it as a child.
    // SOME DeclStmts have multiple declarations. They will add themselves and return NULL
    if (n) chillCS->addChild( n );
  }

  return chillCS;

}



chillAST_node * ConvertRoseFunctionCallExp( SgFunctionCallExp *FCE )
{
  debug_fprintf(stderr, "\nConvertRoseFunctionCallExp()\n");
  SgExpression  *func = FCE->get_function();
  SgExprListExp *args = FCE->get_args();

  const char *funcname = func->unparseToString().c_str();
  debug_fprintf(stderr, "function %s is of type %s\n", funcname, func->class_name().c_str());
  debug_fprintf(stderr, "(rose) args %s\n", args->unparseToString().c_str());

  chillAST_node *chillfunc = ConvertRoseGenericAST(func);

  // create a call expression from the DRE
  chillAST_CallExpr *chillCE = new chillAST_CallExpr( chillfunc );

  // now add the args  - I can't find a clean way to get the args.
  // this will probably die horribly at some point
  std::vector<std::pair< SgNode *, std::string > > subnodes = args->returnDataMemberPointers();
  int numsubnodes = subnodes.size();
  debug_fprintf(stderr, "when looking for args, %d subnodes\n", numsubnodes);

  int firstnull = numsubnodes;
  for (int i=0; i<numsubnodes; i++) {
    SgNode *part =  subnodes[i].first;

    debug_fprintf(stderr, "subnode %d  '%s'\n", i, subnodes[i].second.c_str());
    if (part == NULL)  { firstnull = i; break; }
  }
  debug_fprintf(stderr, "I think function call has up to ? %d arguments\n", firstnull);

  for (int i=0; i<firstnull; i++) {
    string subtype = subnodes[i].second;
    if (subtype == "expressions") {
      SgNode *part =  subnodes[i].first;
      debug_fprintf(stderr, "CONVERTING SUBNODE %d\n", i);
      debug_fprintf(stderr, "%s\n", part->unparseToString().c_str());
      chillCE->addArg( ConvertRoseGenericAST( part ) );
    }
  }

  return chillCE;

}




chillAST_node * ConvertRoseReturnStmt( SgReturnStmt *rs )

{
  debug_fprintf(stderr, "ConvertRoseReturnStmt() parent %p\n");

  chillAST_node *retval = ConvertRoseGenericAST( rs->get_expression());

  chillAST_ReturnStmt * chillRS = new chillAST_ReturnStmt( retval ); // resets p parent
  if (retval) retval->setParent( chillRS );
  return chillRS;
}




chillAST_node * ConvertRoseArrayRefExp( SgPntrArrRefExp *roseARE ) // most specific binop
{
  debug_fprintf(stderr, "ConvertRoseArrayRefExp()\n");

  chillAST_node *base  = ConvertRoseGenericAST( roseARE->get_lhs_operand_i());
  chillAST_node *index = ConvertRoseGenericAST( roseARE->get_rhs_operand_i());

  chillAST_ArraySubscriptExpr * chillASE = new chillAST_ArraySubscriptExpr( base, index, roseARE);
  return chillASE;
}



chillAST_node * ConvertRoseCastExp( SgCastExp *roseCE )
{
  SgCastExp::cast_type_enum casttype = roseCE->get_cast_type();
  const char *types[] = { "error", "default", "C Style", "C++ const", "C++ static", "C++ dynamic", "C++ reinterpret" };

  debug_fprintf(stderr, "ConvertRoseCastExp()  casttype %d = %s    ", casttype, types[casttype] );

  if (casttype != SgCastExp::e_C_style_cast )
    throw std::runtime_error("unhandled cast expression type " + std::to_string(casttype)+ " = " + types[casttype]);

  SgType *towhat = roseCE->get_type();

  chillAST_node *sub = ConvertRoseGenericAST( roseCE->get_operand() ); // temp - wrong parent

  // this sets the parent on sub
  chillAST_CStyleCastExpr *chillCSCE = new chillAST_CStyleCastExpr( towhat->unparseToString().c_str(), sub );
  return chillCSCE;
}



chillAST_node * ConvertRoseAssignInitializer( SgAssignInitializer *roseAI )
{
  SgExpression *Exp = roseAI->get_operand_i();

  return ConvertRoseGenericAST( ( SgNode *) Exp );
}



// this gets called when a structure is declared (defined?)
// is may get called inside a typedef, or within another structure def
//
// confusing. a struct DECLARATION is really a definition of that kind of struct.
// a variable declaration says this variable is that struct.

//struct S { int a; float b; }; // defines S, S::a, and S::b
//struct S; // declares S
//To sum it up: The C++ standard considers struct x; to be a declaration and struct x {}; a definition.

// declare:  there is something with this name, and it has this type
// definition: Defining something means providing all of the necessary information to create that thing in its entirety
//  Once something is defined, that also counts as declaring it;

chillAST_node * ConvertRoseStructDeclaration( SgClassDeclaration *CLASSDEC )  // DEFINITION of a struct
{
  debug_fprintf(stderr, "ConvertRoseStructDeclaration( CLASSDEC )\n");

  const char *origname = strdup( CLASSDEC->get_name().str());
  debug_fprintf(stderr, "struct name is '%s'\n", origname);

  // temp  TODO   DANGER
  char *name = shortenRoseUnnamedName( origname );
  // now name is either the original, or perhaps a short thing for unnamed structs

  char blurb[4096];

  chillAST_RecordDecl *RD = new chillAST_RecordDecl( name, origname );
  RD->setStruct( true );

  SgClassDefinition *def = CLASSDEC->get_definition();
  std::vector< std::pair<SgNode *, std::string > >  subparts = def->returnDataMemberPointers();
  int numsub =  subparts.size();
  for (int i=0; i<numsub; i++) {
    SgNode *thing = subparts[i].first;
    string name   = subparts[i].second;
    if ( name == string("members") ) {
      if (isSgVariableDeclaration( thing )) {
        SgVariableDeclaration *vardecl = (SgVariableDeclaration *)thing;

        chillAST_VarDecl *VD;
        VD = (chillAST_VarDecl *)ConvertRoseVarDecl(vardecl); //subpart is a child of RecordDecl
        RD->addSubpart(VD);
      }
      else
        continue;
    }
  }
  return RD;
}



//  CREATE A VARIABLE OF SOME TYPE  ???
// this should be a vardecl ???
chillAST_node * ConvertRoseStructDefinition( SgClassDefinition *def )
{
  // we just had this
  SgClassDeclaration *CLASSDEC = def->get_declaration();

  return ConvertRoseStructDeclaration( CLASSDEC );  // wrong wrong wrong TODO
}



// typedef says "this name is really that thing"
chillAST_node * ConvertRoseTypeDefDecl( SgTypedefDeclaration *TDD )   {

  //debug_fprintf(stderr, "\n\nConvertRoseTypeDefDecl()\n");
  // what if this was not a struct?  TODO

  char *typedefname = strdup( TDD->get_name().str());
  //debug_fprintf(stderr, "a new type called %s\n", typedefname);

  // we don't know the underlying type yet ...
  chillAST_TypedefDecl *tdd = new chillAST_TypedefDecl( "", typedefname, "");
  //tdd->setStruct( true ); // might not be a struct?
  tdd->setStructName( typedefname );

  std::vector< std::pair<SgNode *, std::string > >  subparts = TDD->returnDataMemberPointers();
  int numsub =  subparts.size();
  //debug_fprintf(stderr, "%d subparts\n", numsub);
  for (int i=0; i<numsub; i++) {
    SgNode *thing = subparts[i].first;
    string name   = subparts[i].second;
    if (!thing) continue;
    if (name == string("declaration")) {
      // doublecheck
      if ( !strcmp( "SgClassDeclaration", roseGlobalVariantNameList[ thing->variantT() ])) {
        debug_fprintf(stderr, "typedef gonna be a struct\n");
        debug_fprintf(stderr, "that's all\n\n");
        SgClassDeclaration *CLASSDEC = (SgClassDeclaration *)thing;
        chillAST_RecordDecl *rd = (chillAST_RecordDecl *)ConvertRoseStructDeclaration( CLASSDEC ); // parent is the TYPEDEF
        debug_fprintf(stderr, "now have recorddecl\n");
        tdd->setStructInfo( rd );
        return tdd;
      }
      else {
        debug_fprintf(stderr, "uhoh, subpart %d %s is a %s, not an SgClassDeclaration\n", name.c_str(), roseGlobalVariantNameList[ thing->variantT() ]);
        exit(-1);
      }
    }

    if ( name == string("members") ) {
      if (isSgVariableDeclaration( thing )) {
        SgVariableDeclaration *vardecl = (SgVariableDeclaration *)thing;
        std::vector<SgInitializedName* > names = vardecl->get_variables();
        char *varname =shortenRoseUnnamedName( names[0]->unparseToString().c_str());   //


        // rose seems to append the struct name to the name, so  structname::membername
        // TODO rethink
        // for now, mimic clang
        char *name = varname;
        if (index(varname, ':')) {
          int i = index(varname, ':') - varname;
          if (varname[i+1] == ':') name = &varname[i+2];
        }
        SgType *typ = names[0]->get_type();

        char *vartype = strdup( typ->unparseToString().c_str());
        char *arraypart = splitTypeInfo(vartype);
        chillAST_VarDecl *VD = NULL;
        //debug_fprintf(stderr, "(typ) %s (name) %s\n", vartype, name);
        // very clunky and incomplete
        VD = new chillAST_VarDecl( vartype, "", name, chillAST_NodeList(), tdd ); // can't handle arrays yet
        tdd->subparts.push_back(VD);
      }
      else  {
        debug_fprintf(stderr, "ir_rose.c, L866, struct member subpart is not a variable declaration???\n");
        exit(-1);
      }
    }


  }

  return tdd;
}




chillAST_node *ConvertRoseIfStmt(SgIfStmt *ifstatement )
{
  debug_fprintf(stderr, "%s", ifstatement->unparseToString().c_str());

  SgStatement *cond     = ifstatement->get_conditional();
  SgStatement *thenpart = ifstatement->get_true_body();
  SgStatement *elsepart = ifstatement->get_false_body();

  chillAST_node *con = ConvertRoseGenericAST( cond);
  chillAST_node *thn = NULL;
  if (thenpart) thn = ConvertRoseGenericAST( thenpart);
  chillAST_node *els = NULL;
  if (elsepart) els = ConvertRoseGenericAST( elsepart );

  return new chillAST_IfStmt(con,thn,els);
}



chillAST_node * ConvertRoseGenericAST( SgNode *n )
{
  if (n == NULL) return NULL;

  debug_fprintf(stderr, "ConvertRoseGenericAST(),  rose AST node of type %s\n", roseGlobalVariantNameList[ n->variantT() ]);

  chillAST_node *ret = NULL;
  if        ( isSgFunctionDeclaration(n) ) { ret = ConvertRoseFunctionDecl    ((SgFunctionDeclaration *)n );
  } else if ( isSgInitializedName(n)     ) { ret = ConvertRoseInitName         ((SgInitializedName *)n );
  } else if ( isSgFunctionRefExp(n)      ) { ret = ConvertRoseFunctionRefExp     ((SgFunctionRefExp *)n );
  } else if ( isSgVariableDeclaration(n) ) { ret = ConvertRoseVarDecl(       (SgVariableDeclaration *)n );
  } else if ( isSgForStatement(n)        ) { ret = ConvertRoseForStatement    ((SgForStatement *)n );
  } else if ( isSgWhileStmt(n)           ) { ret = ConvertRoseWhileStmt       ((SgWhileStmt *)n);
  } else if ( isSgExprStatement(n)       ) { ret = ConvertRoseExprStatement   ((SgExprStatement *)n ); // expression hidden inside exprstatement

  } else if ( isSgIntVal(n)              ) { ret = ConvertRoseIntVal          ((SgIntVal *)n );        //
  } else if ( isSgFloatVal(n)            ) { ret = ConvertRoseFloatVal        ((SgFloatVal *)n );      //
  } else if ( isSgDoubleVal(n)           ) { ret = ConvertRoseDoubleVal       ((SgDoubleVal *)n );     //


  } else if ( isSgPntrArrRefExp(n)       ) { ret = ConvertRoseArrayRefExp     ((SgPntrArrRefExp *)n ); // ALSO a BinaryOp
  } else if ( isSgArrowExp(n)            ) { ret = ConvertRoseArrowExp        ((SgArrowExp *)n ); // ALSO a BinaryOp
  } else if ( isSgDotExp(n)              ) { ret = ConvertRoseMemberExpr      ((SgDotExp *)n ); // ALSO a BinaryOp

  } else if ( isSgBinaryOp(n)            ) { //debug_fprintf(stderr, "\n(a binary op)\n");
    ret = ConvertRoseBinaryOp        ((SgBinaryOp *)n );     // MANY types will trigger this


  } else if ( isSgCastExp(n)             ) { ret = ConvertRoseCastExp         ((SgCastExp *)n );      // ALSO a UnaryOp
  } else if ( isSgUnaryOp(n)             ) { ret = ConvertRoseUnaryOp         ((SgUnaryOp *)n );      // MANY types will trigger this


  } else if ( isSgVarRefExp(n)           ) { /* debug_fprintf(stderr, "parent %p\n", parent); */ret = ConvertRoseVarRefExp       ((SgVarRefExp *)n );
  } else if ( isSgBasicBlock(n)          ) { ret = ConvertRoseBasicBlock      ((SgBasicBlock *)n );
  } else if ( isSgFunctionCallExp(n)     ) { ret = ConvertRoseFunctionCallExp ((SgFunctionCallExp *)n );
  } else if ( isSgReturnStmt(n)          ) { ret = ConvertRoseReturnStmt      ((SgReturnStmt *)n );
  } else if ( isSgIfStmt(n)              ) { ret = ConvertRoseIfStmt          ((SgIfStmt *)n );
  } else if ( isSgAssignInitializer(n)   ) { ret = ConvertRoseAssignInitializer((SgAssignInitializer *)n);
  } else if ( isSgConditionalExp(n)      ) { ret = ConvertRoseConditionalExp  ((SgConditionalExp *)n);
  } else if ( isSgNullExpression(n)      ) {  //  ignore ??
      //return the NULL
  }
  else if ( isSgNullStatement(n)      ) {
    ret = new chillAST_NULL( );
  }
  //} else if ( isSgLessOrEqualOp(n)       ) { ret = ConvertRoseExprStatement((SgLessOrEqualOp *)n, parent ); // binary
    //} else if ( isSgPlusPlusOp(n)       ) { ret = ConvertRoseExprStatement((SgPlusPlusOp *)n, parent );       // unary

  else {
    std::string s = "ConvertRoseGenericAST(), unhandled node of type " + n->class_name() + " " + n->unparseToString();
    throw std::runtime_error(s.c_str());
  }

  ConvertRosePreprocessing( n, ret );  // check for comments, defines, etc attached to this node

  //debug_fprintf(stderr, "ConvertRoseGenericAST()  END\n");
  return ret;
}


// ----------------------------------------------------------------------------
// Class: IR_roseCode
// ----------------------------------------------------------------------------

void parser::Rose::parse(std::string filename, std::string procname) {
  SgProject* project; // Sg is for Sage, the interface to the rose compiler
  int counter = 0;
  char* argv[2];
  argv[0] = strdup( "rose" );
  argv[1] = strdup( filename.c_str() );

  project = OneAndOnlySageProject = frontend(2,argv);// this builds the Rose AST
  // here we would turn the rose AST into chill AST (right?) maybe not just yet
  SgGlobal *firstScope = getFirstGlobalScope(project);
  SgFilePtrList& file_list = project->get_fileList();

  // this can only be one file, since we started with one file. (?)
  int filecount = 0;
  for (SgFilePtrList::iterator it = file_list.begin(); it != file_list.end();
       it++) {
    filecount++;
  }

  for (SgFilePtrList::iterator it=file_list.begin(); it!=file_list.end();it++) {
    SgSourceFile* file = isSgSourceFile(*it);
    SgGlobal *root = file->get_globalScope();
    toplevel = (SgNode *) root;

    std::vector<std::pair< SgNode *, std::string > > topnodes = toplevel->returnDataMemberPointers();

    //if (!is_fortran_) { // Manu:: this macro should not be created if the input code is in fortran
    //  buildCpreprocessorDefineDeclaration(root,
    //                                      "#define __rose_lt(x,y) ((x)<(y)?(x):(y))",
    //                                      PreprocessingInfo::before);
    //  buildCpreprocessorDefineDeclaration(root,
    //                                      "#define __rose_gt(x,y) ((x)>(y)?(x):(y))",
    //                                      PreprocessingInfo::before);
    //}

    SgDeclarationStatementPtrList& declList = root->get_declarations();

    SgDeclarationStatementPtrList::iterator p = declList.begin();

    while (p != declList.end()) {
      SgFunctionDeclaration *func = isSgFunctionDeclaration(*p);
      if (func) {
        if ((func->get_name().getString()) != procname)
          break;

      }
      p++;
      counter++;
    }
    if (p != declList.end())
      break;

  }  // for each of the 1 possible files

  // OK, here we definitely can walk the tree. starting at root
  entire_file_AST = (chillAST_SourceFile *)ConvertRoseFile((SgNode *) firstScope , filename.c_str());
  chill::scanner::DefinitionLinker dl;
  dl.exec(entire_file_AST);
  chill::scanner::SanityCheck sc;
  sc.run(entire_file_AST, std::cout);
  free(argv[1]);
  free(argv[0]);
}
