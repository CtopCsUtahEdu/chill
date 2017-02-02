


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

#ifdef FRONTEND_ROSE 

#include <string>
#include "ir_rose.hh"
#include "ir_rose_utils.hh"
#include <code_gen/CG_roseRepr.h> // TODO remove 
#include <code_gen/CG_chillRepr.h>
#include <code_gen/CG_chillBuilder.h>
#include <code_gen/rose_attributes.h>

#include "chill_ast.hh"
#include "ir_chill.hh"    // should be ir_chill.hh , not ir_clang.hh 

int IR_Code::ir_pointer_counter = 23;  // TODO this dos nothing ??? 
int IR_Code::ir_array_counter = 1;

using namespace SageBuilder;
using namespace SageInterface;
using namespace omega;

// a global variable. it's OK. 
SgProject *OneAndOnlySageProject;  // a global

// more globals. These may be dumb. TODO 
// in ir_chill.cc vector<chillAST_VarDecl *> VariableDeclarations; 
// in ir_chill.cc vector<chillAST_FunctionDecl *> FunctionDeclarations; 

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
      printf ("-----------------------------------------------\n");
      printf ("Found a %s with preprocessing Info attached:\n", n->getTypeString());
      printf ("(memory address: %p Sage type: %s) in file \n%s (line %d column %d) \n",
              locatedNode, 
              locatedNode->class_name ().c_str (),
              (locatedNode->get_file_info ()->get_filenameString ()).c_str (),
              locatedNode->get_file_info ()->get_line(),
              locatedNode->get_file_info ()->get_col()         );
      fflush(stdout); 

      AttachedPreprocessingInfoType::iterator i;
      int counter = 0;
      for (i = comments->begin (); i != comments->end (); i++) counter++;
      debug_fprintf(stderr, "%d preprocessing info\n\n", counter);
      
      counter = 0;
      for (i = comments->begin (); i != comments->end (); i++){
        printf("-------------PreprocessingInfo #%d ----------- : \n",counter++);
        //printf("classification = %s:\n String format = %s\n",
        //       PreprocessingInfo::directiveTypeName((*i)->getTypeOfDirective ()). c_str (), 
        //       (*i)->getString ().c_str ());

        // this logic seems REALLY WRONG 
        CHILL_PREPROCESSING_POSITION p = CHILL_PREPROCESSING_POSITIONUNKNOWN;
        printf ("relative position is = ");
        if ((*i)->getRelativePosition () == PreprocessingInfo::inside) { 
          printf ("inside\n");
          // ??? t = 
        }
        else if ((*i)->getRelativePosition () == PreprocessingInfo::before) { 
          printf ("before\n"); 
          p =  CHILL_PREPROCESSING_LINEBEFORE;
        }
        else if ((*i)->getRelativePosition () == PreprocessingInfo::after) {
          printf ("after\n"); 
          p =  CHILL_PREPROCESSING_TOTHERIGHT;
        }
        fflush(stdout); 

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
      fflush(stdout); 
    } // comments != NULL
    fflush(stdout); 
  } // located node   
} // ConvertRosePreprocessing 





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

          chillAST_node *node =  ConvertRoseFunctionDecl(fd, topnode ); 
          //debug_fprintf(stderr, "after convert, topnode has %d children\n", topnode->getNumChildren()); 
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
        chillAST_node *vd = ConvertRoseVarDecl( rosevd, topnode);
        
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
        
        // topnode->addChild( vd ); // done in convert
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
          chillAST_node *structdef = ConvertRoseStructDefinition( def, topnode ); // really a recorddecl
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
        else { 
          debug_fprintf(stderr, "unhandled top node SgClassDeclaration that is not a struct!\n");
          exit(-1); 
        }
        
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
        chillAST_TypedefDecl *td = (chillAST_TypedefDecl *)ConvertRoseTypeDefDecl( TDD, topnode );
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

  
  //debug_fprintf(stderr, "ConvertRoseFile(), returning topnode\n"); 
  topnode->dump(); 
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


chillAST_node * ConvertRoseFunctionDecl( SgFunctionDeclaration *fd , chillAST_node *parent) 
{
  const char *functionname = strdup( fd->get_name().str());
  debug_fprintf(stderr, "\nConvertRoseFunctionDecl( %s )\n", functionname); 

  
  // need return type 
  SgType *rt = fd->get_orig_return_type(); 
  string temp = rt->unparseToString(); // so it stays in scope !!
  const char *returntype = temp.c_str(); 
  //debug_fprintf(stderr, "return type %s\n", returntype);
  
  chillAST_FunctionDecl *chillFD = new chillAST_FunctionDecl( returntype,  functionname, parent, (void *)fd /* unique */ );
  ConvertRosePreprocessing( fd, chillFD);  // before doing the function decl itself? 

  // add parameters
  std::vector<SgInitializedName*> args = fd->get_args(); 
  int numargs =  args.size();
  for (int i=0; i<numargs; i++) { 
    chillAST_VarDecl *chillPVD = (chillAST_VarDecl *)ConvertRoseParamVarDecl( args[i], chillFD ); 
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
      chillAST_node *n =  ConvertRoseGenericAST( statement, chillCS ); 
      if (n) {
        chillCS->addChild( n ); 
      }

    }
    chillFD->setBody ( chillCS ); 
  }
  else { 
    //debug_fprintf(stderr, "function %s is a forward declaration or external\n", functionname); 
    chillFD->setForward(); 
  }
  
  debug_fprintf(stderr, "ir_rose.cc  adding %s to FunctionDeclarations\n", functionname); 
  FunctionDeclarations.push_back(chillFD); 

  return chillFD; 
}

// todo initname for vardecl ??? 
chillAST_node * ConvertRoseParamVarDecl( SgInitializedName *vardecl, chillAST_node *parent ) 
{
  //debug_fprintf(stderr, "ConvertRoseParamVarDecl()   "); 
  chillAST_VarDecl *chillVD = (chillAST_VarDecl *) ConvertRoseInitName( vardecl, parent );
  chillVD->isAParameter = true;
  debug_fprintf(stderr, "new parameter:\n"); 
  //chillVD->dump(); printf("\n"); fflush(stdout); // dump in ConvertRoseInitName
  
  return chillVD;
}


char *ConvertSgArrayTypeToString( SgArrayType* AT ) { 
  
  char *arraypart = strdup(""); // leak 
  
  SgExpression* indexExp = AT->get_index();
  if(indexExp) {
    
    //debug_fprintf(stderr, "indexExp %s\n", indexExp->unparseToString().c_str());
    if ( SgBinaryOp *BO = isSgBinaryOp(indexExp) ) { 
      //debug_fprintf(stderr, "Binop\n"); 
      chillAST_BinaryOperator *cbo =  (chillAST_BinaryOperator *)ConvertRoseBinaryOp( BO, NULL );
      int val = cbo->evalAsInt(); 
      //cbo->print(); printf(" = %d\n", val); fflush(stdout);
      
      //debug_fprintf(stderr, "manufacturing binop arraypart '[%d]'\n", val);
      char *leak = (char *)malloc( 64 * sizeof(char));
      sprintf(leak, "[%d]\0", val);
      arraypart = leak; 
      
      // fix vartype? 
      //char *tmp = vartype;
      //char *ind = index(tmp, '[');
      //if (ind) { 
      //  char *newstr = (char *)malloc( 1 + sizeof( tmp ));
      //  *ind = '\0'; 
      //  sprintf(newstr, "%s[%d]\0", tmp, val );
      //  vartype = newstr;
      //  free(tmp); 
      //} 
    }
    else { 
      //free(arraypart);
      char *number = ulhack(strdup( indexExp->unparseToString().c_str() )) ;
      arraypart = (char *)malloc (3 + strlen(number)); 
      sprintf(arraypart, "[%s]\0", number);
      free(number); 
    }
    //debug_fprintf(stderr, "arraypart %s\n", arraypart); 
    //arraypart = splitTypeInfo(vartype); // do before possible mucking with vartype
  }
  
  
  SgArrayType* arraybase = isSgArrayType(AT->get_base_type());
  if (arraybase) { 
    char *first = ConvertSgArrayTypeToString( arraybase ); // recurse;
    //debug_fprintf(stderr, "concatting %s %s\n", first, arraypart ); 
    
    // concat 
    int lenfirst = strlen(first);
    int lensecond = strlen(arraypart);
    char *concatted = (char *)malloc( lenfirst + lensecond + 2 ); // could be 1?
    strcpy(concatted, first);
    strcat(concatted, arraypart);
    //debug_fprintf(stderr, "concatted is %s\n", concatted); 
    free( first );
    free( arraypart ); 
    arraypart = concatted;
  }
  
  return arraypart;
}


chillAST_node *find_wacky_vartype( const char *typ, chillAST_node *parent ) { 
  
  // handle most cases quickly 
  char *t = parseUnderlyingType(strdup(typ));
  //debug_fprintf(stderr, "underlying '%s'\n", t);
  if ( 0 == strcmp("int",   t)  || 
       0 == strcmp("double", t)  ||
//       0 == strcmp("float", t)  ||
       0 == strcmp("float", t) ) return NULL;
  
  
  //debug_fprintf(stderr, "OK, looking for %s\n", t);
  if (!parent) { 
    //debug_fprintf(stderr, "no parent?\n"); 
    return NULL;
  }
  chillAST_node *buh = parent->findDatatype( t );
  
  //if (!buh) debug_fprintf(stderr, "could not find typedef for %s\n", t); 
  //else {
  //debug_fprintf(stderr, "buh IS "); buh->print(); fflush(stdout); 
  //} 
  
  return buh; 
  
  
  
  
}



chillAST_node * ConvertRoseInitName( SgInitializedName *initname, chillAST_node *parent ) // TODO probably wrong
{
  debug_fprintf(stderr, "\n\n***  ConvertRoseInitName()  %s\n", initname->unparseToString().c_str()); 
  debug_fprintf(stderr, "initname %s\n", initname->unparseToString().c_str()); 
  
  int numattr = initname->numberOfAttributes();
  //debug_fprintf(stderr, "initname has %d attributes\n", numattr); 
  
  
  char *varname = shortenRoseStructMemberName( initname->unparseToString().c_str() ); 
  debug_fprintf(stderr, "varname '%s'\n", varname); 
  
  //VariantT V;
  //V = initname->variantT();
  //debug_fprintf(stderr,"variantT %d %s\n", V, roseGlobalVariantNameList[V]);
  
  SgType *typ = initname->get_type();
  // !! if typ->unparseToString()->c_str(), the string and therefore the pointer to char are freed before the next statement ! 
  string really = typ->unparseToString(); 
  const char *otype =   really.c_str();
  debug_fprintf(stderr, "original vartype 0x%x '%s'\n", otype, otype);  
  
  
  bool restricted = isRestrict( otype );
  
  // if this is a struct, the vartype may be a huge mess. make it nicer
  char *vartype = parseUnderlyingType(restricthack( shortenRoseUnnamedName( otype ))); 
  //debug_fprintf(stderr, "prettied vartype '%s'\n", vartype); 
  char *arraypart;// = strdup(""); // leak // splitTypeInfo(vartype); // do before possible mucking with vartype
  arraypart =  parseArrayParts( strdup(otype) );
  debug_fprintf(stderr, "HACK vartype %s arraypart %s\n", vartype, arraypart); 
  
  // need underlying type to pass to constructor?  double and arraypart **, not type double ** and arraypart ** 
  
  SgDeclarationStatement *dec = initname->get_declaration();
  SgDeclarationStatement *defdec = dec->get_definingDeclaration();

  if ( !strncmp(vartype, "struct ", 7) ) {
    debug_fprintf(stderr, "this is a struct ???\n"); 
    //debug_fprintf(stderr, "\ndec  %s\n", dec->unparseToString().c_str());
    std::vector< std::pair< SgNode *, std::string > > subparts = dec->returnDataMemberPointers();
    SgClassDeclaration *CD = NULL; 
    
    chillAST_RecordDecl *RD = new chillAST_RecordDecl( vartype, otype, parent); 
    int numsub =  subparts.size();
    //RD->uniquePtr = dec;
    
    //debug_fprintf(stderr, "%d subparts\n", numsub); 
    for (int i=0; i<numsub; i++) { 
      SgNode *thing = subparts[i].first;
      string name   = subparts[i].second;
      //debug_fprintf(stderr, "name %s\n", name.c_str()); 
      //debug_fprintf(stderr, "\nsubpart %2d %s\n", i, name.c_str());
      //if (!thing) debug_fprintf(stderr, "thing NULL\n");
      //else debug_fprintf(stderr, "ConvertRoseInitName()   thing is of type %s\n", roseGlobalVariantNameList[ thing->variantT() ]);
      
      if (name == string("baseTypeDefiningDeclaration")) { 
        CD = (SgClassDeclaration *)thing;
        //debug_fprintf(stderr, "me: %p  defining %p\n", defdecc, CD); 
        //if (CD) 
        //  debug_fprintf(stderr, "\ndefining  %s\n", CD->unparseToString().c_str());
        //else
        //  debug_fprintf(stderr, "\ndefining  (NULL)\n");
        
      }
      //if (name == string("variables")) {  // apparently just the variable name?
      //  SgInitializedName *IN = (SgInitializedName *)thing;
      //  debug_fprintf(stderr, "variables  %s\n", IN->unparseToString().c_str()); 
      //} 
    }
    
    if (CD) { // once more with feeling
      //debug_fprintf(stderr, "\n\n\n"); // CD: %s", CD->unparseToString()); 
      subparts = CD->returnDataMemberPointers();
      numsub =  subparts.size();
      //debug_fprintf(stderr, "%d subparts\n", numsub); 
      for (int i=0; i<numsub; i++) { 
        SgNode *thing = subparts[i].first;
        string name   = subparts[i].second;
        //debug_fprintf(stderr, "\nsubpart %2d %s\n", i, name.c_str());
        //if (!thing) debug_fprintf(stderr, "thing NULL\n");
        //else debug_fprintf(stderr, "ConvertRoseInitName()   thing is of type %s\n", roseGlobalVariantNameList[ thing->variantT() ]);
      }
    }
    
    debug_fprintf(stderr, "OK, NOW WHAT convertroseinitname\n");
    //die(); 
    exit(-1); 
  }  
  
  
  // figure out if this is some non-standard typedef'd type
  chillAST_node *def = find_wacky_vartype( vartype, parent );
  //if (def) debug_fprintf(stderr, "OK, this is a typedef or struct we have to account for\n"); 
  //arraypart =  parseArrayParts( vartype );  // need to use decl before vartype has parts stripped out 
  
  //this is wrong.  "something *"  is not being flagged as array or pointer 
  //in addition, if vartype is a typedef, I think it's being missed.
  
  if (isSgArrayType(typ)) { 
    //debug_fprintf(stderr, "ARRAY TYPE\n"); 
    //if (arraypart) debug_fprintf(stderr, "but arraypart is already '%s'\n", arraypart);
    
    SgArrayType *AT = (SgArrayType *)typ;
    //if (arraypart) free(arraypart); 
    if (!arraypart) arraypart = ConvertSgArrayTypeToString( AT ); 
    debug_fprintf(stderr, "in convertroseinitname(), arraypart %s\n", arraypart); 
    
    //SgArrayType* arraybase = isSgArrayType(t->get_base_type());
    //SgExpression* indexExp = AT->get_index();
    //if(indexExp) { 
    //  
    //  debug_fprintf(stderr, "indexExp %s\n", indexExp->unparseToString().c_str());
    //  if ( SgBinaryOp *BO = isSgBinaryOp(indexExp) ) { 
    //    //debug_fprintf(stderr, "Binop\n"); 
    //    chillAST_BinaryOperator *cbo =  (chillAST_BinaryOperator *)ConvertRoseBinaryOp( BO, NULL );
    //    int val = cbo->evalAsInt(); 
    //    //cbo->print(); printf(" = %d\n", val); fflush(stdout);
    
    //    //debug_fprintf(stderr, "manufacturing binop arraypart '[%d]'\n", val);
    //    char *leak = (char *)malloc( 64 * sizeof(char));
    //    sprintf(leak, "[%d]\0", val);
    //    arraypart = leak; 
    
    // fix vartype? 
    //char *tmp = vartype;
    char *ind = index(vartype, '[');
    if (ind) { 
      //char *newstr = (char *)malloc( 1 + sizeof( tmp ));
      *ind = '\0'; 
      //sprintf(newstr, "%s %s\0", tmp, arraypart );
      //vartype = newstr;
      //free(tmp); 
    }
    //  }
    //  arraypart = splitTypeInfo(vartype); // do before possible mucking with vartype
    
    //  debug_fprintf(stderr, "vartype = '%s'\n", vartype); 
    //  debug_fprintf(stderr, "arraypart = '%s'\n", arraypart); 
    
    //}
  }
  
  
  if (arraypart == NULL) arraypart = strdup(""); // leak
  //debug_fprintf(stderr, "vartype = '%s'\n", vartype); 
  //debug_fprintf(stderr, "arraypart = '%s'\n", arraypart); 
  
  //SgDeclarationStatement *DS = initname->get_declaration();
  //V = DS->variantT();
  //debug_fprintf(stderr,"declaration statement variantT %d %s\n", V, roseGlobalVariantNameList[V]);
  
  
  char *bracket = index(vartype, '{');
  if (bracket) {   // remove extra for structs 
    *bracket = '\0';
    if (*(bracket-1) == ' ')  *(bracket-1) = '\0'; 
  }
  
  //debug_fprintf(stderr,"%s %s   ", vartype, varname); debug_fprintf(stderr,"arraypart = '%s'\n", arraypart);
  chillAST_VarDecl * chillVD = NULL;
  if (def) { 
    if (def->isRecordDecl()) {
      //debug_fprintf(stderr, "vardecl of a STRUCT\n"); 
      chillVD =  new chillAST_VarDecl((chillAST_RecordDecl*)def,  varname, arraypart, parent); 
    }
    else if (def->isTypeDefDecl()) {
      //debug_fprintf(stderr, "vardecl of a typedef\n"); 
      chillVD = new chillAST_VarDecl((chillAST_TypedefDecl*)def, varname, arraypart, parent); 
    }
    else  { 
      debug_fprintf(stderr, "def but not a recorddecl or a typedefdecl?\n");
      exit(-1); 
    }
  }
  else { 
    //debug_fprintf(stderr, "\n*** creating new chillAST_VarDecl ***\n"); 
    chillVD = new chillAST_VarDecl( vartype,  varname, arraypart, (void *)initname, parent); 
  }
  
  chillVD->isRestrict = restricted; // TODO nicer way 
  chillVD->uniquePtr = defdec;
 
  debug_fprintf(stderr, "ConvertRoseInitName()  storing variable declaration '%s' with unique value %p\n", varname,  chillVD->uniquePtr ); 
  // store this away for declrefexpr that references it! 
  VariableDeclarations.push_back(chillVD);
  //debug_fprintf(stderr, "ConvertRoseInitName() END\n"); 

  // check for an initializer    int i = 0;
  SgInitializer * initptr = initname->get_initptr();
  if (initptr) { 
    debug_fprintf(stderr, "%s gets initialized\n", chillVD->varname);
    chillAST_node *init = ConvertRoseGenericAST( initptr, parent);  // NULL); 
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




chillAST_node * ConvertRoseVarDecl( SgVariableDeclaration *vardecl, chillAST_node *parent ) 
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
  SgType *typ = iname->get_type();
  
  char *ovarname = strdup(iname->unparseToString().c_str()); // original 
  char *otype    = strdup(typ->unparseToString().c_str());   // original
  //this otype is useless, as it does not deal with unnamed structs well 
  
  debug_fprintf(stderr, "entiredecl: '%s'\n", entiredecl); 
  debug_fprintf(stderr, "original vartype '%s'\n", otype); 
  debug_fprintf(stderr, "original varname '%s'\n", ovarname);


  
  char *shorttype = otype; // shortenRoseUnnamedName( otype ); // this used to be REALLY LONG for unnamed structs

  bool restricted = isRestrict( otype ) ; // is __restrict__ in there? 
  //if (restricted) debug_fprintf(stderr, "RESTRICTED\n"); 
  //else debug_fprintf(stderr, "NOT RESTRICTED\n");



  //char *fixedtype = fixUnnamedStructType( otype, entiredecl ); 
  
  //debug_fprintf(stderr, "SHORT TYPE IS %s\n", shorttype);
          
  otype = restricthack( otype); // remove  __restrict__
  
  char *vartype   = fixUnnamedStructType( otype, entiredecl ); 
  char *arraypart = splitTypeInfo(vartype);
  char *varname   = shortenRoseStructMemberName( ovarname );
  if (strlen(arraypart) > 1) 
    debug_fprintf(stderr, "vartype: %s\nvarname: %s\narraypart %s\n\n", vartype, varname, arraypart);
  else 
    debug_fprintf(stderr, "vartype: %s\nvarname: %s\n\n", vartype, varname);

  bool unnamedstruct = false; // ususally not
  char *nameOfStruct = vartype;
  
  if ( !strncmp( vartype, "struct ", 7)) { 
    debug_fprintf(stderr, "it is a struct of type '%s' \n", vartype);
    
    nameOfStruct = &vartype[7];
    debug_fprintf(stderr, "struct name is '%s'\n", nameOfStruct);
    if ('{' == nameOfStruct[0]) {
      debug_fprintf(stderr, "it's an unnamed struct\n");
      unnamedstruct = true; 
    }
  }


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
        RD = (chillAST_RecordDecl *) ConvertRoseStructDeclaration( CD, parent );
        
        debug_fprintf(stderr, "\nhere is the struct definition:\n"); RD->print(); printf("\n"); fflush(stdout); 
        
        //debug_fprintf(stderr, "we need to declare a variable of this STRUCT type named %s\n", varname); 

        // do we need to remember this struct type somewhere in case there are more of them?
        
      }
    }
  }

  
  if ( !strncmp( vartype, "struct ", 7)) { 
    debug_fprintf(stderr, "it is a struct of type '%s' \n", vartype);
    
    if (RD) {
      debug_fprintf(stderr, "we know what the struct definition looks like because it was part of this vardecl\n"); 
      
      chillAST_VarDecl *vd = new chillAST_VarDecl( RD, varname, "", parent);
      //parent->addChild( vd ); // ?? 
      vd->setStruct( true );
      RD->setUnnamed ( unnamedstruct ) ;
      vd->uniquePtr = defdecl; // ??
      debug_fprintf(stderr, "dammit setting %s uniquePtr to %p the SgVariableDeclaration that defined it?\n", vd->varname, vd->uniquePtr); 
      
      //debug_fprintf(stderr, "setting that it IS A STRUCT\n");
      //if (vd->isAStruct()) debug_fprintf(stderr, "yes, it is!\n"); else debug_fprintf(stderr, "no, it isn't!\n"); 
      vd->isRestrict = restricted;

      debug_fprintf(stderr, "STORING vardecl %s in global VariableDeclarations %d\n", vd->varname, VariableDeclarations.size()); 
      VariableDeclarations.push_back( vd );
      return vd; 
    }
    
    debug_fprintf(stderr, "we will have to find the recorddecl for %s\n", nameOfStruct);

    RD = parent->findRecordDeclNamed( nameOfStruct );
    if (!RD) {
      debug_fprintf(stderr, "could not find recordDecl named %s\n", nameOfStruct);
      exit(-1);
    }
    
    RD->print();
    
    // make a variable with recorddecl    duplicated code.  rethink.  TODO 
    chillAST_VarDecl *vd = new chillAST_VarDecl( RD, varname, "", parent);
    vd->setStruct( true );
    vd->isRestrict = restricted; 
    VariableDeclarations.push_back( vd );
    return vd; 


    //*************************************************************** can't get to here 
    
    debug_fprintf(stderr, "checking ugly special case   for unnamed struct in an unnamed struct ??? \n");
    debug_fprintf(stderr, "vartype is %s\n", vartype); 
    if (parent) debug_fprintf(stderr, "parent is a %s\n", parent->getTypeString()); 
    
    if (  !strncmp( vartype, "struct ", 7)) { 
      
      debug_fprintf(stderr, "this variable is a struct \n"); 
      
      char *structName = strdup( &(vartype[7]) ); // remove the "struct "
      debug_fprintf(stderr, "structName '%s'\n", structName); 
      
      
      
      // 
      // see if the parent struct has a 
      // parent->print(); printf("\n\n"); fflush(stdout); 
      
      chillAST_RecordDecl *prd = (chillAST_RecordDecl *) parent;
      
      chillAST_VarDecl *subpart = prd->findSubpartByType( structName ); // find the lost unnamed struct definition ??
      
      //debug_fprintf(stderr, "\nsubpart %p\n", subpart);
      if (subpart)  { 
        //  subpart->print(); printf("\n\n"); fflush(stdout);
        //  subpart->dump();  printf("\n\n"); fflush(stdout);
      } 
      else { 
        debug_fprintf(stderr, "member '%s', can't find unnamed struct that is part of a struct\n", varname); 
        exit(-1);
      }
      
      chillAST_RecordDecl *RD = subpart->vardef;
      if (!RD) { 
        debug_fprintf(stderr, "unnamed struct is part of a struct, but I can't find the RecordDecl\n");
        exit(-1);
      }
      //debug_fprintf(stderr, "\nhere is the struct definition:\n"); RD->print(); printf("\n"); fflush(stdout); 
      
      //debug_fprintf(stderr, "we need to declare a variable of this STRUCT type named %s\n", varname); 
      
      chillAST_VarDecl *vd = new chillAST_VarDecl( RD, varname, "", parent);
      vd->setStruct( true ); 
      //debug_fprintf(stderr, "setting that it IS A STRUCT\n");
      //if (vd->isAStruct()) debug_fprintf(stderr, "yes, it is!\n"); else debug_fprintf(stderr, "no, it isn't!\n"); 
      vd->isRestrict = restricted; 
      VariableDeclarations.push_back( vd );
      return vd; 
    }
    
  }  
  
  
  
  //debug_fprintf(stderr, "there WAS NO defining declaration, so an int or float or something\n");
  // OR  we had   struct { a,b,c }  d,e;  
  // and this is e 
  
  
  // call the  ConvertRoseInitName() that takes a SgInitializedName
  chillAST_VarDecl * chillVD = (chillAST_VarDecl *) ConvertRoseInitName( iname, parent ); 
  
  chillVD->isRestrict = restricted; 
  debug_fprintf(stderr, "ConvertRoseVarDecl() storing variable declaration '%s' with unique value %p from  SgInitializedName\n", entiredecl,  chillVD->uniquePtr ); 
  
  // store this away for declrefexpr that references it! 
  // since we called ConvertRoseInitName() which added it already, don't do that again.  
  //VariableDeclarations.push_back(chillVD);
  return chillVD;
}


chillAST_node * ConvertRoseForStatement( SgForStatement *fs, chillAST_node *parent )
{

  debug_fprintf(stderr, "\nConvertRoseForStatement()  parent %p\n", parent); 
  debug_fprintf(stderr, "%s\n", fs->unparseToString().c_str()); 
  std::vector<SgStatement* >inits  = fs->get_init_stmt();  // these 2 seem to be equivalent. 
  //SgForInitStatement *init2 = fs->get_for_init_stmt();  
  //std::vector<SgStatement* >inits = init2->get_init_stmt(); 
  //debug_fprintf(stderr, "%d inits\n", inits.size()); 
  //if (0 < inits.size()) debug_fprintf(stderr, "inits[0] is a %s\n", inits[0]->class_name().c_str()); 
  if (1 < inits.size()) {
    debug_fprintf(stderr, "ConvertRoseForStatement (ir_rose.cc) more than a single statement in the init, not handled\n"); 
    exit(-1);
  }
  SgStatement *roseinit = inits[0]; 
  
  SgExpression *rosecond  = fs->get_test_expr(); 
  SgStatement  *rosecond2 = fs->get_test();
  
  SgExpression *roseincr  = fs->get_increment(); 
  SgStatement  *rosebody  = fs->get_loop_body(); 
  
  //for (int i=0; i<inits.size(); i++) { 
  //  debug_fprintf(stderr, "%s\n", inits[i]->unparseToString().c_str()); 
  //} 
  //debug_fprintf(stderr, "\n"); 
  
  //debug_fprintf(stderr, "%s\n",    rosecond->unparseToString().c_str()); 
  //debug_fprintf(stderr, "%s\n\n", rosecond2->unparseToString().c_str()); 
  
  //debug_fprintf(stderr, "%s\n\n\n", roseincr->unparseToString().c_str()); 
  
  
  
  // create the 4 components of a for statement
  //debug_fprintf(stderr, "\nconvert init %s\n", roseinit->unparseToString().c_str()); 
  chillAST_node *init = ConvertRoseGenericAST( roseinit, parent); // temp parent so it can look up variables   NULL); 

  debug_fprintf(stderr, "FORSTMT INIT ");
  if (init) {
    init->print();
    debug_fprintf(stderr, "\n");
    init->dump();
  }
  else {
    debug_fprintf(stderr, "(NULL)");
    debug_fprintf(stderr, "this should never happen\n");
    exit(-1);
  }
  debug_fprintf(stderr, "\n\n");
  
  //debug_fprintf(stderr, "\nconvert cond %s\n", rosecond->unparseToString().c_str()); 
  chillAST_node *cond = ConvertRoseGenericAST( rosecond, parent); // temp parent so it can look up variables   NULL); 
  
  //debug_fprintf(stderr, "\nconvert incr %s\n", roseincr->unparseToString().c_str()); 
  chillAST_node *incr = ConvertRoseGenericAST( roseincr, parent); // temp parent so it can look up variables   NULL); 
  
  //debug_fprintf(stderr, "\nfor statement, converting body\n"); 
  chillAST_node *body = ConvertRoseGenericAST( rosebody, parent); // temp parent so it can look up variables   NULL);   
  
  // force body to be a compound statement? 
  if (!body->isCompoundStmt()) { 
    chillAST_CompoundStmt *cs = new chillAST_CompoundStmt();
    cs->addChild( body );
    body = cs;
  }
  
  
  chillAST_ForStmt *chill_loop = new  chillAST_ForStmt( init, cond, incr, body, parent); 
  init->setParent( chill_loop );
  cond->setParent( chill_loop );
  incr->setParent( chill_loop );
  body->setParent( chill_loop );
  
  return chill_loop; 
  
}



chillAST_node * ConvertRoseExprStatement( SgExprStatement *exprstatement, chillAST_node *parent )
{
  chillAST_node *ret = NULL; 
  
  debug_fprintf(stderr, "ConvertRoseExprStatement() exprstatement %s\n", exprstatement->unparseToString().c_str()); 
  //debug_fprintf(stderr, "parent %p\n", parent);
  
  SgExpression *expr = exprstatement->get_expression();
  //debug_fprintf(stderr, "ConvertRoseExprStatement() expr %s\n", expr->unparseToString().c_str()); 
  //SgType *typ= expr->get_type();  // this is the type of the returned vale, not what kind of op 
  //debug_fprintf(stderr, "ConvertRoseExprStatement() expression typ %s\n", typ->unparseToString().c_str()); 
  
  //if (isSgExprListExp(expr)) debug_fprintf(stderr, "ExprListExpr\n");
  //if (isSgCommaOpExp(expr))  debug_fprintf(stderr, "commaop expr\n"); // a special kind of Binary op
  
  
  if (isSgBinaryOp(expr)) { 
    
    //debug_fprintf(stderr, "binary op\n"); 
    SgBinaryOp *bo = (SgBinaryOp *) expr;
    //SgType *botyp= bo->get_type(); 
    //debug_fprintf(stderr, "binop typ %s\n", botyp->unparseToString().c_str()); 
    
    //if (isSgCommaOpExp(bo))   { debug_fprintf(stderr, "commaop binop\n");  }
    debug_fprintf(stderr, "calling ConvertRoseBinaryOp, chill parent is %p\n", parent);
    if (parent != NULL) parent->print(); 
    ret = ConvertRoseBinaryOp( bo, parent ); 
  }
  else if ( isSgIntVal(expr)     ) ret = ConvertRoseIntVal   ((SgIntVal *)expr, parent ); 
  else if ( isSgFloatVal(expr)   ) ret = ConvertRoseFloatVal   ((SgFloatVal *)expr, parent );   
  else if ( isSgDoubleVal(expr)  ) ret = ConvertRoseDoubleVal ((SgDoubleVal *)expr, parent );  
  else if ( isSgFunctionCallExp(expr)  ) ret = ConvertRoseFunctionCallExp ((SgFunctionCallExp *)expr, parent );  
  
  
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






chillAST_node * ConvertRoseBinaryOp( SgBinaryOp *rose_binop, chillAST_node *p )
{
  debug_fprintf(stderr, "\nConvertRoseBinaryOp()\n");
  debug_fprintf(stderr, "%s\n", rose_binop->unparseToString().c_str()); 
  
  //size_t ns = rose_binop->get_numberOfTraversalSuccessors(); 
  //debug_fprintf(stderr, "binary op has %d successors\n", ns);      // always 2 I hope?
  //assert( ns == 2 ) ; 
  VariantT typ = rose_binop->variantT();
  //debug_fprintf(stderr,"\nConvertRoseBinaryOp() AST Node is %d %s\n",typ,roseGlobalVariantNameList[typ]);
  
  
#ifdef WORDY 
  std::vector<std::pair< SgNode *, std::string > > subnodes = rose_binop->returnDataMemberPointers(); 
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
#endif
  
  
  
  const char *op = binop_string( typ ); 
  debug_fprintf(stderr, "op is %s\n", op ); 
  debug_fprintf(stderr, "LHS is %s\n", rose_binop->get_lhs_operand_i()->unparseToString().c_str());
  debug_fprintf(stderr, "RHS is %s\n", rose_binop->get_rhs_operand_i()->unparseToString().c_str());
  
  if ( !strcmp(op, ".") ) { // special case. rose says member func is a binop 
    //debug_fprintf(stderr, "this binaryop is really a member expression\n"); 
    return ConvertRoseMemberExpr( rose_binop, p ) ; // TODO put this in the generic 
  }
  
  // Rose encodes Array Subscript Expression as a binary operator array '[]' index
  // make that a chill ArraySubscriptExpr
  if (isSgPntrArrRefExp(rose_binop)) return ConvertRoseArrayRefExp( (SgPntrArrRefExp *)rose_binop, p); 
  
  
  // when this is a . (member) operation, it would be nice for the rhs 
  // to know that, so that it could know to look for the reference variable 
  // in the struct definitions  TODO 

  debug_fprintf(stderr, "ir_rose.cc L1357, making a binop with no LHS no RHS but a parent\n"); 
  chillAST_BinaryOperator *chill_binop = new chillAST_BinaryOperator(NULL, op, NULL, p); 
  
  chillAST_node *l = ConvertRoseGenericAST( rose_binop->get_lhs_operand_i(), chill_binop );
  chill_binop->setLHS(l);
  
  // now rhs can know what the binop is AND what the lhs is 
  chillAST_node *r = ConvertRoseGenericAST( rose_binop->get_rhs_operand_i(), chill_binop ); 
  chill_binop->setRHS(r);
  
  // old: binop created LAST 
  //chillAST_BinaryOperator *chill_binop = new chillAST_BinaryOperator( l, op, r, p );
  //l->setParent( chill_binop );
  //r->setParent( chill_binop );
  
  return chill_binop; 
}



chillAST_node * ConvertRoseMemberExpr( SgBinaryOp *rose_binop, chillAST_node *p ) // rose member exp is a binop
{
  debug_fprintf(stderr, "\nConvertXXXXMemberExp()\n"); 
  
  VariantT typ = rose_binop->variantT();
  //debug_fprintf(stderr, "ConvertRoseMemberExp()  AST Node is %d %s\n", typ, roseGlobalVariantNameList[ typ ] );  
  
  const char *op = binop_string( typ );
  debug_fprintf(stderr, "op is %s\n", op);
  debug_fprintf(stderr, "LHS is %s\n", rose_binop->get_lhs_operand_i()->unparseToString().c_str());
  debug_fprintf(stderr, "RHS is %s\n", rose_binop->get_rhs_operand_i()->unparseToString().c_str());
  
  if ( strcmp(op, ".") ) { // special case. rose says member func is a binop 
    // this should never happen because this test is what got convertrosebinaryop to call 
    // convertrosememberexpr
    debug_fprintf(stderr, "this member expression is NOT a binop with dot as the operation?\n"); 
    die(); 
  }
  
  typ = rose_binop->get_rhs_operand_i()->variantT();
  //debug_fprintf(stderr, "ConvertRoseMemberExp()  member is %d %s\n", typ,roseGlobalVariantNameList[typ]);
  if (strcmp( "SgVarRefExp", roseGlobalVariantNameList[ typ ])) { 
    debug_fprintf(stderr, "rhs of binop dot expression does not seem right\n");
    exit(-1);
  }
  
  
  chillAST_node *base   = ConvertRoseGenericAST( rose_binop->get_lhs_operand_i(), p ); 
  char *member = ConvertRoseMember( (SgVarRefExp*)(rose_binop->get_rhs_operand_i()), base ); 
  //debug_fprintf(stderr, "member (string) is %s\n", member); 
  
  chillAST_MemberExpr *ME = new chillAST_MemberExpr( base, member, p, (void *)rose_binop); 
  
  //debug_fprintf(stderr, "this is the Member Expresion\n");  ME->print();  debug_fprintf(stderr, "\n"); 
  
  return ME; 
}



chillAST_node * ConvertRoseArrowExp( SgBinaryOp *rose_binop, chillAST_node *p ) // rose arrow (member) exp is a binop
{
  debug_fprintf(stderr, "ConvertXXXXArrowExp()  parent %p\n", p); 
  
  VariantT typ = rose_binop->variantT();
  debug_fprintf(stderr, "ConvertRoseMemberExp()  AST Node is %d %s\n", typ, roseGlobalVariantNameList[ typ ] );  
  
  const char *op = binop_string( typ ); 
  if ( strcmp(op, "->") ) { // special case. rose says member func is a binop 
    // this should never happen because this test is what got convertrosebinaryop to call 
    // convertrosememberexpr
    debug_fprintf(stderr, "this member expression is NOT a binop with arrow as the operation?\n"); 
    die(); 
  }
  
  typ = rose_binop->get_rhs_operand_i()->variantT();
  debug_fprintf(stderr, "ConvertRoseMemberExp()  member is %d %s\n", typ,roseGlobalVariantNameList[typ]);
  if (strcmp( "SgVarRefExp", roseGlobalVariantNameList[ typ ])) { 
    debug_fprintf(stderr, "rhs of binop arrow expression does not seem right\n");
    exit(-1);
  }

  debug_fprintf(stderr, "chillAST_MemberExpr *AE = new chillAST_MemberExpr( NULL /* base */, NULL /* member */, p, (void *)rose_binop, CHILL_MEMBER_EXP_ARROW);\n"); 
  chillAST_MemberExpr *AE = new chillAST_MemberExpr( NULL /* base */, NULL /* member */, p, (void *)rose_binop, CHILL_MEMBER_EXP_ARROW); 

  debug_fprintf(stderr, "converting base\n");
  AE->base   = ConvertRoseGenericAST( rose_binop->get_lhs_operand_i(), AE ); 
  AE->member = ConvertRoseMember( (SgVarRefExp*)(rose_binop->get_rhs_operand_i()), AE ); 

  debug_fprintf(stderr, "member (string) is %s\n", AE->member); 
  
  //chillAST_MemberExpr *ME = new chillAST_MemberExpr( base, member, p, (void *)rose_binop); 
  
  debug_fprintf(stderr, "this is the Arrow Expresion\n"); AE->print(); debug_fprintf(stderr, "\n"); 
  
  return AE; 
}





char * ConvertRoseMember( SgVarRefExp* memb, chillAST_node *base ) // the member itself
{
  //debug_fprintf(stderr, "ConvertXXXXXMember()\n");  
  char *member = strdup(memb->unparseToString().c_str());
  return member;   
  
  /*
  // TODO this should be in convert member expression, probably 
  // this will get called when the binaryop is a dot, which means that this member
  // is the rhs of a binop.  We will find the type of the lhs, to make sure we're 
  // accessing the member of the correct struct
  debug_fprintf(stderr, "ConvertRoseMember(), base is\n"); 
  base->print(); printf(" of type %s\n", base->getTypeString()); fflush(stdout); 
  const char * under = base->getUnderlyingType();
  debug_fprintf(stderr, "underlyingtype %s\n", under); 
  //what we really want is the defition of the struct ... 
  
  chillAST_VarDecl*  underdecl = base->getUnderlyingVarDecl(); 
  debug_fprintf(stderr, "underlying VarDecl\n");
  underdecl->dump(); printf("\n"); fflush(stdout); 
  
  
  SgVariableSymbol  *sym = memb->get_symbol();
  SgInitializedName *def = sym->get_declaration(); 
  
  // ugliness to remove "UL" from array sizes 
  char *ugly = strdup( sym->get_type()->unparseToString().c_str() ); 
  ulhack(ugly);
  const char *typ = strdup(ugly); 
  debug_fprintf(stderr, "typ was %s\n", typ);
  debug_fprintf(stderr, "member '%s'\n", member); 
  
  debug_fprintf(stderr, "base "); base->dump(); printf("\n"); fflush(stdout);
  debug_fprintf(stderr, "member (string) %s\n", member); 
  
  */ 
}








chillAST_node * ConvertRoseUnaryOp( SgUnaryOp *rose_unaryop, chillAST_node *parent )
{
  
  //size_t ns = rose_unaryop->get_numberOfTraversalSuccessors(); 
  //debug_fprintf(stderr, "unary op has %d successors\n", ns);      // always 2 I hope?
  //assert( ns == 1 ) ; 
  
  VariantT typ = rose_unaryop->variantT();
  //debug_fprintf(stderr, "ConvertRoseUnaryOp()  AST Node is %d %s\n", typ, roseGlobalVariantNameList[ typ ] );  
  
  if (isSgCastExp(rose_unaryop)) return ConvertRoseCastExp( (SgCastExp *)rose_unaryop, parent); 
  
  
  const char *op = unaryop_string( typ ); 
  //debug_fprintf(stderr, "op is %s\n", op ); 
  
  // prefix/postfix   
  // rose docs say there is no "unknown"
  // SgUnaryOp::Sgop_mode   SgUnaryOp::prefix  SgUnaryOp::postfix 
  bool pre = (SgUnaryOp::prefix == rose_unaryop->get_mode());
  
  
  chillAST_node *sub = ConvertRoseGenericAST( rose_unaryop->get_operand(), parent); // temp parent to find references NULL ); 
  
  chillAST_UnaryOperator *chillUO = new chillAST_UnaryOperator( op, pre, sub, parent ); 
  sub->setParent( chillUO );
  
  return chillUO; 
}



chillAST_node * ConvertRoseVarRefExp( SgVarRefExp *rose_varrefexp, chillAST_node *p )
{
  debug_fprintf(stderr, "\n*** ConvertXXXXVarRefExpr()\n");
  debug_fprintf(stderr, "ref is '%s'\n", rose_varrefexp->unparseToString().c_str());
  //debug_fprintf(stderr, "parent is: %p\n", p );
  if (p == NULL) {
    debug_fprintf(stderr, "(NULL)\n");
    debug_fprintf(stderr, "ConvertRoseVarRefExp() I can't find the reference if I have no context\n");
  }
  else {
    debug_fprintf(stderr, "p is a %s  (perhaps not filled in yet)\n", p->getTypeString()); 
    //p->print();
  }
  debug_fprintf(stderr, "\n\n");
  
  // this is equivalent to chill declrefexpr ??  but we always know it's a variable
  char *varname = strdup(rose_varrefexp->unparseToString().c_str());
  //debug_fprintf(stderr, "varname %s\n", varname);

  // find chill variable decl with this name?
  chillAST_VarDecl *chillvd= p->findVariableNamed( varname );
  //debug_fprintf(stderr, "in  ConvertRoseVarRefExp() found vardecl %p\n", chillvd);
  debug_fprintf(stderr, "in  ConvertRoseVarRefExp() found vardecl\n");
  if (chillvd) {
    chillvd->print();
    debug_fprintf(stderr, "\n\n");
  }
  else {
    debug_fprintf(stderr, "couldn't find variable named '%s'\n", varname); 
  }
  
  
  SgVariableSymbol  *sym = rose_varrefexp->get_symbol();
  SgInitializedName *de = sym->get_declaration();
  //SgDeclarationStatement *defdecl =
  
  // ugliness to remove "UL" from array sizes 
  char *ugly = strdup( sym->get_type()->unparseToString().c_str() ); 
  ulhack(ugly);
  const char *typ = strdup(ugly); 
  char *underlying = parseUnderlyingType( strdup(typ) ); 
  
  
  //debug_fprintf(stderr, "new chillAST_DeclRefExpr( %s, %s, p)\n", typ, varname); 
  chillAST_DeclRefExpr * chillDRE = new chillAST_DeclRefExpr(typ,  varname, p ); 
  
  
  // find the definition (we hope)          TODO this becomes a function?
  // it's a variable reference 
  //int numvars = VariableDeclarations.size();
  //debug_fprintf(stderr, "checking %d variable declarations\n", numvars); 
  //debug_fprintf(stderr, "looking for varname %s   vartype %s   uniquePtr %p\n", varname, typ, dec); 
  //for (int i=0; i<numvars; i++) { 
  //  debug_fprintf(stderr, "checking against '%s' vartype %s   uniquePtr %p \n", VariableDeclarations[i]->varname, VariableDeclarations[i]->vartype, Vari//ableDeclarations[i]->uniquePtr); 
  //  if (VariableDeclarations[i]->uniquePtr == dec) {
  //    //debug_fprintf(stderr, "found it!\n\n"); 
  //    chillvd = VariableDeclarations[i];
  //    //debug_fprintf(stderr, "found it at variabledeclaration of %s at %d of %d\n", rose_varrefexp->unparseToString().c_str(), i, numvars);
  //    //debug_fprintf(stderr, "chillvd "); chillvd->dump(); printf("\n"); fflush(stdout);    
  //    break;
  //  }
  //  if (streq( varname, VariableDeclarations[i]->varname )) { 
  //    //debug_fprintf(stderr, "here's something of the same name!\n");
  //    if (streq(VariableDeclarations[i]->vartype, underlying)) { 
  //      //debug_fprintf(stderr, "and the same type!\n\n");
  //      chillvd = VariableDeclarations[i];
  //      break;
  //    }
  //  }
  //} 
  
  
  if (!chillvd) { 
    // couldn't find the reference as a variable. perhaps it is a member of a struct
    debug_fprintf(stderr, "VarRefExp %s of type %s is not a variable? maybe it's a struct member\n", varname, typ); 
    //debug_fprintf(stderr, "I should look into how to check for that!\n");
    //debug_fprintf(stderr, "parent p is "); p->dump(); printf("\n"); fflush(stdout);
    
    // climb chillAST looking for this named struct/typedef
    vector<chillAST_VarDecl*> decls;
    p->gatherVarDecls( decls );
    //debug_fprintf(stderr, "%d vardecls above this usage\n", decls.size()); 
    //for (int i=0; i<decls.size(); i++) { 
    //  debug_fprintf(stderr, "decl %s of type %s\n", decls[i]->varname, decls[i]->vartype);
    //} 
    
    
    if (p->isBinaryOperator()) { 
      //debug_fprintf(stderr, "parent is a binary op\n");
      chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *) p;
      if (BO->isStructOp()) { 
        //debug_fprintf(stderr, "binop is a struct op %s\n", BO->op); 
        chillAST_node *l = BO->getLHS();
        chillAST_node *r = BO->getRHS();
        //debug_fprintf(stderr, "l %p    r %p\n", l, r); 
        if (l && !r) { 
          //debug_fprintf(stderr, "we must be the rhs of a struct member!\n");
          //l->dump(); printf("\n"); fflush(stdout); 
          
          if (l->isDeclRefExpr()) { 
            chillAST_node *ptr = ((chillAST_DeclRefExpr*)l)->decl;
            //ptr->dump(); printf("\n"); fflush(stdout); 
            if (ptr->isVarDecl()) { 
              chillAST_VarDecl *str = (chillAST_VarDecl *)ptr;
              //debug_fprintf(stderr, "this struct (?) %s\n", str->getTypeString());
              //str->print(); printf("\n"); fflush(stdout);
              //str->dump();  printf("\n"); fflush(stdout);
              //debug_fprintf(stderr, "looking for member %s in struct definition\n", varname);               
              
              chillAST_VarDecl *sub = NULL;
              
              chillAST_RecordDecl *rd = str->vardef;
              if (rd) { 
                //debug_fprintf(stderr, "the vardecl has a recorddecl\n"); rd->print(); printf("\n"); fflush(stdout); 
                sub = rd->findSubpart( varname ); 
              }
              else { 
                chillAST_TypedefDecl *tdd = str->typedefinition;
                if (tdd) { 
                  //debug_fprintf(stderr, "the vardecl has a typedefinition\n"); tdd->print(); printf("\n"); fflush(stdout); 
                  sub = tdd->findSubpart( varname ); 
                }
                else { 
                  debug_fprintf(stderr, "no recorddecl and not typedefinition\n");
                  exit(-1); 
                }
              }
              
              if (!sub) { 
                debug_fprintf(stderr, "could not find subpart %s\n", varname );
                exit(-1);
              }
              
              //debug_fprintf(stderr, "subpart is "); sub->dump(); printf("\n"); sub->print(); printf("\n"); fflush(stdout); 
              
              chillvd = sub; 
            }
          }
        }
      }
    }
  }
  
  
  
  //if (!chillvd) { 
  //  debug_fprintf(stderr, "\nWARNING, ir_rose.cc rose SgVarRefExp %s refers to a declaration I can't find! at ox%x\n", varname, sym); 
  //  debug_fprintf(stderr, "variables I know of are:\n");
  //  for (int i=0; i<numvars; i++) { 
  //    chillAST_VarDecl *adecl = VariableDeclarations[i];
  //    if (adecl->isParmVarDecl()) debug_fprintf(stderr, "(parameter) ");
  //    debug_fprintf(stderr, "%s %s at location 0x%x\n", adecl->vartype, adecl->varname, adecl->uniquePtr); 
  //  }  
  //  debug_fprintf(stderr, "\n"); 
  //} 
  
  if (chillvd == NULL) { debug_fprintf(stderr, "chillDRE->decl = 0x%x\n", chillvd); exit(-1); }
  
  chillDRE->decl = (chillAST_node *)chillvd; // start of spaghetti pointers ...
  
  free(varname); 
  return chillDRE; 
  
}




chillAST_node * ConvertRoseIntVal( SgIntVal *roseintval, chillAST_node *parent )
{
  int val = roseintval->get_value(); 
  chillAST_IntegerLiteral  *chillIL = new chillAST_IntegerLiteral( val, parent );
  return chillIL;   
}


chillAST_node * ConvertRoseFloatVal( SgFloatVal *rosefloatval, chillAST_node *parent )
{
  float val = rosefloatval->get_value(); 
  
  // TODO see if we can find the text version, in case they entered 87 digits of pi
  // the clang version does this
  chillAST_FloatingLiteral  *chillFL = new chillAST_FloatingLiteral( val, parent );
  return chillFL;   
}




chillAST_node * ConvertRoseDoubleVal( SgDoubleVal *rosedoubleval, chillAST_node *parent ) // AST loses precision, stores only float ??? 
{
  double val = rosedoubleval->get_value(); 
  chillAST_FloatingLiteral  *chillFL = new chillAST_FloatingLiteral( val, parent );
  return chillFL;   
}



chillAST_node * ConvertRoseBasicBlock( SgBasicBlock *bb, chillAST_node *parent )
{
  // for now, just a compound statement.  probably need potential for scoping associated with the block
  std::vector<SgStatement* > statements = bb->get_statements(); 
  int numchildren = statements.size(); 
  
  // make an empty CHILL compound statement 
  chillAST_CompoundStmt *chillCS = new chillAST_CompoundStmt; 
  chillCS->setParent( parent );
  
  for (int i=0; i<numchildren; i++) { 
    SgStatement *child = statements[i]; 
    chillAST_node *n =  ConvertRoseGenericAST( (SgNode *)child, chillCS );
    
    // comment from CLANG version.  TODO 
    // usually n will be a statement. We just add it as a child.
    // SOME DeclStmts have multiple declarations. They will add themselves and return NULL
    if (n) chillCS->addChild( n ); 
  }
  
  return chillCS;
  
}



chillAST_node * ConvertRoseFunctionCallExp( SgFunctionCallExp *FCE, chillAST_node *parent)
{
  debug_fprintf(stderr, "\nConvertRoseFunctionCallExp()\n"); 
  SgExpression  *func = FCE->get_function();
  SgExprListExp *args = FCE->get_args();
  
  const char *funcname = func->unparseToString().c_str(); 
  debug_fprintf(stderr, "function %s is of type %s\n", funcname, func->class_name().c_str()); 
  debug_fprintf(stderr, "(rose) args %s\n", args->unparseToString().c_str()); 
  
  if (!isSgFunctionRefExp(func)) { // should never happen  (assert?)
    debug_fprintf(stderr, "ConvertRoseFunctionCallExp() function call not made of SgFunctionRefExp???\n"); 
    exit(-1); 
  }
  
  SgFunctionRefExp * FRE = (SgFunctionRefExp * ) func; 

  //SgFunctionDeclaration *fdecl;
  //SgFunctionSymbol *symbol = FRE->get_symbol_i(); 
  //fdecl = symbol->get_declaration();
  //debug_fprintf(stderr, "symbol->get_declaration() %p  %s\n", fdecl, fdecl->get_name().str() ); 
  
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
  if (chillfd == NULL) { debug_fprintf(stderr, "couldn't find function definition for %s in the locally defined list of functions\n", func->unparseToString().c_str()); exit(-1); }
  
  // make a DeclRefExpr from the function definition
  chillAST_DeclRefExpr *DRE = new  chillAST_DeclRefExpr( chillfd, NULL); 
  
  // create a call expression from the DRE 
  chillAST_CallExpr *chillCE = new chillAST_CallExpr( DRE, parent); 
  DRE->setParent( chillCE ); // ?? 
  
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
      chillCE->addArg( ConvertRoseGenericAST( part, chillCE ) );
    }
  }
  
  chillCE->dump();\
  debug_fprintf(stderr, "\n\n"); 
  chillCE->print(); 
  debug_fprintf(stderr, "\n\n"); 
  //exit(0); 
  
  return chillCE; 
  
}




chillAST_node * ConvertRoseReturnStmt( SgReturnStmt *rs, chillAST_node *p )
  
{
  debug_fprintf(stderr, "ConvertRoseReturnStmt() parent %p\n");
  
  chillAST_node *retval = ConvertRoseGenericAST( rs->get_expression(), p); // temp fake parent 
  
  chillAST_ReturnStmt * chillRS = new chillAST_ReturnStmt( retval, p ); // resets p parent
  if (retval) retval->setParent( chillRS );
  return chillRS; 
}




chillAST_node * ConvertRoseArrayRefExp( SgPntrArrRefExp *roseARE, chillAST_node *p ) // most specific binop 
{
  //debug_fprintf(stderr, "ConvertRoseArrayRefExp()  p %p\n", p); 
  debug_fprintf(stderr, "ConvertRoseArrayRefExp()\n"); 
  //debug_fprintf(stderr, "converting base\n");
  
  chillAST_node *base  = ConvertRoseGenericAST( roseARE->get_lhs_operand_i(), p ); // temp set p as parent so it can find vardecls?
  chillAST_node *index = ConvertRoseGenericAST( roseARE->get_rhs_operand_i(), p ); 
  
  //debug_fprintf(stderr, "ConvertRoseArrayRefExp, base  '"); base->print(); printf("'\n"); fflush(stdout);
  //debug_fprintf(stderr, "ConvertRoseArrayRefExp, index '"); index->print(); printf("'\n"); fflush(stdout);
  
  //debug_fprintf(stderr, "ConvertRoseArrayRefExp, base "); base->dump(); printf("\n"); fflush(stdout);
  
  chillAST_ArraySubscriptExpr * chillASE = new chillAST_ArraySubscriptExpr( base, index, p, roseARE);
  base->setParent ( chillASE );
  index->setParent( chillASE );
  //debug_fprintf(stderr, "ConvertRoseArrayRefExp() result is "); 
  //chillASE->print(); printf("\n"); fflush(stdout); 
  //chillASE->dump(); printf("\n"); fflush(stdout); 
  return chillASE; 
}



chillAST_node * ConvertRoseCastExp( SgCastExp *roseCE, chillAST_node *parent )
{
  SgCastExp::cast_type_enum casttype = roseCE->get_cast_type();
  char *types[] = { "error", "default", "C Style", "C++ const", "C++ static", "C++ dynamic", "C++ reinterpret" }; 
  
  debug_fprintf(stderr, "ConvertRoseCastExp()  casttype %d = %s    ", casttype, types[casttype] ); 
  
  if (casttype != SgCastExp::e_C_style_cast ) { 
    debug_fprintf(stderr, "unhandled cast expression type %d = %s    ", casttype, types[casttype] ); 
    exit(-1); 
  }
  
  
  SgType *towhat = roseCE->get_type(); 
  //debug_fprintf(stderr, "to %s\n", towhat->unparseToString().c_str()); 
  
  chillAST_node *sub = ConvertRoseGenericAST( roseCE->get_operand(), parent ); // temp - wrong parent 

  // this sets the parent on sub
  chillAST_CStyleCastExpr *chillCSCE = new chillAST_CStyleCastExpr( towhat->unparseToString().c_str(), sub, parent ); 
  //sub->setParent( chillCSCE ); // once mor ewith feeling 
  return chillCSCE; 
  
}



chillAST_node * ConvertRoseAssignInitializer( SgAssignInitializer *roseAI, chillAST_node *p )
{
  SgExpression *Exp = roseAI->get_operand_i();

  return ConvertRoseGenericAST( ( SgNode *) Exp, p); 
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

chillAST_node * ConvertRoseStructDeclaration( SgClassDeclaration *CLASSDEC, chillAST_node *p )  // DEFINITION of a struct
{
  debug_fprintf(stderr, "ConvertRoseStructDeclaration( CLASSDEC, p %p )\n",p); 
  if (p) debug_fprintf(stderr, "parent is a %s\n", p->getTypeString()); 
  
  const char *origname = strdup( CLASSDEC->get_name().str());
  debug_fprintf(stderr, "struct name is '%s'\n", origname); 
  
  // temp  TODO   DANGER 
  char *name = shortenRoseUnnamedName( origname ); 
  // now name is either the original, or perhaps a short thing for unnamed structs
  
  
  
  char blurb[4096];
  //debug_fprintf(stderr, "name is %d characters long\n", strlen(name));
  //sprintf(blurb,  "struct %s", name ); 
  //debug_fprintf(stderr, "blurb is '%s'\n", blurb); 
  
  chillAST_RecordDecl *RD = new chillAST_RecordDecl( name, origname, p);
  RD->setStruct( true ); 
  //RD->setStructName( name );
  
  SgClassDefinition *def = CLASSDEC->get_definition(); 
  std::vector< std::pair<SgNode *, std::string > >  subparts = def->returnDataMemberPointers(); 
  int numsub =  subparts.size();
  //debug_fprintf(stderr, "ConvertRoseStructDeclaration %s has %d subparts\n", blurb, numsub); 
  for (int i=0; i<numsub; i++) { 
    SgNode *thing = subparts[i].first;
    string name   = subparts[i].second;
    //debug_fprintf(stderr, "\nConvertRoseStructDeclaration() %s  subpart %d   %s\n", blurb, i, name.c_str());
    //if (thing) debug_fprintf(stderr, "ConvertRoseStructDeclaration()  thing is of type %s\n", roseGlobalVariantNameList[ thing->variantT() ]);
    if ( name == string("members") ) { 
      if (isSgVariableDeclaration( thing )) {
        //debug_fprintf(stderr, "member is a variable declaration\n");
        SgVariableDeclaration *vardecl = (SgVariableDeclaration *)thing;
        std::vector<SgInitializedName* > names = vardecl->get_variables();
        
        // first, get the type. this may be a really ugly thing for an unnamed struct
        SgType *typ = names[0]->get_type();
        
        char *temp =  shortenRoseUnnamedName( typ->unparseToString().c_str() );
        //debug_fprintf(stderr, "temp %s\n", temp);
        bool restricted = isRestrict( temp); // check it for "__restricted__"
        char *vartype = restricthack( temp );  // strip off "__restricted__"
        
        char *arraypart = splitTypeInfo(vartype);
        vartype = parseUnderlyingType( vartype ); // strip out array stuff
        char *varname = shortenRoseStructMemberName( names[0]->unparseToString().c_str()); 
        
        //debug_fprintf(stderr, "vartype: %s\nvarname: %s\narraypart %s\n\n", vartype, varname, arraypart);
        // but then all these are not used ... 
        
        // this doesn't handle the case where vartype is a struct
        //debug_fprintf(stderr, "struct %s has member %d of vartype %s\n", blurb, i, vartype); 
        
#ifdef FIXLATER 
        chillAST_VarDecl *VD=new chillAST_VarDecl(vartype, varname, arraypart, RD);
#endif 
        
        
        chillAST_VarDecl *VD;
        VD = (chillAST_VarDecl *)ConvertRoseVarDecl(vardecl, RD); //subpart is a child of RecordDecl
        VD->isRestrict = restricted; // TODO nicer 
        
        //debug_fprintf(stderr, "\nprinting %s member %d\n", blurb, i); 
        //VD->print(); printf("\n"); fflush(stdout); 
        //VD->dump();  printf("\n"); fflush(stdout); 
        RD->addSubpart(VD); 
      }
      else  { 
        debug_fprintf(stderr, "ir_rose.c, L866, struct member subpart is not a variable declaration???\n");
        exit(-1); 
      }
    }
  }
  
  //debug_fprintf(stderr, "I just defined a struct typedefdecl called %s\n", name); 
  //if (RD->isAStruct()) debug_fprintf(stderr, "yep, RD is a struct\n"); 
  //RD->print(0, stderr);   
  return RD; 
}



//  CREATE A VARIABLE OF SOME TYPE  ??? 
// this should be a vardecl ??? 
chillAST_node * ConvertRoseStructDefinition( SgClassDefinition *def, chillAST_node *p )
{
  // we just had this
  SgClassDeclaration *CLASSDEC = def->get_declaration(); 
  
  return ConvertRoseStructDeclaration( CLASSDEC, p );  // wrong wrong wrong TODO
}



// typedef says "this name is really that thing"
chillAST_node * ConvertRoseTypeDefDecl( SgTypedefDeclaration *TDD, chillAST_node *p )   {
  
  //debug_fprintf(stderr, "\n\nConvertRoseTypeDefDecl()\n"); 
  // what if this was not a struct?  TODO 
  
  char *typedefname = strdup( TDD->get_name().str()); 
  //debug_fprintf(stderr, "a new type called %s\n", typedefname);
  
  // we don't know the underlying type yet ...
  chillAST_TypedefDecl *tdd = new chillAST_TypedefDecl( "", typedefname, NULL);
  //tdd->setStruct( true ); // might not be a struct? 
  tdd->setStructName( typedefname );
  
  std::vector< std::pair<SgNode *, std::string > >  subparts = TDD->returnDataMemberPointers(); 
  int numsub =  subparts.size();
  //debug_fprintf(stderr, "%d subparts\n", numsub); 
  for (int i=0; i<numsub; i++) { 
    SgNode *thing = subparts[i].first;
    string name   = subparts[i].second;
    //debug_fprintf(stderr, "name %s\n", name.c_str()); 
    //debug_fprintf(stderr, "\nsubpart %2d %s\n", i, name.c_str());
    //debug_fprintf(stderr, "ConvertRoseTypeDefDecl()   thing is of type %s\n", roseGlobalVariantNameList[ thing->variantT() ]);
    
    if (name == string("declaration")) { 
      //debug_fprintf(stderr, "it's a declaration!!\n"); 
      // doublecheck
      if ( !strcmp( "SgClassDeclaration", roseGlobalVariantNameList[ thing->variantT() ])) { 
        debug_fprintf(stderr, "typedef gonna be a struct\n"); 
        if (p) p->print(); 
        debug_fprintf(stderr, "that's all\n\n"); 
        SgClassDeclaration *CLASSDEC = (SgClassDeclaration *)thing;
        chillAST_RecordDecl *rd = (chillAST_RecordDecl *)ConvertRoseStructDeclaration( CLASSDEC, tdd ); // parent is the TYPEDEF
        //debug_fprintf(stderr, "definition that this typedecl called %s really is, is:\n", typedefname);
        //rd->print(); printf("\n"); fflush(stdout); 
        //rd->dump();  printf("\n"); fflush(stdout); 
        // add definition to the typedef ... ?? 

        debug_fprintf(stderr, "now have recorddecl\n");
        if (p) p->print(); 
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
        VD = new chillAST_VarDecl( vartype, name, "", tdd ); // can't handle arrays yet 
        tdd->subparts.push_back(VD); 
      }
      else  { 
        debug_fprintf(stderr, "ir_rose.c, L866, struct member subpart is not a variable declaration???\n");
        exit(-1); 
      }
    } 
    
    
  }
  
  debug_fprintf(stderr, "uhoh\n");
  die(); 
  return NULL; 
  
}




chillAST_node *ConvertRoseIfStmt(SgIfStmt *ifstatement , chillAST_node *p)
{
  debug_fprintf(stderr, "ConvertRoseIfStmt() parent %p\n", p);
  debug_fprintf(stderr, "%s", ifstatement->unparseToString().c_str());

  chillAST_IfStmt *ifstmt = new chillAST_IfStmt( NULL, NULL, NULL, p ); // and EMPTY if but with a parent
  
  SgStatement *cond     = ifstatement->get_conditional();
  SgStatement *thenpart = ifstatement->get_true_body();
  SgStatement *elsepart = ifstatement->get_false_body();
  
  chillAST_node *con = ConvertRoseGenericAST( cond, ifstmt);
  chillAST_node *thn = NULL;
  if (thenpart) { 
    thn = ConvertRoseGenericAST( thenpart, ifstmt);
  }
  chillAST_node *els = NULL;
  if (elsepart) els = ConvertRoseGenericAST( elsepart, ifstmt);

  ifstmt->setCond( con );
  ifstmt->setThen( thn );
  ifstmt->setElse( els );
  
  return ifstmt; 
}



chillAST_node * ConvertRoseGenericAST( SgNode *n, chillAST_node *parent ) 
{
  if (n == NULL) return NULL;
  
  //debug_fprintf(stderr, "ConvertRoseGenericAST(),  rose AST node of type %s    parent %p\n", roseGlobalVariantNameList[ n->variantT() ], parent);
  debug_fprintf(stderr, "ConvertRoseGenericAST(),  rose AST node of type %s\n", roseGlobalVariantNameList[ n->variantT() ]);




  chillAST_node *ret = NULL;
  if        ( isSgFunctionDeclaration(n) ) { ret = ConvertRoseFunctionDecl    ((SgFunctionDeclaration *)n, parent ); 
  } else if ( isSgInitializedName(n)     ) { /*debug_fprintf(stderr, "(1)\n"); */ret = ConvertRoseInitName         ((SgInitializedName *)n, parent );    // param?
  } else if ( isSgVariableDeclaration(n) ) { /*debug_fprintf(stderr, "(2)\n"); */ret = ConvertRoseVarDecl(       (SgVariableDeclaration *)n, parent );    
  } else if ( isSgForStatement(n)        ) { ret = ConvertRoseForStatement    ((SgForStatement *)n, parent ); 
  } else if ( isSgExprStatement(n)       ) { ret = ConvertRoseExprStatement   ((SgExprStatement *)n, parent ); // expression hidden inside exprstatement  
    
  } else if ( isSgIntVal(n)              ) { ret = ConvertRoseIntVal          ((SgIntVal *)n, parent );        // 
  } else if ( isSgFloatVal(n)            ) { ret = ConvertRoseFloatVal        ((SgFloatVal *)n, parent );      // 
  } else if ( isSgDoubleVal(n)           ) { ret = ConvertRoseDoubleVal       ((SgDoubleVal *)n, parent );     // 
    
    
  } else if ( isSgPntrArrRefExp(n)       ) { ret = ConvertRoseArrayRefExp     ((SgPntrArrRefExp *)n, parent ); // ALSO a BinaryOp
  } else if ( isSgArrowExp(n)            ) { ret = ConvertRoseArrowExp        ((SgArrowExp *)n, parent ); // ALSO a BinaryOp
  } else if ( isSgDotExp(n)              ) { ret = ConvertRoseMemberExpr      ((SgDotExp *)n, parent ); // ALSO a BinaryOp
    
  } else if ( isSgBinaryOp(n)            ) { //debug_fprintf(stderr, "\n(a binary op)\n"); 
    ret = ConvertRoseBinaryOp        ((SgBinaryOp *)n, parent );     // MANY types will trigger this 
    
    
  } else if ( isSgCastExp(n)             ) { ret = ConvertRoseCastExp         ((SgCastExp *)n, parent );      // ALSO a UnaryOp
  } else if ( isSgUnaryOp(n)             ) { ret = ConvertRoseUnaryOp         ((SgUnaryOp *)n, parent );      // MANY types will trigger this 
    
    
  } else if ( isSgVarRefExp(n)           ) { /* debug_fprintf(stderr, "parent %p\n", parent); */ret = ConvertRoseVarRefExp       ((SgVarRefExp *)n, parent );     
  } else if ( isSgBasicBlock(n)          ) { ret = ConvertRoseBasicBlock      ((SgBasicBlock *)n, parent );     
  } else if ( isSgFunctionCallExp(n)     ) { ret = ConvertRoseFunctionCallExp ((SgFunctionCallExp *)n, parent );     
  } else if ( isSgReturnStmt(n)          ) { ret = ConvertRoseReturnStmt      ((SgReturnStmt *)n, parent );     
  } else if ( isSgIfStmt(n)              ) { ret = ConvertRoseIfStmt          ((SgIfStmt *)n, parent );     
  } else if ( isSgAssignInitializer(n)   ) { ret = ConvertRoseAssignInitializer((SgAssignInitializer *)n, parent);   
  } else if ( isSgNullExpression(n)      ) {  //  ignore ?? 
      //return the NULL
  }
  else if ( isSgNullStatement(n)      ) {
    ret = new chillAST_NULL( parent ); 
  }
  //} else if ( isSgLessOrEqualOp(n)       ) { ret = ConvertRoseExprStatement((SgLessOrEqualOp *)n, parent ); // binary 
    //} else if ( isSgPlusPlusOp(n)       ) { ret = ConvertRoseExprStatement((SgPlusPlusOp *)n, parent );       // unary  
    
  else { 
    debug_fprintf(stderr, "ConvertRoseGenericAST(), unhandled node of type %s   '%s'\n", n->class_name().c_str(), n->unparseToString().c_str()); 
    exit(-1);
  }

  ConvertRosePreprocessing( n, ret );  // check for comments, defines, etc attached to this node 
  
  //debug_fprintf(stderr, "ConvertRoseGenericAST()  END\n"); 
  return ret;
}






// ----------------------------------------------------------------------------
// Class: IR_roseScalarSymbol
// ----------------------------------------------------------------------------

std::string IR_roseScalarSymbol::name() const {
  return std::string(chillvd->varname); 
}

int IR_roseScalarSymbol::size() const {
  debug_fprintf(stderr, "IR_clangScalarSymbol::size()  probably WRONG\n"); 
  return (8); // bytes?? 
}

bool IR_roseScalarSymbol::operator==(const IR_Symbol &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_roseScalarSymbol *l_that =
    static_cast<const IR_roseScalarSymbol *>(&that);
  return this->chillvd == l_that->chillvd;                       
}

IR_Symbol *IR_roseScalarSymbol::clone() const {
  return new IR_roseScalarSymbol(ir_, chillvd );  // clone
}

// ----------------------------------------------------------------------------
// Class: IR_rosePointerSymbol
// ----------------------------------------------------------------------------
std::string IR_rosePointerSymbol::name() const {
  debug_fprintf(stderr, "IR_rosePointerSymbol::name()\n"); 
	return name_;
}



IR_CONSTANT_TYPE IR_rosePointerSymbol::elem_type() const {
	char *typ = chillvd->vartype;
  if (!strcmp("int", typ)) return IR_CONSTANT_INT;
  else  if (!strcmp("float", typ)) return IR_CONSTANT_FLOAT;
  else  if (!strcmp("double", typ)) return IR_CONSTANT_DOUBLE;
  return IR_CONSTANT_UNKNOWN;
}



int IR_rosePointerSymbol::n_dim() const {
	return dim_;
}


void IR_rosePointerSymbol::set_size(int dim, omega::CG_outputRepr*)  { 
  dims.resize(dim); 
};

omega::CG_outputRepr *IR_rosePointerSymbol::size(int dim) const {
	return dims[dim]; // will fail because often we don't have a size for a given dimension
}


bool IR_rosePointerSymbol::operator==(const IR_Symbol &that) const {
	if (typeid(*this) != typeid(that)) return false;

	const IR_rosePointerSymbol *ps_that = static_cast<const IR_rosePointerSymbol *>(&that);
	return this->chillvd == ps_that->chillvd;
}



IR_Symbol *IR_rosePointerSymbol::clone() const {
	return new IR_rosePointerSymbol(ir_, chillvd);
}



// ----------------------------------------------------------------------------
// Class: IR_roseArraySymbol
// ----------------------------------------------------------------------------

std::string IR_roseArraySymbol::name() const {
  // needs to return string from base
  if (base->isMemberExpr()) { 
    debug_fprintf(stderr, "OMG WE'LL ALL BE KILLED\n"); 
    return  std::string("c.i");  // TODO 
  }
  return std::string(chillvd->varname);  // CHILL 
}



int IR_roseArraySymbol::elem_size() const {
  
  debug_fprintf(stderr, "IR_roseArraySymbol::elem_size() gonna die\n"); 

  debug_fprintf(stderr, "var is of type %s\n", chillvd->vartype);
  char *typ = chillvd->vartype;
  if (!typ) { 
    die(); 
  }
  if (!strcmp("int", typ)) return sizeof(int); // ??
  if (!strcmp("float", typ)) return sizeof(float); // ??
  if (!strcmp("double", typ)) return sizeof(double); // ??
  
  die(); 
/* 
   SgType *tn = vs_->get_type();
   SgType* arrType;
   
   int elemsize;
   
   if (arrType = isSgArrayType(tn)) {
   while (isSgArrayType(arrType)) {
   arrType = arrType->findBaseType();
   }
   } else if (arrType = isSgPointerType(tn)) {
   while (isSgPointerType(arrType)) {
   arrType = arrType->findBaseType();
   }
   }
   
   elemsize = (int) arrType->memoryUsage() / arrType->numberOfNodes();
   return elemsize;
*/
  return 8; // TODO 
}

int IR_roseArraySymbol::n_dim() const {
  debug_fprintf(stderr, "IR_roseArraySymbol::n_dim() %d\n",  chillvd->numdimensions);
  //chillvd->print(); printf("\n"); chillvd->dump(); printf("\n"); fflush(stdout); 

  return chillvd->numdimensions;  
}

IR_CONSTANT_TYPE IR_roseArraySymbol::elem_type() const { 
  
  const char *type = chillvd->underlyingtype;
  if (!strcmp(type, "int"))   return IR_CONSTANT_INT; // should be stored instead of a stings
  if (!strcmp(type, "float")) return IR_CONSTANT_FLOAT;
  return IR_CONSTANT_UNKNOWN;
}

namespace {
  char *irTypeString(IR_CONSTANT_TYPE t) {
    switch (t) {
      case IR_CONSTANT_INT:
        return strdup("int");
        break;
      case IR_CONSTANT_FLOAT:
        return strdup("float");
        break;
      case IR_CONSTANT_DOUBLE:
        return strdup("double");
        break; // ??

      case IR_CONSTANT_UNKNOWN:
      default:
        debug_fprintf(stderr, "irTypeString() unknown IR_CONSTANT_TYPE\n");
        exit(-1);
    }
    return NULL; // unreachable
  }
}


omega::CG_outputRepr *IR_roseArraySymbol::size(int dim) const {
  debug_fprintf(stderr, "IR_roseScalarSymbol::size()  probably WRONG\n");  exit(-1); 
//  return (8); // bytes?? 
  return NULL; 
/* 
   SgArrayType* arrType = isSgArrayType(vs_->get_type());
   // SgExprListExp* dimList = arrType->get_dim_info();
   int count = 0;
   SgExpression* expr;
   SgType* pntrType = isSgPointerType(vs_->get_type());
   
   if (arrType != NULL) {
   SgExprListExp* dimList = arrType->get_dim_info();
   if (!static_cast<const IR_roseCode *>(ir_)->is_fortran_) {
   SgExpressionPtrList::iterator it =
   dimList->get_expressions().begin();
   
   while ((it != dimList->get_expressions().end()) && (count < dim)) {
   it++;
   count++;
   }
   
   expr = *it;
   } else {
   SgExpressionPtrList::reverse_iterator i =
   dimList->get_expressions().rbegin();
   for (; (i != dimList->get_expressions().rend()) && (count < dim);
   i++) {
   
   count++;
   }
   
   expr = *i;
   }
   } else if (pntrType != NULL) {
   
   while (count < dim) {
   pntrType = (isSgPointerType(pntrType))->get_base_type();
   count++;
   }
   if (isSgPointerType(pntrType))
   expr = new SgExpression;
   }
   
   if (!expr)
   throw ir_error("Index variable is NULL!!");
   
   // Manu :: debug
   std::cout << "---------- size :: " << isSgNode(expr)->unparseToString().c_str() << "\n";
   
   return new omega::CG_roseRepr(expr); commented out 
*/ 
}

IR_ARRAY_LAYOUT_TYPE IR_roseArraySymbol::layout_type() const {
  if (static_cast<const IR_roseCode *>(ir_)->is_fortran_)
    return IR_ARRAY_LAYOUT_COLUMN_MAJOR;
  else
    return IR_ARRAY_LAYOUT_ROW_MAJOR;
  
}

bool IR_roseArraySymbol::operator==(const IR_Symbol &that) const {
  
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_roseArraySymbol *l_that = static_cast<const IR_roseArraySymbol *>(&that);
  return this->chillvd == l_that->chillvd;
  //return this->vs_ == l_that->vs_;
  
}

IR_Symbol *IR_roseArraySymbol::clone() const {
  return new IR_roseArraySymbol(ir_, chillvd );  // clone
  //return new IR_roseArraySymbol(ir_, vs_);
}

// ----------------------------------------------------------------------------
// Class: IR_roseConstantRef
// ----------------------------------------------------------------------------

bool IR_roseConstantRef::operator==(const IR_Ref &that) const {
  
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_roseConstantRef *l_that =
    static_cast<const IR_roseConstantRef *>(&that);
  
  if (this->type_ != l_that->type_)
    return false;
  
  if (this->type_ == IR_CONSTANT_INT)
    return this->i_ == l_that->i_;
  else
    return this->f_ == l_that->f_;
  
}


omega::CG_outputRepr *IR_roseConstantRef::convert() {
  if (type_ == IR_CONSTANT_INT) {
     debug_fprintf(stderr, "IR_chillConstantRef::convert() unimplemented\n");  exit(-1); 
  } else
    throw ir_error("constant type not supported");
  
}

IR_Ref *IR_roseConstantRef::clone() const {
  if (type_ == IR_CONSTANT_INT)
    return new IR_roseConstantRef(ir_, i_);
  else if (type_ == IR_CONSTANT_FLOAT)
    return new IR_roseConstantRef(ir_, f_);
  else
    throw ir_error("constant type not supported");
  
}

// ----------------------------------------------------------------------------
// Class: IR_roseScalarRef
// ----------------------------------------------------------------------------

bool IR_roseScalarRef::is_write() const {
  if (is_write_ == 1)
    return true;
  
  return false;
}

IR_ScalarSymbol *IR_roseScalarRef::symbol() const {
  chillAST_VarDecl *vd = NULL;
  if (chillvd) vd = chillvd; 
  return new IR_roseScalarSymbol(ir_, vd); // IR_clangScalarRef::symbol()
}

bool IR_roseScalarRef::operator==(const IR_Ref &that) const {
  if (typeid(*this) != typeid(that))
    return false;
  
  const IR_roseScalarRef *l_that =
    static_cast<const IR_roseScalarRef *>(&that);
  
  debug_fprintf(stderr, " IR_roseScalarRef::operator== gonna die\n"); 
  die(); 
  return true; 
}

omega::CG_outputRepr *IR_roseScalarRef::convert() {
  if (!dre) debug_fprintf(stderr, "IR_roseScalarRef::convert()   ROSESCALAR REF has no dre. will probably die soon\n"); 
  omega::CG_chillRepr *result = new omega::CG_chillRepr(dre);
  delete this;
  return result;
}

IR_Ref * IR_roseScalarRef::clone() const {
  if (dre) return new IR_roseScalarRef(ir_, dre); // use declrefexpr if it exists
  return new IR_roseScalarRef(ir_, chillvd); // uses vardecl
}

// ----------------------------------------------------------------------------
// Class: IR_roseArrayRef, also FORMERLY IR_rosePointerArrayRef which was the same ???
// ----------------------------------------------------------------------------
omega::CG_outputRepr *IR_rosePointerArrayRef::index(int dim) const {
  //debug_fprintf(stderr, "IR_roseArrayRef::index( %d )  \n", dim); 
  return new omega::CG_chillRepr( chillASE->getIndex(dim) );// since we may not know index, this could die ???
}

IR_PointerSymbol *IR_rosePointerArrayRef::symbol() const {  // out of ir_clang.cc 
  chillAST_node *mb = chillASE->multibase(); 
  chillAST_VarDecl *vd = (chillAST_VarDecl*)mb; 
  IR_PointerSymbol *PS =  new IR_rosePointerSymbol(ir_, chillASE->basedecl);  // vd); 
  return  PS;
}

bool IR_rosePointerArrayRef::operator!=(const IR_Ref &that) const {
  //debug_fprintf(stderr, "IR_roseArrayRef::operator!=\n"); 
  bool op = (*this) == that; // opposite
  return !op;
}

bool IR_rosePointerArrayRef::operator==(const IR_Ref &that) const {
  const IR_rosePointerArrayRef *l_that = static_cast<const IR_rosePointerArrayRef *>(&that);
  const chillAST_ArraySubscriptExpr* thatASE = l_that->chillASE;
  return (*chillASE) == (*thatASE);
}

omega::CG_outputRepr *IR_rosePointerArrayRef::convert() {
  CG_chillRepr *result = new  CG_chillRepr( chillASE->clone() ); 
  // delete this;  // if you do this, and call convert twice, you're DEAD 
  return result;
}

void IR_rosePointerArrayRef::Dump() const { 
  //debug_fprintf(stderr, "IR_rosePointerArrayRef::Dump()  this 0x%x  chillASE 0x%x\n", this, chillASE); 
  chillASE->print(); printf("\n");fflush(stdout);
}

IR_Ref *IR_rosePointerArrayRef::clone() const {
  return new IR_rosePointerArrayRef(ir_, chillASE, iswrite);
}




bool IR_roseArrayRef::is_write() const {  // out of ir_clang.cc 
  return (iswrite); // TODO  ?? 
}

omega::CG_outputRepr *IR_roseArrayRef::index(int dim) const {
  //debug_fprintf(stderr, "IR_roseArrayRef::index( %d )  \n", dim); 
  return new omega::CG_chillRepr( chillASE->getIndex(dim) );// out of ir_clang.cc
}


IR_ArraySymbol *IR_roseArrayRef::symbol() const {  // out of ir_clang.cc 
  //debug_fprintf(stderr, "IR_roseArrayRef::symbol()\n"); 
  //chillASE->print(); printf("\n"); fflush(stdout); 
  //debug_fprintf(stderr, "base: of type %s ", chillASE->base->getTypeString() );  chillASE->base->print();  printf("\n"); fflush(stdout); 
  
  //debug_fprintf(stderr, "ASE in IR_roseArrayRef is \n");
  //chillASE->print(); printf("\n"); fflush(stdout);
  //chillASE->dump(); printf("\n"); fflush(stdout);
  
  chillAST_node *mb = chillASE->multibase(); 
  chillAST_VarDecl *vd = (chillAST_VarDecl*)mb; 
  //debug_fprintf(stderr, "symbol vardecl "); vd->print(); printf("\n"); fflush(stdout); 
  //debug_fprintf(stderr, "base %s\n", vd->varname); 
  //debug_fprintf(stderr, "IR_roseArrayRef::symbol() returning new IR_roseArraySymbol( %s )\n", vd->varname); 
  //vd->print(); printf("\n"); fflush(stdout); 
  //vd->dump();  printf("\n"); fflush(stdout); 


  IR_ArraySymbol *AS =  new IR_roseArraySymbol(ir_, chillASE->base);  // vd); 
  
  return  AS;
  
}

bool IR_roseArrayRef::operator!=(const IR_Ref &that) const {
  //debug_fprintf(stderr, "IR_roseArrayRef::operator!=\n"); 
  bool op = (*this) == that; // opposite
  return !op;
}

void IR_roseArrayRef::Dump() const { 
  //debug_fprintf(stderr, "IR_roseArrayRef::Dump()  this 0x%x  chillASE 0x%x\n", this, chillASE); 
  chillASE->print(); printf("\n");fflush(stdout);
}

bool IR_roseArrayRef::operator==(const IR_Ref &that) const {
  const IR_roseArrayRef *l_that = static_cast<const IR_roseArrayRef *>(&that);
  const chillAST_ArraySubscriptExpr* thatASE = l_that->chillASE;
  //printf("other is:\n");  thatASE->print(); printf("\n"); fflush(stdout);
  //debug_fprintf(stderr, "addresses are 0x%x  0x%x\n", chillASE, thatASE ); 
  return (*chillASE) == (*thatASE);
  
  // not this ?? TODO 
  //bool op = (*this) == that; // same
  //return op;
  
  
  
/*   if (typeid(*this) != typeid(that))
     return false;
     
     const IR_roseArrayRef *l_that = static_cast<const IR_roseArrayRef *>(&that);
     
     return this->ia_ == l_that->ia_;
*/ 
  
}

omega::CG_outputRepr *IR_roseArrayRef::convert() {
  CG_chillRepr *result = new  CG_chillRepr( chillASE->clone() ); 
  // delete this;  // if you do this, and call it twice, you're DEAD Manu doesn't like this ...
  return result;
}

IR_Ref *IR_roseArrayRef::clone() const {
  return new IR_roseArrayRef(ir_, chillASE, iswrite);
//  return new IR_roseArrayRef(ir_, ia_, is_write_);
}

// ----------------------------------------------------------------------------
// Class: IR_roseLoop
// ----------------------------------------------------------------------------

IR_roseLoop::IR_roseLoop(const IR_Code *ir, chillAST_node *achillnode) {  // directly from ir_clang.cc 
  
  //debug_fprintf(stderr, "IR_roseLoop::IR_roseLoop(ir_, chillforstmt)\n"); 
  
  if (!achillnode->isForStmt()) { 
    debug_fprintf(stderr, "IR_roseLoop::IR_roseLoop(const IR_Code *ir, chillAST_node *achillnode) chillnode is not a ForStmt\n"); 
    exit(-1); 
  }
  
  chillAST_ForStmt *achillforstmt = (chillAST_ForStmt *) achillnode;
  //debug_fprintf(stderr, "loop is:\n"); 
  //achillforstmt->print(); printf("\n"); fflush(stdout); 
  //achillforstmt->dump(); printf("\n"); fflush(stdout);
  
  ir_ = ir; 
  chillforstmt = achillforstmt;
  chillbody    = achillforstmt->getBody(); 
  //debug_fprintf(stderr, "IR_roseLoop::IR_roseLoop()    chillbody\n"); 
  //debug_fprintf(stderr, "body is:\n"); 
  //chillbody->print(); printf("\n\n"); fflush(stdout); 
  //debug_fprintf(stderr, "chillbody of type %s\n", chillbody->getTypeString()); 

  // there should be a better way to do this
  chillAST_node *initnode = chillforstmt->getInit();
  chillAST_BinaryOperator *initBO = (chillAST_BinaryOperator *)chillforstmt->getInit();
  chillAST_VarDecl        *initVD = (chillAST_VarDecl        *)chillforstmt->getInit();

  chillAST_VarDecl *loopvar = NULL;
  
  chillAST_BinaryOperator *cond = (chillAST_BinaryOperator *)chillforstmt->getCond();

  bool initOK = initnode->isNull() || initBO->isAssignmentOp()  || ( initVD->isVarDecl() && initVD->hasInit() );
  
  // check to be sure  (assert) 
  if (!initOK || !cond->isComparisonOp() ) {
    debug_fprintf(stderr, "ir_rose.cc, malformed loop init or cond:\n");
   
    if ( !cond->isComparisonOp() ) {
      debug_fprintf(stderr, "cond is %s, NOT ComparisonOp\n", cond->getTypeString()); 
    }
    achillforstmt->print(); 
    exit(-1); 
  }
  
  // this is STILL too simple. the init can be multipe statements, for example
  //   for (i=0,j=3; i<5; i++)
  //
  if (initnode->isNull()) {
    // TODO we're screwed. we would have to figure out what the lower bound is, which could be impossible
    chilllowerbound = new chillAST_IntegerLiteral( 0 );  // make a fake constant
    debug_fprintf(stderr, "\n\nguessing at the loop's lower bound since it does not have an init!\n");
    chillforstmt->print(); debug_fprintf(stderr, "\n\n\n");
    initVD = NULL;
    initBO = NULL; 
  }
  else if (initBO->isAssignmentOp()) {
    //loopvar         = initBO->getLHS();
    chilllowerbound = initBO->getRHS();
    initVD = NULL; 
  }
  else {
    //loopvar         = initVD;
    chilllowerbound = initVD->getInit();
    initBO = NULL; 
  }
  
  chillupperbound   = cond->getRHS();
  conditionoperator = achillforstmt->conditionoperator; 
  
  chillAST_node *inc  = chillforstmt->getInc();
  // check the increment
  //debug_fprintf(stderr, "increment is of type %s\n", inc->getTypeString()); 
  //inc->print(); printf("\n"); fflush(stdout);
  
  if (inc->getType() == CHILLAST_NODETYPE_UNARYOPERATOR) {
    if (!strcmp(((chillAST_UnaryOperator *) inc)->op, "++")) step_size_ = 1;
    else  step_size_ = -1;
  }
  else if (inc->getType() == CHILLAST_NODETYPE_BINARYOPERATOR) {
    int beets = false;  // slang
    chillAST_BinaryOperator *bop = (chillAST_BinaryOperator *) inc;
    if (bop->isAssignmentOp()) {        // I=I+1   or similar
      chillAST_node *rhs = bop->getRHS();  // (I+1)
      // TODO looks like this will fail for I=1+I or I=J+1 etc. do more checking
      
      char *assop =  bop->getOp(); 
      //debug_fprintf(stderr, "'%s' is an assignment op\n", bop->getOp()); 
      if (streq(assop, "+=") || streq(assop, "-=")) {
        chillAST_node *stride = rhs;
        //debug_fprintf(stderr, "stride is of type %s\n", stride->getTypeString());
        if  (stride->isIntegerLiteral()) {
          int val = ((chillAST_IntegerLiteral *)stride)->value;
          if      (streq( assop, "+=")) step_size_ =  val;
          else if (streq( assop, "-=")) step_size_ = -val;
          else beets = true; 
        }
        else beets = true;  // += or -= but not constant stride
      }
      else if (rhs->isBinaryOperator()) { 
        chillAST_BinaryOperator *binoprhs = (chillAST_BinaryOperator *)rhs;
        chillAST_node *intlit =  binoprhs->getRHS();
        if (intlit->isIntegerLiteral()) {
          int val = ((chillAST_IntegerLiteral *)intlit)->value;
          if      (!strcmp( binoprhs->getOp(), "+")) step_size_ =  val;
          else if (!strcmp( binoprhs->getOp(), "-")) step_size_ = -val;
          else beets = true; 
        }
        else beets = true;
      }
      else beets = true;
    }
    else beets = true;
    
    if (beets) {
      debug_fprintf(stderr, "malformed loop increment (or more likely unhandled case)\n");
      inc->print(); 
      exit(-1); 
    }
  } // binary operator 
  else { 
    debug_fprintf(stderr, "IR_Roseloop constructor, unhandled loop increment\n");
    inc->print(); 
    exit(-1); 
  }
  //inc->print(0, stderr);debug_fprintf(stderr, "\n"); 


  //
  // figure out what the loop index variable is.   there has to be one, for us.
  //
  chillAST_DeclRefExpr *dre = NULL;
  if (initBO) { // TODO ugly
    dre = (chillAST_DeclRefExpr *)initBO->getLHS();
    if (!dre->isDeclRefExpr()) { 
      debug_fprintf(stderr, "malformed loop init. initBO\n"); 
      initBO->print(); 
    }
  }
  else if (initVD) { // TODO ugly  init is a vardecl WITH initialization
    dre = buildDeclRefExpr( initVD ); 
  }
  else { // TODO ugly 
    // init is not going to find the loop var for us 
    // see if we can find the declrefexpr from the cond or incr ??
    //cond->dump(); debug_fprintf(stderr, "\n\n"); 
    //inc->dump(); debug_fprintf(stderr, "\n\n");
    if (cond->isBinaryOperator()) {
      chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *) cond;
      chillAST_node *l = BO->getLHS();
      if (l->isDeclRefExpr()) dre = (chillAST_DeclRefExpr *)l;
    }
    if (!dre) { // we haven't figured it out yet.  try the increment
      if (inc->isBinaryOperator()) { // i = i + 1
        chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *)inc;
        chillAST_node *l = BO->getLHS();
        if (l->isDeclRefExpr()) dre = (chillAST_DeclRefExpr *)l;
      }
      else if (inc->isUnaryOperator()) { // ++ or -- maybe
        chillAST_UnaryOperator *UO = (chillAST_UnaryOperator *)inc;
        chillAST_node *s = UO->subexpr;
        if (s->isDeclRefExpr()) dre = (chillAST_DeclRefExpr *)s;
      }
    }
  }

  if (!dre) {
    debug_fprintf(stderr, "ir_rose.cc L2842, I can't figure out the loop index variable\n");
    chillforstmt->print(); debug_fprintf(stderr, "\n\n\n");
    exit(-1);
  }
  chillindex = dre; // the loop index variable
  
  //debug_fprintf(stderr, "\n\nindex is ");  dre->print(0, stderr);  debug_fprintf(stderr, "\n"); 
  //debug_fprintf(stderr, "init is   "); 
  //chilllowerbound->print(0, stderr);  debug_fprintf(stderr, "\n");
  //debug_fprintf(stderr, "condition is  %s ", "<"); 
  //chillupperbound->print(0, stderr);  debug_fprintf(stderr, "\n");
  //debug_fprintf(stderr, "step size is %d\n\n", step_size_) ; 
  
  //debug_fprintf(stderr, "IR_roseLoop::IR_roseLoop() DONE\n"); 
}

IR_ScalarSymbol *IR_roseLoop::index() const {
  return new IR_roseScalarSymbol(ir_, chillindex->getVarDecl());
}

omega::CG_outputRepr *IR_roseLoop::lower_bound() const {
  debug_fprintf(stderr, "IR_roseLoop::lower_bound()\n"); 
  //chilllowerbound->print(); printf("\n"); fflush(stdout); 
  //chilllowerbound->dump(); printf("\n"); fflush(stdout); 
  omega::CG_outputRepr *OR =  new omega::CG_chillRepr(chilllowerbound);
  //debug_fprintf(stderr, "lower bound DONE\n"); 
  return OR;
}

omega::CG_outputRepr *IR_roseLoop::upper_bound() const {
  debug_fprintf(stderr, "IR_roseLoop::upper_bound()\n"); 
  return new omega::CG_chillRepr(chillupperbound);
}

IR_CONDITION_TYPE IR_roseLoop::stop_cond() const {
  chillAST_BinaryOperator *loopcondition = (chillAST_BinaryOperator*) chillupperbound;
  //debug_fprintf(stderr, "IR_roseLoop::stop_cond()\n"); 
  return conditionoperator; 
  
  
}

IR_Block *IR_roseLoop::body() const {
  debug_fprintf(stderr, "IR_roseLoop::body()\n");
  IR_Block *tmp =  new IR_chillBlock(ir_, chillbody ) ;
  //debug_fprintf(stderr, "new IR_Block is %p\n", tmp); 
  return tmp; 
}

int IR_roseLoop::step_size() const {
  return step_size_;  // is it always an integer?   TODO 
}

IR_Block *IR_roseLoop::convert() {
  //debug_fprintf(stderr, "IR_roseLoop::convert()   maybe \n"); 
  // delete this ??? 
  return new IR_chillBlock( ir_, chillbody ); // ?? 
}

IR_Control *IR_roseLoop::clone() const {
  //debug_fprintf(stderr, "IR_roseLoop::clone()\n"); 
  return new IR_roseLoop(ir_, chillforstmt);
}

// ----------------------------------------------------------------------------
// Class: IR_roseCode
// ----------------------------------------------------------------------------

IR_roseCode::IR_roseCode(const char *file_name, const char* proc_name, const char *dest_name) :
  IR_chillCode() {
  
  debug_fprintf(stderr, "IR_roseCode::IR_roseCode( file_name %s, proc_name %s )\n", file_name, proc_name);
  
  filename = strdup(file_name);          // store in class 
  procedurename = strdup( proc_name );   // store in class
  if (dest_name != NULL)  setOutputName( dest_name ); 
  else { 
    char buf[1024];
    sprintf(buf, "rose_%s\0", file_name); 
    setOutputName( buf ); 
  }
  SgProject* project; // Sg is for Sage, the interface to the rose compiler
  
  int counter = 0;
  
  char* argv[2];
  argv[0] = strdup( "rose" );
  argv[1] = strdup( file_name ); 
  
  // project = (IR_roseCode_Global_Init::Instance(argv))->project;
  //debug_fprintf(stderr, "IR_roseCode::IR_roseCode  actually parsing %s using rose?\n", file_name); 
  project = OneAndOnlySageProject = frontend(2,argv);// this builds the Rose AST
  
  //debug_fprintf(stderr, "IR_roseCode::IR_roseCode()  project defined. file parsed by Rose\n"); 
  
  
  // here we would turn the rose AST into chill AST (right?) maybe not just yet
  
  
  firstScope = getFirstGlobalScope(project);
  SgFilePtrList& file_list = project->get_fileList();
  
  // this can only be one file, since we started with one file. (?)
  int filecount = 0; 
  for (SgFilePtrList::iterator it = file_list.begin(); it != file_list.end();
       it++) {
    filecount++;
  }
  //debug_fprintf(stderr, "%d files\n", filecount); 
  
  
  for (SgFilePtrList::iterator it=file_list.begin(); it!=file_list.end();it++) {
    file = isSgSourceFile(*it);
    if (file->get_outputLanguage() == SgFile::e_Fortran_output_language)
      is_fortran_ = true;
    else
      is_fortran_ = false;
    
    // Manu:: debug
    // if (is_fortran_)
    //   std::cout << "Input is a fortran file\n";
    // else
    //     std::cout << "Input is a C file\n";
    
    root = file->get_globalScope();
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
    
    symtab_ = isSgScopeStatement(root)->get_symbol_table();
    SgDeclarationStatementPtrList& declList = root->get_declarations();
    
    p = declList.begin();
    
    while (p != declList.end()) {
      func = isSgFunctionDeclaration(*p);
      if (func) {
        if (!strcmp((func->get_name().getString()).c_str(), proc_name))
          break;
        
      }
      p++;
      counter++;
    }
    if (p != declList.end())
      break;
    
  }  // for each of the 1 possible files
  
  
  // OK, here we definitely can walk the tree. starting at root
  //debug_fprintf(stderr, "creating chillAST from Rose AST\n");
  //debug_fprintf(stderr, "\nroot is %s  %p\n", root->class_name().c_str(), root ); 
  
  //chillAST_node * 
  //entire_file_AST = (chillAST_SourceFile *)ConvertRoseFile((SgNode *) root , file_name); 
  entire_file_AST = (chillAST_SourceFile *)ConvertRoseFile((SgNode *) firstScope , file_name); 
  
  vector<chillAST_node*> functions;
  chillAST_FunctionDecl *localFD = findFunctionDecl(  entire_file_AST, proc_name );
  //debug_fprintf(stderr, "local Function Definition %p\n", localFD); 
  //localFD->print(); printf("\n\n"); fflush(stdout); 
  chillfunc =  localFD;
  
  //fflush(stdout);
  //debug_fprintf(stderr, "printing whole file\n"); 
  //fprintf(stdout, "\n\n" );   fflush(stdout);
  //entire_file_AST->print();
  //entire_file_AST->dump(); 
  //fflush(stdout);
  
  //debug_fprintf(stderr, "need to create symbol tables?\n"); 
  symtab2_ = func->get_definition()->get_symbol_table();
  symtab3_ = func->get_definition()->get_body()->get_symbol_table();
  
  //debug_fprintf(stderr, "\nir_rose.cc, calling new CG_chillBuilder()\n"); 
  ocg_ = new omega::CG_chillBuilder(entire_file_AST, chillfunc); // transition - use chillAST based builder
  
  i_ = 0; /*i_ handling may need revision */
  
  free(argv[1]);
  free(argv[0]);
  
}

IR_roseCode::~IR_roseCode() {
}

void IR_roseCode::finalizeRose() {
  
  debug_fprintf(stderr, "IR_roseCode::finalizeRose()\n"); 
  // Moved this out of the deconstructor
  // ????
  //SgProject* project = (IR_roseCode_Global_Init::Instance(NULL))->project;
  SgProject* project = OneAndOnlySageProject; 
  // -- Causes coredump. commented out for now -- //
  // processes attributes left in Rose Ast
  //postProcessRoseCodeInsertion(project);
  
  debug_fprintf(stderr, "printing as part of the destructor??\n"); 
  //project->unparse();
  //backend((IR_roseCode_Global_Init::Instance(NULL))->project);
  
  debug_fprintf(stderr, "IR_roseCode::finalizeRose() before cleanup\n"); 
  chillfunc->print(); 
  
  // clean up a bit (TODO this is poorly implemented)
  chillfunc->constantFold(); 
  chillfunc->cleanUpVarDecls(); 
  
  debug_fprintf(stderr, "IR_roseCode::finalizeRose() after cleanup\n"); 
  chillfunc->print(); 
}

std::vector<IR_ScalarRef *> IR_roseCode::FindScalarRef(const omega::CG_outputRepr *repr) const {
  std::vector<IR_ScalarRef *> scalars;
  
  //debug_fprintf(stderr, "IR_roseCode::FindScalarRef()\n"); 
  //debug_fprintf(stderr, "looking for scalar variables in \n"); 
  //repr->dump();
  
  CG_chillRepr *CR = (CG_chillRepr *) repr;
  chillAST_node * chillcode = CR->GetCode();
  //chillcode->dump(); fflush(stdout); 
  
  //vector<chillAST_VarDecl*> decls;
  //chillcode-> gatherVarLHSUsage(decls);
  //chillcode-> gatherVarUsage(decls);
  
  vector<chillAST_DeclRefExpr*> refs;
  chillcode-> gatherDeclRefExprs(refs);
  
  int numdecls = refs.size(); 
  //debug_fprintf(stderr, "found %d variables set in that code\n", numdecls); 
  for (int i=0; i<numdecls; i++) { 
    //refs[i]->print(); printf("\n"); fflush(stdout); 
    
    // create a IR_ScalarRef for each vaiable ?
    IR_roseScalarRef *r = new IR_roseScalarRef( this, refs[i] ); 
    scalars.push_back( r ); 
  }
  
  return scalars;
  
  
}

bool IR_roseCode::parent_is_array(IR_ArrayRef *a) { 
  chillAST_ArraySubscriptExpr* ASE = ((IR_roseArrayRef *)a)->chillASE;
  chillAST_node *p = ASE->getParent();
  if (!p) return false;
  return p->isArraySubscriptExpr();
}

IR_Block *IR_roseCode::GetCode() const {
  debug_fprintf(stderr, "IR_roseCode::GetCode()\n"); 
  IR_chillBlock *chillblock =  new IR_chillBlock(this, chillfunc );
  return chillblock;
}

IR_OPERATION_TYPE IR_roseCode::QueryExpOperation(const omega::CG_outputRepr *repr) const {
  debug_fprintf(stderr, "IR_roseCode::QueryExpOperation()\n");
  
  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  chillAST_node *firstnode = crepr->chillnodes[0];
  //debug_fprintf(stderr, "chillAST node type %s\n", firstnode->getTypeString());
  //firstnode->print(0,stdout); debug_fprintf(stderr, "\n"); 
  //firstnode->dump(0, stdout); debug_fprintf(stderr, "\n"); 

  chillAST_node *node = firstnode;
  if (node->isArraySubscriptExpr()) { 
    debug_fprintf(stderr, "IR_roseCode::QueryExpOperation() returning IR_OP_ARRAY_VARIABLE\n"); 
    return  IR_OP_ARRAY_VARIABLE;
  }
  else if (node->isUnaryOperator()) {
    char *opstring;
    opstring= ((chillAST_UnaryOperator*)node)->op; // TODO enum
    
    //debug_fprintf(stderr, "opstring '%s'\n", opstring);  
    if (!strcmp(opstring, "+"))  return IR_OP_POSITIVE;
    if (!strcmp(opstring, "-"))  return IR_OP_NEGATIVE;
    debug_fprintf(stderr, "ir_rose.cc  IR_roseCode::QueryExpOperation() UNHANDLED Binary Operator op type (%s)\n", opstring); 
    exit(-1);
  }
  else if (node->isBinaryOperator()) {
    char *opstring;
    opstring= ((chillAST_BinaryOperator*)node)->op; // TODO enum
    
    //debug_fprintf(stderr, "opstring '%s'\n", opstring);  
    if (!strcmp(opstring, "+"))  return IR_OP_PLUS;
    if (!strcmp(opstring, "-"))  return IR_OP_MINUS;
    if (!strcmp(opstring, "*"))  return IR_OP_MULTIPLY;
    if (!strcmp(opstring, "/"))  return IR_OP_DIVIDE;
    if (!strcmp(opstring, "="))  return IR_OP_ASSIGNMENT;
    if (!strcmp(opstring, "+=")) return IR_OP_PLUS_ASSIGNMENT;
    if (!strcmp(opstring, "==")) return IR_OP_EQ;
    if (!strcmp(opstring, "!=")) return IR_OP_NEQ;
    if (!strcmp(opstring, ">=")) return IR_OP_GE;
    if (!strcmp(opstring, "<=")) return IR_OP_LE;
    if (!strcmp(opstring, "%"))  return IR_OP_MOD;
    
    debug_fprintf(stderr, "ir_rose.cc  IR_roseCode::QueryExpOperation() UNHANDLED Binary Operator op type (%s)\n", opstring); 
    exit(-1);
  }
  
  // really need to be more rigorous than this hack  // TODO 
  //if (firstnode->isImplicitCastExpr()) node = ((chillAST_ImplicitCastExpr*)firstnode)->subexpr;
  //if (firstnode->isCStyleCastExpr())   node = ((chillAST_CStyleCastExpr*)  firstnode)->subexpr;
  //if (firstnode->isParenExpr())        node = ((chillAST_ParenExpr*)       firstnode)->subexpr;
  node = firstnode->findref(); 
  //debug_fprintf(stderr, "node type is %s\n", node->getTypeString()); 
  
  if (node->isIntegerLiteral() || node->isFloatingLiteral()) {
    debug_fprintf(stderr, "ir_rose.cc  return IR_OP_CONSTANT\n"); 
    return IR_OP_CONSTANT; // but node may be one of the above operations ... ??
  }
  else if (node->isDeclRefExpr() ) { 
    //node->print(0, stderr); debug_fprintf(stderr, "\n"); 
    debug_fprintf(stderr, "return  IR_OP_VARIABLE  ??\n"); 
    return  IR_OP_VARIABLE; // ?? 
  }
  //else if (node->is ) return  something;
  else { 
    debug_fprintf(stderr, "IR_roseCode::QueryExpOperation()  UNHANDLED NODE TYPE %s\n", node->getTypeString());
    exit(-1); 
  }
  
}



IR_CONDITION_TYPE IR_roseCode::QueryBooleanExpOperation( const omega::CG_outputRepr *repr) const {
  //debug_fprintf(stderr, "IR_roseCode::QueryBooleanExpOperation()\n"); 
  // repr should be a CG_chillRepr
  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  chillAST_node *firstnode = crepr->chillnodes[0];
  //debug_fprintf(stderr, "chillAST node type %s\n", firstnode->getTypeString());
  //firstnode->print(); printf("\n"); fflush(stdout); 
  
  if (firstnode->isBinaryOperator()) { // the usual case 
    chillAST_BinaryOperator* BO = ( chillAST_BinaryOperator* ) firstnode;
    const char *op = BO->op;
    
    if (!strcmp("<", op))  return IR_COND_LT;
    if (!strcmp("<=", op)) return IR_COND_LE;
    
    if (!strcmp(">", op))  return IR_COND_GT;
    if (!strcmp(">=", op)) return IR_COND_GE;
    
    if (!strcmp("==", op)) return IR_COND_EQ;
    if (!strcmp("!=", op)) return IR_COND_NE;
  }
  
  debug_fprintf(stderr, "IR_roseCode::QueryBooleanExpOperation() not a binop: %s\n", firstnode->getTypeString());
  printf("\n\n"); firstnode->print(); printf("\n"); fflush(stdout); 
  return IR_COND_UNKNOWN; // what about if (0),  if (1)  etc? 
}




std::vector<omega::CG_outputRepr *> IR_roseCode::QueryExpOperand(const omega::CG_outputRepr *repr) const { 
  //debug_fprintf(stderr, "IR_roseCode::QueryExpOperAND()\n"); 
  std::vector<omega::CG_outputRepr *> v;
  
  CG_chillRepr *crepr = (CG_chillRepr *) repr; 
  
  chillAST_node *e = crepr->chillnodes[0]; // ?? 
  //e->print(); printf("\n"); fflush(stdout); 
  
  // really need to be more rigorous than this hack  // TODO 
  if (e->isImplicitCastExpr()) e = ((chillAST_ImplicitCastExpr*)e)->subexpr;
  if (e->isCStyleCastExpr())   e = ((chillAST_CStyleCastExpr*)  e)->subexpr;
  if (e->isParenExpr())        e = ((chillAST_ParenExpr*)       e)->subexpr;
  
  
  //if(isa<IntegerLiteral>(e) || isa<FloatingLiteral>(e) || isa<DeclRefExpr>(e)) {
  if (e->isIntegerLiteral() || e->isFloatingLiteral() || e->isDeclRefExpr() ) { 
    //debug_fprintf(stderr, "it's a constant\n"); 
    omega::CG_chillRepr *repr = new omega::CG_chillRepr(e);
    v.push_back(repr);
  } else if (e->isBinaryOperator()) { 
    //debug_fprintf(stderr, "binary\n"); 
    
    chillAST_BinaryOperator *bop = (chillAST_BinaryOperator*)e;
    char *op = bop->op;  // TODO enum for operator types
    if (streq(op, "=")) { 
      v.push_back(new omega::CG_chillRepr( bop->rhs ));  // for assign, return RHS
    }
    else if (streq(op, "+") || streq(op, "-") || streq(op, "*") || streq(op, "/") || streq(op, "%") || 
             streq(op, "==") || streq(op, "!=") || 
             streq(op, "<") || streq(op, "<=") ||                        
             streq(op, ">") || streq(op, ">=")
             
      ) {
      //debug_fprintf(stderr, "op\n"); 
      v.push_back(new omega::CG_chillRepr( bop->lhs ));  // for +*-/ == return both lhs and rhs
      v.push_back(new omega::CG_chillRepr( bop->rhs )); 
    }
    else { 
      debug_fprintf(stderr, "ir_rose.cc  IR_roseCode::QueryExpOperand() Binary Operator  UNHANDLED op (%s)\n", op); 
      exit(-1);
    }
  } // BinaryOperator
  
  else if  (e->isUnaryOperator()) { 
    //debug_fprintf(stderr, "unary\n"); 
    omega::CG_chillRepr *repr;
    chillAST_UnaryOperator *uop = (chillAST_UnaryOperator*)e;
    char *op = uop->op; // TODO enum
    if (streq(op, "+") || streq(op, "-")) {
      v.push_back( new omega::CG_chillRepr( uop->subexpr ));
    }
    else { 
      debug_fprintf(stderr, "ir_rose.cc  IR_roseCode::QueryExpOperand() Unary Operator  UNHANDLED op (%s)\n", op); 
      exit(-1);
    }
  } // unaryoperator
  
  else if (e->isArraySubscriptExpr() ) { 
    //debug_fprintf(stderr, "operand is ArraySubscriptExpr()\n"); e->print(); printf("\n"); fflush(stdout);
    v.push_back(new omega::CG_chillRepr( ((chillAST_ArraySubscriptExpr*)e) )); 
    //debug_fprintf(stderr, "Array ref\n"); 
  }
  else { 
    debug_fprintf(stderr, "ir_rose.cc  IR_roseCode::QueryExpOperand() UNHANDLED node type %s\n", e->getTypeString()); 
    exit(-1); 
  }
  //debug_fprintf(stderr, "IR_roseCode::QueryExpOperand() DONE\n"); 
  return v;
}

// things in manu/anand rosecode that need to be done using chillAST
// TODO this seems no different that createarrayref
IR_PointerArrayRef *IR_roseCode::CreatePointerArrayRef(IR_PointerSymbol *sym,
                                                       std::vector<omega::CG_outputRepr *> &index)
{
  debug_fprintf(stderr, "IR_roseCode::CreatePointerArrayRef()\n");
  debug_fprintf(stderr, "symbol name %s    and %d dimensions\n", sym->name().c_str(), sym->n_dim()); 
  //debug_fprintf(stderr, "use createArrayRef instead, they are the same!\n");
  //die(); 

  IR_rosePointerSymbol *RPS = (IR_rosePointerSymbol *)sym;  // chill?
  chillAST_VarDecl *base = RPS->chillvd;
  
  
  std::vector<chillAST_node *> indeces;
  for (int i = 0; i < index.size(); i++) {
    omega::CG_chillRepr *CR = (omega::CG_chillRepr *)index[i]; 
    chillAST_node *chillcode = CR->GetCode(); 
    chillcode->print(0,stderr); debug_fprintf(stderr, "\n");
    indeces.push_back( chillcode ); // TODO error check
  }
  
  chillAST_ArraySubscriptExpr *ASE = new chillAST_ArraySubscriptExpr( base, indeces, NULL);
  return new IR_rosePointerArrayRef( this, ASE,  0); // 0 means not a write so far  
}



void IR_roseCode::CreateDefineMacro(std::string s, 
                                    std::string args,  
                                    omega::CG_outputRepr *repr)
{
  debug_fprintf(stderr, "ir_rose.cc  *IR_roseCode::CreateDefineMacro( string string repr)\n");
  omega::CG_chillRepr *CR = (omega::CG_chillRepr *)repr;
  vector<chillAST_node*> astvec = CR->getChillCode();
  
  chillAST_node *output;
  if (1 < astvec.size()) { 
    // make a compound node?
    debug_fprintf(stderr, " IR_roseCode::CreateDefineMacro(), more than one ast???\n");
    die(); 
  }
  else output = astvec[0]; 
  
  debug_fprintf(stderr, "#define %s%s ", s.c_str(), args.c_str()); 
  debug_fprintf(stderr, "IR_roseCode::CreateDefineMacro(), CR chillnodes:\n"); 
  CR->printChillNodes(); printf("\n"); fflush(stdout); 
  debug_fprintf(stderr, "IR_roseCode::CreateDefineMacro(), CR chillnodes DONE\n"); 
  
  //what do we want ast for the macro to look like? 
  //debug_fprintf(stderr, "entire_file_AST %p\n", entire_file_AST); 
  chillAST_MacroDefinition * macro = new  chillAST_MacroDefinition( s.c_str(), entire_file_AST); // NULL); 
  //debug_fprintf(stderr, "args: '%s'\n", args.c_str()); 
  //debug_fprintf(stderr, "output is of type %s\n", output->getTypeString());
  //macro->addChild( output ); // setBody?
  
  debug_fprintf(stderr, "ir_rose.cc  IR_roseCode::CreateDefineMacro() adding macro to sourcefile\n"); 
  entire_file_AST->addMacro( macro ); // ?? 
  defined_macros.insert(std::pair<std::string, chillAST_node*>(s + args, output)); 
  
  
  // TODO  ALSO put the macro into the SourceFile, so it will be there if that AST is printed
  // TODO one of these should probably go away
  //debug_fprintf(stderr, "entire file had %d children\n",  entire_file_AST->children.size()); 
  entire_file_AST->insertChild(0, macro); 
  //debug_fprintf(stderr, "entire file has %d children\n",  entire_file_AST->children.size()); 
  return;
}



void IR_roseCode::CreateDefineMacro(std::string s, 
                                    std::vector<std::string> args,  
                                    omega::CG_outputRepr *repr)
{
  //debug_fprintf(stderr, "ir_rose.cc *IR_roseCode::CreateDefineMacro( string, VECTOR, repr )\n");
  
  omega::CG_chillRepr *CR = (omega::CG_chillRepr *)repr;
  vector<chillAST_node*> astvec = CR->getChillCode();
  
  if (1 < astvec.size()) { 
    // make a compound node?
    debug_fprintf(stderr, " IR_roseCode::CreateDefineMacro(), more than one ast???\n");
    die(); 
  }
  chillAST_node *sub = astvec[0]; // the thing we'll sub into
  //debug_fprintf(stderr, "sub is of type %s\n", sub->getTypeString()); 
  
  //chillAST_UnaryOperator *unary = new chillAST_UnaryOperator( "*", true, sub, entire_file_AST); // macro parent ?? 
  
  //debug_fprintf(stderr, "#define %s", s.c_str()); 
  //if (args.size()) { 
  //  debug_fprintf(stderr, "( ");
  //  for (int i=0; i<args.size(); i++) { 
  //    if (i) debug_fprintf(stderr, ", ");
  //    debug_fprintf(stderr, "%s", args[i].c_str()); 
  //  }
  //  debug_fprintf(stderr, " )"); 
  //} 
  //debug_fprintf(stderr, "   ");
  //sub->print(); printf("\n\n"); fflush(stdout);  // the body of the macro
  //sub->dump();  printf("\n\n"); fflush(stdout); 
  
  // make the things in the output actually reference the (fake) vardecls we created for the args, so that we can do substitutions later
  
  //what do we want ast for the macro to look like? 
  //debug_fprintf(stderr, "IR_Rosecode entire_file_AST %p\n",  entire_file_AST);
  chillAST_MacroDefinition * macro = new  chillAST_MacroDefinition( s.c_str(), entire_file_AST); // NULL); 
  
  
  // create "parameters" for the #define
  for (int i=0; i<args.size(); i++) { 
    //debug_fprintf(stderr, "'parameter' %s\n", args[i].c_str()); 
    chillAST_VarDecl *vd = new chillAST_VarDecl( "fake", args[i].c_str(), "", NULL); 
    //debug_fprintf(stderr, "adding parameter %d ", i); vd->dump(); fflush(stdout); 
    macro->addParameter( vd );
    
    // find the references to this name in output // TODO 
    // make them point to the vardecl ..
    
  }
  
  macro->setBody( sub ); 
  
  //debug_fprintf(stderr, "macro body is:\nprint()\n"); 
  //sub->print(); printf("\ndump()\n"); fflush(stdout); 
  //sub->dump();  printf("\n"); fflush(stdout);
  
  
  defined_macros.insert(std::pair<std::string, chillAST_node*>(s /* + args */, sub));
  
  // TODO  ALSO put the macro into the SourceFile, so it will be there if that AST is printed
  // TODO one of these should probably go away
  //debug_fprintf(stderr, "entire file had %d children\n",  entire_file_AST->children.size()); 
  entire_file_AST->insertChild(0, macro); 
  //debug_fprintf(stderr, "entire file has %d children\n",  entire_file_AST->children.size()); 
  return;
}





void IR_roseCode::CreateDefineMacro(std::string s,std::string args, std::string repr)
{
  debug_fprintf(stderr, "IR_roseCode::CreateDefine Macro 2( string string string )\n");
  die(); 
  exit(-1); 
}


omega::CG_outputRepr *IR_roseCode::CreateArrayType(IR_CONSTANT_TYPE type, omega::CG_outputRepr* size)
{
  debug_fprintf(stderr, "IR_roseCode::CreateArrayType()   NOT IMPLEMENTED\n");
  //switch (type):  BUH 
  //  case IR_CONSTANT
  //chillAST_VarDecl *vd = new chillAST_VarDecl(
  die(); 
}

omega::CG_outputRepr *IR_roseCode::CreatePointerType(IR_CONSTANT_TYPE type) // why no name???
{
  //debug_fprintf(stderr, "IR_roseCode::CreatePointerType( type )\n");
  const char *typestr = irTypeString( type ); 
  
  // pointer to something, not named  
  // ast doesnt' have a type like this, per se. TODO 
  // Use a variable decl with no name? TODO 
  chillAST_VarDecl *vd = new chillAST_VarDecl( typestr, "", "", NULL);
  vd->numdimensions = 1;
  vd->knownArraySizes = false;
  
  omega::CG_chillRepr *CR = new omega::CG_chillRepr( vd ); 
  return CR; 
}

omega::CG_outputRepr *IR_roseCode::CreatePointerType(omega::CG_outputRepr *type)
{
  debug_fprintf(stderr, "IR_roseCode::CreatePointerType ( CG_outputRepr *type )\n");
  die();
  exit(-1); 
}

omega::CG_outputRepr *IR_roseCode::CreateScalarType(IR_CONSTANT_TYPE type)
{
  debug_fprintf(stderr, "IR_roseCode::CreateScalarType() 1\n");
  const char *typestr = irTypeString( type ); 

  // Use a variable decl with no name? TODO 
  chillAST_VarDecl *vd = new chillAST_VarDecl( typestr, "", "", NULL);
  omega::CG_chillRepr *CR = new omega::CG_chillRepr( vd ); 
  return CR;   
}


bool IR_roseCode::FromSameStmt(IR_ArrayRef *A, IR_ArrayRef *B)
{
  // see if 2 array references are in the same statement (?)
  chillAST_ArraySubscriptExpr* a = ((IR_roseArrayRef *)A)->chillASE;
  chillAST_ArraySubscriptExpr* b = ((IR_roseArrayRef *)B)->chillASE;
  
  //debug_fprintf(stderr, " IR_roseCode::FromSameStmt()\n");
  //a->print(); printf("\n"); 
  //b->print(); printf("\n");  fflush(stdout); 
  
  if (a == b) { 
    //debug_fprintf(stderr, "trivially true because they are exactly the same statement\n"); 
    return true;
  }
  
  chillAST_node *AE = a->getEnclosingStatement();
  chillAST_node *BE = b->getEnclosingStatement();
  //AE->print(); printf("\n"); 
  //BE->print(); printf("\n");  fflush(stdout); 
  return(AE == BE);
}

void IR_roseCode::printStmt(const omega::CG_outputRepr *repr)
{
  debug_fprintf(stderr, "IR_roseCode:: printStmt()\n");
  die();
  exit(-1); 
}

int IR_roseCode::getStmtType(const omega::CG_outputRepr *repr)
{
  // this seems to be 1 == a single statement.
  //  sigh
  
  chillAST_node *n = ((CG_chillRepr *)repr)->GetCode();
  //n->print(); printf("\n"); fflush(stdout);
  //debug_fprintf(stderr, "%s\n", n->getTypeString()); 
  
  if (n->isBinaryOperator()) {
    //debug_fprintf(stderr, "IR_roseCode::getStmtType() returning 1\n"); 
    return 1;
  }
  if (n->isCompoundStmt()) {
    //debug_fprintf(stderr, "IR_roseCode::getStmtType() returning 0\n"); 
    return 0; 
  }
  debug_fprintf(stderr, "IR_roseCode::getStmtType () bailing\n");
  die(); 
}

IR_OPERATION_TYPE IR_roseCode::getReductionOp(const omega::CG_outputRepr *repr)
{
  //debug_fprintf(stderr, "IR_roseCode::getReductionOp()\n");
  chillAST_node *n = ((CG_chillRepr *)repr)->GetCode();
  //debug_fprintf(stderr, "%s\n", n->getTypeString()); 
  //n->print(); printf("\n"); fflush(stdout);
  
  if (n->isBinaryOperator()) { 
    return  QueryExpOperation( repr );  // TODO chillRepr
  }
  
  debug_fprintf(stderr, "IR_roseCode::getReductionOp()\n");
  die();
}

IR_Control *  IR_roseCode::FromForStmt(const omega::CG_outputRepr *repr)
{
  debug_fprintf(stderr, "IR_roseCode::FromForStmt()\n");
  die();
}


IR_Control* IR_roseCode::GetCode(omega::CG_outputRepr* repr) const // what is this ??? 
{
  debug_fprintf(stderr, "IR_roseCode::GetCode(CG_outputRepr*)\n");

  omega::CG_chillRepr* CR = (omega::CG_chillRepr* ) repr;
  chillAST_node *chillcode = CR->GetCode();
  chillcode->print(0,stderr); debug_fprintf(stderr, "\n\n");

  // this routine is supposed to return an IR_Control. 
  // that can be one of 3 things: if, loop, or block
  debug_fprintf(stderr, "chillcode is a %s\n", chillcode->getTypeString()); 
  if (chillcode->isIfStmt()) { 
    return new IR_chillIf( this, chillcode ); 
  }
  if (chillcode->isLoop()) {  // ForStmt
    return new IR_chillLoop( this, (chillAST_ForStmt *)chillcode );
  }
  if (chillcode->isCompoundStmt()) { 
    return new IR_chillBlock( this, (chillAST_CompoundStmt *)chillcode );
  }

  // anything else just wrap it in a compound stmt ???  TODO 

  
  debug_fprintf(stderr, " IR_roseCode::GetCode( repr ),  chillcode is a %s\nDIE\n", chillcode->getTypeString()); 
  die();
  exit(0); 
}


// Manu:: replaces the RHS with a temporary array reference IN PLACE - part of scalar expansion
bool  IR_roseCode::ReplaceRHSExpression(omega::CG_outputRepr *code, IR_Ref *ref){
  //debug_fprintf(stderr, "IR_roseCode::ReplaceRHSExpression()\n");

  // make sure the code has just one statement and that it is an assignment(?)
  CG_chillRepr * CR = (CG_chillRepr * ) code;
  int numnodes = CR->chillnodes.size();

  //debug_fprintf(stderr, "%d chillAST nodes\n", numnodes);
  for (int i=0; i<numnodes; i++) { 
    CR->chillnodes[i]->print(0,stderr); debug_fprintf(stderr, "\n"); 
  }

  if (numnodes == 1) { 
    chillAST_node *nodezero = CR->chillnodes[0];
    if (nodezero-> isAssignmentOp()) {
      chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *)nodezero;

      omega::CG_chillRepr *RR =  (omega::CG_chillRepr *)(ref->convert());
      chillAST_node * n = RR->GetCode(); 
      BO->setRHS(  n );  // replace in place 
      //debug_fprintf(stderr, "binary op with replaced RHS is now\n");
      //BO->print(0,stderr); debug_fprintf(stderr, "\n"); 
      return true;
    }
    debug_fprintf(stderr, "IR_roseCode::ReplaceRHSExpression()  trying to replace the RHS of something that is not an assignment??\n");
    nodezero->print(0,stderr); debug_fprintf(stderr, "\n"); 
  }
  else { 
    debug_fprintf(stderr, "IR_roseCode::ReplaceRHSExpression()  trying to replace the RHS of more than one node ???\n");
  }
  return false; // ?? 

}


bool  IR_roseCode::ReplaceLHSExpression(omega::CG_outputRepr *code, IR_ArrayRef *ref){
  debug_fprintf(stderr, "IR_roseCode::ReplaceLHSExpression()\n");

  // make sure the code has just one statement and that it is an assignment(?)
  CG_chillRepr * CR = (CG_chillRepr * ) code;
  int numnodes = CR->chillnodes.size();
  debug_fprintf(stderr, "%d chillAST nodes\n", numnodes);
  for (int i=0; i<numnodes; i++) { 
    CR->chillnodes[i]->print(0,stderr); debug_fprintf(stderr, "\n"); 
  }

  if (numnodes == 1) { 
    chillAST_node *nodezero = CR->chillnodes[0];
    if (nodezero-> isAssignmentOp()) {
      return new CG_chillRepr(  ((chillAST_BinaryOperator *) nodezero)->rhs ); // clone??
    }
    debug_fprintf(stderr, "IR_roseCode::GetRHSExpression()  trying to find the RHS of something that is not an assignment??\n");
    nodezero->print(0,stderr); debug_fprintf(stderr, "\n"); 
  }
  else { 
    debug_fprintf(stderr, "IR_roseCode::GetRHSExpression()  trying to find the RHS of more than one node ???\n");
  }


  die(); 
  exit(-1);
}



// replaces the RHS with a temporary array reference - part of scalar expansion
omega::CG_outputRepr *  IR_roseCode::GetRHSExpression(omega::CG_outputRepr *code){
  //debug_fprintf(stderr, "IR_roseCode::GetRHSExpression()\n");

  // make sure the code has just one statement and that it is an assignment(?)
  CG_chillRepr * CR = (CG_chillRepr * ) code;
  int numnodes = CR->chillnodes.size();
  debug_fprintf(stderr, "%d chillAST nodes\n", numnodes);
  for (int i=0; i<numnodes; i++) { 
    CR->chillnodes[i]->print(0,stderr); debug_fprintf(stderr, "\n"); 
  }

  if (numnodes == 1) { 
    chillAST_node *nodezero = CR->chillnodes[0];
    if (nodezero-> isAssignmentOp()) {
      return new CG_chillRepr(  ((chillAST_BinaryOperator *) nodezero)->rhs ); // clone??
    }
    debug_fprintf(stderr, "IR_roseCode::GetRHSExpression()  trying to find the RHS of something that is not an assignment??\n");
    nodezero->print(0,stderr); debug_fprintf(stderr, "\n"); 
  }
  else { 
    debug_fprintf(stderr, "IR_roseCode::GetRHSExpression()  trying to find the RHS of more than one node ???\n");
  }

  die(); 
}



omega::CG_outputRepr *  IR_roseCode::GetLHSExpression(omega::CG_outputRepr *code){
  debug_fprintf(stderr, "IR_roseCode::GetLHSExpression()\n");
  // make sure the code has just one statement and that it is an assignment(?)
  CG_chillRepr * CR = (CG_chillRepr * ) code;
  int numnodes = CR->chillnodes.size();
  debug_fprintf(stderr, "%d chillAST nodes\n", numnodes);
  for (int i=0; i<numnodes; i++) { 
    CR->chillnodes[i]->print(0,stderr); debug_fprintf(stderr, "\n"); 
  }

  if (numnodes == 1) { 
    chillAST_node *nodezero = CR->chillnodes[0];
    if (nodezero-> isAssignmentOp()) {
      return new CG_chillRepr(  ((chillAST_BinaryOperator *) nodezero)->lhs ); // clone??
    }
    debug_fprintf(stderr, "IR_roseCode::GetLHSExpression()  trying to find the LHS of something that is not an assignment??\n");
    nodezero->print(0,stderr); debug_fprintf(stderr, "\n"); 
  }
  else { 
    debug_fprintf(stderr, "IR_roseCode::GetLHSExpression()  trying to find the LHS of more than one node ???\n");
  }

  die(); 
  exit(-1);
}


omega::CG_outputRepr *IR_roseCode::CreateMalloc(const IR_CONSTANT_TYPE type, 
                                                std::string lhs, // this is the variable to be assigned the new mwmory! 
                                                omega::CG_outputRepr * size_repr){
  
  debug_fprintf(stderr, "IR_roseCode::CreateMalloc 1()\n");
  char *typ = irTypeString( type );
  debug_fprintf(stderr, "malloc  %s %s \n", typ, lhs.c_str()); 
  
  chillAST_node *siz = ((CG_chillRepr *)size_repr)->GetCode(); 
  //siz->print(0,stderr); debug_fprintf(stderr, "\n");

  chillAST_Malloc* mal = new chillAST_Malloc( typ, siz ); // malloc( sizeof(int) * 248 )   ... no parent
  // this is how it should be 
  // return new CG_chillRepr( mal ); 


  // the rest of this function should not be here 
  chillAST_CStyleCastExpr *CE = new chillAST_CStyleCastExpr( typ, mal );
  // we only have the name of a variable to assign the malloc memory to. Broken
  chillAST_VarDecl *vd = new chillAST_VarDecl( typ, lhs.c_str(), "*", NULL );
  chillAST_BinaryOperator *BO = new chillAST_BinaryOperator( vd, "=", CE );
  BO->print(0, stderr); 
  return new CG_chillRepr( BO ); 
  


}


omega::CG_outputRepr *IR_roseCode::CreateMalloc (omega::CG_outputRepr *type, std::string lhs,
                                                 omega::CG_outputRepr * size_repr) {
  debug_fprintf(stderr, "IR_roseCode::CreateMalloc 2()\n");
  die(); 
  exit(-1);
}

omega::CG_outputRepr *IR_roseCode::CreateFree(  omega::CG_outputRepr *exp){
  debug_fprintf(stderr, "IR_roseCode::CreateFree()\n");
  die(); 
  exit(-1);
}



#endif 
