
#include "ir_chill.hh"
#include "chill_io.hh"
#include "stencil.hh"


bool isLoopConstant( chillAST_node *node) {  // pretty inefficient
  vector<chillAST_ArraySubscriptExpr*> refs;
  node->gatherArrayRefs( refs, false );
  return 0 == refs.size();
}



void stencilInfo::walktree( chillAST_node *node,
               vector< chillAST_node * > &coeffstohere ) { 
  //class stencilInfo &SI /* the output */ ) { 
  
  //debug_fprintf(stderr, "walktree %s ", node->getTypeString()); 
  //node->print(); printf("\n"); fflush(stdout);

  if (node->isArraySubscriptExpr()) {
    // here we've encountered an array subscript expression
    // the coefficients are on a stack.
    chillAST_ArraySubscriptExpr * ASE = (chillAST_ArraySubscriptExpr *)node;
    //debug_fprintf(stderr, "\nASE "); ASE->print(); printf("\n"); fflush(stdout);
    vector <chillAST_node *> coeffs;
    //debug_fprintf(stderr, "coefs:  ");
    for (int i=0; i< coeffstohere.size(); i++) { 
      coeffs.push_back( coeffstohere[i]->clone() ); 

      //bool iscomplicated = coeffstohere[i]->isBinaryOperator();
      //if (iscomplicated) printf("("); 
      //coeffstohere[i]->print(); 
      //if (iscomplicated) printf(")"); 
      //if ( (i+1) < coeffstohere.size()) printf(" * ");
    }
    //printf("\n"); fflush(stdout); 
    coefficients.push_back( coeffs ); 



    vector< chillAST_node * > ind;
    ASE->gatherIndeces( ind );
    int numindeces = ind.size();
    if (numindeces != dimensions) { 
      debug_fprintf(stderr, "src array has %d indeces, stencil has %d\n", numindeces, dimensions); 
    }
    

    vector< int > offsetstostore;
    for (int i=0; i<numindeces; i++) { 
      //ind[i]->print(); printf("\n"); fflush(stdout); 
      
      // verify that the index is using the correct variable
      vector<chillAST_VarDecl*> vds;
      ind[i]->gatherScalarVarDecls( vds );  // gather vardecls 
      int numvd = vds.size();
      
      if (numvd != 1  || vds[0] != indexVariables[i]) { 
        debug_fprintf(stderr, "index %d has %d variables\n", i, numvd);
        ind[i]->print(); printf("\n"); fflush(stdout);
        if (numvd > 0) debug_fprintf(stderr, "variable %s vs %s\n", vds[0]->varname, indexVariables[i]->varname );
        exit(-1);
      }
      //debug_fprintf(stderr, "variable %s is %s\n", vds[0]->varname, indexVariables[i]->varname );

      //debug_fprintf(stderr, "%s  ", ind[i]->getTypeString()); 
      //ind[i]->print(); printf("\n");

      int offset = 0;
      if ( ind[i]->isDeclRefExpr()) {} // just the variable, offset is 0
      if ( ind[i]->isBinaryOperator()) { 
        chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *) ind[i];
        //debug_fprintf(stderr, "LR %s %s\n", BO->lhs->getTypeString(),BO->rhs->getTypeString() ); 

        if (BO->isPlusOp()) {
          if (BO->lhs->isDeclRefExpr()) offset = BO->rhs->evalAsInt();
          else  if (BO->rhs->isDeclRefExpr()) offset = BO->lhs->evalAsInt();
          else { 
            debug_fprintf(stderr, "expecting index to be var + int\n");
            BO->print(); printf("\n"); fflush(stdout);
            exit(-1);
          }
        }
        else if (BO->isMinusOp()) {
          if (BO->lhs->isDeclRefExpr()) offset = -BO->rhs->evalAsInt(); // NEGATIVE
          else { 
            debug_fprintf(stderr, "expecting index to be var - int\n");
            BO->print(); printf("\n"); fflush(stdout);
            exit(-1);
          }
        }
        else { 
          debug_fprintf(stderr, "unhandled index expression: ");
          BO->print(); printf("\n"); fflush(stdout);
          exit(-1);
        }
      }
      //debug_fprintf(stderr, "offset %d\n\n", offset); 
      offsetstostore.push_back( offset );
      // OK, now see if this is outside the rage we already know about
      if (offset < minOffset[ i ]) minOffset[ i ] = offset;
      if (offset > maxOffset[ i ]) maxOffset[ i ] = offset;
    }
    offsets.push_back( offsetstostore ); 
  }
  else if (node->isBinaryOperator()) { 
    chillAST_BinaryOperator *BO = (chillAST_BinaryOperator *) node;

    bool leftisconst  =  isLoopConstant(BO->lhs);
    bool rightisconst =  isLoopConstant(BO->rhs);

    if (leftisconst && !rightisconst) {
      coeffstohere.push_back(BO->lhs ->clone());
      walktree( BO->rhs, coeffstohere);
      coeffstohere.pop_back();
    }
    else if (rightisconst && !leftisconst) {
      coeffstohere.push_back(BO->rhs ->clone());
      walktree( BO->lhs, coeffstohere);
      coeffstohere.pop_back();
    }
    else if  (!rightisconst && !leftisconst) { // neither. just recurse
      walktree( BO->lhs, coeffstohere);
      walktree( BO->rhs, coeffstohere);
    }
    else { // BOTH  should never happen
      debug_fprintf(stderr, "binop has 2 constants??\n");
      BO->print(); printf("\n"); 
      BO->dump();  printf("\n"); 
      fflush(stdout);
      exit(-1);
    }

  }
  else if (node->isParenExpr())
    walktree(static_cast<chillAST_ParenExpr*>(node)->subexpr, coeffstohere);
  else {
    debug_fprintf(stderr, "\n\nnon-constant non BinaryOperator %s?\n", node->getTypeString());
    node->print(); printf("\n\n"); 
    node->dump();  printf("\n\n"); 
    fflush(stdout);
  }

}


stencilInfo::stencilInfo() {
  dimensions = 0; 
  elements = 0;
  srcArrayVariable = NULL;
  dstArrayVariable = NULL;

  minOffset[0] = minOffset[1] = minOffset[2] = 0;  // hardcoded to 3?  TODO 
  maxOffset[0] = maxOffset[1] = maxOffset[2] = 0;
  indexVariables[0] = indexVariables[1] = indexVariables[2] = NULL;
}




stencilInfo::stencilInfo( chillAST_node *topstatement ) { 

  //debug_fprintf(stderr, "\n\nstencil.cc  stencil()\n"); 
  //topstatement->print(); printf("\n\n"); fflush(stdout);


  // warning: hardcoded size of 3
  dimensions = 0; 
  elements = 0;
  srcArrayVariable = NULL;
  dstArrayVariable = NULL;
  minOffset[0] = minOffset[1] = minOffset[2] = 0;  // hardcoded to 3?  TODO 
  maxOffset[0] = maxOffset[1] = maxOffset[2] = 0;
  indexVariables[0] = indexVariables[1] = indexVariables[2] = NULL;

  vector<chillAST_node*> statements;
  if (topstatement->isCompoundStmt()) { 
    statements = ((chillAST_CompoundStmt *) topstatement)->getChildren();
    
    debug_fprintf(stderr, "stencil of %d statements in a compound statement\n", statements.size()); 
  }
  else statements.push_back(topstatement);

  int numstatements = statements.size(); 
  //debug_fprintf(stderr, "\n*** %d statements ***\n\n", numstatements); 


  for (int i=0; i<numstatements; i++) { 
    //debug_fprintf(stderr, "processing statement %d\n", i); 

    chillAST_node* statement = statements[i]; 

    if ( !statement->isBinaryOperator() ) { 
      debug_fprintf(stderr, "top level stencil is not a binop\n");
      statement->print();
      exit(-1);
    }
    
    chillAST_BinaryOperator *binop = (chillAST_BinaryOperator *)statement;
    if ( !binop->isAssignmentOp() ) { 
      debug_fprintf(stderr, "top level stencil calc is not an assignment statement\n");
      binop->print();
      exit(-1);
    }
    
    //debug_fprintf(stderr, "OK, it's an assignment statement, as expected\n");
    vector<chillAST_ArraySubscriptExpr*> lhsarrayrefs; // gather lhs arrayrefs
    binop->lhs->gatherArrayRefs( lhsarrayrefs, 0 );
    //debug_fprintf(stderr, "%d arrayrefs in lhs\n", lhsarrayrefs.size());
    
    // there should be just one. It is the destination of the stencil
    chillAST_node *outvar = NULL;
    if (1 == lhsarrayrefs.size()) { 
      if (NULL == dstArrayVariable) dstArrayVariable = lhsarrayrefs[0]->multibase();
      else { // make sure the statements all have the same dest 
        if (dstArrayVariable != lhsarrayrefs[0]->multibase()) {
          debug_fprintf(stderr, "statement %d of stencil does not have the same dest as previous statements\n"); 
          dstArrayVariable->print(); printf("\n");
          lhsarrayrefs[0]->multibase()->print();  printf("\n\n");
          fflush(stdout);
          exit(-1); 
        }
      }


      //debug_fprintf(stderr, "\n\noutput array variable is "); dstArrayVariable->dump();
      //printf("\n"); fflush(stdout); 
    }
    else { 
      debug_fprintf(stderr, "\n\nlhs has multiple arrayrefs?\n");
      binop->lhs->print(); printf("\n\n"); 
      binop->lhs->dump();  printf("\n\n"); 
    }
    

    vector<chillAST_DeclRefExpr *>lhsrefs; 
    binop->lhs->gatherDeclRefExprs( lhsrefs );  // gather all variable refs on lhs 
    int numdre =  lhsrefs.size(); 
    //debug_fprintf(stderr, "%d declrefs in lhs\n", numdre);
    //for (int i=0; i<numdre; i++) lhsrefs[i]->dump(); 
    
    if (!dimensions) dimensions = numdre-1;
    else { // make sure it's consistent
      if ( dimensions != numdre-1) { 
        debug_fprintf(stderr, "ERROR: stencil dimension was %d now %d\n", dimensions,  numdre-1); 
        exit(-1);
      }
    }

    if (0 == i) { // first time 
      for (int j=1; j<numdre; j++) { // start with second
        chillAST_VarDecl *vd = lhsrefs[j]->getVarDecl();
        indexVariables[j-1] = vd; // should really check that they are not arrays
      }
    }
    else { // check consistency 
      for (int j=1; j<numdre; j++) { // start with second
        chillAST_VarDecl *vd = lhsrefs[j]->getVarDecl();
        if ( indexVariables[j-1] != vd ) { 
          debug_fprintf(stderr, "ERROR statement %d has index variable %s, not %s\n", i, vd->varname, indexVariables[j-1]->varname); 
        }
      }
    }    
    
    
    // look at the rhs of the equation (this can all be commented out. it just prints debug stuff)
    //debug_fprintf(stderr, "\n\n"); 
    //binop->rhs->print(); 
    //debug_fprintf(stderr, "\n\n"); 


    // gather the array refs from the right hand side of the assignment statement into a vector
    vector<chillAST_ArraySubscriptExpr*> refs;
    binop->rhs->gatherArrayRefs( refs, 0 );
    int numarray =  refs.size(); 


    //debug_fprintf(stderr, "%d array refs in rhs\n\n", numarray);
    //for (int i=0; i<numarray; i++) { 
    //  refs[i]->print(); printf("\n"); 
    //  
    //  vector< chillAST_node * > ind;
    //  refs[i]->gatherIndeces( ind );
    //  debug_fprintf(stderr, "%d indeces\n\n", ind.size()); 
    //  for (int j=0; j<ind.size(); j++) { 
    //    ind[j]->print(); printf("\n"); 
    //  }
    //} 
    if (NULL == srcArrayVariable) srcArrayVariable = refs[0]->multibase();
    else { // make sure statements are consistent
      if (srcArrayVariable != refs[0]->multibase()) {
          debug_fprintf(stderr, "statement %d of stencil does not have the same dest as previous statements\n"); 
          srcArrayVariable->print(); printf("\n");
          refs[0]->multibase()->print();  printf("\n\n");
          fflush(stdout);
          exit(-1); 
      }
    }
    
    
    
    vector< chillAST_node * > coeffs;  // none yet
    walktree( binop->rhs, coeffs );  // recursive method 
    
  }

  //print(); // this should probably not be part of the process
  debug_fprintf(stderr, "stencil.cc stencil() DONE\n\n");
}








void stencilInfo::print( FILE *fp) { 
  fprintf(fp, "destination array : %s\n",dstArrayVariable->varname); 
  fprintf(fp, "source array      : %s\n", srcArrayVariable->varname); 
  fprintf(fp, "dimensions        : %d\n\n", dimensions);
  
  fprintf(fp, "Dimension  Variable  MinOffset  MaxOffset  Width\n"); 
  fprintf(fp, "---------  --------  ---------  ---------  -----\n"); 
  for (int i=0; i<dimensions; i++) { 
    //fprintf(fp, "    %d         ",i); indexVariables[i]->print();
    fprintf(fp, "    %d         %s", i, indexVariables[i]->varname); 
    fprintf(fp, "        %3d       %3d       %3d\n", minOffset[i], maxOffset[i], 1+maxOffset[i]-minOffset[i] ); 
    
  }
  
  fprintf(fp, "\n    k    j    i   coefficient\n"); 
  fprintf(fp, " ---- ---- ----   -----------\n"); 
  int numterms = coefficients.size();
  for (int i=0; i<numterms; i++) { 
    for (int j=0; j<dimensions; j++) {
      fprintf(fp, "  %3d", offsets[i][j]);
    }
    
    fprintf(fp, "    ");
    for (int j=0; j<coefficients[i].size(); j++) { 
      bool iscomplicated = coefficients[i][j]->isBinaryOperator();
      if (iscomplicated) fprintf(fp, "("); 
      coefficients[i][j]->print(0, fp);
      if (iscomplicated) fprintf(fp, ")"); 
      if ((j+1) < coefficients[i].size()) fprintf(fp, " * "); 
    }
    fprintf(fp, "\n"); 
  }
  fprintf(fp, "\n"); 
  fflush(stdout);
}


chillAST_node * stencilInfo::find_coefficient( int i, int j, int k ) { 
  int numterms = coefficients.size();
  for (int c=0; c<numterms; c++) { 
    if ( offsets[c][0] == i && 
         offsets[c][1] == j && 
         offsets[c][2] == k ) { 
      if ( coefficients[c].size() == 0 ) return NULL; // or a zero?
      if ( coefficients[c].size() == 1 ) return coefficients[c][0];
      
      // build a series of multiplies?
      chillAST_node *lhs = coefficients[c][0];
      chillAST_node *rhs = coefficients[c][1];
      chillAST_node *mult = new chillAST_BinaryOperator( lhs, "*", rhs );
      for (int x = 2; x< coefficients[c].size(); x++) { 
        mult =  new chillAST_BinaryOperator( mult, "*", coefficients[c][x] );
      }
      return mult;
      
    }
  }
  // there was no coefficient with that offset
  return NULL; // or a FloatingLiteral zero? 
}



int stencilInfo::radius() { 
  int guess = -minOffset[0];  // minOffset should be a negative number 
  for (int i=0; i<dimensions; i++) { 
    if (minOffset[i] != -guess || maxOffset[i] != guess ) { 
      debug_fprintf(stderr, "stencilInfo::radius(), offsets are not symmetric:\n");
      for (int j=0; j<dimensions; j++) {
        debug_fprintf(stderr, "%s %d to %d\n",  indexVariables[j]->varname, minOffset[j], maxOffset[j]);
      }
      exit(-1);
    }
  }
  return guess;
}




bool stencilInfo::isSymmetric() 
{
  debug_fprintf(stderr, "stencilInfo::isSymmetric()\n");

  debug_fprintf(stderr, "%d dimensions\n", dimensions); 

  // don't really need this, I suppose
  for (int i=0; i<dimensions; i++) { 
    if (  -minOffset[i] != maxOffset[i] ) { 
      debug_fprintf(stderr, "stencilInfo::radius(), offsets in dimension %d  are not symmetric\n", i);
      return false;
    }

    debug_fprintf(stderr, "dimension %d   %d to %d\n", i, minOffset[i], maxOffset[i] );
  }

  int numoff = offsets.size();

  for (int o=0; o<numoff; o++) { 
    
    int numindex = offsets[o].size();
    if (numindex != 3)
      throw std::runtime_error("UHOH, stencil is not 3D? " + std::to_string(numindex) + " offsets"); // TODO'

    int ci = offsets[o][0];
    int cj = offsets[o][1];
    int ck = offsets[o][2];

    chillAST_node*  n = find_coefficient( ci, cj, ck );
    

    debug_begin
      fprintf(stderr, "\n\ncoeff %2d %2d %2d  is ", ci, cj, ck);
      n->print(0, stderr); fprintf(stderr, "\n");
    debug_end

    for (int d=0; d<3; d++)  { // dimension 0,1,2

      if (offsets[o][d] != 0) { 
        offsets[o][d] = -offsets[o][d]; // mirror in some dimension 
        
        int i = offsets[o][0];
        int j = offsets[o][1];
        int k = offsets[o][2];
        
        chillAST_node*  mirror = find_coefficient( i, j, k );
        if (!mirror) { 
          debug_fprintf(stderr, "coeff %d %d %d  does not exist\n", i, j, k );
          return false; 
        }
        else {
          debug_begin
            fprintf(stderr, "coeff %2d %2d %2d  is ", i, j, k);
            mirror->print(0, stderr); fprintf(stderr, "\n");
          debug_end
          
          // compare ASTs
          if (! n->isSameAs( mirror )) {
            debug_begin
              debug_fprintf(stderr, "coefficients (%d, %d, %d) and (%d, %d, d) differ\n", ci, cj, ck, i,j,k);
              n->print(0,stderr); debug_fprintf(stderr, "\n");
              mirror->print(0,stderr); debug_fprintf(stderr, "\n");
            debug_end

            return false; 
          }
          
        }
        offsets[o][d] = -offsets[o][d]; // revert
      }
    }



  }

  debug_fprintf(stderr, "yep, it's symmetric\n\n"); 
  return true;
}





#ifdef STANDALONE 

int main(int argc, char *argv[]) {
  
  SgProject* project = frontend(argc, argv);
  SgFilePtrList& file_list = project->get_fileList();
  SgFile *file =  file_list[0]; 
  SgSourceFile *src = isSgSourceFile(file);
  SgNode *toplevel = (SgNode *) src->get_globalScope();
  chillAST_node *entire_file =  ConvertRoseFile( toplevel, argv[1] );

  vector<chillAST_node*> funcs; 
  findmanually( entire_file, "smooth_box_default", funcs); // find functiondecl
  if (1 == funcs.size()) { 
    chillAST_node *body = ((chillAST_FunctionDecl *)funcs[0])->getBody();

    // the next tlien assumes that it is a single statement
    //if (body->isCompoundStmt()) body = body->getChild(0);


    //debug_fprintf(stderr, "\n\n"); 
    //body->print();
    //debug_fprintf(stderr, "\n\n"); 
    //body->dump(); 
    
    
    debug_fprintf(stderr, "body is %s\n", body->getTypeString()); 
    //if (body->isBinaryOperator()) 
    stencil(body);
  }
}

#endif   // STANDALONE 

