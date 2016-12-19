
#ifndef CG_chillRepr_h
#define CG_chillRepr_h

//                                               Repr using chillAst internally
#include <stdio.h>
#include <string.h>
#include <code_gen/CG_outputRepr.h>

#ifndef __STDC_CONSTANT_MACROS 
#define __STDC_CONSTANT_MACROS 
#endif


#include "chill_ast.hh"


namespace omega {

class CG_chillRepr : public CG_outputRepr {
  friend class CG_chillBuilder;
public:
  CG_chillRepr() { stmtclassname = strdup("NOTHING");  }


  char *type() const { return strdup("chill"); }; 
  // 
  vector<chillAST_node*> chillnodes;  // TODO make private
  void printChillNodes() const { 
    for (int i=0; i<chillnodes.size(); i++) {
      chillnodes[i]->print(); printf("\n"); } 
    fflush(stdout); 
  }; 

  CG_chillRepr( vector<chillAST_node*> cnodes ) { 
    chillnodes = cnodes;
  }

  CG_chillRepr( chillAST_node *chillast ) { 
    stmtclassname = strdup(chillast->getTypeString()); 
    //debug_fprintf(stderr, "made new chillRepr of class %s\n", stmtclassname); 
    if (chillast->getType() == CHILLAST_NODETYPE_COMPOUNDSTMT) {
      vector<chillAST_node*> children = chillast->getChildren(); 
      int numchildren = children.size();
      for (int i=0; i<numchildren; i++) {
        chillnodes.push_back( children[i] ); 
        //debug_fprintf(stderr, "adding a statement from a CompoundStmt\n"); 
      }
    }
    else { // for now, assume it's a single statement 
      chillnodes.push_back( chillast );  // ?? 
    }
  }
  void addStatement( chillAST_node *s ) { chillnodes.push_back( s ); } ;
  
  vector<chillAST_node*> getChillCode() const { return chillnodes; }; 

  chillAST_node *GetCode() ;
 
  
  ~CG_chillRepr();
  CG_outputRepr *clone() const;
  void clear();

  
  



  //---------------------------------------------------------------------------
  // Dump operations
  //---------------------------------------------------------------------------
  void dump() const { printChillNodes(); }; 
  void Dump() const;
  //void DumpToFile(FILE *fp = stderr) const;
private:


  char *stmtclassname;    // chill 
  
};



} // namespace

#endif
