#ifndef _CODEGEN_H
#define _CODEGEN_H

#include <omega/Relation.h>
#include <code_gen/CG.h>
#include <code_gen/CG_outputBuilder.h>
#include <vector>
#include <string>

namespace omega {

class CodeGen {
public:
  static const std::string loop_var_name_prefix;
  static const int var_substitution_threshold;
  
protected:
  std::vector<std::vector<Relation> > projected_IS_; //!< projected_IS_[level-1][new stmt#]
  std::vector<Relation> xforms_;  //!< transformations[original stmt#]
  Relation known_; //!< no need to generate code for constraints satisfied in known
  std::vector<int> remap_; //!< map new stmt# to original stmt#

public:
  CodeGen(const std::vector<Relation> &xforms, const std::vector<Relation> &IS, const Relation &known = Relation::Null(),
		                 std::vector< std::vector<int > > smtNonSplitLevels_ =   std::vector< std::vector<int > >(),
		                 std::vector< std::vector<std::string> > loopIdxNames_ =  std::vector< std::vector<std::string> >(),
		                 std::vector< std::pair<int, std::string> >  syncs_ =    std::vector< std::pair<int, std::string> >()
		         );
  ~CodeGen() {}

	//! Build the tree-like loop nest/split relations
  CG_result *buildAST(int effort = 1);
  int num_level() const { return projected_IS_.size(); }
  
private:
  CG_result *buildAST(int level, const BoolSet<> &active, bool split_on_const, const Relation &restriction);

  friend class CG_result;
  friend class CG_split;
  friend class CG_loop;
  friend class CG_leaf;
};

}
#endif
