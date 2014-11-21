#if !defined(Already_Included_code_gen)
#define Already_Included_code_gen

#include <basic/Tuple.h>
#include <omega/Relation.h>
#include <code_gen/CG.h>
#include <code_gen/CG_outputRepr.h>
#include <code_gen/CG_outputBuilder.h>

namespace omega {

typedef Tuple<int> IntTuple;
typedef Tuple<Relation> SetTuple;
typedef Tuple<SetTuple> SetTupleTuple;
typedef Tuple<Relation> RelTuple;
typedef Tuple<RelTuple> RelTupleTuple;

CG_outputRepr *MMGenerateCode(CG_outputBuilder* ocg,
                              Tuple<Relation> &T, Tuple<Relation> &old_IS,
                              const Tuple<CG_outputRepr *> &stmt_content,
                              Relation &known, int effort=1);
std::string MMGenerateCode(Tuple<Relation> &T, Tuple<Relation> &old_IS, Relation &known,
                           int effort=1);

//protonu-adding a new variant to keep Gabe's code happy
CG_outputRepr* MMGenerateCode(CG_outputBuilder* ocg, RelTuple &T, SetTuple &old_IS, 
		const Tuple<CG_outputRepr *> &stmt_content, Relation &known,
		Tuple< IntTuple >& smtNonSplitLevels_,
	       	std::vector< std::pair<int, std::string> > syncs_,
	       	const Tuple< Tuple<std::string> >& loopIdxNames_, 
	       	int effort=1);
//end-protonu

struct Polyhedra {
  int last_level;
  Tuple<Relation> transformations;
  Relation known;

  Tuple<int> remap;  // after initial iteration space's disjoint set splitting, the new statement number maps to old statement number
  Tuple<Tuple<Relation> > projected_nIS;
 
  Polyhedra(const Tuple<Relation> &T, const Tuple<Relation> &old_IS, const Relation &known = Relation::Null());
  ~Polyhedra() {}
};

}
#endif
