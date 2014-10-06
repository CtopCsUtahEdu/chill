#if !defined(Already_included_oc_i)
#define Already_included_oc_i

#include <basic/util.h>
#include <omega/omega_core/oc.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <vector>

namespace omega {

#define maxWildcards 18

extern int findingImplicitEqualities;
extern int firstCheckForRedundantEquations;
extern int use_ugly_names;
extern int doItAgain;
extern int newVar;
extern int conservative;
extern FILE *outputFile; /* printProblem writes its output to this file */
extern int nextWildcard;
extern int trace;
extern int depth;
extern int packing[maxVars];
extern int headerLevel;
extern int inApproximateMode;
extern int inStridesAllowedMode;
extern int addingOuterEqualities;
extern int outerColor;

const int keyMult = 31;
const int hashTableSize =5*maxmaxGEQs;
const int maxKeys = 8*maxmaxGEQs;
extern int hashVersion;
extern eqn hashMaster[hashTableSize];
extern int fastLookup[maxKeys*2];
extern int nextKey;

extern int reduceWithSubs;
extern int pleaseNoEqualitiesInSimplifiedProblems;

#define noProblem ((Problem *) 0)

extern Problem *originalProblem;
int checkIfSingleVar(eqn *e, int i);
/* Solve e = factor alpha for x_j and substitute */

void negateCoefficients(eqn * eqn, int nV);

extern int omegaInitialized;
extern Problem full_answer, context,redProblem;

#if defined BRAIN_DAMAGED_FREE
static inline void free(const Problem *p)
{
  free((char *)p);
}
#endif

#if defined NDEBUG 
#define CHECK_FOR_DUPLICATE_VARIABLE_NAMES 
#else
#define CHECK_FOR_DUPLICATE_VARIABLE_NAMES                              \
  {                                                                     \
    std::vector<std::string> name(nVars);                               \
    for(int i=1; i<=nVars; i++) {                                       \
      name[i-1] = variable(i);                                          \
      assert(!name[i-1].empty());                                       \
      for(int j=1; j<i; j++)                                            \
        assert(!(name[i-1] == name[j-1]));                              \
    }                                                                   \
  }
#endif


} // namespace

#endif
