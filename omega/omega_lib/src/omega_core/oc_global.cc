#include <omega/omega_core/oc_i.h>

namespace omega {

const int Problem::min_alloc = 10;
const int Problem::first_alloc_pad = 5;

int omega_core_debug = 0; // 3: full debugging info

int maxEQs  = 100; // original 35, increased by chun
int maxGEQs = 200; // original 70, increased by chun

int newVar = -1;
int findingImplicitEqualities = 0;
int firstCheckForRedundantEquations = 0;
int doItAgain;
int conservative = 0;
FILE *outputFile = stderr;  /* printProblem writes its output to this file */
char wildName[200][20];
int nextWildcard = 0;
int trace = 1;
int depth = 0;
int headerLevel;
int inApproximateMode = 0;
int inStridesAllowedMode = 0;
int addingOuterEqualities = 0;
int outerColor = 0;
int reduceWithSubs = 1;
int pleaseNoEqualitiesInSimplifiedProblems = 0;
Problem *originalProblem = noProblem;
int omegaInitialized = 0;
int mayBeRed = 0;


// Hash table is used to hash all inequalties for all problems.  It
// persists across problems for quick problem merging in case.  When
// the table is filled to 1/3 full, it is flushed and the filling
// process starts all over again.
int packing[maxVars];
int hashVersion = 0;
eqn hashMaster[hashTableSize];
int fastLookup[maxKeys*2];
int nextKey;

} // namespace
