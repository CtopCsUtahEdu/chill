/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
  Support functions for solving a problem.

 Notes:

 History:
  10/13/08 Complete back substitution process, Chun Chen.
  05/28/09 Extend normalize process to handle redundancy involving
           wilddcards, Chun Chen
*****************************************************************************/

#include <omega/omega_core/oc_i.h>
#include <basic/boolset.h>
#include <algorithm>
#include <vector>

#include "../../../chill_io.hh"

namespace omega {

int checkIfSingleVar(eqn* e, int i) {
  for (; i > 0; i--)
    if (e->coef[i]) {
      i--;
      break;
    }
  for (; i > 0; i--)
    if (e->coef[i])
      break;
  return (i == 0);
}


int singleVarGEQ(eqn* e) {
  return  !e->touched && e->key != 0 && -maxVars <= e->key && e->key <= maxVars;
}


void checkVars(int nVars) {
  if (nVars > maxVars) {
    debug_fprintf(stderr, "\nERROR:\n");
    debug_fprintf(stderr, "An attempt was made to create a conjunction with %d variables.\n", nVars);
    debug_fprintf(stderr, "The current limit on variables in a single conjunction is %d.\n", maxVars);
    debug_fprintf(stderr, "This limit can be changed by changing the #define of maxVars in oc.h.\n\n");
    exit(2);
  }
}


void Problem::difficulty(int &numberNZs, coef_t &maxMinAbsCoef, coef_t &sumMinAbsCoef) const {
  numberNZs=0;
  maxMinAbsCoef=0;
  sumMinAbsCoef=0;
  for (int e = 0; e < nGEQs; e++) {
    coef_t maxCoef = 0;
    for(int i = 1;i <= nVars;i++) 
      if (GEQs[e].coef[i]!=0) {
        coef_t a = abs(GEQs[e].coef[i]);
        maxCoef = max(maxCoef,a);
        numberNZs++;
      }
    coef_t nextCoef = 0;
    for(int i = 1;i <= nVars;i++) 
      if (GEQs[e].coef[i]!=0) {
        coef_t a = abs(GEQs[e].coef[i]);
        if (a < maxCoef) nextCoef = max(nextCoef,a);
        else if (a == maxCoef) maxCoef = 0x7fffffff;
      }
    maxMinAbsCoef = max(maxMinAbsCoef,nextCoef);
    sumMinAbsCoef += nextCoef;
  }

  for (int e = 0; e < nEQs; e++) {
    coef_t maxCoef = 0;
    for(int i = 1;i <= nVars;i++) 
      if (EQs[e].coef[i]!=0) {
        coef_t a = abs(EQs[e].coef[i]);
        maxCoef = max(maxCoef,a);
        numberNZs++;
      }
    coef_t nextCoef = 0;
    for(int i = 1;i <= nVars;i++) 
      if (EQs[e].coef[i]!=0) {
        coef_t a = abs(EQs[e].coef[i]);
        if (a < maxCoef) nextCoef = max(nextCoef,a);
        else if (a == maxCoef) maxCoef = 0x7fffffff;
      }
    maxMinAbsCoef = max(maxMinAbsCoef,nextCoef);
    sumMinAbsCoef += nextCoef;
  }
}

int Problem::countRedGEQs() const {
  int result = 0;
  for (int e = 0; e < nGEQs; e++)
    if (GEQs[e].color == EQ_RED) result++;
  return result;
}

int Problem::countRedEQs() const {
  int result = 0;
  for (int e = 0; e < nEQs; e++)
    if (EQs[e].color == EQ_RED) result++;
  return result;
}

int Problem::countRedEquations() const {
  int result = 0;
  for (int e = 0; e < nEQs; e++)
    if (EQs[e].color == EQ_RED) {
      int i;
      for (i = nVars; i > 0; i--) if (EQs[e].coef[i]) break;
      if (i == 0 && EQs[e].coef[0] != 0) return 0;
      else result+=2;
    }
  for (int e = 0; e < nGEQs; e++)
    if (GEQs[e].color == EQ_RED) result+=1;
  for (int e = 0; e < nMemories; e++)
    switch(redMemory[e].kind ) {
    case redEQ:
    case redStride:
      e++;
    case redLEQ:
    case redGEQ:
      e++;
    case notRed:
      ;    /* avoid warning about notRed not handled */
    }
  return result;
}

void Problem::deleteBlack() {
  int RedVar[maxVars];
  for(int i = safeVars+1;i <= nVars;i++) RedVar[i] = 0;

  assert(nSUBs == 0);

  for (int e = nEQs-1; e >= 0; e--)
    if (EQs[e].color != EQ_RED) {
      eqnncpy(&EQs[e],&EQs[nEQs-1], nVars);
      nEQs--;
    }
    else
      for(int i = safeVars+1;i <= nVars;i++)
        if (EQs[e].coef[i]) RedVar[i] = 1;

  for (int e = nGEQs-1; e >= 0; e--)
    if (GEQs[e].color != EQ_RED) {
      eqnncpy(&GEQs[e],&GEQs[nGEQs-1], nVars);
      nGEQs--;
    }
    else
      for(int i = safeVars+1;i <= nVars;i++)
        if (GEQs[e].coef[i]) RedVar[i] = 1;

  assert(nSUBs == 0);

  for(int i = nVars; i > safeVars;i--) {
    if (!RedVar[i]) deleteVariable(i);
  }
}


void Problem::deleteRed() {
  int BlackVar[maxVars];
  for(int i = safeVars+1;i <= nVars;i++) BlackVar[i] = 0;

  assert(nSUBs == 0);
  for (int e = nEQs-1; e >=0; e--)
    if (EQs[e].color) {
      eqnncpy(&EQs[e],&EQs[nEQs-1], nVars);
      nEQs--;
    }
    else
      for(int i = safeVars+1;i <= nVars;i++)
        if (EQs[e].coef[i]) BlackVar[i] = 1;

  for (int e = nGEQs-1; e >=0; e--)
    if (GEQs[e].color) {
      eqnncpy(&GEQs[e],&GEQs[nGEQs-1], nVars);
      nGEQs--;
    }
    else
      for(int i = safeVars+1;i <= nVars;i++)
        if (GEQs[e].coef[i]) BlackVar[i] = 1;

  assert(nSUBs == 0);

  for(int i = nVars; i> safeVars;i--) {
    if (!BlackVar[i]) deleteVariable(i);
  }
}


void Problem::turnRedBlack() {
  for (int e = nEQs-1; e >= 0; e--) EQs[e].color = 0;
  for (int e = nGEQs-1; e >= 0; e--) GEQs[e].color = 0;
}


void Problem::useWildNames() {
  for(int i = safeVars+1; i <= nVars; i++) nameWildcard(i);
}


void negateCoefficients(eqn* eqn, int nVars) {
  for (int i = nVars; i >= 0; i--)
    eqn-> coef[i] = -eqn->coef[i];
  eqn->touched = true;
}


void Problem::negateGEQ(int e) {
  negateCoefficients(&GEQs[e],nVars);
  GEQs[e].coef[0]--;
}


void Problem:: deleteVariable(int i) {
  if (i < safeVars) {
    int j = safeVars;
    for (int e = nGEQs - 1; e >= 0; e--) {
      GEQs[e].touched = true;
      GEQs[e].coef[i] = GEQs[e].coef[j];
      GEQs[e].coef[j] = GEQs[e].coef[nVars];
    }
    for (int e = nEQs - 1; e >= 0; e--) {
      EQs[e].coef[i] = EQs[e].coef[j];
      EQs[e].coef[j] = EQs[e].coef[nVars];
    }
    for (int e = nSUBs - 1; e >= 0; e--) {
      SUBs[e].coef[i] = SUBs[e].coef[j];
      SUBs[e].coef[j] = SUBs[e].coef[nVars];
    }
    var[i] = var[j];
    var[j] = var[nVars];
  }
  else if (i < nVars) {
    for (int e = nGEQs - 1; e >= 0; e--)
      if (GEQs[e].coef[nVars]) {
        GEQs[e].coef[i] = GEQs[e].coef[nVars];
        GEQs[e].touched = true;
      }
    for (int e = nEQs - 1; e >= 0; e--)
      EQs[e].coef[i] = EQs[e].coef[nVars];
    for (int e = nSUBs - 1; e >= 0; e--)
      SUBs[e].coef[i] = SUBs[e].coef[nVars];
    var[i] = var[nVars];
  }
  if (i <= safeVars)
    safeVars--;
  nVars--;
}


void Problem::setInternals() {
  if (!variablesInitialized) {
    initializeVariables();
  }

  var[0] = 0;
  nextWildcard = 0;
  for(int i = 1;i <= nVars;i++)
    if (var[i] < 0) 
      var[i] = --nextWildcard;
  
  assert(nextWildcard >= -maxWildcards);

  CHECK_FOR_DUPLICATE_VARIABLE_NAMES;

  int v = nSUBs;
  for(int i = 1;i <= safeVars;i++) if (var[i] > 0) v++;
  varsOfInterest = v;

  if (nextKey * 3 > maxKeys) {
    omega::hashVersion++;
    nextKey = maxVars + 1;
    for (int e = nGEQs - 1; e >= 0; e--)
      GEQs[e].touched = true;
    for (int i = 0; i < hashTableSize; i++)
      hashMaster[i].touched = -1;
    hashVersion = omega::hashVersion;
  }
  else if (hashVersion != omega::hashVersion) {
    for (int e = nGEQs - 1; e >= 0; e--)
      GEQs[e].touched = true;
    hashVersion = omega::hashVersion;
  }
}


void Problem::setExternals() {
  for (int i = 1; i <= safeVars; i++)
    forwardingAddress[var[i]] = i;
  for (int i = 0; i < nSUBs; i++)
    forwardingAddress[SUBs[i].key] = -i - 1;
}


void setOutputFile(FILE * file) {
  /* sets the file to which printProblem should send its output to "file" */

  outputFile = file;
}


void setPrintLevel(int level) {
  /* Sets the nber of points printed before constraints in printProblem */
  headerLevel = level;
}


void Problem::putVariablesInStandardOrder() {
  for(int i = 1;i <= safeVars;i++)  {
    int b = i;
    for(int j=i+1;j<=safeVars;j++) {
      if (var[b] < var[j]) b = j;
    }
    if (b != i) swapVars(i,b);
  }
}
 

void Problem::nameWildcard(int i) {
  int j;
  do {
    --nextWildcard;
    if (nextWildcard < -maxWildcards)
      nextWildcard = -1;
    var[i] = nextWildcard;
    for(j = nVars; j > 0;j--) if (i!=j && var[j] == nextWildcard) break;
  } while (j != 0); 
}


int Problem::protectWildcard(int i) {
  assert (i > safeVars);
  if (i != safeVars+1) swapVars(i,safeVars+1);
  safeVars++;
  nameWildcard(safeVars);
  return safeVars;
}


int Problem::addNewProtectedWildcard() {
  int i = ++safeVars;
  nVars++;
  if (nVars != i) {
    for (int e = nGEQs - 1; e >= 0; e--) {
      if (GEQs[e].coef[i] != 0)
        GEQs[e].touched = true;
      GEQs[e].coef[nVars] = GEQs[e].coef[i];
    }
    for (int e = nEQs - 1; e >= 0; e--) {
      EQs[e].coef[nVars] = EQs[e].coef[i];
    }
    for (int e = nSUBs - 1; e >= 0; e--) {
      SUBs[e].coef[nVars] = SUBs[e].coef[i];
    }
    var[nVars] = var[i];
  }
  for (int e = nGEQs - 1; e >= 0; e--)
    GEQs[e].coef[i] = 0;
  for (int e = nEQs - 1; e >= 0; e--)
    EQs[e].coef[i] = 0;
  for (int e = nSUBs - 1; e >= 0; e--)
    SUBs[e].coef[i] = 0;
  nameWildcard(i);
  return (i);
}


int Problem::addNewUnprotectedWildcard() {
  int i = ++nVars;
  for (int e = nGEQs - 1; e >= 0; e--) GEQs[e].coef[i] = 0;
  for (int e = nEQs - 1; e >= 0; e--) EQs[e].coef[i] = 0;
  for (int e = nSUBs - 1; e >= 0; e--) SUBs[e].coef[i] = 0;
  nameWildcard(i);
  return i;
}


void Problem::cleanoutWildcards() {
  bool renormalize = false;

  // substituting wildcard equality
  for (int e = nEQs-1; e >= 0; e--) {
    for (int i = nVars; i >= safeVars+1; i--)
      if (EQs[e].coef[i] != 0) {
        coef_t c = EQs[e].coef[i];
        coef_t a = abs(c);

        bool preserveThisConstraint = true;
        for (int e2 = nEQs-1; e2 >= 0; e2--)
          if (e2 != e && EQs[e2].coef[i] != 0 && EQs[e2].color >= EQs[e].color) {
            preserveThisConstraint = preserveThisConstraint && (gcd(a,abs(EQs[e2].coef[i])) != 1);
            coef_t k = lcm(a, abs(EQs[e2].coef[i]));
            coef_t coef1 = (EQs[e2].coef[i]>0?1:-1) * k / c;
            coef_t coef2 = k / abs(EQs[e2].coef[i]);
            for (int j = nVars; j >= 0; j--)
              EQs[e2].coef[j] = EQs[e2].coef[j] * coef2 - EQs[e].coef[j] * coef1;
            
            coef_t g = 0;
            for (int j = nVars; j >= 0; j--) {
              g = gcd(abs(EQs[e2].coef[j]), g);
              if (g == 1)
                break;
            }
            if (g != 0 && g != 1)
              for (int j = nVars; j >= 0; j--)
                EQs[e2].coef[j] /= g;
          }

        for (int e2 = nGEQs-1; e2 >= 0; e2--)
          if (GEQs[e2].coef[i] != 0 && GEQs[e2].color >= EQs[e].color) {
            coef_t k = lcm(a, abs(GEQs[e2].coef[i]));
            coef_t coef1 = (GEQs[e2].coef[i]>0?1:-1) * k / c;
            coef_t coef2 = k / abs(GEQs[e2].coef[i]);
            for (int j = nVars; j >= 0; j--)
              GEQs[e2].coef[j] = GEQs[e2].coef[j] * coef2 - EQs[e].coef[j] * coef1;
            
            GEQs[e2].touched = 1;
            renormalize = true;
          }
        
        for (int e2 = nSUBs-1; e2 >= 0; e2--)
          if (SUBs[e2].coef[i] != 0 && SUBs[e2].color >= EQs[e].color) {
            coef_t k = lcm(a, abs(SUBs[e2].coef[i]));
            coef_t coef1 = (SUBs[e2].coef[i]>0?1:-1) * k / c;
            coef_t coef2 = k / abs(SUBs[e2].coef[i]);
            for (int j = nVars; j >= 0; j--)
              SUBs[e2].coef[j] = SUBs[e2].coef[j] * coef2 - EQs[e].coef[j] * coef1;
            
            coef_t g = 0;
            for (int j = nVars; j >= 0; j--) {
              g = gcd(abs(SUBs[e2].coef[j]), g);
              if (g == 1)
                break;
            }
            if (g != 0 && g != 1)
              for (int j = nVars; j >= 0; j--)
                SUBs[e2].coef[j] /= g;            
          }

        // remove redundent wildcard equality
        if (!preserveThisConstraint) {
          if (e < nEQs-1)
            eqnncpy (&EQs[e], &EQs[nEQs-1], nVars);
          nEQs--;
          deleteVariable(i);
        }
      
        break;
      }
  }

  // remove multi-wildcard equality in approximation mode
  if (inApproximateMode)
    for (int e = nEQs-1; e >= 0; e--)
      for (int i = nVars; i >= safeVars+1; i--)
        if (EQs[e].coef[i] != 0) {
          int j = i-1;
          for (; j >= safeVars+1; j--)
            if (EQs[e].coef[j] != 0)
              break;

          if (j != safeVars) {
            if (e < nEQs-1)
              eqnncpy (&EQs[e], &EQs[nEQs-1], nVars);
            nEQs--;
          }
      
          break;
        }

  if (renormalize)
    normalize();
}


void Problem:: check() const {
#ifndef NDEBUG
  int v = nSUBs;
  checkVars(nVars+1);
  for(int i = 1; i <= safeVars; i++) if (var[i] > 0) v++;
  assert(v == varsOfInterest);
  for(int e = 0; e < nGEQs; e++) assert(GEQs[e].touched || GEQs[e].key != 0);
  if(!mayBeRed) {
    for(int e = 0; e < nEQs; e++) assert(!EQs[e].color);
    for(int e = 0; e < nGEQs; e++) assert(!GEQs[e].color);
  }
  else 
    for(int i = safeVars+1; i <= nVars; i++) {
      int isBlack = 0;
      int isRed = 0;
      for(int e = 0; e < nEQs; e++)
        if (EQs[e].coef[i]) {
          if (EQs[e].color) isRed = 1;
          else isBlack = 1;
        }
      for(int e = 0; e < nGEQs; e++)
        if (GEQs[e].coef[i]) {
          if (GEQs[e].color) isRed = 1;
          else isBlack = 1;
        }
      if (isBlack && isRed && 0) {
        fprintf(outputFile,"Mixed Red and Black variable:\n");
        printProblem();
      }
    }
#endif 
}


void Problem::rememberRedConstraint(eqn *e, redType type, coef_t stride) {
  // Check if this is really a stride constraint
  if (type == redEQ && newVar == nVars && e->coef[newVar]) {
    type = redStride;
    stride = e->coef[newVar];
  }
  //   else for(int i = safeVars+1; i <= nVars; i++) assert(!e->coef[i]); // outdated -- by chun 10/30/2008

  assert(type != notRed);
  assert(type == redStride || stride == 0);

  if (TRACE) {
    fprintf(outputFile,"being asked to remember red constraint:\n");
    switch(type) {
    case notRed: fprintf(outputFile,"notRed: ");
      break;
    case redGEQ: fprintf(outputFile,"Red: 0 <= ");
      break;
    case redLEQ: fprintf(outputFile,"Red: 0 >= ");
      break;
    case redEQ: fprintf(outputFile,"Red: 0 == ");
      break;
    case redStride: fprintf(outputFile,"Red stride " coef_fmt ": ",stride);
      break;
    }
    printTerm(e,1);
    fprintf(outputFile,"\n");
    printProblem();
    fprintf(outputFile,"----\n");
  }

  // Convert redLEQ to redGEQ
  eqn mem;
  eqnncpy(&mem,e, nVars);
  e = &mem;
  if (type == redLEQ) {
    for(int i = 0; i <= safeVars; i++)
      e->coef[i] = -e->coef[i];
    type = redGEQ;
  }

  // Prepare coefficient array for red constraint
  bool has_wildcard = false;
  coef_t coef[varsOfInterest-nextWildcard+1];
  for (int i = 0; i <= varsOfInterest-nextWildcard; i++)
    coef[i] = 0;
  for (int i = 0; i <= safeVars; i++) {
    if (var[i] < 0) {
      if (e->coef[i] != 0) {
        coef[varsOfInterest-var[i]] = e->coef[i];
        has_wildcard = true;
      }
    }
    else
      coef[var[i]] = e->coef[i];
  }

  // Sophisticated back substituion for wildcards, use Gaussian elimination
  // as a fallback if no simple equations available. -- by chun 10/13/2008
  if (has_wildcard) {
    // Find substitutions involving wildcard
    coef_t *repl_subs[nSUBs];
    int num_wild_in_repl_subs[nSUBs];
    int num_repl_subs = 0;
    for (int i = 0; i < nSUBs; i++) {
      int t = 0;
      for (int j = 1; j <= safeVars; j++) {
        if (var[j] < 0 && SUBs[i].coef[j] != 0)
          t++;
      }
      if (t > 0) {
        repl_subs[num_repl_subs] = new coef_t[varsOfInterest-nextWildcard+1];
        for (int j = 0; j <= varsOfInterest-nextWildcard; j++)
          repl_subs[num_repl_subs][j] = 0;
      
        for (int k = 0; k <= safeVars; k++)
          repl_subs[num_repl_subs][(var[k]<0)?varsOfInterest-var[k]:var[k]] = SUBs[i].coef[k];
        repl_subs[num_repl_subs][SUBs[i].key] = -1;
        num_wild_in_repl_subs[num_repl_subs] = t;
        num_repl_subs++;
      }
    }

    int wild_solved[-nextWildcard+1];
    bool has_unsolved = false;
    for (int i = 1; i <= -nextWildcard; i++) {
      int minimum_wild = 0;
      int pos;
      for (int j = 0; j < num_repl_subs; j++)
        if (repl_subs[j][varsOfInterest+i] != 0 && (minimum_wild == 0 || num_wild_in_repl_subs[j] < minimum_wild)) {
          minimum_wild = num_wild_in_repl_subs[j];
          pos = j;
        }
    
      if (minimum_wild == 0) {
        wild_solved[i] = -1;
        if (coef[varsOfInterest+i] != 0) {
          fprintf(outputFile,"No feasible back substitutions available\n");
          printProblem();
          exit(1);
        }
      }
      else if (minimum_wild == 1)
        wild_solved[i] = pos;
      else {
        wild_solved[i] = -1;
        if (coef[varsOfInterest+i] != 0)
          has_unsolved = true;
      }
    }

    // Gaussian elimination
    while (has_unsolved) {
      for (int i = 0; i < num_repl_subs; i++)
        if (num_wild_in_repl_subs[i] > 1) {
          for (int j = 1; j <= -nextWildcard; j++) {
            if (repl_subs[i][varsOfInterest+j] != 0 && wild_solved[j] >= 0) {
              int s = wild_solved[j];
              coef_t l = lcm(abs(repl_subs[i][varsOfInterest+j]), abs(repl_subs[s][varsOfInterest+j]));
              coef_t scale_1 = l/abs(repl_subs[i][varsOfInterest+j]);
              coef_t scale_2 = l/abs(repl_subs[s][varsOfInterest+j]);
              int sign = ((repl_subs[i][varsOfInterest+j]>0)?1:-1) * ((repl_subs[s][varsOfInterest+j]>0)?1:-1);
              for (int k = 0; k <= varsOfInterest-nextWildcard; k++)
                repl_subs[i][k] = scale_1*repl_subs[i][k] - sign*scale_2*repl_subs[s][k];
              num_wild_in_repl_subs[i]--;
            }
          }

          if (num_wild_in_repl_subs[i] == 1) {
            for (int j = 1; j <= -nextWildcard; j++)
              if (repl_subs[i][varsOfInterest+j] != 0) {
                assert(wild_solved[j]==-1);
                wild_solved[j] = i;
                break;
              }
          }
          else if (num_wild_in_repl_subs[i] > 1) {
            int pos = 0;
            for (int j = 1; j <= -nextWildcard; j++)
              if (repl_subs[i][varsOfInterest+j] != 0) {
                pos = j;
                break;
              }
            assert(pos > 0);
              
            for (int j = i+1; j < num_repl_subs; j++)
              if (repl_subs[j][varsOfInterest+pos] != 0) {
                coef_t l = lcm(abs(repl_subs[i][varsOfInterest+pos]), abs(repl_subs[j][varsOfInterest+pos]));
                coef_t scale_1 = l/abs(repl_subs[i][varsOfInterest+pos]);
                coef_t scale_2 = l/abs(repl_subs[j][varsOfInterest+pos]);
                int sign = ((repl_subs[i][varsOfInterest+pos]>0)?1:-1) * ((repl_subs[j][varsOfInterest+pos]>0)?1:-1);
                for (int k = 0; k <= varsOfInterest-nextWildcard; k++)
                  repl_subs[j][k] = scale_2*repl_subs[j][k] - sign*scale_1*repl_subs[i][k];

                num_wild_in_repl_subs[j] = 0;
                int first_wild = 0;
                for (int k = 1; k <= -nextWildcard; k++)
                  if (repl_subs[j][varsOfInterest+k] != 0) {
                    num_wild_in_repl_subs[j]++;
                    first_wild = k;
                  }

                if (num_wild_in_repl_subs[j] == 1) {
                  if (wild_solved[first_wild] < 0)
                    wild_solved[first_wild] = j;
                }
              }
          }
        }
      
      has_unsolved = false;
      for (int i = 1; i <= -nextWildcard; i++)
        if (coef[varsOfInterest+i] != 0 && wild_solved[i] < 0) {
          has_unsolved = true;
          break;
        }
    }          
              
    // Substitute all widecards in the red constraint
    for (int i = 1; i <= -nextWildcard; i++) {
      if (coef[varsOfInterest+i] != 0) {
        int s = wild_solved[i];
        assert(s >= 0);
      
        coef_t l = lcm(abs(coef[varsOfInterest+i]), abs(repl_subs[s][varsOfInterest+i]));
        coef_t scale_1 = l/abs(coef[varsOfInterest+i]);
        coef_t scale_2 = l/abs(repl_subs[s][varsOfInterest+i]);
        int sign = ((coef[varsOfInterest+i]>0)?1:-1) * ((repl_subs[s][varsOfInterest+i]>0)?1:-1);
        for (int j = 0; j <= varsOfInterest-nextWildcard; j++)
          coef[j] = scale_1*coef[j] - sign*scale_2*repl_subs[s][j];

        if (scale_1 != 1)
          stride *= scale_1;
      }
    }

    for (int i = 0; i < num_repl_subs; i++)
      delete []repl_subs[i];
  }
  
  // Ready to insert into redMemory
  int m = nMemories++;
  redMemory[m].length = 0;
  redMemory[m].kind = type;
  redMemory[m].constantTerm = coef[0];
  for(int i = 1; i <= varsOfInterest; i++)
    if (coef[i]) {
      int j = redMemory[m].length++;
      redMemory[m].coef[j] = coef[i];
      redMemory[m].var[j] = i;
    }
  if (type == redStride) redMemory[m].stride = stride;
  if (DBUG) {
    fprintf(outputFile,"Red constraint remembered\n");
    printProblem();
  }
}

void Problem::recallRedMemories() {
  if (nMemories) {
    if (TRACE) {
      fprintf(outputFile,"Recalling red memories\n");
      printProblem();
    }

    eqn* e = 0;
    for(int m = 0; m < nMemories; m++) {
      switch(redMemory[m].kind) {
      case redGEQ:
      {
        int temporary_eqn = newGEQ();
        e = &GEQs[temporary_eqn];
        eqnnzero(e, nVars);
        e->touched = 1;
        break;
      }
      case redEQ:
      {
        int temporary_eqn = newEQ();
        e = &EQs[temporary_eqn];
        eqnnzero(e, nVars);
        break;
      }
      case redStride:
      {
        int temporary_eqn = newEQ();
        e = &EQs[temporary_eqn];
        eqnnzero(e, nVars);
        int i = addNewUnprotectedWildcard();
        e->coef[i] = -redMemory[m].stride;
        break;
      }
      default:
        assert(0);
      }
      e->color = EQ_RED;
      e->coef[0] = redMemory[m].constantTerm;
      for(int i = 0; i < redMemory[m].length; i++) {
        int v = redMemory[m].var[i];
        assert(var[forwardingAddress[v]] == v);
        e->coef[forwardingAddress[v]] = redMemory[m].coef[i];
      }
    }
        
    nMemories = 0;
    if (TRACE) {
      fprintf(outputFile,"Red memories recalled\n");
      printProblem();
    }
  }
}

void Problem::swapVars(int i, int j) {
  if (DEBUG) {
    use_ugly_names++;
    fprintf(outputFile, "Swapping %d and %d\n", i, j);
    printProblem();
    use_ugly_names--;
  }
  std::swap(var[i], var[j]);
  for (int e = nGEQs - 1; e >= 0; e--)
    if (GEQs[e].coef[i] != GEQs[e].coef[j]) {
      GEQs[e].touched = true;
      coef_t t = GEQs[e].coef[i];
      GEQs[e].coef[i] = GEQs[e].coef[j];
      GEQs[e].coef[j] = t;
    }
  for (int e = nEQs - 1; e >= 0; e--)
    if (EQs[e].coef[i] != EQs[e].coef[j]) {
      coef_t t = EQs[e].coef[i];
      EQs[e].coef[i] = EQs[e].coef[j];
      EQs[e].coef[j] = t;
    }
  for (int e = nSUBs - 1; e >= 0; e--)
    if (SUBs[e].coef[i] != SUBs[e].coef[j]) {
      coef_t t = SUBs[e].coef[i];
      SUBs[e].coef[i] = SUBs[e].coef[j];
      SUBs[e].coef[j] = t;
    }
  if (DEBUG) {
    use_ugly_names++;
    fprintf(outputFile, "Swapping complete \n");
    printProblem();
    fprintf(outputFile, "\n");
    use_ugly_names--;
  }
}

void Problem::addingEqualityConstraint(int e) {
  if (addingOuterEqualities && originalProblem != noProblem &&
      originalProblem != this && !conservative) {
    int e2 = originalProblem->newEQ();
    if (TRACE)
      fprintf(outputFile, "adding equality constraint %d to outer problem\n", e2);
    eqnnzero(&originalProblem->EQs[e2], originalProblem->nVars);
    for (int i = nVars; i >= 1; i--) {
      int j;
      for (j = originalProblem->nVars; j >= 1; j--)
        if (originalProblem->var[j] == var[i])
          break;
      if (j <= 0 || (outerColor && j > originalProblem->safeVars)) {
        if (DBUG)
          fprintf(outputFile, "retracting\n");
        originalProblem->nEQs--;
        return;
      }
      originalProblem->EQs[e2].coef[j] = EQs[e].coef[i];
    }
    originalProblem->EQs[e2].coef[0] = EQs[e].coef[0];
 
    originalProblem->EQs[e2].color = outerColor;
    if (DBUG)
      originalProblem->printProblem();
  }
}


// Initialize hash codes for inequalities, remove obvious redundancy.
// Case 1:
// a1*x1+a2*x2+...>=c  (1)
// a1*x2+a2*x2+...>=c' (2)
// if c>=c' then (2) is redundant, and vice versa.
//
// case 2:
// a1*x1+a2*x2+...>=c  (1)
// a1*x1+a2*x2+...<=c' (2)
// if c=c' then add equality of (1) or (2),
// if c>c' then no solution.
//
// Finally it calls extended normalize process which handles
// wildcards in redundacy removal.
normalizeReturnType Problem::normalize() {
  int i, j;
  bool coupledSubscripts = false;

  check();

  for (int e = 0; e < nGEQs; e++) {
    if (!GEQs[e].touched) {
      if (!singleVarGEQ(&GEQs[e]))
        coupledSubscripts = true;
    }
    else { // normalize e
      coef_t g;
      int topVar;
      int i0;
      coef_t hashCode;

      {
        int *p = &packing[0];
        for (int k = 1; k <= nVars; k++)
          if (GEQs[e].coef[k]) {
            *(p++) = k;
          }
        topVar = (p - &packing[0]) - 1;
      }

      if (topVar == -1) {
        if (GEQs[e].coef[0] < 0) {
          // e has no solution
          return (normalize_false);
        }
        deleteGEQ(e);
        e--;
        continue;
      }
      else if (topVar == 0) {
        int singleVar = packing[0];
        g = GEQs[e].coef[singleVar];
        if (g > 0) {
          GEQs[e].coef[singleVar] = 1;
          GEQs[e].key = singleVar;
        }
        else {
          g = -g;
          GEQs[e].coef[singleVar] = -1;
          GEQs[e].key = -singleVar;
        }
        if (g > 1)
          GEQs[e].coef[0] = int_div(GEQs[e].coef[0], g);
      }
      else {
        coupledSubscripts = true;
        i0 = topVar;
        i = packing[i0--];
        g = GEQs[e].coef[i];
        hashCode = g * (i + 3);
        if (g < 0)
          g = -g;
        for (; i0 >= 0; i0--) {
          coef_t x;
          i = packing[i0];
          x = GEQs[e].coef[i];
          hashCode = hashCode * keyMult * (i + 3) + x;
          if (x < 0)
            x = -x;
          if (x == 1) {
            g = 1;
            i0--;
            break;
          }
          else
            g = gcd(x, g);
        }
        for (; i0 >= 0; i0--) {
          coef_t x;
          i = packing[i0];
          x = GEQs[e].coef[i];
          hashCode = hashCode * keyMult * (i + 3) + x;
        }
        if (g > 1) {
          GEQs[e].coef[0] = int_div(GEQs[e].coef[0], g);
          i0 = topVar;
          i = packing[i0--];
          GEQs[e].coef[i] = GEQs[e].coef[i] / g;
          hashCode = GEQs[e].coef[i] * (i + 3);
          for (; i0 >= 0; i0--) {
            i = packing[i0];
            GEQs[e].coef[i] = GEQs[e].coef[i] / g;
            hashCode = hashCode * keyMult * (i + 3) + GEQs[e].coef[i];
          }
        }

        {
          coef_t g2 = abs(hashCode);  // get e's hash code
          j = static_cast<int>(g2 % static_cast<coef_t>(hashTableSize));
          assert (g2 % (coef_t) hashTableSize == j);
          while (1) {
            eqn *proto = &(hashMaster[j]);
            if (proto->touched == g2) {
              if (proto->coef[0] == topVar) {
                if (hashCode >= 0)
                  for (i0 = topVar; i0 >= 0; i0--) {
                    i = packing[i0];
                    if (GEQs[e].coef[i] != proto->coef[i])
                      break;
                  }
                else
                  for (i0 = topVar; i0 >= 0; i0--) {
                    i = packing[i0];
                    if (GEQs[e].coef[i] != -proto->coef[i])
                      break;
                  }

                if (i0 < 0) {
                  if (hashCode >= 0)
                    GEQs[e].key = proto->key;
                  else
                    GEQs[e].key = -proto->key;
                  break;
                }
              }
            }
            else if (proto->touched < 0) { //insert e into the empty entry in hash table
              eqnnzero(proto, nVars);
              if (hashCode >= 0)
                for (i0 = topVar; i0 >= 0; i0--) {
                  i = packing[i0];
                  proto->coef[i] = GEQs[e].coef[i];
                }
              else
                for (i0 = topVar; i0 >= 0; i0--) {
                  i = packing[i0];
                  proto->coef[i] = -GEQs[e].coef[i];
                }
              proto->coef[0] = topVar;
              proto->touched = g2;
              proto->key = nextKey++;

              if (proto->key > maxKeys) {
                fprintf(outputFile, "too many hash keys generated \n");
                fflush(outputFile);
                exit(2);
              }
              if (hashCode >= 0)
                GEQs[e].key = proto->key;
              else
                GEQs[e].key = -proto->key;
              break;
            }
            j = (j + 1) % hashTableSize;
          }
        }
      }
    }

    GEQs[e].touched = false;

    {
      int eKey = GEQs[e].key;
      int e2;
      if (e > 0) {
        e2 = fastLookup[maxKeys - eKey];
        if (e2 >= 0 && e2 < e && GEQs[e2].key == -eKey) {
          // confirm it is indeed a match  -- by chun 10/29/2008
          int k;
          for (k = nVars; k >= 1; k--)
            if (GEQs[e2].coef[k] != -GEQs[e].coef[k])
              break;

          if (k == 0) {    
            if (GEQs[e2].coef[0] < -GEQs[e].coef[0]) {
              // there is no solution from e and e2
              return (normalize_false);
            }
            else if (GEQs[e2].coef[0] == -GEQs[e].coef[0]) {
              // reduce e and e2 to an equation
              int neweq = newEQ();
              eqnncpy(&EQs[neweq], &GEQs[e], nVars);
              EQs[neweq].color = GEQs[e].color || GEQs[e2].color;
              addingEqualityConstraint(neweq);
            }
          }
        }

        e2 = fastLookup[maxKeys + eKey];
        if (e2 >= 0 && e2 < e && GEQs[e2].key == eKey) {
          // confirm it is indeed a match  -- by chun 10/29/2008
          int k;
          for (k = nVars; k >= 1; k--)
            if (GEQs[e2].coef[k] != GEQs[e].coef[k])
              break;
          
          if (k == 0) {
            if (GEQs[e2].coef[0] > GEQs[e].coef[0] ||
                (GEQs[e2].coef[0] == GEQs[e].coef[0] && GEQs[e2].color)) {
              // e2 is redundant
              GEQs[e2].coef[0] = GEQs[e].coef[0];
              GEQs[e2].color =  GEQs[e].color;
              deleteGEQ(e);
              e--;
              continue;
            }
            else {
              // e is redundant
              deleteGEQ(e);
              e--;
              continue;
            }
          }
        }
      }
      fastLookup[maxKeys + eKey] = e;
    }
  }

  // bypass entended normalization for temporary problem
  if (!isTemporary && !inApproximateMode)
    normalize_ext();
    
  return coupledSubscripts ? normalize_coupled : normalize_uncoupled;
}

//
// Extended normalize process, remove redundancy involving wildcards.
// e.g.
// exists alpha, beta:
// v1+8*alpha<=v2<=15+8*alpha (1)
// v1+8*beta<=v2<=15+8*beta   (2)
// if there are no other inequalities involving alpha or beta,
// then either (1) or (2) is redundant.  Such case can't be simplified
// by fourier-motzkin algorithm due to special meanings of existentials.
//
void Problem::normalize_ext() {
  std::vector<BoolSet<> > disjoint_wildcards(nVars-safeVars, BoolSet<>(nVars-safeVars));
  std::vector<BoolSet<> > wildcards_in_inequality(nVars-safeVars, BoolSet<>(nGEQs));
  for (int i = 0; i < nVars-safeVars; i++) {
    disjoint_wildcards[i].set(i);
  }

  // create disjoint wildcard sets according to inequalities
  for (int e = 0; e < nGEQs; e++) {
    std::vector<BoolSet<> >::iterator first_set = disjoint_wildcards.end();
    for (int i = 0; i < nVars-safeVars; i++)
      if (GEQs[e].coef[i+safeVars+1] != 0) {
        wildcards_in_inequality[i].set(e);

        std::vector<BoolSet<> >::iterator cur_set = disjoint_wildcards.end();
        for (std::vector<BoolSet<> >::iterator j = disjoint_wildcards.begin(); j != disjoint_wildcards.end(); j++)
          if ((*j).get(i)) {
            cur_set = j;
            break;
          }
        assert(cur_set!=disjoint_wildcards.end());
        if (first_set == disjoint_wildcards.end())
          first_set = cur_set;
        else if (first_set != cur_set) {
          *first_set |= *cur_set;
          disjoint_wildcards.erase(cur_set);
        }
      }
  }

  // do not consider wildcards appearing in equalities
  for (int e = 0; e < nEQs; e++)
    for (int i = 0; i < nVars-safeVars; i++)
      if (EQs[e].coef[i+safeVars+1] != 0) {
        for (std::vector<BoolSet<> >::iterator j = disjoint_wildcards.begin(); j != disjoint_wildcards.end(); j++)
          if ((*j).get(i)) {
            disjoint_wildcards.erase(j);
            break;
          }
      }
  
  // create disjoint inequality sets
  std::vector<BoolSet<> > disjoint_inequalities(disjoint_wildcards.size());
  for (size_t i = 0; i < disjoint_wildcards.size(); i++)
    for (int j = 0; j < nVars-safeVars; j++)
      if (disjoint_wildcards[i].get(j))
        disjoint_inequalities[i] |= wildcards_in_inequality[j];

  // hash the inequality again, this time separate wildcard variables from
  // regular variables
  coef_t hash_safe[nGEQs];
  coef_t hash_wild[nGEQs];
  for (int e = 0; e < nGEQs; e++) {
    coef_t hashCode = 0;
    for (int i = 1; i <= safeVars; i++)
      if (GEQs[e].coef[i] != 0)
        hashCode = hashCode * keyMult * (i+3) + GEQs[e].coef[i];
    hash_safe[e] = hashCode;
    
    hashCode = 0;
    for (int i = safeVars+1; i <= nVars; i++)
      if (GEQs[e].coef[i] != 0)
        hashCode = hashCode * keyMult + GEQs[e].coef[i];
    hash_wild[e] = hashCode;
  }

  // sort hash keys for each disjoint set
  std::vector<std::vector<std::pair<int, std::pair<coef_t, coef_t> > > > disjoint_hash(disjoint_inequalities.size());
  for (size_t i = 0; i < disjoint_inequalities.size(); i++)
    for (int e = 0; e < nGEQs; e++)
      if (disjoint_inequalities[i].get(e)) {
        std::vector<std::pair<int, std::pair<coef_t, coef_t> > >::iterator j = disjoint_hash[i].begin();
        for (; j != disjoint_hash[i].end(); j++)
          if ((hash_safe[e] > (*j).second.first) ||
              (hash_safe[e] == (*j).second.first && hash_wild[e] > (*j).second.second))
            break;
        disjoint_hash[i].insert(j, std::make_pair(e, std::make_pair(hash_safe[e], hash_wild[e])));
      }

  // test wildcard equivalance
  std::vector<bool> is_dead(nGEQs, false);
  for (size_t i = 0; i < disjoint_wildcards.size(); i++) {
    if (disjoint_inequalities[i].num_elem() == 0)
      continue;
    
    for (size_t j = i+1; j < disjoint_wildcards.size(); j++) {
      if (disjoint_wildcards[i].num_elem() != disjoint_wildcards[j].num_elem() ||
          disjoint_hash[i].size() != disjoint_hash[j].size())
        continue;

      bool match = true;
      for (size_t k = 0; k < disjoint_hash[i].size(); k++) {
        if (disjoint_hash[i][k].second != disjoint_hash[j][k].second) {
          match = false;
          break;
        }
      }
      if (!match)
        continue;
      
      // confirm same coefficients for regular variables
      for (size_t k = 0; k < disjoint_hash[i].size(); k++) {
        for (int p = 1; p <= safeVars; p++)
          if (GEQs[disjoint_hash[i][k].first].coef[p] != GEQs[disjoint_hash[j][k].first].coef[p]) {
            match = false;
            break;
          }
        if (!match)
          break;
      }
      if (!match)
        continue;

      // now try combinatory wildcard matching
      std::vector<int> wild_map(nVars-safeVars, -1);
      for (size_t k = 0; k < disjoint_hash[i].size(); k++) {
        int e1 = disjoint_hash[i][k].first;
        int e2 = disjoint_hash[j][k].first;
        
        for (int p = 0; p < nVars-safeVars; p++)
          if (GEQs[e1].coef[p+safeVars+1] != 0) {
            if (wild_map[p] == -1) {
              for (int q = 0; q < nVars-safeVars; q++)
                if (wild_map[q] == -1 &&
                    GEQs[e2].coef[q+safeVars+1] == GEQs[e1].coef[p+safeVars+1]) {
                  wild_map[p] = q;
                  wild_map[q] = p;
                  break;
                }
              if (wild_map[p] == -1) {
                match = false;
                break;
              }
            }
            else if (GEQs[e2].coef[wild_map[p]+safeVars+1] != GEQs[e1].coef[p+safeVars+1]) {
              match = false;
              break;
            }   
          }             
        if (!match)
          break;

        for (int p = 0; p < nVars-safeVars; p++)
          if (GEQs[e2].coef[p+safeVars+1] != 0 &&
              (wild_map[p] == -1 || GEQs[e2].coef[p+safeVars+1] != GEQs[e1].coef[wild_map[p]+safeVars+1])) {
            match = false;
            break;
          }
        if (!match)
          break;
      }
      if (!match)
        continue;

      // check constants
      int dir = 0;
      for (size_t k = 0; k < disjoint_hash[i].size(); k++) {
        if (GEQs[disjoint_hash[i][k].first].coef[0] > GEQs[disjoint_hash[j][k].first].coef[0]) {
          if (dir == 0)
            dir = 1;
          else if (dir == -1) {
            match = false;
            break;
          }
        }
        else if (GEQs[disjoint_hash[i][k].first].coef[0] < GEQs[disjoint_hash[j][k].first].coef[0]) {
          if (dir == 0)
            dir = -1;
          else if (dir == 1) {
            match = false;
            break;
          }
        }
      }
      if (!match)
        continue;

      // check redness
      int red_dir = 0;
      for (size_t k = 0; k < disjoint_hash[i].size(); k++) {
        if (GEQs[disjoint_hash[i][k].first].color > GEQs[disjoint_hash[j][k].first].color) {
          if (red_dir == 0)
            red_dir = 1;
          else if (red_dir == -1) {
            match = false;
            break;
          }
        }
        else if (GEQs[disjoint_hash[i][k].first].color < GEQs[disjoint_hash[j][k].first].color) {
          if (red_dir == 0)
            red_dir = -1;
          else if (red_dir == 1) {
            match = false;
            break;
          }
        }
      }
      if (!match)
        continue;
        
      // remove redundant inequalities
      if (dir == 1 || (dir == 0 && red_dir == 1)) {
        for (size_t k = 0; k < disjoint_hash[i].size(); k++) {
          GEQs[disjoint_hash[i][k].first].coef[0] = GEQs[disjoint_hash[j][k].first].coef[0];
          GEQs[disjoint_hash[i][k].first].color = GEQs[disjoint_hash[j][k].first].color;
          is_dead[disjoint_hash[j][k].first] = true;
        }
      }
      else {
        for (size_t k = 0; k < disjoint_hash[i].size(); k++) {
          is_dead[disjoint_hash[j][k].first] = true;
        }
      }
    }
  }

  // eliminate dead inequalities
  for (int e = nGEQs-1; e >= 0; e--)
    if (is_dead[e]) {
      deleteGEQ(e);
    }
}


void initializeOmega(void) {
  if (omegaInitialized)
    return;

//   assert(sizeof(eqn)==sizeof(int)*(headerWords)+sizeof(coef_t)*(1+maxVars));
  nextWildcard = 0;
  nextKey = maxVars + 1;
  for (int i = 0; i < hashTableSize; i++)
    hashMaster[i].touched = -1;

  sprintf(wildName[1], "__alpha");
  sprintf(wildName[2], "__beta");
  sprintf(wildName[3], "__gamma");
  sprintf(wildName[4], "__delta");
  sprintf(wildName[5], "__tau");
  sprintf(wildName[6], "__sigma");
  sprintf(wildName[7], "__chi");
  sprintf(wildName[8], "__omega");
  sprintf(wildName[9], "__pi");
  sprintf(wildName[10], "__ni");
  sprintf(wildName[11], "__Alpha");
  sprintf(wildName[12], "__Beta");
  sprintf(wildName[13], "__Gamma");
  sprintf(wildName[14], "__Delta");
  sprintf(wildName[15], "__Tau");
  sprintf(wildName[16], "__Sigma");
  sprintf(wildName[17], "__Chi");
  sprintf(wildName[18], "__Omega");
  sprintf(wildName[19], "__Pi");

  omegaInitialized = 1;
}

//
// This is experimental (I would say, clinical) fact:
// If the code below is removed then simplifyProblem cycles.
//
class brainDammage {
public:
  brainDammage();
};
 
brainDammage::brainDammage() {
  initializeOmega();
}
 
static brainDammage Podgorny;

} // namespace
