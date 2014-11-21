/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   Expensive inequality elimination.

 Notes:

 History:
   03/31/09 Use BoolSet, Chun Chen
*****************************************************************************/

#include <omega/omega_core/oc_i.h>
#include <basic/boolset.h>
#include <vector>

namespace omega {

int Problem::expensiveKill() {
  int e;
  if (TRACE) fprintf(outputFile,"Performing expensive kill tests: [\n");
  if (DBUG) printProblem();
  Problem tmpProblem;
  int oldTrace = trace;
  int constraintsRemoved = 0;

  trace = 0;
  conservative++;

  for (e = nGEQs - 1; e >= 0; e--)
    if (!GEQs[e].essential) {
      if (DBUG) {
        fprintf(outputFile, "checking equation %d to see if it is redundant: ", e);
        printGEQ(&(GEQs[e]));
        fprintf(outputFile, "\n");
      }
      tmpProblem = *this;
      tmpProblem.negateGEQ(e);
      tmpProblem.varsOfInterest = 0;
      tmpProblem.nSUBs = 0;
      tmpProblem.nMemories = 0;
      tmpProblem.safeVars = 0;
      tmpProblem.variablesFreed = 0;
      tmpProblem.isTemporary = true;

      if (!tmpProblem.solve(false)) {
        if (DBUG)
          fprintf(outputFile, "redundant!\n");
        constraintsRemoved++;
        deleteGEQ(e);
      }
    }
  
  if (constraintsRemoved) {
    if (TRACE) fprintf(outputFile,"%d Constraints removed!!\n",constraintsRemoved);
  }

  trace = oldTrace;
  conservative--;
  if (TRACE) fprintf(outputFile,"] expensive kill tests done\n");
  return 1;
}

int Problem::expensiveRedKill() {
  int e;
  if (TRACE) fprintf(outputFile,"Performing expensive red kill tests: [\n");
  Problem tmpProblem;
  int oldTrace = trace;
  int constraintsRemoved = 0;

  trace = 0;
  conservative++;

  for (e = nGEQs - 1; e >= 0; e--)
    if (!GEQs[e].essential && GEQs[e].color) {
      if (DEBUG) {
        fprintf(outputFile, "checking equation %d to see if it is redundant: ", e);
        printGEQ(&(GEQs[e]));
        fprintf(outputFile, "\n");
      }
      tmpProblem = *this;
      tmpProblem.negateGEQ(e);
      tmpProblem.varsOfInterest = 0;
      tmpProblem.nSUBs = 0;
      tmpProblem.nMemories = 0;
      tmpProblem.safeVars = 0;
      tmpProblem.variablesFreed = 0;
      tmpProblem.isTemporary = true;
      tmpProblem.turnRedBlack();
      if (!tmpProblem.solve(false)) {
        constraintsRemoved++;
        deleteGEQ(e);
      }
    }
  
  if (constraintsRemoved) {
    if (TRACE) fprintf(outputFile,"%d Constraints removed!!\n",constraintsRemoved);
  }

  trace = oldTrace;
  conservative--;
  if (TRACE) fprintf(outputFile,"] expensive red kill tests done\n");
  return 1;
}


int Problem::expensiveEqualityCheck() {
  int e;
  return 1;
  if (TRACE) fprintf(outputFile,"Performing expensive equality tests: [\n");
  Problem tmpProblem;
  int oldTrace = trace;
  int equalitiesFound = 0;

  trace = 0;
  conservative++;

  for (e = nGEQs - 1; e >= 0; e--) {
    if (DEBUG) {
      fprintf(outputFile, "checking equation %d to see if it is an equality: ", e);
      printGEQ(&(GEQs[e]));
      fprintf(outputFile, "\n");
    }
    tmpProblem = *this;
    tmpProblem.GEQs[e].coef[0]--;
    tmpProblem.varsOfInterest = 0;
    tmpProblem.nSUBs = 0;
    tmpProblem.nMemories = 0;
    tmpProblem.safeVars = 0;
    tmpProblem.variablesFreed = 0;
    tmpProblem.isTemporary = true;
    if (!tmpProblem.solve(false)) {
      int neweq = newEQ();
      eqnncpy(&EQs[neweq], &GEQs[e], nVars);
      equalitiesFound++;
      addingEqualityConstraint(neweq);
    }
  }
  if (equalitiesFound) {
    if (TRACE) fprintf(outputFile,"%d Equalities found!!\n",equalitiesFound);
  }

  trace = oldTrace;
  conservative--;
  if (equalitiesFound) {
    if (!solveEQ()) return 0;
    if (!normalize()) return 0;
  }
  if (TRACE) fprintf(outputFile,"] expensive equality tests done\n");
  return 1;
}


void Problem::quickRedKill(int computeGist) {
  if (DBUG) {
    fprintf(outputFile, "in quickRedKill: [\n");
    printProblem();
  }

  noteEssential(0);
  int moreToDo = chainKill(1,0);

#ifdef NDEBUG
  if (!moreToDo) {
    if (DBUG) fprintf(outputFile, "] quickRedKill\n");
    return;
  }
#endif

  int isDead[nGEQs];
  int deadCount = 0;
  std::vector<BoolSet<> > P(nGEQs, BoolSet<>(nVars)), Z(nGEQs, BoolSet<>(nVars)), N(nGEQs, BoolSet<>(nVars));
  BoolSet<> PP, PZ, PN; /* possible Positives, possible zeros & possible negatives */
  BoolSet<> MZ; /* must zeros */
  
  int equationsToKill = 0;
  for (int e = nGEQs - 1; e >= 0; e--) {
    isDead[e] = 0;
    if (GEQs[e].color && !GEQs[e].essential) equationsToKill++;
    if (GEQs[e].color && GEQs[e].essential && !computeGist) 
      if (!moreToDo) {
        if (DBUG) fprintf(outputFile, "] quickRedKill\n");
        return;
      }
    for (int i = nVars; i >= 1; i--) {
      if (GEQs[e].coef[i] > 0)
        P[e].set(i-1);
      else if (GEQs[e].coef[i] < 0)
        N[e].set(i-1);
      else
        Z[e].set(i-1);
    }
  }

  if (!equationsToKill) 
    if (!moreToDo) {
      if (DBUG) fprintf(outputFile, "] quickRedKill\n");
      return;
    }
  
  for (int e = nGEQs - 1; e > 0; e--)
    if (!isDead[e])
      for (int e2 = e - 1; e2 >= 0; e2--)
        if (!isDead[e2]) {
          coef_t a = 0;
          int i, j;
          for (i = nVars; i > 1; i--) {
            for (j = i - 1; j > 0; j--) {
              a = (GEQs[e].coef[i] * GEQs[e2].coef[j] - GEQs[e2].coef[i] * GEQs[e].coef[j]);
              if (a != 0)
                goto foundPair;
            }
          }
          continue;

        foundPair:
          if (DEBUG) {
            fprintf(outputFile, "found two equations to combine, i = %s, ", variable(i));
            fprintf(outputFile, "j = %s, alpha = " coef_fmt "\n", variable(j), a);
            printGEQ(&(GEQs[e]));
            fprintf(outputFile, "\n");
            printGEQ(&(GEQs[e2]));
            fprintf(outputFile, "\n");
          }

          MZ = (Z[e] & Z[e2]);
          PZ = MZ |  (P[e] & N[e2]) | (N[e] & P[e2]);
          PP = P[e] | P[e2];
          PN = N[e] | N[e2];

          for (int e3 = nGEQs - 1; e3 >= 0; e3--)
            if (e3 != e && e3 != e2 && GEQs[e3].color && !GEQs[e3].essential) {
              coef_t alpha1, alpha2, alpha3;

              if (!PZ.imply(Z[e3]) || MZ.imply(~Z[e3])) continue;
              if (!PP.imply(P[e3]) || !PN.imply(N[e3])) continue;

              if (a > 0) {
                alpha1 = GEQs[e2].coef[j] * GEQs[e3].coef[i] - GEQs[e2].coef[i] * GEQs[e3].coef[j];
                alpha2 = -(GEQs[e].coef[j] * GEQs[e3].coef[i] - GEQs[e].coef[i] * GEQs[e3].coef[j]);
                alpha3 = a;
              }
              else {
                alpha1 = -(GEQs[e2].coef[j] * GEQs[e3].coef[i] - GEQs[e2].coef[i] * GEQs[e3].coef[j]);
                alpha2 = -(-(GEQs[e].coef[j] * GEQs[e3].coef[i] - GEQs[e].coef[i] * GEQs[e3].coef[j]));
                alpha3 = -a;
              }
    
              if (alpha1 > 0 && alpha2 > 0) {
                if (DEBUG) {
                  fprintf(outputFile, "alpha1 = " coef_fmt ", alpha2 = " coef_fmt "; comparing against: ", alpha1, alpha2);
                  printGEQ(&(GEQs[e3]));
                  fprintf(outputFile, "\n");
                }
                coef_t c;
                int k;
                for (k = nVars; k >= 0; k--) {
                  c = alpha1 * GEQs[e].coef[k] + alpha2 * GEQs[e2].coef[k];
                  if (DEBUG) {
                    if (k>0) 
                      fprintf(outputFile, " %s: " coef_fmt ", " coef_fmt "\n", variable(k), c, alpha3 * GEQs[e3].coef[k]);
                    else fprintf(outputFile, " constant: " coef_fmt ", " coef_fmt "\n", c, alpha3 * GEQs[e3].coef[k]);
                  }
                  if (c != alpha3 * GEQs[e3].coef[k])
                    break;
                }
                if (k < 0 || (k == 0 && c < alpha3 * (GEQs[e3].coef[k]+1))) {
                  if (DEBUG) {
                    deadCount++;
                    fprintf(outputFile, "red equation#%d is dead (%d dead so far, %d remain)\n", e3, deadCount, nGEQs - deadCount);
                    printGEQ(&(GEQs[e]));
                    fprintf(outputFile, "\n");
                    printGEQ(&(GEQs[e2]));
                    fprintf(outputFile, "\n");
                    printGEQ(&(GEQs[e3]));
                    fprintf(outputFile, "\n");
                    assert(moreToDo);
                  }
                  isDead[e3] = 1;
                }
              }
            }
        }
  
  for (int e = nGEQs - 1; e >= 0; e--)
    if (isDead[e])
      deleteGEQ(e);

  if (DBUG) {
    fprintf(outputFile,"] quickRedKill\n");
    printProblem();
  }
}

} // namespace
