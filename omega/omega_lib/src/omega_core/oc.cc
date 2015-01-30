/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
  Simplify a problem.

 Notes:

 History:
   12/10/06 Improved gist function, by Chun Chen.
*****************************************************************************/

#include <omega/omega_core/oc_i.h>

namespace omega {

eqn SUBs[maxVars+1];
Memory redMemory[maxVars+1];

int Problem::reduceProblem() {
  int result;
  checkVars(nVars+1);
  assert(omegaInitialized);
  if (nVars > nEQs + 3 * safeVars)
    freeEliminations(safeVars);

  check();
  if (!mayBeRed && nSUBs == 0 && safeVars == 0) {
    result = solve(OC_SOLVE_UNKNOWN);
    nGEQs = 0;
    nEQs = 0;
    nSUBs = 0;
    nMemories = 0;
    if (!result) {
      int e = newEQ();
      assert(e == 0);
      eqnnzero(&EQs[0], nVars);
      EQs[0].color = EQ_BLACK;
      EQs[0].coef[0] = 1;
    }
    check();
    return result;
  }
  return solve(OC_SOLVE_SIMPLIFY);
}


int Problem::simplifyProblem(int verify, int subs, int redundantElimination) {
  checkVars(nVars+1);
  assert(omegaInitialized);
  setInternals();
  check();
  if (!reduceProblem())  goto returnFalse;
  if (verify) {
    addingOuterEqualities++;
    int r = verifyProblem();
    addingOuterEqualities--;
    if (!r) goto returnFalse;
    if (nEQs) { // found some equality constraints during verification
      int numRed = 0;
      if (mayBeRed) 
        for (int e = nGEQs - 1; e >= 0; e--) if (GEQs[e].color == EQ_RED) numRed++;
      if (mayBeRed && nVars == safeVars && numRed == 1)
        nEQs = 0; // discard them
      else if (!reduceProblem()) {
        assert(0 && "Added equality constraint to verified problem generates false");
      }
    }
  }
  if (redundantElimination) {
    if (redundantElimination > 1) {
      if (!expensiveEqualityCheck()) goto returnFalse;
    }
    if (!quickKill(0)) goto returnFalse;
    if (redundantElimination > 1) {
      if (!expensiveKill()) goto returnFalse;
    }
  }
  resurrectSubs();
  if (redundantElimination) 
    simplifyStrideConstraints();
  if (redundantElimination > 2 && safeVars < nVars) {
    if (!quickKill(0)) goto returnFalse;
    return simplifyProblem(verify, subs, redundantElimination-2);
  }
  setExternals();
  assert(nMemories == 0);
  return (1);
  
returnFalse:
  nGEQs = 0;
  nEQs = 0;
  resurrectSubs();
  nGEQs = 0;
  nEQs = 0;
  int neweq = newEQ();
  assert(neweq == 0);
  eqnnzero(&EQs[neweq], nVars);
  EQs[neweq].color = EQ_BLACK;
  EQs[neweq].coef[0] = 1;
  nMemories = 0;
  return 0;
}

int Problem::simplifyApproximate(bool strides_allowed) {
  int result;
  checkVars(nVars+1);
  assert(inApproximateMode  == 0);

  inApproximateMode = 1;
  inStridesAllowedMode = strides_allowed;
  if (TRACE)
    fprintf(outputFile, "Entering Approximate Mode [\n");

  assert(omegaInitialized);
  result = simplifyProblem(0,0,0);

  while (result && !strides_allowed && nVars > safeVars) {
    int e;
    for (e = nGEQs - 1; e >= 0; e--) 
      if (GEQs[e].coef[nVars]) deleteGEQ(e);
    for (e = nEQs - 1; e >= 0; e--) 
      if (EQs[e].coef[nVars]) deleteEQ(e);
    nVars--;
    result = simplifyProblem(0,0,0);
  }

  if (TRACE)
    fprintf(outputFile, "] Leaving Approximate Mode\n");
    
  assert(inApproximateMode  == 1);
  inApproximateMode=0;
  inStridesAllowedMode = 0;

  assert(nMemories == 0);
  return (result);
}




/*
 * Return 1 if red equations constrain the set of possible
 * solutions. We assume that there are solutions to the black
 * equations by themselves, so if there is no solution to the combined
 * problem, we return 1.
 */

#ifdef GIST_CHECK
Problem full_answer, context;
Problem redProblem;
#endif

redCheck Problem::redSimplifyProblem(int effort, int computeGist) {
  int result;
  int e;

  checkVars(nVars+1);
  assert(mayBeRed >= 0);
  mayBeRed++;

  assert(omegaInitialized);
  if (TRACE) {
    fprintf(outputFile, "Checking for red equations:\n");
    printProblem();
  }
  setInternals();

#ifdef GIST_CHECK
  int r1,r2;
  if (TRACE) 
    fprintf(outputFile,"Set-up for gist invariant checking[\n");
  redProblem = *this;
  redProblem.check();
  full_answer = *this;
  full_answer.check();
  full_answer.turnRedBlack();
  full_answer.check();
  r1 = full_answer.simplifyProblem(1,0,1);
  full_answer.check();
  if (DBUG) fprintf(outputFile,"Simplifying context [\n");
  context = *this; 
  context.check();
  context.deleteRed();
  context.check();
  r2 = context.simplifyProblem(1,0,1);
  context.check();
  if (DBUG) fprintf(outputFile,"] Simplifying context\n");

  if (!r2 && TRACE) fprintf(outputFile, "WARNING: Gist context is false!\n");
  if (TRACE) 
    fprintf(outputFile,"] Set-up for gist invariant checking done\n");
#endif

  // Save known integer modular equations, -- by chun 12/10/2006
  eqn ModularEQs[nEQs];
  int nModularEQs = 0;
  int old_nVars = nVars;
  for (int e = 0; e < nEQs; e++)
    if (EQs[e].color != EQ_RED)
      for (int i = safeVars+1; i <= nVars; i++)
        if (EQs[e].coef[i] != 0) {
          eqnncpy(&(ModularEQs[nModularEQs++]), &(EQs[e]), nVars);
          break;
        }
  
  
  if (solveEQ() == false) {
    if (TRACE)
      fprintf(outputFile, "Gist is FALSE\n");
    if (computeGist) {
      nMemories = 0;
      nGEQs = 0;
      nEQs = 0;
      resurrectSubs();
      nGEQs = 0;
      nEQs = 0;
      int neweq = newEQ();
      assert(neweq == 0);
      eqnnzero(&EQs[neweq], nVars);
      EQs[neweq].color = EQ_RED;
      EQs[neweq].coef[0] = 1;
    }
    mayBeRed--;
    return redFalse;
  }

  if (!computeGist && nMemories) 
    return redConstraints;
  if (normalize() == normalize_false) {
    if (TRACE)
      fprintf(outputFile, "Gist is FALSE\n");
    if (computeGist) {
      nGEQs = 0;
      nEQs = 0;
      resurrectSubs();
      nMemories = 0;
      nGEQs = 0;
      nEQs = 0;
      int neweq = newEQ();
      assert(neweq == 0);
      eqnnzero(&EQs[neweq], nVars);
      EQs[neweq].color = EQ_RED;
      EQs[neweq].coef[0] = 1;
    }
    mayBeRed--;
    return redFalse;
  }

  result = 0;
  for (e = nGEQs - 1; e >= 0; e--) if (GEQs[e].color == EQ_RED) result = 1;
  for (e = nEQs - 1; e >= 0; e--) if (EQs[e].color == EQ_RED) result = 1;
  if (nMemories) result = 1;
  if (!result) {
    if (computeGist) {
      nGEQs = 0;
      nEQs = 0;
      resurrectSubs();
      nGEQs = 0;
      nMemories = 0;
      nEQs = 0;
    }
    mayBeRed--;
    return noRed;
  }

  result = simplifyProblem(effort?1:0,1,0);
#ifdef GIST_CHECK
  if (!r1 && TRACE && result) 
    fprintf(outputFile, "Gist is False but not detected\n");
#endif
  if (!result) {
    if (TRACE)
      fprintf(outputFile, "Gist is FALSE\n");
    if (computeGist) {
      nGEQs = 0;
      nEQs = 0;
      resurrectSubs();
      nGEQs = 0;
      nEQs = 0;
      int neweq = newEQ();
      assert(neweq == 0);
      nMemories = 0;
      eqnnzero(&EQs[neweq], nVars);
      EQs[neweq].color = EQ_RED;
      EQs[neweq].coef[0] = 1;
    }
    mayBeRed--;
    return redFalse;
  }

  freeRedEliminations();

  result = 0;
  for (e = nGEQs - 1; e >= 0; e--) if (GEQs[e].color == EQ_RED) result = 1;
  for (e = nEQs - 1; e >= 0; e--) if (EQs[e].color == EQ_RED) result = 1;
  if (nMemories) result = 1;
  if (!result) {
    if (computeGist) {
      nGEQs = 0;
      nEQs = 0;
      resurrectSubs();
      nGEQs = 0;
      nMemories = 0;
      nEQs = 0;
    }
    mayBeRed--;
    return noRed;
  }

  if (effort && (computeGist || !nMemories)) {
    if (TRACE)
      fprintf(outputFile, "*** Doing potentially expensive elimination tests for red equations [\n");
    quickRedKill(computeGist);
    checkGistInvariant();
    result = nMemories;
    for (e = nGEQs - 1; e >= 0; e--) if (GEQs[e].color == EQ_RED) result++;
    for (e = nEQs - 1; e >= 0; e--) if (EQs[e].color == EQ_RED) result++;
    if (result && effort > 1 && (computeGist || !nMemories))  {
      expensiveRedKill();
      result = nMemories;
      for (e = nGEQs-1; e >= 0; e--) if (GEQs[e].color == EQ_RED) result++;
      for (e = nEQs-1; e >= 0; e--) if (EQs[e].color == EQ_RED) result++;
    }
 
    if (!result)  {
      if (TRACE)
        fprintf(outputFile, "]******************** Redudant Red Equations eliminated!!\n");
      if (computeGist) {
        nGEQs = 0;
        nEQs = 0;
        resurrectSubs();
        nGEQs = 0;
        nMemories = 0;
        nEQs = 0;
      }
      mayBeRed--;
      return noRed;
    }
     
    if (TRACE) fprintf(outputFile, "]******************** Red Equations remain\n");
    if (DEBUG) printProblem();
  }
  
  if (computeGist) {
    resurrectSubs(); 
    cleanoutWildcards();


    // Restore saved modular equations into EQs without affecting the problem
    if (nEQs+nModularEQs > allocEQs) {
      allocEQs = padEQs(allocEQs, nEQs+nModularEQs);
      eqn *new_eqs = new eqn[allocEQs];
      for (int e = 0; e < nEQs; e++)
        eqnncpy(&(new_eqs[e]), &(EQs[e]), nVars);
      delete[] EQs;
      EQs= new_eqs;
    }
   
    for (int e = 0; e < nModularEQs; e++) {
      eqnncpy(&(EQs[nEQs+e]), &(ModularEQs[e]), old_nVars);
      EQs[nEQs+e].color = EQ_RED;
      Tuple<coef_t> t(safeVars);
      for (int i = 1; i <= safeVars; i++)
        t[i] = ModularEQs[e].coef[var[i]];
      for (int i = 1; i <= safeVars; i++)
        EQs[nEQs+e].coef[i] = t[i];
    }
      

    // Now simplify modular equations using Chinese remainder theorem -- by chun 12/10/2006
    for (int e = 0; e < nEQs; e++)
      if (EQs[e].color == EQ_RED) {
        int wild_pos = -1;
        for (int i = safeVars+1; i <= nVars; i++)
          if (EQs[e].coef[i] != 0) {
            wild_pos = i;
            break;
          }

        if (wild_pos == -1)
          continue;

        for (int e2 = e+1; e2 < nEQs+nModularEQs; e2++)
          if (EQs[e2].color == EQ_RED) {
            int wild_pos2 = -1;
            for (int i = safeVars+1; i <= ((e2<nEQs)?nVars:old_nVars); i++)
              if (EQs[e2].coef[i] != 0) {
                wild_pos2 = i;
                break;
              }

            if (wild_pos2 == -1)
              continue;

            coef_t g = gcd(abs(EQs[e].coef[wild_pos]), abs(EQs[e2].coef[wild_pos2]));
            coef_t g2 = 1;
            coef_t g3;
            EQs[e].color = EQs[e2].color = EQ_BLACK;
            while ((g3 = factor(g)) != 1) {
              coef_t gg = g2 * g3;
              g = g/g3;
              
              bool match = true;
              coef_t c = EQs[e].coef[0];
              coef_t c2 = EQs[e2].coef[0];
              bool change_sign = false;
              for (int i = 1; i <= safeVars; i++) {
                coef_t coef = int_mod_hat(EQs[e].coef[i], gg);
                coef_t coef2 = int_mod_hat(EQs[e2].coef[i], gg);

                if (coef == 0 && coef2 == 0)
                  continue;
              
                if (change_sign && coef == -coef2)
                  continue;

                if (!change_sign) {
                  if (coef == coef2)
                    continue;
                  else if (coef == -coef2) {
                    change_sign = true;
                    continue;
                  }
                }
                
                if (coef != 0) {
                  coef_t t = query_variable_mod(i, gg/gcd(abs(coef), gg), EQ_RED, nModularEQs, old_nVars);
                  if (t == posInfinity) {
                    match = false;
                    break;
                  }
                
                  c += coef * t;
                }
                if (coef2 != 0) {
                  coef_t t = query_variable_mod(i, gg/gcd(abs(coef2), gg), EQ_RED, nModularEQs, old_nVars);
                  if (t == posInfinity) {
                    match = false;
                    break;
                  }

                  c2 += coef2 * t;
                }
              }
              if ((change_sign && int_mod_hat(c, gg) != -int_mod_hat(c2, gg)) ||
                  (!change_sign && int_mod_hat(c, gg) != int_mod_hat(c2, gg)))
                match = false;

              if (match)
                g2 = gg;
            }
            EQs[e].color = EQs[e2].color = EQ_RED;

            if (g2 == 1)
              continue;
            
            if (g2 == abs(EQs[e].coef[wild_pos])) {
              EQs[e].color = EQ_BLACK;
              break;
            }
            else if (e2 < nEQs && g2 == abs(EQs[e2].coef[wild_pos2]))
              EQs[e2].color = EQ_BLACK;
            else {
              coef_t g4 = abs(EQs[e].coef[wild_pos])/g2;
              while (lcm(g2, g4) != abs(EQs[e].coef[wild_pos])) {
                assert(lcm(g2, g4) < abs(EQs[e].coef[wild_pos]));
                g4 *= abs(EQs[e].coef[wild_pos])/lcm(g2, g4);
              }
              
              for (int i = 0; i <= safeVars; i++)
                EQs[e].coef[i] = int_mod_hat(EQs[e].coef[i], g4);
              EQs[e].coef[wild_pos] = (EQs[e].coef[wild_pos]>0?1:-1)*g4;
            }
          }
      }
    
    deleteBlack();
  }
  
  setExternals();
  mayBeRed--;
  assert(nMemories == 0);
  return redConstraints;
}


void Problem::convertEQstoGEQs(bool excludeStrides) {
  int i;
  int e;
  if (DBUG)
    fprintf(outputFile, "Converting all EQs to GEQs\n");
  simplifyProblem(0,0,0);
  for(e=0;e<nEQs;e++) {
    bool isStride = 0;
    int e2 = newGEQ();
    if (excludeStrides)
      for(i = safeVars+1; i <= nVars; i++)
        isStride = isStride || (EQs[e].coef[i] != 0);
    if (isStride) continue;
    eqnncpy(&GEQs[e2], &EQs[e], nVars);
    GEQs[e2].touched = 1;
    e2 = newGEQ();
    eqnncpy(&GEQs[e2], &EQs[e], nVars);
    GEQs[e2].touched = 1;
    for (i = 0; i <= nVars; i++)
      GEQs[e2].coef[i] = -GEQs[e2].coef[i];
  }
  // If we have eliminated all EQs, can set nEQs to 0
  // If some strides are left, we don't know the position of them in the EQs
  // array, so decreasing nEQs might remove wrong EQs -- we just leave them
  // all in. (could sort the EQs to move strides to the front, but too hard.)
  if (!excludeStrides) nEQs=0; 
  if (DBUG)
    printProblem();
}


void Problem::convertEQtoGEQs(int eq) {
  int i;
  if (DBUG)
    fprintf(outputFile, "Converting EQ %d to GEQs\n",eq);
  int e2 = newGEQ();
  eqnncpy(&GEQs[e2], &EQs[eq], nVars);
  GEQs[e2].touched = 1;
  e2 = newGEQ();
  eqnncpy(&GEQs[e2], &EQs[eq], nVars);
  GEQs[e2].touched = 1;
  for (i = 0; i <= nVars; i++)
    GEQs[e2].coef[i] = -GEQs[e2].coef[i];
  if (DBUG)
    printProblem();
}


/*
 * Calculate value of variable modulo integer from problem's equation
 * set plus additional saved modular equations embedded in the same
 * EQs array (hinted by nModularEQs) if available. If there is no
 * solution, return posInfinity.
 */
coef_t Problem::query_variable_mod(int v, coef_t factor, int color, int nModularEQs, int nModularVars) const {
  if (safeVars < v)
    return posInfinity;
  
  Tuple<bool> working_on(safeVars);
  for (int i = 1; i <= safeVars; i++)
    working_on[i] = false;

  return query_variable_mod(v, factor, color, nModularEQs, nModularVars, working_on);
}

coef_t Problem::query_variable_mod(int v, coef_t factor, int color, int nModularEQs, int nModularVars, Tuple<bool> &working_on) const {
  working_on[v] = true;

  for (int e = 0; e < nEQs+nModularEQs; e++)
    if (EQs[e].color == color) {
      coef_t coef = int_mod_hat(EQs[e].coef[v], factor);
      if (abs(coef) != 1)
        continue;

      bool wild_factored = true;
      for (int i = safeVars+1; i <= ((e<nEQs)?nVars:nModularVars); i++)
        if (int_mod_hat(EQs[e].coef[i], factor) != 0) {
          wild_factored = false;
          break;
        }
      if (!wild_factored)
        continue;
      
      coef_t result = 0;
      for (int i = 1; i <= safeVars; i++)
        if (i != v) {
          coef_t p = int_mod_hat(EQs[e].coef[i], factor);

          if (p == 0)
            continue;

          if (working_on[i] == true) {
            result = posInfinity;
            break;
          }

          coef_t q = query_variable_mod(i, factor, color, nModularEQs, nModularVars, working_on);
          if (q == posInfinity) {
            result = posInfinity;
            break;
          }
          result += p*q;
        }

      if (result != posInfinity) {
        result += EQs[e].coef[0];
        if (coef == 1)
          result = -result;
        working_on[v] = false;

        return int_mod_hat(result, factor);
      }
    }

  working_on[v] = false;
  return posInfinity;
}          



#ifdef GIST_CHECK
enum compareAnswer {apparentlyEqual, mightNotBeEqual, NotEqual};

static compareAnswer checkEquiv(Problem *p1, Problem *p2) {
  int r1,r2;

  p1->check();
  p2->check();
  p1->resurrectSubs(); 
  p2->resurrectSubs(); 
  p1->check();
  p2->check();
  p1->putVariablesInStandardOrder(); 
  p2->putVariablesInStandardOrder(); 
  p1->check();
  p2->check();
  p1->ordered_elimination(0); 
  p2->ordered_elimination(0); 
  p1->check();
  p2->check();
  r1 = p1->simplifyProblem(1,1,0);
  r2 = p2->simplifyProblem(1,1,0);
  p1->check();
  p2->check();

  if (!r1 || !r2) {
    if (r1 == r2) return apparentlyEqual;
    return NotEqual;
  }
  if (p1->nVars != p2->nVars 
      || p1->nGEQs != p2->nGEQs 
      || p1->nSUBs != p2->nSUBs
      || p1->checkSum()  != p2->checkSum()) {
    r1 = p1->simplifyProblem(0,1,1);
    r2 = p2->simplifyProblem(0,1,1);
    assert(r1 && r2);
    p1->check();
    p2->check();
    if (p1->nVars != p2->nVars 
        || p1->nGEQs != p2->nGEQs 
        || p1->nSUBs != p2->nSUBs
        || p1->checkSum()  != p2->checkSum()) {
      r1 = p1->simplifyProblem(0,1,2);
      r2 = p2->simplifyProblem(0,1,2);
      p1->check();
      p2->check();
      assert(r1 && r2);
      if (p1->nVars != p2->nVars 
          || p1->nGEQs != p2->nGEQs 
          || p1->nSUBs != p2->nSUBs
          || p1->checkSum()  != p2->checkSum()) {
        p1->check();
        p2->check();
        p1->resurrectSubs(); 
        p2->resurrectSubs(); 
        p1->check();
        p2->check();
        p1->putVariablesInStandardOrder(); 
        p2->putVariablesInStandardOrder(); 
        p1->check();
        p2->check();
        p1->ordered_elimination(0); 
        p2->ordered_elimination(0); 
        p1->check();
        p2->check();
        r1 = p1->simplifyProblem(1,1,0);
        r2 = p2->simplifyProblem(1,1,0);
        p1->check();
        p2->check();
      }
    }
  }
  
  if (p1->nVars != p2->nVars 
      || p1->nSUBs != p2->nSUBs
      || p1->nGEQs != p2->nGEQs 
      || p1->nSUBs != p2->nSUBs) return NotEqual;
  if (p1->checkSum()  != p2->checkSum()) return mightNotBeEqual;
  return apparentlyEqual;
}
#endif

void Problem::checkGistInvariant() const {
#ifdef GIST_CHECK
  Problem new_answer;
  int r;

  check();
  fullAnswer.check();
  context.check();

  if (safeVars < nVars) {
    if (DBUG) {
      fprintf(outputFile,"Can't check gist invariant due to wildcards\n");
      printProblem();
    }
    return;
  }
  if (DBUG) {
    fprintf(outputFile,"Checking gist invariant on: [\n");
    printProblem();
  }

  new_answer = *this;
  new_answer->resurrectSubs();
  new_answer->cleanoutWildcards();
  if (DEBUG) {
    fprintf(outputFile,"which is: \n");
    printProblem();
  }
  deleteBlack(&new_answer);
  turnRedBlack(&new_answer);
  if (DEBUG) {
    fprintf(outputFile,"Black version of answer: \n");
    printProblem(&new_answer);
  }
  problem_merge(&new_answer,&context);

  r = checkEquiv(&full_answer,&new_answer);
  if (r != apparentlyEqual) {
    fprintf(outputFile,"GIST INVARIANT REQUIRES MANUAL CHECK:[\n");
    fprintf(outputFile,"Original problem:\n");
    printProblem(&redProblem);

    fprintf(outputFile,"Context:\n");
    printProblem(&context);

    fprintf(outputFile,"Computed gist:\n");
    printProblem();

    fprintf(outputFile,"Combined answer:\n");
    printProblem(&full_answer);

    fprintf(outputFile,"Context && red constraints:\n");
    printProblem(&new_answer);
    fprintf(outputFile,"]\n");
  }
  
  if (DBUG) {
    fprintf(outputFile,"] Done checking gist invariant on\n");
  }
#endif
}

} // namespace
