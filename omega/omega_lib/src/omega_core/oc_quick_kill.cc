/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   Quick inequality elimination.

 Notes:

 History:
   03/31/09 Use BoolSet, Chun Chen
*****************************************************************************/

#include <omega/omega_core/oc_i.h>
#include <vector>
#include <algorithm>
#include <basic/boolset.h>

namespace omega {

int Problem::combineToTighten() {
  int effort = min(12+5*(nVars-safeVars),23);

  if (DBUG) {
    fprintf(outputFile, "\nin combineToTighten (%d,%d):\n",effort,nGEQs);
    printProblem();
    fprintf(outputFile, "\n");
  }
  if (nGEQs > effort) {
    if (TRACE) {
      fprintf(outputFile, "too complicated to tighten\n");
    }
    return 1;
  }

  for(int e = 1; e < nGEQs; e++) {
    for(int e2 = 0; e2 < e; e2++) {
      coef_t g = 0;

      bool has_wildcard = false;
      bool has_wildcard2 = false;
      for (int i = nVars; i > safeVars; i--) {
        coef_t a = GEQs[e].coef[i];
        coef_t b = GEQs[e2].coef[i];
        g = gcd(g, abs(a+b));
        if (a != 0)
          has_wildcard = true;
        if (b != 0)
          has_wildcard2 = true;
      }
        
      coef_t c, c2;
      if ((has_wildcard && !has_wildcard2) || (!has_wildcard && has_wildcard2))
        c = 0;
      else
        c = -1;
      for (int i = safeVars; i >= 1; i--) {
        coef_t a = GEQs[e].coef[i];
        coef_t b = GEQs[e2].coef[i];
        if (a != 0 || b != 0) {
          g = gcd(g, abs(a+b));

          if (c < 0) {
            if (g == 1)
              break;
          }
          else if ((a>0 && b<0) || (a<0 && b>0)) {
            if (c == 0) {
              try {
                coef_t prod = lcm(abs(a), abs(b));
                c = prod/abs(a);
                c2 = prod/abs(b);
              }
              catch (std::overflow_error) {
                c = -1;
              }
            }
            else {
              if (c*a+c2*b != 0)
                c = -1;
            }
          }
          else {
            c = -1;
          }
        }
      }

      bool done_unit_combine = false;
      if (g > 1 && (GEQs[e].coef[0] + GEQs[e2].coef[0]) % g != 0) {
        int e3 = newGEQ();
        for(int i = nVars; i >= 1; i--) {
          GEQs[e3].coef[i] = (GEQs[e].coef[i] + GEQs[e2].coef[i])/g;
        }
        GEQs[e3].coef[0] = int_div(GEQs[e].coef[0] + GEQs[e2].coef[0], g);
        GEQs[e3].color = GEQs[e].color || GEQs[e2].color;
        GEQs[e3].touched = 1;
        if (DBUG) {
          fprintf(outputFile,  "Combined     ");
          printGEQ(&GEQs[e]);
          fprintf(outputFile,"\n         and ");
          printGEQ(&GEQs[e2]);
          fprintf(outputFile,"\n to get #%d: ",e3);
          printGEQ(&GEQs[e3]);
          fprintf(outputFile,"\n\n");
        }

        done_unit_combine = true;
        if (nGEQs > effort+5 || nGEQs > maxmaxGEQs-10) goto doneCombining;
      }

      if (c > 0 && !(c == 1 && c2 == 1 && done_unit_combine)) {
        bool still_has_wildcard = false;
        coef_t p[nVars-safeVars];
        for (int i = nVars; i > safeVars; i--) {
          p[i-safeVars-1] = c * GEQs[e].coef[i] + c2 * GEQs[e2].coef[i];
          if (p[i-safeVars-1] != 0)
            still_has_wildcard = true;
        }
        if (still_has_wildcard) {
          int e3 = newGEQ();
          for(int i = nVars; i > safeVars; i--)
            GEQs[e3].coef[i] = p[i-safeVars-1];
          for (int i = safeVars; i > 0; i--)
            GEQs[e3].coef[i] = 0;          
          GEQs[e3].coef[0] = c * GEQs[e].coef[0] + c2 * GEQs[e2].coef[0];
          GEQs[e3].color = GEQs[e].color || GEQs[e2].color;
          GEQs[e3].touched = 1;
          if (DBUG) {
            fprintf(outputFile,  "Additionally combined     ");
            printGEQ(&GEQs[e]);
            fprintf(outputFile,"\n         and ");
            printGEQ(&GEQs[e2]);
            fprintf(outputFile,"\n to get #%d: ",e3);
            printGEQ(&GEQs[e3]);
            fprintf(outputFile,"\n\n");
          }

          if (nGEQs > effort+5 || nGEQs > maxmaxGEQs-10) goto doneCombining;
        }
      }      
    }
  }

doneCombining:
  if (normalize() == normalize_false) return 0;
  while (nEQs) {
    if (!solveEQ()) return 0;
    if (normalize() == normalize_false) return 0;
  }
  return 1;
}


void Problem::noteEssential(int onlyWildcards) {
  for (int e = nGEQs - 1; e >= 0; e--) {
    GEQs[e].essential = 0;
    GEQs[e].varCount = 0;
  }
  if (onlyWildcards) {
    for (int e = nGEQs - 1; e >= 0; e--) {
      GEQs[e].essential = 1;
      for (int i = nVars; i > safeVars; i--) 
        if (GEQs[e].coef[i] < -1 || GEQs[e].coef[i] > 1) {
          GEQs[e].essential = 0;
          break;
        }
    }
  }
  for (int i = nVars; i >= 1; i--) {
    int onlyLB = -1;
    int onlyUB = -1;
    for (int e = nGEQs - 1; e >= 0; e--) 
      if (GEQs[e].coef[i] > 0) {
        GEQs[e].varCount ++;
        if (onlyLB == -1) onlyLB = e;
        else onlyLB = -2;
      }
      else if (GEQs[e].coef[i] < 0) {
        GEQs[e].varCount ++;
        if (onlyUB == -1) onlyUB = e;
        else onlyUB = -2;
      }
    if (onlyUB >= 0) {
      if (DBUG) {
        fprintf(outputFile,"only UB: ");
        printGEQ(&GEQs[onlyUB]);
        fprintf(outputFile,"\n");
      }
      GEQs[onlyUB].essential = 1;
    }
    if (onlyLB >= 0) {
      if (DBUG) {
        fprintf(outputFile,"only LB: ");
        printGEQ(&GEQs[onlyLB]);
        fprintf(outputFile,"\n");
      }
      GEQs[onlyLB].essential = 1;
    }
  }
  for (int e = nGEQs - 1; e >= 0; e--) 
    if (!GEQs[e].essential && GEQs[e].varCount > 1) {
      int i1,i2,i3;
      for (i1 = nVars; i1 >= 1; i1--) if (GEQs[e].coef[i1]) break;
      for (i2 = i1-1; i2 >= 1; i2--)  if (GEQs[e].coef[i2]) break;
      for (i3 = i2-1; i3 >= 1; i3--)  if (GEQs[e].coef[i3]) break;
      assert(i2 >= 1);
      int e2;
      for (e2 = nGEQs - 1; e2 >= 0; e2--)
        if (e!=e2) {
          coef_t crossProduct;
          crossProduct = GEQs[e].coef[i1]*GEQs[e2].coef[i1];
          crossProduct += GEQs[e].coef[i2]*GEQs[e2].coef[i2];
          for (int i = i3; i >= 1; i--)
            if (GEQs[e2].coef[i])
              crossProduct += GEQs[e].coef[i]*GEQs[e2].coef[i];
          if (crossProduct > 0) {
            if (DBUG) fprintf(outputFile,"Cross product of %d and %d is " coef_fmt "\n", e, e2, crossProduct);
            break;
          }
        }
      if (e2 < 0) GEQs[e].essential = 1;
    }
  if (DBUG) {
    fprintf(outputFile,"Computed essential equations\n");
    fprintf(outputFile,"essential equations:\n");
    for (int e = 0; e < nGEQs; e++)
      if (GEQs[e].essential) {
        printGEQ(&GEQs[e]);
        fprintf(outputFile,"\n");
      }
    fprintf(outputFile,"potentially redundant equations:\n");
    for (int e = 0; e < nGEQs; e++)
      if (!GEQs[e].essential) {
        printGEQ(&GEQs[e]);
        fprintf(outputFile,"\n");
      }
  }
}


int Problem::findDifference(int e, int &v1, int &v2) {
  // if 1 returned, eqn E is of form v1 -coef >= v2 
  for(v1=1;v1<=nVars;v1++)
    if (GEQs[e].coef[v1]) break;
  for(v2=v1+1;v2<=nVars;v2++)
    if (GEQs[e].coef[v2]) break;
  if (v2 > nVars) {
    if (GEQs[e].coef[v1] == -1) {
      v2 = v1;
      v1 = 0; 
      return 1;
    }
    if (GEQs[e].coef[v1] == 1) {
      v2 = 0;
      return 1;
    }
    return 0;
  }
  if (GEQs[e].coef[v1] * GEQs[e].coef[v2] != -1)  return 0;
  if (GEQs[e].coef[v1] < 0) std::swap(v1,v2);
  return 1;
}


namespace {
  struct succListStruct {
    int    num;
    int    notEssential;
    int    var[maxVars];
    coef_t diff[maxVars];
    int    eqn[maxVars];
  };
}
  

int Problem::chainKill(int color, int onlyWildcards) {
  int v1,v2,e;
  int essentialPred[maxVars];
  int redundant[maxmaxGEQs];
  int inChain[maxVars];
  int goodStartingPoint[maxVars];
  int tryToEliminate[maxmaxGEQs];
  int triedDoubleKill = 0;

  succListStruct succ[maxVars];

restart:

  int anyToKill = 0;
  int anyKilled = 0;
  int canHandle = 0;
   
  for(v1=0;v1<=nVars;v1++) {
    succ[v1].num = 0;
    succ[v1].notEssential = 0;
    goodStartingPoint[v1] = 0;
    inChain[v1] = -1;
    essentialPred[v1] = 0;
  }

  int essentialEquations = 0;
  for (e = 0; e < nGEQs; e++) {
    redundant[e] = 0;
    tryToEliminate[e] = !GEQs[e].essential;
    if (GEQs[e].essential) essentialEquations++;
    if (color && !GEQs[e].color) tryToEliminate[e] = 0;
  }
  if (essentialEquations == nGEQs) return 0;
  if (2*essentialEquations < nVars) return 1;
   
  for (e = 0; e < nGEQs; e++) 
    if (tryToEliminate[e] && GEQs[e].varCount <= 2 && findDifference(e,v1,v2)) {
      assert(v1 == 0 || GEQs[e].coef[v1] == 1);
      assert(v2 == 0 || GEQs[e].coef[v2] == -1);
      succ[v2].notEssential++;
      int s = succ[v2].num++;
      succ[v2].eqn[s] = e;
      succ[v2].var[s] = v1;
      succ[v2].diff[s] = -GEQs[e].coef[0];
      goodStartingPoint[v2] = 1;
      anyToKill++;
      canHandle++;
    }
  if (!anyToKill) {
    return canHandle < nGEQs;
  }
  for (e = 0; e < nGEQs; e++) 
    if (!tryToEliminate[e] && GEQs[e].varCount <= 2 && findDifference(e,v1,v2)) {
      assert(v1 == 0 || GEQs[e].coef[v1] == 1);
      assert(v2 == 0 || GEQs[e].coef[v2] == -1);
      int s = succ[v2].num++;
      essentialPred[v1]++;
      succ[v2].eqn[s] = e;
      succ[v2].var[s] = v1;
      succ[v2].diff[s] = -GEQs[e].coef[0];
      canHandle++;
    }


  if (DBUG) {
    int s;
    fprintf(outputFile,"In chainkill: [\n");
    for(v1 = 0;v1<=nVars;v1++) {
      fprintf(outputFile,"#%d <=   %s: ",essentialPred[v1],variable(v1));
      for(s=0;s<succ[v1].notEssential;s++) 
        fprintf(outputFile," %s(" coef_fmt ") ",variable(succ[v1].var[s]), succ[v1].diff[s]);
      for(;s<succ[v1].num;s++) 
        fprintf(outputFile," %s[" coef_fmt "] ",variable(succ[v1].var[s]), succ[v1].diff[s]);
      fprintf(outputFile,"\n");
    }
  }

  for(;v1<=nVars;v1++)
    if (succ[v1].num == 1 && succ[v1].notEssential == 1) {
      succ[v1].notEssential--;
      essentialPred[succ[v1].var[succ[v1].notEssential]]++;
    }

  if (DBUG) fprintf(outputFile,"Trying quick double kill:\n");
  int s1a,s1b,s2;
  int v3;
  for(v1 = 0;v1<=nVars;v1++) 
    for(s1a=0;s1a<succ[v1].notEssential;s1a++) {
      v3 = succ[v1].var[s1a];
      for(s1b=0;s1b<succ[v1].num;s1b++)
        if (s1a != s1b) {
          v2 = succ[v1].var[s1b];
          for(s2=0;s2<succ[v2].num;s2++)
            if (succ[v2].var[s2] == v3 && succ[v1].diff[s1b] + succ[v2].diff[s2] >= succ[v1].diff[s1a]) {
              if (DBUG) {
                fprintf(outputFile,"quick double kill: "); 
                printGEQ(&GEQs[succ[v1].eqn[s1a]]);
                fprintf(outputFile,"\n"); 
              }
              redundant[succ[v1].eqn[s1a]] = 1;
              anyKilled++;
              anyToKill--;
              goto nextVictim;
            }
        }
    nextVictim: v1 = v1;
    }
  if (anyKilled) {
    for (e = nGEQs-1; e >= 0;e--)
      if (redundant[e]) {
        if (DBUG) {
          fprintf(outputFile,"Deleting ");
          printGEQ(&GEQs[e]);
          fprintf(outputFile,"\n");
        }
        deleteGEQ(e);
      }

    if (!anyToKill) return canHandle < nGEQs;
    noteEssential(onlyWildcards);
    triedDoubleKill = 1;
    goto restart;
  }

  for(v1 = 0;v1<=nVars;v1++)
    if (succ[v1].num == succ[v1].notEssential && succ[v1].notEssential > 0) {
      succ[v1].notEssential--;
      essentialPred[succ[v1].var[succ[v1].notEssential]]++;
    }
  
  while (1) {
    int chainLength;
    int chain[maxVars];
    coef_t distance[maxVars];
    // pick a place to start
    for(v1 = 0;v1<=nVars;v1++)
      if (essentialPred[v1] == 0 && succ[v1].num > succ[v1].notEssential)
        break;
    if (v1 > nVars) 
      for(v1 = 0;v1<=nVars;v1++)
        if (goodStartingPoint[v1] && succ[v1].num > succ[v1].notEssential)
          break;
    if (v1 > nVars) break;

    chainLength = 1;
    chain[0] = v1;
    distance[0] = 0;
    inChain[v1] = 0;
    int s;
   
    while (succ[v1].num > succ[v1].notEssential) {
      s = succ[v1].num-1;
      if (inChain[succ[v1].var[s]] >= 0) {
        // Found cycle, don't do anything with them yet
        break;
      }
      succ[v1].num = s;

      distance[chainLength]= distance[chainLength-1] + succ[v1].diff[s];
      v1 = chain[chainLength] = succ[v1].var[s];
      essentialPred[v1]--;
      assert(essentialPred[v1] >= 0);
      inChain[v1] = chainLength;
      chainLength++;
    }


    int c;
    if (DBUG) {
      fprintf(outputFile,"Found chain: \n");
      for (c = 0; c < chainLength; c++) 
        fprintf(outputFile,"%s:" coef_fmt "  ",variable(chain[c]), distance[c]);
      fprintf(outputFile,"\n");
    }
 
 
    for (c = 0; c < chainLength; c++) {
      v1 = chain[c];
      for(s=0;s<succ[v1].notEssential;s++) {
        if (DBUG)
          fprintf(outputFile,"checking for %s + " coef_fmt " <= %s \n", variable(v1), succ[v1].diff[s], variable(succ[v1].var[s]));
        if (inChain[succ[v1].var[s]] > c+1) {
          if (DBUG)
            fprintf(outputFile,"%s + " coef_fmt " <= %s is in chain\n", variable(v1), distance[inChain[succ[v1].var[s]]]- distance[c], variable(succ[v1].var[s]));
          if ( distance[inChain[succ[v1].var[s]]]- distance[c] >= succ[v1].diff[s]) {
            if (DBUG) 
              fprintf(outputFile,"%s + " coef_fmt " <= %s is redundant\n", variable(v1),succ[v1].diff[s], variable(succ[v1].var[s]));
            redundant[succ[v1].eqn[s]] = 1;
          }
        }
      }
    } 
    for (c = 0; c < chainLength; c++) 
      inChain[chain[c]] = -1;
  }
  
  for (e = nGEQs-1; e >= 0;e--)
    if (redundant[e]) {
      if (DBUG) {
        fprintf(outputFile,"Deleting ");
        printGEQ(&GEQs[e]);
        fprintf(outputFile,"\n");
      }
      deleteGEQ(e);
      anyKilled = 1;
    }

  if (anyKilled) noteEssential(onlyWildcards);

  if (anyKilled && DBUG) {
    fprintf(outputFile,"\nResult:\n");
    printProblem();
  }
  if (DBUG)  {
    fprintf(outputFile,"] end chainkill\n");
    printProblem();
  }
  return canHandle < nGEQs;
}


namespace {
  struct varCountStruct {
    int e;
    int safeVarCount;
    int wildVarCount;
    varCountStruct(int e_, int count1_, int count2_) {
      e = e_;
      safeVarCount = count1_;
      wildVarCount = count2_; }
  };
  bool operator<(const varCountStruct &a, const varCountStruct &b) {
    if (a.wildVarCount < b.wildVarCount)
      return true;
    else if (a.wildVarCount > b.wildVarCount)
      return false;
    else
      return a.safeVarCount < b.safeVarCount;
  }
}


//
// Deduct redundant inequalities by combination of any two inequalities.
// Return value: 0 (no solution),
//               1 (nothing killed),
//               2 (some inequality killed).
//
int Problem::quickKill(int onlyWildcards, bool desperate) {
  if (!onlyWildcards && !combineToTighten())
    return 0;
  noteEssential(onlyWildcards);
  int moreToDo = chainKill(0, onlyWildcards);

#ifdef NDEBUG
  if (!moreToDo) return 1;
#endif

  
  if (!desperate && nGEQs > 256) {  // original 60, increased by chun
    if (TRACE) {
      fprintf(outputFile, "%d inequalities are too complicated to quick kill\n", nGEQs);
    }
    return 1;
  }

  if (DBUG) {
    fprintf(outputFile, "in eliminate Redudant:\n");
    printProblem();
  }

  int isDead[nGEQs];
  std::vector<varCountStruct> killOrder;
  std::vector<BoolSet<> > P(nGEQs, BoolSet<>(nVars)), Z(nGEQs, BoolSet<>(nVars)), N(nGEQs, BoolSet<>(nVars));
  BoolSet<> PP, PZ, PN; // possible Positives, possible zeros & possible negatives

  for (int e = nGEQs - 1; e >= 0; e--) {
    isDead[e] = 0;
    int safeVarCount = 0;
    int wildVarCount = 0;
    for (int i = nVars; i >= 1; i--) {
      if (GEQs[e].coef[i] == 0)
        Z[e].set(i-1);
      else {
        if (i > safeVars)
          wildVarCount++;
        else
          safeVarCount++;
        if (GEQs[e].coef[i] < 0)
          N[e].set(i-1);
        else
          P[e].set(i-1);
      }
    }

    if (!GEQs[e].essential || wildVarCount > 0)
      killOrder.push_back(varCountStruct(e, safeVarCount, wildVarCount));
  }

  sort(killOrder.begin(), killOrder.end());
  
  if (DEBUG) {
    fprintf(outputFile,"Prefered kill order:\n");
    for (int e3I = killOrder.size()-1; e3I >= 0; e3I--) {
      fprintf(outputFile,"%2d: ",nGEQs-1-e3I);
      printGEQ(&GEQs[killOrder[e3I].e]);
      fprintf(outputFile,"\n");
    }
  }

  int e3U = killOrder.size()-1;
  while (e3U >= 0) {
    // each round of elimination is for inequalities of same complexity and rounds are at descending complexity order
    int e3L = e3U-1;
    for(; e3L >= 0; e3L--)
      if (killOrder[e3L].safeVarCount+killOrder[e3L].wildVarCount != killOrder[e3U].safeVarCount + killOrder[e3U].wildVarCount)
        break;

    // check if e3 can be eliminated from combination of e1 and e2
    for (int e1 = 0; e1 < nGEQs; e1++)
      if (!isDead[e1])
        for (int e2 = e1+1; e2 < nGEQs; e2++)
          if (!isDead[e2]) {
            coef_t alpha = 0;
            int p, q;
            for (p = nVars; p > 1; p--) 
              for (q = p - 1; q > 0; q--) {
                try {
                  alpha = check_mul(GEQs[e1].coef[p], GEQs[e2].coef[q]) - check_mul(GEQs[e2].coef[p], GEQs[e1].coef[q]);
                }
                catch (std::overflow_error) {
                  continue;
                }
                if (alpha != 0)
                  goto foundPQ;
              }
            continue;

          foundPQ:
            PZ = (Z[e1] & Z[e2]) | (P[e1] & N[e2]) | (N[e1] & P[e2]);
            PP = P[e1] | P[e2];
            PN = N[e1] | N[e2];
            if (DEBUG) {
              fprintf(outputFile,"Considering combination of ");
              printGEQ(&(GEQs[e1]));
              fprintf(outputFile," and  ");
              printGEQ(&(GEQs[e2]));
              fprintf(outputFile,"\n");
            }

            for (int e3I = e3U; e3I > e3L; e3I--) {
              int e3 = killOrder[e3I].e;
              if (!isDead[e3] && e3 != e1 && e3 != e2)
                try {
                  coef_t alpha1, alpha2, alpha3;

                  if (!PZ.imply(Z[e3]))
                    goto nextE3;

                  alpha1 = check_mul(GEQs[e2].coef[q], GEQs[e3].coef[p]) - check_mul(GEQs[e2].coef[p], GEQs[e3].coef[q]);
                  alpha2 = -(check_mul(GEQs[e1].coef[q], GEQs[e3].coef[p]) - check_mul(GEQs[e1].coef[p], GEQs[e3].coef[q]));
                  alpha3 = alpha;

                  if (alpha1 < 0) {
                    alpha1 = -alpha1;
                    alpha2 = -alpha2;
                    alpha3 = -alpha3;
                  }
                  if (alpha1 == 0 || alpha2 <= 0)
                    goto nextE3;

                  {
                    coef_t g = gcd(gcd(alpha1, alpha2), abs(alpha3));
                    alpha1 /= g;
                    alpha2 /= g;
                    alpha3 /= g;
                  }
              
                  if (DEBUG) {
                    fprintf(outputFile, coef_fmt "e1 + " coef_fmt "e2 = " coef_fmt "e3: ",alpha1,alpha2,alpha3);
                    printGEQ(&(GEQs[e3]));
                    fprintf(outputFile,"\n");
                  }

                  if (alpha3 > 0) { // trying to prove e3 is redundant
                    if (!GEQs[e3].color && (GEQs[e1].color || GEQs[e2].color)) {
                      goto nextE3;
                    }
                    if (!PP.imply(P[e3]) | !PN.imply(N[e3]))
                      goto nextE3;

                    // verify alpha1*v1+alpha2*v2 = alpha3*v3
                    for (int k = nVars; k >= 1; k--)
                      if (check_mul(alpha3,  GEQs[e3].coef[k]) != check_mul(alpha1, GEQs[e1].coef[k]) + check_mul(alpha2, GEQs[e2].coef[k]))
                        goto nextE3;

                    coef_t c = check_mul(alpha1, GEQs[e1].coef[0]) + check_mul(alpha2, GEQs[e2].coef[0]);
                    if (c < check_mul(alpha3, (GEQs[e3].coef[0] + 1))) {
                      if (DBUG) {
                        fprintf(outputFile, "found redundant inequality\n");
                        fprintf(outputFile, "alpha1, alpha2, alpha3 = " coef_fmt "," coef_fmt "," coef_fmt "\n", alpha1, alpha2, alpha3);
                        printGEQ(&(GEQs[e1]));
                        fprintf(outputFile, "\n");
                        printGEQ(&(GEQs[e2]));
                        fprintf(outputFile, "\n=> ");
                        printGEQ(&(GEQs[e3]));
                        fprintf(outputFile, "\n\n");
                        assert(moreToDo);
                      }

                      isDead[e3] = 1;
                    }
                  } 
                  else { // trying to prove e3 <= 0 or e3 = 0
                    if (!PN.imply(P[e3]) | !PP.imply(N[e3]))
                      goto nextE3;

                    // verify alpha1*v1+alpha2*v2 = alpha3*v3
                    for (int k = nVars; k >= 1; k--)
                      if (check_mul(alpha3, GEQs[e3].coef[k]) != check_mul(alpha1, GEQs[e1].coef[k]) + check_mul(alpha2, GEQs[e2].coef[k]))
                        goto nextE3;

                    if (DEBUG) {
                      fprintf(outputFile,"All but constant term checked\n");
                    }
                    coef_t c = check_mul(alpha1, GEQs[e1].coef[0]) + check_mul(alpha2, GEQs[e2].coef[0]);
                    if (DEBUG) {
                      fprintf(outputFile,"All but constant term checked\n");
                      fprintf(outputFile,"Constant term is " coef_fmt " vs " coef_fmt "\n",
                              alpha3*GEQs[e3].coef[0],
                              alpha3*(GEQs[e3].coef[0]-1));
                    }
                    if (c < check_mul(alpha3, (GEQs[e3].coef[0]))) {
                      // we just proved e3 < 0, so no solutions exist
                      if (DBUG) {
                        fprintf(outputFile, "found implied over tight inequality\n");
                        fprintf(outputFile, "alpha1, alpha2, alpha3 = " coef_fmt "," coef_fmt "," coef_fmt "\n", alpha1, alpha2, -alpha3);
                        printGEQ(&(GEQs[e1]));
                        fprintf(outputFile, "\n");
                        printGEQ(&(GEQs[e2]));
                        fprintf(outputFile, "\n=> not ");
                        printGEQ(&(GEQs[e3]));
                        fprintf(outputFile, "\n\n");
                      }
                      return 0;
                    }
                    else if (!GEQs[e3].color && (GEQs[e1].color || GEQs[e2].color)) {
                      goto nextE3;
                    }
                    else if (c < check_mul(alpha3, (GEQs[e3].coef[0] - 1))) {
                      // we just proved e3 <= 0, so e3 = 0
                      if (DBUG) {
                        fprintf(outputFile, "found implied tight inequality\n");
                        fprintf(outputFile, "alpha1, alpha2, alpha3 = " coef_fmt "," coef_fmt "," coef_fmt "\n", alpha1, alpha2, -alpha3);
                        printGEQ(&(GEQs[e1]));
                        fprintf(outputFile, "\n");
                        printGEQ(&(GEQs[e2]));
                        fprintf(outputFile, "\n=> inverse ");
                        printGEQ(&(GEQs[e3]));
                        fprintf(outputFile, "\n\n");
                      }
                      int neweq = newEQ();
                      eqnncpy(&EQs[neweq], &GEQs[e3], nVars);
                      addingEqualityConstraint(neweq);
                      isDead[e3] = 1;
                    }
                  }
                nextE3:;
                }
                catch (std::overflow_error) {
                  continue;
                }
            }
          }

    e3U = e3L;
  }

  bool anything_killed = false;
  for (int e = nGEQs - 1; e >= 0; e--) {
    if (isDead[e]) {
      anything_killed = true;
      deleteGEQ(e);
    }
  }

  if (DBUG) {
    fprintf(outputFile,"\nResult:\n");
    printProblem();
  }

  if (anything_killed)
    return 2;
  else
    return 1;
}

} // namespace
