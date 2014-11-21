/*****************************************************************************
 Copyright (C) 1994-2000 University of Maryland
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   Various hull calculations.

 Notes:

 History:
  06/15/09  ConvexRepresentation, Chun Chen
  11/25/09  RectHull, Chun Chen
*****************************************************************************/

#include <omega.h>
#include <omega/farkas.h>
#include <omega/hull.h>
#include <basic/Bag.h>
#include <basic/Map.h>
#include <basic/omega_error.h>
#include <list>
#include <vector>
#include <set>

namespace omega {

int hull_debug = 0; 

Relation ConvexHull(NOT_CONST Relation &R) {
  Relation S = Approximate(consume_and_regurgitate(R));
  if (!S.is_upper_bound_satisfiable())
    return S;
  if (S.has_single_conjunct())
    return S;
  return Farkas(Farkas(S,Basic_Farkas), Convex_Combination_Farkas);
}

Relation DecoupledConvexHull(NOT_CONST Relation &R) {
  Relation S = Approximate(consume_and_regurgitate(R));
  if (!S.is_upper_bound_satisfiable())
    return S;
  if (S.has_single_conjunct())
    return S;
  return Farkas(Farkas(S,Decoupled_Farkas), Convex_Combination_Farkas);
}

Relation AffineHull(NOT_CONST Relation &R) {
  Relation S = Approximate(consume_and_regurgitate(R));
  if (!S.is_upper_bound_satisfiable())
    return S;
  return Farkas(Farkas(S,Basic_Farkas), Affine_Combination_Farkas);
}

Relation LinearHull(NOT_CONST Relation &R) {
  Relation S = Approximate(consume_and_regurgitate(R));
  if (!S.is_upper_bound_satisfiable())
    return S;
  return Farkas(Farkas(S,Basic_Farkas), Linear_Combination_Farkas);
}

Relation ConicHull(NOT_CONST Relation &R) {
  Relation S = Approximate(consume_and_regurgitate(R));
  if (!S.is_upper_bound_satisfiable())
    return S;  
  return Farkas(Farkas(S,Basic_Farkas), Positive_Combination_Farkas);
}


Relation FastTightHull(NOT_CONST Relation &input_R, NOT_CONST Relation &input_H) {
  Relation R = Approximate(consume_and_regurgitate(input_R));
  Relation H = Approximate(consume_and_regurgitate(input_H));

  if (hull_debug) {
    fprintf(DebugFile,"[ Computing FastTightHull of:\n");
    R.prefix_print(DebugFile);
    fprintf(DebugFile,"given known hull of:\n");
    H.prefix_print(DebugFile);
  }

  if (!H.has_single_conjunct()) {
    if (hull_debug) 
      fprintf(DebugFile, "] bailing out of FastTightHull, known hull not convex\n");
    return H;
  }

  if (!H.is_obvious_tautology()) {
    R = Gist(R,copy(H));
    R.simplify(1,0);
  }

  if (R.has_single_conjunct()) {
    R = Intersection(R,H);
    if (hull_debug)  {
      fprintf(DebugFile, "] quick easy answer to FastTightHull\n");
      R.prefix_print(DebugFile);
    }
    return R;
  }
  if (R.has_local(coefficient_of_constant_term)) {
    if (hull_debug) {
      fprintf(DebugFile, "] Can't handle recursive application of Farkas lemma\n");
    }
    return H;
  }
  
  if (hull_debug) {
    fprintf(DebugFile,"Gist of R given H is:\n");
    R.prefix_print(DebugFile);
  }

  if (1) {
    Set<Variable_ID> vars;
    int conjuncts = 0;
    for (DNF_Iterator s(R.query_DNF()); s.live(); s.next()) {
      conjuncts++;
      for (Variable_ID_Iterator v(*((*s)->variables())); v.live(); v++) {
        bool found = false;
        for (EQ_Iterator eq = (*s)->EQs(); eq.live(); eq.next())
          if ((*eq).get_coef(*v) != 0) {
            if (!found) vars.insert(*v);
            found = true;
            break;
          }
        if (!found)
          for (GEQ_Iterator geq = (*s)->GEQs(); geq.live(); geq.next())
            if ((*geq).get_coef(*v) != 0) {
              if (!found) vars.insert(*v);
              found = true;
              break;
            }
      }
    }

     
    // We now know which variables appear in R
    if (hull_debug) {
      fprintf(DebugFile,"Variables we need a better hull on are: ");
      foreach(v,Variable_ID,vars,
              fprintf(DebugFile," %s",v->char_name()));
      fprintf(DebugFile,"\n");
    }
    Conjunct *c = H.single_conjunct();
    int total=0;
    int copied = 0;
    for (EQ_Iterator eq = c->EQs(); eq.live(); eq.next()) {
      total++;
      foreach(v,Variable_ID,vars,
              if ((*eq).get_coef(v) != 0) {
                R.and_with_EQ(*eq);
                copied++;
                break; // out of variable loop
              }
        );
    }
    for (GEQ_Iterator geq = c->GEQs(); geq.live(); geq.next()) {
      total++;
      foreach(v,Variable_ID,vars,
              if ((*geq).get_coef(v) != 0) {
                R.and_with_GEQ(*geq);
                copied++;
                break; // out of variable loop
              }
        );
    }
    if (copied < total) {
      R = Approximate(R);

      if (hull_debug) { 
        fprintf(DebugFile,"Decomposed relation, copied only %d of %d constraints\n",copied,total);
        fprintf(DebugFile,"Original R:\n");
        R.prefix_print(DebugFile);
        fprintf(DebugFile,"Known hull:\n");
        H.prefix_print(DebugFile);
        fprintf(DebugFile,"New R:\n");
        R.prefix_print(DebugFile);
      }
    }
  }

  Relation F = Farkas(copy(R), Basic_Farkas, true);
  if (hull_debug)  
    fprintf(DebugFile,"Farkas Difficulty = " coef_fmt "\n", farkasDifficulty);
  if (farkasDifficulty > 260) {
    if (hull_debug)  {
      fprintf(DebugFile, "] bailing out, farkas is way too complex\n");
      fprintf(DebugFile,"Farkas:\n");
      F.prefix_print(DebugFile);
    }
    return H;
  }
  else if (farkasDifficulty > 130) {
    // Bail out
    if (hull_debug)  {
      fprintf(DebugFile, coef_fmt " non-zeros in original farkas\n", farkasDifficulty);
    }
    Relation tmp = Farkas(R, Decoupled_Farkas, true);
    
    if (hull_debug)  {
      fprintf(DebugFile, coef_fmt " non-zeros in decoupled farkas\n", farkasDifficulty);
    }
    if (farkasDifficulty > 260)  {
      if (hull_debug)  {
        fprintf(DebugFile, "] bailing out, farkas is way too complex\n");
        fprintf(DebugFile,"Farkas:\n");
        F.prefix_print(DebugFile);
      }
      return H;
    }
    else {
      if (farkasDifficulty > 130) 
        R = Intersection(H, Farkas(tmp, Affine_Combination_Farkas, true));
      else R = Intersection(H,
                            Intersection(Farkas(tmp, Convex_Combination_Farkas, true),
                                         Farkas(F, Affine_Combination_Farkas, true)));
      if (hull_debug)  {
        fprintf(DebugFile, "] bailing out, farkas is too complex, using affine hull\n");
        fprintf(DebugFile,"Farkas:\n");
        F.prefix_print(DebugFile);
        fprintf(DebugFile,"Affine hull:\n");
        R.prefix_print(DebugFile);
      }
      return R;
    }
  }
  
  R = Intersection(H, Farkas(F, Convex_Combination_Farkas, true));
  if (hull_debug)  {
    fprintf(DebugFile, "] Result of FastTightHull:\n");
    R.prefix_print(DebugFile);
  }
  return R;
}



namespace {
  bool parallel(const GEQ_Handle &g1, const GEQ_Handle &g2) {
    for(Constr_Vars_Iter cvi(g1, false); cvi; cvi++) {
      coef_t c1 = (*cvi).coef;
      coef_t c2 = g2.get_coef((*cvi).var);
      if (c1 != c2) return false;
    }
    {
      for(Constr_Vars_Iter cvi(g2, false); cvi; cvi++) {
        coef_t c1 = g1.get_coef((*cvi).var);
        coef_t c2 = (*cvi).coef;
        if (c1 != c2) return false;
      }
    }
    return true;
  }


  bool hull(const EQ_Handle &e, const GEQ_Handle &g, coef_t &hull) {
    int sign = 0;
    for(Constr_Vars_Iter cvi(e, false); cvi; cvi++) {
      coef_t c1 = (*cvi).coef;
      coef_t c2 = g.get_coef((*cvi).var);
      if (sign == 0) sign = (c1*c2>=0?1:-1);
      if (sign*c1 != c2) return false;
    }
    assert(sign != 0);
    {
      for(Constr_Vars_Iter cvi(g, false); cvi; cvi++) {
        coef_t c1 = e.get_coef((*cvi).var);
        coef_t c2 = (*cvi).coef;
        if (sign*c1 != c2) return false;
      }
    }
    hull = max(sign * e.get_const(), g.get_const());
    if (hull_debug) {
      fprintf(DebugFile,"Hull of:\n %s\n", e.print_to_string().c_str());
      fprintf(DebugFile," %s\n", g.print_to_string().c_str());
      fprintf(DebugFile,"is " coef_fmt "\n\n",hull);
    }
    return true;
  }

  bool eq(const EQ_Handle &e1, const EQ_Handle &e2) {
    int sign = 0;
    for(Constr_Vars_Iter cvi(e1, false); cvi; cvi++) {
      coef_t c1 = (*cvi).coef;
      coef_t c2 = e2.get_coef((*cvi).var);
      if (sign == 0) sign = (c1*c2>=0?1:-1);
      if (sign*c1 != c2) return false;
    }
    assert(sign != 0);
    {
      for(Constr_Vars_Iter cvi(e2, false); cvi; cvi++) {
        coef_t c1 = e1.get_coef((*cvi).var);
        coef_t c2 = (*cvi).coef;
        if (sign*c1 != c2) return false;
      }
    }
    return sign * e1.get_const() == e2.get_const();
  }
}


// This function is deprecated!!!
Relation QuickHull(Relation &R) {
  Tuple<Relation> Rs(1);
  Rs[1] = R;
  return QuickHull(Rs);
}


// This function is deprecated!!!
Relation QuickHull(Tuple<Relation> &Rs) {
  assert(!Rs.empty());

  // if (Rs.size() == 1) return Rs[1];

  Tuple<Relation> l_Rs;
  for (int i = 1; i <= Rs.size(); i++)
    for (DNF_Iterator c(Rs[i].query_DNF()); c; c++) {
      Relation r = Relation(Rs[i], c.curr());
      l_Rs.append(Approximate(r));
    }
    
  if (l_Rs.size() == 1)
    return l_Rs[1];
  
  Relation result = Relation::True(Rs[1]);
  result.copy_names(Rs[1]);

  use_ugly_names++; 

  if (hull_debug > 1) 
    for (int i = 1; i <= l_Rs.size(); i++) {
      fprintf(DebugFile,"#%d \n",i);
      l_Rs[i].prefix_print(DebugFile);
    }

  
//   Relation R = copy(Rs[1]);
//   for (int i = 2; i <= Rs.size(); i++) 
//     R = Union(R,copy(Rs[i]));

// #if 0
//   if (!R.is_set()) {
//     if (R.n_inp() == R.n_out()) {
//       Relation AC = DeltasToRelation(Hull(Deltas(copy(R),
//                                                  min(R.n_inp(),R.n_out()))),
//                                      R.n_inp(),R.n_out());
//       Relation dH = Hull(Domain(copy(R)),false);
//       Relation rH = Hull(Range(copy(R)),false);
//       result = Intersection(AC,Cross_Product(dH,rH)); 
//     }
//     else {
//       Relation dH = Hull(Domain(copy(R)),false);
//       Relation rH = Hull(Range(copy(R)),false);
//       result = Cross_Product(dH,rH); 
//       assert(Must_Be_Subset(copy(R),copy(result)));
//     }
//   }

// #endif
 
  Conjunct *first;
  l_Rs[1] = EQs_to_GEQs(l_Rs[1]);
  first = l_Rs[1].single_conjunct();
  for (GEQ_Iterator candidate(first->GEQs()); candidate.live(); candidate.next()) {
    coef_t maxConstantTerm = (*candidate).get_const();
    bool found = true; 
    if (hull_debug > 1) {
      fprintf(DebugFile,"searching for bound on:\n %s\n", (*candidate).print_to_string().c_str());
    }
    for (int i = 2; i <= l_Rs.size(); i++) {
      Conjunct *other = l_Rs[i].single_conjunct();
      bool found_for_i = false;
      for (GEQ_Iterator target(other->GEQs()); target.live(); target.next()) {
        if (hull_debug > 2) {
          fprintf(DebugFile,"candidate:\n %s\n", (*candidate).print_to_string().c_str());
          fprintf(DebugFile,"target:\n %s\n", (*target).print_to_string().c_str());
        }
        if (parallel(*candidate,*target)) {
          if (hull_debug > 1)
            fprintf(DebugFile,"Found bound:\n %s\n", (*target).print_to_string().c_str());
          maxConstantTerm = max(maxConstantTerm,(*target).get_const());
          found_for_i = true;
          break;
        }
      }
      if (!found_for_i) {
        for (EQ_Iterator target_e(other->EQs()); target_e.live(); target_e.next()) {
          coef_t h;
          if (hull(*target_e,*candidate,h)) {
            if (hull_debug > 1)
              fprintf(DebugFile,"Found bound of " coef_fmt ":\n %s\n", h, (*target_e).print_to_string().c_str());
            maxConstantTerm = max(maxConstantTerm,h);
            found_for_i = true;
            break;
          }
        };
        if (!found_for_i) {
          if (hull_debug > 1) {
            fprintf(DebugFile,"No bound found in:\n");
            fprintf(DebugFile, "%s", l_Rs[i].print_with_subs_to_string().c_str());
          }
          //if nothing found 
          found = false;
          break;
        }
      }
    }
   
    if (found) {
      GEQ_Handle  h = result.and_with_GEQ();
      copy_constraint(h,*candidate);
      if (hull_debug > 1)
        fprintf(DebugFile,"Setting constant term to " coef_fmt " in\n %s\n", maxConstantTerm, h.print_to_string().c_str());
      h.update_const(maxConstantTerm - (*candidate).get_const());
      if (hull_debug > 1)
        fprintf(DebugFile,"Updated constraint is\n %s\n", h.print_to_string().c_str());
    }
  }


  for (EQ_Iterator candidate_eq(first->EQs()); candidate_eq.live(); candidate_eq.next()) {
    bool found = true;
    for (int i = 2; i <= l_Rs.size(); i++) {
      Conjunct *C = l_Rs[i].single_conjunct();
      bool found_for_i = false;

      for (EQ_Iterator target(C->EQs()); target.live(); target.next()) {
        if (eq(*candidate_eq,*target)) {
          found_for_i = true;
          break;
        }
      }
      if (!found_for_i) {
        //if nothing found 
        found = false;
        break;
      }
    }
   
    if (found) {
      EQ_Handle  h = result.and_with_EQ();
      copy_constraint(h,*candidate_eq);
      if (hull_debug > 1)
        fprintf(DebugFile,"Adding eq constraint: %s\n", h.print_to_string().c_str());
    }
  }

  use_ugly_names--;
  if (hull_debug > 1) {
    fprintf(DebugFile,"quick hull is of:");
    result.print_with_subs(DebugFile);
  }
  return result;
}


// Relation Hull2(Tuple<Relation> &Rs, Tuple<int> &active) {
//   assert(Rs.size() == active.size() && Rs.size() > 0);

//   Tuple<Relation> l_Rs;
//   for (int i = 1; i <= Rs.size(); i++)
//     if (active[i])
//       l_Rs.append(copy(Rs[i]));

//   if (l_Rs.size() == 0)
//     return Relation::False(Rs[1]);
    
//   try {
//     Relation r = l_Rs[1];
//     for (int i = 2; i <= l_Rs.size(); i++) {
//       r = Union(r, copy(l_Rs[i]));
//       r.simplify();
//     }

//     // Relation F = Farkas(r, Basic_Farkas, true);
//     // if (farkasDifficulty >= 500)
//     //   throw std::overflow_error("loop convex hull too complicated.");
//     // F = Farkas(F, Convex_Combination_Farkas, true);
//     return Farkas(Farkas(r, Basic_Farkas, true), Convex_Combination_Farkas, true);
//   }
//   catch (std::overflow_error) {
//     return QuickHull(l_Rs);
//   }
// }
  

namespace {
  void printRs(Tuple<Relation> &Rs) {
    fprintf(DebugFile,"Rs:\n");
    for (int i = 1; i <= Rs.size(); i++)
      fprintf(DebugFile,"#%d : %s\n",i,
              Rs[i].print_with_subs_to_string().c_str());
  }
}

Relation BetterHull(Tuple<Relation> &Rs, bool stridesAllowed, bool checkSubsets,
                    NOT_CONST Relation &input_knownHull = Relation::Null()) {
  Relation knownHull = consume_and_regurgitate(input_knownHull);
  static int OMEGA_WHINGE = -1;
  if (OMEGA_WHINGE < 0) {
    OMEGA_WHINGE = getenv("OMEGA_WHINGE") ? atoi(getenv("OMEGA_WHINGE")) : 0;
  }
  assert(!Rs.empty());
  if (Rs.size() == 1) {
    if (stridesAllowed) return Rs[1];
    else return Approximate(Rs[1]);
  }

  if (checkSubsets) {
    Tuple<bool> live(Rs.size());
    if (hull_debug) {
      fprintf(DebugFile,"Checking subsets in hull computation:\n");
      printRs(Rs);
    }
    int i;
    for(i=1;i <=Rs.size(); i++) live[i] = true;
    for(i=1;i <=Rs.size(); i++) 
      for(int j=1;j <=Rs.size(); j++) if (i != j && live[j]) {
          if (hull_debug) fprintf(DebugFile,"checking %d Is_Obvious_Subset %d\n",i,j);
          if (Is_Obvious_Subset(copy(Rs[i]),copy(Rs[j]))) {
            if (hull_debug) fprintf(DebugFile,"yes...\n");
            live[i] = false;
            break;
          }
        }
    for(i=1;i <=Rs.size(); i++) if (!live[i]) {
        if (i < Rs.size()) {
          Rs[i] = Rs[Rs.size()];
          live[i] = live[Rs.size()];
        }
        Rs[Rs.size()] = Relation();
        Rs.delete_last();
        i--;
      }
  }
  Relation hull;
  if (hull_debug) {
    fprintf(DebugFile,"Better Hull:\n");
    printRs(Rs);
    fprintf(DebugFile,"known hull: %s\n", knownHull.print_with_subs_to_string().c_str());
  }
  if (knownHull.is_null()) hull = QuickHull(Rs);
  else hull = Intersection(QuickHull(Rs),knownHull);
  // for (int i = 1; i <= Rs.size(); i++)
  //   hull = RectHull(Union(hull, copy(Rs[i])));
  // hull = Intersection(hull, knownHull);
  hull.simplify();
  if (hull_debug) {
    fprintf(DebugFile,"quick hull: %s\n", hull.print_with_subs_to_string().c_str());
  }

  Relation orig = Relation::False(Rs[1]);
  int i;
  for (i = 1; i <= Rs.size(); i++) 
    orig = Union(orig,copy(Rs[i]));

  orig.simplify();

  for (i = 1; i <= Rs.size(); i++) {
    if (!hull.is_obvious_tautology()) Rs[i] = Gist(Rs[i],copy(hull));
    Rs[i].simplify();
    if (Rs[i].is_obvious_tautology()) return hull;
    if (Rs[i].has_single_conjunct()) {
      Rs[i] = EQs_to_GEQs(Rs[i]);
      if (hull_debug) {
        fprintf(DebugFile,"Checking for hull constraints in:\n  %s\n", Rs[i].print_with_subs_to_string().c_str());
      }
      Conjunct *c = Rs[i].single_conjunct();
      for (GEQ_Iterator g(c->GEQs()); g.live(); g.next()) {
        Relation tmp = Relation::True(Rs[i]);
        tmp.and_with_GEQ(*g);
        if (!Difference(copy(orig),tmp).is_upper_bound_satisfiable()) 
          hull.and_with_GEQ(*g);
      }
      for (EQ_Iterator e(c->EQs()); e.live(); e.next()) {
        Relation tmp = Relation::True(Rs[i]);
        tmp.and_with_EQ(*e);
        if (!Difference(copy(orig),tmp).is_upper_bound_satisfiable()) 
          hull.and_with_EQ(*e);
      }
    }
  }

  hull = FastTightHull(orig,hull);
  assert(hull.has_single_conjunct());

  if (stridesAllowed) return hull;
  else return Approximate(hull);

}



Relation  Hull(NOT_CONST Relation &S, 
               bool stridesAllowed,
               int effort,
               NOT_CONST Relation &knownHull) {
  Relation R = consume_and_regurgitate(S);
  R.simplify(1,0);
  if (!R.is_upper_bound_satisfiable()) return R;
  Tuple<Relation> Rs;
  for (DNF_Iterator c(R.query_DNF()); c.live(); ) {
    Rs.append(Relation(R,c.curr()));
    c.next();
  }
  if (effort == 1)
    return BetterHull(Rs,stridesAllowed,false,knownHull);
  else
    return QuickHull(Rs);
}



Relation Hull(Tuple<Relation> &Rs, 
              Tuple<int> &validMask, 
              int effort, 
              bool stridesAllowed,
              NOT_CONST Relation &knownHull) {
  // Use relation of index i only when validMask[i] != 0
  Tuple<Relation> Rs2;
  for(int i = 1; i <= Rs.size(); i++) {
    if (validMask[i]) {
      Rs[i].simplify();
      for (DNF_Iterator c(Rs[i].query_DNF()); c.live(); ) {
        Rs2.append(Relation(Rs[i],c.curr()));
        c.next();
      }
    }
  }
  assert(effort == 0 || effort == 1);
  if (effort == 1)
    return BetterHull(Rs2,stridesAllowed,true,knownHull);
  else
    return QuickHull(Rs2);
}


// This function is deprecated!!!
Relation CheckForConvexRepresentation(NOT_CONST Relation &R_In) {
  Relation R = consume_and_regurgitate(R_In);
  Relation h = Hull(copy(R));
  if (!Difference(copy(h),copy(R)).is_upper_bound_satisfiable())
    return h;
  else
    return R;
}

// This function is deprecated!!!
Relation CheckForConvexPairs(NOT_CONST Relation &S) {
  Relation R = consume_and_regurgitate(S);
  Relation hull = FastTightHull(copy(R),Relation::True(R));
  R.simplify(1,0);
  if (!R.is_upper_bound_satisfiable() || R.number_of_conjuncts() < 2) return R;
  Tuple<Relation> Rs;
  for (DNF_Iterator c(R.query_DNF()); c.live(); ) {
    Rs.append(Relation(R,c.curr()));
    c.next();
  }

  bool *dead = new bool[Rs.size()+1];
  int i;
  for(i = 1; i<=Rs.size();i++) dead[i] = false;

  for(i = 1; i<=Rs.size();i++)
    if (!dead[i]) 
      for(int j = i+1; j<=Rs.size();j++) if (!dead[j]) {
          if (hull_debug) {
            fprintf(DebugFile,"Comparing #%d and %d\n",i,j);
          }
          Relation U = Union(copy(Rs[i]),copy(Rs[j]));
          Relation H_ij = FastTightHull(copy(U),copy(hull));
          if (!Difference(copy(H_ij),U).is_upper_bound_satisfiable()) {
            Rs[i] = H_ij;
            dead[j] = true;
            if (hull_debug)  {
              fprintf(DebugFile,"Combined them\n");
            }
          }
        }
  i = 1;
  while(i<=Rs.size() && dead[i]) i++;
  assert(i<=Rs.size());
  R = Rs[i];
  i++;
  for(; i<=Rs.size();i++)
    if (!dead[i]) 
      R = Union(R,Rs[i]);
  delete []dead;
  return R;
}

//
// Supporting functions for ConvexRepresentation
//
namespace {
struct Interval {
  std::list<std::pair<Relation, Relation> >::iterator pos;
  coef_t lb;
  coef_t ub;
  bool change;
  coef_t modulo;    
  Interval(std::list<std::pair<Relation, Relation> >::iterator pos_, coef_t lb_, coef_t ub_):
    pos(pos_), lb(lb_), ub(ub_) {}
  friend bool operator<(const Interval &a, const Interval &b);
};

bool operator<(const Interval &a, const Interval &b) {
  return a.lb < b.lb;
}

struct Modulo_Interval {
  coef_t modulo;
  coef_t start;
  coef_t size;
  Modulo_Interval(coef_t modulo_, coef_t start_, coef_t size_):
    modulo(modulo_), start(start_), size(size_) {}
  friend bool operator<(const Interval &a, const Interval &b);
};
  
bool operator<(const Modulo_Interval &a, const Modulo_Interval &b) {
  if (a.modulo == b.modulo) {
    if (a.start == b.start)
      return a.size < b.size;
    else
      return a.start < b.start;
  }
  else
    return a.modulo < b.modulo;
}

void merge_intervals(std::list<Interval> &intervals, coef_t modulo, std::list<std::pair<Relation, Relation> > &Rs, std::list<std::pair<Relation, Relation> >::iterator orig) {
  // normalize  intervals
  for (std::list<Interval>::iterator i = intervals.begin(); i != intervals.end(); i++) {
    (*i).modulo = modulo;
    (*i).change = false;
    if ((*i).ub - (*i).lb + 1>= modulo) {
      (*i).lb = 0;
      (*i).ub = modulo - 1;
    } 
    else if ((*i).ub < 0 || (*i).lb >= modulo) {
      coef_t range = (*i).ub - (*i).lb;
      (*i).lb = int_mod((*i).lb, modulo);
      (*i).ub = (*i).lb + range;
    }
  }

  intervals.sort();

  // merge neighboring intervals
  std::list<Interval>::iterator p = intervals.begin();
  while (p != intervals.end()) {
    std::list<Interval>::iterator q = p;
    q++;
    while (q != intervals.end()) {
      if ((*p).ub + 1 >= (*q).lb) {
        Relation hull = ConvexHull(Union(copy((*(*p).pos).first), copy((*(*q).pos).first)));
        Relation remainder = Difference(Difference(copy(hull), copy((*(*p).pos).first)), copy((*(*q).pos).first));
        if (!remainder.is_upper_bound_satisfiable()) {
          if ((*q).pos == orig)
            std::swap((*p).pos, (*q).pos);
          (*(*p).pos).first = hull;
          (*p).ub = max((*p).ub, (*q).ub);
          (*p).change = true;
          Rs.erase((*q).pos);
          q = intervals.erase(q);
        }
        else
          break;
      }
      else
        break;
    }

    bool p_moved = false;
    q = p;
    q++;
    while (q != intervals.end()) {
      if ((*q).ub >= modulo && int_mod((*q).ub, modulo) + 1 >= (*p).lb) {
        Relation hull = ConvexHull(Union(copy((*(*p).pos).first), copy((*(*q).pos).first)));
        Relation remainder = Difference(Difference(copy(hull), copy((*(*p).pos).first)), copy((*(*q).pos).first));
        if (!remainder.is_upper_bound_satisfiable()) {
          if ((*p).pos == orig)
            std::swap((*p).pos, (*q).pos);
          (*(*q).pos).first = hull;
          coef_t t = (*p).ub - int_mod((*q).ub, modulo);
          if (t > 0)
            (*q).ub = (*q).ub + t;              
          (*q).change = true;
          Rs.erase((*p).pos);
          p = intervals.erase(p);
          p_moved = true;
          break;
        }
        else
          q++;
      }
      else
        q++;
    }

    if (!p_moved)
      p++;
  }

  // merge by reducing the strengh of modulo
  std::list<Modulo_Interval> modulo_intervals;
  coef_t max_distance = modulo/2;
  for (std::list<Interval>::iterator p = intervals.begin(); p != intervals.end(); p++) {
    if ((*p).lb >= max_distance)
      break;

    coef_t size = (*p).ub - (*p).lb;
      
    std::list<Interval>::iterator q = p;
    q++;
    while (q != intervals.end()) {
      coef_t distance = (*q).lb - (*p).lb;
      if (distance > max_distance)
        break;

      if ((*q).ub - (*q).lb != size || int_mod(modulo, distance) != 0) {
        q++;
        continue;
      }

      int num_reduced = 0;
      coef_t looking_for = int_mod((*p).lb, distance);
      for (std::list<Interval>::iterator k = intervals.begin(); k != intervals.end(); k++) {
        if ((*k).lb == looking_for && (*k).ub - (*k).lb == size) {            
          num_reduced++;
          looking_for += distance;
          if (looking_for >= modulo)
            break;
        }            
        else if ((*k).lb <= looking_for && (*k).ub >= looking_for + size) {
          looking_for += distance;
          if (looking_for >= modulo)
            break;
        }
        else if ((*k).lb > looking_for)
          break;
      }

      if (looking_for >= modulo && num_reduced > 1)
        modulo_intervals.push_back(Modulo_Interval(distance, int_mod((*p).lb, distance), size));

      q++;
    }
  }
          
  modulo_intervals.sort();

  // remove redundant reduced-strength intervals
  std::list<Modulo_Interval>::iterator p2 = modulo_intervals.begin();
  while (p2 != modulo_intervals.end()) {
    std::list<Modulo_Interval>::iterator q2 = p2;
    q2++;
    while (q2 != modulo_intervals.end()) {
      if ((*p2).modulo == (*q2).modulo && (*p2).start == (*q2).start)
        q2 = modulo_intervals.erase(q2);
      else if (int_mod((*q2).modulo, (*p2).modulo) == 0 &&
               (*p2).start == int_mod((*q2).start, (*p2).modulo) &&
               (*p2).size >= (*q2).size)
        q2 = modulo_intervals.erase(q2);
      else
        q2++;
    }
    p2++;
  }
                
  // replace original intervals with new reduced-strength ones
  for (std::list<Modulo_Interval>::iterator i = modulo_intervals.begin(); i != modulo_intervals.end(); i++) {
    std::vector<Relation *> candidates;
    int num_replaced = 0;
    for (std::list<Interval>::iterator j = intervals.begin(); j != intervals.end(); j++)
      if (int_mod((*j).modulo, (*i).modulo) == 0 &&
          (*j).ub - (*j).lb >= (*i).size &&
          (int_mod((*j).lb, (*i).modulo) == (*i).start ||
           int_mod((*j).ub, (*i).modulo) == (*i).start + (*i).size)) {
        candidates.push_back(&((*(*j).pos).first));
        if (int_mod((*j).lb, (*i).modulo) == (*i).start &&
            (*j).ub - (*j).lb == (*i).size)
          num_replaced++;
      }
    if (num_replaced <= 1)
      continue;
      
    Relation R = copy(*candidates[0]);
    for (size_t k = 1; k < candidates.size(); k++)
      R = Union(R, copy(*candidates[k]));
    Relation hull = ConvexHull(copy(R));
    Relation remainder = Difference(copy(hull), copy(R));
    if (!remainder.is_upper_bound_satisfiable()) {
      std::list<Interval>::iterator replaced_one = intervals.end();
      for (std::list<Interval>::iterator j = intervals.begin(); j != intervals.end();)
        if (int_mod((*j).modulo, (*i).modulo) == 0 &&
            (*j).ub - (*j).lb >= (*i).size &&
            (int_mod((*j).lb, (*i).modulo) == (*i).start ||
             int_mod((*j).ub, (*i).modulo) == (*i).start + (*i).size)) {
          if (int_mod((*j).lb, (*i).modulo) == (*i).start &&
              (*j).ub - (*j).lb == (*i).size) {
            if (replaced_one == intervals.end()) {
              (*(*j).pos).first = hull;
              (*j).lb = int_mod((*j).lb, (*i).modulo);
              (*j).ub = int_mod((*j).ub, (*i).modulo);
              (*j).modulo = (*i).modulo;
              (*j).change = true;
              replaced_one = j;
              j++;
            }
            else {
              if ((*j).pos == orig) {
                std::swap((*replaced_one).pos, (*j).pos);
                (*(*replaced_one).pos).first = (*(*j).pos).first;
              }
              Rs.erase((*j).pos);
              j = intervals.erase(j);
            }
          }
          else {
            if (int_mod((*j).lb, (*i).modulo) == (*i).start)
              (*j).lb = (*j).lb + (*i).size + 1;
            else
              (*j).ub = (*j).ub - (*i).size - 1;
            (*j).change = true;
            j++;
          }
        }
        else
          j++;
    }
  }                
}    
} // namespace


//
// Simplify a union of sets/relations to a minimal (may not be
// optimal) number of convex regions.  It intends to replace
// CheckForConvexRepresentation and CheckForConvexPairs functions.
//
Relation ConvexRepresentation(NOT_CONST Relation &R) {
  Relation l_R = copy(R);
  if (!l_R.is_upper_bound_satisfiable() || l_R.number_of_conjuncts() < 2)
    return R;

  // separate each conjunct into smooth convex region and holes
  std::list<std::pair<Relation, Relation> > Rs; // pair(smooth region, hole condition)
  for (DNF_Iterator c(l_R.query_DNF()); c.live(); c++) {
    Relation r1 = Relation(l_R, c.curr());
    Relation r2 = Approximate(copy(r1));
    r1 = Gist(r1, copy(r2));
    Rs.push_back(std::make_pair(r2, r1));
  }

  try {
    bool change = true;
    while (change) {
      change = false;

      std::list<std::pair<Relation, Relation> >::iterator i = Rs.begin();
      while (i != Rs.end()) {
        // find regions with identical hole conditions to merge
        {
          std::list<std::pair<Relation, Relation> >::iterator j = i;
          j++;
          while (j != Rs.end()) {
            if (!Difference(copy((*i).second), copy((*j).second)).is_upper_bound_satisfiable() &&
                !Difference(copy((*j).second), copy((*i).second)).is_upper_bound_satisfiable()) {
              if (Must_Be_Subset(copy((*j).first), copy((*i).first))) {
                j = Rs.erase(j);
              }
              else if (Must_Be_Subset(copy((*i).first), copy((*j).first))) {
                (*i).first = (*j).first;
                j = Rs.erase(j);
                change = true;
              }
              else {
                Relation r;
                bool already_use_recthull = false;
                try {
                  // chun's debug
                  // throw std::runtime_error("dfdf");
                
                  r = ConvexHull(Union(copy((*i).first), copy((*j).first)));
                }
                catch (const std::overflow_error &e) {
                  r = RectHull(Union(copy((*i).first), copy((*j).first)));
                  already_use_recthull = true;
                }
                retry_recthull:
                Relation r2 = Difference(Difference(copy(r), copy((*i).first)), copy((*j).first));
                if (!r2.is_upper_bound_satisfiable()) { // convex hull is tight
                  (*i).first = r;
                  j = Rs.erase(j);
                  change = true;
                }
                else {
                  if (!already_use_recthull) {
                    r = RectHull(Union(copy((*i).first), copy((*j).first)));
                    already_use_recthull = true;
                    goto retry_recthull;
                  }
                  else
                    j++;
                }
              }
            }
            else
              j++;
          }
        }

        // find identical smooth regions as candidates for hole merge
        std::list<std::list<std::pair<Relation, Relation> >::iterator> s;
        for (std::list<std::pair<Relation, Relation> >::iterator j = Rs.begin(); j != Rs.end(); j++) 
          if (j != i) {
            if (!Intersection(Difference(copy((*i).first), copy((*j).first)), copy((*j).second)).is_upper_bound_satisfiable() &&
                !Intersection(Difference(copy((*j).first), copy((*i).first)), copy((*i).second)).is_upper_bound_satisfiable())
              s.push_back(j);
          }
      
        if (s.size() != 0) {
          // convert hole condition c1*x1+c2*x2+... = c*alpha+d to a pair of inequalities
          (*i).second = EQs_to_GEQs((*i).second, false);

          // find potential wildcards that can be used for hole conditions
          std::set<Variable_ID> nonsingle_wild;
          for (EQ_Iterator ei((*i).second.single_conjunct()); ei; ei++)
            if ((*ei).has_wildcards())
              for (Constr_Vars_Iter cvi(*ei, true); cvi; cvi++)
                nonsingle_wild.insert(cvi.curr_var());
          for (GEQ_Iterator gei((*i).second.single_conjunct()); gei; gei++)
            if ((*gei).has_wildcards()) {
              Constr_Vars_Iter cvi(*gei, true);
              Constr_Vars_Iter cvi2 = cvi;
              cvi2++;
              if (cvi2) {
                nonsingle_wild.insert(cvi.curr_var());
                for (; cvi2; cvi2++)
                  nonsingle_wild.insert(cvi2.curr_var());
              }
            }

          // find hole condition in c*alpha+d1<=c1*x1+c2*x2+...<=c*alpha+d2 format
          for (GEQ_Iterator gei((*i).second.single_conjunct()); gei; gei++)
            if ((*gei).has_wildcards()) {
              coef_t c;
              Variable_ID v;
              {
                Constr_Vars_Iter cvi(*gei, true);
                v = cvi.curr_var();
                c = cvi.curr_coef();
                if (c < 0 || nonsingle_wild.find(v) != nonsingle_wild.end())
                  continue;
              }
          
              coef_t lb = posInfinity;
              for (GEQ_Iterator gei2((*i).second.single_conjunct()); gei2; gei2++) {
                if (!(*gei2 == *gei) && (*gei2).get_coef(v) != 0) {
                  if (lb != posInfinity) {
                    nonsingle_wild.insert(v);
                    break;
                  }
                
                  bool match = true;
                  for (Constr_Vars_Iter cvi2(*gei); cvi2; cvi2++)
                    if (cvi2.curr_coef() != -((*gei2).get_coef(cvi2.curr_var()))) {
                      match = false;
                      break;
                    }
                  if (match)
                    for (Constr_Vars_Iter cvi2(*gei2); cvi2; cvi2++)
                      if (cvi2.curr_coef() != -((*gei).get_coef(cvi2.curr_var()))) {
                        match = false;
                        break;
                      }
                  if (!match) {
                    nonsingle_wild.insert(v);
                    break;
                  }

                  lb = -(*gei2).get_const();
                }
              }

              if (nonsingle_wild.find(v) != nonsingle_wild.end())
                continue;
          
              Relation stride_cond = Relation::True((*i).second);
              F_Exists *f_exists = stride_cond.and_with_and()->add_exists();
              Variable_ID e = f_exists->declare();
              F_And *f_root = f_exists->add_and();
              GEQ_Handle h1 = f_root->add_GEQ();
              GEQ_Handle h2 = f_root->add_GEQ();
              for (Constr_Vars_Iter cvi2(*gei); cvi2; cvi2++) {
                Variable_ID v = cvi2.curr_var();
                switch (v->kind()) {
                case Wildcard_Var: 
                  h1.update_coef(e, cvi2.curr_coef());
                  h2.update_coef(e, -cvi2.curr_coef());
                  break;
                case Global_Var: {
                  Global_Var_ID g = v->get_global_var();
                  Variable_ID v2;
                  if (g->arity() == 0)
                    v2 = stride_cond.get_local(g);
                  else
                    v2 = stride_cond.get_local(g, v->function_of());
                  h1.update_coef(v2, cvi2.curr_coef());
                  h2.update_coef(v2, -cvi2.curr_coef());
                  break;
                }
                default:
                  h1.update_coef(v, cvi2.curr_coef());
                  h2.update_coef(v, -cvi2.curr_coef());
                }
              }
              h1.update_const((*gei).get_const());
              h2.update_const(-lb);

              stride_cond.simplify();
              Relation other_cond = Gist(copy((*i).second), copy(stride_cond));

              // find regions with potential mergeable stride condition with this one
              std::list<Interval> intervals;
              intervals.push_back(Interval(i, lb, (*gei).get_const()));
            
              for (std::list<std::list<std::pair<Relation, Relation> >::iterator>::iterator j = s.begin(); j != s.end(); j++)
                if (Must_Be_Subset(copy((**j).second), copy(other_cond))) {
                  Relation stride_cond2 = Gist(copy((**j).second), copy(other_cond));

                  // interval can be removed
                  if (stride_cond2.is_obvious_tautology()) {
                    intervals.push_back(Interval(*j, 0, c-1));
                    continue;
                  }

                  stride_cond2 = EQs_to_GEQs(stride_cond2, false);
                  coef_t lb, ub;
                  GEQ_Iterator gei2(stride_cond2.single_conjunct());
                  coef_t sign = 0;
                  for (Constr_Vars_Iter cvi(*gei2, true); cvi; cvi++)
                    if (sign != 0) {
                      sign = 0;
                      break;
                    }
                    else if (cvi.curr_coef() == c)
                      sign = 1;
                    else if (cvi.curr_coef() == -c)
                      sign = -1;
                    else {
                      sign = 0;
                      break;
                    }                
                  if (sign == 0)
                    continue;

                  bool match = true;
                  for (Constr_Vars_Iter cvi(*gei2); cvi; cvi++) {
                    Variable_ID v = cvi.curr_var();
                    if (v->kind() == Wildcard_Var)
                      continue;
                    else if (v->kind() == Global_Var) {
                      Global_Var_ID g = v->get_global_var();
                      if (g->arity() == 0)
                        v = (*i).second.get_local(g);
                      else
                        v = (*i).second.get_local(g, v->function_of());
                    }
                    
                    if (cvi.curr_coef() != sign * (*gei).get_coef(v)) {
                      match = false;
                      break;
                    }
                  }
                  if (!match)
                    continue;
                
                  for (Constr_Vars_Iter cvi(*gei); cvi; cvi++) {
                    Variable_ID v = cvi.curr_var();
                    if (v->kind() == Wildcard_Var)
                      continue;
                    else if (v->kind() == Global_Var) {
                      Global_Var_ID g = v->get_global_var();
                      if (g->arity() == 0)
                        v = stride_cond2.get_local(g);
                      else
                        v = stride_cond2.get_local(g, v->function_of());
                    }
                    
                    if (cvi.curr_coef() != sign * (*gei2).get_coef(v)) {
                      match = false;
                      break;
                    }
                  }
                  if (!match)
                    continue;
                  if (sign > 0)
                    ub = (*gei2).get_const();
                  else
                    lb = -(*gei2).get_const();                
                
                  gei2++;
                  if (!gei2)
                    continue;

                  coef_t sign2 = 0;
                  for (Constr_Vars_Iter cvi(*gei2, true); cvi; cvi++)
                    if (sign2 != 0) {
                      sign2 = 0;
                      break;
                    }
                    else if (cvi.curr_coef() == c)
                      sign2 = 1;
                    else if (cvi.curr_coef() == -c)
                      sign2 = -1;
                    else {
                      sign2 = 0;
                      break;
                    }
                  if (sign2 != -sign)
                    continue;

                  for (Constr_Vars_Iter cvi(*gei2); cvi; cvi++) {
                    Variable_ID v = cvi.curr_var();
                    if (v->kind() == Wildcard_Var)
                      continue;
                    else if (v->kind() == Global_Var) {
                      Global_Var_ID g = v->get_global_var();
                      if (g->arity() == 0)
                        v = (*i).second.get_local(g);
                      else
                        v = (*i).second.get_local(g, v->function_of());
                    }
                    
                    if (cvi.curr_coef() != sign2 * (*gei).get_coef(v)) {
                      match = false;
                      break;
                    }
                  }
                  if (!match)
                    continue;
                
                  for (Constr_Vars_Iter cvi(*gei); cvi; cvi++) {
                    Variable_ID v = cvi.curr_var();
                    if (v->kind() == Wildcard_Var)
                      continue;
                    else if (v->kind() == Global_Var) {
                      Global_Var_ID g = v->get_global_var();
                      if (g->arity() == 0)
                        v = stride_cond2.get_local(g);
                      else
                        v = stride_cond2.get_local(g, v->function_of());
                    }
                    
                    if (cvi.curr_coef() != sign2 * (*gei2).get_coef(v)) {
                      match = false;
                      break;
                    }
                  }
                  if (!match)
                    continue;
                  if (sign2 > 0)
                    ub = (*gei2).get_const();
                  else
                    lb = -(*gei2).get_const();
                
                  gei2++;
                  if (gei2)
                    continue;
                    
                  intervals.push_back(Interval(*j, lb, ub));                
                }

              merge_intervals(intervals, c, Rs, i);

              // make current region the last one being updated
              bool invalid = false;
              for (std::list<Interval>::iterator ii = intervals.begin(); ii != intervals.end(); ii++)
                if ((*ii).change && (*ii).pos == i) {
                  invalid = true;
                  intervals.push_back(*ii);
                  intervals.erase(ii);
                  break;
                }

              // update hole condition for each region
              for (std::list<Interval>::iterator ii = intervals.begin(); ii != intervals.end(); ii++)
                if ((*ii).change) {
                  change = true;

                  if ((*ii).ub - (*ii).lb + 1 >= (*ii).modulo)
                    (*(*ii).pos).second = copy(other_cond);
                  else {
                    Relation stride_cond = Relation::True((*i).second);
                    F_Exists *f_exists = stride_cond.and_with_and()->add_exists();
                    Variable_ID e = f_exists->declare();
                    F_And *f_root = f_exists->add_and();
                    GEQ_Handle h1 = f_root->add_GEQ();
                    GEQ_Handle h2 = f_root->add_GEQ();
                    for (Constr_Vars_Iter cvi2(*gei); cvi2; cvi2++) {
                      Variable_ID v = cvi2.curr_var();
                      switch (v->kind()) {
                      case Wildcard_Var: 
                        h1.update_coef(e, (*ii).modulo);
                        h2.update_coef(e, -(*ii).modulo);
                        break;
                      case Global_Var: {
                        Global_Var_ID g = v->get_global_var();
                        Variable_ID v2;
                        if (g->arity() == 0)
                          v2 = stride_cond.get_local(g);
                        else
                          v2 = stride_cond.get_local(g, v->function_of());
                        h1.update_coef(v2, cvi2.curr_coef());
                        h2.update_coef(v2, -cvi2.curr_coef());
                        break;
                      }
                      default:
                        h1.update_coef(v, cvi2.curr_coef());
                        h2.update_coef(v, -cvi2.curr_coef());
                      }
                    }
                    h1.update_const((*ii).ub);
                    h2.update_const(-(*ii).lb);

                    (*(*ii).pos).second = Intersection(copy(other_cond), stride_cond);
                    (*(*ii).pos).second.simplify();
                  }
                }
            
              if (invalid)
                break;
            }
        }      
        i++;
      }
    }
  }
  catch (const presburger_error &e) {
    throw e;
  }

  Relation R2 = Relation::False(l_R);
  for (std::list<std::pair<Relation, Relation> >::iterator i = Rs.begin(); i != Rs.end(); i++)
    R2 = Union(R2, Intersection((*i).first, (*i).second));
  R2.simplify(0, 1);
  
  return R2;
}


//
// Use gist and value range to calculate a quick rectangular hull. It
// intends to replace all hull calculations (QuickHull, BetterHull,
// FastTightHull) beyond the method of ConvexHull (dual
// representations). In the future, it will support max(...)-like
// upper bound. So RectHull complements ConvexHull in two ways: first
// for relations that ConvexHull gets too complicated, second for
// relations where different conjuncts have different symbolic upper
// bounds.
// 
Relation RectHull(NOT_CONST Relation &Rel) {
  Relation R = Approximate(consume_and_regurgitate(Rel));
  if (!R.is_upper_bound_satisfiable())
    return R;
  if (R.has_single_conjunct())
    return R;

  std::vector<std::string> input_names(R.n_inp());
  for (int i = 1; i <= R.n_inp(); i++)
    input_names[i-1] = R.input_var(i)->name();
  std::vector<std::string> output_names(R.n_out());
  for (int i = 1; i <= R.n_out(); i++)
    output_names[i-1] = R.output_var(i)->name();
  
  DNF_Iterator c(R.query_DNF());
  Relation r = Relation(R, c.curr());
  c++;
  std::vector<std::pair<coef_t, coef_t> > bounds1(R.n_inp());
  std::vector<std::pair<coef_t, coef_t> > bounds2(R.n_out());
  {
    Relation t = Project_Sym(copy(r));
    t.simplify();
    for (int i = 1; i <= R.n_inp(); i++) {
      Tuple<Variable_ID> v;
      for (int j = 1; j <= R.n_inp(); j++)
        if (j != i)
          v.append(r.input_var(j));
      for (int j = 1; j <= R.n_out(); j++)
        v.append(r.output_var(j));
      Relation t2 = Project(copy(t), v);
      t2.query_variable_bounds(t2.input_var(i), bounds1[i-1].first, bounds1[i-1].second);
    }
    for (int i = 1; i <= R.n_out(); i++) {
      Tuple<Variable_ID> v;
      for (int j = 1; j <= R.n_out(); j++)
        if (j != i)
          v.append(r.output_var(j));
      for (int j = 1; j <= R.n_inp(); j++)
        v.append(r.input_var(j));
      Relation t2 = Project(copy(t), v);
      t2.query_variable_bounds(t2.output_var(i), bounds2[i-1].first, bounds2[i-1].second);
    }
  }
    
  while (c.live()) {
    Relation r2 = Relation(R, c.curr());
    c++;
    Relation x = Gist(copy(r), Gist(copy(r), copy(r2), 1), 1);
    if (Difference(copy(r2), copy(x)).is_upper_bound_satisfiable())
      x = Relation::True(R);
    Relation y = Gist(copy(r2), Gist(copy(r2), copy(r), 1), 1);
    if (Difference(copy(r), copy(y)).is_upper_bound_satisfiable())
      y = Relation::True(R);
    r = Intersection(x, y);
    
    {
      Relation t = Project_Sym(copy(r2));
      t.simplify();
      for (int i = 1; i <= R.n_inp(); i++) {
        Tuple<Variable_ID> v;
        for (int j = 1; j <= R.n_inp(); j++)
          if (j != i)
            v.append(r2.input_var(j));
        for (int j = 1; j <= R.n_out(); j++)
          v.append(r2.output_var(j));
        Relation t2 = Project(copy(t), v);
        coef_t lbound, ubound;
        t2.query_variable_bounds(t2.input_var(i), lbound, ubound);
        bounds1[i-1].first = min(bounds1[i-1].first, lbound);
        bounds1[i-1].second = max(bounds1[i-1].second, ubound);
      }
      for (int i = 1; i <= R.n_out(); i++) {
        Tuple<Variable_ID> v;
        for (int j = 1; j <= R.n_out(); j++)
          if (j != i)
            v.append(r2.output_var(j));
        for (int j = 1; j <= R.n_inp(); j++)
          v.append(r2.input_var(j));
        Relation t2 = Project(copy(t), v);
        coef_t lbound, ubound;
        t2.query_variable_bounds(t2.output_var(i), lbound, ubound);
        bounds2[i-1].first = min(bounds2[i-1].first, lbound);
        bounds2[i-1].second = max(bounds2[i-1].second, ubound);
      }
    }

    Relation r3(R.n_inp(), R.n_out());
    F_And *f_root = r3.add_and();
    for (int i = 1; i <= R.n_inp(); i++) {
      if (bounds1[i-1].first != -posInfinity) {
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(r3.input_var(i), 1);
        h.update_const(-bounds1[i-1].first);
      }
      if (bounds1[i-1].second != posInfinity) {
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(r3.input_var(i), -1);
        h.update_const(bounds1[i-1].second);
      }
    }
    for (int i = 1; i <= R.n_out(); i++) {
      if (bounds2[i-1].first != -posInfinity) {
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(r3.output_var(i), 1);
        h.update_const(-bounds2[i-1].first);
      }
      if (bounds2[i-1].second != posInfinity) {
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(r3.output_var(i), -1);
        h.update_const(bounds2[i-1].second);
      }
    }
    r = Intersection(r, r3);
    r.simplify();
  }

  for (int i = 1; i <= r.n_inp(); i++)
    r.name_input_var(i, input_names[i-1]);
  for (int i = 1; i <= r.n_out(); i++)
    r.name_output_var(i, output_names[i-1]);
  r.setup_names();
  return r;
}

} // namespace
