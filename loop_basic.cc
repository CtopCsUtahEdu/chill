/*
 * loop_basic.cc
 *
 *  Created on: Nov 12, 2012
 *      Author: anand
 */

#include "loop.hh"
#include "chill_error.hh"
#include <omega.h>
#include "omegatools.hh"
#include <code_gen/CG_utils.h>
#include "iegenlib.h"


using namespace omega;

void Loop::permute(const std::vector<int> &pi) {
  std::set<int> active;
  for (int i = 0; i < stmt.size(); i++)
    active.insert(i);
  
  permute(active, pi);
}

void Loop::original() {
  std::set<int> active;
  for (int i = 0; i < stmt.size(); i++)
    active.insert(i);
  setLexicalOrder(0, active);
  invalidateCodeGen();
  //apply_xform();
}
void Loop::permute(int stmt_num, int level, const std::vector<int> &pi) {
  // check for sanity of parameters
  int starting_order;
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument(
      "invalid statement number " + to_string(stmt_num));
  std::set<int> active;
  if (level < 0 || level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("3invalid loop level " + to_string(level));
  else if (level == 0) {
    for (int i = 0; i < stmt.size(); i++)
      active.insert(i);
    level = 1;
    starting_order = 0;
  } else {
    std::vector<int> lex = getLexicalOrder(stmt_num);
    active = getStatements(lex, 2 * level - 2);
    starting_order = lex[2 * level - 2];
    lex[2 * level - 2]++;
    shiftLexicalOrder(lex, 2 * level - 2, active.size() - 1);
  }
  std::vector<int> pi_inverse(pi.size(), 0);
  for (int i = 0; i < pi.size(); i++) {
    if (pi[i] >= level + pi.size() || pi[i] < level
        || pi_inverse[pi[i] - level] != 0)
      throw std::invalid_argument("invalid permuation");
    pi_inverse[pi[i] - level] = level + i;
  }
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++)
    if (level + pi.size() - 1 > stmt[*i].loop_level.size())
      throw std::invalid_argument(
        "invalid permutation for statement " + to_string(*i));

  invalidateCodeGen();

  // Update transformation relations
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    int n = stmt[*i].xform.n_out();
    omega::Relation mapping(n, n);
    F_And *f_root = mapping.add_and();
    for (int j = 1; j <= 2 * level - 2; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_coef(mapping.input_var(j), -1);
    }
    for (int j = level; j <= level + pi.size() - 1; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2 * j), 1);
      h.update_coef(mapping.input_var(2 * pi[j - level]), -1);
    }
    for (int j = level; j <= level + pi.size() - 1; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2 * j - 1), 1);
      h.update_coef(mapping.input_var(2 * j - 1), -1);
    }
    for (int j = 2 * (level + pi.size() - 1) + 1; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_coef(mapping.input_var(j), -1);
    }
    stmt[*i].xform = Composition(mapping, stmt[*i].xform);
    stmt[*i].xform.simplify();
  }
  
  // get the permuation for dependence vectors
  std::vector<int> t;
  for (int i = 0; i < pi.size(); i++)
    if (stmt[stmt_num].loop_level[pi[i] - 1].type == LoopLevelOriginal)
      t.push_back(stmt[stmt_num].loop_level[pi[i] - 1].payload);
  int max_dep_dim = -1;
  int min_dep_dim = dep.num_dim();
  for (int i = 0; i < t.size(); i++) {
    if (t[i] > max_dep_dim)
      max_dep_dim = t[i];
    if (t[i] < min_dep_dim)
      min_dep_dim = t[i];
  }
  if (min_dep_dim > max_dep_dim)
    return;
  if (max_dep_dim - min_dep_dim + 1 != t.size())
    throw loop_error("cannot update the dependence graph after permuation");
  std::vector<int> dep_pi(dep.num_dim());
  for (int i = 0; i < min_dep_dim; i++)
    dep_pi[i] = i;
  for (int i = min_dep_dim; i <= max_dep_dim; i++)
    dep_pi[i] = t[i - min_dep_dim];
  for (int i = max_dep_dim + 1; i < dep.num_dim(); i++)
    dep_pi[i] = i;
  
  dep.permute(dep_pi, active);
  
  // update the dependence graph
  DependenceGraph g(dep.num_dim());
  for (int i = 0; i < dep.vertex.size(); i++)
    g.insert();
  for (int i = 0; i < dep.vertex.size(); i++)
    for (DependenceGraph::EdgeList::iterator j =
           dep.vertex[i].second.begin(); j != dep.vertex[i].second.end();
         j++) {
      if ((active.find(i) != active.end()
           && active.find(j->first) != active.end())) {
        std::vector<DependenceVector> dv = j->second;
        for (int k = 0; k < dv.size(); k++) {
          switch (dv[k].type) {
          case DEP_W2R:
          case DEP_R2W:
          case DEP_W2W:
          case DEP_R2R: {
            std::vector<coef_t> lbounds(dep.num_dim());
            std::vector<coef_t> ubounds(dep.num_dim());
            for (int d = 0; d < dep.num_dim(); d++) {
              lbounds[d] = dv[k].lbounds[dep_pi[d]];
              ubounds[d] = dv[k].ubounds[dep_pi[d]];
            }
            dv[k].lbounds = lbounds;
            dv[k].ubounds = ubounds;
            break;
          }
          case DEP_CONTROL: {
            break;
          }
          default:
            throw loop_error("unknown dependence type");
          }
        }
        g.connect(i, j->first, dv);
      } else if (active.find(i) == active.end()
                 && active.find(j->first) == active.end()) {
        std::vector<DependenceVector> dv = j->second;
        g.connect(i, j->first, dv);
      } else {
        std::vector<DependenceVector> dv = j->second;
        for (int k = 0; k < dv.size(); k++)
          switch (dv[k].type) {
          case DEP_W2R:
          case DEP_R2W:
          case DEP_W2W:
          case DEP_R2R: {
            for (int d = 0; d < dep.num_dim(); d++)
              if (dep_pi[d] != d) {
                dv[k].lbounds[d] = -posInfinity;
                dv[k].ubounds[d] = posInfinity;
              }
            break;
          }
          case DEP_CONTROL:
            break;
          default:
            throw loop_error("unknown dependence type");
          }
        g.connect(i, j->first, dv);
      }
    }
  dep = g;
  
  // update loop level information
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    int cur_dep_dim = min_dep_dim;
    std::vector<LoopLevel> new_loop_level(stmt[*i].loop_level.size());
    for (int j = 1; j <= stmt[*i].loop_level.size(); j++)
      if (j >= level && j < level + pi.size()) {
        switch (stmt[*i].loop_level[pi_inverse[j - level] - 1].type) {
        case LoopLevelOriginal:
          new_loop_level[j - 1].type = LoopLevelOriginal;
          new_loop_level[j - 1].payload = cur_dep_dim++;
          new_loop_level[j - 1].parallel_level =
            stmt[*i].loop_level[pi_inverse[j - level] - 1].parallel_level;
          break;
        case LoopLevelTile: {
          new_loop_level[j - 1].type = LoopLevelTile;
          int ref_level = stmt[*i].loop_level[pi_inverse[j - level]
                                              - 1].payload;
          if (ref_level >= level && ref_level < level + pi.size())
            new_loop_level[j - 1].payload = pi_inverse[ref_level
                                                       - level];
          else
            new_loop_level[j - 1].payload = ref_level;
          new_loop_level[j - 1].parallel_level = stmt[*i].loop_level[j
                                                                     - 1].parallel_level;
          break;
        }
        default:
          throw loop_error(
            "unknown loop level information for statement "
            + to_string(*i));
        }
      } else {
        switch (stmt[*i].loop_level[j - 1].type) {
        case LoopLevelOriginal:
          new_loop_level[j - 1].type = LoopLevelOriginal;
          new_loop_level[j - 1].payload =
            stmt[*i].loop_level[j - 1].payload;
          new_loop_level[j - 1].parallel_level = stmt[*i].loop_level[j
                                                                     - 1].parallel_level;
          break;
        case LoopLevelTile: {
          new_loop_level[j - 1].type = LoopLevelTile;
          int ref_level = stmt[*i].loop_level[j - 1].payload;
          if (ref_level >= level && ref_level < level + pi.size())
            new_loop_level[j - 1].payload = pi_inverse[ref_level
                                                       - level];
          else
            new_loop_level[j - 1].payload = ref_level;
          new_loop_level[j - 1].parallel_level = stmt[*i].loop_level[j
                                                                     - 1].parallel_level;
          break;
        }
        default:
          throw loop_error(
            "unknown loop level information for statement "
            + to_string(*i));
        }
      }
    stmt[*i].loop_level = new_loop_level;
  }
  
  setLexicalOrder(2 * level - 2, active, starting_order);
}
void Loop::permute(const std::set<int> &active, const std::vector<int> &pi) {
  if (active.size() == 0 || pi.size() == 0)
    return;
  
  // check for sanity of parameters
  int level = pi[0];
  for (int i = 1; i < pi.size(); i++)
    if (pi[i] < level)
      level = pi[i];
  if (level < 1)
    throw std::invalid_argument("invalid permuation");
  std::vector<int> reverse_pi(pi.size(), 0);
  for (int i = 0; i < pi.size(); i++)
    if (pi[i] >= level + pi.size())
      throw std::invalid_argument("invalid permutation");
    else
      reverse_pi[pi[i] - level] = i + level;
  for (int i = 0; i < reverse_pi.size(); i++)
    if (reverse_pi[i] == 0)
      throw std::invalid_argument("invalid permuation");
  int ref_stmt_num;
  std::vector<int> lex;
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    if (*i < 0 || *i >= stmt.size())
      throw std::invalid_argument("invalid statement " + to_string(*i));
    if (i == active.begin()) {
      ref_stmt_num = *i;
      lex = getLexicalOrder(*i);
    } else {
      if (level + pi.size() - 1 > stmt[*i].loop_level.size())
        throw std::invalid_argument("invalid permuation");
      std::vector<int> lex2 = getLexicalOrder(*i);
      for (int j = 0; j < 2 * level - 3; j += 2)
        if (lex[j] != lex2[j])
          throw std::invalid_argument(
            "statements to permute must be in the same subloop");
      for (int j = 0; j < pi.size(); j++)
        if (!(stmt[*i].loop_level[level + j - 1].type
              == stmt[ref_stmt_num].loop_level[level + j - 1].type
              && stmt[*i].loop_level[level + j - 1].payload
              == stmt[ref_stmt_num].loop_level[level + j - 1].payload))
          throw std::invalid_argument(
            "permuted loops must have the same loop level types");
    }
  }

  invalidateCodeGen();

  // Update transformation relations
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    int n = stmt[*i].xform.n_out();
    omega::Relation mapping(n, n);
    F_And *f_root = mapping.add_and();
    for (int j = 1; j <= n; j += 2) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(j), 1);
      h.update_coef(mapping.input_var(j), -1);
    }
    for (int j = 0; j < pi.size(); j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2 * (level + j)), 1);
      h.update_coef(mapping.input_var(2 * pi[j]), -1);
    }
    for (int j = 1; j < level; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2 * j), 1);
      h.update_coef(mapping.input_var(2 * j), -1);
    }
    for (int j = level + pi.size(); j <= stmt[*i].loop_level.size(); j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(2 * j), 1);
      h.update_coef(mapping.input_var(2 * j), -1);
    }
    
    stmt[*i].xform = Composition(mapping, stmt[*i].xform);
    stmt[*i].xform.simplify();
  }
  
  // get the permuation for dependence vectors
  std::vector<int> t;
  for (int i = 0; i < pi.size(); i++)
    if (stmt[ref_stmt_num].loop_level[pi[i] - 1].type == LoopLevelOriginal)
      t.push_back(stmt[ref_stmt_num].loop_level[pi[i] - 1].payload);
  int max_dep_dim = -1;
  int min_dep_dim = num_dep_dim;
  for (int i = 0; i < t.size(); i++) {
    if (t[i] > max_dep_dim)
      max_dep_dim = t[i];
    if (t[i] < min_dep_dim)
      min_dep_dim = t[i];
  }
  if (min_dep_dim > max_dep_dim)
    return;
  if (max_dep_dim - min_dep_dim + 1 != t.size())
    throw loop_error("cannot update the dependence graph after permuation");
  std::vector<int> dep_pi(num_dep_dim);
  for (int i = 0; i < min_dep_dim; i++)
    dep_pi[i] = i;
  for (int i = min_dep_dim; i <= max_dep_dim; i++)
    dep_pi[i] = t[i - min_dep_dim];
  for (int i = max_dep_dim + 1; i < num_dep_dim; i++)
    dep_pi[i] = i;
  
  dep.permute(dep_pi, active);
  
  // update the dependence graph
  DependenceGraph g(dep.num_dim());
  for (int i = 0; i < dep.vertex.size(); i++)
    g.insert();
  for (int i = 0; i < dep.vertex.size(); i++)
    for (DependenceGraph::EdgeList::iterator j =
           dep.vertex[i].second.begin(); j != dep.vertex[i].second.end();
         j++) {         //
      if ((active.find(i) != active.end()
           && active.find(j->first) != active.end())) {
        std::vector<DependenceVector> dv = j->second;
        for (int k = 0; k < dv.size(); k++) {
          switch (dv[k].type) {
          case DEP_W2R:
          case DEP_R2W:
          case DEP_W2W:
          case DEP_R2R: {
            std::vector<coef_t> lbounds(num_dep_dim);
            std::vector<coef_t> ubounds(num_dep_dim);
            for (int d = 0; d < num_dep_dim; d++) {
              lbounds[d] = dv[k].lbounds[dep_pi[d]];
              ubounds[d] = dv[k].ubounds[dep_pi[d]];
            }
            dv[k].lbounds = lbounds;
            dv[k].ubounds = ubounds;
            break;
          }
          case DEP_CONTROL: {
            break;
          }
          default:
            throw loop_error("unknown dependence type");
          }
        }
        g.connect(i, j->first, dv);
      } else if (active.find(i) == active.end()
                 && active.find(j->first) == active.end()) {
        std::vector<DependenceVector> dv = j->second;
        g.connect(i, j->first, dv);
      } else {
        std::vector<DependenceVector> dv = j->second;
        for (int k = 0; k < dv.size(); k++)
          switch (dv[k].type) {
          case DEP_W2R:
          case DEP_R2W:
          case DEP_W2W:
          case DEP_R2R: {
            for (int d = 0; d < num_dep_dim; d++)
              if (dep_pi[d] != d) {
                dv[k].lbounds[d] = -posInfinity;
                dv[k].ubounds[d] = posInfinity;
              }
            break;
          }
          case DEP_CONTROL:
            break;
          default:
            throw loop_error("unknown dependence type");
          }
        g.connect(i, j->first, dv);
      }
    }
  dep = g;
  
  // update loop level information
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++) {
    int cur_dep_dim = min_dep_dim;
    std::vector<LoopLevel> new_loop_level(stmt[*i].loop_level.size());
    for (int j = 1; j <= stmt[*i].loop_level.size(); j++)
      if (j >= level && j < level + pi.size()) {
        switch (stmt[*i].loop_level[reverse_pi[j - level] - 1].type) {
        case LoopLevelOriginal:
          new_loop_level[j - 1].type = LoopLevelOriginal;
          new_loop_level[j - 1].payload = cur_dep_dim++;
          new_loop_level[j - 1].parallel_level =
            stmt[*i].loop_level[reverse_pi[j - level] - 1].parallel_level;
          break;
        case LoopLevelTile: {
          new_loop_level[j - 1].type = LoopLevelTile;
          int ref_level = stmt[*i].loop_level[reverse_pi[j - level]-1].payload;
          if (ref_level >= level && ref_level < level + pi.size())
            new_loop_level[j - 1].payload = reverse_pi[ref_level
                                                       - level];
          else
            new_loop_level[j - 1].payload = ref_level;
          new_loop_level[j - 1].parallel_level =
            stmt[*i].loop_level[reverse_pi[j - level] - 1].parallel_level;
          break;
        }
        default:
          throw loop_error(
            "unknown loop level information for statement "
            + to_string(*i));
        }
      } else {
        switch (stmt[*i].loop_level[j - 1].type) {
        case LoopLevelOriginal:
          new_loop_level[j - 1].type = LoopLevelOriginal;
          new_loop_level[j - 1].payload =
            stmt[*i].loop_level[j - 1].payload;
          new_loop_level[j - 1].parallel_level = stmt[*i].loop_level[j
                                                                     - 1].parallel_level;
          break;
        case LoopLevelTile: {
          new_loop_level[j - 1].type = LoopLevelTile;
          int ref_level = stmt[*i].loop_level[j - 1].payload;
          if (ref_level >= level && ref_level < level + pi.size())
            new_loop_level[j - 1].payload = reverse_pi[ref_level
                                                       - level];
          else
            new_loop_level[j - 1].payload = ref_level;
          new_loop_level[j - 1].parallel_level = stmt[*i].loop_level[j
                                                                     - 1].parallel_level;
          break;
        }
        default:
          throw loop_error(
            "unknown loop level information for statement "
            + to_string(*i));
        }
      }
    stmt[*i].loop_level = new_loop_level;
  }
  
  setLexicalOrder(2 * level - 2, active);
}


void Loop::set_array_size(std::string name, int size ){
  array_dims.insert(std::pair<std::string, int >(name, size));
}


std::set<int> Loop::split(int stmt_num, int level, const omega::Relation &cond) {
  // check for sanity of parameters
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invalid statement " + to_string(stmt_num));
  if (level <= 0 || level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("4invalid loop level " + to_string(level));
  
  std::set<int> result;
  int dim = 2 * level - 1;
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_loop = getStatements(lex, dim - 1);
  
  omega::Relation cond2 = copy(cond);
  cond2.simplify();
  cond2 = EQs_to_GEQs(cond2);
  Conjunct *c = cond2.single_conjunct();
  int cur_lex = lex[dim - 1];
  bool shifted = false;

  for (GEQ_Iterator gi(c->GEQs()); gi; gi++) {
    int max_level = (*gi).max_tuple_pos();
    omega::Relation single_cond(max_level);
    single_cond.and_with_GEQ(*gi);
    
    // TODO: should decide where to place newly created statements with
    // complementary split condition from dependence graph.
    bool place_after;
    if (max_level == 0)
      place_after = true;
    else if ((*gi).get_coef(cond2.set_var(max_level)) < 0)
      place_after = true;
    else
      place_after = false;
    
    bool temp_place_after;      // = place_after;
    bool assigned = false;
    // original statements with split condition,
    // new statements with complement of split condition
    int old_num_stmt = stmt.size();
    std::map<int, int> what_stmt_num;
    apply_xform(same_loop);
    for (std::set<int>::iterator i = same_loop.begin();
         i != same_loop.end(); i++) {
      int n = stmt[*i].IS.n_set();
      omega::Relation part1, part2;
      if (max_level > n) {
        part1 = copy(stmt[*i].IS);
        part2 = omega::Relation::False(0);
      } else {
        part1 = Intersection(copy(stmt[*i].IS),
                             Extend_Set(copy(single_cond), n - max_level));
        part2 = Intersection(copy(stmt[*i].IS),
                             Extend_Set(Complement(copy(single_cond)),
                                        n - max_level));
      }
      
      //split dependence check
      
      if (max_level > level) {
        
        DNF_Iterator di1(stmt[*i].IS.query_DNF());
        DNF_Iterator di2(part1.query_DNF());
        for (; di1 && di2; di1++, di2++) {
          //printf("In next conjunct,\n");
          EQ_Iterator ei1 = (*di1)->EQs();
          EQ_Iterator ei2 = (*di2)->EQs();
          for (; ei1 && ei2; ei1++, ei2++) {
            //printf(" In next equality constraint,\n");
            Constr_Vars_Iter cvi1(*ei1);
            Constr_Vars_Iter cvi2(*ei2);
            int dimension = (*cvi1).var->get_position();
            int same = 0;
            bool identical = false;
            if (identical = !strcmp((*cvi1).var->char_name(),
                                    (*cvi2).var->char_name())) {
              
              for (; cvi1 && cvi2; cvi1++, cvi2++) {
                
                if (((*cvi1).coef != (*cvi2).coef
                     || (*ei1).get_const()
                     != (*ei2).get_const())
                    || (strcmp((*cvi1).var->char_name(),
                               (*cvi2).var->char_name()))) {
                  
                  same++;
                }
              }
            }
            if ((same != 0) || !identical) {
              
              dimension = dimension - 1;
              
              while (stmt[*i].loop_level[dimension].type
                     == LoopLevelTile)
                dimension =
                  stmt[*i].loop_level[dimension].payload;
              
              dimension = stmt[*i].loop_level[dimension].payload;

              for (int i = 0; i < stmt.size(); i++) {
                std::vector<std::pair<int, DependenceVector> > D;
                for (DependenceGraph::EdgeList::iterator j =
                       dep.vertex[i].second.begin();
                     j != dep.vertex[i].second.end(); j++) {
                  for (int k = 0; k < j->second.size(); k++) {
                    DependenceVector dv = j->second[k];
                    if (dv.type != DEP_CONTROL)
                      if (dv.hasNegative(dimension)
                          && !dv.quasi)
                        throw loop_error(
                          "loop error: Split is illegal, dependence violation!");
                    
                  }
                }
              }
              
            }
            
            GEQ_Iterator gi1 = (*di1)->GEQs();
            GEQ_Iterator gi2 = (*di2)->GEQs();
            
            for (; gi1 && gi2; gi++, gi2++) {
              
              Constr_Vars_Iter cvi1(*gi1);
              Constr_Vars_Iter cvi2(*gi2);
              int dimension = (*cvi1).var->get_position();
              int same = 0;
              bool identical = false;
              if (identical = !strcmp((*cvi1).var->char_name(),
                                      (*cvi2).var->char_name())) {
                
                for (; cvi1 && cvi2; cvi1++, cvi2++) {
                  
                  if (((*cvi1).coef != (*cvi2).coef
                       || (*gi1).get_const()
                       != (*gi2).get_const())
                      || (strcmp((*cvi1).var->char_name(),
                                 (*cvi2).var->char_name()))) {
                    
                    same++;
                  }
                }
              }
              if ((same != 0) || !identical) {
                dimension = dimension - 1;

                while (stmt[*i].loop_level[dimension].type
                       == LoopLevelTile)
                  dimension =
                    stmt[*i].loop_level[dimension].payload;
                
                dimension =
                  stmt[*i].loop_level[dimension].payload;
                
                for (int i = 0; i < stmt.size(); i++) {
                  std::vector<std::pair<int, DependenceVector> > D;
                  for (DependenceGraph::EdgeList::iterator j =
                         dep.vertex[i].second.begin();
                       j != dep.vertex[i].second.end();
                       j++) {
                    for (int k = 0; k < j->second.size();
                         k++) {
                      DependenceVector dv = j->second[k];
                      if (dv.type != DEP_CONTROL)
                        if (dv.hasNegative(dimension)
                            && !dv.quasi)
                          
                          throw loop_error(
                            "loop error: Split is illegal, dependence violation!");
                      
                    }
                  }
                }
                
              }
              
            }
            
          }
          
        }
        
        DNF_Iterator di3(stmt[*i].IS.query_DNF());
        DNF_Iterator di4(part2.query_DNF());        //
        for (; di3 && di4; di3++, di4++) {
          EQ_Iterator ei1 = (*di3)->EQs();
          EQ_Iterator ei2 = (*di4)->EQs();
          for (; ei1 && ei2; ei1++, ei2++) {
            Constr_Vars_Iter cvi1(*ei1);
            Constr_Vars_Iter cvi2(*ei2);
            int dimension = (*cvi1).var->get_position();
            int same = 0;
            bool identical = false;
            if (identical = !strcmp((*cvi1).var->char_name(),
                                    (*cvi2).var->char_name())) {
              
              for (; cvi1 && cvi2; cvi1++, cvi2++) {
                
                if (((*cvi1).coef != (*cvi2).coef
                     || (*ei1).get_const()
                     != (*ei2).get_const())
                    || (strcmp((*cvi1).var->char_name(),
                               (*cvi2).var->char_name()))) {
                  
                  same++;
                }
              }
            }
            if ((same != 0) || !identical) {
              dimension = dimension - 1;
              
              while (stmt[*i].loop_level[dimension].type
                     == LoopLevelTile)
                dimension = stmt[*i].loop_level[dimension].payload;
              
              dimension = stmt[*i].loop_level[dimension].payload;
              
              for (int i = 0; i < stmt.size(); i++) {
                std::vector<std::pair<int, DependenceVector> > D;
                for (DependenceGraph::EdgeList::iterator j =
                       dep.vertex[i].second.begin();
                     j != dep.vertex[i].second.end(); j++) {
                  for (int k = 0; k < j->second.size(); k++) {
                    DependenceVector dv = j->second[k];
                    if (dv.type != DEP_CONTROL)
                      if (dv.hasNegative(dimension)
                          && !dv.quasi)
                        
                        throw loop_error(
                          "loop error: Split is illegal, dependence violation!");
                    
                  }
                }
              }
              
            }
            
          }
          GEQ_Iterator gi1 = (*di3)->GEQs();
          GEQ_Iterator gi2 = (*di4)->GEQs();
          
          for (; gi1 && gi2; gi++, gi2++) {
            Constr_Vars_Iter cvi1(*gi1);
            Constr_Vars_Iter cvi2(*gi2);
            int dimension = (*cvi1).var->get_position();
            int same = 0;
            bool identical = false;
            if (identical = !strcmp((*cvi1).var->char_name(),
                                    (*cvi2).var->char_name())) {
              
              for (; cvi1 && cvi2; cvi1++, cvi2++) {
                
                if (((*cvi1).coef != (*cvi2).coef
                     || (*gi1).get_const()
                     != (*gi2).get_const())
                    || (strcmp((*cvi1).var->char_name(),
                               (*cvi2).var->char_name()))) {
                  
                  same++;
                }
              }
            }
            if ((same != 0) || !identical) {
              dimension = dimension - 1;
              
              while (stmt[*i].loop_level[dimension].type        //
                     == LoopLevelTile)
                dimension = stmt[*i].loop_level[dimension].payload;
              
              dimension = stmt[*i].loop_level[dimension].payload;
              
              for (int i = 0; i < stmt.size(); i++) {
                std::vector<std::pair<int, DependenceVector> > D;
                for (DependenceGraph::EdgeList::iterator j =
                       dep.vertex[i].second.begin();
                     j != dep.vertex[i].second.end(); j++) {
                  for (int k = 0; k < j->second.size(); k++) {
                    DependenceVector dv = j->second[k];
                    if (dv.type != DEP_CONTROL)
                      if (dv.hasNegative(dimension)
                          && !dv.quasi)
                        
                        throw loop_error(
                          "loop error: Split is illegal, dependence violation!");
                    
                  }
                }
              }
              
            }
            
          }
          
        }
        
      }
      
      stmt[*i].IS = part1;
      
      int n1 = part2.n_set();
      int m = this->known.n_set();
      omega::Relation test;
      if(m > n1)
        test = Intersection(copy(this->known),
                            Extend_Set(copy(part2), m - part2.n_set()));
      else
        test = Intersection(copy(part2),
                            Extend_Set(copy(this->known), n1 - this->known.n_set()));
      
      if (test.is_upper_bound_satisfiable()) {
        Statement new_stmt;
        new_stmt.code = stmt[*i].code->clone();
        new_stmt.IS = part2;
        new_stmt.xform = copy(stmt[*i].xform);
        new_stmt.ir_stmt_node = NULL;
        new_stmt.loop_level = stmt[*i].loop_level;
        
        new_stmt.has_inspector = stmt[*i].has_inspector;
        new_stmt.reduction = stmt[*i].reduction;
        new_stmt.reductionOp = stmt[*i].reductionOp;
        
        stmt_nesting_level_.push_back(stmt_nesting_level_[*i]);
        
        if (!shifted) {
          shifted = true;
          // make adjacent lexical number available for new statements
          if (place_after) {
            lex[dim - 1] = cur_lex + 1;
            shiftLexicalOrder(lex, dim - 1, 1);
          } else {
            lex[dim - 1] = cur_lex - 1;
            shiftLexicalOrder(lex, dim - 1, -1);
          }
        }
        if (place_after)
          assign_const(new_stmt.xform, dim - 1, cur_lex + 1);
        else
          assign_const(new_stmt.xform, dim - 1, cur_lex - 1);
        
        debug_fprintf(stderr, "loop_basic.cc L828 adding stmt %d\n", stmt.size()); 
        stmt.push_back(new_stmt);

        uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
        uninterpreted_symbols_stringrepr.push_back(uninterpreted_symbols_stringrepr[stmt_num]);
        dep.insert();
        what_stmt_num[*i] = stmt.size() - 1;
        if (*i == stmt_num)
          result.insert(stmt.size() - 1);
      }
      
    }
    // update dependence graph
    int dep_dim = get_dep_dim_of(stmt_num, level);
    for (int i = 0; i < old_num_stmt; i++) {
      std::vector<std::pair<int, std::vector<DependenceVector> > > D;
      
      for (DependenceGraph::EdgeList::iterator j =
             dep.vertex[i].second.begin();
           j != dep.vertex[i].second.end(); j++) {
        if (same_loop.find(i) != same_loop.end()) {
          if (same_loop.find(j->first) != same_loop.end()) {
            if (what_stmt_num.find(i) != what_stmt_num.end()
                && what_stmt_num.find(j->first)
                != what_stmt_num.end())
              dep.connect(what_stmt_num[i],
                          what_stmt_num[j->first], j->second);
            if (place_after
                && what_stmt_num.find(j->first)
                != what_stmt_num.end()) {
              std::vector<DependenceVector> dvs;
              for (int k = 0; k < j->second.size(); k++) {
                DependenceVector dv = j->second[k];
                if (dv.is_data_dependence() && dep_dim != -1) {
                  dv.lbounds[dep_dim] = -posInfinity;
                  dv.ubounds[dep_dim] = posInfinity;
                }
                dvs.push_back(dv);
              }
              if (dvs.size() > 0)
                D.push_back(
                  std::make_pair(what_stmt_num[j->first],
                                 dvs));
            } else if (!place_after
                       && what_stmt_num.find(i)
                       != what_stmt_num.end()) {
              std::vector<DependenceVector> dvs;
              for (int k = 0; k < j->second.size(); k++) {
                DependenceVector dv = j->second[k];
                if (dv.is_data_dependence() && dep_dim != -1) {
                  dv.lbounds[dep_dim] = -posInfinity;
                  dv.ubounds[dep_dim] = posInfinity;
                }
                dvs.push_back(dv);
              }
              if (dvs.size() > 0)
                dep.connect(what_stmt_num[i], j->first, dvs);
              
            }
          } else {
            if (what_stmt_num.find(i) != what_stmt_num.end())
              dep.connect(what_stmt_num[i], j->first, j->second);
          }
        } else if (same_loop.find(j->first) != same_loop.end()) {
          if (what_stmt_num.find(j->first) != what_stmt_num.end())
            D.push_back(
              std::make_pair(what_stmt_num[j->first],
                             j->second));
        }
      }
      
      for (int j = 0; j < D.size(); j++)
        dep.connect(i, D[j].first, D[j].second);
    }
    
  }
  
  return result;
}

void Loop::skew(const std::set<int> &stmt_nums, int level,
                const std::vector<int> &skew_amount) {
  if (stmt_nums.size() == 0)
    return;
  
  // check for sanity of parameters
  int ref_stmt_num = *(stmt_nums.begin());
  for (std::set<int>::const_iterator i = stmt_nums.begin();
       i != stmt_nums.end(); i++) {
    if (*i < 0 || *i >= stmt.size())
      throw std::invalid_argument(
        "invalid statement number " + to_string(*i));
    if (level < 1 || level > stmt[*i].loop_level.size())
      throw std::invalid_argument(
        "5invalid loop level " + to_string(level));
    for (int j = stmt[*i].loop_level.size(); j < skew_amount.size(); j++)
      if (skew_amount[j] != 0)
        throw std::invalid_argument("invalid skewing formula");
  }

  invalidateCodeGen();

  // set trasformation relations
  for (std::set<int>::const_iterator i = stmt_nums.begin();
       i != stmt_nums.end(); i++) {
    int n = stmt[*i].xform.n_out();
    omega::Relation r(n, n);
    F_And *f_root = r.add_and();
    for (int j = 1; j <= n; j++)
      if (j != 2 * level) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(r.input_var(j), 1);
        h.update_coef(r.output_var(j), -1);
      }
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(r.output_var(2 * level), -1);
    for (int j = 0; j < skew_amount.size(); j++)
      if (skew_amount[j] != 0)
        h.update_coef(r.input_var(2 * (j + 1)), skew_amount[j]);
    
    stmt[*i].xform = Composition(r, stmt[*i].xform);
    stmt[*i].xform.simplify();
  }
  
  // update dependence graph
  if (stmt[ref_stmt_num].loop_level[level - 1].type == LoopLevelOriginal) {
    int dep_dim = stmt[ref_stmt_num].loop_level[level - 1].payload;
    for (std::set<int>::const_iterator i = stmt_nums.begin();
         i != stmt_nums.end(); i++)
      for (DependenceGraph::EdgeList::iterator j =
             dep.vertex[*i].second.begin();
           j != dep.vertex[*i].second.end(); j++)
        if (stmt_nums.find(j->first) != stmt_nums.end()) {
          // dependence between skewed statements
          std::vector<DependenceVector> dvs = j->second;
          for (int k = 0; k < dvs.size(); k++) {
            DependenceVector &dv = dvs[k];
            if (dv.is_data_dependence()) {
              coef_t lb = 0;
              coef_t ub = 0;
              for (int kk = 0; kk < skew_amount.size(); kk++) {
                int cur_dep_dim = get_dep_dim_of(*i, kk + 1);
                if (skew_amount[kk] > 0) {
                  if (lb != -posInfinity
                      && stmt[*i].loop_level[kk].type == LoopLevelOriginal
                      && dv.lbounds[cur_dep_dim] != -posInfinity)
                    lb += skew_amount[kk] * dv.lbounds[cur_dep_dim];
                  else {
                    if (cur_dep_dim != -1
                        && !(dv.lbounds[cur_dep_dim] == 0
                             && dv.ubounds[cur_dep_dim]== 0))
                      lb = -posInfinity;
                  }
                  if (ub != posInfinity
                      && stmt[*i].loop_level[kk].type == LoopLevelOriginal
                      && dv.ubounds[cur_dep_dim] != posInfinity)
                    ub += skew_amount[kk] * dv.ubounds[cur_dep_dim];
                  else {
                    if (cur_dep_dim != -1
                        && !(dv.lbounds[cur_dep_dim] == 0
                             && dv.ubounds[cur_dep_dim] == 0))
                      ub = posInfinity;
                  }
                } else if (skew_amount[kk] < 0) {
                  if (lb != -posInfinity
                      && stmt[*i].loop_level[kk].type == LoopLevelOriginal
                      && dv.ubounds[cur_dep_dim] != posInfinity)
                    lb += skew_amount[kk] * dv.ubounds[cur_dep_dim];
                  else {
                    if (cur_dep_dim != -1
                        && !(dv.lbounds[cur_dep_dim] == 0
                             && dv.ubounds[cur_dep_dim] == 0))
                      lb = -posInfinity;
                  }
                  if (ub != posInfinity
                      && stmt[*i].loop_level[kk].type == LoopLevelOriginal
                      && dv.lbounds[cur_dep_dim] != -posInfinity)
                    ub += skew_amount[kk] * dv.lbounds[cur_dep_dim];
                  else {
                    if (cur_dep_dim != -1
                        && !(dv.lbounds[cur_dep_dim] == 0
                             && dv.ubounds[cur_dep_dim] == 0))
                      ub = posInfinity;
                  }
                }
              }
              dv.lbounds[dep_dim] = lb;
              dv.ubounds[dep_dim] = ub;
              if ((dv.isCarried(dep_dim) && dv.hasPositive(dep_dim)) 
                  && dv.quasi)
                dv.quasi = false;
              
              if ((dv.isCarried(dep_dim) && dv.hasNegative(dep_dim)) 
                  && !dv.quasi)
                throw loop_error(
                  "loop error: Skewing is illegal, dependence violation!");
              dv.lbounds[dep_dim] = lb;
              dv.ubounds[dep_dim] = ub;
              if ((dv.isCarried(dep_dim)
                   && dv.hasPositive(dep_dim)) && dv.quasi)
                dv.quasi = false;
              
              if ((dv.isCarried(dep_dim)
                   && dv.hasNegative(dep_dim)) && !dv.quasi)
                throw loop_error(
                  "loop error: Skewing is illegal, dependence violation!");
            }
          }
          j->second = dvs;
        } else {
          // dependence from skewed statement to unskewed statement becomes jumbled,
          // put distance value at skewed dimension to unknown
          std::vector<DependenceVector> dvs = j->second;
          for (int k = 0; k < dvs.size(); k++) {
            DependenceVector &dv = dvs[k];
            if (dv.is_data_dependence()) {
              dv.lbounds[dep_dim] = -posInfinity;
              dv.ubounds[dep_dim] = posInfinity;
            }
          }
          j->second = dvs;
        }
    for (int i = 0; i < dep.vertex.size(); i++)
      if (stmt_nums.find(i) == stmt_nums.end())
        for (DependenceGraph::EdgeList::iterator j =
               dep.vertex[i].second.begin();
             j != dep.vertex[i].second.end(); j++)
          if (stmt_nums.find(j->first) != stmt_nums.end()) {
            // dependence from unskewed statement to skewed statement becomes jumbled,
            // put distance value at skewed dimension to unknown
            std::vector<DependenceVector> dvs = j->second;
            for (int k = 0; k < dvs.size(); k++) {
              DependenceVector &dv = dvs[k];
              if (dv.is_data_dependence()) {
                dv.lbounds[dep_dim] = -posInfinity;
                dv.ubounds[dep_dim] = posInfinity;
              }
            }
            j->second = dvs;
          }
  }
}


void Loop::shift(const std::set<int> &stmt_nums, int level, int shift_amount) {
  if (stmt_nums.size() == 0)
    return;
  
  // check for sanity of parameters
  int ref_stmt_num = *(stmt_nums.begin());
  for (std::set<int>::const_iterator i = stmt_nums.begin();
       i != stmt_nums.end(); i++) {
    if (*i < 0 || *i >= stmt.size())
      throw std::invalid_argument(
        "invalid statement number " + to_string(*i));
    if (level < 1 || level > stmt[*i].loop_level.size())
      throw std::invalid_argument(
        "6invalid loop level " + to_string(level));
  }
  
  // do nothing
  if (shift_amount == 0)
    return;

  invalidateCodeGen();

  // set trasformation relations
  for (std::set<int>::const_iterator i = stmt_nums.begin();
       i != stmt_nums.end(); i++) {
    int n = stmt[*i].xform.n_out();
    
    omega::Relation r(n, n);
    F_And *f_root = r.add_and();
    for (int j = 1; j <= n; j++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(r.input_var(j), 1);
      h.update_coef(r.output_var(j), -1);
      if (j == 2 * level)
        h.update_const(shift_amount);
    }
    
    stmt[*i].xform = Composition(r, stmt[*i].xform);
    stmt[*i].xform.simplify();
  }
  
  // update dependence graph
  if (stmt[ref_stmt_num].loop_level[level - 1].type == LoopLevelOriginal) {
    int dep_dim = stmt[ref_stmt_num].loop_level[level - 1].payload;
    for (std::set<int>::const_iterator i = stmt_nums.begin();
         i != stmt_nums.end(); i++)
      for (DependenceGraph::EdgeList::iterator j =
             dep.vertex[*i].second.begin();
           j != dep.vertex[*i].second.end(); j++)
        if (stmt_nums.find(j->first) == stmt_nums.end()) {
          // dependence from shifted statement to unshifted statement
          std::vector<DependenceVector> dvs = j->second;
          for (int k = 0; k < dvs.size(); k++) {
            DependenceVector &dv = dvs[k];
            if (dv.is_data_dependence()) {
              if (dv.lbounds[dep_dim] != -posInfinity)
                dv.lbounds[dep_dim] -= shift_amount;
              if (dv.ubounds[dep_dim] != posInfinity)
                dv.ubounds[dep_dim] -= shift_amount;
            }
          }
          j->second = dvs;
        }
    for (int i = 0; i < dep.vertex.size(); i++)
      if (stmt_nums.find(i) == stmt_nums.end())
        for (DependenceGraph::EdgeList::iterator j =
               dep.vertex[i].second.begin();
             j != dep.vertex[i].second.end(); j++)
          if (stmt_nums.find(j->first) != stmt_nums.end()) {
            // dependence from unshifted statement to shifted statement
            std::vector<DependenceVector> dvs = j->second;
            for (int k = 0; k < dvs.size(); k++) {
              DependenceVector &dv = dvs[k];
              if (dv.is_data_dependence()) {
                if (dv.lbounds[dep_dim] != -posInfinity)
                  dv.lbounds[dep_dim] += shift_amount;
                if (dv.ubounds[dep_dim] != posInfinity)
                  dv.ubounds[dep_dim] += shift_amount;
              }
            }
            j->second = dvs;
          }
  }
}

void Loop::scale(const std::set<int> &stmt_nums, int level, int scale_amount) {
  std::vector<int> skew_amount(level, 0);
  skew_amount[level - 1] = scale_amount;
  skew(stmt_nums, level, skew_amount);
}

void Loop::reverse(const std::set<int> &stmt_nums, int level) {
  scale(stmt_nums, level, -1);
}

void Loop::fuse(const std::set<int> &stmt_nums, int level) {
  if (stmt_nums.size() == 0 || stmt_nums.size() == 1)
    return;

  invalidateCodeGen();

  int dim = 2 * level - 1;
  // check for sanity of parameters
  std::vector<int> ref_lex;
  int ref_stmt_num;
  apply_xform(); 
  for (std::set<int>::const_iterator i = stmt_nums.begin();
       i != stmt_nums.end(); i++) {
    if (*i < 0 || *i >= stmt.size()) {
      debug_fprintf(stderr, "statement number %d   should be in [0, %d)\n", *i, stmt.size()); 
      throw std::invalid_argument(
        "FUSE invalid statement number " + to_string(*i));
    }
    if (level <= 0
        //    || (level > (stmt[*i].xform.n_out() - 1) / 2
        //    || level > stmt[*i].loop_level.size())
      ) {
      debug_fprintf(stderr, "FUSE level %d ", level);
      debug_fprintf(stderr, "must be greater than zero and \n");
      debug_fprintf(stderr, "must NOT be greater than (%d - 1)/2 == %d   and\n", stmt[*i].xform.n_out(), (stmt[*i].xform.n_out() - 1) / 2);
      debug_fprintf(stderr, "must NOT be greater than %d\n", stmt[*i].loop_level.size());
      throw std::invalid_argument(
        "FUSE invalid loop level " + to_string(level));
    }
    if (ref_lex.size() == 0) {
      ref_lex = getLexicalOrder(*i);
      ref_stmt_num = *i;
    } else {
      std::vector<int> lex = getLexicalOrder(*i);
      for (int j = 0; j < dim - 1; j += 2)
        if (lex[j] != ref_lex[j])
          throw std::invalid_argument(
            "statements for fusion must be in the same level-"
            + to_string(level - 1) + " subloop");
    }
  }
  
  // collect lexicographical order values from to-be-fused statements
  std::set<int> lex_values;
  for (std::set<int>::const_iterator i = stmt_nums.begin();
       i != stmt_nums.end(); i++) {
    std::vector<int> lex = getLexicalOrder(*i);
    lex_values.insert(lex[dim - 1]);
  }
  if (lex_values.size() == 1)
    return;
  // negative dependence would prevent fusion
  
  int dep_dim = get_dep_dim_of(ref_stmt_num, level);
  
  for (std::set<int>::iterator i = lex_values.begin(); i != lex_values.end();
       i++) {
    ref_lex[dim - 1] = *i;
    std::set<int> a = getStatements(ref_lex, dim - 1);
    std::set<int>::iterator j = i;
    j++;
    for (; j != lex_values.end(); j++) {
      ref_lex[dim - 1] = *j;
      std::set<int> b = getStatements(ref_lex, dim - 1);
      for (std::set<int>::iterator ii = a.begin(); ii != a.end(); ii++)
        for (std::set<int>::iterator jj = b.begin(); jj != b.end();
             jj++) {
          std::vector<DependenceVector> dvs;
          dvs = dep.getEdge(*ii, *jj);
          for (int k = 0; k < dvs.size(); k++)
            if (dvs[k].isCarried(dep_dim)
                && dvs[k].hasNegative(dep_dim))
              throw loop_error(
                "loop error: statements " + to_string(*ii)
                + " and " + to_string(*jj)
                + " cannot be fused together due to negative dependence");
          dvs = dep.getEdge(*jj, *ii);
          for (int k = 0; k < dvs.size(); k++)
            if (dvs[k].isCarried(dep_dim)
                && dvs[k].hasNegative(dep_dim))
              throw loop_error(
                "loop error: statements " + to_string(*jj)
                + " and " + to_string(*ii)
                + " cannot be fused together due to negative dependence");
        }
    }
  }
  
  std::set<int> same_loop = getStatements(ref_lex, dim - 3);
  
  std::vector<std::set<int> > s = sort_by_same_loops(same_loop, level);
  std::vector<bool> s2;

  for (int i = 0; i < s.size(); i++) {
    s2.push_back(false);
  }

  for (std::set<int>::iterator kk = stmt_nums.begin(); kk != stmt_nums.end();
       kk++)
    for (int i = 0; i < s.size(); i++)
      if (s[i].find(*kk) != s[i].end()) {
        
        s2[i] = true;
      }
  
  try {
    
    //Dependence Check for Ordering Constraint
    //Graph<std::set<int>, bool> dummy = construct_induced_graph_at_level(s5,
    //    dep, dep_dim)

    Graph<std::set<int>, bool> g = construct_induced_graph_at_level(s, dep,
                                                                    dep_dim);
    s = typed_fusion(g, s2);
  } catch (const loop_error &e) {
    
    throw loop_error(
      "statements cannot be fused together due to negative dependence");
    
  }
  
  int order = 0;
  for (int i = 0; i < s.size(); i++) {
    int adj = 0;
    int lex = INT16_MIN;
    int llex = 0;
    for (std::set<int>::iterator it = s[i].begin(); it != s[i].end(); it++) {
      int nlex = get_const(stmt[*it].xform, 2 * level - 2, Output_Var);
      int nnlex = get_const(stmt[*it].xform, 2 * level, Output_Var);
      if (nlex != lex) {
        if (lex > INT16_MIN)
          adj = llex + 1 - nnlex;
        lex = nlex;
      }
      llex = nnlex + adj;
      assign_const(stmt[*it].xform, 2 * level, llex);
      assign_const(stmt[*it].xform, 2 * level - 2, order);
    }
    order++;
  }


  //plan for selective typed fusion
  
  /*
    1. sort the lex values of the statements
    2. construct induced graph on sorted statements
    3. pick a node from the graph, check if it is before/after from the candidate set for fusion
    equal-> set the max fused node of this node to be the start/target node for fusion
    before -> augment and continue
    
    4. once target node identified and is on work queue update successors and other nodes to start node
    5. augment and continue
    6. if all candidate nodes dont end up in start node throw error
    7. Get nodes and update lexical values
    
  */

  /* for (std::set<int>::iterator kk = stmt_nums.begin(); kk != stmt_nums.end();
       kk++)
    for (int i = 0; i < s.size(); i++)
      if (s[i].find(*kk) != s[i].end()) {
        s1.insert(s[i].begin(), s[i].end());
        s2.insert(i);
      }
  
  s3.push_back(s1);
  for (int i = 0; i < s.size(); i++)
    if (s2.find(i) == s2.end()) {
      s3.push_back(s[i]);
      s4.insert(s[i].begin(), s[i].end());
    }
  try {
    std::vector<std::set<int> > s5;
    s5.push_back(s1);
    s5.push_back(s4);
    
    //Dependence Check for Ordering Constraint
    //Graph<std::set<int>, bool> dummy = construct_induced_graph_at_level(s5,
    //      dep, dep_dim);
    
    Graph<std::set<int>, bool> g = construct_induced_graph_at_level(s3, dep,
                                                                    dep_dim);
    std::cout<< g;
    s = typed_fusion(g);
  } catch (const loop_error &e) {
    
    throw loop_error(
      "statements cannot be fused together due to negative dependence");
    
  }
  
  if (s3.size() == s.size()) {
    int order = 0;
    for (int i = 0; i < s.size(); i++) {
      
      for (std::set<int>::iterator it = s[i].begin(); it != s[i].end();
           it++) {
        
        assign_const(stmt[*it].xform, 2 * level - 2, order);
        
      }
      
      order++;
    }
  } else if (s3.size() > s.size()) {
    
    int order = 0;
    for (int j = 0; j < s.size(); j++) {
      std::set<int>::iterator it3;
      for (it3 = s1.begin(); it3 != s1.end(); it3++) {
        if (s[j].find(*it3) != s[j].end())
          break;
      }
      if (it3 != s1.end()) {
        for (std::set<int>::iterator it = s1.begin(); it != s1.end();
             it++)
          assign_const(stmt[*it].xform, 2 * level - 2, order);
        
        order++;
        
      }
      
      for (int i = 0; i < s3.size(); i++) {
        std::set<int>::iterator it2;
        
        for (it2 = s3[i].begin(); it2 != s3[i].end(); it2++) {
          if (s[j].find(*it2) != s[j].end())
            break;
        }
        
        if (it2 != s3[i].end()) {
          for (std::set<int>::iterator it = s3[i].begin();
               it != s3[i].end(); it++)
            assign_const(stmt[*it].xform, 2 * level - 2, order);
          
          order++;
          
        }
      }
    }
    
  } else
    throw loop_error("Typed Fusion Error");
  */
}



void Loop::distribute(const std::set<int> &stmt_nums, int level) {
  if (stmt_nums.size() == 0 || stmt_nums.size() == 1)
    return;
  debug_fprintf(stderr, "Loop::distribute()\n");

  invalidateCodeGen();
  int dim = 2 * level - 1;
  int ref_stmt_num;
  // check for sanity of parameters
  std::vector<int> ref_lex;
  for (std::set<int>::const_iterator i = stmt_nums.begin();
       i != stmt_nums.end(); i++) {
    if (*i < 0 || *i >= stmt.size())
      throw std::invalid_argument(
        "invalid statement number " + to_string(*i));
    
    if (level < 1
        || (level > (stmt[*i].xform.n_out() - 1) / 2
            || level > stmt[*i].loop_level.size()))
      throw std::invalid_argument(
        "8invalid loop level " + to_string(level));
    if (ref_lex.size() == 0) {
      ref_lex = getLexicalOrder(*i);
      ref_stmt_num = *i;
    } else {
      std::vector<int> lex = getLexicalOrder(*i);
      for (int j = 0; j <= dim - 1; j += 2)
        if (lex[j] != ref_lex[j])
          throw std::invalid_argument(
            "statements for distribution must be in the same level-"
            + to_string(level) + " subloop");
    }
  }

  // find SCC in the to-be-distributed loop
  int dep_dim = get_dep_dim_of(ref_stmt_num, level);
  std::set<int> same_loop = getStatements(ref_lex, dim - 1);
  Graph<int, Empty> g;
  for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end();
       i++)
    g.insert(*i);
  for (int i = 0; i < g.vertex.size(); i++)
    for (int j = i + 1; j < g.vertex.size(); j++) {
      std::vector<DependenceVector> dvs;
      dvs = dep.getEdge(g.vertex[i].first, g.vertex[j].first);
      for (int k = 0; k < dvs.size(); k++)
        if (dvs[k].isCarried(dep_dim)) {
          g.connect(i, j);
          break;
        }
      dvs = dep.getEdge(g.vertex[j].first, g.vertex[i].first);
      for (int k = 0; k < dvs.size(); k++)
        if (dvs[k].isCarried(dep_dim)) {
          g.connect(j, i);
          break;
        }
    }
  std::vector<std::set<int> > s = g.topoSort();
  // find statements that cannot be distributed due to dependence cycle
  Graph<std::set<int>, Empty> g2;
  for (int i = 0; i < s.size(); i++) {
    std::set<int> t;
    for (std::set<int>::iterator j = s[i].begin(); j != s[i].end(); j++)
      if (stmt_nums.find(g.vertex[*j].first) != stmt_nums.end())
        t.insert(g.vertex[*j].first);
    if (!t.empty())
      g2.insert(t);
  }
  for (int i = 0; i < g2.vertex.size(); i++)
    for (int j = i + 1; j < g2.vertex.size(); j++)
      for (std::set<int>::iterator ii = g2.vertex[i].first.begin();
           ii != g2.vertex[i].first.end(); ii++)
        for (std::set<int>::iterator jj = g2.vertex[j].first.begin();
             jj != g2.vertex[j].first.end(); jj++) {
          std::vector<DependenceVector> dvs;
          dvs = dep.getEdge(*ii, *jj);
          for (int k = 0; k < dvs.size(); k++)
            if (dvs[k].isCarried(dep_dim)) {
              g2.connect(i, j);
              break;
            }
          dvs = dep.getEdge(*jj, *ii);
          for (int k = 0; k < dvs.size(); k++)
            if (dvs[k].isCarried(dep_dim)) {
              g2.connect(j, i);
              break;
            }
        }
  std::vector<std::set<int> > s2 = g2.topoSort();
  // nothing to distribute
  if (s2.size() == 1)
    throw loop_error(
      "loop error: no statement can be distributed due to dependence cycle");
  std::vector<std::set<int> > s3;
  for (int i = 0; i < s2.size(); i++) {
    std::set<int> t;
    for (std::set<int>::iterator j = s2[i].begin(); j != s2[i].end(); j++)
      std::set_union(t.begin(), t.end(), g2.vertex[*j].first.begin(),
                     g2.vertex[*j].first.end(), inserter(t, t.begin()));
    s3.push_back(t);
  }
  // associate other affected statements with the right distributed statements
  for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end();
       i++)
    if (stmt_nums.find(*i) == stmt_nums.end()) {
      bool is_inserted = false;
      int potential_insertion_point = 0;
      for (int j = 0; j < s3.size(); j++) {
        for (std::set<int>::iterator k = s3[j].begin();
             k != s3[j].end(); k++) {
          std::vector<DependenceVector> dvs;
          dvs = dep.getEdge(*i, *k);
          for (int kk = 0; kk < dvs.size(); kk++)
            if (dvs[kk].isCarried(dep_dim)) {
              s3[j].insert(*i);
              is_inserted = true;
              break;
            }
          dvs = dep.getEdge(*k, *i);
          for (int kk = 0; kk < dvs.size(); kk++)
            if (dvs[kk].isCarried(dep_dim))
              potential_insertion_point = j;
        }
        if (is_inserted)
          break;
      }
      if (!is_inserted)
        s3[potential_insertion_point].insert(*i);
    }
  // set lexicographical order after distribution
  int order = ref_lex[dim - 1];
  int shift = 0;
  shiftLexicalOrder(ref_lex, dim - 1, s3.size() - 1);
  for (std::vector<std::set<int> >::iterator i = s3.begin(); i != s3.end();
       i++) {
    ref_lex[dim-1] = order;
    for (std::set<int>::iterator j = (*i).begin(); j != (*i).end(); j++)
      assign_const(stmt[*j].xform, dim - 1, order);
    ref_lex[dim+1] = INT32_MAX;
    shiftLexicalOrder(ref_lex, dim+1, -shift);
    shift += i->size();
    order++;
  }
  // no need to update dependence graph
  
  return;
}




std::vector<IR_ArrayRef *> FindOuterArrayRefs(IR_Code *ir,
                                              std::vector<IR_ArrayRef *> &arr_refs) {
    std::vector<IR_ArrayRef *> to_return;
  for (int i = 0; i < arr_refs.size(); i++)
    if (!ir->parent_is_array(arr_refs[i])) {
      int j;
      for (j = 0; j < to_return.size(); j++)
        if (*to_return[j] == *arr_refs[i])
          break;
      if (j == to_return.size())
        to_return.push_back(arr_refs[i]);
    }
  return to_return;
}





std::vector<std::vector<std::string> > constructInspectorVariables(IR_Code *ir,
                                                                   std::vector<IR_ArrayRef *> &arr, std::vector<std::string> &index) {
  
  debug_fprintf(stderr, "constructInspectorVariables()\n"); 

  std::vector<std::vector<std::string> > to_return;
  
  for (auto i: arr) {
    
    std::vector<std::string> per_index;
    
    CG_outputRepr *subscript = i->index(0);
    
    if (i->n_dim() > 1)
      throw ir_error(
        "multi-dimensional array support non-existent for flattening currently");
    
    while (ir->QueryExpOperation(subscript) == IR_OP_ARRAY_VARIABLE) {
      
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(subscript);
      
      IR_ArrayRef *ref = static_cast<IR_ArrayRef *>(ir->Repr2Ref(v[0]));
      //per_index.push_back(ref->name());
      
      subscript = ref->index(0);
      
    }
    
    if (ir->QueryExpOperation(subscript) == IR_OP_VARIABLE) {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(subscript);
      IR_ScalarRef *ref = static_cast<IR_ScalarRef *>(ir->Repr2Ref(v[0]));
      per_index.push_back(ref->name());
      int j;
      for (j = 0; j < index.size(); j++)
        if (index[j] == ref->name())
          break;
      
      if (j == index.size())
        throw ir_error("Non index variable in array expression");
      
      int k;
      for (k = 0; k < to_return.size(); k++)
        if (to_return[k][0] == ref->name())
          break;
      if (k == to_return.size()) { 
        to_return.push_back(per_index);
        debug_fprintf(stderr, "adding index %s\n", ref->name().c_str()); 
      }
      
    }
    
  }
  
  return to_return;
  
}

/*std::vector<CG_outputRepr *> constructInspectorData(IR_Code *ir, std::vector<std::vector<std::string> >  &indices){
  
  std::vector<CG_outputRepr *> to_return;
  
  for(int i =0; i < indices.size(); i++)
  ir->CreateVariableDeclaration(indices[i][0]);
  return to_return;
  }
  
  
  CG_outputRepr* constructInspectorFunction(IR_Code* ir, std::vector<std::vector<std::string> >  &indices){
  
  CG_outputRepr *to_return;
  
  
  
  return to_return;
  }
  
*/

CG_outputRepr * checkAndGenerateIndirectMappings(CG_outputBuilder * ocg,
                                                 std::vector<std::vector<std::string> > &indices,
                                                 CG_outputRepr * instance, CG_outputRepr * class_def,
                                                 CG_outputRepr * count_var) {
  
  CG_outputRepr *to_return = NULL;
  
  for (int i = 0; i < indices.size(); i++)
    if (indices[i].size() > 1) {
      std::string index = indices[i][indices[i].size() - 1];
      CG_outputRepr *rep = ocg->CreateArrayRefExpression(
        ocg->CreateDotExpression(instance,
                                 ocg->lookup_member_data(class_def, index, instance)),
        count_var);
      for (int j = indices[i].size() - 2; j >= 0; j--)
        rep = ocg->CreateArrayRefExpression(indices[i][j], rep);
      
      CG_outputRepr *lhs = ocg->CreateArrayRefExpression(
        ocg->CreateDotExpression(instance,
                                 ocg->lookup_member_data(class_def, indices[i][0], instance)),
        count_var);
      
      to_return = ocg->StmtListAppend(to_return,
                                      ocg->CreateAssignment(0, lhs, rep));
      
    }
  
  return to_return;
  
}

CG_outputRepr *generatePointerAssignments(CG_outputBuilder *ocg,
                                          std::string prefix_name,
                                          std::vector<std::vector<std::string> > &indices,
                                          CG_outputRepr *instance, 
                                          CG_outputRepr *class_def) {
  
  debug_fprintf(stderr, "generatePointerAssignments()\n");
  CG_outputRepr *list = NULL;

  debug_fprintf(stderr, "prefix '%s',   %d indices\n",  prefix_name.c_str(), indices.size()); 
  for (int i = 0; i < indices.size(); i++) {
    
    std::string s = prefix_name + "_" + indices[i][0];

    debug_fprintf(stderr, "s %s\n", s.c_str()); 
    
    // create a variable definition for a pointer to int with this name
    // that seems to be the only actual result of this routine ... 
    //chillAST_VarDecl *vd = new chillAST_VarDecl( "int",  prefix_name.c_str(), "*", NULL);
    //vd->print(); printf("\n"); fflush(stdout); 
    //vd->dump(); printf("\n"); fflush(stdout); 
    
    CG_outputRepr *ptr_exp = ocg->CreatePointer(s); // but dropped on the floor. unused 
    //debug_fprintf(stderr, "ptr_exp created\n"); 
    
    //CG_outputRepr *rhs = ocg->CreateDotExpression(instance,
    //                                              ocg->lookup_member_data(class_def, indices[i][0], instance));
    
    //CG_outputRepr *ptr_assignment = ocg->CreateAssignment(0, ptr_exp, rhs);
    
    //list = ocg->StmtListAppend(list, ptr_assignment);
    
  }
  
  debug_fprintf(stderr, "generatePointerAssignments() DONE\n\n");
  return list;
}



void Loop::flatten(int stmt_num, std::string index_name ,std::vector<int> &loop_levels,
                   std::string inspector_name) {
  debug_fprintf(stderr, "Loop::flatten( stmt_num %d )\n", stmt_num); 
     
  //  apply_xform();
  
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument(
      
      "invalid statement number " + to_string(stmt_num));
  
  std::sort(loop_levels.begin(), loop_levels.end());
  std::vector<int>::iterator set_it = loop_levels.begin();
  
  int current = *(loop_levels.begin());
  set_it++;
  
  if (current <= 0)
    throw std::invalid_argument("9invalid loop level " + to_string(current));
  if (current > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument(
      "there is no loop level " + to_string(current)
      + " for statement " + to_string(stmt_num));
  
  for (; set_it != loop_levels.end(); set_it++) {
    
    //if (*set_it - current != 1)
    //  throw std::invalid_argument(
    
    //  "loop levels for flattening must be continuous!");
    
    current = *set_it;
    
    if (current <= 0)
      throw std::invalid_argument(
        "10invalid loop level " + to_string(current));
    if (current > stmt[stmt_num].loop_level.size())
      throw std::invalid_argument(
        "there is no loop level " + to_string(current)
        + " for statement " + to_string(stmt_num));
    
  }
  
  if (loop_levels[loop_levels.size() - 1] != stmt[stmt_num].loop_level.size())
    throw std::invalid_argument(
      "11invalid loop level: currrently flattened loop levels must extend to last dimension");
  
  //apply_xform(stmt_num);
  
  //a). Look up array references in stmt[stmt_num].code
  
  std::vector<IR_ArrayRef *> arr_refs = ir->FindArrayRef(stmt[stmt_num].code);
  
  std::vector<IR_ArrayRef *> outer_arr_refs = FindOuterArrayRefs(ir,
                                                                 arr_refs);
  
  std::vector<std::vector<std::string> > indices;
  
  std::vector<std::string> loop_indices;
  
  for (std::vector<int>::iterator i = loop_levels.begin();
       i != loop_levels.end(); i++){
    
    
    
    loop_indices.push_back(stmt[stmt_num].IS.set_var(*i)->name());
  }
  std::vector<IR_ArrayRef *> set_refs;
  
  for (int i = 0; i < outer_arr_refs.size(); i++) {
    bool found = false;
    for (auto j: set_refs) {
      if (j->name() == outer_arr_refs[i]->name()) {
        found = true;
        break;
      }
      
    }
    if (!found)
      set_refs.push_back(outer_arr_refs[i]);
    
  }
  
  indices = constructInspectorVariables(ir, set_refs, loop_indices);
  
  /////////////////////////////////////////////////////////////////////////////
  //Collect Inspector data arrays, types, & sizes to make a typedef of a struct 
  /////////////////////////////////////////////////////////////////////////////
  std::vector<CG_outputRepr *> class_data_types;
  std::vector<std::string> class_data;

  //indices to be passed to inspector function as parameters
  std::vector<std::string> function_param;
  for (int i = 0; i < indices.size(); i++) {
    class_data.push_back(indices[i][0]);
    int j;
    for (j = 0; j < loop_indices.size(); j++)
      if (loop_indices[j] == indices[i][0])
        break;
    if (j != loop_indices.size())
      function_param.push_back(indices[i][0]);
    
    for (j = 0; j < arr_refs.size(); j++) {
      
      std::string comp;
      if (ir->QueryExpOperation(arr_refs[j]->index(0))
          == IR_OP_VARIABLE) {
        std::vector<CG_outputRepr *> v = ir->QueryExpOperand(
          arr_refs[j]->index(0));
        
        IR_ScalarRef *index = static_cast<IR_ScalarRef *>(ir->Repr2Ref(v[0]));
        comp = index->name();
      } else if (ir->QueryExpOperation(arr_refs[j]->index(0))
                 == IR_OP_ARRAY_VARIABLE) {
        std::vector<CG_outputRepr *> v = ir->QueryExpOperand(
          arr_refs[j]->index(0));
        
        IR_ArrayRef *index = static_cast<IR_ArrayRef *>(ir->Repr2Ref(v[0]));
        comp = index->name();
      }
      
      if (comp == indices[i][0]) {
        debug_fprintf(stderr, "making a pointer to int type for %s\n", comp.c_str()); 
        CG_outputRepr *arr_type = ir->CreatePointerType(IR_CONSTANT_INT);
        
        //ir->CreateArrayType(IR_CONSTANT_INT,
        //arr_refs[j]->symbol()->size(0));
        class_data_types.push_back(arr_type); 
        break;
      }
      
    }
    if (j == arr_refs.size())
      throw ir_error("unable to find index for inspector");
    
  }
  
  std::vector<std::string> inspector_member_functions;
  
  inspector_member_functions.push_back("inspector"); // a constructor, if it were actually a class and not a struct

  class_data.push_back("count"); // but no type !! TODO 

  class_data_types.push_back( ir->CreateScalarType(IR_CONSTANT_INT)); // explicitly make count an int
  
//  std::vector<std::vector<std::string> > function_param_in;
  
//  function_param_in.push_back(function_param);
//--Construct Inspector Class + data
  

  ////////////////////////////////////////////////////
  // build the typedef for the inspector struct    - WHAT IS THE SCOPE? global? 
  ////////////////////////////////////////////////////
  CG_outputBuilder *ocg = ir->builder(); 
  CG_outputRepr *class_def = ocg->CreateStruct("inspector", 
                                              class_data,
                                              class_data_types);
  
  // TODO how do you tell what scope to create the class in?
  // so far, no scope, so it does not really exist and is not printed

  //  CG_outputRepr *func_def_constructor = ocg->lookup_member_function(class_def,
  //      "inspector");
  //CG_outputRepr *func_def_method = ocg->lookup_member_function(class_def,
  //    "nnz");
  
  //  CG_outputRepr *constructor_body = ocg->CreateAssignment(0, constructor_data->clone(),
  //      ocg->CreateInt(0));
  //ocg->setFunctionBody(func_def_constructor, constructor_body);
  
  CG_outputRepr *list = NULL;

  // make a declaration of the newly-defined struct  - WHAT IS THE SCOPE? function?  TODO 
  debug_fprintf(stderr, "loop_basic.cc L1910 creating an instance of the inspector named %s\n", inspector_name.c_str()); 
  CG_outputRepr *instance = ocg->CreateClassInstance(inspector_name,   // instance is a VARDECL seems wrong
                                                     class_def);

  CG_outputRepr *constructor_data = ocg->lookup_member_data(class_def,
                                                            "count", instance);
  for (int i = 0; i < function_param.size(); i++) {
    debug_fprintf(stderr, "making assignment for class data %d %s\n", i, class_data[i].c_str());  
    
    // what is this? 
    CG_outputRepr *func_mem_data = ocg->lookup_member_data(class_def,     // scope 
                                                           class_data[i], // variable name 
                                                           instance); 
    //std::vector<CG_outputRepr *> args; // never used !! 
    //args.push_back(ocg->CreateInt(4)); // never used !! 
    
    
    // c.j[c.count] = t4;
    // c.i[c.count] = t2;
    debug_fprintf(stderr, "\nloop_basic.cc L1958  calling CreateDotExpression with CLONE  (c.j)\n"); 
    // clone is because CreateDotetc DELETES the args (why?) 
    CG_outputRepr *array = ocg->CreateDotExpression(instance, // ->clone(),                   //clone is WRONG?
                                                    func_mem_data ); // ->clone()); // c.j
    debug_fprintf(stderr, "\nloop_basic.cc L1962  calling CreateDotExpression with CLONE  (c.count)\n"); 
    CG_outputRepr *index = ocg->CreateDotExpression(instance, // ->clone(), // c.count
                                                    ocg->lookup_member_data(class_def, "count", instance));
    
    debug_fprintf(stderr, "\nloop_basic.cc L1967  calling CreateAssignment()\n"); 
    CG_outputRepr *func_mem_body = ocg->CreateAssignment(0,
                                                         ocg->CreateArrayRefExpression(array, 
                                                                                       index), 
                                                         ocg->CreateIdent(function_param[i]));
    
   
    list = ocg->StmtListAppend(list, func_mem_body);
    
  }
  
  //ocg->setFunctionBody(func_def_method, list);
  
  //CG_outputRepr *count = ocg->CreateIdent("count");
  //CG_outputRepr *count_init = ocg->CreateAssignment(0, count->clone(),
  //    ocg->CreateInt(0));
  
  //init_code = ocg->StmtListAppend(init_code, count_init);


  CG_outputRepr *count_plusplus = ocg->CreateAssignment(0,
                                                        ocg->CreateDotExpression(instance->clone(),
                                                                                 constructor_data->clone()),
                                                        ocg->CreatePlus(
                                                                        ocg->CreateDotExpression(instance->clone(),
                                                                                                 constructor_data->clone()), ocg->CreateInt(1)));

  //  CG_outputRepr *stmts_to_append = checkAndGenerateIndirectMappings(ocg,
  //      indices, instance, class_def, count->clone());

  debug_fprintf(stderr, "CALLING unused code generatePointerAssignments();\n"); 
  // unused  ?? but seems to have needed side effects   
  CG_outputRepr *initializations = generatePointerAssignments(ocg, // never used !! TODO 
                                                              inspector_name, indices, instance, class_def);
  


  /*
   * Generate
   * Substitution Mappings here
   *
   */
  
  debug_fprintf(stderr, "*** BLAAT\n");
  init_code = ocg->StmtListAppend(init_code,
                                  ocg->CreateAssignment(0,
                                                        ocg->CreateDotExpression(instance->clone(),
                                                                                 constructor_data->clone()), ocg->CreateInt(0)));
  
//--end Construct Inspector Class + data
  
//Attach function as data member to class;
  
//b). Generate mappings and store mappings in global data structure
//    eg. y[i] += a[j]*x[col[j]]
//    y: i-> i  a: j->j x:j-> col[j]
  
//c). Generate the Inspector code corresponding to the mapping
//    using NNZ.nnz(t2,t4);
  
//d) generate pointers for each of the array references
//Eg.  float *nnz_i = NNZ.i;
  
//e) Replace the expressions within col[j]
//eg.  j->col[nnz.j] and  you have x[col[nnz.j]] for x then replace with nnz.x
  
//f) replace nnz(i,j) with *count;
  
//1. Create Omega omega::Relation for Inspector Iteration Space
//   using uninterpreted function symbols
  
//1.a) Create Mapping from i->[i,k,j]
  
  int n = stmt[stmt_num].IS.n_set();
  /*
  //1.b) Modify xform
  
  int m = stmt[stmt_num].xform.n_out();
  
  omega::Relation r(m, m + 2);
  F_And *f_root = r.add_and();
  
  for (int j = 1; j <= m; j++) {
  EQ_Handle e = f_root->add_EQ();
  e.update_coef(r.input_var(j), 1);
  e.update_coef(r.output_var(j), -1);
  }
  
  for (int j = m + 2; j <= m + 2; j += 2) {
  EQ_Handle e = f_root->add_EQ();
  e.update_coef(r.output_var(j), 1);
  
  }
  
  // a = NNZ(i,j)
  Free_Var_Decl *nnz = new Free_Var_Decl(inspector_name, loop_levels.size());
  
  Variable_ID nnz_in = r.get_local(nnz, Input_Tuple);
  EQ_Handle e2 = f_root->add_EQ();
  e2.update_coef(r.output_var(m + 1), 1);
  e2.update_coef(nnz_in, -1);
  
  */
  
//2. Create Omega omega::Relation for Flattened Iteration Space
//   using uninterpreted function symbolsstd::vector<IR_ArrayRef *> arr_refs = ir->FindArrayRef(stmt[stmt_num].code);
  
  
// omega::Relation flattened_IS = copy(stmt[stmt_num].IS);
  //IeGenlib relation construction
  //1. Create a map from array reference names to outputReprs to pass into CreateSubstituted statement
  std::map<std::string, std::string> array_name_to_relation;
  
  for (auto it: set_refs) {
    std::string s;
    if (it->n_dim() > 1)
      throw loop_error(
        "only statements with single dimensional arrays are handled for flattening currently");
    
    CG_outputRepr *index = it->index(0);
    if (ir->QueryExpOperation(index) != IR_OP_ARRAY_VARIABLE) {
      omega::Relation r(stmt[stmt_num].IS.n_set(), 1);
      //std::cout << (*it)->name() << std::endl;
      F_And *f_root = r.add_and();
      
      for (int j = 1; j <= stmt[stmt_num].IS.n_set(); j++)          // all the iteration space indeces enclosing this
        r.name_input_var(j, stmt[stmt_num].IS.set_var(j)->name());
      Variable_ID v = r.output_var(1);
      exp2formula(
        this, ir, r, f_root, freevar, index, v, 'w', IR_COND_EQ,
                  true,uninterpreted_symbols[stmt_num],uninterpreted_symbols_stringrepr[stmt_num], unin_rel[stmt_num]);
      r.setup_names();
      r.simplify(2, 4);
      s = r.print_with_subs_to_string();
      //std::cout << s << std::endl;
      
    } else {
      
      std::vector<IR_ArrayRef *> inner_refs = ir->FindArrayRef(index);
      
      CG_outputRepr *index2 = inner_refs[0]->index(0);
      if (inner_refs.size() > 1)
        throw loop_error(
          "more than one level of indirection in arrays not supported currently");
      
      std::string function_name = inner_refs[0]->name();
      
      if (ir->QueryExpOperation(index2) == IR_OP_ARRAY_VARIABLE)
        throw loop_error(
          "more than one level of indirection in arrays not supported currently");
      
      omega::Relation r(stmt[stmt_num].IS.n_set(), 1);
      std::cout << it->name() << std::endl;
      F_And *f_root = r.add_and();
      
      for (int j = 1; j <= stmt[stmt_num].IS.n_set(); j++)
        r.name_input_var(j, stmt[stmt_num].IS.set_var(j)->name());
      Variable_ID v = r.output_var(1);
      exp2formula(this, ir, r, f_root, freevar, index2, v, 'w', IR_COND_EQ,
                  true,uninterpreted_symbols[stmt_num],uninterpreted_symbols_stringrepr[stmt_num], unin_rel[stmt_num]);
      r.setup_names();
      r.simplify(2, 4);
      std::string output_var = r.print_outputs_with_subs_to_string();
      
      s = "{[";
      s += stmt[stmt_num].IS.set_var(1)->name();
      for (int j = 2; j <= stmt[stmt_num].IS.n_set(); j++)
        s = s + "," + stmt[stmt_num].IS.set_var(j)->name();
      
      s += "]->[r2] : r2 = " + function_name + "(" + output_var + ")}";
      
    }
    array_name_to_relation.insert(
      std::pair<std::string, std::string>(it->name(), s));
    std::cout << "check" << std::endl;
    std::cout << s << std::endl;
  }
  
  // from CHILL, Anand declares the uninterpreted function
  // that performs coalescing and defines the coalescing
  // transformation.
  iegenlib::setCurrEnv(); // Clears out the environment
  
  omega::Relation IS_for_iegen = copy(stmt[stmt_num].IS);
  printf("IS_for_iegen   "); IS_for_iegen.print(); printf("\n"); fflush(stdout); 

  std::string set_string = "{[";
  for (int j = 1; j <= IS_for_iegen.n_set(); j++) {
    if (j != 1)
      set_string += ",";
    set_string += stmt[stmt_num].IS.set_var(j)->name();
  }
  
  set_string += "] ";
  std::string iegen_is = print_to_iegen_string(IS_for_iegen);
  iegen_is = set_string + " : " + iegen_is + "}";
  std::cout << iegen_is << std::endl;
  
  omega::Relation flattened_index(1);
  flattened_index.name_set_var(1, index_name);
  F_And *f_root_iegen = flattened_index.add_and();
  
  GEQ_Handle iegen1 = f_root_iegen->add_GEQ();
  iegen1.update_coef(flattened_index.set_var(1), 1);
  
  GEQ_Handle iegen2 = f_root_iegen->add_GEQ();
  iegen2.update_coef(flattened_index.set_var(1), -1);
  Free_Var_Decl *t = new Free_Var_Decl("nnz");
  Variable_ID e = flattened_index.get_local(t);
  iegen2.update_coef(e, 1);
  iegen2.update_const(-1);
  flattened_index.setup_names();
  flattened_index.simplify();
  printf("\n*** flattened_index   ");   flattened_index.print(); printf("\n"); fflush(stdout); 

 
  std::string set_string2 = "{[" + flattened_index.set_var(1)->name()
    + "] : ";
  
  std::string iegen_flattened = print_to_iegen_string(flattened_index);
  iegen_flattened = set_string2 + iegen_flattened + "}";
  std::cout << iegen_flattened << std::endl;
  
  omega::Relation r1(stmt[stmt_num].IS.n_set(), 1);
  
  std::vector<std::string> index_names;
  for (int j = 1; j <= stmt[stmt_num].IS.n_set(); j++) {
    r1.name_input_var(j, stmt[stmt_num].IS.set_var(j)->name());
    index_names.push_back(stmt[stmt_num].IS.set_var(j)->name());
  }
  r1.name_output_var(1, index_name);

  Free_Var_Decl *nnz_iegen = new Free_Var_Decl(inspector_name,
                                               loop_levels.size());
  Variable_ID iegen_nnz = r1.get_local(nnz_iegen, Input_Tuple);
  
  F_And *f_root_coalesce = r1.add_and();
  
  EQ_Handle e_iegen1 = f_root_coalesce->add_EQ();
  e_iegen1.update_coef(iegen_nnz, 1);
  e_iegen1.update_coef(r1.output_var(1), -1);
  
  printf("\n*** r1   ");   r1.print(); printf("\n"); fflush(stdout);

  std::string r1_coalesce = print_to_iegen_string(r1);
  std::string set_string3 = set_string + " -> [" + r1.output_var(1)->name()
    + "] : ";
  r1_coalesce = set_string3 + r1_coalesce + "}";
  std::cout << r1_coalesce << std::endl;
  
  debug_fprintf(stderr, "%s\n%s\n%s\n", inspector_name.c_str(), iegen_is.c_str(), iegen_flattened.c_str());

  debug_fprintf(stderr, "\ndomain\n"); 
  iegenlib::Set *domain   = new iegenlib::Set(iegen_is);
  debug_fprintf(stderr, "\nrange\n"); 
  iegenlib::Set *range = new iegenlib::Set(iegen_flattened); 


  debug_fprintf(stderr, "Loop::flatten(), line 2202, calling iegenlib::appendCurrEnv()\n"); 
  iegenlib::appendCurrEnv(inspector_name, // UF name
                          // UF domain
                          domain, // new iegenlib::Set(iegen_is),
                          // UF range
                          range, // new iegenlib::Set(iegen_flattened),
                          // c function is bijective
                          true,
                          iegenlib::Monotonic_NONE);
  debug_fprintf(stderr, "FINISHED iegenlib::appendCurrEnv()\n\n"); 


  iegenlib::Relation* T_coalesce = new iegenlib::Relation(r1_coalesce);
  std::cout<< T_coalesce->prettyPrintString();
  debug_fprintf(stderr, "T_coalesce\n"); 

  // then applies the transformation to each of the access relation
  iegenlib::Relation* T_coalesce_inv = T_coalesce->Inverse();
  //EXPECT_EQ("{ [k] -> [i, j] : k - j = 0 && k - c(i, j) = 0 }",
  std::cout << T_coalesce_inv->prettyPrintString() << std::endl;
  
  omega::Relation flattened_IS(stmt[stmt_num].IS.n_set() + 1);
  F_And *f_and = flattened_IS.add_and();
  
  for (int j = 1; j <= flattened_IS.n_set() - 1; j++) {
    EQ_Handle e = f_and->add_EQ();
    e.update_coef(flattened_IS.set_var(j), 1);
    
  }
  
  flattened_IS.copy_names(stmt[stmt_num].IS);
  //flattened_IS.setup_names();
  flattened_IS.name_set_var(flattened_IS.n_set(), index_name);
  
  Free_Var_Decl *nnz = new Free_Var_Decl(inspector_name, loop_levels.size());
  Variable_ID nnz_in_ = flattened_IS.get_local(nnz, Input_Tuple);
  
  GEQ_Handle g3_ = f_and->add_GEQ();
  
  g3_.update_coef(nnz_in_, 1);
  g3_.update_coef(flattened_IS.set_var(flattened_IS.n_set()), -1);
  g3_.update_const(-1);
  
  GEQ_Handle g4_ = f_and->add_GEQ();
  g4_.update_coef(flattened_IS.set_var(flattened_IS.n_set()), 1);
  
  flattened_IS.simplify();
  printf("\n*** flattened_IS   ");   flattened_IS.print(); printf("\n"); fflush(stdout); 


  omega::Relation flattened_xform(flattened_IS.n_set(),
                           2 * flattened_IS.n_set() + 1);
  
  F_And *fr = flattened_xform.add_and();
  
  for (int j = 1; j <= flattened_IS.n_set(); j++) {
    EQ_Handle e = fr->add_EQ();
    
    if (j != flattened_IS.n_set()) {
      e.update_coef(flattened_xform.output_var(2 * j), 1);
      e.update_const(-1);
    } else {
      e.update_coef(flattened_xform.input_var(j), -1);
      e.update_coef(flattened_xform.output_var(2 * j), 1);
      
    }
    
  }
  
  for (int j = 1; j <= 2 * (flattened_IS.n_set()) + 1; j += 2) {
    EQ_Handle e = fr->add_EQ();
    
    e.update_coef(flattened_xform.output_var(j), 1);
    e.update_const(-1);
    
  }
  flattened_IS.setup_names();
  copy(flattened_IS).print();
  copy(flattened_xform).print();
  
  flattened_IS.simplify();
  flattened_xform.simplify();
  printf("\n*** flattened_xform   ");   flattened_xform.print(); printf("\n"); fflush(stdout); 


  /*  stmt[stmt_num].xform = Composition(r, stmt[stmt_num].xform);
      stmt[stmt_num].xform.simplify();
      stmt[stmt_num].IS.simplify();
  */
  Statement new_stmt;
  new_stmt.IS = flattened_IS;
  new_stmt.xform = flattened_xform;
  new_stmt.loop_level = stmt[stmt_num].loop_level;
  new_stmt.reduction = stmt[stmt_num].reduction;
  new_stmt.reductionOp = stmt[stmt_num].reductionOp;
  LoopLevel ll;
  ll.type = LoopLevelOriginal;
  ll.payload = n + 1;
  ll.parallel_level = 0;
  if (new_stmt.reduction > 0) {
    ll.segreducible = true;
    ll.segment_descriptor = inspector_name + "."
      + new_stmt.IS.set_var(1)->name();
  } else
    ll.segreducible = false;
  
  new_stmt.loop_level.push_back(ll);
  
//Anand: extra redundant loop_level
  /*  LoopLevel ll2;
      ll2.type = LoopLevelOriginal;
      ll2.payload = n + 2;
      ll2.parallel_level = 0;
      new_stmt.loop_level.push_back(ll2);
  */
  new_stmt.ir_stmt_node = NULL;
  new_stmt.has_inspector = false;
  new_stmt.code = stmt[stmt_num].code->clone();
  
  delete stmt[stmt_num].code;
  stmt[stmt_num].code = ocg->StmtListAppend(list, count_plusplus);
  stmt[stmt_num].has_inspector = true;
  //stmt[stmt_num].code = ocg->CreateNullStatement();
  ocg->CreateIdent(index_name);

  // create  #define c(i,j) c.count
  debug_fprintf(stderr, "*** create  #define c(i,j) (*c_count)\n");

  std::vector< std::string> args;
  args.push_back( std::string("i") );
  args.push_back( std::string("j") );

  debug_fprintf(stderr, "\nloop_basic.cc in loop_flatten(), calling CreateDefineMacro( %s, i,j, count)\n", inspector_name.c_str()); 
  ir->CreateDefineMacro(inspector_name, 
                        args, // "(i,j)", // (i,j) hardcoded ???  TODO 
                        ocg->ObtainInspectorRange(inspector_name, "count"));  // ?? 
  debug_fprintf(stderr, "*** create  #define c(i,j) (*c_count) DONE\n");
  
  // add macro to sourceFile  ( should be in CreateDefineMacro ?? ) 
  //IR_rosecode *irrose = (IR_rosecode *)ir_; 
  //ChillAST_SourceFile *sf = irrose->entire_file_AST;
  //sf->addMacro( 
  
  debug_fprintf(stderr, "loop_basic.cc L2353 adding stmt %d\n", stmt.size()); 
  stmt.push_back(new_stmt);
  
  uninterpreted_symbols.push_back(uninterpreted_symbols[stmt_num]);
  uninterpreted_symbols_stringrepr.push_back(uninterpreted_symbols_stringrepr[stmt_num]);
  int new_stmt_num = stmt.size() - 1;
  //Anand adding dependence graph update :
  
  dep.insert();
  
  DependenceVector dv;
  dv.type = DEP_W2R;
  dv.lbounds.push_back(0);
  dv.ubounds.push_back(posInfinity);
  
  for (int i = 1; i < loop_levels.size(); i++) {
    
    dv.lbounds.push_back(0);
    dv.ubounds.push_back(0);
    
  }
  
  //Connect original statement and flattened statement with unknown dependence
  dep.connect(stmt_num, stmt.size() - 1, dv);
// Update known relation
  int known_dim = (this->known).n_set();
  
  omega::Relation known = Extend_Set(copy(this->known),
                              stmt[stmt_num].xform.n_out() - this->known.n_set());
  
  Variable_ID nnz_in__ = known.get_local(nnz, Input_Tuple);
  F_And *f_and_ = known.and_with_and();
  GEQ_Handle g5_ = f_and_->add_GEQ();
  g5_.update_coef(nnz_in__, 1);
  g5_.update_const(-4096);
  //this->known = known;
  
  int n_ = stmt[new_stmt_num].loop_level.size();
  
  std::vector<int> lex = getLexicalOrder(new_stmt_num);
  
  omega::Relation mapping(2 * n_ + 1, n_);
  F_And *f_root = mapping.add_and();
  for (int j = 1; j <= n_; j++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.output_var(j), 1);
    h.update_coef(mapping.input_var(2 * j), -1);
    
  }
  mapping = Composition(mapping, stmt[new_stmt_num].xform);
  
  F_And *f_and__ = mapping.and_with_and();
  
  GEQ_Handle g6_ = f_and__->add_GEQ();
  g6_.update_coef(mapping.input_var(n_), -1);
  
  Variable_ID nnz_in___ = mapping.get_local(nnz, Input_Tuple);
  g6_.update_coef(nnz_in___, 1);
  
  mapping.simplify();
  printf("\n*** mapping1   ");   mapping.print(); printf("\n"); fflush(stdout); 

  
  // match omega input/output variables to variable names in the code
  for (int j = 1; j <= stmt[new_stmt_num].IS.n_set(); j++)
    mapping.name_input_var(j, stmt[new_stmt_num].IS.set_var(j)->name());
  for (int j = 1; j <= n_; j++)
    mapping.name_output_var(j,
                            tmp_loop_var_name_prefix
                            + to_string(tmp_loop_var_name_counter + j - 1));
  
  mapping.setup_names();
  printf("\n*** mapping2   ");   mapping.print(); printf("\n"); fflush(stdout); 

  //stmt[*i].code = outputStatement(ocg, stmt[*i].code, 0, mapping, known, std::vector<CG_outputRepr *>(mapping.n_out(), NULL));
  std::vector<std::string> loop_vars;
  for (int j = 1; j <= stmt[new_stmt_num].IS.n_set(); j++)
    loop_vars.push_back(stmt[new_stmt_num].IS.set_var(j)->name());
  
/*  std::vector<CG_outputRepr *> subs = output_substitutions(ocg,
    Inverse(copy(mapping)),
    std::vector<std::pair<CG_outputRepr *, int> >(mapping.n_out(),
    std::make_pair(static_cast<CG_outputRepr *>(NULL), 0)),
    uninterpreted_symbols);
    stmt[new_stmt_num].code = ocg->CreateSubstitutedStmt(0,
    stmt[new_stmt_num].code, loop_vars, subs);
    
*/
  
  
  
  //2. For each array reference get the index and initialize the string relation
  
  std::vector<IR_ArrayRef *> arr_refs_ = ir->FindArrayRef(stmt[new_stmt_num].code);
  
  std::vector<IR_ArrayRef *> outer_arr_refs_ = FindOuterArrayRefs(ir,
                                                                  arr_refs_);
  
  
  std::vector<IR_ArrayRef *> set_refs_;
  
  for (int i = 0; i < outer_arr_refs_.size(); i++) {
    bool found = false;
    for (auto j: set_refs_)
      if (j->name() == outer_arr_refs_[i]->name()) {
        found = true;
        break;
      }
    if (!found)
      set_refs_.push_back(outer_arr_refs_[i]);
    
  }
  
  
  for (std::map<std::string, std::string>::iterator it =
         array_name_to_relation.begin(); it != array_name_to_relation.end();
       it++) {
    
    std::string array_name = it->first;
    std::string array_exp = it->second;
    std::cout<<array_exp<<std::endl;
    iegenlib::Relation* r0_v0 = new iegenlib::Relation(array_exp);
    
    //3. double indirection introduce the uninterpreted function
    
    //4. Create the coalesce relation by reading the IS and the  flattening relation levels names
    // Also do the computation for updating the access mappings
    
    iegenlib::Relation* r0_v1 = r0_v0->Compose(T_coalesce_inv);
    
    ////// To get the updated access relation do the following:
    // Find tuple variable 1 as a function of tuple variables 0 through 0
    iegenlib::Exp* r0_v1_exp = r0_v1->findFunction(1, 0, 0);
    
    std::string s = r0_v1_exp->prettyPrintString(r0_v1->getTupleDecl());
    debug_fprintf(stderr, "*** s %s\n", s.c_str()); 

    CG_outputRepr * subscript = iegen_parser(s, index_names);
    
    //6. Pass in  the mappings to CreateSubstitutedStmt
    CG_outputRepr *new_array_exp = ir->builder()->CreateArrayRefExpression( ir->builder()->CreateIdent(array_name), subscript);
    
    IR_ArrayRef *to_replace;
    auto it2 = set_refs_.begin();
    for(; it2 != set_refs_.end(); ++it2)
      if((*it2)->name() == array_name) {
        to_replace = *it2;
        break;
      }
    
    set_refs_.erase(it2);
    ir->ReplaceExpression(to_replace, new_array_exp);
    
    
    delete r0_v0;
    delete r0_v1;
    delete r0_v1_exp;
    
  }
  
  std::vector<CG_outputRepr *> subs = output_substitutions(ocg,
                                                           Inverse(copy(mapping)),
                                                           std::vector<std::pair<CG_outputRepr *, int> >(mapping.n_out(),
                                                                                                         std::make_pair(static_cast<CG_outputRepr *>(NULL), 0)),
                                                           uninterpreted_symbols[stmt_num]);
  stmt[new_stmt_num].code = ocg->CreateSubstitutedStmt(0,
                                                       stmt[new_stmt_num].code, loop_vars, subs, true);
  

  debug_fprintf(stderr, "BEFORE changing stmt[%d].IS\n", new_stmt_num); 
  //debugomega::Relations();

  omega::Relation buh = Restrict_Domain(mapping, stmt[new_stmt_num].IS);
  debug_fprintf(stderr, "*** buh: \n");   buh.print(); printf("\n"); fflush(stdout); 
  stmt[new_stmt_num].IS = omega::Range( buh ); 
  debug_fprintf(stderr, "*** buh: \n");   buh.print(); printf("\n"); fflush(stdout); 
 
  printf("\n*** mapping3   ");   mapping.print(); printf("\n"); fflush(stdout); 

  // above breaks the following into parts 
  //stmt[new_stmt_num].IS = omega::Range(Restrict_Domain(mapping, stmt[new_stmt_num].IS));

  debug_fprintf(stderr, "AFTER changing stmt[%d].IS\n", new_stmt_num); 
  //debugomega::Relations();

  stmt[new_stmt_num].IS.simplify();
  
  debug_fprintf(stderr, "AFTER simplify\n"); 
  //debugomega::Relations();

  // replace original transformation relation with straight 1-1 mapping
  mapping = omega::Relation(n_, 2 * n_ + 1);
  f_root = mapping.add_and();
  for (int j = 1; j <= n_; j++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.output_var(2 * j), 1);
    h.update_coef(mapping.input_var(j), -1);
  }
  for (int j = 1; j <= 2 * n_ + 1; j += 2) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.output_var(j), 1);
    h.update_const(-lex[j - 1]);
  }
  stmt[new_stmt_num].xform = mapping;
  tmp_loop_var_name_counter += n_;
  this->known = known;


  printf("\n*** mapping   ");   mapping.print(); printf("\n"); fflush(stdout); 
  printf("\n*** known     ");   known.print(); printf("\n"); fflush(stdout); 


  delete T_coalesce;
  delete T_coalesce_inv;
  
  debug_fprintf(stderr, "Loop::flatten() END\n"); 
}










void Loop::normalize(int stmt_num, int loop_level) {
  
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument(
      
      "invalid statement number " + to_string(stmt_num));
  
  if (loop_level <= 0)
    throw std::invalid_argument(
      "12invalid loop level " + to_string(loop_level));
  if (loop_level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument(
      "there is no loop level " + to_string(loop_level)
      + " for statement " + to_string(stmt_num));
  
  apply_xform(stmt_num);
  
  omega::Relation r = copy(stmt[stmt_num].IS);
  
  omega::Relation bound = get_loop_bound(r, loop_level, this->known);
  if (!bound.has_single_conjunct() || !bound.is_satisfiable()
      || bound.is_tautology())
    throw loop_error("unable to extract loop bound for normalize");
  
  // extract the loop stride
  coef_t stride;
  std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(bound,
                                                                  bound.set_var(loop_level));
  if (result.second == NULL)
    stride = 1;
  else
    stride = abs(result.first.get_coef(result.second))
      / gcd(abs(result.first.get_coef(result.second)),
            abs(result.first.get_coef(bound.set_var(loop_level))));
  
  if (stride != 1)
    throw loop_error(
      "normalize currently only handles unit stride, non unit stride present in loop bounds");
  
  GEQ_Handle lb;
  
  Conjunct *c = bound.query_DNF()->single_conjunct();
  for (GEQ_Iterator gi(c->GEQs()); gi; gi++) {
    int coef = (*gi).get_coef(bound.set_var(loop_level));
    if (coef > 0)
      lb = *gi;
  }
  
  //Loop bound already zero
  //Nothing to do.
  if (lb.is_const(bound.set_var(loop_level)) && lb.get_const() == 0)
    return;
  
  if (lb.is_const_except_for_global(bound.set_var(loop_level))) {
    
    int n = stmt[stmt_num].xform.n_out();
    
    omega::Relation r(n, n);
    F_And *f_root = r.add_and();
    for (int j = 1; j <= n; j++)
      if (j != 2 * loop_level) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(r.input_var(j), 1);
        h.update_coef(r.output_var(j), -1);
      }
    
    stmt[stmt_num].xform = Composition(r, stmt[stmt_num].xform);
    stmt[stmt_num].xform.simplify();
    
    for (Constr_Vars_Iter ci(lb); ci; ci++) {
      if ((*ci).var->kind() == Global_Var) {
        Global_Var_ID g = (*ci).var->get_global_var();
        Variable_ID v;
        if (g->arity() == 0)
          v = stmt[stmt_num].xform.get_local(g);
        else
          v = stmt[stmt_num].xform.get_local(g,
                                             (*ci).var->function_of());
        
        F_And *f_super_root = stmt[stmt_num].xform.and_with_and();
        F_Exists *f_exists = f_super_root->add_exists();
        F_And *f_root = f_exists->add_and();
        
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(stmt[stmt_num].xform.output_var(2 * loop_level),
                      1);
        h.update_coef(stmt[stmt_num].xform.input_var(loop_level), -1);
        h.update_coef(v, 1);
        
        stmt[stmt_num].xform.simplify();
      }
      
    }
    
  } else
    throw loop_error("loop bounds too complex for normalize!");
  
}

