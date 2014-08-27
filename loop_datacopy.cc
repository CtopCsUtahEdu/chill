/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   Various data copy schemes.

 Notes:

 History:
   02/20/09 Created by Chun Chen by splitting original datacopy from loop.cc
*****************************************************************************/

#include <codegen.h>
#include <code_gen/CG_utils.h>
#include "loop.hh"
#include "omegatools.hh"
#include "ir_code.hh"
#include "chill_error.hh"

using namespace omega;

//
// data copy function by referring arrays by numbers.
// e.g. A[i] = A[i-1] + B[i]
//      parameter array_ref_num=[0,2] means to copy data touched by A[i-1] and A[i]
//
bool Loop::datacopy(const std::vector<std::pair<int, std::vector<int> > > &array_ref_nums, int level,
                    bool allow_extra_read, int fastest_changing_dimension, int padding_stride, int padding_alignment, int memory_type) {
  // check for sanity of parameters
  std::set<int> same_loop;
  for (int i = 0; i < array_ref_nums.size(); i++) {
    int stmt_num = array_ref_nums[i].first;
    if (stmt_num < 0 || stmt_num >= stmt.size())
      throw std::invalid_argument("invalid statement number " + to_string(stmt_num));
    if (level <= 0 || level > stmt[stmt_num].loop_level.size())
      throw std::invalid_argument("invalid loop level " + to_string(level));
    if (i == 0) {
      std::vector<int> lex = getLexicalOrder(stmt_num);
      same_loop = getStatements(lex, 2*level-2);
    }
    else if (same_loop.find(stmt_num) == same_loop.end())
      throw std::invalid_argument("array references for data copy must be located in the same subloop");
  }
  
  // convert array reference numbering scheme to actual array references
  std::vector<std::pair<int, std::vector<IR_ArrayRef *> > > selected_refs;
  for (int i = 0; i < array_ref_nums.size(); i++) {
    if (array_ref_nums[i].second.size() == 0)
      continue;
    
    int stmt_num = array_ref_nums[i].first;
    selected_refs.push_back(std::make_pair(stmt_num, std::vector<IR_ArrayRef *>()));
    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[stmt_num].code);
    std::vector<bool> selected(refs.size(), false);
    for (int j = 0; j < array_ref_nums[i].second.size(); j++) {
      int ref_num = array_ref_nums[i].second[j];
      if (ref_num < 0 || ref_num >= refs.size()) {
        for (int k = 0; k < refs.size(); k++)
          delete refs[k];
        throw std::invalid_argument("invalid array reference number " + to_string(ref_num) + " in statement " + to_string(stmt_num));
      }
      selected_refs[selected_refs.size()-1].second.push_back(refs[ref_num]);
      selected[ref_num] = true;
    }
    for (int j = 0; j < refs.size(); j++)
      if (!selected[j])
        delete refs[j];
  }
  if (selected_refs.size() == 0)
    throw std::invalid_argument("found no array references to copy");
  
  // do the copy
  return datacopy_privatized(selected_refs, level, std::vector<int>(), allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, memory_type);
}

//
// data copy function by referring arrays by name.
// e.g. A[i] = A[i-1] + B[i]
//      parameter array_name=A means to copy data touched by A[i-1] and A[i]
//
bool Loop::datacopy(int stmt_num, int level, const std::string &array_name,
                    bool allow_extra_read, int fastest_changing_dimension, int padding_stride, int padding_alignment, int memory_type) {
  // check for sanity of parameters
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invalid statement number " + to_string(stmt_num));
  if (level <= 0 || level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("invalid loop level " + to_string(level));
  
  // collect array references by name
  std::vector<int> lex = getLexicalOrder(stmt_num);
  int dim = 2*level - 1;
  std::set<int> same_loop = getStatements(lex, dim-1);
  
  std::vector<std::pair<int, std::vector<IR_ArrayRef *> > > selected_refs;
  for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
    std::vector<IR_ArrayRef *> t;
    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[*i].code);  
    for (int j = 0; j < refs.size(); j++)
      if (refs[j]->name() == array_name)
        t.push_back(refs[j]);
      else
        delete refs[j];
    if (t.size() != 0)
      selected_refs.push_back(std::make_pair(*i, t)); 
  }
  if (selected_refs.size() == 0)
    throw std::invalid_argument("found no array references with name " + to_string(array_name) + " to copy");
  
  // do the copy
  return datacopy_privatized(selected_refs, level, std::vector<int>(), allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, memory_type);
}


bool Loop::datacopy_privatized(int stmt_num, int level, const std::string &array_name, const std::vector<int> &privatized_levels,
                               bool allow_extra_read, int fastest_changing_dimension, int padding_stride, int padding_alignment, int memory_type) {
  // check for sanity of parameters
  if (stmt_num < 0 || stmt_num >= stmt.size())
    throw std::invalid_argument("invalid statement number " + to_string(stmt_num));
  if (level <= 0 || level > stmt[stmt_num].loop_level.size())
    throw std::invalid_argument("invalid loop level " + to_string(level));
  
  // collect array references by name
  std::vector<int> lex = getLexicalOrder(stmt_num);
  int dim = 2*level - 1;
  std::set<int> same_loop = getStatements(lex, dim-1);
  
  std::vector<std::pair<int, std::vector<IR_ArrayRef *> > > selected_refs;
  for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
    selected_refs.push_back(std::make_pair(*i, std::vector<IR_ArrayRef *>()));
    
    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[*i].code);  
    for (int j = 0; j < refs.size(); j++)
      if (refs[j]->name() == array_name)
        selected_refs[selected_refs.size()-1].second.push_back(refs[j]);
      else
        delete refs[j];
  }
  if (selected_refs.size() == 0)
    throw std::invalid_argument("found no array references with name " + to_string(array_name) + " to copy");
  
  // do the copy
  return datacopy_privatized(selected_refs, level, privatized_levels, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, memory_type);
}


bool Loop::datacopy_privatized(const std::vector<std::pair<int, std::vector<int> > > &array_ref_nums, int level, const std::vector<int> &privatized_levels, bool allow_extra_read, int fastest_changing_dimension, int padding_stride, int padding_alignment, int memory_type) {
  // check for sanity of parameters
  std::set<int> same_loop;
  for (int i = 0; i < array_ref_nums.size(); i++) {
    int stmt_num = array_ref_nums[i].first;
    if (stmt_num < 0 || stmt_num >= stmt.size())
      throw std::invalid_argument("invalid statement number " + to_string(stmt_num));
    if (level <= 0 || level > stmt[stmt_num].loop_level.size())
      throw std::invalid_argument("invalid loop level " + to_string(level));
    if (i == 0) {
      std::vector<int> lex = getLexicalOrder(stmt_num);
      same_loop = getStatements(lex, 2*level-2);
    }
    else if (same_loop.find(stmt_num) == same_loop.end())
      throw std::invalid_argument("array references for data copy must be located in the same subloop");
  }
  
  // convert array reference numbering scheme to actual array references
  std::vector<std::pair<int, std::vector<IR_ArrayRef *> > > selected_refs;
  for (int i = 0; i < array_ref_nums.size(); i++) {
    if (array_ref_nums[i].second.size() == 0)
      continue;
    
    int stmt_num = array_ref_nums[i].first;
    selected_refs.push_back(std::make_pair(stmt_num, std::vector<IR_ArrayRef *>()));
    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[stmt_num].code);
    std::vector<bool> selected(refs.size(), false);
    for (int j = 0; j < array_ref_nums[i].second.size(); j++) {
      int ref_num = array_ref_nums[i].second[j];
      if (ref_num < 0 || ref_num >= refs.size()) {
        for (int k = 0; k < refs.size(); k++)
          delete refs[k];
        throw std::invalid_argument("invalid array reference number " + to_string(ref_num) + " in statement " + to_string(stmt_num));
      }
      selected_refs[selected_refs.size()-1].second.push_back(refs[ref_num]);
      selected[ref_num] = true;
    }
    for (int j = 0; j < refs.size(); j++)
      if (!selected[j])
        delete refs[j];
  }
  if (selected_refs.size() == 0)
    throw std::invalid_argument("found no array references to copy");
  
  // do the copy
  return datacopy_privatized(selected_refs, level, privatized_levels, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, memory_type);
}


//
// Implement low level datacopy function with lots of options.
//
/*bool Loop::datacopy_privatized(const std::vector<std::pair<int, std::vector<IR_ArrayRef *> > > &stmt_refs, int level,
  const std::vector<int> &privatized_levels,
  bool allow_extra_read, int fastest_changing_dimension,
  int padding_stride, int padding_alignment, int memory_type) {
  if (stmt_refs.size() == 0)
  return true;
  
  // check for sanity of parameters
  IR_ArraySymbol *sym = NULL;
  std::vector<int> lex;
  std::set<int> active;
  if (level <= 0)
  throw std::invalid_argument("invalid loop level " + to_string(level));
  for (int i = 0; i < privatized_levels.size(); i++) {
  if (i == 0) {
  if (privatized_levels[i] < level)
  throw std::invalid_argument("privatized loop levels must be no less than level " + to_string(level));
  }
  else if (privatized_levels[i] <= privatized_levels[i-1])
  throw std::invalid_argument("privatized loop levels must be in ascending order");
  }
  for (int i = 0; i < stmt_refs.size(); i++) {
  int stmt_num = stmt_refs[i].first;
  active.insert(stmt_num);
  if (stmt_num < 0 || stmt_num >= stmt.size())
  throw std::invalid_argument("invalid statement number " + to_string(stmt_num));
  if (privatized_levels.size() != 0) {
  if (privatized_levels[privatized_levels.size()-1] > stmt[stmt_num].loop_level.size())
  throw std::invalid_argument("invalid loop level " + to_string(privatized_levels[privatized_levels.size()-1]) + " for statement " + to_string(stmt_num));
  }
  else {
  if (level > stmt[stmt_num].loop_level.size())
  throw std::invalid_argument("invalid loop level " + to_string(level) + " for statement " + to_string(stmt_num));
  }
  for (int j = 0; j < stmt_refs[i].second.size(); j++) {
  if (sym == NULL) {
  sym = stmt_refs[i].second[j]->symbol();
  lex = getLexicalOrder(stmt_num);
  }
  else {
  IR_ArraySymbol *t = stmt_refs[i].second[j]->symbol();
  if (t->name() != sym->name()) {
  delete t;
  delete sym;
  throw std::invalid_argument("try to copy data from different arrays");
  }
  delete t;
  }
  }
  }
  if (!(fastest_changing_dimension >= -1 && fastest_changing_dimension < sym->n_dim()))
  throw std::invalid_argument("invalid fastest changing dimension for the array to be copied");
  if (padding_stride < 0)
  throw std::invalid_argument("invalid temporary array stride requirement");
  if (padding_alignment == -1 || padding_alignment == 0)
  throw std::invalid_argument("invalid temporary array alignment requirement");
  
  int dim = 2*level - 1;
  int n_dim = sym->n_dim();
  
  if (fastest_changing_dimension == -1)
  switch (sym->layout_type()) {
  case IR_ARRAY_LAYOUT_ROW_MAJOR:
  fastest_changing_dimension = n_dim - 1;
  break;
  case IR_ARRAY_LAYOUT_COLUMN_MAJOR:
  fastest_changing_dimension = 0;
  break;
  default:
  throw loop_error("unsupported array layout");
  }
  
  
  // build iteration spaces for all reads and for all writes separately
  apply_xform(active);
  bool has_write_refs = false;
  bool has_read_refs = false;
  Relation wo_copy_is = Relation::False(level-1+privatized_levels.size()+n_dim);
  Relation ro_copy_is = Relation::False(level-1+privatized_levels.size()+n_dim);
  for (int i = 0; i < stmt_refs.size(); i++) {
  int stmt_num = stmt_refs[i].first;
  
  for (int j = 0; j < stmt_refs[i].second.size(); j++) {
  Relation mapping(stmt[stmt_num].IS.n_set(), level-1+privatized_levels.size()+n_dim);
  for (int k = 1; k <= mapping.n_inp(); k++)
  mapping.name_input_var(k, stmt[stmt_num].IS.set_var(k)->name());
  mapping.setup_names();
  F_And *f_root = mapping.add_and();
  for (int k = 1; k <= level-1; k++) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(mapping.input_var(k), 1);
  h.update_coef(mapping.output_var(k), -1);
  }
  for (int k = 0; k < privatized_levels.size(); k++) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(mapping.input_var(privatized_levels[k]), 1);
  h.update_coef(mapping.output_var(level+k), -1);
  }
  for (int k = 0; k < n_dim; k++) {
  CG_outputRepr *repr = stmt_refs[i].second[j]->index(k);
  exp2formula(ir, mapping, f_root, freevar, repr, mapping.output_var(level-1+privatized_levels.size()+k+1), 'w', IR_COND_EQ, false);
  repr->clear();
  delete repr;
  }
  Relation r = Range(Restrict_Domain(mapping, Intersection(copy(stmt[stmt_num].IS), Extend_Set(copy(this->known), stmt[stmt_num].IS.n_set() - this->known.n_set()))));
  if (stmt_refs[i].second[j]->is_write()) {
  has_write_refs = true;
  wo_copy_is = Union(wo_copy_is, r);
  wo_copy_is.simplify(2, 4);
  }
  else {
  has_read_refs = true;
  //protonu--removing the next line for now
  ro_copy_is = Union(ro_copy_is, r);
  ro_copy_is.simplify(2, 4);
  //ro_copy_is = ConvexRepresentation(Union(ro_copy_is, r));
  
  }
  }
  }
  
  if (allow_extra_read) {
  Relation t = DecoupledConvexHull(copy(ro_copy_is));
  if (t.number_of_conjuncts() > 1)
  ro_copy_is = RectHull(ro_copy_is);
  else
  ro_copy_is = t;
  }
  else {
  Relation t = ConvexRepresentation(copy(ro_copy_is));
  if (t.number_of_conjuncts() > 1)
  ro_copy_is = RectHull(ro_copy_is);
  else
  ro_copy_is = t;
  }
  wo_copy_is = ConvexRepresentation(wo_copy_is);
  
  if (allow_extra_read) {
  Tuple<Relation> Rs;
  Tuple<int> active;
  for (DNF_Iterator di(ro_copy_is.query_DNF()); di; di++) {
  Rs.append(Relation(ro_copy_is, di.curr()));
  active.append(1);
  }
  Relation the_gcs = Relation::True(ro_copy_is.n_set());
  for (int i = level-1+privatized_levels.size()+1; i <= level-1+privatized_levels.size()+n_dim; i++) {
  Relation r = greatest_common_step(Rs, active, i, Relation::Null());
  the_gcs = Intersection(the_gcs, r);
  }
  
  ro_copy_is = Approximate(ro_copy_is);
  ro_copy_is = ConvexRepresentation(ro_copy_is);
  ro_copy_is = Intersection(ro_copy_is, the_gcs);
  ro_copy_is.simplify();
  }
  
  
  
  for (int i = 1; i < level; i++) {
  std::string s = stmt[*active.begin()].IS.input_var(i)->name();
  wo_copy_is.name_set_var(i, s);
  ro_copy_is.name_set_var(i, s);
  }
  for (int i = 0; i < privatized_levels.size(); i++) {
  std::string s = stmt[*active.begin()].IS.input_var(privatized_levels[i])->name();
  wo_copy_is.name_set_var(level+i, s);
  ro_copy_is.name_set_var(level+i, s);
  }
  for (int i = level+privatized_levels.size(); i < level+privatized_levels.size()+n_dim; i++) {
  std::string s = tmp_loop_var_name_prefix + to_string(tmp_loop_var_name_counter+i-level-privatized_levels.size());
  wo_copy_is.name_set_var(i, s);
  ro_copy_is.name_set_var(i, s);
  }
  tmp_loop_var_name_counter += n_dim;
  
  //protonu--end change
  
  wo_copy_is.setup_names();
  ro_copy_is.setup_names();
  
  // build merged iteration space for calculating temporary array size
  bool already_use_recthull = false;
  Relation untampered_copy_is = ConvexRepresentation(Union(copy(wo_copy_is), copy(ro_copy_is)));
  Relation copy_is = untampered_copy_is;
  if (copy_is.number_of_conjuncts() > 1) {
  try {
  copy_is = ConvexHull(copy(untampered_copy_is));
  }
  catch (const std::overflow_error &e) {
  copy_is = RectHull(copy(untampered_copy_is));
  already_use_recthull = true;
  }
  }
  
  
  Retry_copy_is:
  // extract temporary array information
  CG_outputBuilder *ocg = ir->builder();
  std::vector<CG_outputRepr *> index_lb(n_dim); // initialized to NULL
  std::vector<coef_t> index_stride(n_dim, 1);
  std::vector<bool> is_index_eq(n_dim, false);
  std::vector<std::pair<int, CG_outputRepr *> > index_sz(0);  
  Relation reduced_copy_is = copy(copy_is);
  
  for (int i = 0; i < n_dim; i++) {
  if (i != 0)
  reduced_copy_is = Project(reduced_copy_is, level-1+privatized_levels.size()+i, Set_Var);
  Relation bound = get_loop_bound(reduced_copy_is, level-1+privatized_levels.size()+i);
  
  // extract stride
  EQ_Handle stride_eq;
  {
  bool simple_stride = true;
  int strides = countStrides(bound.query_DNF()->single_conjunct(), bound.set_var(level-1+privatized_levels.size()+i+1), stride_eq, simple_stride);
  if (strides > 1) {
  throw loop_error("too many strides");
  }
  else if (strides == 1) {
  int sign = stride_eq.get_coef(bound.set_var(level-1+privatized_levels.size()+i+1));
  Constr_Vars_Iter it(stride_eq, true);
  index_stride[i] = abs((*it).coef/sign);
  }
  }
  
  // check if this arary index requires loop
  Conjunct *c = bound.query_DNF()->single_conjunct();
  for (EQ_Iterator ei(c->EQs()); ei; ei++) {
  if ((*ei).has_wildcards())
  continue;
  
  int coef = (*ei).get_coef(bound.set_var(level-1+privatized_levels.size()+i+1));
  if (coef != 0) {
  int sign = 1;
  if (coef < 0) {
  coef = -coef;
  sign = -1;
  }
  
  CG_outputRepr *op = NULL;
  for (Constr_Vars_Iter ci(*ei); ci; ci++) {
  switch ((*ci).var->kind()) {
  case Input_Var:
  {
  if ((*ci).var != bound.set_var(level-1+privatized_levels.size()+i+1))
  if ((*ci).coef*sign == 1)
  op = ocg->CreateMinus(op, ocg->CreateIdent((*ci).var->name()));
  else if ((*ci).coef*sign == -1)
  op = ocg->CreatePlus(op, ocg->CreateIdent((*ci).var->name()));
  else if ((*ci).coef*sign > 1)
  op = ocg->CreateMinus(op, ocg->CreateTimes(ocg->CreateInt(abs((*ci).coef)), ocg->CreateIdent((*ci).var->name())));
  else // (*ci).coef*sign < -1
  op = ocg->CreatePlus(op, ocg->CreateTimes(ocg->CreateInt(abs((*ci).coef)), ocg->CreateIdent((*ci).var->name())));
  break;
  }
  case Global_Var:
  {
  Global_Var_ID g = (*ci).var->get_global_var();
  if ((*ci).coef*sign == 1)
  op = ocg->CreateMinus(op, ocg->CreateIdent(g->base_name()));
  else if ((*ci).coef*sign == -1)
  op = ocg->CreatePlus(op, ocg->CreateIdent(g->base_name()));
  else if ((*ci).coef*sign > 1)
  op = ocg->CreateMinus(op, ocg->CreateTimes(ocg->CreateInt(abs((*ci).coef)), ocg->CreateIdent(g->base_name())));
  else // (*ci).coef*sign < -1
  op = ocg->CreatePlus(op, ocg->CreateTimes(ocg->CreateInt(abs((*ci).coef)), ocg->CreateIdent(g->base_name())));
  break;
  }
  default:
  throw loop_error("unsupported array index expression");
  }
  }
  if ((*ei).get_const() != 0)
  op = ocg->CreatePlus(op, ocg->CreateInt(-sign*((*ei).get_const())));
  if (coef != 1)
  op = ocg->CreateIntegerDivide(op, ocg->CreateInt(coef));
  
  index_lb[i] = op;
  is_index_eq[i] = true;
  break;
  }
  }
  if (is_index_eq[i])
  continue;
  
  // seperate lower and upper bounds
  std::vector<GEQ_Handle> lb_list, ub_list;
  for (GEQ_Iterator gi(c->GEQs()); gi; gi++) {
  int coef = (*gi).get_coef(bound.set_var(level-1+privatized_levels.size()+i+1));
  if (coef != 0 && (*gi).has_wildcards()) {
  bool clean_bound = true;
  GEQ_Handle h;
  for (Constr_Vars_Iter cvi(*gi, true); gi; gi++)
  if (!findFloorInequality(bound, (*cvi).var, h, bound.set_var(level-1+privatized_levels.size()+i+1))) {
  clean_bound = false;
  break;
  }
  if (!clean_bound)
  continue;
  }
  
  if (coef > 0)
  lb_list.push_back(*gi);
  else if (coef < 0)
  ub_list.push_back(*gi);
  }
  if (lb_list.size() == 0 || ub_list.size() == 0)
  if (already_use_recthull)
  throw loop_error("failed to calcuate array footprint size");
  else {
  copy_is = RectHull(copy(untampered_copy_is));
  already_use_recthull = true;
  goto Retry_copy_is;
  }
  
  // build lower bound representation
  Tuple<CG_outputRepr *> lb_repr_list;
  for (int j = 0; j < lb_list.size(); j++)
  lb_repr_list.append(outputLBasRepr(ocg, lb_list[j], bound,
  bound.set_var(level-1+privatized_levels.size()+i+1), 
  index_stride[i], stride_eq, Relation::True(bound.n_set()),
  std::vector<CG_outputRepr *>(bound.n_set())));
  
  if (lb_repr_list.size() > 1)
  index_lb[i] = ocg->CreateInvoke("max", lb_repr_list);
  else if (lb_repr_list.size() == 1)
  index_lb[i] = lb_repr_list[1];
  
  // build temporary array size representation
  {
  Relation cal(copy_is.n_set(), 1);
  F_And *f_root = cal.add_and();
  for (int j = 0; j < ub_list.size(); j++)
  for (int k = 0; k < lb_list.size(); k++) {
  GEQ_Handle h = f_root->add_GEQ();
  
  for (Constr_Vars_Iter ci(ub_list[j]); ci; ci++) {
  switch ((*ci).var->kind()) {
  case Input_Var:
  {
  int pos = (*ci).var->get_position();
  h.update_coef(cal.input_var(pos), (*ci).coef);
  break;
  }
  case Global_Var:
  {
  Global_Var_ID g = (*ci).var->get_global_var();
  Variable_ID v;
  if (g->arity() == 0)
  v = cal.get_local(g);
  else
  v = cal.get_local(g, (*ci).var->function_of());
  h.update_coef(v, (*ci).coef);
  break;
  }
  default:
  throw loop_error("cannot calculate temporay array size statically");
  }
  }
  h.update_const(ub_list[j].get_const());
  
  for (Constr_Vars_Iter ci(lb_list[k]); ci; ci++) {
  switch ((*ci).var->kind()) {
  case Input_Var:
  {
  int pos = (*ci).var->get_position();
  h.update_coef(cal.input_var(pos), (*ci).coef);
  break;
  }
  case Global_Var:
  {
  Global_Var_ID g = (*ci).var->get_global_var();
  Variable_ID v;
  if (g->arity() == 0)
  v = cal.get_local(g);
  else
  v = cal.get_local(g, (*ci).var->function_of());
  h.update_coef(v, (*ci).coef);
  break;
  }
  default:
  throw loop_error("cannot calculate temporay array size statically");
  }
  }
  h.update_const(lb_list[k].get_const());
  
  h.update_const(1);
  h.update_coef(cal.output_var(1), -1);
  }
  
  cal = Restrict_Domain(cal, copy(copy_is));
  for (int j = 1; j <= cal.n_inp(); j++)
  cal = Project(cal, j, Input_Var);
  cal.simplify();
  
  // pad temporary array size
  // TODO: for variable array size, create padding formula
  Conjunct *c = cal.query_DNF()->single_conjunct();
  bool is_index_bound_const = false;
  for (GEQ_Iterator gi(c->GEQs()); gi && !is_index_bound_const; gi++)
  if ((*gi).is_const(cal.output_var(1))) {
  coef_t size = (*gi).get_const() / (-(*gi).get_coef(cal.output_var(1)));
  if (padding_stride != 0) {
  size = (size + index_stride[i] - 1) / index_stride[i];
  if (i == fastest_changing_dimension)
  size = size * padding_stride;
  }
  if (i == fastest_changing_dimension) {
  if (padding_alignment > 1) { // align to boundary for data packing
  int residue = size % padding_alignment;
  if (residue)
  size = size+padding_alignment-residue;
  }
  else if (padding_alignment < -1) {  // un-alignment for memory bank conflicts
  while (gcd(size, static_cast<coef_t>(-padding_alignment)) != 1)
  size++;
  }
  }
  index_sz.push_back(std::make_pair(i, ocg->CreateInt(size)));
  is_index_bound_const = true;
  }
  
  if (!is_index_bound_const) {
  for (GEQ_Iterator gi(c->GEQs()); gi && !is_index_bound_const; gi++) {
  int coef = (*gi).get_coef(cal.output_var(1));
  if (coef < 0) {
  CG_outputRepr *op = NULL;
  for (Constr_Vars_Iter ci(*gi); ci; ci++) {
  if ((*ci).var != cal.output_var(1)) {
  switch((*ci).var->kind()) {
  case Global_Var:
  {
  Global_Var_ID g = (*ci).var->get_global_var();
  if ((*ci).coef == 1)
  op = ocg->CreatePlus(op, ocg->CreateIdent(g->base_name()));
  else if ((*ci).coef == -1)
  op = ocg->CreateMinus(op, ocg->CreateIdent(g->base_name()));
  else if ((*ci).coef > 1)
  op = ocg->CreatePlus(op, ocg->CreateTimes(ocg->CreateInt((*ci).coef), ocg->CreateIdent(g->base_name())));
  else // (*ci).coef < -1
  op = ocg->CreateMinus(op, ocg->CreateTimes(ocg->CreateInt(-(*ci).coef), ocg->CreateIdent(g->base_name())));
  break;
  }
  default:
  throw loop_error("failed to generate array index bound code");
  }
  }
  }
  int c = (*gi).get_const();
  if (c > 0)
  op = ocg->CreatePlus(op, ocg->CreateInt(c));
  else if (c < 0)
  op = ocg->CreateMinus(op, ocg->CreateInt(-c));
  if (padding_stride != 0) {
  if (i == fastest_changing_dimension) {
  coef_t g = gcd(index_stride[i], static_cast<coef_t>(padding_stride));
  coef_t t1 = index_stride[i] / g;
  if (t1 != 1)
  op = ocg->CreateIntegerDivide(ocg->CreatePlus(op, ocg->CreateInt(t1-1)), ocg->CreateInt(t1));
  coef_t t2 = padding_stride / g;
  if (t2 != 1)
  op = ocg->CreateTimes(op, ocg->CreateInt(t2));
  }
  else if (index_stride[i] != 1) {
  op = ocg->CreateIntegerDivide(ocg->CreatePlus(op, ocg->CreateInt(index_stride[i]-1)), ocg->CreateInt(index_stride[i]));
  }
  }
  
  index_sz.push_back(std::make_pair(i, op));
  break;
  }
  }
  }
  }
  }
  
  // change the temporary array index order
  for (int i = 0; i < index_sz.size(); i++)
  if (index_sz[i].first == fastest_changing_dimension)
  switch (sym->layout_type()) {
  case IR_ARRAY_LAYOUT_ROW_MAJOR:
  std::swap(index_sz[index_sz.size()-1], index_sz[i]);
  break;
  case IR_ARRAY_LAYOUT_COLUMN_MAJOR:
  std::swap(index_sz[0], index_sz[i]);
  break;
  default:
  throw loop_error("unsupported array layout");
  }
  
  // declare temporary array or scalar
  IR_Symbol *tmp_sym;
  if (index_sz.size() == 0) {
  tmp_sym = ir->CreateScalarSymbol(sym, memory_type);
  }
  else {
  std::vector<CG_outputRepr *> tmp_array_size(index_sz.size());
  for (int i = 0; i < index_sz.size(); i++)
  tmp_array_size[i] = index_sz[i].second->clone();
  tmp_sym = ir->CreateArraySymbol(sym, tmp_array_size, memory_type);
  }
  
  // create temporary array read initialization code
  CG_outputRepr *copy_code_read;
  if (has_read_refs)
  if (index_sz.size() == 0) {
  IR_ScalarRef *tmp_scalar_ref = ir->CreateScalarRef(static_cast<IR_ScalarSymbol *>(tmp_sym));
  
  std::vector<CG_outputRepr *> rhs_index(n_dim);
  for (int i = 0; i < index_lb.size(); i++)
  if (is_index_eq[i])
  rhs_index[i] = index_lb[i]->clone();
  else
  rhs_index[i] = ir->builder()->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+i+1)->name());
  IR_ArrayRef *copied_array_ref = ir->CreateArrayRef(sym, rhs_index);
  
  copy_code_read = ir->builder()->CreateAssignment(0, tmp_scalar_ref->convert(), copied_array_ref->convert());
  }
  else {
  std::vector<CG_outputRepr *> lhs_index(index_sz.size());
  for (int i = 0; i < index_sz.size(); i++) {
  int cur_index_num = index_sz[i].first;
  CG_outputRepr *cur_index_repr = ocg->CreateMinus(ocg->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+cur_index_num+1)->name()), index_lb[cur_index_num]->clone());
  if (padding_stride != 0) {
  if (i == n_dim-1) {
  coef_t g = gcd(index_stride[cur_index_num], static_cast<coef_t>(padding_stride));
  coef_t t1 = index_stride[cur_index_num] / g;
  if (t1 != 1)
  cur_index_repr = ocg->CreateIntegerDivide(cur_index_repr, ocg->CreateInt(t1));
  coef_t t2 = padding_stride / g;
  if (t2 != 1)
  cur_index_repr = ocg->CreateTimes(cur_index_repr, ocg->CreateInt(t2));
  }
  else if (index_stride[cur_index_num] != 1) {
  cur_index_repr = ocg->CreateIntegerDivide(cur_index_repr, ocg->CreateInt(index_stride[cur_index_num]));
  }
  }
  
  if (ir->ArrayIndexStartAt() != 0)
  cur_index_repr = ocg->CreatePlus(cur_index_repr, ocg->CreateInt(ir->ArrayIndexStartAt()));
  lhs_index[i] = cur_index_repr;
  }
  
  IR_ArrayRef *tmp_array_ref = ir->CreateArrayRef(static_cast<IR_ArraySymbol *>(tmp_sym), lhs_index);
  
  std::vector<CG_outputRepr *> rhs_index(n_dim);
  for (int i = 0; i < index_lb.size(); i++)
  if (is_index_eq[i])
  rhs_index[i] = index_lb[i]->clone();
  else
  rhs_index[i] = ir->builder()->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+i+1)->name());
  IR_ArrayRef *copied_array_ref = ir->CreateArrayRef(sym, rhs_index);
  
  copy_code_read = ir->builder()->CreateAssignment(0, tmp_array_ref->convert(), copied_array_ref->convert());
  }
  
  // create temporary array write back code
  CG_outputRepr *copy_code_write;
  if (has_write_refs)
  if (index_sz.size() == 0) {
  IR_ScalarRef *tmp_scalar_ref = ir->CreateScalarRef(static_cast<IR_ScalarSymbol *>(tmp_sym));
  
  std::vector<CG_outputRepr *> rhs_index(n_dim);
  for (int i = 0; i < index_lb.size(); i++)
  if (is_index_eq[i])
  rhs_index[i] = index_lb[i]->clone();
  else
  rhs_index[i] = ir->builder()->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+i+1)->name());
  IR_ArrayRef *copied_array_ref = ir->CreateArrayRef(sym, rhs_index);
  
  copy_code_write = ir->builder()->CreateAssignment(0, copied_array_ref->convert(), tmp_scalar_ref->convert());
  }
  else {
  std::vector<CG_outputRepr *> lhs_index(n_dim);
  for (int i = 0; i < index_lb.size(); i++)
  if (is_index_eq[i])
  lhs_index[i] = index_lb[i]->clone();
  else
  lhs_index[i] = ir->builder()->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+i+1)->name());
  IR_ArrayRef *copied_array_ref = ir->CreateArrayRef(sym, lhs_index);
  
  std::vector<CG_outputRepr *> rhs_index(index_sz.size());
  for (int i = 0; i < index_sz.size(); i++) {
  int cur_index_num = index_sz[i].first;
  CG_outputRepr *cur_index_repr = ocg->CreateMinus(ocg->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+cur_index_num+1)->name()), index_lb[cur_index_num]->clone());
  if (padding_stride != 0) {
  if (i == n_dim-1) {
  coef_t g = gcd(index_stride[cur_index_num], static_cast<coef_t>(padding_stride));
  coef_t t1 = index_stride[cur_index_num] / g;
  if (t1 != 1)
  cur_index_repr = ocg->CreateIntegerDivide(cur_index_repr, ocg->CreateInt(t1));
  coef_t t2 = padding_stride / g;
  if (t2 != 1)
  cur_index_repr = ocg->CreateTimes(cur_index_repr, ocg->CreateInt(t2));
  }
  else if (index_stride[cur_index_num] != 1) {
  cur_index_repr = ocg->CreateIntegerDivide(cur_index_repr, ocg->CreateInt(index_stride[cur_index_num]));
  }
  }
  
  if (ir->ArrayIndexStartAt() != 0)
  cur_index_repr = ocg->CreatePlus(cur_index_repr, ocg->CreateInt(ir->ArrayIndexStartAt()));
  rhs_index[i] = cur_index_repr;
  }
  IR_ArrayRef *tmp_array_ref = ir->CreateArrayRef(static_cast<IR_ArraySymbol *>(tmp_sym), rhs_index);
  
  copy_code_write = ir->builder()->CreateAssignment(0, copied_array_ref->convert(), tmp_array_ref->convert());
  }
  
  // now we can remove those loops for array indexes that are
  // dependent on others
  if (!(index_sz.size() == n_dim && (sym->layout_type() == IR_ARRAY_LAYOUT_ROW_MAJOR || n_dim <= 1))) {
  Relation mapping(level-1+privatized_levels.size()+n_dim, level-1+privatized_levels.size()+index_sz.size());
  F_And *f_root = mapping.add_and();
  for (int i = 1; i <= level-1+privatized_levels.size(); i++) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(mapping.input_var(i), 1);
  h.update_coef(mapping.output_var(i), -1);
  }
  
  int cur_index = 0;
  std::vector<int> mapped_index(index_sz.size());
  for (int i = 0; i < n_dim; i++)
  if (!is_index_eq[i]) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(mapping.input_var(level-1+privatized_levels.size()+i+1), 1);
  switch (sym->layout_type()) {
  case IR_ARRAY_LAYOUT_COLUMN_MAJOR: {
  h.update_coef(mapping.output_var(level-1+privatized_levels.size()+index_sz.size()-cur_index), -1);
  mapped_index[index_sz.size()-cur_index-1] = i;
  break;
  }
  case IR_ARRAY_LAYOUT_ROW_MAJOR: {
  h.update_coef(mapping.output_var(level-1+privatized_levels.size()+cur_index+1), -1);
  mapped_index[cur_index] = i;
  break;
  }
  default:
  throw loop_error("unsupported array layout");
  }
  cur_index++;
  }
  
  wo_copy_is = Range(Restrict_Domain(copy(mapping), wo_copy_is));
  ro_copy_is = Range(Restrict_Domain(copy(mapping), ro_copy_is));
  
  // protonu--replacing Chun's old code 
  for (int i = 1; i <= level-1+privatized_levels.size(); i++) {
  wo_copy_is.name_set_var(i, copy_is.set_var(i)->name());
  ro_copy_is.name_set_var(i, copy_is.set_var(i)->name());
  }
  
  
  
  for (int i = 0; i < index_sz.size(); i++) {
  wo_copy_is.name_set_var(level-1+privatized_levels.size()+i+1, copy_is.set_var(level-1+privatized_levels.size()+mapped_index[i]+1)->name());
  ro_copy_is.name_set_var(level-1+privatized_levels.size()+i+1, copy_is.set_var(level-1+privatized_levels.size()+mapped_index[i]+1)->name());
  }      
  wo_copy_is.setup_names();
  ro_copy_is.setup_names();
  }
  
  // insert read copy statement
  int old_num_stmt = stmt.size();
  int ro_copy_stmt_num = -1;
  if (has_read_refs) {
  Relation copy_xform(ro_copy_is.n_set(), 2*ro_copy_is.n_set()+1);
  {
  F_And *f_root = copy_xform.add_and();
  for (int i = 1; i <= ro_copy_is.n_set(); i++) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(copy_xform.input_var(i), 1);
  h.update_coef(copy_xform.output_var(2*i), -1);
  }
  for (int i = 1; i <= dim; i+=2) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(copy_xform.output_var(i), -1);
  h.update_const(lex[i-1]);
  }
  for (int i = dim+2; i <= copy_xform.n_out(); i+=2) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(copy_xform.output_var(i), 1);
  }
  }
  
  Statement copy_stmt_read;
  copy_stmt_read.IS = ro_copy_is;
  copy_stmt_read.xform = copy_xform;
  copy_stmt_read.code = copy_code_read;
  copy_stmt_read.loop_level = std::vector<LoopLevel>(ro_copy_is.n_set());
  copy_stmt_read.ir_stmt_node = NULL;
  for (int i = 0; i < level-1; i++) {
  copy_stmt_read.loop_level[i].type = stmt[*(active.begin())].loop_level[i].type;
  if (stmt[*(active.begin())].loop_level[i].type == LoopLevelTile &&
  stmt[*(active.begin())].loop_level[i].payload >= level) {
  int j;
  for (j = 0; j < privatized_levels.size(); j++)
  if (privatized_levels[j] == stmt[*(active.begin())].loop_level[i].payload)
  break;
  if (j == privatized_levels.size())
  copy_stmt_read.loop_level[i].payload = -1;
  else
  copy_stmt_read.loop_level[i].payload = level + j;
  }
  else
  copy_stmt_read.loop_level[i].payload = stmt[*(active.begin())].loop_level[i].payload;
  copy_stmt_read.loop_level[i].parallel_level = stmt[*(active.begin())].loop_level[i].parallel_level;
  }
  for (int i = 0; i < privatized_levels.size(); i++) {
  copy_stmt_read.loop_level[level-1+i].type = stmt[*(active.begin())].loop_level[privatized_levels[i]].type;
  copy_stmt_read.loop_level[level-1+i].payload = stmt[*(active.begin())].loop_level[privatized_levels[i]].payload;
  copy_stmt_read.loop_level[level-1+i].parallel_level = stmt[*(active.begin())].loop_level[privatized_levels[i]].parallel_level;
  }
  int left_num_dim = num_dep_dim - (get_last_dep_dim_before(*(active.begin()), level) + 1);
  for (int i = 0; i < min(left_num_dim, static_cast<int>(index_sz.size())); i++) {
  copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].type = LoopLevelOriginal;
  copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].payload = num_dep_dim-left_num_dim+i;
  copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].parallel_level = 0;
  }
  for (int i = min(left_num_dim, static_cast<int>(index_sz.size())); i < index_sz.size(); i++) {
  copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].type = LoopLevelUnknown;
  copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].payload = -1;
  copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].parallel_level = 0;
  }
  
  shiftLexicalOrder(lex, dim-1, 1);
  stmt.push_back(copy_stmt_read);
  ro_copy_stmt_num = stmt.size() - 1;
  dep.insert();
  }
  
  // insert write copy statement
  int wo_copy_stmt_num = -1;
  if (has_write_refs) {
  Relation copy_xform(wo_copy_is.n_set(), 2*wo_copy_is.n_set()+1);
  {
  F_And *f_root = copy_xform.add_and();
  for (int i = 1; i <= wo_copy_is.n_set(); i++) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(copy_xform.input_var(i), 1);
  h.update_coef(copy_xform.output_var(2*i), -1);
  }
  for (int i = 1; i <= dim; i+=2) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(copy_xform.output_var(i), -1);
  h.update_const(lex[i-1]);
  }
  for (int i = dim+2; i <= copy_xform.n_out(); i+=2) {
  EQ_Handle h = f_root->add_EQ();
  h.update_coef(copy_xform.output_var(i), 1);
  }
  }
  
  Statement copy_stmt_write;
  copy_stmt_write.IS = wo_copy_is;
  copy_stmt_write.xform = copy_xform;
  copy_stmt_write.code = copy_code_write;
  copy_stmt_write.loop_level = std::vector<LoopLevel>(wo_copy_is.n_set());
  copy_stmt_write.ir_stmt_node = NULL;
  
  for (int i = 0; i < level-1; i++) {
  copy_stmt_write.loop_level[i].type = stmt[*(active.begin())].loop_level[i].type;
  if (stmt[*(active.begin())].loop_level[i].type == LoopLevelTile &&
  stmt[*(active.begin())].loop_level[i].payload >= level) {
  int j;
  for (j = 0; j < privatized_levels.size(); j++)
  if (privatized_levels[j] == stmt[*(active.begin())].loop_level[i].payload)
  break;
  if (j == privatized_levels.size())
  copy_stmt_write.loop_level[i].payload = -1;
  else
  copy_stmt_write.loop_level[i].payload = level + j;
  }
  else
  copy_stmt_write.loop_level[i].payload = stmt[*(active.begin())].loop_level[i].payload;
  copy_stmt_write.loop_level[i].parallel_level = stmt[*(active.begin())].loop_level[i].parallel_level;
  }
  for (int i = 0; i < privatized_levels.size(); i++) {
  copy_stmt_write.loop_level[level-1+i].type = stmt[*(active.begin())].loop_level[privatized_levels[i]].type;
  copy_stmt_write.loop_level[level-1+i].payload = stmt[*(active.begin())].loop_level[privatized_levels[i]].payload;
  copy_stmt_write.loop_level[level-1+i].parallel_level = stmt[*(active.begin())].loop_level[privatized_levels[i]].parallel_level;
  }
  int left_num_dim = num_dep_dim - (get_last_dep_dim_before(*(active.begin()), level) + 1);
  for (int i = 0; i < min(left_num_dim, static_cast<int>(index_sz.size())); i++) {
  copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].type = LoopLevelOriginal;
  copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].payload = num_dep_dim-left_num_dim+i;
  copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].parallel_level = 0;
  }
  for (int i = min(left_num_dim, static_cast<int>(index_sz.size())); i < index_sz.size(); i++) {
  copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].type = LoopLevelUnknown;
  copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].payload = -1;
  copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].parallel_level = 0;
  }
  
  lex[dim-1]++;
  shiftLexicalOrder(lex, dim-1, -2);
  stmt.push_back(copy_stmt_write);
  wo_copy_stmt_num = stmt.size() - 1;
  dep.insert();
  } 
  
  // replace original array accesses with temporary array accesses
  for (int i =0; i < stmt_refs.size(); i++)
  for (int j = 0; j < stmt_refs[i].second.size(); j++) {
  if (index_sz.size() == 0) {
  IR_ScalarRef *tmp_scalar_ref = ir->CreateScalarRef(static_cast<IR_ScalarSymbol *>(tmp_sym));
  ir->ReplaceExpression(stmt_refs[i].second[j], tmp_scalar_ref->convert());
  }
  else {
  std::vector<CG_outputRepr *> index_repr(index_sz.size());
  for (int k = 0; k < index_sz.size(); k++) {
  int cur_index_num = index_sz[k].first;
  
  CG_outputRepr *cur_index_repr = ocg->CreateMinus(stmt_refs[i].second[j]->index(cur_index_num), index_lb[cur_index_num]->clone());
  if (padding_stride != 0) {
  if (k == n_dim-1) {
  coef_t g = gcd(index_stride[cur_index_num], static_cast<coef_t>(padding_stride));
  coef_t t1 = index_stride[cur_index_num] / g;
  if (t1 != 1)
  cur_index_repr = ocg->CreateIntegerDivide(cur_index_repr, ocg->CreateInt(t1));
  coef_t t2 = padding_stride / g;
  if (t2 != 1)
  cur_index_repr = ocg->CreateTimes(cur_index_repr, ocg->CreateInt(t2));
  }
  else if (index_stride[cur_index_num] != 1) {
  cur_index_repr = ocg->CreateIntegerDivide(cur_index_repr, ocg->CreateInt(index_stride[cur_index_num]));
  }
  }
  
  if (ir->ArrayIndexStartAt() != 0)
  cur_index_repr = ocg->CreatePlus(cur_index_repr, ocg->CreateInt(ir->ArrayIndexStartAt()));
  index_repr[k] = cur_index_repr;
  }
  
  IR_ArrayRef *tmp_array_ref = ir->CreateArrayRef(static_cast<IR_ArraySymbol *>(tmp_sym), index_repr);
  ir->ReplaceExpression(stmt_refs[i].second[j], tmp_array_ref->convert());
  }
  }
  
  // update dependence graph
  int dep_dim = get_last_dep_dim_before(*(active.begin()), level) + 1;
  if (ro_copy_stmt_num != -1) {
  for (int i = 0; i < old_num_stmt; i++) {
  std::vector<std::vector<DependenceVector> > D;
  
  for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end();) {
  if (active.find(i) != active.end() && active.find(j->first) == active.end()) {
  std::vector<DependenceVector> dvs1, dvs2;
  for (int k = 0; k < j->second.size(); k++) {
  DependenceVector dv = j->second[k];
  if (dv.sym != NULL && dv.sym->name() == sym->name() && (dv.type == DEP_R2R || dv.type == DEP_R2W))
  dvs1.push_back(dv);
  else
  dvs2.push_back(dv);
  }
  j->second = dvs2;
  if (dvs1.size() > 0)
  dep.connect(ro_copy_stmt_num, j->first, dvs1);
  }
  else if (active.find(i) == active.end() && active.find(j->first) != active.end()) {
  std::vector<DependenceVector> dvs1, dvs2;
  for (int k = 0; k < j->second.size(); k++) {
  DependenceVector dv = j->second[k];
  if (dv.sym != NULL && dv.sym->name() == sym->name() && (dv.type == DEP_R2R || dv.type == DEP_W2R))
  dvs1.push_back(dv);
  else
  dvs2.push_back(dv);
  }
  j->second = dvs2;
  if (dvs1.size() > 0)
  D.push_back(dvs1);
  }
  
  if (j->second.size() == 0)
  dep.vertex[i].second.erase(j++);
  else
  j++;
  }
  
  for (int j = 0; j < D.size(); j++)
  dep.connect(i, ro_copy_stmt_num, D[j]);
  }
  
  // insert dependences from copy statement loop to copied statements
  DependenceVector dv;
  dv.type = DEP_W2R;
  dv.sym = tmp_sym->clone();
  dv.lbounds = std::vector<coef_t>(num_dep_dim, 0);
  dv.ubounds = std::vector<coef_t>(num_dep_dim, 0);
  for (int i = dep_dim; i < num_dep_dim; i++) {
  dv.lbounds[i] = -posInfinity;
  dv.ubounds[i] = posInfinity;
  } 
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++)
  dep.connect(ro_copy_stmt_num, *i, dv);
  }
  
  if (wo_copy_stmt_num != -1) {
  for (int i = 0; i < old_num_stmt; i++) {
  std::vector<std::vector<DependenceVector> > D;
  
  for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end();) {
  if (active.find(i) != active.end() && active.find(j->first) == active.end()) {
  std::vector<DependenceVector> dvs1, dvs2;
  for (int k = 0; k < j->second.size(); k++) {
  DependenceVector dv = j->second[k];
  if (dv.sym != NULL && dv.sym->name() == sym->name() && (dv.type == DEP_W2R || dv.type == DEP_W2W))
  dvs1.push_back(dv);
  else
  dvs2.push_back(dv);
  }
  j->second = dvs2;
  if (dvs1.size() > 0)
  dep.connect(wo_copy_stmt_num, j->first, dvs1);
  }
  else if (active.find(i) == active.end() && active.find(j->first) != active.end()) {
  std::vector<DependenceVector> dvs1, dvs2;
  for (int k = 0; k < j->second.size(); k++) {
  DependenceVector dv = j->second[k];
  if (dv.sym != NULL && dv.sym->name() == sym->name() && (dv.type == DEP_R2W || dv.type == DEP_W2W))
  dvs1.push_back(dv);
  else
  dvs2.push_back(dv);
  }
  j->second = dvs2;
  if (dvs1.size() > 0)
  D.push_back(dvs1);
  }
  
  if (j->second.size() == 0)
  dep.vertex[i].second.erase(j++);
  else
  j++;
  }
  
  for (int j = 0; j < D.size(); j++)
  dep.connect(i, wo_copy_stmt_num, D[j]);
  }
  
  // insert dependences from copied statements to write statements
  DependenceVector dv;
  dv.type = DEP_W2R;
  dv.sym = tmp_sym->clone();
  dv.lbounds = std::vector<coef_t>(num_dep_dim, 0);
  dv.ubounds = std::vector<coef_t>(num_dep_dim, 0);
  for (int i = dep_dim; i < num_dep_dim; i++) {
  dv.lbounds[i] = -posInfinity;
  dv.ubounds[i] = posInfinity;
  } 
  for (std::set<int>::iterator i = active.begin(); i != active.end(); i++)
  dep.connect(*i, wo_copy_stmt_num, dv);
  
  }
  
  // update variable name for dependences among copied statements
  for (int i = 0; i < old_num_stmt; i++) {
  if (active.find(i) != active.end())
  for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end(); j++)
  if (active.find(j->first) != active.end())
  for (int k = 0; k < j->second.size(); k++) {
  IR_Symbol *s = tmp_sym->clone();
  j->second[k].sym = s;
  }
  }
  
  // insert anti-dependence from write statement to read statement
  if (ro_copy_stmt_num != -1 && wo_copy_stmt_num != -1)
  if (dep_dim >= 0) {
  DependenceVector dv;
  dv.type = DEP_R2W;
  dv.sym = tmp_sym->clone();
  dv.lbounds = std::vector<coef_t>(num_dep_dim, 0);
  dv.ubounds = std::vector<coef_t>(num_dep_dim, 0);
  for (int k = dep_dim; k < num_dep_dim; k++) {
  dv.lbounds[k] = -posInfinity;
  dv.ubounds[k] = posInfinity;
  }
  for (int k = 0; k < dep_dim; k++) {
  if (k != 0) {
  dv.lbounds[k-1] = 0;
  dv.ubounds[k-1] = 0;
  }
  dv.lbounds[k] = 1;
  dv.ubounds[k] = posInfinity;
  dep.connect(wo_copy_stmt_num, ro_copy_stmt_num, dv);
  }
  }
  
  
  // cleanup
  delete sym;
  delete tmp_sym;
  for (int i = 0; i < index_lb.size(); i++) {
  index_lb[i]->clear();
  delete index_lb[i];
  }
  for (int i = 0; i < index_sz.size(); i++) {
  index_sz[i].second->clear();
  delete index_sz[i].second;
  }
  
  return true;
  }
*/
bool Loop::datacopy_privatized(const std::vector<std::pair<int, std::vector<IR_ArrayRef *> > > &stmt_refs, int level,
                               const std::vector<int> &privatized_levels,
                               bool allow_extra_read, int fastest_changing_dimension,
                               int padding_stride, int padding_alignment, int memory_type) {
  if (stmt_refs.size() == 0)
    return true;
  
  // check for sanity of parameters
  IR_ArraySymbol *sym = NULL;
  std::vector<int> lex;
  std::set<int> active;
  if (level <= 0)
    throw std::invalid_argument("invalid loop level " + to_string(level));
  for (int i = 0; i < privatized_levels.size(); i++) {
    if (i == 0) {
      if (privatized_levels[i] < level)
        throw std::invalid_argument("privatized loop levels must be no less than level " + to_string(level));
    }
    else if (privatized_levels[i] <= privatized_levels[i-1])
      throw std::invalid_argument("privatized loop levels must be in ascending order");
  }
  for (int i = 0; i < stmt_refs.size(); i++) {
    int stmt_num = stmt_refs[i].first;
    active.insert(stmt_num);
    if (stmt_num < 0 || stmt_num >= stmt.size())
      throw std::invalid_argument("invalid statement number " + to_string(stmt_num));
    if (privatized_levels.size() != 0) {
      if (privatized_levels[privatized_levels.size()-1] > stmt[stmt_num].loop_level.size())
        throw std::invalid_argument("invalid loop level " + to_string(privatized_levels[privatized_levels.size()-1]) + " for statement " + to_string(stmt_num));
    }
    else {
      if (level > stmt[stmt_num].loop_level.size())
        throw std::invalid_argument("invalid loop level " + to_string(level) + " for statement " + to_string(stmt_num));
    }
    for (int j = 0; j < stmt_refs[i].second.size(); j++) {
      if (sym == NULL) {
        sym = stmt_refs[i].second[j]->symbol();
        lex = getLexicalOrder(stmt_num);
      }
      else {
        IR_ArraySymbol *t = stmt_refs[i].second[j]->symbol();
        if (t->name() != sym->name()) {
          delete t;
          delete sym;
          throw std::invalid_argument("try to copy data from different arrays");
        }
        delete t;
      }
    }
  }
  if (!(fastest_changing_dimension >= -1 && fastest_changing_dimension < sym->n_dim()))
    throw std::invalid_argument("invalid fastest changing dimension for the array to be copied");
  if (padding_stride < 0)
    throw std::invalid_argument("invalid temporary array stride requirement");
  if (padding_alignment == -1 || padding_alignment == 0)
    throw std::invalid_argument("invalid temporary array alignment requirement");
  
  int dim = 2*level - 1;
  int n_dim = sym->n_dim();
  

  if (fastest_changing_dimension == -1)
    switch (sym->layout_type()) {
    case IR_ARRAY_LAYOUT_ROW_MAJOR:
      fastest_changing_dimension = n_dim - 1;
      break;
    case IR_ARRAY_LAYOUT_COLUMN_MAJOR:
      fastest_changing_dimension = 0;
      break;
    default:
      throw loop_error("unsupported array layout");
    }

  
  // invalidate saved codegen computation
  delete last_compute_cgr_;
  last_compute_cgr_ = NULL;
  delete last_compute_cg_;
  last_compute_cg_ = NULL;
  
  // build iteration spaces for all reads and for all writes separately
  apply_xform(active);
  
  bool has_write_refs = false;
  bool has_read_refs = false;
  Relation wo_copy_is = Relation::False(level-1+privatized_levels.size()+n_dim);
  Relation ro_copy_is = Relation::False(level-1+privatized_levels.size()+n_dim);
  for (int i = 0; i < stmt_refs.size(); i++) {
    int stmt_num = stmt_refs[i].first;
    
    for (int j = 0; j < stmt_refs[i].second.size(); j++) {
      Relation mapping(stmt[stmt_num].IS.n_set(), level-1+privatized_levels.size()+n_dim);
      for (int k = 1; k <= mapping.n_inp(); k++)
        mapping.name_input_var(k, stmt[stmt_num].IS.set_var(k)->name());
      mapping.setup_names();
      F_And *f_root = mapping.add_and();
      for (int k = 1; k <= level-1; k++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(mapping.input_var(k), 1);
        h.update_coef(mapping.output_var(k), -1);
      }
      for (int k = 0; k < privatized_levels.size(); k++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(mapping.input_var(privatized_levels[k]), 1);
        h.update_coef(mapping.output_var(level+k), -1);
      }
      for (int k = 0; k < n_dim; k++) {
        CG_outputRepr *repr = stmt_refs[i].second[j]->index(k);
        exp2formula(ir, mapping, f_root, freevar, repr, mapping.output_var(level-1+privatized_levels.size()+k+1), 'w', IR_COND_EQ, false);
        repr->clear();
        delete repr;
      }
      Relation r = Range(Restrict_Domain(mapping, Intersection(copy(stmt[stmt_num].IS), Extend_Set(copy(this->known), stmt[stmt_num].IS.n_set() - this->known.n_set()))));
      if (stmt_refs[i].second[j]->is_write()) {
        has_write_refs = true;
        wo_copy_is = Union(wo_copy_is, r);
        wo_copy_is.simplify(2, 4);
        
        
      }
      else {
        has_read_refs = true;
        ro_copy_is = Union(ro_copy_is, r);
        ro_copy_is.simplify(2, 4);
        
      }
    }
  }
  
  // simplify read and write footprint iteration space
  {
    if (allow_extra_read)
      ro_copy_is = SimpleHull(ro_copy_is, true, true);
    else
      ro_copy_is = ConvexRepresentation(ro_copy_is);
    
    wo_copy_is = ConvexRepresentation(wo_copy_is);
    if (wo_copy_is.number_of_conjuncts() > 1) {
      Relation t = SimpleHull(wo_copy_is, true, true);
      if (Must_Be_Subset(copy(t), copy(ro_copy_is)))
        wo_copy_is = t;
      else if (Must_Be_Subset(copy(wo_copy_is), copy(ro_copy_is)))
        wo_copy_is = ro_copy_is;
    }
  }
  
  // make copy statement variable names match the ones in the original statements which
  // already have the same names due to apply_xform
  {
    int ref_stmt = *active.begin();
    for (std::set<int>::iterator i = active.begin(); i != active.end(); i++)
      if (stmt[*i].IS.n_set() > stmt[ref_stmt].IS.n_set())
        ref_stmt = *i;
    for (int i = 1; i < level; i++) {
      std::string s = stmt[ref_stmt].IS.input_var(i)->name();
      wo_copy_is.name_set_var(i, s);
      ro_copy_is.name_set_var(i, s);
    }
    for (int i = 0; i < privatized_levels.size(); i++) {
      std::string s = stmt[ref_stmt].IS.input_var(privatized_levels[i])->name();
      wo_copy_is.name_set_var(level+i, s);
      ro_copy_is.name_set_var(level+i, s);
    }
    for (int i = level+privatized_levels.size(); i < level+privatized_levels.size()+n_dim; i++) {
      std::string s = tmp_loop_var_name_prefix + to_string(tmp_loop_var_name_counter+i-level-privatized_levels.size());
      wo_copy_is.name_set_var(i, s);
      ro_copy_is.name_set_var(i, s);
    }
    tmp_loop_var_name_counter += n_dim;
    wo_copy_is.setup_names();
    ro_copy_is.setup_names();
  }
  
  // build merged footprint iteration space for calculating temporary array size
  Relation copy_is = SimpleHull(Union(copy(ro_copy_is), copy(wo_copy_is)), true, true);
  
  // extract temporary array information
  CG_outputBuilder *ocg = ir->builder();
  std::vector<CG_outputRepr *> index_lb(n_dim); // initialized to NULL
  std::vector<coef_t> index_stride(n_dim);
  std::vector<bool> is_index_eq(n_dim, false);
  std::vector<std::pair<int, CG_outputRepr *> > index_sz(0);
  Relation reduced_copy_is = copy(copy_is);
  
  for (int i = 0; i < n_dim; i++) {
    if (i != 0)
      reduced_copy_is = Project(reduced_copy_is, level-1+privatized_levels.size()+i, Set_Var);
    Relation bound = get_loop_bound(reduced_copy_is, level-1+privatized_levels.size()+i);
    
    // extract stride
    std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(bound, bound.set_var(level-1+privatized_levels.size()+i+1));
    if (result.second != NULL)
      index_stride[i] = abs(result.first.get_coef(result.second))/gcd(abs(result.first.get_coef(result.second)), abs(result.first.get_coef(bound.set_var(level-1+privatized_levels.size()+i+1))));
    else
      index_stride[i] = 1;
    
    // check if this arary index requires loop
    Conjunct *c = bound.query_DNF()->single_conjunct();
    for (EQ_Iterator ei(c->EQs()); ei; ei++) {
      if ((*ei).has_wildcards())
        continue;
      
      int coef = (*ei).get_coef(bound.set_var(level-1+privatized_levels.size()+i+1));
      if (coef != 0) {
        int sign = 1;
        if (coef < 0) {
          coef = -coef;
          sign = -1;
        }
        
        CG_outputRepr *op = NULL;
        for (Constr_Vars_Iter ci(*ei); ci; ci++) {
          switch ((*ci).var->kind()) {
          case Input_Var:
          {
            if ((*ci).var != bound.set_var(level-1+privatized_levels.size()+i+1))
              if ((*ci).coef*sign == 1)
                op = ocg->CreateMinus(op, ocg->CreateIdent((*ci).var->name()));
              else if ((*ci).coef*sign == -1)
                op = ocg->CreatePlus(op, ocg->CreateIdent((*ci).var->name()));
              else if ((*ci).coef*sign > 1)
                op = ocg->CreateMinus(op, ocg->CreateTimes(ocg->CreateInt(abs((*ci).coef)), ocg->CreateIdent((*ci).var->name())));
              else // (*ci).coef*sign < -1
                op = ocg->CreatePlus(op, ocg->CreateTimes(ocg->CreateInt(abs((*ci).coef)), ocg->CreateIdent((*ci).var->name())));
            break;
          }
          case Global_Var:
          {
            Global_Var_ID g = (*ci).var->get_global_var();
            if ((*ci).coef*sign == 1)
              op = ocg->CreateMinus(op, ocg->CreateIdent(g->base_name()));
            else if ((*ci).coef*sign == -1)
              op = ocg->CreatePlus(op, ocg->CreateIdent(g->base_name()));
            else if ((*ci).coef*sign > 1)
              op = ocg->CreateMinus(op, ocg->CreateTimes(ocg->CreateInt(abs((*ci).coef)), ocg->CreateIdent(g->base_name())));
            else // (*ci).coef*sign < -1
              op = ocg->CreatePlus(op, ocg->CreateTimes(ocg->CreateInt(abs((*ci).coef)), ocg->CreateIdent(g->base_name())));
            break;
          }
          default:
            throw loop_error("unsupported array index expression");
          }
        }
        if ((*ei).get_const() != 0)
          op = ocg->CreatePlus(op, ocg->CreateInt(-sign*((*ei).get_const())));
        if (coef != 1)
          op = ocg->CreateIntegerFloor(op, ocg->CreateInt(coef));
        
        index_lb[i] = op;
        is_index_eq[i] = true;
        break;
      }
    }
    if (is_index_eq[i])
      continue;
    
    // seperate lower and upper bounds
    std::vector<GEQ_Handle> lb_list, ub_list;
    std::set<Variable_ID> excluded_floor_vars;
    excluded_floor_vars.insert(bound.set_var(level-1+privatized_levels.size()+i+1));
    for (GEQ_Iterator gi(c->GEQs()); gi; gi++) {
      int coef = (*gi).get_coef(bound.set_var(level-1+privatized_levels.size()+i+1));
      if (coef != 0 && (*gi).has_wildcards()) {
        bool clean_bound = true;
        GEQ_Handle h;
        for (Constr_Vars_Iter cvi(*gi, true); gi; gi++)
          if (!find_floor_definition(bound, (*cvi).var, excluded_floor_vars).first) {
            clean_bound = false;
            break;
          }
        if (!clean_bound)
          continue;
      }
      
      if (coef > 0)
        lb_list.push_back(*gi);
      else if (coef < 0)
        ub_list.push_back(*gi);
    }
    if (lb_list.size() == 0 || ub_list.size() == 0)
      throw loop_error("failed to calcuate array footprint size");
    
    // build lower bound representation
    std::vector<CG_outputRepr *> lb_repr_list;
    for (int j = 0; j < lb_list.size(); j++){
      if(this->known.n_set() == 0)
        lb_repr_list.push_back(output_lower_bound_repr(ocg, lb_list[j], bound.set_var(level-1+privatized_levels.size()+i+1), result.first, result.second, bound, Relation::True(bound.n_set()), std::vector<std::pair<CG_outputRepr *, int> >(bound.n_set(), std::make_pair(static_cast<CG_outputRepr *>(NULL), 0))));
      else
        lb_repr_list.push_back(output_lower_bound_repr(ocg, lb_list[j], bound.set_var(level-1+privatized_levels.size()+i+1), result.first, result.second, bound, this->known, std::vector<std::pair<CG_outputRepr *, int> >(bound.n_set(), std::make_pair(static_cast<CG_outputRepr *>(NULL), 0))));
    }
    if (lb_repr_list.size() > 1)
      index_lb[i] = ocg->CreateInvoke("max", lb_repr_list);
    else if (lb_repr_list.size() == 1)
      index_lb[i] = lb_repr_list[0];
    
    // build temporary array size representation
    {
      Relation cal(copy_is.n_set(), 1);
      F_And *f_root = cal.add_and();
      for (int j = 0; j < ub_list.size(); j++)
        for (int k = 0; k < lb_list.size(); k++) {
          GEQ_Handle h = f_root->add_GEQ();
          
          for (Constr_Vars_Iter ci(ub_list[j]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var:
            {
              int pos = (*ci).var->get_position();
              h.update_coef(cal.input_var(pos), (*ci).coef);
              break;
            }
            case Global_Var:
            {
              Global_Var_ID g = (*ci).var->get_global_var();
              Variable_ID v;
              if (g->arity() == 0)
                v = cal.get_local(g);
              else
                v = cal.get_local(g, (*ci).var->function_of());
              h.update_coef(v, (*ci).coef);
              break;
            }
            default:
              throw loop_error("cannot calculate temporay array size statically");
            }
          }
          h.update_const(ub_list[j].get_const());
          
          for (Constr_Vars_Iter ci(lb_list[k]); ci; ci++) {
            switch ((*ci).var->kind()) {
            case Input_Var:
            {
              int pos = (*ci).var->get_position();
              h.update_coef(cal.input_var(pos), (*ci).coef);
              break;
            }
            case Global_Var:
            {
              Global_Var_ID g = (*ci).var->get_global_var();
              Variable_ID v;
              if (g->arity() == 0)
                v = cal.get_local(g);
              else
                v = cal.get_local(g, (*ci).var->function_of());
              h.update_coef(v, (*ci).coef);
              break;
            }
            default:
              throw loop_error("cannot calculate temporay array size statically");
            }
          }
          h.update_const(lb_list[k].get_const());
          
          h.update_const(1);
          h.update_coef(cal.output_var(1), -1);
        }
      
      cal = Restrict_Domain(cal, copy(copy_is));
      for (int j = 1; j <= cal.n_inp(); j++)
        cal = Project(cal, j, Input_Var);
      cal.simplify();
      
      // pad temporary array size
      // TODO: for variable array size, create padding formula
      Conjunct *c = cal.query_DNF()->single_conjunct();
      bool is_index_bound_const = false;
      for (GEQ_Iterator gi(c->GEQs()); gi && !is_index_bound_const; gi++)
        if ((*gi).is_const(cal.output_var(1))) {
          coef_t size = (*gi).get_const() / (-(*gi).get_coef(cal.output_var(1)));
          if (padding_stride != 0) {
            size = (size + index_stride[i] - 1) / index_stride[i];
            if (i == fastest_changing_dimension)
              size = size * padding_stride;
          }
          if (i == fastest_changing_dimension) {
            if (padding_alignment > 1) { // align to boundary for data packing
              int residue = size % padding_alignment;
              if (residue)
                size = size+padding_alignment-residue;
            }
            else if (padding_alignment < -1) {  // un-alignment for memory bank conflicts
              while (gcd(size, static_cast<coef_t>(-padding_alignment)) != 1)
                size++;
            }
          }
          index_sz.push_back(std::make_pair(i, ocg->CreateInt(size)));
          is_index_bound_const = true;
        }
      
      if (!is_index_bound_const) {
        for (GEQ_Iterator gi(c->GEQs()); gi && !is_index_bound_const; gi++) {
          int coef = (*gi).get_coef(cal.output_var(1));
          if (coef < 0) {
            CG_outputRepr *op = NULL;
            for (Constr_Vars_Iter ci(*gi); ci; ci++) {
              if ((*ci).var != cal.output_var(1)) {
                switch((*ci).var->kind()) {
                case Global_Var:
                {
                  Global_Var_ID g = (*ci).var->get_global_var();
                  if ((*ci).coef == 1)
                    op = ocg->CreatePlus(op, ocg->CreateIdent(g->base_name()));
                  else if ((*ci).coef == -1)
                    op = ocg->CreateMinus(op, ocg->CreateIdent(g->base_name()));
                  else if ((*ci).coef > 1)
                    op = ocg->CreatePlus(op, ocg->CreateTimes(ocg->CreateInt((*ci).coef), ocg->CreateIdent(g->base_name())));
                  else // (*ci).coef < -1
                    op = ocg->CreateMinus(op, ocg->CreateTimes(ocg->CreateInt(-(*ci).coef), ocg->CreateIdent(g->base_name())));
                  break;
                }
                default:
                  throw loop_error("failed to generate array index bound code");
                }
              }
            }
            int c = (*gi).get_const();
            if (c > 0)
              op = ocg->CreatePlus(op, ocg->CreateInt(c));
            else if (c < 0)
              op = ocg->CreateMinus(op, ocg->CreateInt(-c));
            if (padding_stride != 0) {
              if (i == fastest_changing_dimension) {
                coef_t g = gcd(index_stride[i], static_cast<coef_t>(padding_stride));
                coef_t t1 = index_stride[i] / g;
                if (t1 != 1)
                  op = ocg->CreateIntegerFloor(ocg->CreatePlus(op, ocg->CreateInt(t1-1)), ocg->CreateInt(t1));
                coef_t t2 = padding_stride / g;
                if (t2 != 1)
                  op = ocg->CreateTimes(op, ocg->CreateInt(t2));
              }
              else if (index_stride[i] != 1) {
                op = ocg->CreateIntegerFloor(ocg->CreatePlus(op, ocg->CreateInt(index_stride[i]-1)), ocg->CreateInt(index_stride[i]));
              }
            }
            
            index_sz.push_back(std::make_pair(i, op));
            break;
          }
        }
      }
    }
  }
  
  // change the temporary array index order
  for (int i = 0; i < index_sz.size(); i++)
    if (index_sz[i].first == fastest_changing_dimension)
      switch (sym->layout_type()) {
      case IR_ARRAY_LAYOUT_ROW_MAJOR:
        std::swap(index_sz[index_sz.size()-1], index_sz[i]);
        break;
      case IR_ARRAY_LAYOUT_COLUMN_MAJOR:
        std::swap(index_sz[0], index_sz[i]);
        break;
      default:
        throw loop_error("unsupported array layout");
      }
  
  // declare temporary array or scalar
  IR_Symbol *tmp_sym;
  if (index_sz.size() == 0) {
    tmp_sym = ir->CreateScalarSymbol(sym, memory_type);
  }
  else {
    std::vector<CG_outputRepr *> tmp_array_size(index_sz.size());
    for (int i = 0; i < index_sz.size(); i++)
      tmp_array_size[i] = index_sz[i].second->clone();
    tmp_sym = ir->CreateArraySymbol(sym, tmp_array_size, memory_type);
  }
  
  // create temporary array read initialization code
  CG_outputRepr *copy_code_read;
  if (has_read_refs)
    if (index_sz.size() == 0) {
      IR_ScalarRef *tmp_scalar_ref = ir->CreateScalarRef(static_cast<IR_ScalarSymbol *>(tmp_sym));
      
      std::vector<CG_outputRepr *> rhs_index(n_dim);
      for (int i = 0; i < index_lb.size(); i++)
        if (is_index_eq[i])
          rhs_index[i] = index_lb[i]->clone();
        else
          rhs_index[i] = ir->builder()->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+i+1)->name());
      IR_ArrayRef *copied_array_ref = ir->CreateArrayRef(sym, rhs_index);
      
      copy_code_read = ir->builder()->CreateAssignment(0, tmp_scalar_ref->convert(), copied_array_ref->convert());
    }
    else {
      std::vector<CG_outputRepr *> lhs_index(index_sz.size());
      for (int i = 0; i < index_sz.size(); i++) {
        int cur_index_num = index_sz[i].first;
        CG_outputRepr *cur_index_repr = ocg->CreateMinus(ocg->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+cur_index_num+1)->name()), index_lb[cur_index_num]->clone());
        if (padding_stride != 0) {
          if (i == n_dim-1) {
            coef_t g = gcd(index_stride[cur_index_num], static_cast<coef_t>(padding_stride));
            coef_t t1 = index_stride[cur_index_num] / g;
            if (t1 != 1)
              cur_index_repr = ocg->CreateIntegerFloor(cur_index_repr, ocg->CreateInt(t1));
            coef_t t2 = padding_stride / g;
            if (t2 != 1)
              cur_index_repr = ocg->CreateTimes(cur_index_repr, ocg->CreateInt(t2));
          }
          else if (index_stride[cur_index_num] != 1) {
            cur_index_repr = ocg->CreateIntegerFloor(cur_index_repr, ocg->CreateInt(index_stride[cur_index_num]));
          }
        }
        
        if (ir->ArrayIndexStartAt() != 0)
          cur_index_repr = ocg->CreatePlus(cur_index_repr, ocg->CreateInt(ir->ArrayIndexStartAt()));
        lhs_index[i] = cur_index_repr;
      }
      
      IR_ArrayRef *tmp_array_ref = ir->CreateArrayRef(static_cast<IR_ArraySymbol *>(tmp_sym), lhs_index);
      
      std::vector<CG_outputRepr *> rhs_index(n_dim);
      for (int i = 0; i < index_lb.size(); i++)
        if (is_index_eq[i])
          rhs_index[i] = index_lb[i]->clone();
        else
          rhs_index[i] = ir->builder()->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+i+1)->name());
      IR_ArrayRef *copied_array_ref = ir->CreateArrayRef(sym, rhs_index);
      
      copy_code_read = ir->builder()->CreateAssignment(0, tmp_array_ref->convert(), copied_array_ref->convert());
    }
  
  // create temporary array write back code
  CG_outputRepr *copy_code_write;
  if (has_write_refs)
    if (index_sz.size() == 0) {
      IR_ScalarRef *tmp_scalar_ref = ir->CreateScalarRef(static_cast<IR_ScalarSymbol *>(tmp_sym));
      
      std::vector<CG_outputRepr *> rhs_index(n_dim);
      for (int i = 0; i < index_lb.size(); i++)
        if (is_index_eq[i])
          rhs_index[i] = index_lb[i]->clone();
        else
          rhs_index[i] = ir->builder()->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+i+1)->name());
      IR_ArrayRef *copied_array_ref = ir->CreateArrayRef(sym, rhs_index);
      
      copy_code_write = ir->builder()->CreateAssignment(0, copied_array_ref->convert(), tmp_scalar_ref->convert());
    }
    else {
      std::vector<CG_outputRepr *> lhs_index(n_dim);
      for (int i = 0; i < index_lb.size(); i++)
        if (is_index_eq[i])
          lhs_index[i] = index_lb[i]->clone();
        else
          lhs_index[i] = ir->builder()->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+i+1)->name());
      IR_ArrayRef *copied_array_ref = ir->CreateArrayRef(sym, lhs_index);
      
      std::vector<CG_outputRepr *> rhs_index(index_sz.size());
      for (int i = 0; i < index_sz.size(); i++) {
        int cur_index_num = index_sz[i].first;
        CG_outputRepr *cur_index_repr = ocg->CreateMinus(ocg->CreateIdent(copy_is.set_var(level-1+privatized_levels.size()+cur_index_num+1)->name()), index_lb[cur_index_num]->clone());
        if (padding_stride != 0) {
          if (i == n_dim-1) {
            coef_t g = gcd(index_stride[cur_index_num], static_cast<coef_t>(padding_stride));
            coef_t t1 = index_stride[cur_index_num] / g;
            if (t1 != 1)
              cur_index_repr = ocg->CreateIntegerFloor(cur_index_repr, ocg->CreateInt(t1));
            coef_t t2 = padding_stride / g;
            if (t2 != 1)
              cur_index_repr = ocg->CreateTimes(cur_index_repr, ocg->CreateInt(t2));
          }
          else if (index_stride[cur_index_num] != 1) {
            cur_index_repr = ocg->CreateIntegerFloor(cur_index_repr, ocg->CreateInt(index_stride[cur_index_num]));
          }
        }
        
        if (ir->ArrayIndexStartAt() != 0)
          cur_index_repr = ocg->CreatePlus(cur_index_repr, ocg->CreateInt(ir->ArrayIndexStartAt()));
        rhs_index[i] = cur_index_repr;
      }
      IR_ArrayRef *tmp_array_ref = ir->CreateArrayRef(static_cast<IR_ArraySymbol *>(tmp_sym), rhs_index);
      
      copy_code_write = ir->builder()->CreateAssignment(0, copied_array_ref->convert(), tmp_array_ref->convert());
    }
  
  // now we can remove those loops for array indexes that are
  // dependent on others
  if (!(index_sz.size() == n_dim && (sym->layout_type() == IR_ARRAY_LAYOUT_ROW_MAJOR || n_dim <= 1))) {
    Relation mapping(level-1+privatized_levels.size()+n_dim, level-1+privatized_levels.size()+index_sz.size());
    F_And *f_root = mapping.add_and();
    for (int i = 1; i <= level-1+privatized_levels.size(); i++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.input_var(i), 1);
      h.update_coef(mapping.output_var(i), -1);
    }
    
    int cur_index = 0;
    std::vector<int> mapped_index(index_sz.size());
    for (int i = 0; i < n_dim; i++)
      if (!is_index_eq[i]) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(mapping.input_var(level-1+privatized_levels.size()+i+1), 1);
        switch (sym->layout_type()) {
        case IR_ARRAY_LAYOUT_COLUMN_MAJOR: {
          h.update_coef(mapping.output_var(level-1+privatized_levels.size()+index_sz.size()-cur_index), -1);
          mapped_index[index_sz.size()-cur_index-1] = i;
          break;
        }
        case IR_ARRAY_LAYOUT_ROW_MAJOR: {
          h.update_coef(mapping.output_var(level-1+privatized_levels.size()+cur_index+1), -1);
          mapped_index[cur_index] = i;
          break;
        }
        default:
          throw loop_error("unsupported array layout");
        }
        cur_index++;
      }
    
    wo_copy_is = Range(Restrict_Domain(copy(mapping), wo_copy_is));
    ro_copy_is = Range(Restrict_Domain(copy(mapping), ro_copy_is));
    for (int i = 1; i <= level-1+privatized_levels.size(); i++) {
      wo_copy_is.name_set_var(i, copy_is.set_var(i)->name());
      ro_copy_is.name_set_var(i, copy_is.set_var(i)->name());
    }
    for (int i = 0; i < index_sz.size(); i++) {
      wo_copy_is.name_set_var(level-1+privatized_levels.size()+i+1, copy_is.set_var(level-1+privatized_levels.size()+mapped_index[i]+1)->name());
      ro_copy_is.name_set_var(level-1+privatized_levels.size()+i+1, copy_is.set_var(level-1+privatized_levels.size()+mapped_index[i]+1)->name());
    }
    wo_copy_is.setup_names();
    ro_copy_is.setup_names();
  }
  
  // insert read copy statement
  int old_num_stmt = stmt.size();
  int ro_copy_stmt_num = -1;
  if (has_read_refs) {
    Relation copy_xform(ro_copy_is.n_set(), 2*ro_copy_is.n_set()+1);
    {
      F_And *f_root = copy_xform.add_and();
      for (int i = 1; i <= ro_copy_is.n_set(); i++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(copy_xform.input_var(i), 1);
        h.update_coef(copy_xform.output_var(2*i), -1);
      }
      for (int i = 1; i <= dim; i+=2) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(copy_xform.output_var(i), -1);
        h.update_const(lex[i-1]);
      }
      for (int i = dim+2; i <= copy_xform.n_out(); i+=2) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(copy_xform.output_var(i), 1);
      }
    }
    
    Statement copy_stmt_read;
    copy_stmt_read.IS = ro_copy_is;
    copy_stmt_read.xform = copy_xform;
    copy_stmt_read.code = copy_code_read;
    copy_stmt_read.loop_level = std::vector<LoopLevel>(ro_copy_is.n_set());
    copy_stmt_read.ir_stmt_node = NULL;
    for (int i = 0; i < level-1; i++) {
      copy_stmt_read.loop_level[i].type = stmt[*(active.begin())].loop_level[i].type;
      if (stmt[*(active.begin())].loop_level[i].type == LoopLevelTile &&
          stmt[*(active.begin())].loop_level[i].payload >= level) {
        int j;
        for (j = 0; j < privatized_levels.size(); j++)
          if (privatized_levels[j] == stmt[*(active.begin())].loop_level[i].payload)
            break;
        if (j == privatized_levels.size())
          copy_stmt_read.loop_level[i].payload = -1;
        else
          copy_stmt_read.loop_level[i].payload = level + j;
      }
      else
        copy_stmt_read.loop_level[i].payload = stmt[*(active.begin())].loop_level[i].payload;
      copy_stmt_read.loop_level[i].parallel_level = stmt[*(active.begin())].loop_level[i].parallel_level;
    }
    for (int i = 0; i < privatized_levels.size(); i++) {
      copy_stmt_read.loop_level[level-1+i].type = stmt[*(active.begin())].loop_level[privatized_levels[i]].type;
      copy_stmt_read.loop_level[level-1+i].payload = stmt[*(active.begin())].loop_level[privatized_levels[i]].payload;
      copy_stmt_read.loop_level[level-1+i].parallel_level = stmt[*(active.begin())].loop_level[privatized_levels[i]].parallel_level;
    }
    int left_num_dim = num_dep_dim - (get_last_dep_dim_before(*(active.begin()), level) + 1);
    for (int i = 0; i < min(left_num_dim, static_cast<int>(index_sz.size())); i++) {
      copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].type = LoopLevelOriginal;
      copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].payload = num_dep_dim-left_num_dim+i;
      copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].parallel_level = 0;
    }
    for (int i = min(left_num_dim, static_cast<int>(index_sz.size())); i < index_sz.size(); i++) {
      copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].type = LoopLevelUnknown;
      copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].payload = -1;
      copy_stmt_read.loop_level[level-1+privatized_levels.size()+i].parallel_level = 0;
    }
    
    
    shiftLexicalOrder(lex, dim-1, 1);
    stmt.push_back(copy_stmt_read);
    ro_copy_stmt_num = stmt.size() - 1;
    dep.insert();
  }
  
  // insert write copy statement
  int wo_copy_stmt_num = -1;
  if (has_write_refs) {
    Relation copy_xform(wo_copy_is.n_set(), 2*wo_copy_is.n_set()+1);
    {
      F_And *f_root = copy_xform.add_and();
      for (int i = 1; i <= wo_copy_is.n_set(); i++) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(copy_xform.input_var(i), 1);
        h.update_coef(copy_xform.output_var(2*i), -1);
      }
      for (int i = 1; i <= dim; i+=2) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(copy_xform.output_var(i), -1);
        h.update_const(lex[i-1]);
      }
      for (int i = dim+2; i <= copy_xform.n_out(); i+=2) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(copy_xform.output_var(i), 1);
      }
    }
    
    Statement copy_stmt_write;
    copy_stmt_write.IS = wo_copy_is;
    copy_stmt_write.xform = copy_xform;
    copy_stmt_write.code = copy_code_write;
    copy_stmt_write.loop_level = std::vector<LoopLevel>(wo_copy_is.n_set());
    copy_stmt_write.ir_stmt_node = NULL;
    
    for (int i = 0; i < level-1; i++) {
      copy_stmt_write.loop_level[i].type = stmt[*(active.begin())].loop_level[i].type;
      if (stmt[*(active.begin())].loop_level[i].type == LoopLevelTile &&
          stmt[*(active.begin())].loop_level[i].payload >= level) {
        int j;
        for (j = 0; j < privatized_levels.size(); j++)
          if (privatized_levels[j] == stmt[*(active.begin())].loop_level[i].payload)
            break;
        if (j == privatized_levels.size())
          copy_stmt_write.loop_level[i].payload = -1;
        else
          copy_stmt_write.loop_level[i].payload = level + j;
      }
      else
        copy_stmt_write.loop_level[i].payload = stmt[*(active.begin())].loop_level[i].payload;
      copy_stmt_write.loop_level[i].parallel_level = stmt[*(active.begin())].loop_level[i].parallel_level;
    }
    for (int i = 0; i < privatized_levels.size(); i++) {
      copy_stmt_write.loop_level[level-1+i].type = stmt[*(active.begin())].loop_level[privatized_levels[i]].type;
      copy_stmt_write.loop_level[level-1+i].payload = stmt[*(active.begin())].loop_level[privatized_levels[i]].payload;
      copy_stmt_write.loop_level[level-1+i].parallel_level = stmt[*(active.begin())].loop_level[privatized_levels[i]].parallel_level;
    }
    int left_num_dim = num_dep_dim - (get_last_dep_dim_before(*(active.begin()), level) + 1);
    for (int i = 0; i < min(left_num_dim, static_cast<int>(index_sz.size())); i++) {
      copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].type = LoopLevelOriginal;
      copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].payload = num_dep_dim-left_num_dim+i;
      copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].parallel_level = 0;
    }
    for (int i = min(left_num_dim, static_cast<int>(index_sz.size())); i < index_sz.size(); i++) {
      copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].type = LoopLevelUnknown;
      copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].payload = -1;
      copy_stmt_write.loop_level[level-1+privatized_levels.size()+i].parallel_level = 0;
    }
    lex[dim-1]++;
    shiftLexicalOrder(lex, dim-1, -2);
    stmt.push_back(copy_stmt_write);
    wo_copy_stmt_num = stmt.size() - 1;
    dep.insert();
  }
  
  // replace original array accesses with temporary array accesses
  for (int i =0; i < stmt_refs.size(); i++)
    for (int j = 0; j < stmt_refs[i].second.size(); j++) {
      if (index_sz.size() == 0) {
        IR_ScalarRef *tmp_scalar_ref = ir->CreateScalarRef(static_cast<IR_ScalarSymbol *>(tmp_sym));
        ir->ReplaceExpression(stmt_refs[i].second[j], tmp_scalar_ref->convert());
      }
      else {
        std::vector<CG_outputRepr *> index_repr(index_sz.size());
        for (int k = 0; k < index_sz.size(); k++) {
          int cur_index_num = index_sz[k].first;
          
          CG_outputRepr *cur_index_repr = ocg->CreateMinus(stmt_refs[i].second[j]->index(cur_index_num), index_lb[cur_index_num]->clone());
          if (padding_stride != 0) {
            if (k == n_dim-1) {
              coef_t g = gcd(index_stride[cur_index_num], static_cast<coef_t>(padding_stride));
              coef_t t1 = index_stride[cur_index_num] / g;
              if (t1 != 1)
                cur_index_repr = ocg->CreateIntegerFloor(cur_index_repr, ocg->CreateInt(t1));
              coef_t t2 = padding_stride / g;
              if (t2 != 1)
                cur_index_repr = ocg->CreateTimes(cur_index_repr, ocg->CreateInt(t2));
            }
            else if (index_stride[cur_index_num] != 1) {
              cur_index_repr = ocg->CreateIntegerFloor(cur_index_repr, ocg->CreateInt(index_stride[cur_index_num]));
            }
          }
          
          if (ir->ArrayIndexStartAt() != 0)
            cur_index_repr = ocg->CreatePlus(cur_index_repr, ocg->CreateInt(ir->ArrayIndexStartAt()));
          index_repr[k] = cur_index_repr;
        }
        
        IR_ArrayRef *tmp_array_ref = ir->CreateArrayRef(static_cast<IR_ArraySymbol *>(tmp_sym), index_repr);
        ir->ReplaceExpression(stmt_refs[i].second[j], tmp_array_ref->convert());
      }
    }
  
  // update dependence graph
  int dep_dim = get_last_dep_dim_before(*(active.begin()), level) + 1;
  if (ro_copy_stmt_num != -1) {
    for (int i = 0; i < old_num_stmt; i++) {
      std::vector<std::vector<DependenceVector> > D;
      
      for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end();) {
        if (active.find(i) != active.end() && active.find(j->first) == active.end()) {
          std::vector<DependenceVector> dvs1, dvs2;
          for (int k = 0; k < j->second.size(); k++) {
            DependenceVector dv = j->second[k];
            if (dv.sym != NULL && dv.sym->name() == sym->name() && (dv.type == DEP_R2R || dv.type == DEP_R2W))
              dvs1.push_back(dv);
            else
              dvs2.push_back(dv);
          }
          j->second = dvs2;
          if (dvs1.size() > 0)
            dep.connect(ro_copy_stmt_num, j->first, dvs1);
        }
        else if (active.find(i) == active.end() && active.find(j->first) != active.end()) {
          std::vector<DependenceVector> dvs1, dvs2;
          for (int k = 0; k < j->second.size(); k++) {
            DependenceVector dv = j->second[k];
            if (dv.sym != NULL && dv.sym->name() == sym->name() && (dv.type == DEP_R2R || dv.type == DEP_W2R))
              dvs1.push_back(dv);
            else
              dvs2.push_back(dv);
          }
          j->second = dvs2;
          if (dvs1.size() > 0)
            D.push_back(dvs1);
        }
        
        if (j->second.size() == 0)
          dep.vertex[i].second.erase(j++);
        else
          j++;
      }
      
      for (int j = 0; j < D.size(); j++)
        dep.connect(i, ro_copy_stmt_num, D[j]);
    }
    
    // insert dependences from copy statement loop to copied statements
    DependenceVector dv;
    dv.type = DEP_W2R;
    dv.sym = tmp_sym->clone();
    dv.lbounds = std::vector<coef_t>(dep.num_dim(), 0);
    dv.ubounds = std::vector<coef_t>(dep.num_dim(), 0);
    for (int i = dep_dim; i < dep.num_dim(); i++) {
      dv.lbounds[i] = -posInfinity;
      dv.ubounds[i] = posInfinity;
    }
    for (std::set<int>::iterator i = active.begin(); i != active.end(); i++)
      dep.connect(ro_copy_stmt_num, *i, dv);
  }
  
  if (wo_copy_stmt_num != -1) {
    for (int i = 0; i < old_num_stmt; i++) {
      std::vector<std::vector<DependenceVector> > D;
      
      for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end();) {
        if (active.find(i) != active.end() && active.find(j->first) == active.end()) {
          std::vector<DependenceVector> dvs1, dvs2;
          for (int k = 0; k < j->second.size(); k++) {
            DependenceVector dv = j->second[k];
            if (dv.sym != NULL && dv.sym->name() == sym->name() && (dv.type == DEP_W2R || dv.type == DEP_W2W))
              dvs1.push_back(dv);
            else
              dvs2.push_back(dv);
          }
          j->second = dvs2;
          if (dvs1.size() > 0)
            dep.connect(wo_copy_stmt_num, j->first, dvs1);
        }
        else if (active.find(i) == active.end() && active.find(j->first) != active.end()) {
          std::vector<DependenceVector> dvs1, dvs2;
          for (int k = 0; k < j->second.size(); k++) {
            DependenceVector dv = j->second[k];
            if (dv.sym != NULL && dv.sym->name() == sym->name() && (dv.type == DEP_R2W || dv.type == DEP_W2W))
              dvs1.push_back(dv);
            else
              dvs2.push_back(dv);
          }
          j->second = dvs2;
          if (dvs1.size() > 0)
            D.push_back(dvs1);
        }
        
        if (j->second.size() == 0)
          dep.vertex[i].second.erase(j++);
        else
          j++;
      }
      
      for (int j = 0; j < D.size(); j++)
        dep.connect(i, wo_copy_stmt_num, D[j]);
    }
    
    // insert dependences from copied statements to write statements
    DependenceVector dv;
    dv.type = DEP_W2R;
    dv.sym = tmp_sym->clone();
    dv.lbounds = std::vector<coef_t>(dep.num_dim(), 0);
    dv.ubounds = std::vector<coef_t>(dep.num_dim(), 0);
    for (int i = dep_dim; i < dep.num_dim(); i++) {
      dv.lbounds[i] = -posInfinity;
      dv.ubounds[i] = posInfinity;
    }
    for (std::set<int>::iterator i = active.begin(); i != active.end(); i++)
      dep.connect(*i, wo_copy_stmt_num, dv);
    
  }
  
  // update variable name for dependences among copied statements
  for (int i = 0; i < old_num_stmt; i++) {
    if (active.find(i) != active.end())
      for (DependenceGraph::EdgeList::iterator j = dep.vertex[i].second.begin(); j != dep.vertex[i].second.end(); j++)
        if (active.find(j->first) != active.end())
          for (int k = 0; k < j->second.size(); k++) {
            IR_Symbol *s = tmp_sym->clone();
            j->second[k].sym = s;
          }
  }
  
  // insert anti-dependence from write statement to read statement
  if (ro_copy_stmt_num != -1 && wo_copy_stmt_num != -1)
    if (dep_dim >= 0) {
      DependenceVector dv;
      dv.type = DEP_R2W;
      dv.sym = tmp_sym->clone();
      dv.lbounds = std::vector<coef_t>(dep.num_dim(), 0);
      dv.ubounds = std::vector<coef_t>(dep.num_dim(), 0);
      for (int k = dep_dim; k < dep.num_dim(); k++) {
        dv.lbounds[k] = -posInfinity;
        dv.ubounds[k] = posInfinity;
      }
      for (int k = 0; k < dep_dim; k++) {
        if (k != 0) {
          dv.lbounds[k-1] = 0;
          dv.ubounds[k-1] = 0;
        }
        dv.lbounds[k] = 1;
        dv.ubounds[k] = posInfinity;
        dep.connect(wo_copy_stmt_num, ro_copy_stmt_num, dv);
      }
    }
  
  // cleanup
  delete sym;
  delete tmp_sym;
  for (int i = 0; i < index_lb.size(); i++) {
    index_lb[i]->clear();
    delete index_lb[i];
  }
  for (int i = 0; i < index_sz.size(); i++) {
    index_sz[i].second->clear();
    delete index_sz[i].second;
  }
  
  return true;
}
