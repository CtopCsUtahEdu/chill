/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
 CG node classes, used to build AST tree from polyhedra scanning.

 Notes:
 Parameter "restriction" is always tighter than "known" since CG_split
 node does not correspond to any code for enforcement. This property is
 destroyed after hoistGuard since "restriction" is not used anymore.
 CG node's children are guaranteed not to be NULL, either NULL child is
 removed from the children or the parent node itself becomes NULL.

 History:
 04/20/96 printRepr added by D people. Lei Zhou
 10/24/06 hoistGuard added by chun
 08/03/10 collect CG classes into one place, by Chun Chen
 08/04/10 track dynamically substituted variables in printRepr, by chun
 04/02/11 rewrite the CG node classes, by chun
 *****************************************************************************/

#include <typeinfo>
#include <assert.h>
#include <omega.h>
#include <code_gen/codegen.h>
#include <code_gen/CG.h>
#include <code_gen/CG_outputBuilder.h>
#include <code_gen/CG_stringBuilder.h>
#include <code_gen/CG_utils.h>
#include <code_gen/codegen_error.h>
#include <stack>
#include <string.h>

namespace omega {

extern std::vector<std::vector<int> > smtNonSplitLevels;
extern std::vector<std::vector<std::string> > loopIdxNames; //per stmt
extern std::vector<std::pair<int, std::string> > syncs;

extern int checkLoopLevel;
extern int stmtForLoopCheck;
extern int upperBoundForLevel;
extern int lowerBoundForLevel;
extern bool fillInBounds;

//-----------------------------------------------------------------------------
// Class: CG_result
//-----------------------------------------------------------------------------

CG_outputRepr *CG_result::printRepr(CG_outputBuilder *ocg,
		const std::vector<CG_outputRepr *> &stmts) const {
	return printRepr(1, ocg, stmts,
			std::vector<std::pair<CG_outputRepr *, int> >(num_level(),
					std::make_pair(static_cast<CG_outputRepr *>(NULL), 0)));
}

std::string CG_result::printString() const {
	CG_stringBuilder ocg;
	std::vector<CG_outputRepr *> stmts(codegen_->xforms_.size());
	for (int i = 0; i < stmts.size(); i++)
		stmts[i] = new CG_stringRepr("s" + to_string(i));
	CG_stringRepr *repr = static_cast<CG_stringRepr *>(printRepr(&ocg, stmts));
	for (int i = 0; i < stmts.size(); i++)
		delete stmts[i];

	if (repr != NULL) {
		std::string s = repr->GetString();
		delete repr;
		return s;
	} else
		return std::string();
}

int CG_result::num_level() const {
	return codegen_->num_level();
}

//-----------------------------------------------------------------------------
// Class: CG_split
//-----------------------------------------------------------------------------

CG_result *CG_split::recompute(const BoolSet<> &parent_active,
		const Relation &known, const Relation &restriction) {
	active_ &= parent_active;
	if (active_.empty()) {
		delete this;
		return NULL;
	}


	int i = 0;
	while (i < restrictions_.size()) {
		Relation new_restriction = Intersection(copy(restrictions_[i]),
				copy(restriction));

		new_restriction.simplify(2, 4);
		//new_restriction.simplify();
		clauses_[i] = clauses_[i]->recompute(active_, copy(known),
				new_restriction);
		if (clauses_[i] == NULL) {
			restrictions_.erase(restrictions_.begin() + i);
			clauses_.erase(clauses_.begin() + i);
		} else
			i++;
	}


	if (restrictions_.size() == 0) {
		delete this;
		return NULL;
	} else
		return this;
}

int CG_split::populateDepth() {
	int max_depth = 0;
	for (int i = 0; i < clauses_.size(); i++) {
		int t = clauses_[i]->populateDepth();
		if (t > max_depth)
			max_depth = t;
	}
	return max_depth;
}

std::pair<CG_result *, Relation> CG_split::liftOverhead(int depth,
		bool propagate_up) {
	for (int i = 0; i < clauses_.size();) {
		std::pair<CG_result *, Relation> result = clauses_[i]->liftOverhead(
				depth, propagate_up);
		if (result.first == NULL)
			clauses_.erase(clauses_.begin() + i);
		else {
			clauses_[i] = result.first;
			if (!result.second.is_obvious_tautology())
				return std::make_pair(this, result.second);
			i++;
		}

	}

	if (clauses_.size() == 0) {
		delete this;
		return std::make_pair(static_cast<CG_result *>(NULL),
				Relation::True(num_level()));
	} else
		return std::make_pair(this, Relation::True(num_level()));
}

Relation CG_split::hoistGuard() {
	std::vector<Relation> guards;
	for (int i = 0; i < clauses_.size(); i++)
		guards.push_back(clauses_[i]->hoistGuard());

	return SimpleHull(guards, true, true);
}

void CG_split::removeGuard(const Relation &guard) {
	for (int i = 0; i < clauses_.size(); i++)
		clauses_[i]->removeGuard(guard);
}

std::vector<CG_result *> CG_split::findNextLevel() const {
	std::vector<CG_result *> result;
	for (int i = 0; i < clauses_.size(); i++) {
		CG_split *splt = dynamic_cast<CG_split *>(clauses_[i]);
		if (splt != NULL) {
			std::vector<CG_result *> t = splt->findNextLevel();
			result.insert(result.end(), t.begin(), t.end());
		} else
			result.push_back(clauses_[i]);
	}

	return result;
}

CG_outputRepr *CG_split::printRepr(int indent, CG_outputBuilder *ocg,
		const std::vector<CG_outputRepr *> &stmts,
		const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) const {
	CG_outputRepr *stmtList = NULL;
	std::vector<CG_result *> next_level = findNextLevel();

	std::vector<CG_loop *> cur_loops;
	for (int i = 0; i < next_level.size(); i++) {
		CG_loop *lp = dynamic_cast<CG_loop *>(next_level[i]);
		if (lp != NULL) {
			cur_loops.push_back(lp);
		} else {
			stmtList = ocg->StmtListAppend(stmtList,
					loop_print_repr(cur_loops, 0, cur_loops.size(),
							Relation::True(num_level()), NULL, indent, ocg,
							stmts, assigned_on_the_fly));
			stmtList = ocg->StmtListAppend(stmtList,
					next_level[i]->printRepr(indent, ocg, stmts,
							assigned_on_the_fly));
			cur_loops.clear();
		}
	}

	stmtList = ocg->StmtListAppend(stmtList,
			loop_print_repr(cur_loops, 0, cur_loops.size(),
					Relation::True(num_level()), NULL, indent, ocg, stmts,
					assigned_on_the_fly));
	return stmtList;
}

CG_result *CG_split::clone() const {
	std::vector<CG_result *> clauses(clauses_.size());
	for (int i = 0; i < clauses_.size(); i++)
		clauses[i] = clauses_[i]->clone();
	return new CG_split(codegen_, active_, restrictions_, clauses);
}

void CG_split::dump(int indent) const {
	std::string prefix;
	for (int i = 0; i < indent; i++)
		prefix += "  ";
	std::cout << prefix << "SPLIT: " << active_ << std::endl;
	for (int i = 0; i < restrictions_.size(); i++) {
		std::cout << prefix << "restriction: ";
		const_cast<CG_split *>(this)->restrictions_[i].print();
		clauses_[i]->dump(indent + 1);
	}

}

//-----------------------------------------------------------------------------
// Class: CG_loop
//-----------------------------------------------------------------------------

CG_result *CG_loop::recompute(const BoolSet<> &parent_active,
		const Relation &known, const Relation &restriction) {
	known_ = copy(known);
	restriction_ = copy(restriction);
	active_ &= parent_active;

	std::vector<Relation> Rs;
	for (BoolSet<>::iterator i = active_.begin(); i != active_.end(); i++) {
		Relation r = Intersection(copy(restriction),
				copy(codegen_->projected_IS_[level_ - 1][*i]));

		//r.simplify(2, 4);
		r.simplify();
		if (!r.is_upper_bound_satisfiable()) {
			active_.unset(*i);
			continue;
		}
		Rs.push_back(copy(r));
	}

	if (active_.empty()) {
		delete this;
		return NULL;
	}

	Relation hull = SimpleHull(Rs, true, true);

	//hull.simplify(2,4);

	// check if actual loop is needed
	std::pair<EQ_Handle, int> result = find_simplest_assignment(hull,
			hull.set_var(level_));
	if (result.second < INT_MAX) {
		needLoop_ = false;

		bounds_ = Relation(hull.n_set());
		F_Exists *f_exists = bounds_.add_and()->add_exists();
		F_And *f_root = f_exists->add_and();
		std::map<Variable_ID, Variable_ID> exists_mapping;
		EQ_Handle h = f_root->add_EQ();
		for (Constr_Vars_Iter cvi(result.first); cvi; cvi++) {
			Variable_ID v = cvi.curr_var();
			switch (v->kind()) {
			case Input_Var:
				h.update_coef(bounds_.input_var(v->get_position()),
						cvi.curr_coef());
				break;
			case Wildcard_Var: {
				Variable_ID v2 = replicate_floor_definition(hull, v, bounds_,
						f_exists, f_root, exists_mapping);
				h.update_coef(v2, cvi.curr_coef());
				break;
			}
			case Global_Var: {
				Global_Var_ID g = v->get_global_var();
				Variable_ID v2;
				if (g->arity() == 0)
					v2 = bounds_.get_local(g);
				else
					v2 = bounds_.get_local(g, v->function_of());
				h.update_coef(v2, cvi.curr_coef());
				break;
			}
			default:
				assert(false);
			}
		}
		h.update_const(result.first.get_const());
		bounds_.simplify();
	}
	// loop iterates more than once, extract bounds now
	else {
		needLoop_ = true;

		bounds_ = Relation(hull.n_set());
		F_Exists *f_exists = bounds_.add_and()->add_exists();
		F_And *f_root = f_exists->add_and();
		std::map<Variable_ID, Variable_ID> exists_mapping;

		Relation b = Gist(copy(hull), copy(known), 1);
		bool has_unresolved_bound = false;

		std::set<Variable_ID> excluded_floor_vars;
		excluded_floor_vars.insert(b.set_var(level_));
		for (GEQ_Iterator e(b.single_conjunct()->GEQs()); e; e++)
			if ((*e).get_coef(b.set_var(level_)) != 0) {
				bool is_bound = true;
				for (Constr_Vars_Iter cvi(*e, true); cvi; cvi++) {
					std::pair<bool, GEQ_Handle> result = find_floor_definition(
							b, cvi.curr_var(), excluded_floor_vars);
					if (!result.first) {
						is_bound = false;
						has_unresolved_bound = true;
						break;
					}
				}

				if (!is_bound)
					continue;

				GEQ_Handle h = f_root->add_GEQ();
				for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
					Variable_ID v = cvi.curr_var();
					switch (v->kind()) {
					case Input_Var:
						h.update_coef(bounds_.input_var(v->get_position()),
								cvi.curr_coef());
						break;
					case Wildcard_Var: {
						Variable_ID v2 = replicate_floor_definition(b, v,
								bounds_, f_exists, f_root, exists_mapping);
						h.update_coef(v2, cvi.curr_coef());
						break;
					}
					case Global_Var: {
						Global_Var_ID g = v->get_global_var();
						Variable_ID v2;
						if (g->arity() == 0)
							v2 = bounds_.get_local(g);
						else
							v2 = bounds_.get_local(g, v->function_of());
						h.update_coef(v2, cvi.curr_coef());
						break;
					}
					default:
						assert(false);
					}
				}
				h.update_const((*e).get_const());
			}

		if (has_unresolved_bound) {
			b = Approximate(b);
			b.simplify(2, 4);
			//Simplification of Hull
			hull = Approximate(hull);
			hull.simplify(2, 4);
			//end : Anand
			for (GEQ_Iterator e(b.single_conjunct()->GEQs()); e; e++)
				if ((*e).get_coef(b.set_var(level_)) != 0)
					f_root->add_GEQ(*e);
		}
		bounds_.simplify();
                hull.simplify(2,4);
		// Since current SimpleHull does not support max() upper bound or min() lower bound,
		// we have to forcefully split the loop when hull approximation does not return any bound.
		bool has_lb = false;
		bool has_ub = false;
		for (GEQ_Iterator e = bounds_.single_conjunct()->GEQs(); e; e++) {
			if ((*e).get_coef(bounds_.set_var(level_)) > 0)
				has_lb = true;
			else if ((*e).get_coef(bounds_.set_var(level_)) < 0)
				has_ub = true;
			if (has_lb && has_ub)
				break;
		}

		if (!has_lb) {
			for (int i = 0; i < Rs.size(); i++) {
				Relation r = Approximate(copy(Rs[i]));
				r.simplify(2, 4);
				for (GEQ_Iterator e = r.single_conjunct()->GEQs(); e; e++)
					if ((*e).get_coef(r.input_var(level_)) > 0) {
						Relation r2 = Relation::True(num_level());
						r2.and_with_GEQ(*e);
						r2.simplify();
						std::vector<Relation> restrictions(2);
						restrictions[0] = Complement(copy(r2));
						restrictions[0].simplify();
						restrictions[1] = r2;
						std::vector<CG_result *> clauses(2);
						clauses[0] = this;
						clauses[1] = this->clone();
						CG_result *cgr = new CG_split(codegen_, active_,
								restrictions, clauses);
						cgr = cgr->recompute(active_, copy(known),
								copy(restriction));
						return cgr;
					}
			}
			for (int i = 0; i < Rs.size(); i++) {
				Relation r = Approximate(copy(Rs[i]));
				r.simplify(2, 4);
				for (EQ_Iterator e = r.single_conjunct()->EQs(); e; e++)
					if ((*e).get_coef(r.input_var(level_)) != 0) {
						Relation r2 = Relation::True(num_level());
						r2.and_with_GEQ(*e);
						r2.simplify();
						std::vector<Relation> restrictions(2);
						if ((*e).get_coef(r.input_var(level_)) > 0) {
							restrictions[0] = Complement(copy(r2));
							restrictions[0].simplify();
							restrictions[1] = r2;
						} else {
							restrictions[0] = r2;
							restrictions[1] = Complement(copy(r2));
							restrictions[1].simplify();
						}
						std::vector<CG_result *> clauses(2);
						clauses[0] = this;
						clauses[1] = this->clone();
						CG_result *cgr = new CG_split(codegen_, active_,
								restrictions, clauses);
						cgr = cgr->recompute(active_, copy(known),
								copy(restriction));
						return cgr;
					}
			}
		} else if (!has_ub) {
			for (int i = 0; i < Rs.size(); i++) {
				Relation r = Approximate(copy(Rs[i]));
				r.simplify(2, 4);
				for (GEQ_Iterator e = r.single_conjunct()->GEQs(); e; e++)
					if ((*e).get_coef(r.input_var(level_)) < 0) {
						Relation r2 = Relation::True(num_level());
						r2.and_with_GEQ(*e);
						r2.simplify();
						std::vector<Relation> restrictions(2);
						restrictions[1] = Complement(copy(r2));
						restrictions[1].simplify();
						restrictions[0] = r2;
						std::vector<CG_result *> clauses(2);
						clauses[0] = this;
						clauses[1] = this->clone();
						CG_result *cgr = new CG_split(codegen_, active_,
								restrictions, clauses);
						cgr = cgr->recompute(active_, copy(known),
								copy(restriction));
						return cgr;
					}
			}
			for (int i = 0; i < Rs.size(); i++) {
				Relation r = Approximate(copy(Rs[i]));
				r.simplify(2, 4);
				for (EQ_Iterator e = r.single_conjunct()->EQs(); e; e++)
					if ((*e).get_coef(r.input_var(level_)) != 0) {
						Relation r2 = Relation::True(num_level());
						r2.and_with_GEQ(*e);
						r2.simplify();
						std::vector<Relation> restrictions(2);
						if ((*e).get_coef(r.input_var(level_)) > 0) {
							restrictions[0] = Complement(copy(r2));
							restrictions[0].simplify();
							restrictions[1] = r2;
						} else {
							restrictions[0] = r2;
							restrictions[1] = Complement(copy(r2));
							restrictions[1].simplify();
						}
						std::vector<CG_result *> clauses(2);
						clauses[0] = this;
						clauses[1] = this->clone();
						CG_result *cgr = new CG_split(codegen_, active_,
								restrictions, clauses);
						cgr = cgr->recompute(active_, copy(known),
								copy(restriction));
						return cgr;
					}
			}
		}

		if (!has_lb && !has_ub)
			throw codegen_error(
					"can't find any bound at loop level " + to_string(level_));
		else if (!has_lb)
			throw codegen_error(
					"can't find lower bound at loop level "
							+ to_string(level_));
		else if (!has_ub)
			throw codegen_error(
					"can't find upper bound at loop level "
							+ to_string(level_));
	}
	bounds_.copy_names(hull);
	bounds_.setup_names();

	// additional guard/stride condition extraction
	if (needLoop_) {
		Relation cur_known = Intersection(copy(bounds_), copy(known_));
		cur_known.simplify();
		hull = Gist(hull, copy(cur_known), 1);

		std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(hull,
				hull.set_var(level_));
		if (result.second != NULL)
			if (abs(result.first.get_coef(hull.set_var(level_))) == 1) {
				F_Exists *f_exists = bounds_.and_with_and()->add_exists();
				F_And *f_root = f_exists->add_and();
				std::map<Variable_ID, Variable_ID> exists_mapping;
				EQ_Handle h = f_root->add_EQ();
				for (Constr_Vars_Iter cvi(result.first); cvi; cvi++) {
					Variable_ID v = cvi.curr_var();
					switch (v->kind()) {
					case Input_Var:
						h.update_coef(bounds_.input_var(v->get_position()),
								cvi.curr_coef());
						break;
					case Wildcard_Var: {
						Variable_ID v2;
						if (v == result.second)
							v2 = f_exists->declare();
						else
							v2 = replicate_floor_definition(hull, v, bounds_,
									f_exists, f_root, exists_mapping);
						h.update_coef(v2, cvi.curr_coef());
						break;
					}
					case Global_Var: {
						Global_Var_ID g = v->get_global_var();
						Variable_ID v2;
						if (g->arity() == 0)
							v2 = bounds_.get_local(g);
						else
							v2 = bounds_.get_local(g, v->function_of());
						h.update_coef(v2, cvi.curr_coef());
						break;
					}
					default:
						assert(false);
					}
				}
				h.update_const(result.first.get_const());
			} else {
				// since gist is not powerful enough on modular constraints for now,
				// make an educated guess
				coef_t stride = abs(result.first.get_coef(result.second))
						/ gcd(abs(result.first.get_coef(result.second)),
								abs(
										result.first.get_coef(
												hull.set_var(level_))));

				Relation r1(hull.n_inp());
				F_Exists *f_exists = r1.add_and()->add_exists();
				F_And *f_root = f_exists->add_and();
				std::map<Variable_ID, Variable_ID> exists_mapping;
				EQ_Handle h = f_root->add_EQ();
				for (Constr_Vars_Iter cvi(result.first); cvi; cvi++) {
					Variable_ID v = cvi.curr_var();
					switch (v->kind()) {
					case Input_Var:
						h.update_coef(r1.input_var(v->get_position()),
								cvi.curr_coef());
						break;
					case Wildcard_Var: {
						Variable_ID v2;
						if (v == result.second)
							v2 = f_exists->declare();
						else
							v2 = replicate_floor_definition(hull, v, r1,
									f_exists, f_root, exists_mapping);
						h.update_coef(v2, cvi.curr_coef());
						break;
					}
					case Global_Var: {
						Global_Var_ID g = v->get_global_var();
						Variable_ID v2;
						if (g->arity() == 0)
							v2 = r1.get_local(g);
						else
							v2 = r1.get_local(g, v->function_of());
						h.update_coef(v2, cvi.curr_coef());
						break;
					}
					default:
						assert(false);
					}
				}
				h.update_const(result.first.get_const());
				r1.simplify();

				bool guess_success = false;
				for (GEQ_Iterator e(bounds_.single_conjunct()->GEQs()); e; e++)
					if ((*e).get_coef(bounds_.set_var(level_)) == 1) {
						Relation r2(hull.n_inp());
						F_Exists *f_exists = r2.add_and()->add_exists();
						F_And *f_root = f_exists->add_and();
						std::map<Variable_ID, Variable_ID> exists_mapping;
						EQ_Handle h = f_root->add_EQ();
						h.update_coef(f_exists->declare(), stride);
						for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
							Variable_ID v = cvi.curr_var();
							switch (v->kind()) {
							case Input_Var:
								h.update_coef(r2.input_var(v->get_position()),
										cvi.curr_coef());
								break;
							case Wildcard_Var: {
								Variable_ID v2 = replicate_floor_definition(
										hull, v, r2, f_exists, f_root,
										exists_mapping);
								h.update_coef(v2, cvi.curr_coef());
								break;
							}
							case Global_Var: {
								Global_Var_ID g = v->get_global_var();
								Variable_ID v2;
								if (g->arity() == 0)
									v2 = r2.get_local(g);
								else
									v2 = r2.get_local(g, v->function_of());
								h.update_coef(v2, cvi.curr_coef());
								break;
							}
							default:
								assert(false);
							}
						}
						h.update_const((*e).get_const());
						r2.simplify();

						if (Gist(copy(r1),
								Intersection(copy(cur_known), copy(r2)), 1).is_obvious_tautology()
								&& Gist(copy(r2),
										Intersection(copy(cur_known), copy(r1)),
										1).is_obvious_tautology()) {
							bounds_ = Intersection(bounds_, r2);
							bounds_.simplify();
							guess_success = true;
							break;
						}
					}

				// this is really a stride with non-unit coefficient for this loop variable
				if (!guess_success) {
					// TODO: for stride ax = b mod n it might be beneficial to
					//       generate modular linear equation solver code for
					//       runtime to get the starting position in printRepr,
					//       and stride would be n/gcd(|a|,n), thus this stride
					//       can be put into bounds_ too.
				}

			}

		hull = Project(hull, hull.set_var(level_));
		hull.simplify(2, 4);
		guard_ = Gist(hull, Intersection(copy(bounds_), copy(known_)), 1);
	}
	// don't generate guard for non-actual loop, postpone it. otherwise
	// redundant if-conditions might be generated since for-loop semantics
	// includes implicit comparison checking.  -- by chun 09/14/10
	else
		guard_ = Relation::True(num_level());
	guard_.copy_names(bounds_);
	guard_.setup_names();

        //guard_.simplify();  
	// recursively down the AST
	Relation new_known = Intersection(copy(known),
			Intersection(copy(bounds_), copy(guard_)));
	new_known.simplify(2, 4);
	Relation new_restriction = Intersection(copy(restriction),
			Intersection(copy(bounds_), copy(guard_)));
	new_restriction.simplify(2, 4);
	body_ = body_->recompute(active_, new_known, new_restriction);
	if (body_ == NULL) {
		delete this;
		return NULL;
	} else
		return this;
}

int CG_loop::populateDepth() {
	int depth = body_->populateDepth();
	if (needLoop_)
		depth_ = depth + 1;
	else
		depth_ = depth;
	return depth_;
}

std::pair<CG_result *, Relation> CG_loop::liftOverhead(int depth,
		bool propagate_up) {
	if (depth_ > depth) {
		assert(propagate_up == false);
		std::pair<CG_result *, Relation> result = body_->liftOverhead(depth,
				false);
		body_ = result.first;
		return std::make_pair(this, Relation::True(num_level()));
	} else { // (depth_ <= depth)
		if (propagate_up) {
			Relation r = pick_one_guard(guard_, level_);
			if (!r.is_obvious_tautology())
				return std::make_pair(this, r);
		}

		std::pair<CG_result *, Relation> result;
		if (propagate_up || needLoop_)
			result = body_->liftOverhead(depth, true);
		else
			result = body_->liftOverhead(depth, false);
		body_ = result.first;
		if (result.second.is_obvious_tautology())
			return std::make_pair(this, result.second);

		// loop is an assignment, replace this loop variable in overhead condition
		if (!needLoop_) {
			result.second = Intersection(result.second, copy(bounds_));
			result.second = Project(result.second,
					result.second.set_var(level_));
			result.second.simplify(2, 4);
		}


		int max_level = 0;
		bool has_wildcard = false;
		bool direction = true;
		for (EQ_Iterator e(result.second.single_conjunct()->EQs()); e; e++)
			if ((*e).has_wildcards()) {
				if (has_wildcard)
					assert(false);
				else
					has_wildcard = true;
				for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
					if (cvi.curr_var()->kind() == Input_Var
							&& cvi.curr_var()->get_position() > max_level)
						max_level = cvi.curr_var()->get_position();
			} else
				assert(false);

		if (!has_wildcard) {
			int num_simple_geq = 0;
			for (GEQ_Iterator e(result.second.single_conjunct()->GEQs()); e;
					e++)
				if (!(*e).has_wildcards()) {
					num_simple_geq++;
					for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
						if (cvi.curr_var()->kind() == Input_Var
								&& cvi.curr_var()->get_position() > max_level) {
							max_level = cvi.curr_var()->get_position();
							direction = (cvi.curr_coef() < 0) ? true : false;
						}
				} else {
					has_wildcard = true;
					for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
						if (cvi.curr_var()->kind() == Input_Var
								&& cvi.curr_var()->get_position() > max_level) {
							max_level = cvi.curr_var()->get_position();
						}
				}
			assert(
					(has_wildcard && num_simple_geq == 0) || (!has_wildcard && num_simple_geq == 1));
		}

		// check if this is the top loop level for splitting for this overhead
		if (!propagate_up || (has_wildcard && max_level == level_ - 1)
				|| (!has_wildcard && max_level == level_)) {
			std::vector<Relation> restrictions(2);
			std::vector<CG_result *> clauses(2);
			int saved_num_level = num_level();
			if (has_wildcard || direction) {
				restrictions[1] = Complement(copy(result.second));
				restrictions[1].simplify();
				clauses[1] = this->clone();
				restrictions[0] = result.second;
				clauses[0] = this;
			} else {
				restrictions[0] = Complement(copy(result.second));
				restrictions[0].simplify();
				clauses[0] = this->clone();
				restrictions[1] = result.second;
				clauses[1] = this;
			}
			CG_result *cgr = new CG_split(codegen_, active_, restrictions,
					clauses);
			CG_result *new_cgr = cgr->recompute(active_, copy(known_),
					copy(restriction_));
			new_cgr->populateDepth();
			assert(new_cgr==cgr);
			if (static_cast<CG_split *>(new_cgr)->clauses_.size() == 1)
				// infinite recursion detected, bail out
				return std::make_pair(new_cgr, Relation::True(saved_num_level));
			else
				return cgr->liftOverhead(depth, propagate_up);
		} else
			return std::make_pair(this, result.second);
	}
}

Relation CG_loop::hoistGuard() {

	Relation r = body_->hoistGuard();

	// TODO: should bookkeep catched contraints in loop output as enforced and check if anything missing
	// if (!Gist(copy(b), copy(enforced)).is_obvious_tautology()) {
	//   fprintf(stderr, "need to generate extra guard inside the loop\n");
	// }

	  if (!needLoop_)
	    r = Intersection(r, copy(bounds_));
	  r = Project(r, r.set_var(level_));
	  r = Gist(r, copy(known_), 1);

	  Relation eliminate_existentials_r;
	  Relation eliminate_existentials_known;

	  eliminate_existentials_r = copy(r);
	  if (!r.is_obvious_tautology()) {
		  eliminate_existentials_r = Approximate(copy(r));
		  eliminate_existentials_r.simplify(2,4);
		  eliminate_existentials_known = Approximate(copy(known_));
		  eliminate_existentials_known.simplify(2,4);

		  eliminate_existentials_r = Gist( eliminate_existentials_r, eliminate_existentials_known, 1);
	  }
          

	  if (!eliminate_existentials_r.is_obvious_tautology()) {
	 // if (!r.is_obvious_tautology()) {
	    body_->removeGuard(r);
	    guard_ = Intersection(guard_, copy(r));
	    guard_.simplify();
	  }

	  return guard_;

	//   return ifList;
	// }


}

void CG_loop::removeGuard(const Relation &guard) {
	known_ = Intersection(known_, copy(guard));
	known_.simplify();

	guard_ = Gist(guard_, copy(known_), 1);
	guard_.copy_names(known_);
	guard_.setup_names();
}

CG_outputRepr *CG_loop::printRepr(int indent, CG_outputBuilder *ocg,
		const std::vector<CG_outputRepr *> &stmts,
		const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) const {
	return printRepr(true, indent, ocg, stmts, assigned_on_the_fly);
}

CG_outputRepr *CG_loop::printRepr(bool do_print_guard, int indent,
		CG_outputBuilder *ocg, const std::vector<CG_outputRepr *> &stmts,
		const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) const {
	CG_outputRepr *guardRepr;
	if (do_print_guard)
		guardRepr = output_guard(ocg, guard_, assigned_on_the_fly);
	else
		guardRepr = NULL;

	Relation cur_known = Intersection(copy(known_), copy(guard_));
	cur_known.simplify();
	if (needLoop_) {

		if (checkLoopLevel)
			if (level_ == checkLoopLevel)
				if (active_.get(stmtForLoopCheck))
					fillInBounds = true;

		CG_outputRepr *ctrlRepr = output_loop(ocg, bounds_, level_, cur_known,
				assigned_on_the_fly);

		fillInBounds = false;

		CG_outputRepr *bodyRepr = body_->printRepr(
				(guardRepr == NULL) ? indent + 1 : indent + 2, ocg, stmts,
				assigned_on_the_fly);
		CG_outputRepr * loopRepr;

		if (guardRepr == NULL)
			loopRepr = ocg->CreateLoop(indent, ctrlRepr, bodyRepr);
		else
			loopRepr = ocg->CreateLoop(indent + 1, ctrlRepr, bodyRepr);

		if (!smtNonSplitLevels.empty()) {
			bool blockLoop = false;
			bool threadLoop = false;
			bool sync = false;
			int firstActiveStmt = -1;
			for (int s = 0; s < active_.size(); s++) {
				if (active_.get(s)) {
					if (firstActiveStmt < 0)
						firstActiveStmt = s;
					//We assume smtNonSplitLevels is only used to mark the first of
					//the block or thread loops to be reduced in CUDA-CHiLL. Here we
					//place some comments to help with final code generation.
					//int idx = smtNonSplitLevels[s].index(level_);

					if (s < smtNonSplitLevels.size()) {
						if (smtNonSplitLevels[s].size() > 0)
							if (smtNonSplitLevels[s][0] == level_) {
								blockLoop = true;
							}
						//Assume every stmt marked with a thread loop index also has a block loop idx
						if (smtNonSplitLevels[s].size() > 1)
							if (smtNonSplitLevels[s][1] == level_) {
								threadLoop = true;
							}
					}
				}
			}
			if (blockLoop && threadLoop) {
				fprintf(stderr,
						"Warning, have %d level more than once in smtNonSplitLevels\n",
						level_);
				threadLoop = false;
			}
			std::string preferredIdx;
			if (loopIdxNames.size()
					&& (level_ / 2) - 1 < loopIdxNames[firstActiveStmt].size())
				preferredIdx = loopIdxNames[firstActiveStmt][(level_ / 2) - 1];
			for (int s = 0; s < active_.size(); s++) {
				if (active_.get(s)) {
					for (int i = 0; i < syncs.size(); i++) {
						if (syncs[i].first == s
								&& strcmp(syncs[i].second.c_str(),
										preferredIdx.c_str()) == 0) {
							sync = true;
							//printf("FOUND SYNC\n");
						}

					}
				}
		
			}
			if (threadLoop || blockLoop || preferredIdx.length() != 0) {
				char buf[1024];
				std::string loop;
				if (blockLoop)
					loop = "blockLoop ";
				if (threadLoop)
					loop = "threadLoop ";
				if (preferredIdx.length() != 0 && sync) {
					sprintf(buf, "~cuda~ %spreferredIdx: %s sync", loop.c_str(),
							preferredIdx.c_str());
				} else if (preferredIdx.length() != 0) {
					sprintf(buf, "~cuda~ %spreferredIdx: %s", loop.c_str(),
							preferredIdx.c_str());
				} else {
					sprintf(buf, "~cuda~ %s", loop.c_str());
				}


				loopRepr = ocg->CreateAttribute(loopRepr, buf);
			}

		}
		if (guardRepr == NULL)
			return loopRepr;
		else
			return ocg->CreateIf(indent, guardRepr, loopRepr, NULL);
	} else {
		std::pair<CG_outputRepr *, std::pair<CG_outputRepr *, int> > result =
				output_assignment(ocg, bounds_, level_, cur_known,
						assigned_on_the_fly);
		guardRepr = ocg->CreateAnd(guardRepr, result.first);

		if (result.second.second < CodeGen::var_substitution_threshold) {
			std::vector<std::pair<CG_outputRepr *, int> > atof =
					assigned_on_the_fly;
			atof[level_ - 1] = result.second;
			CG_outputRepr *bodyRepr = body_->printRepr(
					(guardRepr == NULL) ? indent : indent + 1, ocg, stmts,
					atof);
			delete atof[level_ - 1].first;
			if (guardRepr == NULL)
				return bodyRepr;
			else
				return ocg->CreateIf(indent, guardRepr, bodyRepr, NULL);
		} else {
			CG_outputRepr *assignRepr = ocg->CreateAssignment(
					(guardRepr == NULL) ? indent : indent + 1,
					output_ident(ocg, bounds_,
							const_cast<CG_loop *>(this)->bounds_.set_var(
									level_), assigned_on_the_fly),
					result.second.first);
			CG_outputRepr *bodyRepr = body_->printRepr(
					(guardRepr == NULL) ? indent : indent + 1, ocg, stmts,
					assigned_on_the_fly);
			if (guardRepr == NULL)
				return ocg->StmtListAppend(assignRepr, bodyRepr);
			else
				return ocg->CreateIf(indent, guardRepr,
						ocg->StmtListAppend(assignRepr, bodyRepr), NULL);
		}

	}
}

CG_result *CG_loop::clone() const {
	return new CG_loop(codegen_, active_, level_, body_->clone());
}

void CG_loop::dump(int indent) const {
	std::string prefix;
	for (int i = 0; i < indent; i++)
		prefix += "  ";
	std::cout << prefix << "LOOP (level " << level_ << "): " << active_
			<< std::endl;
	std::cout << prefix << "known: ";
	const_cast<CG_loop *>(this)->known_.print();
	std::cout << prefix << "restriction: ";
	const_cast<CG_loop *>(this)->restriction_.print();
	std::cout << prefix << "bounds: ";
	const_cast<CG_loop *>(this)->bounds_.print();
	std::cout << prefix << "guard: ";
	const_cast<CG_loop *>(this)->guard_.print();
	body_->dump(indent + 1);
}

//-----------------------------------------------------------------------------
// Class: CG_leaf
//-----------------------------------------------------------------------------

CG_result* CG_leaf::recompute(const BoolSet<> &parent_active,
		const Relation &known, const Relation &restriction) {
	active_ &= parent_active;
	known_ = copy(known);

	guards_.clear();
	for (BoolSet<>::iterator i = active_.begin(); i != active_.end(); i++) {
		Relation r = Intersection(
				copy(codegen_->projected_IS_[num_level() - 1][*i]),
				copy(restriction));
		r.simplify(2, 4);
		if (!r.is_upper_bound_satisfiable())
			active_.unset(*i);
		else {
			r = Gist(r, copy(known), 1);
			if (!r.is_obvious_tautology()) {
				guards_[*i] = r;
				guards_[*i].copy_names(known);
				guards_[*i].setup_names();
			}
		}
	}


	if (active_.empty()) {
		delete this;
		return NULL;
	} else
		return this;
}

std::pair<CG_result *, Relation> CG_leaf::liftOverhead(int depth, bool) {
	if (depth == 0)
		return std::make_pair(this, Relation::True(num_level()));

	for (std::map<int, Relation>::iterator i = guards_.begin();
			i != guards_.end(); i++) {
		Relation r = pick_one_guard(i->second);
		if (!r.is_obvious_tautology()) {
			bool has_wildcard = false;
			int max_level = 0;
			for (EQ_Iterator e(r.single_conjunct()->EQs()); e; e++) {
				if ((*e).has_wildcards())
					has_wildcard = true;
				for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
					if (cvi.curr_var()->kind() == Input_Var
							&& cvi.curr_var()->get_position() > max_level)
						max_level = cvi.curr_var()->get_position();
			}
			for (GEQ_Iterator e(r.single_conjunct()->GEQs()); e; e++) {
				if ((*e).has_wildcards())
					has_wildcard = true;
				for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
					if (cvi.curr_var()->kind() == Input_Var
							&& cvi.curr_var()->get_position() > max_level)
						max_level = cvi.curr_var()->get_position();
			}

			if (!(has_wildcard && max_level == codegen_->num_level()))
				return std::make_pair(this, r);
		}
	}

	return std::make_pair(this, Relation::True(num_level()));
}

Relation CG_leaf::hoistGuard() {
	std::vector<Relation> guards;
	for (BoolSet<>::iterator i = active_.begin(); i != active_.end(); i++) {
		std::map<int, Relation>::iterator j = guards_.find(*i);
		if (j == guards_.end()) {
			Relation r = Relation::True(num_level());
			r.copy_names(known_);
			r.setup_names();
			return r;
		} else {
			guards.push_back(j->second);
		}
	}

	return SimpleHull(guards, true, true);
}

void CG_leaf::removeGuard(const Relation &guard) {
	known_ = Intersection(known_, copy(guard));
	known_.simplify();

	std::map<int, Relation>::iterator i = guards_.begin();
	while (i != guards_.end()) {
		i->second = Gist(i->second, copy(known_), 1);
		if (i->second.is_obvious_tautology())
			guards_.erase(i++);
		else
			++i;
	}
}

CG_outputRepr *CG_leaf::printRepr(int indent, CG_outputBuilder *ocg,
		const std::vector<CG_outputRepr *> &stmts,
		const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly) const {
	return leaf_print_repr(active_, guards_, NULL, known_, indent, ocg,
			codegen_->remap_, codegen_->xforms_, stmts, assigned_on_the_fly);
}

CG_result *CG_leaf::clone() const {
	return new CG_leaf(codegen_, active_);
}

void CG_leaf::dump(int indent) const {
	std::string prefix;
	for (int i = 0; i < indent; i++)
		prefix += "  ";
	std::cout << prefix << "LEAF: " << active_ << std::endl;
	std::cout << prefix << "known: ";
	const_cast<CG_leaf *>(this)->known_.print();
	for (std::map<int, Relation>::const_iterator i = guards_.begin();
			i != guards_.end(); i++) {
		std::cout << prefix << "guard #" << i->first << ":";
		const_cast<Relation &>(i->second).print();
	}
}

}
