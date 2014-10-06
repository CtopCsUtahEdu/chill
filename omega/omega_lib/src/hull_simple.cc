/*****************************************************************************
 Copyright (C) 2011 Chun Chen
 All Rights Reserved.

 Purpose:
 Hull approximation including lattice and irregular constraints that
 involves wildcards.

 Notes:

 History:
 03/12/11 Created by Chun Chen
 *****************************************************************************/

#include <assert.h>
#include <omega.h>
#include <basic/boolset.h>
#include <vector>
#include <list>
#include <set>
#include <string>
#include <algorithm>

namespace omega {

Relation SimpleHull(const Relation &R, bool allow_stride_constraint,
		bool allow_irregular_constraint) {
	std::vector<Relation> Rs;
	Rs.push_back(R);
	return SimpleHull(Rs, allow_stride_constraint, allow_irregular_constraint);
}

Relation SimpleHull(const std::vector<Relation> &Rs,
		bool allow_stride_constraint, bool allow_irregular_constraint) {
	// check for sanity of parameters
	if (Rs.size() == 0)
		return Relation::False(0);
	int num_dim = -1;
	int first_non_null;
	for (int i = 0; i < Rs.size(); i++) {
		if (Rs[i].is_null())
			continue;

		if (num_dim == -1) {
			num_dim = Rs[i].n_inp();
			first_non_null = i;
		}

		if (Rs[i].n_inp() != num_dim)
			throw std::invalid_argument(
					"relations for hull must have same dimension");
		if (Rs[i].n_out() != 0)
			throw std::invalid_argument(
					"hull calculation must be set relation");
	}

	// convert to a list of relations each with a single conjunct
	std::vector<Relation> l_Rs;
	for (int i = 0; i < Rs.size(); i++) {
		if (Rs[i].is_null())
			continue;

		Relation r = copy(Rs[i]);

		 //r.simplify(2, 4);
	        r.simplify();
		DNF_Iterator c(r.query_DNF());
		int top = l_Rs.size();
		while (c.live()) {
			Relation r2 = Relation(r, c.curr());

			// quick elimination of redundant conjuncts
			bool already_included = false;
			for (int j = 0; j < top; j++)
				if (Must_Be_Subset(copy(r2), copy(l_Rs[j]))) {
					already_included = true;
					break;
				} else if (Must_Be_Subset(copy(l_Rs[j]), copy(r2))) {
					l_Rs.erase(l_Rs.begin() + j);
					top--;
					break;
				}

			if (!already_included)
				l_Rs.push_back(r2);
			c++;
		}
	}

	// shortcut for simple case
	if (l_Rs.size() == 0) {
		if (num_dim == -1)
			return Relation::False(0);
		else {
			Relation r = Relation::False(num_dim);
			r.copy_names(Rs[first_non_null]);
			r.setup_names();
			return r;
		}
	} else if (l_Rs.size() == 1) {
		if (allow_stride_constraint && allow_irregular_constraint) {
			l_Rs[0].copy_names(Rs[first_non_null]);
			l_Rs[0].setup_names();
			return l_Rs[0];
		} else if (!allow_stride_constraint && !allow_irregular_constraint) {
			l_Rs[0] = Approximate(l_Rs[0]);
			l_Rs[0].copy_names(Rs[first_non_null]);
			l_Rs[0].setup_names();
			return l_Rs[0];
		}
	}

	Relation hull = Relation::True(num_dim);

	// lattice union approximation
	if (allow_stride_constraint) {
		std::vector<std::vector<std::pair<EQ_Handle, BoolSet<> > > > strides(
				l_Rs.size());
		for (int i = 0; i < l_Rs.size(); i++)
			for (EQ_Iterator e = l_Rs[i].single_conjunct()->EQs(); e; e++)
				if ((*e).has_wildcards()) {
					int num_wildcard = 0;
					BoolSet<> affected(num_dim);
					for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
						if (cvi.curr_var()->kind() == Wildcard_Var)
							num_wildcard++;
						else if (cvi.curr_var()->kind() == Input_Var)
							affected.set(cvi.curr_var()->get_position() - 1);
					}
					if (num_wildcard == 1)
						strides[i].push_back(std::make_pair(*e, affected));
				}

		for (int i = 0; i < strides[0].size(); i++) {
			coef_t c =
					abs(
							strides[0][i].first.get_coef(
									Constr_Vars_Iter(strides[0][i].first, true).curr_var()));
			coef_t old_c = c;
			bool is_stride = true;
			for (int j = 1; j < l_Rs.size(); j++) {
				std::list<coef_t> candidates;
				for (int k = 0; k < strides[j].size(); k++)
					if (!(strides[0][i].second & strides[j][k].second).empty()) {
						coef_t t = gcd(c,
								abs(
										strides[j][k].first.get_coef(
												Constr_Vars_Iter(
														strides[j][k].first,
														true).curr_var())));
						if (t != 1) {
							std::list<coef_t>::iterator p = candidates.begin();
							while (p != candidates.end() && *p > t)
								++p;
							if (p == candidates.end() || *p != t)
								candidates.insert(p, t);

							t = gcd(t, abs(strides[0][i].first.get_const()));
							t = gcd(t, abs(strides[j][k].first.get_const()));
							if (t != 1) {
								std::list<coef_t>::iterator p =
										candidates.begin();
								while (p != candidates.end() && *p > t)
									++p;
								if (p == candidates.end() || *p != t)
									candidates.insert(p, t);
							}
						}
					}

				bool found_matched_stride = false;
				for (std::list<coef_t>::iterator k = candidates.begin();
						k != candidates.end(); k++) {
					Relation r = Relation::True(num_dim);
					EQ_Handle h = r.and_with_EQ(strides[0][i].first);
					h.update_coef(Constr_Vars_Iter(h, true).curr_var(),
							-old_c + *k);
					r.simplify();
					if (Must_Be_Subset(copy(l_Rs[j]), copy(r))) {
						c = *k;
						found_matched_stride = true;
						break;
					}
				}

				if (!found_matched_stride) {
					is_stride = false;
					break;
				}
			}

			if (is_stride) {
				Relation r = Relation::True(num_dim);
				EQ_Handle h = r.and_with_EQ(strides[0][i].first);
				h.update_coef(Constr_Vars_Iter(h, true).curr_var(), -old_c + c);
				r.simplify();
				hull = Intersection(hull, r);
			}
		}
	}

	// consider some special wildcard constraints
	if (allow_irregular_constraint) {
		std::vector<
				std::vector<
						std::pair<Variable_ID, std::map<Variable_ID, coef_t> > > > ranges(
				l_Rs.size());
		for (int i = 0; i < l_Rs.size(); i++) {
			std::vector<std::pair<GEQ_Handle, std::map<Variable_ID, coef_t> > > geqs_ub;
			std::vector<std::pair<GEQ_Handle, std::map<Variable_ID, coef_t> > > geqs_lb;
			for (GEQ_Iterator e = l_Rs[i].single_conjunct()->GEQs(); e; e++)
				if ((*e).has_wildcards()) {
					int num_wildcard = 0;
					std::map<Variable_ID, coef_t> formula;
					int direction;
					for (Constr_Vars_Iter cvi(*e); cvi; cvi++) {
						Variable_ID v = cvi.curr_var();
						switch (v->kind()) {
						case Wildcard_Var:
							num_wildcard++;
							if (cvi.curr_coef() > 0)
								direction = true;
							else
								direction = false;
							break;
						case Input_Var:
						case Global_Var:
							formula[cvi.curr_var()] = cvi.curr_coef();
							break;
						default:
							assert(false);
						}
					}
					if (num_wildcard == 1) {
						if (direction) {
							for (std::map<Variable_ID, coef_t>::iterator j =
									formula.begin(); j != formula.end(); j++)
								j->second = -j->second;
							geqs_ub.push_back(std::make_pair(*e, formula));
						} else
							geqs_lb.push_back(std::make_pair(*e, formula));
					}
				}
			for (int j = 0; j < geqs_lb.size(); j++) {
				Variable_ID v =
						Constr_Vars_Iter(geqs_lb[j].first, true).curr_var();
				for (int k = 0; k < geqs_ub.size(); k++)
					if (v == Constr_Vars_Iter(geqs_ub[k].first, true).curr_var()
							&& geqs_lb[j].second == geqs_ub[k].second)
						ranges[i].push_back(
								std::make_pair(v, geqs_lb[j].second));
			}
		}

		// find compatible wildcard match
		// TODO: evaluate to find the best match, also avoid mapping two wildcards
		//       in a single conjunct to one variable (incorrect)
		std::vector<std::vector<int> > all_match;
		for (int i = 0; i < ranges[0].size(); i++) {
			std::vector<int> match(l_Rs.size(), -1);
			match[0] = i;
			for (int j = 1; j < l_Rs.size(); j++) {
				for (int k = 0; k < ranges[j].size(); k++)
					if (ranges[0][i].second == ranges[j][k].second) {
						match[j] = k;
						break;
					}
				if (match[j] == -1)
					break;
			}
			if (match[l_Rs.size() - 1] != -1)
				all_match.push_back(match);
		}

		// map compatible wildcards to input variables
		std::vector<Relation> ll_Rs(l_Rs.size());
		for (int i = 0; i < l_Rs.size(); i++) {
			Relation r(num_dim + all_match.size());
			F_Exists *f_exists = r.add_and()->add_exists();
			F_And *f_root = f_exists->add_and();
			std::map<Variable_ID, Variable_ID> wc_map;
			for (int j = 0; j < all_match.size(); j++)
				wc_map[ranges[i][all_match[j][i]].first] = r.set_var(j + 1);

			for (EQ_Iterator e(l_Rs[i].single_conjunct()->EQs()); e; e++)
				if (!(*e).has_wildcards()) {
					EQ_Handle h = f_root->add_EQ();
					for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
						switch (cvi.curr_var()->kind()) {
						case Input_Var: {
							h.update_coef(
									r.set_var(
											cvi.curr_var()->get_position()
													+ all_match.size()),
									cvi.curr_coef());
							break;
						}
						case Global_Var: {
							Global_Var_ID g = cvi.curr_var()->get_global_var();
							Variable_ID v;
							if (g->arity() == 0)
								v = r.get_local(g);
							else
								v = r.get_local(g,
										cvi.curr_var()->function_of());
							h.update_coef(v, cvi.curr_coef());
							break;
						}
						default:
							assert(false);
						}
					h.update_const((*e).get_const());
				}

			for (GEQ_Iterator e(l_Rs[i].single_conjunct()->GEQs()); e; e++) {
				GEQ_Handle h = f_root->add_GEQ();
				for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
					switch (cvi.curr_var()->kind()) {
					case Input_Var: {
						h.update_coef(
								r.set_var(
										cvi.curr_var()->get_position()
												+ all_match.size()),
								cvi.curr_coef());
						break;
					}
					case Global_Var: {
						Global_Var_ID g = cvi.curr_var()->get_global_var();
						Variable_ID v;
						if (g->arity() == 0)
							v = r.get_local(g);
						else
							v = r.get_local(g, cvi.curr_var()->function_of());
						h.update_coef(v, cvi.curr_coef());
						break;
					}
					case Wildcard_Var: {
						std::map<Variable_ID, Variable_ID>::iterator p =
								wc_map.find(cvi.curr_var());
						Variable_ID v;
						if (p == wc_map.end()) {
							v = f_exists->declare();
							wc_map[cvi.curr_var()] = v;
						} else
							v = p->second;
						h.update_coef(v, cvi.curr_coef());
						break;
					}
					default:
						assert(false);
					}
				h.update_const((*e).get_const());
			}

			r.simplify();
			ll_Rs[i] = r;
		}

		// now use SimpleHull on regular bounds only
		Relation result = SimpleHull(ll_Rs, false, false);

		// convert imaginary input variables back to wildcards
		Relation mapping(num_dim + all_match.size(), num_dim);
		F_And *f_root = mapping.add_and();
		for (int i = 0; i < num_dim; i++) {
			EQ_Handle h = f_root->add_EQ();
			h.update_coef(mapping.input_var(all_match.size() + i + 1), 1);
			h.update_coef(mapping.output_var(i + 1), -1);
		}
		result = Range(Restrict_Domain(mapping, result));

		hull = Intersection(hull, result);
		hull.simplify();
		hull.copy_names(Rs[first_non_null]);
		hull.setup_names();
		return hull;
	}

	// check regular bounds
	if (l_Rs.size() == 1) {
		hull = Intersection(hull, Approximate(copy(l_Rs[0])));
	} else {
		for (int i = 0; i < l_Rs.size(); i++) {
			l_Rs[i] = Approximate(l_Rs[i]);
			l_Rs[i].simplify(2, 4);
		}

		// check global variables
		// TODO: global variable function_of() is not considered for now
		std::map<Global_Var_ID, int> globals;
		for (int i = 0; i < l_Rs.size(); i++)
			for (Constraint_Iterator e(
					l_Rs[i].single_conjunct()->constraints()); e; e++)
				for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
					if (cvi.curr_var()->kind() == Global_Var)
						globals[cvi.curr_var()->get_global_var()] = -1;

		if (globals.size() > 0) {
			int count = 1;
			for (std::map<Global_Var_ID, int>::iterator i = globals.begin();
					i != globals.end(); i++)
				i->second = count++;

			std::vector<Relation> ll_Rs(l_Rs.size());
			for (int i = 0; i < l_Rs.size(); i++) {
				Relation r(num_dim + globals.size());
				F_And *f_root = r.add_and();
				for (EQ_Iterator e(l_Rs[i].single_conjunct()->EQs()); e; e++) {
					EQ_Handle h = f_root->add_EQ();
					for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
						switch (cvi.curr_var()->kind()) {
						case Input_Var: {
							h.update_coef(
									r.set_var(
											cvi.curr_var()->get_position()
													+ globals.size()),
									cvi.curr_coef());
							break;
						}
						case Global_Var: {
							h.update_coef(
									r.set_var(
											globals[cvi.curr_var()->get_global_var()]),
									cvi.curr_coef());
							break;
						}
						default:
							assert(false);
						}
					h.update_const((*e).get_const());
				}
				for (GEQ_Iterator e(l_Rs[i].single_conjunct()->GEQs()); e;
						e++) {
					GEQ_Handle h = f_root->add_GEQ();
					for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
						switch (cvi.curr_var()->kind()) {
						case Input_Var: {
							h.update_coef(
									r.set_var(
											cvi.curr_var()->get_position()
													+ globals.size()),
									cvi.curr_coef());
							break;
						}
						case Global_Var: {
							h.update_coef(
									r.set_var(
											globals[cvi.curr_var()->get_global_var()]),
									cvi.curr_coef());
							break;
						}
						default:
							assert(false);
						}
					h.update_const((*e).get_const());
				}

				ll_Rs[i] = r;
			}

			Relation result = SimpleHull(ll_Rs, false, false);

			std::map<int, Global_Var_ID> globals_reverse;
			for (std::map<Global_Var_ID, int>::iterator i = globals.begin();
					i != globals.end(); i++)
				globals_reverse[i->second] = i->first;

			Relation r(num_dim);
			F_And *f_root = r.add_and();
			for (EQ_Iterator e(result.single_conjunct()->EQs()); e; e++) {
				EQ_Handle h = f_root->add_EQ();
				for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
					switch (cvi.curr_var()->kind()) {
					case Input_Var: {
						int pos = cvi.curr_var()->get_position();
						if (pos > globals_reverse.size())
							h.update_coef(
									r.set_var(pos - globals_reverse.size()),
									cvi.curr_coef());
						else {
							Global_Var_ID g = globals_reverse[pos];
							h.update_coef(r.get_local(g), cvi.curr_coef());
						}
						break;
					}
					default:
						assert(false);
					}
				h.update_const((*e).get_const());
			}
			for (GEQ_Iterator e(result.single_conjunct()->GEQs()); e; e++) {
				GEQ_Handle h = f_root->add_GEQ();
				for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
					switch (cvi.curr_var()->kind()) {
					case Input_Var: {
						int pos = cvi.curr_var()->get_position();
						if (pos > globals_reverse.size())
							h.update_coef(
									r.set_var(pos - globals_reverse.size()),
									cvi.curr_coef());
						else {
							Global_Var_ID g = globals_reverse[pos];
							h.update_coef(r.get_local(g), cvi.curr_coef());
						}
						break;
					}
					default:
						assert(false);
					}
				h.update_const((*e).get_const());
			}

			hull = Intersection(hull, r);
			hull.simplify();
			hull.copy_names(Rs[first_non_null]);
			hull.setup_names();
			return hull;
		} else {
			std::vector<std::vector<Relation> > projected(num_dim + 1,
					std::vector<Relation>(l_Rs.size()));
			for (int i = 0; i < l_Rs.size(); i++) {
				projected[num_dim][i] = copy(l_Rs[i]);
				for (int j = num_dim - 1; j >= 0; j--) {
					projected[j][i] = Project(copy(projected[j + 1][i]),
							projected[j + 1][i].input_var(j + 1));
					projected[j][i].simplify(2, 4);
				}
			}

			std::vector<bool> has_lb(num_dim, false);
			std::vector<bool> has_ub(num_dim, false);
			for (int i = 0; i < num_dim; i++) {
				bool skip_lb = false;
				bool skip_ub = false;
				std::vector<Relation> bound(l_Rs.size());
				for (int j = 0; j < l_Rs.size(); j++) {
					bound[j] = Gist(copy(projected[i + 1][j]),
							copy(projected[i][j]), 1);
					bound[j] = Approximate(bound[j]);
					bound[j] = EQs_to_GEQs(bound[j]);

					bool has_lb_not_in_hull = false;
					bool has_ub_not_in_hull = false;
					for (GEQ_Iterator e = bound[j].single_conjunct()->GEQs(); e;
							e++) {
						coef_t coef = (*e).get_coef(bound[j].input_var(i + 1));
						if (!skip_lb && coef > 0) {
							Relation r = Relation::True(bound[j].n_inp());
							r.and_with_GEQ(*e);
							r.simplify();

							if (j != 0 && l_Rs.size() > 2
									&& Must_Be_Subset(copy(hull), copy(r)))
								continue;

							bool belong_to_hull = true;
							for (int k = 0; k < l_Rs.size(); k++)
								if (k != j
										&& !Must_Be_Subset(copy(l_Rs[k]),
												copy(r))) {
									belong_to_hull = false;
									break;
								}
							if (belong_to_hull) {
								hull.and_with_GEQ(*e);
								has_lb[i] = true;
							} else
								has_lb_not_in_hull = true;
						} else if (!skip_ub && coef < 0) {
							Relation r = Relation::True(bound[j].n_inp());
							r.and_with_GEQ(*e);
							r.simplify();

							if (j != 0 && l_Rs.size() > 2
									&& Must_Be_Subset(copy(hull), copy(r)))
								continue;

							bool belong_to_hull = true;
							for (int k = 0; k < l_Rs.size(); k++)
								if (k != j
										&& !Must_Be_Subset(copy(l_Rs[k]),
												copy(r))) {
									belong_to_hull = false;
									break;
								}
							if (belong_to_hull) {
								hull.and_with_GEQ(*e);
								has_ub[i] = true;
							} else
								has_ub_not_in_hull = true;
						}
					}

					if (!has_lb_not_in_hull)
						skip_lb = true;
					if (!has_ub_not_in_hull)
						skip_ub = true;
					if (skip_lb && skip_ub)
						break;
				}

				// no ready lower bound, approximate it
				bool got_rect_lb = false;
				if (!skip_lb) {
					for (int j = 0; j < l_Rs.size(); j++) {
						std::set<BoolSet<> > S;
						for (GEQ_Iterator e =
								bound[j].single_conjunct()->GEQs(); e; e++) {
							coef_t coef = (*e).get_coef(
									bound[j].input_var(i + 1));
							if (coef > 0) {
								BoolSet<> s(i);
								for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
									if ((*cvi).var->kind() == Input_Var
											&& (*cvi).var->get_position() - 1
													!= i) {
										if (((*cvi).coef > 0
												&& has_ub[(*cvi).var->get_position()
														- 1])
												|| ((*cvi).coef < 0
														&& has_lb[(*cvi).var->get_position()
																- 1]))
											s.set(
													(*cvi).var->get_position()
															- 1);
										else {
											for (GEQ_Iterator e2 =
													bound[j].single_conjunct()->GEQs();
													e2; e2++)
												if (e2 != e
														&& (((*cvi).coef > 0
																&& (*e2).get_coef(
																		(*cvi).var)
																		< 0)
																|| ((*cvi).coef
																		< 0
																		&& (*e2).get_coef(
																				(*cvi).var)
																				> 0))) {
													s.set(
															(*cvi).var->get_position()
																	- 1);
													break;
												}
										}
									}

								if (s.num_elem() > 0)
									S.insert(s);
							}
						}

						if (S.size() > 0) {
							BoolSet<> s(i);
							for (std::set<BoolSet<> >::iterator k = S.begin();
									k != S.end(); k++)
								s |= *k;
							for (int k = 0; k < i; k++)
								if (s.get(k)) {
									BoolSet<> t(i);
									t.set(k);
									S.insert(t);
								}

							for (std::set<BoolSet<> >::iterator k = S.begin();
									k != S.end(); k++) {

								bool do_again = false;
								std::set<int> vars;
								int round_trip = 0;
								do {
									Relation r = copy(projected[i + 1][j]);

									if (!do_again) {
										for (int kk = 0; kk < i; kk++)
											if ((*k).get(kk)) {
												r = Project(r,
														r.input_var(kk + 1));
												vars.insert(kk + 1);
											}
									} else {
										for (std::set<int>::iterator vars_it =
												vars.begin();
												vars_it != vars.end();
												vars_it++)
											if (*vars_it < i + 1)
												r = Project(r,
														r.input_var(*vars_it));
									}

									r.simplify(2, 4);
									Relation r2 = Project(copy(r),
											r.input_var(i + 1));
									Relation b = Gist(copy(r), copy(r2), 1);
									// Relation c = Project(copy(r),r.input_var(4) );

									// c.simplify(2,4);
									// Relation d = Project(copy(c), r.input_var(i+1));
									// Relation e = Gist(copy(c), copy(d), 1);

									b = Approximate(b);
									b = EQs_to_GEQs(b);

									for (GEQ_Iterator e =
											b.single_conjunct()->GEQs(); e;
											e++) {
										coef_t coef = (*e).get_coef(
												b.input_var(i + 1));
										if (coef > 0) {
											Relation r = Relation::True(
													b.n_inp());
											r.and_with_GEQ(*e);
											r.simplify();

											if (Must_Be_Subset(copy(hull),
													copy(r)))
												continue;

											bool belong_to_hull = true;
											for (int k = 0; k < l_Rs.size();
													k++)
												if (k != j
														&& !Must_Be_Subset(
																copy(l_Rs[k]),
																copy(r))) {
													belong_to_hull = false;
													break;
												}
											if (belong_to_hull) {
												hull.and_with_GEQ(*e);
												got_rect_lb = true;
											}
										}
									}
									do_again = false;
									if (!got_rect_lb) {
										bool found = false;
										for (GEQ_Iterator e =
												b.single_conjunct()->GEQs(); e;
												e++) {
											coef_t coef = (*e).get_coef(
													b.input_var(i + 1));

											if (coef > 0) {
												for (Constr_Vars_Iter cvi(*e);
														cvi; cvi++)
													if ((*cvi).var->kind()
															== Input_Var
															&& (*cvi).var->get_position()
																	- 1 != i) {

														if (((*cvi).coef > 0
																&& has_ub[(*cvi).var->get_position()
																		- 1])
																|| ((*cvi).coef
																		< 0
																		&& has_lb[(*cvi).var->get_position()
																				- 1])) {
															vars.insert(
																	(*cvi).var->get_position());
															found = true;
														} else {
															for (GEQ_Iterator e2 =
																	b.single_conjunct()->GEQs();
																	e2; e2++)
																if (e2 != e
																		&& (((*cvi).coef
																				> 0
																				&& (*e2).get_coef(
																						(*cvi).var)
																						< 0)
																				|| ((*cvi).coef
																						< 0
																						&& (*e2).get_coef(
																								(*cvi).var)
																								> 0))) {
																	vars.insert(
																			(*cvi).var->get_position());
																	found =
																			true;
																	break;
																}
														}

													}

											}
											if (found)
												break;

										}
										if (found && (round_trip < i))
											do_again = true;

									}
									round_trip++;
								} while (do_again);
							}

							if (got_rect_lb)
								break;
						}
					}
				}

				// no ready upper bound, approximate it
				bool got_rect_ub = false;
				if (!skip_ub) {
					for (int j = 0; j < l_Rs.size(); j++) {
						std::set<BoolSet<> > S;
						for (GEQ_Iterator e =
								bound[j].single_conjunct()->GEQs(); e; e++) {
							coef_t coef = (*e).get_coef(
									bound[j].input_var(i + 1));
							if (coef < 0) {
								BoolSet<> s(i);
								for (Constr_Vars_Iter cvi(*e); cvi; cvi++)
									if ((*cvi).var->kind() == Input_Var
											&& (*cvi).var->get_position() - 1
													!= i) {
										if (((*cvi).coef > 0
												&& has_ub[(*cvi).var->get_position()
														- 1])
												|| ((*cvi).coef < 0
														&& has_lb[(*cvi).var->get_position()
																- 1]))
											s.set(
													(*cvi).var->get_position()
															- 1);
										else {
											for (GEQ_Iterator e2 =
													bound[j].single_conjunct()->GEQs();
													e2; e2++)
												if (e2 != e
														&& (((*cvi).coef > 0
																&& (*e2).get_coef(
																		(*cvi).var)
																		< 0)
																|| ((*cvi).coef
																		< 0
																		&& (*e2).get_coef(
																				(*cvi).var)
																				> 0))) {
													s.set(
															(*cvi).var->get_position()
																	- 1);
													break;
												}
										}
									}

								if (s.num_elem() > 0)
									S.insert(s);
							}
						}

						if (S.size() > 0) {
							BoolSet<> s(i);
							for (std::set<BoolSet<> >::iterator k = S.begin();
									k != S.end(); k++)
								s |= *k;
							for (int k = 0; k < i; k++)
								if (s.get(k)) {
									BoolSet<> t(i);
									t.set(k);
									S.insert(t);
								}

							for (std::set<BoolSet<> >::iterator k = S.begin();
									k != S.end(); k++) {

								bool do_again = false;
								std::set<int> vars;
								int round_trip = 0;
								do {

									Relation r = copy(projected[i + 1][j]);

									if (!do_again) {
										for (int kk = 0; kk < i; kk++)
											if ((*k).get(kk)) {
												r = Project(r,
														r.input_var(kk + 1));
												vars.insert(kk + 1);
											}
									} else {
										for (std::set<int>::iterator vars_it =
												vars.begin();
												vars_it != vars.end();
												vars_it++)
											if (*vars_it < i + 1)
												r = Project(r,
														r.input_var(*vars_it));
									}

									r.simplify(2, 4);
									Relation r2 = Project(copy(r),
											r.input_var(i + 1));
									// r2.simplify(2,4);
									Relation b = Gist(r, r2, 1);
									b = Approximate(b);
									b = EQs_to_GEQs(b);

									for (GEQ_Iterator e =
											b.single_conjunct()->GEQs(); e;
											e++) {
										coef_t coef = (*e).get_coef(
												b.input_var(i + 1));
										if (coef < 0) {
											Relation r = Relation::True(
													b.n_inp());
											r.and_with_GEQ(*e);
											r.simplify();

											if (Must_Be_Subset(copy(hull),
													copy(r)))
												continue;

											bool belong_to_hull = true;
											for (int k = 0; k < l_Rs.size();
													k++)
												if (k != j
														&& !Must_Be_Subset(
																copy(l_Rs[k]),
																copy(r))) {
													belong_to_hull = false;
													break;
												}
											if (belong_to_hull) {
												hull.and_with_GEQ(*e);
												got_rect_ub = true;
											}
										}
									}
                                    do_again = false;
									if (!got_rect_ub) {
										bool found = false;
										for (GEQ_Iterator e =
												b.single_conjunct()->GEQs(); e;
												e++) {
											coef_t coef = (*e).get_coef(
													b.input_var(i + 1));
											if (coef < 0) {
												for (Constr_Vars_Iter cvi(*e);
														cvi; cvi++)
													if ((*cvi).var->kind()
															== Input_Var
															&& (*cvi).var->get_position()
																	- 1 != i) {

														if (((*cvi).coef > 0
																&& has_ub[(*cvi).var->get_position()
																		- 1])
																|| ((*cvi).coef
																		< 0
																		&& has_lb[(*cvi).var->get_position()
																				- 1])) {
															vars.insert(
																	(*cvi).var->get_position());
															found = true;
														} else {
															for (GEQ_Iterator e2 =
																	b.single_conjunct()->GEQs();
																	e2; e2++)
																if (e2 != e
																		&& (((*cvi).coef
																				> 0
																				&& (*e2).get_coef(
																						(*cvi).var)
																						< 0)
																				|| ((*cvi).coef
																						< 0
																						&& (*e2).get_coef(
																								(*cvi).var)
																								> 0))) {
																	vars.insert(
																			(*cvi).var->get_position());
																	found =
																			true;
																	break;
																}
														}

													}
											}
											if (found)
												break;
										}
										if (found && (round_trip < i))
											do_again = true;

									}
									round_trip++;
								} while (do_again);
							}

							if (got_rect_ub)
								break;
						}
					}
				}
			}
		}
	}

	hull.simplify();
	hull.copy_names(Rs[first_non_null]);
	hull.setup_names();
	return hull;
}

}

