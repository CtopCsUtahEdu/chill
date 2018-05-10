/*
* loop_sparse.cc
*  Loop Class for Sparse Matrix Format Transformations
*  Created on: October 7, 2013
*      Author: Anand Venkat
*/

#include "loop.hh"
#include <iegenlib.h>
#include <code_gen/CG_utils.h>
#include "omegatools.hh"
#include <sstream>

using namespace omega;

omega::Relation flip_var_exclusive(omega::Relation &r1, Variable_ID &v1) {
  omega::Relation r;
  if (r1.is_set())
    r = omega::Relation(r1.n_set());
  else
    r = omega::Relation(r1.n_inp(), r1.n_out());

  r.copy_names(r1);
  r.setup_names();
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  bool found = false;
  for (DNF_Iterator di(const_cast<omega::Relation &>(r1).query_DNF()); di; di++) {

    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      if ((*gi).get_coef(v1) != 0) {
        GEQ_Handle h = f_root->add_GEQ();
        found = true;
        //&& (*gi).is_const_except_for_global(
        //		const_cast<Relation &>(old_relation).set_var(
        //				old_pos))) {
        bool not_const = true;
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
            case Output_Var: {

              if (v->get_position() == v1->get_position())
                h.update_coef(r.output_var(v->get_position()),
                              -cvi.curr_coef());
              else
                h.update_coef(r.output_var(v->get_position()),
                              cvi.curr_coef());
              /*else if (v->get_position() < old_pos)
               h.update_coef(r.input_var(v->get_position()),
               cvi.curr_coef());   //Anand: hack//
               else
               not_const = true;
               */
              //else
              //	throw omega_error(
              //			"relation contains set vars other than that to be replicated!");
              break;

            }
            case Input_Var: {

              if (v->get_position() == v1->get_position())
                h.update_coef(r.input_var(v->get_position()),
                              -cvi.curr_coef());
              else
                h.update_coef(r.input_var(v->get_position()),
                              cvi.curr_coef());
              /*else if (v->get_position() < old_pos)
               h.update_coef(r.input_var(v->get_position()),
               cvi.curr_coef());   //Anand: hack//
               else
               not_const = true;
               */
              //else
              //	throw omega_error(
              //			"relation contains set vars other than that to be replicated!");
              break;

            }

            case Wildcard_Var: {
              Variable_ID v2 = replicate_floor_definition(r1, v, r,
                                                          f_exists, f_root, exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            case Global_Var: {
              Global_Var_ID g = v->get_global_var();
              Variable_ID v2;
              if (g->arity() == 0)
                v2 = r.get_local(g);
              else
                v2 = r.get_local(g, v->function_of());
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            default:
              assert(false);
          }
        }
        h.update_const((*gi).get_const());
      }
    }
    for (EQ_Iterator ei((*di)->EQs()); ei; ei++) {
      if ((*ei).get_coef(v1) != 0) {
        found = true;
        EQ_Handle h = f_root->add_EQ();

        for (Constr_Vars_Iter cvi(*ei); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {

            case Output_Var: {

              if (v->get_position() == v1->get_position())
                h.update_coef(r.output_var(v->get_position()),
                              -cvi.curr_coef());
              else
                h.update_coef(r.output_var(v->get_position()),
                              cvi.curr_coef());
              /*else if (v->get_position() < old_pos)
               h.update_coef(r.input_var(v->get_position()),
               cvi.curr_coef());   //Anand: hack//
               else
               not_const = true;
               */
              //else
              //	throw omega_error(
              //			"relation contains set vars other than that to be replicated!");
              break;

            }
            case Input_Var: {

              if (v->get_position() == v1->get_position())
                h.update_coef(r.input_var(v->get_position()),
                              -cvi.curr_coef());
              else
                h.update_coef(r.input_var(v->get_position()),
                              cvi.curr_coef());
              //	else
              //		throw omega_error(
              //				"relation contains set vars other than that to be replicated!");
              break;

            }
            case Wildcard_Var: {
              Variable_ID v2 = replicate_floor_definition(r1, v, r,
                                                          f_exists, f_root, exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            case Global_Var: {
              Global_Var_ID g = v->get_global_var();
              Variable_ID v2;
              if (g->arity() == 0)
                v2 = r.get_local(g);
              else
                v2 = r.get_local(g, v->function_of());
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
            default:
              assert(false);
          }
        }
        h.update_const((*ei).get_const());
      }

    }
  }

  r.simplify();
  if (found)
    return r;
  return r1;
}

omega::Relation flip_var(omega::Relation &r1, Variable_ID &v1) {
  omega::Relation r;
  if (r1.is_set())
    r = omega::Relation(r1.n_set());
  else
    r = omega::Relation(r1.n_inp(), r1.n_out());

  r.copy_names(r1);
  r.setup_names();
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;

  for (DNF_Iterator di(const_cast<omega::Relation &>(r1).query_DNF()); di; di++) {
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();

      //&& (*gi).is_const_except_for_global(
      //		const_cast<Relation &>(old_relation).set_var(
      //				old_pos))) {
      bool not_const = true;
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
          case Output_Var: {

            if (v->get_position() == v1->get_position())
              h.update_coef(r.output_var(v->get_position()),
                            -cvi.curr_coef());
            else
              h.update_coef(r.output_var(v->get_position()),
                            cvi.curr_coef());
            /*else if (v->get_position() < old_pos)
             h.update_coef(r.input_var(v->get_position()),
             cvi.curr_coef());   //Anand: hack//
             else
             not_const = true;
             */
            //else
            //	throw omega_error(
            //			"relation contains set vars other than that to be replicated!");
            break;

          }
          case Input_Var: {

            if (v->get_position() == v1->get_position())
              h.update_coef(r.input_var(v->get_position()),
                            -cvi.curr_coef());
            else
              h.update_coef(r.input_var(v->get_position()),
                            cvi.curr_coef());
            /*else if (v->get_position() < old_pos)
             h.update_coef(r.input_var(v->get_position()),
             cvi.curr_coef());   //Anand: hack//
             else
             not_const = true;
             */
            //else
            //	throw omega_error(
            //			"relation contains set vars other than that to be replicated!");
            break;

          }

          case Wildcard_Var: {
            Variable_ID v2 = replicate_floor_definition(r1, v, r, f_exists,
                                                        f_root, exists_mapping);
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
          case Global_Var: {
            Global_Var_ID g = v->get_global_var();
            Variable_ID v2;
            if (g->arity() == 0)
              v2 = r.get_local(g);
            else
              v2 = r.get_local(g, v->function_of());
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
          default:
            assert(false);
        }
      }
      h.update_const((*gi).get_const());

    }
    for (EQ_Iterator ei((*di)->EQs()); ei; ei++) {
      EQ_Handle h = f_root->add_EQ();

      for (Constr_Vars_Iter cvi(*ei); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {

          case Output_Var: {

            if (v->get_position() == v1->get_position())
              h.update_coef(r.output_var(v->get_position()),
                            -cvi.curr_coef());
            else
              h.update_coef(r.output_var(v->get_position()),
                            cvi.curr_coef());
            /*else if (v->get_position() < old_pos)
             h.update_coef(r.input_var(v->get_position()),
             cvi.curr_coef());   //Anand: hack//
             else
             not_const = true;
             */
            //else
            //	throw omega_error(
            //			"relation contains set vars other than that to be replicated!");
            break;

          }
          case Input_Var: {

            if (v->get_position() == v1->get_position())
              h.update_coef(r.input_var(v->get_position()),
                            -cvi.curr_coef());
            else
              h.update_coef(r.input_var(v->get_position()),
                            cvi.curr_coef());
            //	else
            //		throw omega_error(
            //				"relation contains set vars other than that to be replicated!");
            break;

          }
          case Wildcard_Var: {
            Variable_ID v2 = replicate_floor_definition(r1, v, r, f_exists,
                                                        f_root, exists_mapping);
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
          case Global_Var: {
            Global_Var_ID g = v->get_global_var();
            Variable_ID v2;
            if (g->arity() == 0)
              v2 = r.get_local(g);
            else
              v2 = r.get_local(g, v->function_of());
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
          default:
            assert(false);
        }
      }
      h.update_const((*ei).get_const());
    }

  }

  r.simplify();
  return r;

}

omega::Relation parseISLStringToOmegaRelation(std::string s,
                                              const omega::Relation &r1, const omega::Relation &r2,
                                              std::map<std::string, std::string> &unin,
                                              std::vector<omega::Free_Var_Decl*> &freevar, IR_Code *ir) {

  // find first occurrence of '[' and ]'

  std::size_t pos, pos2;

  pos = s.find("{");
  pos2 = s.find("}");

  std::string var_string = s.substr(pos + 1, pos2 - pos - 1);
  pos = var_string.find("[");
  pos2 = var_string.find("]");

  var_string = var_string.substr(pos + 1, pos2 - pos - 1);
  char * str = new char[var_string.length() + 1];
  strcpy(str, var_string.c_str());

  char *pch;
  pch = strtok(str, " ,");
  std::set<std::string> vars;
  while (pch != NULL) {
    std::string var(pch);
    vars.insert(var);
    pch = strtok(NULL, " ,");

  }
  std::vector<std::string> ordered_vars;
  for (std::set<std::string>::iterator i = vars.begin(); i != vars.end();
       i++) {

    for (int j = 1; j <= const_cast<omega::Relation &>(r1).n_set(); j++) {
      if (const_cast<omega::Relation &>(r1).set_var(j)->name() == *i) {
        ordered_vars.push_back(*i);
        break;
      }
    }

  }

  for (std::set<std::string>::iterator i = vars.begin(); i != vars.end();
       i++) {

    for (int j = 1; j <= const_cast<omega::Relation &>(r2).n_set(); j++) {
      if (const_cast<omega::Relation &>(r2).set_var(j)->name() == *i) {
        ordered_vars.push_back(*i);
        break;
      }
    }

  }

  // tokenize and add variables in order first without p and then with p tokenize ", "
  // look for first occurence of ':' and '}'
  pos = s.find(":");

  std::string constraint_string = s.substr(pos + 1);

  char * constr_str = new char[constraint_string.length() + 1];
  strcpy(constr_str, constraint_string.c_str());
  std::vector<std::string> conjuncts;
  pch = strtok(constr_str, "&");
  while (pch != NULL) {
    std::string conjunct(pch);
    conjuncts.push_back(conjunct);
    pch = strtok(NULL, "&");
    //if(pch != NULL)
    //pch = strtok(NULL, "&");
  }

  std::set<std::string> terms;
  omega::Relation R(ordered_vars.size());

  for (int j = 1; j <= ordered_vars.size(); j++)
    R.name_set_var(j, ordered_vars[j - 1]);

  F_And *f_root = R.add_and();

  std::map<std::string, Variable_ID> gathered;
  for (int j = 0; j < conjuncts.size(); j++) {
    std::string conjunct = conjuncts[j];
    bool eq = false;
    bool geq = false;
    if (conjunct.find(">=") == std::string::npos)
      geq = false;
    else
      geq = true;

    if (geq == false)
      if (conjunct.find("=") != std::string::npos)
        eq = true;

    GEQ_Handle h;
    EQ_Handle e;

    if (eq)
      e = f_root->add_EQ();
    if (geq)
      h = f_root->add_GEQ();

    assert(eq || geq);
    char * conj_str = new char[conjunct.length() + 1];
    strcpy(conj_str, conjunct.c_str());
    //char *conj_str = conjunct.c_str();
    char *pch2 = strtok(conj_str, ">=<");
    std::string part1(pch2);

    char * first_part = new char[part1.length() + 1];
    strcpy(first_part, part1.c_str());
    //	char *first_part = part1.c_str();
    char *pch3 = strtok(first_part, " ");
    char bef = '\0';
    while (pch3 != NULL) {

      if (isdigit(*pch3)) {
        int constant = atoi(pch3);
        if (bef == '-') {
          constant *= (-1);

        }
        if (eq)
          e.update_const(constant);
        if (geq)
          h.update_const(constant);

        bef = '\0';
      } else if (*pch3 == '+') {
        bef = '+';

      } else if (*pch3 == '-') {

        bef = '-';
        if (strlen(pch3) > 1 && isalpha(*(pch3 + 1)))
          pch3 += 1;

      }
      if (isalpha(*pch3)) {

        std::string term(pch3);

        int coef = 1;
        if (bef == '-')
          coef = -1;
        //if (term.find("(") == std::string::npos) {
        bef = '\0';
        bool found_set_var = false;
        for (int k = 0; k < ordered_vars.size(); k++)
          if (ordered_vars[k] == term) {
            found_set_var = true;
            if (eq)
              e.update_coef(R.set_var(k + 1), coef);
            else if (geq)
              h.update_coef(R.set_var(k + 1), coef);
          }

        if (!found_set_var) {
          if (term.find("(") == std::string::npos) {
            bool found2 = false;
            for (int k = 0; k < freevar.size(); k++)
              if (freevar[k]->base_name() == term) {
                Variable_ID v;
                found2 = true;
                if (gathered.find(term) == gathered.end()) {
                  v = R.get_local(freevar[k]);
                  gathered.insert(
                      std::pair<std::string, Variable_ID>(term, v));
                } else
                  v = gathered.find(term)->second;
                if (eq)
                  e.update_coef(v, coef);
                else if (geq)
                  h.update_coef(v, coef);
              }
            if (!found2) {
              Free_Var_Decl *repl = new Free_Var_Decl(term);
              Variable_ID v = R.get_local(repl);
              freevar.push_back(repl);
              gathered.insert(
                  std::pair<std::string, Variable_ID>(term, v));
              if (eq)
                e.update_coef(v, coef);
              else if (geq)
                h.update_coef(v, coef);
            }

          } else {

            int brace_count = 0;
            int offset = 0;
            while (term.find("(", offset) != std::string::npos) {
              brace_count++;
              offset = term.find("(", offset);
              offset += 1;
            }
            offset = 0;
            int count = 0;
            while (brace_count > count) {
              int prev = offset;
              offset = term.find(")", offset);
              if (offset != std::string::npos) {
                count++;
                offset += 1;
              } else {
                offset = prev;

                pch3 = strtok(NULL, " ");

                term += std::string(pch3);
              }

            }

            for (std::map<std::string, std::string>::iterator it =
                unin.begin(); it != unin.end(); it++) {
              if (it->second == term) {

                std::string to_comp = it->first.substr(0,
                                                       it->first.find("("));
                bool found = false;
                bool found2 = false;
                for (int k = 0; k < freevar.size(); k++)

                  if (freevar[k]->base_name() == to_comp) {
                    Variable_ID v;
                    if (gathered.find(term) == gathered.end()) {

                      std::vector<std::string> var;
                      for (int j = 1;
                           j
                           <= const_cast<omega::Relation &>(r2).n_set();
                           j++) {
                        if (term.find(
                            const_cast<omega::Relation &>(r2).set_var(
                                j)->name())
                            != std::string::npos) {
                          var.push_back(
                              const_cast<omega::Relation &>(r2).set_var(
                                  j)->name());
                          //ordered_vars.push_back(*i);
                          found2 = true;
                          break;
                        }
                      }

                      if (found2) {
                        int pos = -1;
                        for (int j = 1; j <= R.n_set(); j++) {

                          if (var[var.size() - 1]
                              == R.set_var(j)->name())
                            pos = j;
                        }
                        bool found3 = false;
                        Variable_ID v2;
                        for (int k1 = 0; k1 < freevar.size();
                             k1++) {
                          if (freevar[k1]->base_name()
                              == to_comp + "p") {
                            found3 = true;
                            v2 = R.get_local(freevar[k1],
                                             Input_Tuple);
                            break;
                          }

                        }
                        if (found3) {
                          if (eq)
                            e.update_coef(v2, coef);
                          else if (geq)
                            h.update_coef(v2, coef);

                        } else {
                          Free_Var_Decl *repl = new Free_Var_Decl(
                              to_comp + "p", pos);
                          v2 = R.get_local(repl, Input_Tuple);

                          std::vector<std::string> args;

                          for (int j = 1; j <= pos; j++) {
                            args.push_back(R.set_var(j)->name());

                          }
                          CG_outputRepr *rhs = ir->RetrieveMacro(
                              to_comp);
                          rhs = rhs->clone();

                          for (int k = 0; k < var.size(); k++) {

                            std::string mirror_var =
                                var[k].substr(0,
                                              var[k].find("p"));

                            CG_outputRepr *exp =
                                ir->builder()->CreateIdent(
                                    var[k]);
                            std::vector<IR_ScalarRef *> refs =
                                ir->FindScalarRef(rhs);
                            for (int l = 0; l < refs.size();
                                 l++) {
                              if (refs[l]->name() == mirror_var)
                                ir->ReplaceExpression(refs[l],
                                                      exp->clone());

                            }
                          }

                          ir->CreateDefineMacro(to_comp + "p",
                                                args, rhs);

                          freevar.push_back(repl);
                          //gathered.insert(
                          //		std::pair<std::string, Variable_ID>(
                          //				to_comp + "p", v2));]
                          if (eq)
                            e.update_coef(v2, coef);
                          else if (geq)
                            h.update_coef(v2, coef);

                        }
                        break;
                      } else {
                        v = R.get_local(freevar[k], Input_Tuple);
                        gathered.insert(
                            std::pair<std::string, Variable_ID>(
                                term, v));
                      }
                    } else
                      v = gathered.find(term)->second;
                    if (eq)
                      e.update_coef(v, coef);
                    else if (geq)
                      h.update_coef(v, coef);
                    found = true;
                    break;
                  }
                if (found || found2)
                  break;
              }
            }
          }
        }

      }
      if (pch3 != NULL)
        pch3 = strtok(NULL, " ");
    }

    //pch = strtok(NULL, "&");
  }
  R.simplify();
// current conjunct  strtok with "&&"
// switch  on  identifier, "=", ">=", number
// add UF logic later..
  return R;

}

bool checkIfEqual(omega::Relation &r, omega::Relation s) {

  std::vector<std::set<std::pair<int, int> > > output_vars;
  std::vector<std::set<std::pair<int, int> > > input_vars;
  std::vector<std::set<std::pair<int, std::string> > > global_vars;
  std::vector<int> constants;
  for (DNF_Iterator di(const_cast<omega::Relation &>(r).query_DNF()); di; di++) {
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++) {
      int constant = (*gi).get_const();
      constants.push_back(constant);
      std::set<std::pair<int, int> > ov;
      std::set<std::pair<int, int> > iv;
      std::set<std::pair<int, std::string> > gv;
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();

        switch (v->kind()) {
          case Output_Var:

            ov.insert(
                std::pair<int, int>(v->get_position(), cvi.curr_coef()));

            break;
          case Input_Var:

            iv.insert(
                std::pair<int, int>(v->get_position(), cvi.curr_coef()));

            break;
          case Wildcard_Var: {
            throw loop_error(
                "wildcard variable not handled in equality constraint for level set parallelization");
            break;
          }
          case Global_Var: {
            Global_Var_ID g = v->get_global_var();
            gv.insert(
                std::pair<int, std::string>(cvi.curr_coef(),
                                            (const char *)(g->base_name())));

            break;
          }
          default:
            break;
        }
      }
      output_vars.push_back(ov);
      input_vars.push_back(iv);
      global_vars.push_back(gv);
    }
  }

  std::vector<std::set<std::pair<int, int> > > output_vars2;
  std::vector<std::set<std::pair<int, int> > > input_vars2;
  std::vector<std::set<std::pair<int, std::string> > > global_vars2;
  std::vector<int> constants2;
  for (DNF_Iterator di(const_cast<omega::Relation &>(s).query_DNF()); di; di++) {
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++) {
      int constant = (*gi).get_const();
      constants2.push_back(constant);
      std::set<std::pair<int, int> > ov;
      std::set<std::pair<int, int> > iv;
      std::set<std::pair<int, std::string> > gv;
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();

        switch (v->kind()) {
          case Output_Var:

            ov.insert(
                std::pair<int, int>(v->get_position(), cvi.curr_coef()));

            break;
          case Input_Var:

            iv.insert(
                std::pair<int, int>(v->get_position(), cvi.curr_coef()));

            break;
          case Wildcard_Var: {
            throw loop_error(
                "wildcard variable not handled ine quality constraint for level set parallelization");
            break;
          }
          case Global_Var: {
            Global_Var_ID g = v->get_global_var();
            gv.insert(
                std::pair<int, std::string>(cvi.curr_coef(),
                                            (const char *)(g->base_name())));

            break;
          }
          default:
            break;
        }
      }
      output_vars2.push_back(ov);
      input_vars2.push_back(iv);
      global_vars2.push_back(gv);
    }
  }
  if (output_vars.size() != output_vars2.size())
    return false;

  for (int i = 0; i < output_vars.size(); i++) {
    bool found = false;

    for (int k = 0; k < output_vars2.size(); k++)
      if (output_vars2[k] == output_vars[i])
        found = true;

    if (!found)
      return false;
  }

  if (input_vars.size() != input_vars2.size())
    return false;

  for (int i = 0; i < input_vars.size(); i++) {
    bool found = false;

    for (int k = 0; k < input_vars2.size(); k++)
      if (input_vars2[k] == input_vars[i])
        found = true;

    if (!found)
      return false;
  }
  if (constants.size() != constants2.size())
    return false;

  for (int i = 0; i < constants.size(); i++) {
    bool found = false;

    for (int k = 0; k < constants2.size(); k++)
      if (constants[k] == constants[i])
        found = true;

    if (!found)
      return false;
  }
  if (global_vars.size() != global_vars2.size())
    return false;

  for (int i = 0; i < global_vars.size(); i++) {
    bool found = false;

    for (int k = 0; k < global_vars2.size(); k++)
      if (global_vars2[k] == global_vars[i])
        found = true;

    if (!found)
      return false;
  }

  return true;
}

std::map<int, omega::Relation> removeRedundantConstraints(
    std::map<int, omega::Relation> &rels) {

  std::map<int, omega::Relation> to_return;
  for (std::map<int, omega::Relation>::iterator i = rels.begin(); i != rels.end();
       i++) {
    bool found = false;
    if (i->second.is_finalized()) {

      for (std::map<int, omega::Relation>::iterator j = rels.begin();
           j != rels.end(); j++) {
        if (j->first > i->first) {

          found |= checkIfEqual(i->second, j->second);
          found |= checkIfEqual(i->second, j->second);
        }
      }
      if (!found) {
        to_return.insert(std::pair<int, omega::Relation>(i->first, i->second));

      }

    }

    /*	if (rels[i].second.is_finalized()) {


     for (int j = 0; j < rels.size(); j++) {

     if (i != j) {
     found |= checkIfEqual(rels[i].second, rels[j].first);
     found |= checkIfEqual(rels[i].second, rels[j].second);
     }
     }
     if (!found){
     to_return.push_back(rels[i]);
     }
     }
     */
  }

  return to_return;

}

std::string get_lb_string(omega::Relation &result, Variable_ID &v2,
                          bool range = true) {
  std::string a = "";
  Global_Var_ID uf;

  for (DNF_Iterator di(const_cast<omega::Relation &>(result).query_DNF()); di; di++) {
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      bool found_c = false;
      bool dont_consider = false;
      std::pair<int, int> coefs = std::pair<int, int>(0, 0);
      int constant = (*gi).get_const();
      std::string other_global = "";
      bool found_uf = false;
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
          case Input_Var: {
            if (v2->kind() == Input_Var) {
              if (v->name() == v2->name() && cvi.curr_coef() == 1)
                found_c = true;
              else
                dont_consider = true;
            } else {

              dont_consider = true;

            }

            break;
          }

          case Global_Var: {

            Global_Var_ID g = v->get_global_var();
            if (g->arity() > 0) {
              std::string name = g->base_name();
              if (v2->kind() == Global_Var) {
                if (name == v2->get_global_var()->base_name()
                    && cvi.curr_coef() == 1)
                  found_c = true;
                else
                  dont_consider = true;
              } else {
                if (other_global == "" && cvi.curr_coef() < 0) {
                  other_global = name;
                  uf = g;
                  found_uf = true;
                } else
                  dont_consider = true;
              }
            } else {
              if (other_global == "" & cvi.curr_coef() < 0) {
                std::string name = g->base_name();
                other_global = name;
              } else
                dont_consider = true;
            }
            break;
          }
          default:
            dont_consider = true;
            break;
        }
      }
      if (!dont_consider && found_c) {
        if (found_uf) {

          Variable_ID v2 = result.get_local(uf, Input_Tuple);
          return get_lb_string(result, v2, range);

        }

        if (constant == 0 && other_global == "") {
          if (found_c) {
            // a = monotonic3[i].first.substr(0, monotonic3[i].first.find("_"));
            if (range)
              a = std::string("0 <= ") + std::string("j");
            else
              a = std::string("0 <= ") + std::string("i");
          }
        } else if (constant > 0 && other_global == "") {
          if (found_c) {
            std::ostringstream convert;
            convert << constant;
            //  a = monotonic3[i].first.substr(0, monotonic3[i].first.find("_"));
            if (range)
              a = convert.str() + std::string(" <= ") + std::string("j");
            else
              a = convert.str() + std::string(" <= ") + std::string("i");
          }

        } else if (constant < 0 && other_global == "") {
          if (found_c) {
            std::ostringstream convert;
            convert << constant;
            //		  a = monotonic3[i].first.substr(0, monotonic3[i].first.find("_"));
            if (range)
              a = convert.str() + std::string(" <= ") + std::string("j");
            else
              a = convert.str() + std::string(" <= ") + std::string("i");
          }

        } else if (other_global != "") {
          if (constant > 0)
            constant *= -1;
          std::ostringstream convert;
          convert << constant;
          a = other_global + convert.str();
          if (range)
            a = a + std::string(" <= ") + std::string("j");
          else
            a = a + std::string(" <= ") + std::string("i");
        }

        return a;
      }
    }
  }
  return a;
}

std::string get_ub_string(omega::Relation &result, Variable_ID &v2,
                          bool range = true) {
  std::string b = "";
  Global_Var_ID uf;

  for (DNF_Iterator di(const_cast<omega::Relation &>(result).query_DNF()); di; di++) {
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      bool found_c = false;
      bool dont_consider = false;
      std::pair<int, int> coefs = std::pair<int, int>(0, 0);
      int constant = (*gi).get_const();
      bool found_uf = false;
      std::string other_global = "";
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();

        switch (v->kind()) {
          case Input_Var: {
            if (v2->kind() == Input_Var) {
              if (v->name() == v2->name() && cvi.curr_coef() == -1)
                found_c = true;
              else
                dont_consider = true;
            } else
              dont_consider = true;
            break;
          }

          case Global_Var: {

            Global_Var_ID g = v->get_global_var();
            if (g->arity() > 0) {
              std::string name = g->base_name();
              if (v2->kind() == Global_Var) {
                if (name == v2->get_global_var()->base_name()
                    && cvi.curr_coef() == -1)
                  found_c = true;
                else
                  dont_consider = true;
              } else {
                if (other_global == "" && cvi.curr_coef() > 0) {
                  std::string s = g->base_name();
                  other_global = s;
                  uf = g;
                  found_uf = true;
                } else
                  dont_consider = true;
              }
            } else {
              if (other_global == "" && cvi.curr_coef() > 0) {
                std::string s = g->base_name();
                other_global = s;
              } else
                dont_consider = true;
            }
            break;
          }
          default:
            dont_consider = true;
            break;
        }
      }

      if (!dont_consider && found_c) {
        if (found_uf) {

          Variable_ID v2 = result.get_local(uf, Input_Tuple);
          return get_ub_string(result, v2, range);

        }
        b = "";

        if (constant < 0) {
          std::ostringstream convert;
          convert << constant;
          b += convert.str();
        } else {
          std::ostringstream convert;
          convert << constant;
          b += "+" + convert.str();

        }

        if (constant == 0) {

          if (range)
            b = std::string("j") + std::string(" <=") + other_global;
          else
            b = std::string("i") + std::string(" <=") + other_global;
        } else if (constant == -1) {
          if (range)
            b = std::string("j") + std::string(" < ") + other_global;
          else
            b = std::string("i") + std::string(" < ") + other_global;

        } else {

          if (range)
            b = std::string("j") + std::string(" <=") + other_global;
          else
            b = std::string("i") + std::string(" <=") + other_global;

        }

        return b;
      }
    }

  }
  return b;
}

std::pair<std::vector<std::string>,
    std::pair<
        std::vector<std::pair<bool, std::pair<std::string, std::string> > >,
        std::vector<std::pair<bool, std::pair<std::string, std::string> > > > > determineUFsForIegen(
    omega::Relation &result, IR_Code *ir, std::vector<Free_Var_Decl *> freevar) {

  std::vector<std::pair<bool, std::pair<std::string, std::string> > > to_return;
  std::vector<std::pair<bool, std::pair<std::string, std::string> > > to_return2;
  //outer most function call Eg. rowptr(colidx(k)) would be just rowptr
  //would be stored as rowptr_colidx__ so just parse till first '_' for base UF name
  std::set<std::string> outer_most_ufs2;

  for (DNF_Iterator di(const_cast<omega::Relation &>(result).query_DNF()); di; di++) {
    for (Constraint_Iterator gi((*di)->constraints()); gi; gi++) {
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();

        switch (v->kind()) {
          case Global_Var: {

            Global_Var_ID g = v->get_global_var();
            if (g->arity() > 0) {
              std::string name = g->base_name();
              char * name_str = new char[name.length() + 1];
              strcpy(name_str, name.c_str());
              //char *conj_str = conjunct.c_str();
              char *pch2 = strtok(name_str, "_");
              if (pch2) {
                pch2 = strtok(NULL, "_");
                //if nothing after '_'
                if (!pch2)
                  outer_most_ufs2.insert(name);
              }
            }
            break;
          }
          default:
            break;
        }
      }
    }
  }
  std::vector<std::string> outer_most_ufs;
  for (std::set<string>::iterator i = outer_most_ufs2.begin();
       i != outer_most_ufs2.end(); i++) {

    outer_most_ufs.push_back(*i);

  }

  std::vector<std::pair<std::string, std::pair<std::string, std::string> > > monotonic;
  for (int i = 0; i < outer_most_ufs.size(); i++)
    for (int j = i + 1; j < outer_most_ufs.size(); j++) {
      for (DNF_Iterator di(const_cast<omega::Relation &>(result).query_DNF()); di;
           di++) {
        for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
          bool found_c = false;
          bool dont_consider = false;
          std::pair<int, int> coefs = std::pair<int, int>(0, 0);
          int constant = (*gi).get_const();
          for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();

            switch (v->kind()) {
              case Global_Var: {

                Global_Var_ID g = v->get_global_var();
                if (g->arity() > 0) {
                  std::string name = g->base_name();
                  if (name == outer_most_ufs[i])
                    coefs.first = cvi.curr_coef();
                  else if (name == outer_most_ufs[j])
                    coefs.second = cvi.curr_coef();

                } else
                  dont_consider = true;

                break;
              }
              default:
                dont_consider = true;
                break;
            }
          }
          if (!dont_consider) {
            if (coefs.first == -1 && coefs.second == 1) {
              if (constant == -1)
                monotonic.push_back(
                    std::pair<std::string,
                        std::pair<std::string, std::string> >("<",
                                                              std::pair<std::string, std::string>(
                                                                  outer_most_ufs[i],
                                                                  outer_most_ufs[j])));
              else if (constant == 0)
                monotonic.push_back(
                    std::pair<std::string,
                        std::pair<std::string, std::string> >("<=",
                                                              std::pair<std::string, std::string>(
                                                                  outer_most_ufs[i],
                                                                  outer_most_ufs[j])));

            } else if (coefs.first == 1 && coefs.second == -1) {

              if (constant == -1)
                monotonic.push_back(
                    std::pair<std::string,
                        std::pair<std::string, std::string> >("<",
                                                              std::pair<std::string, std::string>(
                                                                  outer_most_ufs[j],
                                                                  outer_most_ufs[i])));
              else if (constant == 0)
                monotonic.push_back(
                    std::pair<std::string,
                        std::pair<std::string, std::string> >("<=",
                                                              std::pair<std::string, std::string>(
                                                                  outer_most_ufs[j],
                                                                  outer_most_ufs[i])));

            }

          }
        }
      }

    }

  std::set<std::string> monotonic2;
  std::set<std::string> not_monotonic2;
  std::vector<std::pair<std::string, std::string> > monotonic3;
  std::set<std::string> other_constraints;

  for (int i = 0; i < monotonic.size(); i++) {
    std::string first = monotonic[i].second.first;
    std::string second = monotonic[i].second.second;

    std::string a = first.substr(0, first.find("_"));
    std::string b = second.substr(0, second.find("_"));
    if (a == b) {
      CG_outputRepr *rep = ir->RetrieveMacro(first);
      std::vector<IR_PointerArrayRef *> refs = ir->FindPointerArrayRef(rep);
      rep = NULL;
      if (refs.size() > 0) {
        rep = refs[0]->index(0);
      }
      CG_outputRepr *rep2 = ir->RetrieveMacro(second);
      std::vector<IR_PointerArrayRef *> refs2 = ir->FindPointerArrayRef(
          rep2);
      rep2 = NULL;
      if (refs2.size() > 0) {
        rep2 = refs2[0]->index(0);
      }

      if (ir->QueryExpOperation(rep) == IR_OP_VARIABLE
          || ir->QueryExpOperation(rep) == IR_OP_MULTIPLY) {
        std::string one;
        if (ir->QueryExpOperation(ir->QueryExpOperand(rep)[0])
            == IR_OP_VARIABLE)
          one = ir->Repr2Ref(ir->QueryExpOperand(rep)[0])->name();
        if (ir->QueryExpOperation(ir->QueryExpOperand(rep)[1])
            == IR_OP_VARIABLE)
          one = ir->Repr2Ref(ir->QueryExpOperand(rep)[1])->name();

        if (ir->QueryExpOperation(rep2) == IR_OP_PLUS) {
          std::string two;

          if (ir->QueryExpOperation(ir->QueryExpOperand(rep2)[0])
              == IR_OP_MULTIPLY) {
            CG_outputRepr *tmp = ir->QueryExpOperand(rep2)[0];
            if (ir->QueryExpOperation(ir->QueryExpOperand(tmp)[0])
                == IR_OP_VARIABLE)
              two = ir->Repr2Ref(ir->QueryExpOperand(tmp)[0])->name();
            else
              two = ir->Repr2Ref(ir->QueryExpOperand(tmp)[1])->name();
          } else if (ir->QueryExpOperation(ir->QueryExpOperand(rep)[0])
                     == IR_OP_VARIABLE) {
            two = ir->Repr2Ref(ir->QueryExpOperand(rep2)[0])->name();
          }
          if (ir->QueryExpOperation(ir->QueryExpOperand(rep2)[1])
              == IR_OP_MULTIPLY) {
            CG_outputRepr *tmp = ir->QueryExpOperand(rep2)[1];
            if (ir->QueryExpOperation(ir->QueryExpOperand(tmp)[0])
                == IR_OP_VARIABLE)
              two = ir->Repr2Ref(ir->QueryExpOperand(tmp)[0])->name();
            else
              two = ir->Repr2Ref(ir->QueryExpOperand(tmp)[1])->name();
          } else if (ir->QueryExpOperation(ir->QueryExpOperand(rep2)[1])
                     == IR_OP_VARIABLE) {
            two = ir->Repr2Ref(ir->QueryExpOperand(rep2)[1])->name();

          }

          if (one == two) {

            if (ir->QueryExpOperation(ir->QueryExpOperand(rep2)[1])
                == IR_OP_CONSTANT) {
              if (static_cast<IR_ConstantRef *>(ir->Repr2Ref(
                  ir->QueryExpOperand(rep2)[1]))->integer() == 1) {

                monotonic2.insert(a);
                monotonic3.push_back(
                    std::pair<std::string, std::string>(first,
                                                        second));

              }
            } else if (ir->QueryExpOperation(ir->QueryExpOperand(rep2)[0])
                       == IR_OP_CONSTANT) {
              if (static_cast<IR_ConstantRef *>(ir->Repr2Ref(
                  ir->QueryExpOperand(rep2)[0]))->integer() == 1) {

                monotonic2.insert(a);
                monotonic3.push_back(
                    std::pair<std::string, std::string>(first,
                                                        second));
              }
            }
          }
        }
      } else if (ir->QueryExpOperation(rep2) == IR_OP_VARIABLE
                 || ir->QueryExpOperation(rep2) == IR_OP_MULTIPLY) {
        std::string one;
        if (ir->QueryExpOperation(ir->QueryExpOperand(rep2)[0])
            == IR_OP_VARIABLE)
          one = ir->Repr2Ref(ir->QueryExpOperand(rep2)[0])->name();
        if (ir->QueryExpOperation(ir->QueryExpOperand(rep2)[1])
            == IR_OP_VARIABLE)
          one = ir->Repr2Ref(ir->QueryExpOperand(rep2)[1])->name();

        if (ir->QueryExpOperation(rep) == IR_OP_PLUS) {
          std::string two;

          if (ir->QueryExpOperation(ir->QueryExpOperand(rep)[0])
              == IR_OP_MULTIPLY) {
            CG_outputRepr *tmp = ir->QueryExpOperand(rep)[0];
            if (ir->QueryExpOperation(ir->QueryExpOperand(tmp)[0])
                == IR_OP_VARIABLE)
              two = ir->Repr2Ref(ir->QueryExpOperand(tmp)[0])->name();
            else
              two = ir->Repr2Ref(ir->QueryExpOperand(tmp)[1])->name();
          } else if (ir->QueryExpOperation(ir->QueryExpOperand(rep)[0])
                     == IR_OP_VARIABLE) {
            two = ir->Repr2Ref(ir->QueryExpOperand(rep)[0])->name();
          }
          if (ir->QueryExpOperation(ir->QueryExpOperand(rep)[1])
              == IR_OP_MULTIPLY) {
            CG_outputRepr *tmp = ir->QueryExpOperand(rep)[1];
            if (ir->QueryExpOperation(ir->QueryExpOperand(tmp)[0])
                == IR_OP_VARIABLE)
              two = ir->Repr2Ref(ir->QueryExpOperand(tmp)[0])->name();
            else
              two = ir->Repr2Ref(ir->QueryExpOperand(tmp)[1])->name();
          } else if (ir->QueryExpOperation(ir->QueryExpOperand(rep)[1])
                     == IR_OP_VARIABLE) {
            two = ir->Repr2Ref(ir->QueryExpOperand(rep)[1])->name();

          }

          if (one == two) {

            if (ir->QueryExpOperation(ir->QueryExpOperand(rep)[1])
                == IR_OP_CONSTANT) {
              if (static_cast<IR_ConstantRef *>(ir->Repr2Ref(
                  ir->QueryExpOperand(rep)[1]))->integer() == 1) {

                monotonic2.insert(a);
                monotonic3.push_back(
                    std::pair<std::string, std::string>(first,
                                                        second));

              }
            } else if (ir->QueryExpOperation(ir->QueryExpOperand(rep)[0])
                       == IR_OP_CONSTANT) {
              if (static_cast<IR_ConstantRef *>(ir->Repr2Ref(
                  ir->QueryExpOperand(rep)[0]))->integer() == 1) {

                monotonic2.insert(a);
                monotonic3.push_back(
                    std::pair<std::string, std::string>(first,
                                                        second));
              }
            }
          }
        }

        /*	if (ir->QueryExpOperation(rep2) == IR_OP_VARIABLE) {
         std::string one =
         ir->Repr2Ref(ir->QueryExpOperand(rep2)[0])->name();
         if (ir->QueryExpOperation(rep) == IR_OP_PLUS) {
         std::string two =
         ir->Repr2Ref(ir->QueryExpOperand(rep)[0])->name();
         if (one == two)
         if (static_cast<IR_ConstantRef *>(ir->Repr2Ref(
         ir->QueryExpOperand(rep)[1]))->integer() == 1) {

         monotonic2.insert(a);
         monotonic3.push_back(
         std::pair<std::string, std::string>(second,
         first));
         }
         }
         }
         }*/

      }
    }

    else {
      other_constraints.insert(a + " " + monotonic[i].first + " " + b);

    }
  }
  for (int i = 0; i < outer_most_ufs.size(); i++) {
    bool found = false;
    for (int j = 0; j < monotonic3.size(); j++) {
      if (monotonic3[j].first == outer_most_ufs[i]
          || monotonic3[j].second == outer_most_ufs[i]) {
        found = true;
        break;
      }

    }
    if (!found) {
      //std::string a = outer_most_ufs[i].substr(0,outer_most_ufs[i].find("_") );
      not_monotonic2.insert(outer_most_ufs[i]);

    }

  }

  //range info
  for (std::set<std::string>::iterator i = not_monotonic2.begin();
       i != not_monotonic2.end(); i++) {
    std::string lb, ub;
    Free_Var_Decl *d = 0;
    for (int j = 0; j < freevar.size(); j++)
      if (freevar[j]->base_name() == *i) {
        d = freevar[j];
        break;

      }

    assert(d);
    Variable_ID v = result.get_local(d, Input_Tuple);

    lb = get_lb_string(result, v);
    ub = get_ub_string(result, v);

    to_return.push_back(
        std::pair<bool, std::pair<std::string, std::string> >(false,
                                                              std::pair<std::string, std::string>(
                                                                  (*i).substr(0, (*i).find("_")), lb + " &&" + ub)));
    int arity = d->arity();
    Variable_ID var = result.set_var(arity);
    lb = get_lb_string(result, var, false);
    ub = get_ub_string(result, var, false);

    to_return2.push_back(
        std::pair<bool, std::pair<std::string, std::string> >(false,
                                                              std::pair<std::string, std::string>(
                                                                  (*i).substr(0, (*i).find("_")), lb + " &&" + ub)));
  }
  //range info
  for (int i = 0; i < monotonic3.size(); i++) {
    Free_Var_Decl *d = 0;
    Free_Var_Decl *e = 0;
    for (int j = 0; j < freevar.size(); j++) {
      if (freevar[j]->base_name() == monotonic3[i].first) {
        d = freevar[j];
        //break;

      } else if (freevar[j]->base_name() == monotonic3[i].second) {
        e = freevar[j];
        //break;

      }
      if (e != NULL && d != NULL)
        break;
    }

    assert(d);
    assert(e);
    Variable_ID v = result.get_local(d, Input_Tuple);
    Variable_ID v2 = result.get_local(e, Input_Tuple);
    std::string a = get_lb_string(result, v);
    std::string b = get_ub_string(result, v2);
    to_return.push_back(
        std::pair<bool, std::pair<std::string, std::string> >(true,
                                                              std::pair<std::string, std::string>(
                                                                  monotonic3[i].first.substr(0,
                                                                                             monotonic3[i].first.find("_")), a + " &&" + b)));

    int arity = d->arity();
    Variable_ID var = result.set_var(arity);
    std::string lb = get_lb_string(result, var, false);
    std::string ub = get_ub_string(result, var, false);
    //  ub = ub + "+ 1";
    ub = ub.replace(ub.find("<"), 2, "<=");

    to_return2.push_back(
        std::pair<bool, std::pair<std::string, std::string> >(true,
                                                              std::pair<std::string, std::string>(
                                                                  monotonic3[i].first.substr(0,
                                                                                             (monotonic3[i].first).find("_")),
                                                                  lb + " &&" + ub)));
  }
  //domain
  std::vector<string> other_constraints2;
  for (std::set<string>::iterator i = other_constraints.begin();
       i != other_constraints.end(); i++)
    other_constraints2.push_back(*i);
  return (std::pair<std::vector<std::string>,
      std::pair<
          std::vector<std::pair<bool, std::pair<std::string, std::string> > >,
          std::vector<std::pair<bool, std::pair<std::string, std::string> > > > >(
      other_constraints2,
      std::pair<
          std::vector<std::pair<bool, std::pair<std::string, std::string> > >,
          std::vector<std::pair<bool, std::pair<std::string, std::string> > > >(
          to_return, to_return2)));

}

omega::Relation Loop::parseExpWithWhileToRel(omega::CG_outputRepr *repr,
                                             omega::Relation &R, int loc) {

  std::vector<IR_Loop *> loops = ir->FindLoops(repr);
  int count = 0;
  for (int i = 0; i < loops.size(); i++)
    // TODO assume each loop uses 1 index
    count ++;

  omega::Relation r(count + R.n_set());
  F_And *f_root = r.add_and();
  std::map<int, int> pos_map;

  for (int i = 1; i <= R.n_set(); i++) {
    pos_map.insert(std::pair<int, int>(i, i));
  }

  for (int i = 1; i <= R.n_set(); i++) {
    r.name_set_var(i, R.set_var(i)->name());
    r = replace_set_var_as_another_set_var(r, R, i, i, pos_map);
  }

  count = 1 + R.n_set();
  f_root = r.and_with_and();
  for (int i = 0; i < loops.size(); i++) {
    r.name_set_var(count, loops[i]->index()->name());
    Variable_ID v = r.set_var(count++);
    CG_outputRepr *lb = loops[i]->lower_bound();

    exp2formula(this, ir, r, f_root, freevar, lb, v, 's', IR_COND_GE, true,
                uninterpreted_symbols[loc],
                uninterpreted_symbols_stringrepr[loc], unin_rel[loc]);
    CG_outputRepr *ub = loops[i]->upper_bound();

    IR_CONDITION_TYPE cond = loops[i]->stop_cond();
    if (cond == IR_COND_LT || cond == IR_COND_LE)
      exp2formula(this, ir, r, f_root, freevar, ub, v, 's', cond, true,
                  uninterpreted_symbols[loc],
                  uninterpreted_symbols_stringrepr[loc], unin_rel[loc]);

  }

  return r;

}



/*!
 * Mahdi: This functions extarcts and returns the data dependence relations 
 * that are needed for generating inspectors for wavefront paralleization of a 
 * specific loop level
 * Loop levels start with 0 (being outer most loop), outer most loop is the default
 * Input:  loop level for parallelization
 * Output: dependence relations in teh form of strings that are in ISL (IEGenLib) syntax  
 */
std::vector<std::pair<std::string, std::string >> 
 Loop::depRelsForParallelization(int parallelLoopLevel){

  int stmt_num = 1, level = 1, whileLoop_stmt_num = 1;

  // Mahdi: a temporary hack for getting dependence extraction changes integrated
  replaceCode_ind = 0;

//std::cout<<"Start of printDependenceUFs!\n";
//std::cout<<"\n|_|-|_| START UFCs:\n";
//  for (std::map<std::string, std::string >::iterator it=unin_symbol_for_iegen.begin(); it!=unin_symbol_for_iegen.end(); ++it)
//    std::cout<<"\n *****UFS = " << it->first << " => " << it->second << '\n';
//for(int i = 0; i<stmt.size() ; i++){
//  std::cout<<"\nR "<<i<<" = "<<stmt[i].IS<<"\nXform = "<<stmt[i].xform<<"\nSt NL = "<<stmt_nesting_level_[i]<<"\n";
//}

  omega::Relation r = parseExpWithWhileToRel(stmt[whileLoop_stmt_num].code, 
                                             stmt[whileLoop_stmt_num].IS, stmt_num);

  r.simplify();

  // Adding some extra constraints that are input from chill script and stored in Loop::known
  omega::Relation result;
  if (known.n_set() < r.n_set()) {
    omega::Relation known = omega::Extend_Set(omega::copy(this->known),
                                              r.n_set() - this->known.n_set());
    result = omega::Intersection(copy(known), copy(r));
  } else {
    omega::Relation tmp = omega::Extend_Set(omega::copy(r),
                                     this->known.n_set() - r.n_set());
    result = omega::Intersection(tmp, copy(known));
  }
  result.simplify();

  bool is_negative = false;
  Variable_ID v2 = r.set_var(1);
  //need to correct following logic
  for (DNF_Iterator di(const_cast<omega::Relation &>(r).query_DNF()); di; di++)
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++)
      if ((*gi).is_const(v2))
        if ((*gi).get_coef(v2) < 0 && (*gi).get_const() == 0)
          is_negative = true;
  if (is_negative) {
    r = flip_var(r, v2);	//convert to positive_IS
    Variable_ID v3 = result.set_var(1);
    result = flip_var(result, v3);
  }

  // FIXME: Mahdi: Do we need following? Seems to me all the UFCs shold have already been 
  //               stored in Loop::dep_rel_for_iegen in previous phases.
  // UFCs and their representations in omega::relations are extracted and stored in
  // Loop::dep_rel_for_iegen. We can later use them to generate IEGenLib(ISL)::Relations
  // out of omeg::Relations and vice versa. We also need symbolic constants for generating
  // ISL relations specifically.
  std::pair<std::vector<std::string>,
      std::pair<std::vector<std::pair<bool, std::pair<std::string, std::string> > >,
          std::vector<std::pair<bool, std::pair<std::string, std::string> > > > > syms =
      determineUFsForIegen(result, ir, freevar);


  // Getting the statement number for all statements in the parallel loop
  int first_stmt_in_parallel_loop = 0;
  for(int i = 0; i<stmt.size() ; i++){
    if (stmt_nesting_level_[i] == parallelLoopLevel)
      first_stmt_in_parallel_loop = i;
  }
  std::vector<int> lex_ = getLexicalOrder(first_stmt_in_parallel_loop);
  std::set<int> same_loop_ = getStatements(lex_, level - 1);

  // Store all accesses in all the statements of the parallel loop level in access
  std::vector<IR_ArrayRef *> access;
  int access_st[100]={-1},ct=0;
  for (std::set<int>::iterator i = same_loop_.begin(); i != same_loop_.end();
       i++) {
    std::vector<IR_ArrayRef *> access2 = ir->FindArrayRef(stmt[*i].code);
    for (int j = 0; j < access2.size(); j++){
      access.push_back(access2[j]);
      access_st[ct++] = *i; 
    }
  }

int relCounter = 1;

  // Checking pairs of all accesses for comming up with data access equalities.
  std::vector<std::pair< std::pair<omega::Relation, omega::Relation> , omega::Relation>> dep_relation_;
  std::map<int, omega::Relation> rels;
  std::vector<std::pair<omega::Relation, omega::Relation> > rels2;
  for (int i = 0; i < access.size(); i++) {
    IR_ArrayRef *a = access[i];
    IR_ArraySymbol *sym_a = a->symbol();
//std::cout<<"\ni #"<<i<<" : "<<*sym_a<<"\n";
    for (int j = i; j < access.size(); j++) {
      IR_ArrayRef *b = access[j];
      IR_ArraySymbol *sym_b = b->symbol();

      if (*sym_a == *sym_b && (a->is_write() || b->is_write())) {
        omega::Relation write_r, read_r;
        if ( a->is_write() ) {
          write_r = stmt[access_st[i]].IS;
          read_r = stmt[access_st[j]].IS;
        } else {
          write_r = stmt[access_st[j]].IS;
          read_r = stmt[access_st[i]].IS;
        }
        omega::Relation r1(write_r.n_set(), read_r.n_set());
        for (int i = 1; i <= r.n_set(); i++)
          r1.name_input_var(i, write_r.set_var(i)->name());

        for (int i = 1; i <= r.n_set(); i++)
          r1.name_output_var(i, read_r.set_var(i)->name() + "p");

//std::cout<<"\n#############Initial r1#"<<relCounter<<" = "<<r1<<"  write_r = "<<write_r<<"  read_r = "<<read_r<<"\n";

        F_And *f_root = r1.add_and(); 

        // Mahdi: The reason behind the loop over n_dim is to have equalities between 
        // different indices of array accesses; e.g y[i][col[j]] and y[k][l] -> { i = k and col(j) = l }
        for (int i = 0; i < a->n_dim(); i++) {
          F_Exists *f_exists = f_root->add_exists();
          Variable_ID e1 = f_exists->declare(tmp_e());
          Variable_ID e2 = f_exists->declare(tmp_e());
          F_And *f_and = f_exists->add_and();

          CG_outputRepr *repr_src; // = a->index(i);
          CG_outputRepr *repr_dst; // = b->index(i);
          if ( a->is_write() ) {  
            repr_src = a->index(i);
            repr_dst = b->index(i);

          } else {
            repr_src = b->index(i);
            repr_dst = a->index(i);
          }
          bool has_complex_formula = false;

//std::cout<<"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Before exp2for r1 = "<<r1<<"\n";
//repr_src->dump();
//repr_dst->dump();
          try {   
            exp2formula(this, ir, r1, f_and, freevar, repr_src, e1, 'w',
                        IR_COND_EQ, false, uninterpreted_symbols[stmt_num],
                        uninterpreted_symbols_stringrepr[stmt_num],
                        unin_rel[stmt_num]);
//std::cout<<"After 1st exp2for r1 = "<<r1<<"   E1 = "<<e1->name()<<"\n";
            exp2formula(this, ir, r1, f_and, freevar, repr_dst, e2, 'r',
                        IR_COND_EQ, false, uninterpreted_symbols[stmt_num],
                        uninterpreted_symbols_stringrepr[stmt_num],
                        unin_rel[stmt_num]);
//std::cout<<"After 2nd exp2for r1 = "<<r1<<"   E2 = "<<e2->char_name()<<"\n";
          } catch (const ir_exp_error &e) {
            has_complex_formula = true;
          }

          if (!has_complex_formula) {
            EQ_Handle h = f_and->add_EQ();
            h.update_coef(e1, -1);
            h.update_coef(e2, 1);
          }
//std::cout<<"\nAfter creating the equality r1 = "<<r1<<"\n";

          repr_src->clear();
          repr_dst->clear();
          delete repr_src;
          delete repr_dst;
        }   // Mahdi: end of n_dim loop

        r1.simplify();
        if (is_negative) {
          Variable_ID v1, v2;

          v1 = r1.input_var(1);
          r1 = flip_var_exclusive(r1, v1);
          v2 = r1.output_var(1);
          r1 = flip_var_exclusive(r1, v2);
        }

        rels.insert(std::pair<int, omega::Relation>(dep_relation_.size(), copy(r1)));
        dep_relation_.push_back(
            std::pair<std::pair<omega::Relation, omega::Relation>, omega::Relation>( 
                   (std::pair<omega::Relation, omega::Relation>(write_r, read_r)) , r1) );

//std::cout<<"\nEnd of access check loop: r#"<<relCounter++<<" = "<<r1<<" \nwrite________r = "<<write_r<<"read________r = "<<read_r<<"\n";
      }
    }
  }

relCounter = 1;

  // The loop that creates the relations for IEGen
  for (int i = 0; i < dep_relation_.size(); i++) {
    omega::Relation write_constraints = copy(dep_relation_[i].first.first);
    omega::Relation read_constraints(dep_relation_[i].first.second.n_set()); 
    omega::Relation equality_constraints = copy(dep_relation_[i].second);

    for (int j = 1; j <= read_constraints.n_set(); j++){
      read_constraints.name_set_var(j, dep_relation_[i].first.second.set_var(j)->name() + "p");
    }
    read_constraints.copy_names(read_constraints);
    read_constraints.setup_names();

    F_And *f_root = read_constraints.add_and();

    std::map<int, int> pos_map;
    for (int j = 1; j <= read_constraints.n_set(); j++) {
      pos_map.insert(std::pair<int, int>(i, i));
    }
//std::cout<<"\n----^^^----Before replace Read Constraints #"<<relCounter<<" = "<<dep_relation_[i].first.second<<"\n";
//    for (int j = 1; j <= read_constraints.n_set(); j++) {
     read_constraints = replace_set_vars(read_constraints, 
                                  dep_relation_[i].first.second, relCounter);//, j, j, pos_map);
//    }
    read_constraints.simplify();
//std::cout<<"----^^^----After replace Read Constraints #"<<relCounter<<" = "<<read_constraints<<"\nwrite = "<<write_constraints<<"\n";

    std::string tuple_decl_RAW = "[";
    for (int j = 1; j <= write_constraints.n_set(); j++) {
      if (j > 1)
        tuple_decl_RAW += ",";
      tuple_decl_RAW += write_constraints.set_var(j)->name();
    }
    tuple_decl_RAW += "] -> [";
    for (int j = 1; j <= read_constraints.n_set(); j++) {
      if (j > 1)
        tuple_decl_RAW += ",";
      tuple_decl_RAW += read_constraints.set_var(j)->name();
    }
    tuple_decl_RAW += "]";
    std::string tuple_decl_WAR = "[";
    for (int j = 1; j <= read_constraints.n_set(); j++) {
      if (j > 1)
        tuple_decl_WAR += ",";
      tuple_decl_WAR += read_constraints.set_var(j)->name();
    }
    tuple_decl_WAR += "] -> [";
    for (int j = 1; j <= write_constraints.n_set(); j++) {
      if (j > 1)
        tuple_decl_WAR += ",";
      tuple_decl_WAR += write_constraints.set_var(j)->name();
    }

    tuple_decl_WAR += "]";

    std::string lex_order1 = write_constraints.set_var(1)->name()  + " < "
                     + read_constraints.set_var(1)->name();
    std::string lex_order2 = read_constraints.set_var(1)->name() + " < "
                     + write_constraints.set_var(1)->name();

//std::cout<<"\nLex ord1 = "<<lex_order1<<"    Lex ord2 = "<<lex_order2<<"\n";

    // Generating symbolic constants for the dependencs
    std::set<std::string> global_vars_write = get_global_vars(write_constraints);
    std::set<std::string> global_vars_read = get_global_vars(read_constraints);
    std::set<std::string> global_vars_equality = get_global_vars(equality_constraints);
    std::set<std::string> global_vars;
    for (std::set<std::string>::iterator it = global_vars_write.begin();
         it != global_vars_write.end(); it++)  global_vars.insert(*it);
    for (std::set<std::string>::iterator it = global_vars_read.begin();
         it != global_vars_read.end(); it++)  global_vars.insert(*it);
    for (std::set<std::string>::iterator it = global_vars_equality.begin();
         it != global_vars_equality.end(); it++)  global_vars.insert(*it);

    string symbolic_constants = "[";
    for (std::set<std::string>::iterator it = global_vars.begin(); 
         it != global_vars.end(); it++){
      if (it != global_vars.begin())
        symbolic_constants += ",";
      symbolic_constants += *it;
    }
    symbolic_constants += "]";
//std::cout<<"\nsymbolic_constants = "<<symbolic_constants<<"\n";


    std::string main_constraints = print_to_iegen_string(write_constraints);
    main_constraints += " && " + print_to_iegen_string(read_constraints);
    main_constraints += " && " + print_to_iegen_string(equality_constraints);
//std::cout<<"\n*******************Before replace  main_constraints #"<<relCounter<<" = "<<main_constraints<<"\n";
    // Here we replace the UFCs in the omega::relation that kind of have 
    // a symbolic constant form, with their actual equivalent in the code:
    // Note: omega can only handle certain type of UFCs that is why 
    // they are not stored in their original form at the first place.
    // For instance: omega::relation would have row(i+1) as row__(i).
    for (std::map<std::string, std::string>::iterator it =
         unin_symbol_for_iegen.begin(); it != unin_symbol_for_iegen.end(); it++) {
      std::size_t found = main_constraints.find(it->first);

      while (found != std::string::npos) {
        // Mahdi: If UFC is not at the beginning of the string we need to check for nested calls.
        if( found > 0 ){
           // Mahdi: This is necessary to skip nested calls, e.g A_(B_(i))
          if(main_constraints.substr(found - 1, 1) != "_"){
            main_constraints.replace(found, it->first.length(), it->second);
          }
        // Mahdi: If UFC is at the beginning of the string we do not need to check for nested calls.
        // Also, since checking for nested UFCalls involves looking up 1-character before found,
        // when found == 0, we would have a seg fault in our hand looking up the -1 position of a string.
        } else if( found == 0 ){ 
          main_constraints.replace(found, it->first.length(), it->second);
        }

        found = main_constraints.find(it->first, found + 1);
      }
    }
//std::cout<<"\n******************* main_constraints "<<relCounter<<" = "<<main_constraints<<"\n";

    std::string s1 = symbolic_constants + " -> " + "{" + tuple_decl_RAW + " : " + lex_order1 + " && " + main_constraints + "}";
    std::string s2 = symbolic_constants + " -> " + "{" + tuple_decl_WAR + " : " + lex_order2 + " && " + main_constraints + "}";

    rels2.push_back(std::pair<omega::Relation, omega::Relation>(write_constraints, read_constraints));
    std::cout << "\nS"<<relCounter++<<" = " << s1;
    std::cout << "\nS"<<relCounter++<<" = " << s2 <<std::endl;
    dep_rel_for_iegen.push_back(std::pair<std::string, std::string>(s1, s2));

  }  // Mahdi: End of the loop that creates dependencs for IEGenLib

  rels = removeRedundantConstraints(rels);

  return dep_rel_for_iegen;
}
