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

// Mahdi: A helper function to correct embedded iteration space: from Tuowen's topdown branch
// buildIS is basically suppose to replace init_loop in Tuowens branch, and init_loop commented out
// however since Tuowen may want to keep somethings from init_loop I am leaving it there for now
extern std::string index_name(int level);

typedef struct depRelationParts{
  omega::Relation write_st_is;
  int write_st_no;
  omega::Relation read_st_is;
  int read_st_no;
  omega::Relation equality_st_is;
}depRelParts;


void replace_tv_name(std::string &str, std::string old_name, std::string new_name){
  std::size_t found = str.find(old_name);
  while (found != std::string::npos) {
    if( (found + old_name.length()) >= str.size() ){
      str.replace(found, old_name.length(), new_name);
    } else if(str[(found + old_name.length())] != 'p') {
      str.replace(found, old_name.length(), new_name);
    }
    found = str.find(old_name, found + 1);
  }
} 


// Printing omega::Relation to string 
std::string omega_rel_to_string(omega::Relation rel){

  std::string result,tuple_decl = "[", constraints;

// Mahdi: Think more generally about if following change is correct
//  constraints = rel.print_formula_to_string();
  constraints = print_to_iegen_string(rel);

  if (rel.is_set()){
    for (int j = 1; j <= rel.n_set(); j++) {
      if (j > 1)
        tuple_decl += ",";
      tuple_decl += rel.set_var(j)->name();
    }
  } else {
    for (int j = 1; j <= rel.n_inp(); j++) {
      if (j > 1)
        tuple_decl += ",";
      tuple_decl += rel.input_var(j)->name();
    }
    tuple_decl += "] -> [";
    for (int j = 1; j <= rel.n_out(); j++) {
      if (j > 1)
        tuple_decl += ",";
      tuple_decl += rel.output_var(j)->name();
    }
  }
  tuple_decl += "]";
  
  result = "{" + tuple_decl + ": " + constraints + "}"; 

  return result;
}

// Helper function to remove extrac spaces after "," from a string
std::string remExtraSpace(std::string str){

  std::string result = str;

  for(int i=0 ; i < result.length()-1; i++){
    if(result[i] == ',') while(result[i+1] == ' ') result.erase (i+1,1); 
  }

  return result;
}

// Extracts names privatizable arrays from a string
std::set<std::string> privateArrays(std::string str){

  std::set<std::string> prArrays;
  str = iegenlib::trim(str);

  if(str.length() == 0 )   return prArrays;

  std::size_t start = 0, end = 0 ;
  while( start < str.length() ){
    end = str.find_first_of(',', start);
    if( end == std::string::npos ) 
      end = str.length();
    prArrays.insert( trim( str.substr(start,end-start) ) ); 
    start = end + 1;
  }

  return prArrays;
}

// Extracts statement number for reduction operations
std::set<int> reductionOps(std::string str){

  std::set<int> redOps;
  int tint;
  std::string tstr;
  str = iegenlib::trim(str);

  if(str.length() == 0 )   return redOps;

  std::size_t start = 0, end = 0 ;
  while( start < str.length() ){
    end = str.find_first_of(',', start);
    if( end == std::string::npos ) 
      end = str.length();
    str = trim( str.substr(start,end-start) );
    sscanf(str.c_str(), "%d", &tint); 
    redOps.insert(tint);
    start = end + 1;
  }

  return redOps;
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
Loop::depRelsForParallelization(std::string privatizable_arrays, 
                                std::string reduction_operations, int parallelLoopLevel){

  int stmt_num = 1, level = 1, whileLoop_stmt_num = 1, maxDim = stmt[0].IS.n_set();

  std::set<std::string> prArrays = privateArrays(privatizable_arrays);
  std::set<int> redOps = reductionOps(reduction_operations);

  // Mahdi: a temporary hack for getting dependence extraction changes integrated
  replaceCode_ind = 0; 

//std::cout<<"\n\nStart of depRelsForParallelization!\n\n";

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
      // Excluding private arrays 
      IR_ArrayRef *a = access2[j];
      IR_ArraySymbol *sym_a = a->symbol();
      std::string f_name = sym_a->name();
      //std::cout<<"\n access : "<<f_name <<"\n";
      if( prArrays.find(f_name) != prArrays.end() ) continue;

      access.push_back(access2[j]);
      access_st[ct++] = *i; 
    }
  }

int relCounter = 1;
  
  // Checking pairs of all accesses for comming up with data access equalities.
  std::vector<depRelParts> depRels_Parts;
  for (int i = 0; i < access.size(); i++) {
    int write_st_no, read_st_no;
    IR_ArrayRef *a = access[i];
    IR_ArraySymbol *sym_a = a->symbol();

    for (int j = i; j < access.size(); j++) {

      IR_ArrayRef *b = access[j];
      IR_ArraySymbol *sym_b = b->symbol();

      // Excluding reduction operations
      if( access_st[i] == access_st[j] && redOps.find(access_st[i]) != redOps.end() && *a == *b )
        continue;

      if (*sym_a == *sym_b && (a->is_write() || b->is_write())) {
        // Mahdi: write_r, read_r are useless remove them.
        omega::Relation write_r, read_r;
        omega::Relation a_rel = omega::Range(omega::Restrict_Domain(omega::copy(stmt[access_st[i]].xform), 
                                omega::copy(stmt[access_st[i]].IS)));
        omega::Relation b_rel = omega::Range(omega::Restrict_Domain(omega::copy(stmt[access_st[j]].xform), 
                                omega::copy(stmt[access_st[j]].IS)));

        if ( a->is_write() ) {
          write_r = stmt[access_st[i]].IS;//a_rel;
          read_r = stmt[access_st[j]].IS;//b_rel;//
          write_st_no = access_st[i]; read_st_no = access_st[j];
        } else {
          write_r = stmt[access_st[j]].IS;//b_rel;//
          read_r = stmt[access_st[i]].IS;//a_rel;//
          write_st_no = access_st[j]; read_st_no = access_st[i];
        }
        omega::Relation accessEqRel(maxDim, maxDim);
        for (int i = 1; i <= maxDim; i++)
          accessEqRel.name_input_var(i, index_name(i));//r1.name_input_var(i, write_r.set_var(i)->name());

        for (int i = 1; i <= maxDim; i++)
          accessEqRel.name_output_var(i, index_name(i) + "p");//r1.name_output_var(i, read_r.set_var(i)->name() + "p");

        F_And *f_root = accessEqRel.add_and(); 

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

          try {   
            exp2formula(this, ir, accessEqRel, f_and, freevar, repr_src, e1, 'w',
                        IR_COND_EQ, false, uninterpreted_symbols[write_st_no],
                        uninterpreted_symbols_stringrepr[write_st_no],
                        unin_rel[write_st_no], true);
//std::cout<<"\nAfter 1st exp2for accessEqRel = "<<accessEqRel<<"\n";
            exp2formula(this, ir, accessEqRel, f_and, freevar, repr_dst, e2, 'r',
                        IR_COND_EQ, false, uninterpreted_symbols[read_st_no],
                        uninterpreted_symbols_stringrepr[read_st_no],
                        unin_rel[read_st_no], true);
//std::cout<<"\nAfter 2nd exp2for accessEqRel = "<<accessEqRel<<"\n";
          } catch (const ir_exp_error &e) {
            has_complex_formula = true;
          }

          if (!has_complex_formula) {
            EQ_Handle h = f_and->add_EQ();
            h.update_coef(e1, -1);
            h.update_coef(e2, 1);
          }

          repr_src->clear();
          repr_dst->clear();
          delete repr_src;
          delete repr_dst;
        }   // Mahdi: end of n_dim loop

        accessEqRel.simplify();

        depRelParts tempDepRelParts;
        tempDepRelParts.write_st_is = write_r;
        tempDepRelParts.write_st_no = write_st_no;
        tempDepRelParts.read_st_is = read_r;
        tempDepRelParts.read_st_no = read_st_no;
        tempDepRelParts.equality_st_is = accessEqRel;
        depRels_Parts.push_back( tempDepRelParts );
      }
    }
  }

relCounter = 1;

  // Creating a map for tuple variable name adjusment because of applying schedule to Iteration Space
  std::vector<std::pair<std::string,std::string>> replace_tuple_var;
  for (int j = 1; j <= maxDim; j++){
    std::string is_tv = index_name(j);
    std::string is_tvp = is_tv+"p";
    replace_tuple_var.push_back( std::pair<std::string,std::string>(is_tv, ("In_"+to_string(j*2))) );
    replace_tuple_var.push_back( std::pair<std::string,std::string>(is_tvp, ("Out_"+to_string(j*2))) );
  }

  // Replacing tuple variable names in UFC map kept for turning omega::Relaiton to iegen::Relation
  std::map<std::string, std::string > omega2iegen_ufc_map;
  for (std::map<std::string, std::string>::iterator it =
       unin_symbol_for_iegen.begin(); it != unin_symbol_for_iegen.end(); it++) {
    std::string omega_name = it->first, iegen_name = it->second;
    for (int j=0; j < replace_tuple_var.size(); j++) {
      replace_tv_name(omega_name, replace_tuple_var[j].first, replace_tuple_var[j].second);
      replace_tv_name(iegen_name, replace_tuple_var[j].first, replace_tuple_var[j].second);
    }
    omega2iegen_ufc_map.insert(std::pair<std::string, std::string>(omega_name, iegen_name));
  }

//  std::ofstream outf;
//  outf.open (output_filename.c_str(), std::ofstream::out);
  
  // The loop that creates the relations for IEGen
  for (int i = 0; i < depRels_Parts.size(); i++) {

    omega::Relation write_sch = stmt[(depRels_Parts[i].write_st_no)].xform;
    omega::Relation read_sch = stmt[depRels_Parts[i].read_st_no].xform;
    omega::Relation write_orig_IS = stmt[(depRels_Parts[i].write_st_no)].IS;
    omega::Relation read_orig_IS = stmt[depRels_Parts[i].read_st_no].IS;
    omega::Relation equality_constraints = copy(depRels_Parts[i].equality_st_is);

    omega::Relation read_orig_IS_p(read_orig_IS.n_set()); 
    for (int j = 1; j <= read_orig_IS_p.n_set(); j++){
      read_orig_IS_p.name_set_var(j, read_orig_IS.set_var(j)->name() + "p");
    }
    read_orig_IS_p.copy_names(read_orig_IS_p);
    read_orig_IS_p.setup_names();
    F_And *f_root = read_orig_IS_p.add_and();
    read_orig_IS_p = replace_set_vars(read_orig_IS_p, read_orig_IS);
    read_orig_IS_p.simplify();

    // We are going to use IEGenLib to apply schedule to iteration space
    std::string omega_orig_write_is = omega_rel_to_string(write_orig_IS);
    std::string omega_orig_write_sch = omega_rel_to_string(write_sch);
    iegenlib::Set *iegen_write_is = new iegenlib::Set(omega_orig_write_is);
    iegenlib::Relation *iegen_write_sch = new iegenlib::Relation(omega_orig_write_sch);
    iegenlib::Set *iegen_write;
    iegen_write = iegen_write_sch->Apply(iegen_write_is);
    std::string iegen_write_str = iegen_write->getString();//prettyPrintString();//
    replace_tv_name(iegen_write_str,std::string("Out"),std::string("In"));

    std::string omega_orig_read_is = omega_rel_to_string(read_orig_IS_p);
    std::string omega_orig_read_sch = omega_rel_to_string(read_sch);
    iegenlib::Set *iegen_read_is = new iegenlib::Set(omega_orig_read_is);
    iegenlib::Relation *iegen_read_sch = new iegenlib::Relation(omega_orig_read_sch);
    iegenlib::Set *iegen_read;
    iegen_read = iegen_read_sch->Apply(iegen_read_is);
    std::string iegen_read_str = iegen_read->getString();//prettyPrintString();//
    // Mahdi FIXME: exception for smSmMul: make this general by getting it from json file
    replace_tv_name(iegen_read_str,std::string("nz"),std::string("nzp"));

    srParts iegen_write_parts, iegen_read_parts;
    iegen_write_parts = getPartsFromStr(iegen_write_str);
    iegen_read_parts = getPartsFromStr(iegen_read_str);

    // Generating tuple declaration for the dependencs
    std::string tuple_decl_RAW = iegen_write_parts.tupDecl + "->" + iegen_read_parts.tupDecl;
    std::string tuple_decl_WAR = iegen_read_parts.tupDecl + "->" + iegen_write_parts.tupDecl;

    // Generating lexographical ordering for the dependencs
    std::string lex_order1 = "In_2 < Out_2";
    std::string lex_order2 = "Out_2 < In_2";

    // Replacing tuple variable names in equality constraint that is 
    // built based on names in original iteration space, e.g 
    // chill_idx1 = colidx__(chill_idx1p,chill_idx2p) -> In_2 = colidx__(Out_2,Out_4)
    std::string equality_constraints_str = print_to_iegen_string(equality_constraints);

    for (int j=0; j < replace_tuple_var.size(); j++) {
      replace_tv_name(equality_constraints_str, replace_tuple_var[j].first, 
                      replace_tuple_var[j].second);
    }

    std::string main_constraints = remExtraSpace(iegen_write_parts.constraints);
    main_constraints += " && " + remExtraSpace(iegen_read_parts.constraints);
    if( iegenlib::trim(equality_constraints_str) != std::string("") )  // sc = sc would be empty string
      main_constraints += " && " + equality_constraints_str;

    // Here we replace the UFCs in the omega::relation that kind of have 
    // a symbolic constant form, with their actual equivalent in the code:
    // Note: omega can only handle certain type of UFCs that is why 
    // they are not stored in their original form at the first place.
    // For instance: omega::relation would have row(i+1) as row__(i).
    for (std::map<std::string, std::string>::iterator it =
         omega2iegen_ufc_map.begin(); it != omega2iegen_ufc_map.end(); it++) {
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

    std::string s1 = "{" + tuple_decl_RAW + " : " + lex_order1 + " && " + main_constraints + "}";
    std::string s2 = "{" + tuple_decl_WAR + " : " + lex_order2 + " && " + main_constraints + "}";

    //std::cout << "\nS"<<relCounter++<<" = " << s1;
    //std::cout << "\nS"<<relCounter++<<" = " << s2 <<std::endl;
    //outf<<s1<<std::endl<<s2<<std::endl;
    dep_rel_for_iegen.push_back(std::pair<std::string, std::string>(s1, s2));

  }  // Mahdi: End of the loop that creates dependencs for IEGenLib

  //outf.close();
  
//  rels = removeRedundantConstraints(rels);

  return dep_rel_for_iegen;
}

