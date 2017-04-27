/*
* loop_sparse.cc
*  Loop Class for Sparse Matrix Format Transformations
*  Created on: October 7, 2013
*      Author: Anand Venkat
*/

#include "loop.hh"
#include <iegenlib.h>
#include <omega/code_gen/include/code_gen/CG_utils.h>
#include "omegatools.hh"
#include <sstream>

using namespace omega;

Relation flip_var_exclusive(Relation &r1, Variable_ID &v1) {
  Relation r;
  if (r1.is_set())
    r = Relation(r1.n_set());
  else
    r = Relation(r1.n_inp(), r1.n_out());

  r.copy_names(r1);
  r.setup_names();
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  bool found = false;
  for (DNF_Iterator di(const_cast<Relation &>(r1).query_DNF()); di; di++) {

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

Relation flip_var(Relation &r1, Variable_ID &v1) {
  Relation r;
  if (r1.is_set())
    r = Relation(r1.n_set());
  else
    r = Relation(r1.n_inp(), r1.n_out());

  r.copy_names(r1);
  r.setup_names();
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;

  for (DNF_Iterator di(const_cast<Relation &>(r1).query_DNF()); di; di++) {
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
  Relation R(ordered_vars.size());

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

                          std::string args = "(";

                          for (int j = 1; j <= pos; j++) {
                            if (j > 1)
                              args += ",";
                            args += R.set_var(j)->name();

                          }
                          args += ")";
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

bool checkIfEqual(Relation &r, Relation s) {

  std::vector<std::set<std::pair<int, int> > > output_vars;
  std::vector<std::set<std::pair<int, int> > > input_vars;
  std::vector<std::set<std::pair<int, std::string> > > global_vars;
  std::vector<int> constants;
  for (DNF_Iterator di(const_cast<Relation &>(r).query_DNF()); di; di++) {
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
  for (DNF_Iterator di(const_cast<Relation &>(s).query_DNF()); di; di++) {
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

std::map<int, Relation> removeRedundantConstraints(
    std::map<int, Relation> &rels) {

  std::map<int, Relation> to_return;
  for (std::map<int, Relation>::iterator i = rels.begin(); i != rels.end();
       i++) {
    bool found = false;
    if (i->second.is_finalized()) {

      for (std::map<int, Relation>::iterator j = rels.begin();
           j != rels.end(); j++) {
        if (j->first > i->first) {

          found |= checkIfEqual(i->second, j->second);
          found |= checkIfEqual(i->second, j->second);
        }
      }
      if (!found) {
        to_return.insert(std::pair<int, Relation>(i->first, i->second));

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

std::string get_lb_string(Relation &result, Variable_ID &v2,
                          bool range = true) {
  std::string a = "";
  Global_Var_ID uf;

  for (DNF_Iterator di(const_cast<Relation &>(result).query_DNF()); di; di++) {
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

std::string get_ub_string(Relation &result, Variable_ID &v2,
                          bool range = true) {
  std::string b = "";
  Global_Var_ID uf;

  for (DNF_Iterator di(const_cast<Relation &>(result).query_DNF()); di; di++) {
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
    Relation &result, IR_Code *ir, std::vector<Free_Var_Decl *> freevar) {

  std::vector<std::pair<bool, std::pair<std::string, std::string> > > to_return;
  std::vector<std::pair<bool, std::pair<std::string, std::string> > > to_return2;
  //outer most function call Eg. rowptr(colidx(k)) would be just rowptr
  //would be stored as rowptr_colidx__ so just parse till first '_' for base UF name
  std::set<std::string> outer_most_ufs2;

  for (DNF_Iterator di(const_cast<Relation &>(result).query_DNF()); di; di++) {
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
      for (DNF_Iterator di(const_cast<Relation &>(result).query_DNF()); di;
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

void Loop::printDependenceUFs(int stmt_num, int level) {
  Relation R = stmt[stmt_num].IS;

  Relation r = parseExpWithWhileToRel(stmt[stmt_num].code, R, stmt_num);
  r.simplify();

  Relation result;
  if (known.n_set() < r.n_set()) {
    omega::Relation known = omega::Extend_Set(omega::copy(this->known),
                                              r.n_set() - this->known.n_set());

    result = omega::Intersection(copy(known), copy(r));

  } else {

    Relation tmp = omega::Extend_Set(omega::copy(r),
                                     this->known.n_set() - r.n_set());

    result = omega::Intersection(tmp, copy(known));

  }
  result.simplify();
//r.print();

  bool is_negative = false;

  Variable_ID v2 = r.set_var(1);
  //need to correct following logic
  for (DNF_Iterator di(const_cast<Relation &>(r).query_DNF()); di; di++)
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++)
      if ((*gi).is_const(v2))
        if ((*gi).get_coef(v2) < 0 && (*gi).get_const() == 0)
          is_negative = true;

  if (is_negative) {
    r = flip_var(r, v2);	//convert to positive_IS
    Variable_ID v3 = result.set_var(1);
    result = flip_var(result, v3);
  }
  std::pair<std::vector<std::string>,
      std::pair<
          std::vector<std::pair<bool, std::pair<std::string, std::string> > >,
          std::vector<std::pair<bool, std::pair<std::string, std::string> > > > > syms =
      determineUFsForIegen(result, ir, freevar);

  std::vector<int> lex_ = getLexicalOrder(stmt_num);
  std::set<int> same_loop_ = getStatements(lex_, level - 1);
  std::vector<IR_ArrayRef *> access;
//get the equality, and_with_EQ,

  for (std::set<int>::iterator i = same_loop_.begin(); i != same_loop_.end();
       i++) {

    std::vector<IR_ArrayRef *> access2 = ir->FindArrayRef(stmt[*i].code);
    for (int j = 0; j < access2.size(); j++)
      access.push_back(access2[j]);
  }

  std::vector<std::pair<omega::Relation, omega::Relation> > dep_relation_;
  std::map<int, Relation> rels;
  std::vector<std::pair<Relation, Relation> > rels2;
  for (int i = 0; i < access.size(); i++) {
    IR_ArrayRef *a = access[i];

    IR_ArraySymbol *sym_a = a->symbol();

    for (int j = i + 1; j < access.size(); j++) {
      IR_ArrayRef *b = access[j];
      IR_ArraySymbol *sym_b = b->symbol();

      if (*sym_a == *sym_b && (a->is_write() || b->is_write())) {

        Relation r1(r.n_set(), r.n_set());

        for (int i = 1; i <= r.n_set(); i++)
          r1.name_input_var(i, r.set_var(i)->name());

        for (int i = 1; i <= r.n_set(); i++)
          r1.name_output_var(i, r.set_var(i)->name() + "p");

        IR_Symbol *sym_src;	// = a->symbol();
        IR_Symbol *sym_dst;	// = b->symbol();

        if (!a->is_write()) {
          sym_src = a->symbol();
          sym_dst = b->symbol();

        } else {
          sym_src = b->symbol();
          sym_dst = a->symbol();
        }

        if (*sym_src != *sym_dst) {
          r1.add_or(); // False Relation
          delete sym_src;
          delete sym_dst;
          //return r;
        } else {
          delete sym_src;
          delete sym_dst;
        }

        F_And *f_root = r1.add_and();

        for (int i = 0; i < a->n_dim(); i++) {
          F_Exists *f_exists = f_root->add_exists();
          Variable_ID e1 = f_exists->declare(tmp_e());
          Variable_ID e2 = f_exists->declare(tmp_e());
          F_And *f_and = f_exists->add_and();

          CG_outputRepr *repr_src; // = a->index(i);
          CG_outputRepr *repr_dst; // = b->index(i);
          if (!a->is_write()) {
            repr_src = a->index(i);
            repr_dst = b->index(i);

          } else {
            repr_src = b->index(i);
            repr_dst = a->index(i);
          }
          bool has_complex_formula = false;

          if (ir->QueryExpOperation(repr_src) == IR_OP_ARRAY_VARIABLE
              || ir->QueryExpOperation(repr_dst) == IR_OP_ARRAY_VARIABLE)
            ; //has_complex_formula = true;

          //if (!has_complex_formula) {

          try {
            exp2formula(this, ir, r1, f_and, freevar, repr_src, e1, 'w',
                        IR_COND_EQ, false, uninterpreted_symbols[stmt_num],
                        uninterpreted_symbols_stringrepr[stmt_num],
                        unin_rel[stmt_num]);
            exp2formula(this, ir, r1, f_and, freevar, repr_dst, e2, 'r',
                        IR_COND_EQ, false, uninterpreted_symbols[stmt_num],
                        uninterpreted_symbols_stringrepr[stmt_num],
                        unin_rel[stmt_num]);
          } catch (const ir_exp_error &e) {
            has_complex_formula = true;
          }

          if (!has_complex_formula) {
            EQ_Handle h = f_and->add_EQ();
            h.update_coef(e1, 1);
            h.update_coef(e2, -1);
          }
          //}
          repr_src->clear();
          repr_dst->clear();
          delete repr_src;
          delete repr_dst;
        }
        r1.simplify();
        if (is_negative) {
          Variable_ID v1, v2;

          v1 = r1.input_var(1);
          r1 = flip_var_exclusive(r1, v1);
          v2 = r1.output_var(1);
          r1 = flip_var_exclusive(r1, v2);

        }

        rels.insert(
            std::pair<int, Relation>(dep_relation_.size(), copy(r1)));
        dep_relation_.push_back(
            std::pair<omega::Relation, omega::Relation>(r, r1));

      }
    }
  }

// Call arrays2Relation

  for (int i = 0; i < dep_relation_.size(); i++) {
    Relation t(dep_relation_[i].first.n_set());
    for (int j = 1; j <= t.n_set(); j++)
      t.name_set_var(j, dep_relation_[i].first.set_var(j)->name() + "p");
    F_And *f_root = t.add_and();

    std::map<int, int> pos_map;
    for (int j = 1; j <= t.n_set(); j++) {
      pos_map.insert(std::pair<int, int>(i, i));

    }

    for (int j = 1; j <= t.n_set(); j++) {

      t = replace_set_var_as_another_set_var(t, dep_relation_[i].first, j, j,
                                             pos_map);

    }
    t.simplify();

    std::string initial = "{[";

    for (int j = 1; j <= dep_relation_[i].first.n_set(); j++) {
      if (j > 1)
        initial += ",";

      initial += dep_relation_[i].first.set_var(j)->name();
      initial += ",";
      initial += dep_relation_[i].first.set_var(j)->name() + "p";
    }

    initial += "]: ";

    std::string s1 = t.set_var(1)->name() + " <  "
                     + dep_relation_[i].first.set_var(1)->name();
    std::string s2 = t.set_var(1)->name() + " >  "
                     + dep_relation_[i].first.set_var(1)->name();

    Relation s_ = copy(dep_relation_[i].first);
    Relation t_ = copy(dep_relation_[i].second);
    /*for (DNF_Iterator di(const_cast<Relation &>(this->known).query_DNF());
     di; di++) {
     for (GEQ_Iterator e((*di)->GEQs()); e; e++) {
     s_.and_with_GEQ(*e);
     t.and_with_GEQ(*e);

     }
     }*/

    std::set<std::string> global_vars = get_global_vars(s_);
    std::set<std::string> global_vars2 = get_global_vars(t);

    std::string start = "[";

    for (std::set<std::string>::iterator j = global_vars.begin();
         j != global_vars.end(); j++) {
      if (j != global_vars.begin())
        start += ",";
      start += *j;

    }

    std::string start2 = "[";

    for (std::set<std::string>::iterator j = global_vars2.begin();
         j != global_vars2.end(); j++) {
      if (j != global_vars2.begin())
        start2 += ",";
      start2 += *j;

    }
    start += "] ->";
    start2 += "] ->";
    std::string s = " && " + print_to_iegen_string(s_);
    s += " && " + print_to_iegen_string(t);
    s += " && " + print_to_iegen_string(t_);
//std::cout<<s<<std::endl<<std::endl;
    for (std::map<std::string, std::string>::iterator a =
        unin_symbol_for_iegen.begin(); a != unin_symbol_for_iegen.end();
         a++) {
      std::size_t found = s.find(a->first);
      while (found != std::string::npos) {
        if (s.substr(found - 1, 1) != "_")
          s.replace(found, a->first.length(), a->second);

        found = s.find(a->first, found + 1);

      }
    }
    s1 = start + initial + s1 + s + "}";
    s2 = start2 + initial + s2 + s + "}";
    rels2.push_back(std::pair<Relation, Relation>(s_, t));
    std::cout << s1 << std::endl;
    std::cout << s2 << std::endl;
    std::cout << std::endl;
    dep_rel_for_iegen.push_back(std::pair<std::string, std::string>(s1, s2));
  }

  std::map<int, omega::Relation> for_codegen;
  std::map<int, omega::Relation> for_codegen2;
  int count = 0;
  rels = removeRedundantConstraints(rels);

  /*std::string try_ =
   "[ m, nnz ] -> { [i, ip, k] : ip - colidx(k) = 0 && colidx(k) >= 0 && rowptr(i) >= 0 && rowptr(ip) >= 0 && rowptr(colidx(k)) >= 0  && k - rowptr(i) >= 0 && nnz - diagptr(colidx(k) + 1) >= 0  && nnz - rowptr(colidx(k) + 1) >= 0 && diagptr(i + 1) - rowptr(i + 1) >= 0 && diagptr(ip + 1) - rowptr(ip + 1) >= 0 && diagptr(colidx(k)) - rowptr(colidx(k)) >= 0 && diagptr(colidx(k) + 1) + 1 >= 0 && rowptr(colidx(k) + 1) + 1 >= 0  && -i + m - 2 >= 0 && i - colidx(k) - 1 >= 0 && -k + diagptr(i) - 1 >= 0 && -k + rowptr(i + 1) - 2 >= 0 && nnz - diagptr(i) - 1 >= 0 && nnz - diagptr(i + 1) - 1 >= 0 && nnz - diagptr(ip) - 1 >= 0 && nnz - diagptr(ip + 1) - 1 >= 0 && -diagptr(colidx(k)) + rowptr(ip + 1) - 2 >= 0 && -diagptr(colidx(k)) + rowptr(colidx(k) + 1) + 2 >= 0 && diagptr(colidx(k) + 1) - rowptr(colidx(k) + 1) + 8 >= 0}";
   Relation a = parseISLStringToOmegaRelation(try_, rels2[0].first,
   rels2[0].second, unin_symbol_for_iegen, freevar, ir);
   std::cout << "First Result\n\n";
   copy(a).print();
   */

  //get IS intersect Known
  // check for domain and range info"
  // if rowptr_ < rowptr__  add monotonic info
  //
  for (int j = 0; j < dep_rel_for_iegen.size(); j++) {

    if (rels.find(j) != rels.end()) {

      iegenlib::setCurrEnv();
      for (int k = 0; k < syms.second.first.size(); k++) {
        std::string symbol = syms.second.first[k].second.first;
        std::string range_ = syms.second.first[k].second.second;
        std::string domain_ = syms.second.second[k].second.second;

        if (syms.second.first[k].first == true) {
          iegenlib::appendCurrEnv(symbol,
                                  new iegenlib::Set("{[i]:" + domain_ + "}"), // Domain
                                  new iegenlib::Set("{[j]:" + range_ + "}"),  // Range
                                  false,                              // Bijective?!
                                  iegenlib::Monotonic_Increasing       // monotonicity
          );

        } else {
          iegenlib::appendCurrEnv(symbol,
                                  new iegenlib::Set("{[i]:" + domain_ + "}"), // Domain
                                  new iegenlib::Set("{[j]:" + range_ + "}"),  // Range
                                  false,                              // Bijective?!
                                  iegenlib::Monotonic_NONE            // monotonicity
          );

        }

      }

      ;

      /*		iegenlib::appendCurrEnv("colidx",
       new iegenlib::Set("{[i]:0<=i &&i<nnz}"), // Domain
       new iegenlib::Set("{[j]:0<=j &&j<m}"),        // Range
       false,                              // Bijective?!
       iegenlib::Monotonic_NONE            // monotonicity
       );
       iegenlib::appendCurrEnv("rowptr",
       new iegenlib::Set("{[i]:0<=i &&i<=m}"),
       new iegenlib::Set("{[j]:0<=j &&j<nnz}"), false,
       iegenlib::Monotonic_Increasing);
       iegenlib::appendCurrEnv("diagptr",
       new iegenlib::Set("{[i]:0<=i &&i<=m}"),
       new iegenlib::Set("{[j]:0<=j &&j<nnz}"), false,
       iegenlib::Monotonic_Increasing);
       */
// (2)
// Putting constraints in an iegenlib::Set
// Data access dependence from ILU CSR code
      iegenlib::Set *A1 = new iegenlib::Set(dep_rel_for_iegen[j].first);

// (3)
// How to add user defined constraint
      //iegenlib::Set* A1_extend;
      std::cout << dep_rel_for_iegen[j].first << std::endl;
      for (int i = 0; i < syms.first.size(); i++) {
        //		cout << syms.first[i] << endl;
        std::string comparator_str;
        if (syms.first[i].find("<=") != std::string::npos) {
          comparator_str = "<=";
        } else if (syms.first[i].find("<") != std::string::npos)
          comparator_str = "<";
        else
          assert(false);

        char *copy_str = new char[syms.first[i].length() + 1];
        ;

        strcpy(copy_str, syms.first[i].c_str());
        //	char *first_part = part1.c_str();
        char *pch3 = strtok(copy_str, " ><=");
        std::string token1(pch3);

        pch3 = strtok(NULL, " ><=");
        std::string token2(pch3);
        iegenlib::Set *tmp = A1->addUFConstraints(token1, comparator_str,
                                                  token2);
        delete A1;
        A1 = tmp;
      }
// (4)
// Specify loops that are going to be parallelized, so we are not going to
// project them out. Here "i" and "ip"
      std::set<int> parallelTvs;
      parallelTvs.insert(0);
      parallelTvs.insert(1);

// expected output  (for testing purposes)
//std::string ex_A1_str("Not Satisfiable");
// (5)
// Simplifyng the constraints set

      iegenlib::Set* A1_sim = A1->simplifyForPartialParallel(parallelTvs);

// (6)
// Print out results
// If set is not satisfiable simplifyForPartialParallel is going to return
// NULL, we should check this before getting result's string with
// toISLString.
      std::string A1_sim_str("Not Satisfiable");

      omega::Relation r, r2;
      //r = Relation::False(1);
      //r2 = Relation::False(1);
      if (A1_sim) {
        A1_sim_str = A1_sim->toISLString();
        r = parseISLStringToOmegaRelation(A1_sim_str, rels2[j].first,
                                          rels2[j].second, unin_symbol_for_iegen, freevar, ir);
        std::cout << "Result\n\n";
        copy(r).print();

        for_codegen.insert(std::pair<int, Relation>(count++, r));
      }
      std::cout << "\n\nA1 simplified = " << A1_sim_str << "\n\n";

      iegenlib::Set *A2 = new iegenlib::Set(dep_rel_for_iegen[j].second);
      for (int i = 0; i < syms.first.size(); i++) {
        //cout << syms.first[i] << endl;
        std::string comparator_str;
        if (syms.first[i].find("<=") != std::string::npos) {
          comparator_str = "<=";
        } else if (syms.first[i].find("<") != std::string::npos)
          comparator_str = "<";
        else
          assert(false);

        char *copy_str = new char[syms.first[i].length() + 1];
        ;

        strcpy(copy_str, syms.first[i].c_str());
        //	char *first_part = part1.c_str();
        char *pch3 = strtok(copy_str, " ><=");
        std::string token1(pch3);

        pch3 = strtok(NULL, " ><=");
        std::string token2(pch3);

        iegenlib::Set *tmp = A2->addUFConstraints(token1, comparator_str,
                                                  token2);
        delete A2;
        A2 = tmp;
      }
// (3)
// How to add user defined constraint
      //	iegenlib::Set* A2_extend = A2->addUFConstraints("rowptr", "<=",
      //			"diagptr");

// (4)
// Specify loops that are going to be parallelized, so we are not going to
// project them out. Here "i" and "ip"
      std::set<int> parallelTvs2;
      parallelTvs2.insert(0);
      parallelTvs2.insert(1);

// expected output  (for testing purposes)
//std::string ex_A1_str("Not Satisfiable");
// (5)
// Simplifyng the constraints set

      iegenlib::Set* A2_sim = A2->simplifyForPartialParallel(parallelTvs2);

// (6)
// Print out results
// If set is not satisfiable simplifyForPartialParallel is going to return
// NULL, we should check this before getting result's string with
// toISLString.
      std::string A2_sim_str("Not Satisfiable");
      if (A2_sim) {
        A2_sim_str = A2_sim->toISLString();
        r2 = parseISLStringToOmegaRelation(A2_sim_str, rels2[j].first,
                                           rels2[j].second, unin_symbol_for_iegen, freevar, ir);
        std::cout << "Result\n\n";
        copy(r2).print();

        for_codegen2.insert(std::pair<int, Relation>(count, r2));
      }

      std::cout << "\n\nA2 simplified = " << A2_sim_str << "\n\n";
// For testing purposes
//EXPECT_EQ(ex_A1_str, A1_sim_str);
      delete A1;
      delete A2;
      //delete A1_extend;
      delete A1_sim;
      //delete A2_extend;
      delete A2_sim;
    }
  }
}