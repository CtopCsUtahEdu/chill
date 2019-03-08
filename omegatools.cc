/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2010 University of Utah
 All Rights Reserved.

 Purpose:
   Useful tools involving Omega manipulation.

 Notes:

 History:
   01/2006 Created by Chun Chen.
   03/2009 Upgrade Omega's interaction with compiler to IR_Code, by Chun Chen.
*****************************************************************************/

#include <codegen.h>

#include "omegatools.hh"
#include "ir_code.hh"
#include "chill_error.hh"

#include "chill_ast.hh"
#include "code_gen/CG_stringRepr.h"
#include "code_gen/CG_chillRepr.h"
#include <code_gen/CG_utils.h>

using namespace omega;

namespace {
  struct DependenceLevel {
    Relation r;
    int level;
    int dir; // direction upto current level:
    // -1:negative, 0: undetermined, 1: postive
    std::vector<coef_t> lbounds;
    std::vector<coef_t> ubounds;

    DependenceLevel(const Relation &_r, int _dims) :
        r(_r), level(0), dir(0), lbounds(_dims), ubounds(_dims) {}
  };

  struct LinearTerm {
    int coefficient;
    std::vector<std::string> term;
    bool is_function;
    std::vector<LinearTerm> args;

    void dump() {
      if (term.size() == 0)
        std::cout << "NO STRINGS IN LINEAR TERM\n";
      for (int i = 0; i < term.size(); i++) {
        printf(" term is: %s\n", term[i].c_str());
      }
      std::cout << coefficient << "\n";
    }

    bool operator==(const LinearTerm &that) const {

      if (this->coefficient != that.coefficient)
        return false;
      if (this->is_function != that.is_function)
        return false;
      if ((this->term.size() == 0) && (that.term.size() == 0))
        return true;

      if ((this->term.size() > 0) && (that.term.size() > 0)) {

        if (this->term.size() != that.term.size())
          return false;

        for (int i = 0; i < this->term.size(); i++) {
          int j;
          bool matched = false;;
          for (j = 0; j < that.term.size(); j++) {

            if (this->term[i].size() == that.term[j].size()
                && this->term[i] == that.term[j]) {

              if (this->is_function) {
                for (int k = 0; k < this->args.size(); k++) {
                  int j;
                  for (; j < that.args.size(); j++)
                    if (this->args[k] == that.args[j])
                      break;
                  if (j == that.args.size())
                    return false;
                  else {
                    matched = true;
                    break;
                  }
                }
              } else {
                matched = true;
                break;
              }
            }
            if (matched)
              break;

          }
          if (j == that.term.size())
            return false;
        }

        return true;

      }

      return false;
    }

    bool operator!=(const LinearTerm &that) const {

      return !(*this == that);
    }

    bool operator<(const LinearTerm &that) const {

      if (this->term.size() < that.term.size())
        return true;
      else if (this->term.size() > that.term.size())
        return false;
      else if (this->coefficient < that.coefficient)
        return true;
      else if (this->coefficient > that.coefficient)
        return false;
      else if (this->term < that.term)
        return true;
      else
        return false;

    }

  };

  std::string dumpargs(std::vector<std::string> &v) {
// Mahdi: fixed a bug, empty argument list was not handlesd correctly.

    if (v.size() == 0) return "";

    std::string str = "(";
    for (int i = 0 ; i < v.size() ; i++ ) {
      if ( i ) str += ",";
      str += v[i];
    }
    str += ")";

    return str;
  }

  bool compareTerm(LinearTerm one, LinearTerm two) {

    if ((one.term.size() == 0) && (two.term.size() == 0))
      return true;

    if ((one.term.size() > 0) && (two.term.size() > 0)) {

      if (one.term.size() != two.term.size())
        return false;

      for (int i = 0; i < one.term.size(); i++) {
        int j;
        for (j = 0; j < two.term.size(); j++)

          if (one.term[i] == two.term[j])
            break;

        if (j == two.term.size())
          return false;
      }

      return true;

    }

    return false;
  }

  bool checkEquivalence(std::vector<LinearTerm> v1, std::vector<LinearTerm> v2) {

    if (v1.size() != v2.size())
      return false;

    for (int i = 0; i < v1.size(); i++) {
      int j;
      for (j = 0; j < v2.size(); j++) {
        if (v1[i] == v2[j] && (v1[i].coefficient == v2[j].coefficient))
          break;
      }
      if (j == v2.size())
        return false;

    }

    return true;

  }

  std::vector<LinearTerm> recursiveConstructLinearExpression(
      CG_outputRepr *repr_src, IR_Code *ir, char side) {
    std::vector<LinearTerm> v;
    switch (ir->QueryExpOperation(repr_src)) {

      case IR_OP_CONSTANT: {
        std::vector<CG_outputRepr *> v_ = ir->QueryExpOperand(repr_src);
        IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(v_[0]));
        LinearTerm to_push;
        to_push.is_function = false;
        to_push.coefficient = ref->integer();
        to_push.term = std::vector<std::string>();
        v.push_back(to_push);
        break;
      }
      case IR_OP_MACRO: {
        std::vector<CG_outputRepr *> v_ = ir->QueryExpOperand(repr_src);
        // Fixme: I'm a hack
        IR_FunctionRef *ref;
        {
          CG_chillRepr *crepr = (CG_chillRepr *) v_[0];
          chillAST_node *node = crepr->chillnodes[0];
          ref = new IR_chillFunctionRef(ir, static_cast<chillAST_DeclRefExpr *>(node));
        }
        LinearTerm to_push;
        to_push.is_function = true;

        std::string s = ref->name();

        CG_outputRepr *arguments = ir->RetrieveMacro(s);

        CG_outputRepr *inner = NULL;
        if (ir->QueryExpOperation(arguments) == IR_OP_ARRAY_VARIABLE) {
          std::vector<CG_outputRepr *> v = ir->QueryExpOperand(arguments);
          IR_Ref *ref_ = ir->Repr2Ref(v[0]);

          if (ref_->n_dim() > 1)
            throw ir_error(
                "Multi dimensional array in loop bounds: not supported currently!\n");

          if (dynamic_cast<IR_PointerArrayRef *>(ref_) != NULL) {

            inner = dynamic_cast<IR_PointerArrayRef *>(ref_)->index(0);

          } else if (dynamic_cast<IR_ArrayRef *>(ref_) != NULL) {

            inner = dynamic_cast<IR_ArrayRef *>(ref_)->index(0);

          }

        }
        if (inner == NULL)
          throw ir_error(
              "Unrecognized IR node in recursiveConstructLinearExpression\n");

        std::vector<std::string> to_push_2;
        to_push_2.push_back(s);
        to_push.coefficient = 1;
        to_push.term = to_push_2;
        to_push.args = recursiveConstructLinearExpression(inner->clone(), ir,
                                                          side);
        v.push_back(to_push);
        break;
      }

      case IR_OP_VARIABLE: {
        std::vector<CG_outputRepr *> v_ = ir->QueryExpOperand(repr_src);
        IR_ScalarRef *ref = static_cast<IR_ScalarRef *>(ir->Repr2Ref(v_[0]));
        LinearTerm to_push;
        to_push.is_function = false;

        std::string s = ref->name();
        if (side == 'r')
          s += "p";
        std::vector<std::string> to_push_2;
        to_push_2.push_back(s);
        to_push.coefficient = 1;
        to_push.term = to_push_2;
        v.push_back(to_push);
        break;
      }
      case IR_OP_ARRAY_VARIABLE: {
        std::vector<CG_outputRepr *> v_ = ir->QueryExpOperand(repr_src);
        IR_Ref *ref = ir->Repr2Ref(v_[0]);

        std::string s = ref->name();

        CG_outputRepr *repr2;

        assert(ref->n_dim() == 1);

        if (dynamic_cast<IR_PointerArrayRef *>(ref) != NULL) {

          repr2 = dynamic_cast<IR_PointerArrayRef *>(ref)->index(0);

        } else if (dynamic_cast<IR_ArrayRef *>(ref) != NULL) {

          repr2 = dynamic_cast<IR_ArrayRef *>(ref)->index(0);

        }

        LinearTerm to_push;
        to_push.is_function = true;
        to_push.args = recursiveConstructLinearExpression(repr2, ir, side);
        //std::string s = ref->name();
        std::vector<std::string> to_push_2;
        to_push_2.push_back(s);
        to_push.coefficient = 1;
        to_push.term = to_push_2;
        v.push_back(to_push);
        break;
      }
      case IR_OP_PLUS: {
        std::vector<CG_outputRepr *> v_ = ir->QueryExpOperand(repr_src);
        std::vector<LinearTerm> v1 = recursiveConstructLinearExpression(v_[0],
                                                                        ir, side);
        std::vector<LinearTerm> v2 = recursiveConstructLinearExpression(v_[1],
                                                                        ir, side);

        std::set<int> two_;
        for (int i = 0; i < v1.size(); i++) {
          int j;
          for (j = 0; j < v2.size(); j++)
            if (compareTerm(v1[i], v2[j])) {
              LinearTerm temp;
              temp.is_function = v1[i].is_function;
              temp.coefficient = v1[i].coefficient + v2[j].coefficient;
              temp.term = v1[i].term;
              temp.args = v1[i].args;
              temp.is_function = v1[i].is_function;
              v.push_back(temp);
              two_.insert(j);
              break;
            }
          if (j == v2.size())
            v.push_back(v1[i]);
        }

        for (int i = 0; i < v2.size(); i++)
          if (two_.find(i) == two_.end())
            v.push_back(v2[i]);

        break;
      }
      case IR_OP_MINUS: {
        std::vector<CG_outputRepr *> v_ = ir->QueryExpOperand(repr_src);
        std::vector<LinearTerm> v1 = recursiveConstructLinearExpression(v_[0],
                                                                        ir, side);
        std::vector<LinearTerm> v2 = recursiveConstructLinearExpression(v_[1],
                                                                        ir, side);

        std::set<int> two_;
        for (int i = 0; i < v1.size(); i++) {
          int j;
          for (j = 0; j < v2.size(); j++)
            if (compareTerm(v1[i], v2[j])) {
              LinearTerm temp;
              temp.is_function = v1[i].is_function;
              temp.coefficient = v1[i].coefficient - v2[j].coefficient;
              temp.term = v1[i].term;
              temp.args = v1[i].args;
              temp.is_function = v1[i].is_function;
              v.push_back(temp);
              two_.insert(j);
              break;
            }
          if (j == v2.size())
            v.push_back(v1[i]);
        }

        for (int i = 0; i < v2.size(); i++)
          if (two_.find(i) == two_.end())
            v.push_back(v2[i]);

        break;
      }
      case IR_OP_MULTIPLY: {
        std::vector<CG_outputRepr *> v_ = ir->QueryExpOperand(repr_src);
        std::vector<LinearTerm> v1 = recursiveConstructLinearExpression(v_[0],
                                                                        ir, side);
        std::vector<LinearTerm> v2 = recursiveConstructLinearExpression(v_[1],
                                                                        ir, side);

        std::set<int> two_;
        for (int i = 0; i < v1.size(); i++) {
          for (int j = 0; j < v2.size(); j++) {
            LinearTerm temp;

            temp.coefficient = v1[i].coefficient * v2[j].coefficient;
            if (v1[i].term.size() > 0 && v2[j].term.size() > 0) {
              temp.term = v1[i].term;
              if (v1[i].is_function)
                temp.args = v1[i].args;

              temp.is_function = v1[i].is_function | v2[j].is_function;
              temp.term.insert(temp.term.end(), v2[j].term.begin(),
                               v2[j].term.end());
              if (v2[j].is_function)
                temp.args.insert(temp.args.end(), v2[j].args.begin(),
                                 v2[j].args.end());
            } else if (v1[i].term.size() == 0 && v2[j].term.size() > 0) {
              temp.term = v2[j].term;
              temp.is_function = v2[j].is_function;
              if (v2[j].is_function)
                temp.args = v2[j].args;
            } else if (v1[i].term.size() > 0 && v2[j].term.size() == 0) {
              temp.term = v1[i].term;
              temp.is_function = v1[i].is_function;
              if (v1[i].is_function)
                temp.args = v1[i].args;
            } else {
              temp.term = std::vector<std::string>();
              temp.is_function = false;
            }
            int k;
            for (k = 0; k < v.size(); k++) {

              if (v[k] == temp) {
                v[k].coefficient = v[k].coefficient + temp.coefficient;
                break;
              }
            }
            if (k == v.size())
              v.push_back(temp);

          }
        }

        break;
      }

      default: {
        throw loop_error("unsupported operation type in array subscript");
        break;
      }
    }
    return v;

  }
}

std::string tmp_e() {
  static int counter = 1;
  return std::string("e")+to_string(counter++);
}


// Mahdi: This is needed for a change in exp2formula's concept that itself was related to
//        how iteration space is cacluated for imperfect loops.
//* When faced with nested function calls while calculated experssion for a dependence
//  exp2formual, now, passes output tuple of relation (r relation in the input list) to recursive 
//  call to exp2formual, instead of passing input tuple of relation that previously was getting passed.
std::string rmvExtraPrime(std::string str){
  if(str[(str.size()-1)] == 'p' || str[(str.size()-1)] == '\'')
    str.erase(str.end()-1, str.end()); // removing extra "p" or "'"
}

// Mahdi: A helper function to correct embedded iteration space: from Tuowen's topdown branch
// buildIS is basically suppose to replace init_loop in Tuowens branch, and init_loop commented out
// however since Tuowen may want to keep somethings from init_loop I am leaving it there for now
extern std::string index_name(int level);

//-----------------------------------------------------------------------------
// Convert expression tree to omega relation.  "destroy" means shallow
// deallocation of "repr", not freeing the actual code inside.
// -----------------------------------------------------------------------------
void exp2formula(Loop *loop, IR_Code *ir, Relation &r, F_And *f_root,
                 std::vector<Free_Var_Decl*> &freevars, CG_outputRepr *repr,
                 Variable_ID lhs, char side, IR_CONDITION_TYPE rel, bool destroy,
                 std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols,
                 std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols_stringrepr,
                 std::map<std::string, std::vector<omega::Relation> > &index_variables, bool extractingDepRel) {

  switch (ir->QueryExpOperation(repr)) {

    case IR_OP_MACRO: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
      IR_FunctionRef *ref = static_cast<IR_FunctionRef *>(ir->Repr2Ref(v[0]));
      std::string s = ref->name();
      Variable_ID e;
      bool exists = false;
      for (int i = 0; i < freevars.size(); i++)
        if (freevars[i]->base_name() == s) {
          e = r.get_local(freevars[i], Input_Tuple);
          exists = true;
        }
      if (!exists) {
        Free_Var_Decl *t = new Free_Var_Decl(s, 1);
        e = r.get_local(t, Input_Tuple);
      }

      EQ_Handle h = f_root->add_EQ();
      h.update_coef(lhs, 1);
      h.update_coef(e, -1);

      break;
    }
    case IR_OP_CONSTANT: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
      IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(v[0]));
      if (!ref->is_integer())
        throw ir_exp_error("non-integer constant coefficient");

      coef_t c = ref->integer();
      if (rel == IR_COND_GE || rel == IR_COND_GT) {
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(lhs, 1);
        if (rel == IR_COND_GE)
          h.update_const(-c);
        else
          h.update_const(-c - 1);
      } else if (rel == IR_COND_LE || rel == IR_COND_LT) {
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(lhs, -1);
        if (rel == IR_COND_LE)
          h.update_const(c);
        else
          h.update_const(c - 1);
      } else if (rel == IR_COND_EQ) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(lhs, 1);
        h.update_const(-c);
      } else
        throw std::invalid_argument("unsupported condition type");

      delete v[0];
      delete ref;
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_ARRAY_VARIABLE: {
// Mahdi: comment: Whenever you see something like if(side == 'r'); += "p"; += "\'"; 
// these rae related to building data access equality for dependence relations. 
// if(side == 'r') is checking to see if we have gotten an expression related to read access. 
// Note, iterators for read access, at the end, are going to have 
// an extra "p" in built for IEGenLib relations and an extra "\'" in CHILL specific relations. 
// So exp2formula needs to know that input relation for building
//  a dependence would be something like: Relation &r = {[chillidx_1]->[chillidx1p] :}
// But, both chillidx_1 and chillidx1p are (probably) representing the same iterator in the code (IR_Code *ir). 
 

      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);

      IR_Ref *ref = ir->Repr2Ref(v[0]);

      std::string s = ref->name();

      int max_dim = 0;
      bool need_new_fsymbol = false;

      // Mahdi: comment: vars stores the prefix tuple variable up to (and including) 
      // the input iterator parameter to an index array.
      std::set<std::string> vars;
      std::vector<Relation> reprs3;
      std::vector<EQ_Handle> reprs_3;

// Mahdi: comment: These are the names that are going to built for an uninetreted function call (UFC). 
// Then if the UFC does not already exists in the maps, its going to be added to them. 
// Different names are for differnt maps (code to omega, omega to iegenlib).
      std::set<std::string> unin_syms;
      CG_outputRepr *curr_repr = NULL;
      CG_outputRepr *curr_repr_s = NULL;
      CG_outputRepr *curr_repr_s2 = NULL;
      CG_outputRepr *curr_repr_no_const = NULL;
      CG_outputRepr *curr_repr_s_no_const = NULL;
      CG_outputRepr *curr_repr_s2_no_const = NULL;


      // Mahdi: comment: This loop traverses different dimentions of a multi dimentional array, e.g A[i1][i2][i3].
      for (int i = 0; i < ref->n_dim(); i++) {

/* Mahdi: Here exp2formula recursively traverses nested calls, e.g a[b[c[i]]].
To tell the recursive calls what is the tuple declaration for the relation that 
includes the overall term, here a temporary set is created out of tuple declaration of 
the input omega relation to first exp2formula call. It used to be the case that 
this set was created  ONLY with INPUT tuple declaration of the original input relation. 
And, it used to work fine, because we call exp2formula in 2 general case: 
(1) when chill is building the iteration space of the statements, in which case 
the input relation for first call to exp2formual is already just a set, e.g r = {[...] : ...}. 
So, when we build temporary set out of set, their tuple declaration is the same. 
(2) chill calls exp2formula when building some sort of dependence relation. 
Depenedence relations are actually relations, meaning their tuple declaration have 
input and output parts, e.g r = {[...]->[...] : ...}. Now, in this case, 
the tuple declaration of the temporary set would not be same as input relation to 
first call of exp2formula. The old behaivour of using only input part of tuple declaration 
used to work since chill had certain form of creating iteration space for 
imperfectly nested loops that was skewed toward this behaivour working. 
But chill previous way of creating iteration space was not correct for general cases. 
Now, that Tuowen has implemented a better way of generating iteration sapce for loop nests, 
following code cannot just use the input part of tuple declaration of input relation to 
create the temporary set. It needs to create the temporary set with either input or output part of 
the input relation depending on whether we are creating constraints related to 
write-access (input part), or constraints for read-access (output part).
*/
        Relation temp(r.n_inp());

        r.setup_names();
        if (r.is_set())

          for (int j = 1; j <= r.n_set(); j++) {

            temp.name_set_var(j, r.set_var(j)->name());

          }
        else{
          for (int j = 1; j <= r.n_inp(); j++) {
            if(side == 'w') temp.name_input_var(j, r.input_var(j)->name());
            else if(side == 'r') temp.name_input_var(j, r.output_var(j)->name());
          }
        }
        F_And *temp_root = temp.add_and();

        CG_outputRepr* repr2 = repr->clone();

        IR_Ref *ref_tmp = ref;

        if (dynamic_cast<IR_PointerArrayRef *>(ref) != NULL) {

          repr2 = dynamic_cast<IR_PointerArrayRef *>(ref)->index(i);

        } else if (dynamic_cast<IR_ArrayRef *>(ref) != NULL) {

          repr2 = dynamic_cast<IR_ArrayRef *>(ref)->index(i);

        }

        //std::vector<Free_Var_Decl*> freevars_;
        Free_Var_Decl *t = new Free_Var_Decl(s);
        Variable_ID e = temp.get_local(t);
        freevars.insert(freevars.end(), t);

        exp2formula(loop, ir, temp, temp_root, freevars, repr2, e, side,
                    IR_COND_EQ, false, uninterpreted_symbols,
                    uninterpreted_symbols_stringrepr, index_variables);

        EQ_Handle e1;

// Mahdi: comment: when there are nested index arrays (Val[A[B[i+1]]]). 
// I think exp2formula's output is kept inide omega::Relation temp in a way that is not straightforward.
// Since things can get complicated, when we reach inner most parameter for instance for Val[A[B[i+1]] example, 
// and say input relation to first call to exp2formula was:
// r = {[i] : }
// the output for inner most call would be something like:
// r = {[i] : B = i+1}
// And, then:
// // r = {[i] : A = B__(i)}
// And finally, "e" (Variable_ID lhs) should return A_(B__(i))) for the very first call to exp2formula.
// Following trying to fish out the output of recursive calls from the output relation

        //std::set<std::string> arg_set_vars;
        for (DNF_Iterator di(temp.query_DNF()); di; di++) {
          for (EQ_Iterator ei = (*di)->EQs(); ei; ei++) {

            e1 = *ei;
            for (Constr_Vars_Iter cvi(*ei); cvi; cvi++)
              if ((*cvi).var->kind() == Input_Var) {
                // Mahdi: change to consider changes of iteration space
                std::string tv_name = (*cvi).var->name();
                if(tv_name[(tv_name.size()-1)] == 'p' || tv_name[(tv_name.size()-1)] == '\'')
                  tv_name.erase(tv_name.end()-1, tv_name.end()); // removing extra "p" or "'"

                if ((*cvi).var->get_position() > max_dim)
                  max_dim = (*cvi).var->get_position();
                std::string name = tv_name;
                if (side == 'r')
                  name += "p";
                curr_repr = ir->builder()->CreatePlus(curr_repr,
                                                      ir->builder()->CreateTimes(
                                                          ir->builder()->CreateInt(
                                                              -(*cvi).coef),
                                                          ir->builder()->CreateIdent(name)));
                if (-(*cvi).coef != 1) {
                  curr_repr_s = ir->builder_s().CreatePlus(
                      curr_repr_s,
                      ir->builder_s().CreateTimes(
                          ir->builder_s().CreateInt(
                              -(*cvi).coef),
                          ir->builder_s().CreateIdent(
                              name)));
                  if (side == 'r') {
                    std::string name = tv_name;
                    curr_repr_s2 = ir->builder_s().CreatePlus(
                        curr_repr_s2,
                        ir->builder_s().CreateTimes(
                            ir->builder_s().CreateInt(
                                -(*cvi).coef),
                            ir->builder_s().CreateIdent(
                                name)));

                  } else {

                  name += "p";
                    curr_repr_s2 = ir->builder_s().CreatePlus(
                        curr_repr_s2,
                        ir->builder_s().CreateTimes(
                            ir->builder_s().CreateInt(
                                -(*cvi).coef),
                            ir->builder_s().CreateIdent(
                                name)));
                  }
                } else {
                  curr_repr_s = ir->builder_s().CreatePlus(
                      curr_repr_s,
                      ir->builder_s().CreateIdent(name));
                  if (side == 'r') {
                    std::string name = tv_name;
                    curr_repr_s2 = ir->builder_s().CreatePlus(
                        curr_repr_s2,
                        ir->builder_s().CreateIdent(name));
                  } else {

                    name += "p";
                    curr_repr_s2 = ir->builder_s().CreatePlus(
                        curr_repr_s2,
                        ir->builder_s().CreateIdent(name));
                  }

                }
                vars.insert(
                    r.input_var((*cvi).var->get_position())->name());
                int j;
                for (j = 0; j < reprs_3.size(); j++)
                  if (reprs_3[j] == e1)
                    break;
                if (j == reprs_3.size()) {
                  Relation c(temp.n_set());
                  c.copy_names(temp);
                  c.setup_names();
                  c.add_and();
                  reprs_3.push_back(e1);
                  c.and_with_EQ(e1);
                }

              } else if ((*cvi).var->kind() == Global_Var
                         && (*cvi).var->get_global_var()->base_name()
                            != s) {
                Global_Var_ID g = (*cvi).var->get_global_var();
                if (g->arity() > 0) {
                  std::vector<CG_outputRepr *> args;
                  std::vector<CG_outputRepr *> args2;
                  //ir->RetrieveMacro(g->base_name());
                  std::string arg_string = "(";
                  std::string arg_string2 = "(";
                  for (int l = 1; l <= g->arity(); l++) {
                    std::string tv_name = temp.set_var(l)->name();
                    if(tv_name[(tv_name.size()-1)] == 'p' || tv_name[(tv_name.size()-1)] == '\'')
                      tv_name.erase(tv_name.end()-1, tv_name.end()); // removing extra "p" or "'"
                    if (l > 1) {
                      arg_string += ",";
                      arg_string2 += ",";
                    }
                    args.push_back(
                        ir->builder()->CreateIdent(
                            tv_name));
                    args2.push_back(
                        ir->builder_s().CreateIdent(
                            tv_name));
                    arg_string += tv_name;
                    arg_string2 += tv_name;
                    if (side == 'r')
                      arg_string += "p";
                    else
                      arg_string2 += "p";

                  }
                  arg_string += ")";
                  arg_string2 += ")";
                  //std::string base_name = g->base_name();
                  std::string s = g->base_name();
                  arg_string = s + arg_string;
                  arg_string2 = s + arg_string2;
                  //base_name = base_name.substr(0, base_name.find_first_of("_") );
                  std::map<std::string, std::string>::iterator lookup2 =
                      loop->unin_symbol_for_iegen.find(
                          arg_string);
                  std::map<std::string, std::string>::iterator lookup3 =
                      loop->unin_symbol_for_iegen.find(
                          arg_string2);
                  curr_repr = ir->builder()->CreatePlus(curr_repr,
                                                        ir->builder()->CreateTimes(
                                                            ir->builder()->CreateInt(
                                                                -(*cvi).coef),
                                                            ir->builder()->CreateInvoke(
                                                                g->base_name(), args)));
                  if (-(*cvi).coef != 1) {
                    if (lookup2
                        != loop->unin_symbol_for_iegen.end()){
                      curr_repr_s =
                          ir->builder_s().CreatePlus(
                              curr_repr_s,
                              ir->builder_s().CreateTimes(
                                  ir->builder_s().CreateInt(
                                      -(*cvi).coef),
                                  new CG_stringRepr(
                                      lookup2->second)));}
                    if (lookup3
                        != loop->unin_symbol_for_iegen.end())
                      curr_repr_s2 =
                          ir->builder_s().CreatePlus(
                              curr_repr_s2,
                              ir->builder_s().CreateTimes(
                                  ir->builder_s().CreateInt(
                                      -(*cvi).coef),
                                  new CG_stringRepr(
                                      lookup3->second)));
                  } else {
                    if (lookup2
                        != loop->unin_symbol_for_iegen.end())
                      curr_repr_s =
                          ir->builder_s().CreatePlus(
                              curr_repr_s,
                              new CG_stringRepr(
                                  lookup2->second));
                    if (lookup3
                        != loop->unin_symbol_for_iegen.end())
                      curr_repr_s2 =
                          ir->builder_s().CreatePlus(
                              curr_repr_s2,
                              new CG_stringRepr(
                                  lookup3->second));
                  }
                  unin_syms.insert(g->base_name());
                  std::map<std::string, std::set<std::string> >::iterator lookup =
                      loop->unin_symbol_args.find(
                          g->base_name());

                  for (std::set<std::string>::iterator i =
                      lookup->second.begin();
                       i != lookup->second.end(); i++) {
                    for (int j = 1; j <= temp.n_inp(); j++)
                      if (temp.input_var(j)->name() == *i)
                        if (j > max_dim)
                          max_dim = j;

                    vars.insert(*i);
                  }

                } else {
                  curr_repr = ir->builder()->CreateTimes(
                      ir->builder()->CreateInt(-(*cvi).coef),
                      ir->builder()->CreateIdent(
                          g->base_name()));
                  curr_repr_s = ir->builder_s().CreatePlus(
                      curr_repr_s,
                      ir->builder_s().CreateTimes(
                          ir->builder_s().CreateInt(
                              -(*cvi).coef),
                          ir->builder_s().CreateIdent(
                              g->base_name())));
                  vars.insert(g->base_name());
                }
              }   // Mahdi: end of: for (Constr_Vars_Iter cvi(*ei); cvi; cvi++)
            if ((*ei).get_const() != 0) {
              curr_repr_no_const = curr_repr->clone();
              curr_repr_s_no_const = curr_repr_s->clone();
              curr_repr_s2_no_const = curr_repr_s2->clone();

              curr_repr = ir->builder()->CreatePlus(curr_repr,
                                                    ir->builder()->CreateInt(-(*ei).get_const()));
              curr_repr_s = ir->builder_s().CreatePlus(curr_repr_s,
                                                       ir->builder_s().CreateInt(-(*ei).get_const()));
              curr_repr_s2 = ir->builder_s().CreatePlus(curr_repr_s2,
                                                        ir->builder_s().CreateInt(-(*ei).get_const()));

              //need_new_fsymbol = true;
              //true;
              //s += "_";
            } else {

              curr_repr_no_const = ir->builder()->CreatePlus(
                  curr_repr->clone(),
                  ir->builder()->CreateInt(1));
              curr_repr_s_no_const = ir->builder_s().CreatePlus(
                  curr_repr_s->clone(),
                  ir->builder_s().CreateInt(1));
              curr_repr_s2_no_const = ir->builder_s().CreatePlus(
                  curr_repr_s2->clone(),
                  ir->builder_s().CreateInt(1));
            }

          }

        }

        //if (max_diref->n_dim())
        //	need_new_fsymbol = true;

      }

      //if(side == 'r')
      //	s+="'";

      // Mahdi: comment: The s+= "_" are related to building a UFC like A__(i) out of A[i+1]
      for (std::set<std::string>::iterator i = unin_syms.begin();
           i != unin_syms.end(); i++)
        s += "_" + *i;

      //if(need_new_fsymbol)
      //	s+="_";

      need_new_fsymbol = true;
      Variable_ID e = find_index(r, s, side);

      if (e == NULL) { // must be free variable
        Free_Var_Decl *t = NULL;
        Free_Var_Decl *ta = NULL;
        bool changed = true;
        do {
          changed = false;
          s += "_";
          t = NULL;

          for (unsigned i = 0; i < freevars.size(); i++) {
            std::string ss = freevars[i]->base_name();
            t = freevars[i];
            if (s == ss) {
              std::vector<CG_outputRepr *> curr;
              curr.push_back(curr_repr->clone());
              std::vector<LinearTerm> a =
                  recursiveConstructLinearExpression(curr_repr,
                                                     ir, side);
              CG_outputRepr *args = ir->RetrieveMacro(s);
              CG_outputRepr * inner = NULL;
              if (ir->QueryExpOperation(args)
                  == IR_OP_ARRAY_VARIABLE) {
                std::vector<CG_outputRepr *> v =
                    ir->QueryExpOperand(args);
                IR_Ref * ref_ = ir->Repr2Ref(v[0]);

                if (ref_->n_dim() > 1)
                  throw ir_error(
                      "Multi dimensional array in loop bounds: not supported currently!\n");

                if (dynamic_cast<IR_PointerArrayRef *>(ref_) != NULL) {

                  inner =
                      dynamic_cast<IR_PointerArrayRef *>(ref_)->index(
                          0);

                } else if (dynamic_cast<IR_ArrayRef *>(ref_) != NULL) {

                  inner =
                      dynamic_cast<IR_ArrayRef *>(ref_)->index(
                          0);

                }

              }

              if (inner == NULL)
                throw ir_error(
                    "Unrecognized exppression in exp2formula!\n");
              std::vector<LinearTerm> b =
                  recursiveConstructLinearExpression(
                      inner->clone(), ir, side);

              if (!checkEquivalence(a, b)) {
                need_new_fsymbol = true;
                changed = true;
                break;
              } else {
                need_new_fsymbol = false;
                //changed = true;
                break;

              }

            }
          }
        } while (changed);

        if (need_new_fsymbol) {
          t = new Free_Var_Decl(s, max_dim);
          //else
          //	t = new Free_Var_Decl(s, max_dim);
          freevars.insert(freevars.end(), t);

        }
        Free_Var_Decl *t1 = NULL;
        std::string s1 = s + "_";
        bool need_new_fsymbol2 = true;
        if (curr_repr_no_const != NULL) {

          bool changed = true;
          do {
            changed = false;
            s1 += "_";
            t1 = NULL;

            for (unsigned i = 0; i < freevars.size(); i++) {
              std::string ss = freevars[i]->base_name();
              t1 = freevars[i];
              if (s1 == ss) {

                std::vector<CG_outputRepr *> curr;
                curr.push_back(curr_repr->clone());
                std::vector<LinearTerm> a =
                    recursiveConstructLinearExpression(
                        curr_repr, ir, side);

                CG_outputRepr *args = ir->RetrieveMacro(s);
                CG_outputRepr * inner = NULL;
                if (ir->QueryExpOperation(args)
                    == IR_OP_ARRAY_VARIABLE) {
                  std::vector<CG_outputRepr *> v =
                      ir->QueryExpOperand(args);
                  IR_Ref * ref_ = ir->Repr2Ref(v[0]);

                  if (ref_->n_dim() > 1)
                    throw ir_error(
                        "Multi dimensional array in loop bounds: not supported currently!\n");

                  if (dynamic_cast<IR_PointerArrayRef *>(ref_)
                      != NULL) {

                    inner =
                        dynamic_cast<IR_PointerArrayRef *>(ref_)->index(
                            0);

                  } else if (dynamic_cast<IR_ArrayRef *>(ref_)
                             != NULL) {

                    inner =
                        dynamic_cast<IR_ArrayRef *>(ref_)->index(
                            0);

                  }

                }

                if (inner == NULL)
                  throw ir_error(
                      "Unrecognized exppression in exp2formula!\n");
                std::vector<LinearTerm> b =
                    recursiveConstructLinearExpression(
                        inner->clone(), ir, side);

                if (!checkEquivalence(a, b)) {
                  need_new_fsymbol2 = true;
                  changed = true;
                  break;
                } else {
                  need_new_fsymbol2 = false;
                  //changed = true;
                  break;

                }

              }
            }
          } while (changed);

          if (need_new_fsymbol2) {
            t1 = new Free_Var_Decl(s1, max_dim);
            //else
            //	t = new Free_Var_Decl(s, max_dim);
            freevars.insert(freevars.end(), t1);

          }
        }

        if (need_new_fsymbol) {
          loop->unin_symbol_args.insert(
              std::pair<std::string, std::set<std::string> >(s,
                                                             vars));

        }
        if (need_new_fsymbol2 && curr_repr_no_const != NULL) {

          loop->unin_symbol_args.insert(
              std::pair<std::string, std::set<std::string> >(s1,
                                                             vars));
        }

        // Mahdi: comment: vars stores the prefix tuple variable up to (and including) 
        // the input iterator parameter to an index array.
        for (int j = 1; j <= max_dim; j++) {

          if (vars.find(r.input_var(j)->name()) == vars.end())
            vars.insert(r.input_var(j)->name());
          Relation tmp(r.n_inp(), 1);
          F_And *f_root = tmp.add_and();
          EQ_Handle e = f_root->add_EQ();
          e.update_coef(tmp.input_var(j), -1);
          e.update_coef(tmp.output_var(1), 1);
          tmp.simplify();

          reprs3.push_back(tmp);

        }

        std::vector<std::string> args;
        std::vector<std::string> args2;
        std::vector<omega::CG_outputRepr *> reprs;
        std::vector<omega::CG_outputRepr *> reprs2;

// Mahdi: CODE changed: Following is going to generate the prefix tuple declaration for an UFC.
//        However, the old implementation was changing the order of tuple variables. That was because
//        the set of prefix tuple variables were getting stored and read from a c++ std::set, and
//        this data structure orders its elements based on its own metrics (alphabetical) not order of insertion.
//
//        for (std::set<std::string>::iterator it = vars.begin();
//             it != vars.end(); it++) {
//          if (side == 'r') {
//            args.push_back(*it + "p");
//            args2.push_back(*it);
//          } else {
//            args.push_back(*it);
//            args2.push_back(*it + "p");
//          }
//          reprs.push_back(ir->builder()->CreateIdent(*it));
//          reprs2.push_back(ir->builder_s().CreateIdent(*it));
//        }

        for (std::set<std::string>::iterator it = vars.begin();
             it != vars.end(); it++) {
          reprs.push_back(ir->builder()->CreateIdent(*it));
          reprs2.push_back(ir->builder_s().CreateIdent(*it));
        }

        // Mahdi: comment:  max_dim represents the number of prefix iterators up to 
        // parameter iterator of UFC (e.g A[i3], A_(i1,i2,i3), max_dim=3). 
        for (int j = 1; j <= max_dim; j++) {
          std::string tv_name;// = r.input_var(j)->name();

          // Mahdi: comment: I am first removing extra "p" or "'" if they already exists.
          // But I would add them later.
          if (r.is_set() ) {
            tv_name = r.set_var(j)->name();
            if(tv_name[(tv_name.size()-1)] == 'p' || tv_name[(tv_name.size()-1)] == '\'')
              tv_name.erase(tv_name.end()-1, tv_name.end()); // removing extra "p" or "'"
          } else if (side == 'r') {

            tv_name = r.output_var(j)->name();
            if(tv_name[(tv_name.size()-1)] == 'p' || tv_name[(tv_name.size()-1)] == '\'')
              tv_name.erase(tv_name.end()-1, tv_name.end()); // removing extra "p" or "'"
          } else {
            tv_name = r.input_var(j)->name();          // There is no extra "p" or "'"
          }
          //if (vars.find( tv_name ) != vars.end()){
            //  Mahdi: comment: we always build 2 version of an UFC, A_(i1,i2,i3) and A_(i1p,i2p,i3p)
            //  The A_(i1p,i2p,i3p) is for read part of a dependence relation. 
            //  Note, we might end up needing only one of them, or both.
            if (side == 'r') {
              args.push_back(tv_name + "p");
              args2.push_back(tv_name);
            } else {
              args.push_back(tv_name);
              args2.push_back(tv_name + "p");
            }
          //}
          std::string parameters = static_cast<CG_stringRepr*>(curr_repr_s)->GetString();
          if( parameters.find(tv_name) !=std::string::npos ) break;
        }

        // Mahdi: comment: Checking to see if UFC is already in the maps that keep track of mapping 
        // UFCs between code, omega, and IEGenLib representations.
        if (need_new_fsymbol) {
          std::vector<CG_outputRepr *> a;
          a.push_back(curr_repr_s);
          curr_repr_s = ir->builder_s().CreateInvoke(ref->name(), a);
          loop->unin_symbol_for_iegen.insert(
              std::pair<std::string, std::string>(s + dumpargs(args),
                                                  static_cast<CG_stringRepr*>(curr_repr_s)->GetString()));
          std::vector<CG_outputRepr *> b;
          b.push_back(curr_repr_s2);
          curr_repr_s2 = ir->builder_s().CreateInvoke(ref->name(), b);
          loop->unin_symbol_for_iegen.insert(
              std::pair<std::string, std::string>(s + dumpargs(args2),
                                                  static_cast<CG_stringRepr*>(curr_repr_s2)->GetString()));
        }
        if (need_new_fsymbol2 && curr_repr_no_const != NULL) {
          std::vector<CG_outputRepr *> a;
          a.push_back(curr_repr_s_no_const);
          curr_repr_s_no_const = ir->builder_s().CreateInvoke(ref->name(),
                                                              a);
          loop->unin_symbol_for_iegen.insert(
              std::pair<std::string, std::string>(s1 + dumpargs(args),
                                                  static_cast<CG_stringRepr*>(curr_repr_s_no_const)->GetString()));
          std::vector<CG_outputRepr *> b;
          b.push_back(curr_repr_s2_no_const);
          curr_repr_s2_no_const = ir->builder_s().CreateInvoke(
              ref->name(), b);
          loop->unin_symbol_for_iegen.insert(
              std::pair<std::string, std::string>(s1 + dumpargs(args2),
                                                  static_cast<CG_stringRepr*>(curr_repr_s2_no_const)->GetString()));
        }

        if (need_new_fsymbol) {
          ir->CreateDefineMacro(s, args,
                                ir->builder()->CreateArrayRefExpression(ref->convert(),
                                                                        curr_repr));
        }

        if (uninterpreted_symbols.find(s) == uninterpreted_symbols.end()){
          uninterpreted_symbols.insert(
              std::pair<std::string,
                  std::vector<omega::CG_outputRepr *> >(s,
                                                        reprs));
        }
        if (uninterpreted_symbols_stringrepr.find(s)
            == uninterpreted_symbols_stringrepr.end())
          uninterpreted_symbols_stringrepr.insert(
              std::pair<std::string,
                  std::vector<omega::CG_outputRepr *> >(s,
                                                        reprs2));

        if (index_variables.find(s) == index_variables.end())
          index_variables.insert(
              std::pair<std::string, std::vector<omega::Relation> >(s,
                                                                    reprs3));

        if (need_new_fsymbol2) {
          if (curr_repr_no_const != NULL)
            ir->CreateDefineMacro(s1, args,
                                  ir->builder()->CreateArrayRefExpression(ref->convert(),
                                                                          curr_repr_no_const));
        }
        if (curr_repr_no_const != NULL) {
          if (uninterpreted_symbols.find(s1)
              == uninterpreted_symbols.end())
            uninterpreted_symbols.insert(
                std::pair<std::string,
                    std::vector<omega::CG_outputRepr *> >(s1,
                                                          reprs));

          if (uninterpreted_symbols_stringrepr.find(s1)
              == uninterpreted_symbols_stringrepr.end())
            uninterpreted_symbols_stringrepr.insert(
                std::pair<std::string,
                    std::vector<omega::CG_outputRepr *> >(s1,
                                                          reprs2));

          if (index_variables.find(s1) == index_variables.end())
            index_variables.insert(
                std::pair<std::string, std::vector<omega::Relation> >(
                    s1, reprs3));
        }

        if (side == 'r')
          e = r.get_local(t, Output_Tuple);
        else
          e = r.get_local(t, Input_Tuple);
        if (rel == IR_COND_GE || rel == IR_COND_GT) {
          GEQ_Handle h = f_root->add_GEQ();
          h.update_coef(lhs, 1);
          h.update_coef(e, -1);
          if (rel == IR_COND_GT)
            h.update_const(-1);
        } else if (rel == IR_COND_LE || rel == IR_COND_LT) {
          GEQ_Handle h = f_root->add_GEQ();
          h.update_coef(lhs, -1);
          h.update_coef(e, 1);
          if (rel == IR_COND_LT)
            h.update_const(-1);
        } else if (rel == IR_COND_EQ) {
          EQ_Handle h = f_root->add_EQ();
          h.update_coef(lhs, 1);
          h.update_coef(e, -1);

        } else
          throw std::invalid_argument("unsupported condition type");
      }
      //	}

//  delete v[0];
      delete ref;
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_VARIABLE: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
      IR_ScalarRef *ref = static_cast<IR_ScalarRef *>(ir->Repr2Ref(v[0]));

      std::string s = ref->name();
      Variable_ID e = find_index(r, s, side);

      if (e == NULL) { // must be free variable
        if( extractingDepRel && side == 'r' ){
          s = s+"p";
        }
        Free_Var_Decl *t = NULL;
        for (unsigned i = 0; i < freevars.size(); i++) {
          std::string ss = freevars[i]->base_name();

          if (s == ss) {
            t = freevars[i];
            break;
          }
        }


        if (t == NULL) {
          t = new Free_Var_Decl(s);
          freevars.insert(freevars.end(), t);
        }

        e = r.get_local(t);

      }
      if (rel == IR_COND_GE || rel == IR_COND_GT) {
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(lhs, 1);
        h.update_coef(e, -1);
        if (rel == IR_COND_GT)
          h.update_const(-1);
      } else if (rel == IR_COND_LE || rel == IR_COND_LT) {
        GEQ_Handle h = f_root->add_GEQ();
        h.update_coef(lhs, -1);
        h.update_coef(e, 1);
        if (rel == IR_COND_LT)
          h.update_const(-1);
      } else if (rel == IR_COND_EQ) {
        EQ_Handle h = f_root->add_EQ();
        h.update_coef(lhs, -1);
        h.update_coef(e, 1);
      } else
        throw std::invalid_argument("unsupported condition type");

//  delete v[0];
      delete ref;
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_ASSIGNMENT: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
      exp2formula(loop, ir, r, f_root, freevars, v[0], lhs, side, rel, true,
                  uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_PLUS: {
      F_Exists *f_exists = f_root->add_exists();
      Variable_ID e1 = f_exists->declare(tmp_e());
      Variable_ID e2 = f_exists->declare(tmp_e());
      F_And *f_and = f_exists->add_and();

      if (rel == IR_COND_GE || rel == IR_COND_GT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, 1);
        h.update_coef(e1, -1);
        h.update_coef(e2, -1);
        if (rel == IR_COND_GT)
          h.update_const(-1);
      } else if (rel == IR_COND_LE || rel == IR_COND_LT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, -1);
        h.update_coef(e1, 1);
        h.update_coef(e2, 1);
        if (rel == IR_COND_LT)
          h.update_const(-1);
      } else if (rel == IR_COND_EQ) {
        EQ_Handle h = f_and->add_EQ();
        h.update_coef(lhs, 1);
        h.update_coef(e1, -1);
        h.update_coef(e2, -1);
      } else
        throw std::invalid_argument("unsupported condition type");

      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
      exp2formula(loop, ir, r, f_and, freevars, v[0], e1, side, IR_COND_EQ,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      exp2formula(loop, ir, r, f_and, freevars, v[1], e2, side, IR_COND_EQ,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_MINUS: {
      F_Exists *f_exists = f_root->add_exists();
      Variable_ID e1 = f_exists->declare(tmp_e());
      Variable_ID e2 = f_exists->declare(tmp_e());
      F_And *f_and = f_exists->add_and();

      if (rel == IR_COND_GE || rel == IR_COND_GT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, 1);
        h.update_coef(e1, -1);
        h.update_coef(e2, 1);
        if (rel == IR_COND_GT)
          h.update_const(-1);
      } else if (rel == IR_COND_LE || rel == IR_COND_LT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, -1);
        h.update_coef(e1, 1);
        h.update_coef(e2, -1);
        if (rel == IR_COND_LT)
          h.update_const(-1);
      } else if (rel == IR_COND_EQ) {
        EQ_Handle h = f_and->add_EQ();
        h.update_coef(lhs, 1);
        h.update_coef(e1, -1);
        h.update_coef(e2, 1);
      } else
        throw std::invalid_argument("unsupported condition type");

      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);
      exp2formula(loop, ir, r, f_and, freevars, v[0], e1, side, IR_COND_EQ,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      exp2formula(loop, ir, r, f_and, freevars, v[1], e2, side, IR_COND_EQ,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_MULTIPLY: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);

      coef_t coef;
      CG_outputRepr *term;
      if (ir->QueryExpOperation(v[0]) == IR_OP_CONSTANT) {
        IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(
            v[0]));
        coef = ref->integer();
        delete v[0];
        delete ref;
        term = v[1];
      } else if (ir->QueryExpOperation(v[1]) == IR_OP_CONSTANT) {
        IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(
            v[1]));
        coef = ref->integer();
        delete v[1];
        delete ref;
        term = v[0];
      } else
        throw ir_exp_error("not presburger expression");

      F_Exists *f_exists = f_root->add_exists();
      Variable_ID e = f_exists->declare(tmp_e());
      F_And *f_and = f_exists->add_and();

      if (rel == IR_COND_GE || rel == IR_COND_GT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, 1);
        h.update_coef(e, -coef);
        if (rel == IR_COND_GT)
          h.update_const(-1);
      } else if (rel == IR_COND_LE || rel == IR_COND_LT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, -1);
        h.update_coef(e, coef);
        if (rel == IR_COND_LT)
          h.update_const(-1);
      } else if (rel == IR_COND_EQ) {
        EQ_Handle h = f_and->add_EQ();
        h.update_coef(lhs, 1);
        h.update_coef(e, -coef);
      } else
        throw std::invalid_argument("unsupported condition type");

      exp2formula(loop, ir, r, f_and, freevars, term, e, side, IR_COND_EQ,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_DIVIDE: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);

      assert(ir->QueryExpOperation(v[1]) == IR_OP_CONSTANT);
      IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(v[1]));
      coef_t coef = ref->integer();
      delete v[1];
      delete ref;

      F_Exists *f_exists = f_root->add_exists();
      Variable_ID e = f_exists->declare(tmp_e());
      F_And *f_and = f_exists->add_and();

      if (rel == IR_COND_GE || rel == IR_COND_GT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, coef);
        h.update_coef(e, -1);
        if (rel == IR_COND_GT)
          h.update_const(-1);
      } else if (rel == IR_COND_LE || rel == IR_COND_LT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, -coef);
        h.update_coef(e, 1);
        if (rel == IR_COND_LT)
          h.update_const(-1);
      } else if (rel == IR_COND_EQ) {
        EQ_Handle h = f_and->add_EQ();
        h.update_coef(lhs, coef);
        h.update_coef(e, -1);
      } else
        throw std::invalid_argument("unsupported condition type");

      exp2formula(loop, ir, r, f_and, freevars, v[0], e, side, IR_COND_EQ,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_MOD:
    {
      debug_fprintf(stderr, "IR_OP_MOD\n");
      /* the left hand of a mod can be a var but the right must be a const */
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);

      assert(ir->QueryExpOperation(v[1]) == IR_OP_CONSTANT);
      IR_ConstantRef *ref = static_cast<IR_ConstantRef *>(ir->Repr2Ref(v[1]));
      coef_t coef = ref->integer();
      delete v[1];
      delete ref;

      F_Exists *f_exists = f_root->add_exists();
      Variable_ID e = f_exists->declare(tmp_e());
      Variable_ID b = f_exists->declare(tmp_e());
      F_And *f_and = f_exists->add_and();


      if (rel == IR_COND_EQ)
        {
          EQ_Handle h = f_and->add_EQ();
          h.update_coef(lhs, 1);
          h.update_coef(b, coef);
          h.update_coef(e, -1);
        }

      else  if (rel == IR_COND_GE || rel == IR_COND_GT) {
        //i = CONST alpha + beta && beta >= const ( handled higher up ) && beta < CONST
        EQ_Handle h = f_and->add_EQ();
        h.update_coef(lhs, 1);
        h.update_coef(b, coef);
        h.update_coef(e, -1);

        GEQ_Handle k = f_and->add_GEQ();
        k.update_coef(lhs, -1 );
        k.update_const(coef-1);

      }

      else if (rel == IR_COND_LE || rel == IR_COND_LT) {
        //i = CONST alpha + beta && beta <= const ( handled higher up ) && beta >= 0
        EQ_Handle h = f_and->add_EQ();
        h.update_coef(lhs, 1);
        h.update_coef(b, coef);
        h.update_coef(e, -1);

        GEQ_Handle k = f_and->add_GEQ();
        k.update_coef(lhs, 1 );

      }

      exp2formula(loop, ir, r, f_and, freevars, v[0], e, side, IR_COND_EQ, true,
                  uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      if (destroy)
        delete repr;

      break;
    }
    case IR_OP_POSITIVE: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);

      exp2formula(loop, ir, r, f_root, freevars, v[0], lhs, side, rel, true,
                  uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_NEGATIVE: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);

      F_Exists *f_exists = f_root->add_exists();
      Variable_ID e = f_exists->declare(tmp_e());
      F_And *f_and = f_exists->add_and();

      if (rel == IR_COND_GE || rel == IR_COND_GT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, 1);
        h.update_coef(e, 1);
        if (rel == IR_COND_GT)
          h.update_const(-1);
      } else if (rel == IR_COND_LE || rel == IR_COND_LT) {
        GEQ_Handle h = f_and->add_GEQ();
        h.update_coef(lhs, -1);
        h.update_coef(e, -1);
        if (rel == IR_COND_LT)
          h.update_const(-1);
      } else if (rel == IR_COND_EQ) {
        EQ_Handle h = f_and->add_EQ();
        h.update_coef(lhs, 1);
        h.update_coef(e, 1);
      } else
        throw std::invalid_argument("unsupported condition type");

      exp2formula(loop, ir, r, f_and, freevars, v[0], e, side, IR_COND_EQ,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      if (destroy)
        delete repr;
      break;
    }
    case IR_OP_MIN: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);

      F_Exists *f_exists = f_root->add_exists();

      if (rel == IR_COND_GE || rel == IR_COND_GT) {
        F_Or *f_or = f_exists->add_and()->add_or();
        for (int i = 0; i < v.size(); i++) {
          Variable_ID e = f_exists->declare(tmp_e());
          F_And *f_and = f_or->add_and();
          GEQ_Handle h = f_and->add_GEQ();
          h.update_coef(lhs, 1);
          h.update_coef(e, -1);
          if (rel == IR_COND_GT)
            h.update_const(-1);

          exp2formula(loop, ir, r, f_and, freevars, v[i], e, side,
                      IR_COND_EQ, true, uninterpreted_symbols,
                      uninterpreted_symbols_stringrepr, index_variables);
        }
      } else if (rel == IR_COND_LE || rel == IR_COND_LT) {
        F_And *f_and = f_exists->add_and();
        for (int i = 0; i < v.size(); i++) {
          Variable_ID e = f_exists->declare(tmp_e());
          GEQ_Handle h = f_and->add_GEQ();
          h.update_coef(lhs, -1);
          h.update_coef(e, 1);
          if (rel == IR_COND_LT)
            h.update_const(-1);

          exp2formula(loop, ir, r, f_and, freevars, v[i], e, side,
                      IR_COND_EQ, true, uninterpreted_symbols,
                      uninterpreted_symbols_stringrepr, index_variables);
        }
      } else if (rel == IR_COND_EQ) {
        F_Or *f_or = f_exists->add_and()->add_or();
        for (int i = 0; i < v.size(); i++) {
          Variable_ID e = f_exists->declare(tmp_e());
          F_And *f_and = f_or->add_and();

          EQ_Handle h = f_and->add_EQ();
          h.update_coef(lhs, 1);
          h.update_coef(e, -1);

          exp2formula(loop, ir, r, f_and, freevars, v[i], e, side,
                      IR_COND_EQ, false, uninterpreted_symbols,
                      uninterpreted_symbols_stringrepr, index_variables);

          for (int j = 0; j < v.size(); j++)
            if (j != i) {
              Variable_ID e2 = f_exists->declare(tmp_e());
              GEQ_Handle h2 = f_and->add_GEQ();
              h2.update_coef(e, -1);
              h2.update_coef(e2, 1);

              exp2formula(loop, ir, r, f_and, freevars, v[j], e2,
                          side, IR_COND_EQ, false, uninterpreted_symbols,
                          uninterpreted_symbols_stringrepr,
                          index_variables);
            }
        }

        for (int i = 0; i < v.size(); i++)
          delete v[i];
      } else
        throw std::invalid_argument("unsupported condition type");

      if (destroy)
        delete repr;
    }
    case IR_OP_MAX: {
      std::vector<CG_outputRepr *> v = ir->QueryExpOperand(repr);

      F_Exists *f_exists = f_root->add_exists();

      if (rel == IR_COND_LE || rel == IR_COND_LT) {
        F_Or *f_or = f_exists->add_and()->add_or();
        for (int i = 0; i < v.size(); i++) {
          Variable_ID e = f_exists->declare(tmp_e());
          F_And *f_and = f_or->add_and();
          GEQ_Handle h = f_and->add_GEQ();
          h.update_coef(lhs, -1);
          h.update_coef(e, 1);
          if (rel == IR_COND_LT)
            h.update_const(-1);

          exp2formula(loop, ir, r, f_and, freevars, v[i], e, side,
                      IR_COND_EQ, true, uninterpreted_symbols,
                      uninterpreted_symbols_stringrepr, index_variables);
        }
      } else if (rel == IR_COND_GE || rel == IR_COND_GT) {
        F_And *f_and = f_exists->add_and();
        for (int i = 0; i < v.size(); i++) {
          Variable_ID e = f_exists->declare(tmp_e());
          GEQ_Handle h = f_and->add_GEQ();
          h.update_coef(lhs, 1);
          h.update_coef(e, -1);
          if (rel == IR_COND_GT)
            h.update_const(-1);

          exp2formula(loop, ir, r, f_and, freevars, v[i], e, side,
                      IR_COND_EQ, true, uninterpreted_symbols,
                      uninterpreted_symbols_stringrepr, index_variables);
        }
      } else if (rel == IR_COND_EQ) {
        F_Or *f_or = f_exists->add_and()->add_or();
        for (int i = 0; i < v.size(); i++) {
          Variable_ID e = f_exists->declare(tmp_e());
          F_And *f_and = f_or->add_and();

          EQ_Handle h = f_and->add_EQ();
          h.update_coef(lhs, 1);
          h.update_coef(e, -1);

          exp2formula(loop, ir, r, f_and, freevars, v[i], e, side,
                      IR_COND_EQ, false, uninterpreted_symbols,
                      uninterpreted_symbols_stringrepr, index_variables);

          for (int j = 0; j < v.size(); j++)
            if (j != i) {
              Variable_ID e2 = f_exists->declare(tmp_e());
              GEQ_Handle h2 = f_and->add_GEQ();
              h2.update_coef(e, 1);
              h2.update_coef(e2, -1);

              exp2formula(loop, ir, r, f_and, freevars, v[j], e2,
                          side, IR_COND_EQ, false, uninterpreted_symbols,
                          uninterpreted_symbols_stringrepr,
                          index_variables);
            }
        }

        for (int i = 0; i < v.size(); i++)
          delete v[i];
      } else
        throw std::invalid_argument("unsupported condition type");

      if (destroy)
        delete repr;
    }
    case IR_OP_NULL:
      break;
    default:
      throw ir_exp_error("unsupported operand type");
  }
}


//-----------------------------------------------------------------------------
// Build dependence relation for two array references.
// -----------------------------------------------------------------------------
Relation arrays2relation(Loop *loop, IR_Code *ir,
                         std::vector<Free_Var_Decl*> &freevars, const IR_ArrayRef *ref_src,
                         const Relation &IS_w, const IR_ArrayRef *ref_dst, const Relation &IS_r,
                         std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols,
                         std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols_stringrepr,
                         std::map<std::string, std::vector<omega::Relation> > &unin_rel) {

  Relation &IS1 = const_cast<Relation &>(IS_w);
  Relation &IS2 = const_cast<Relation &>(IS_r);

  Relation r(IS1.n_set(), IS2.n_set());

  for (int i = 1; i <= IS1.n_set(); i++)
    r.name_input_var(i, IS1.set_var(i)->name());

  for (int i = 1; i <= IS2.n_set(); i++)
    r.name_output_var(i, IS2.set_var(i)->name() + "'");

  IR_Symbol *sym_src = ref_src->symbol();
  IR_Symbol *sym_dst = ref_dst->symbol();
  if (*sym_src != *sym_dst) {
    r.add_or(); // False Relation
    delete sym_src;
    delete sym_dst;
    return r;
  } else {
    delete sym_src;
    delete sym_dst;
  }

  F_And *f_root = r.add_and();

  for (int i = 0; i < ref_src->n_dim(); i++) {
    F_Exists *f_exists = f_root->add_exists();
    Variable_ID e1 = f_exists->declare(tmp_e());
    Variable_ID e2 = f_exists->declare(tmp_e());
    F_And *f_and = f_exists->add_and();

    CG_outputRepr *repr_src = ref_src->index(i);
    CG_outputRepr *repr_dst = ref_dst->index(i);

    bool has_complex_formula = false;

    try {
      exp2formula(loop, ir, r, f_and, freevars, repr_src, e1, 'w',
                  IR_COND_EQ, false, uninterpreted_symbols,
                  uninterpreted_symbols_stringrepr, unin_rel);
      exp2formula(loop, ir, r, f_and, freevars, repr_dst, e2, 'r',
                  IR_COND_EQ, false, uninterpreted_symbols,
                  uninterpreted_symbols_stringrepr, unin_rel);
    } catch (const ir_exp_error &e) {
      has_complex_formula = true;
    }

    if (!has_complex_formula) {
      EQ_Handle h = f_and->add_EQ();
      h.update_coef(e1, 1);
      h.update_coef(e2, -1);
    }
    repr_src->clear();
    repr_dst->clear();
    delete repr_src;
    delete repr_dst;
  }

// add iteration space restriction
  r = Restrict_Domain(r, copy(IS1));
  r = Restrict_Range(r, copy(IS2));

// reset the output variable names lost in restriction
  for (int i = 1; i <= IS2.n_set(); i++)
    r.name_output_var(i, IS2.set_var(i)->name() + "'");
  r.setup_names();
  return r;
}

//-----------------------------------------------------------------------------
// Convert array dependence relation into set of dependence vectors, assuming
// ref_w is lexicographically before ref_r in the source code.
// -----------------------------------------------------------------------------
std::pair<std::vector<DependenceVector>, std::vector<DependenceVector> > relation2dependences (
                                                                                               const IR_ArrayRef *ref_src, 
                                                                                               const IR_ArrayRef *ref_dst, 
                                                                                               const Relation &r) {
  //debug_fprintf(stderr, "relation2dependences()\n"); 
  assert(r.n_inp() == r.n_out());
  
  std::vector<DependenceVector> dependences1, dependences2;  
  //std::vector<DependenceVector*> dep1, dep2;  
  std::stack<DependenceLevel> working;
  working.push(DependenceLevel(r, r.n_inp()));
  
  while (!working.empty()) {
    //debug_fprintf(stderr, "!empty  size %d\n", working.size()); 
    
    DependenceLevel dep = working.top();
    working.pop();
    
    //if (!dep.r.is_satisfiable()) debug_fprintf(stderr, "NOT dep.r.is_satisfiable()\n"); 
    //else                         debug_fprintf(stderr, "    dep.r.is_satisfiable()\n"); 
    
    // No dependence exists, move on.
    if (!dep.r.is_satisfiable()) { 
      //debug_fprintf(stderr, "No dependence exists, move on.\n"); 
      continue;
    }
    
    //debug_fprintf(stderr, "satisfiable\n"); 
    //debug_fprintf(stderr, "dep.level %d   r.n_inp() %d\n", dep.level, r.n_inp()); 
    if (dep.level == r.n_inp()) {
      //debug_fprintf(stderr, "dep.level == r.n_inp()\n"); 
      DependenceVector dv;
      
      //debug_fprintf(stderr, "\ndv created in if                                         ***\n"); 
      //DependenceVector *dv2 = new DependenceVector; 
      
      //debug_fprintf(stderr, "for loop independent dependence  dep.dir %d\n", dep.dir); 
      // for loop independent dependence, use lexical order to
      // determine the correct source and destination
      if (dep.dir == 0) {
        //debug_fprintf(stderr, "dep.dir == 0\n"); 
        if (*ref_src == *ref_dst) {                             // c == c
          //debug_fprintf(stderr, "trivial\n"); 
          continue; // trivial self zero-dependence
        }
        
        if (ref_src->is_write()) {
          if (ref_dst->is_write())
            dv.type = DEP_W2W;
          else
            dv.type = DEP_W2R;
        }
        else {
          if (ref_dst->is_write())
            dv.type = DEP_R2W;
          else
            dv.type = DEP_R2R;
        }
        
      }
      else if (dep.dir == 1) {
        //debug_fprintf(stderr, "dep.dir == 1\n"); 
        if (ref_src->is_write()) {
          if (ref_dst->is_write())
            dv.type = DEP_W2W;
          else
            dv.type = DEP_W2R;
        }
        else {
          if (ref_dst->is_write())
            dv.type = DEP_R2W;
          else
            dv.type = DEP_R2R;
        }
      }
      else { // dep.dir == -1
        //debug_fprintf(stderr, "dep.dir == -1\n"); 
        if (ref_dst->is_write()) {
          if (ref_src->is_write())
            dv.type = DEP_W2W;
          else
            dv.type = DEP_W2R;
        }
        else {
          if (ref_src->is_write())
            dv.type = DEP_R2W;
          else
            dv.type = DEP_R2R;
        }
      }
      
      dv.lbounds = dep.lbounds;
      dv.ubounds = dep.ubounds;
      
      //debug_fprintf(stderr, "omegatools.cc calling ref_src->symbol();\n"); 
      dv.sym = ref_src->symbol();
      //debug_fprintf(stderr, "dv.sym = %p\n", dv.sym); 
      
      //debug_fprintf(stderr, "symbol %s  ADDING A DEPENDENCE OF TYPE ", dv.sym->name().c_str()); 
      //switch (dv.type) { 
      //case DEP_W2W: debug_fprintf(stderr, "DEP_W2W to "); break; 
      //case DEP_W2R: debug_fprintf(stderr, "DEP_W2R to "); break; 
      //case DEP_R2W: debug_fprintf(stderr, "DEP_R2W to "); break; 
      //case DEP_R2R: debug_fprintf(stderr, "DEP_R2R to "); break; 
      //default: debug_fprintf(stderr, "DEP_UNKNOWN to "); break;
      //} 
      //if (dep.dir == 0 || dep.dir == 1) debug_fprintf(stderr, "dependences1\n"); 
      //else debug_fprintf(stderr, "dependences2\n"); 
      
      if (dep.dir == 0 || dep.dir == 1) {
        //debug_fprintf(stderr, "pushing dv\n"); 
        dependences1.push_back(dv);
        //debug_fprintf(stderr, "DONE pushing dv\n"); 
        
        //debug_fprintf(stderr, "now %d dependences1\n", dependences1.size() ); 
        //for (int i=0; i<dependences1.size(); i++) {
        //  debug_fprintf(stderr, "dependences1[%d]: ", i ); 
        //  //debug_fprintf(stderr, "symbol %p ", dependences1[i].sym);
        //  debug_fprintf(stderr, "symbol ");
        //  debug_fprintf(stderr, "%s\n", dependences1[i].sym->name().c_str()); 
        //} 
        //debug_fprintf(stderr, "\n"); 
      }
      else { 
        //debug_fprintf(stderr, "pushing dv\n"); 
        dependences2.push_back(dv);
        //debug_fprintf(stderr, "DONE pushing dv\n"); 
        
        //debug_fprintf(stderr, "now %d dependences2\n", dependences2.size() ); 
        //for (int i=0; i<dependences2.size(); i++) {
        //  debug_fprintf(stderr, "dependences2[%d]: ", i); 
        //  //debug_fprintf(stderr, "symbol %p ", dependences2[i].sym);
        //  debug_fprintf(stderr, "symbol "); 
        //  debug_fprintf(stderr, "%s\n", dependences2[i].sym->name().c_str()); 
        //} 
        //debug_fprintf(stderr, "\n"); 
      }
      
      //debug_fprintf(stderr, "dv goes out of scope                                      ***\n"); 
    }
    else {
      //debug_fprintf(stderr, "now work on the next dimension level\n"); 
      // now work on the next dimension level
      int level = ++dep.level;
      //debug_fprintf(stderr, "level %d\n", level); 
      
      coef_t lbound, ubound;
      Relation delta = Deltas(copy(dep.r));
      //delta.print(); fflush(stdout); 
      delta.query_variable_bounds(delta.set_var(level), lbound, ubound);
      //debug_fprintf(stderr, "delta   lb " coef_fmt " 0x%llx    ub " coef_fmt " 0x%llx\n", lbound,lbound,ubound,ubound);
      
      
      if (dep.dir == 0) {
        //debug_fprintf(stderr, "dep.dir == 0\n"); 
        if (lbound > 0) {
          dep.dir = 1;
          dep.lbounds[level-1] = lbound;
          dep.ubounds[level-1] = ubound;
          
          //debug_fprintf(stderr, "push 1\n"); 
          working.push(dep);
        }
        else if (ubound < 0) {
          dep.dir = -1;
          dep.lbounds[level-1] = -ubound;
          dep.ubounds[level-1] = -lbound;
          
          //debug_fprintf(stderr, "push 2\n"); 
          working.push(dep);
        }
        else {
          // split the dependence vector into flow- and anti-dependence
          // for the first non-zero distance, also separate zero distance
          // at this level.
          {
            DependenceLevel dep2 = dep;
            
            dep2.lbounds[level-1] =  0;
            dep2.ubounds[level-1] =  0;
            
            F_And *f_root = dep2.r.and_with_and();
            EQ_Handle h = f_root->add_EQ();
            h.update_coef(dep2.r.input_var(level), 1);
            h.update_coef(dep2.r.output_var(level), -1);
            
            //debug_fprintf(stderr, "push 3\n"); 
            working.push(dep2);
          }
          
          //debug_fprintf(stderr, "lbound %lld 0x%llx\n", lbound, lbound); 
          //if (lbound < 0LL) debug_fprintf(stderr, "lbound < 0LL\n"); 
          //if (*ref_src != *ref_dst) debug_fprintf(stderr, "(*ref_src != *ref_dst)\n"); 
          //else debug_fprintf(stderr, "(*ref_src EQUAL *ref_dst)\n"); 
          
          if (lbound < 0LL && (*ref_src != *ref_dst)) {                // c == c
            DependenceLevel dep2 = dep;
            
            F_And *f_root = dep2.r.and_with_and();
            GEQ_Handle h = f_root->add_GEQ();
            h.update_coef(dep2.r.input_var(level), 1);
            h.update_coef(dep2.r.output_var(level), -1);
            h.update_const(-1);
            
            // get tighter bounds under new constraints
            coef_t lbound, ubound;
            delta = Deltas(copy(dep2.r));
            delta.query_variable_bounds(delta.set_var(level),
                                        lbound, ubound);
            
            dep2.dir = -1;            
            dep2.lbounds[level-1] = std::max(-ubound,static_cast<coef_t>(1)); // use max() to avoid Omega retardedness
            dep2.ubounds[level-1] = -lbound;
            
            //debug_fprintf(stderr, "push 4\n"); 
            working.push(dep2);
          }
          
          //debug_fprintf(stderr, "ubound %d\n", ubound); 
          if (ubound > 0) {
            DependenceLevel dep2 = dep;
            
            F_And *f_root = dep2.r.and_with_and();
            GEQ_Handle h = f_root->add_GEQ();
            h.update_coef(dep2.r.input_var(level), -1);
            h.update_coef(dep2.r.output_var(level), 1);
            h.update_const(-1);
            
            // get tighter bonds under new constraints
            coef_t lbound, ubound;
            delta = Deltas(copy(dep2.r));
            delta.query_variable_bounds(delta.set_var(level),
                                        lbound, ubound);
            dep2.dir = 1;
            dep2.lbounds[level-1] = std::max(lbound,static_cast<coef_t>(1)); // use max() to avoid Omega retardness
            dep2.ubounds[level-1] = ubound;
            
            //debug_fprintf(stderr, "push 5\n"); 
            working.push(dep2);
          }
        }
      }
      // now deal with dependence vector with known direction
      // determined at previous levels
      else {
        //debug_fprintf(stderr, "else messy\n"); 
        // For messy bounds, further test to see if the dependence distance
        // can be reduced to positive/negative.  This is an omega hack.
        if (lbound == negInfinity && ubound == posInfinity) {
          {
            Relation t = dep.r;
            F_And *f_root = t.and_with_and();
            GEQ_Handle h = f_root->add_GEQ();
            h.update_coef(t.input_var(level), 1);
            h.update_coef(t.output_var(level), -1);
            h.update_const(-1);
            
            if (!t.is_satisfiable()) {
              lbound = 0;
            }
          }
          {
            Relation t = dep.r;
            F_And *f_root = t.and_with_and();
            GEQ_Handle h = f_root->add_GEQ();
            h.update_coef(t.input_var(level), -1);
            h.update_coef(t.output_var(level), 1);
            h.update_const(-1);
            
            if (!t.is_satisfiable()) {
              ubound = 0;
            }
          }
        }
        
        // Same thing as above, test to see if zero dependence
        // distance possible.
        if (lbound == 0 || ubound == 0) {
          Relation t = dep.r;
          F_And *f_root = t.and_with_and();
          EQ_Handle h = f_root->add_EQ();
          h.update_coef(t.input_var(level), 1);
          h.update_coef(t.output_var(level), -1);
          
          if (!t.is_satisfiable()) {
            if (lbound == 0)
              lbound = 1;
            if (ubound == 0)
              ubound = -1;
          }
        }
        
        if (dep.dir == -1) {
          dep.lbounds[level-1] = -ubound;
          dep.ubounds[level-1] = -lbound;
        }
        else { // dep.dir == 1
          dep.lbounds[level-1] = lbound;
          dep.ubounds[level-1] = ubound;
        }
        
        //debug_fprintf(stderr, "push 6\n"); 
        working.push(dep);
      }
    }
    //debug_fprintf(stderr, "at bottom, size %d\n", working.size()); 
    
  }
  
  //debug_fprintf(stderr, "leaving relation2dependences, %d and %d dependences\n", dependences1.size(), dependences2.size()); 
  
  
  //for (int i=0; i<dependences1.size(); i++) {
  //debug_fprintf(stderr, "dependences1[%d]: ", i); 
  //debug_fprintf(stderr, "symbol %s\n", dependences1[i].sym->name().c_str()); 
  
  //debug_fprintf(stderr, "symbol %s  HAS A left  DEPENDENCE OF TYPE ", dependences1[i].sym->name().c_str()); 
  //switch (dependences1[i].type) { 
  //case DEP_W2W: debug_fprintf(stderr, "DEP_W2W\n");     break; 
  //case DEP_W2R: debug_fprintf(stderr, "DEP_W2R\n");     break; 
  //case DEP_R2W: debug_fprintf(stderr, "DEP_R2W\n");     break; 
  //case DEP_R2R: debug_fprintf(stderr, "DEP_R2R\n");     break; 
  //default:      debug_fprintf(stderr, "DEP_UNKNOWN\n"); break;
  //} 
  //} 
  
  
  //for (int i=0; i<dependences2.size(); i++) {
  
  //debug_fprintf(stderr, "symbol %s  HAS A right DEPENDENCE OF TYPE ", dependences2[i].sym->name().c_str()); 
  //switch (dependences2[i].type) { 
  //case DEP_W2W: debug_fprintf(stderr, "DEP_W2W\n");     break; 
  //case DEP_W2R: debug_fprintf(stderr, "DEP_W2R\n");     break; 
  //case DEP_R2W: debug_fprintf(stderr, "DEP_R2W\n");     break; 
  //case DEP_R2R: debug_fprintf(stderr, "DEP_R2R\n");     break; 
  //default:      debug_fprintf(stderr, "DEP_UNKNOWN\n"); break;
  //} 
  //} 
  
  
  
  return std::make_pair(dependences1, dependences2);
}


//-----------------------------------------------------------------------------
// Convert a boolean expression to omega relation.  "destroy" means shallow
// deallocation of "repr", not freeing the actual code inside.
//-----------------------------------------------------------------------------
void exp2constraint(Loop *loop, IR_Code *ir, Relation &r, F_And *f_root,
                    std::vector<Free_Var_Decl *> &freevars, CG_outputRepr *repr,
                    bool destroy,
                    std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols,
                    std::map<std::string, std::vector<omega::CG_outputRepr *> > &uninterpreted_symbols_stringrepr,
                    std::map<std::string, std::vector<omega::Relation> > &index_variables) {
  IR_CONDITION_TYPE cond = ir->QueryBooleanExpOperation(repr);
  switch (cond) {
    case IR_COND_LT:
    case IR_COND_LE:
    case IR_COND_EQ:
    case IR_COND_GT:
    case IR_COND_GE: {
      F_Exists *f_exist = f_root->add_exists();
      Variable_ID e = f_exist->declare();
      F_And *f_and = f_exist->add_and();
      std::vector<omega::CG_outputRepr *> op = ir->QueryExpOperand(repr);
      exp2formula(loop, ir, r, f_and, freevars, op[0], e, 's', IR_COND_EQ,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      exp2formula(loop, ir, r, f_and, freevars, op[1], e, 's', cond, true,
                  uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      if (destroy)
        delete repr;
      break;
    }
    case IR_COND_NE: {
      F_Exists *f_exist = f_root->add_exists();
      Variable_ID e = f_exist->declare();
      F_Or *f_or = f_exist->add_or();
      F_And *f_and = f_or->add_and();
      std::vector<omega::CG_outputRepr *> op = ir->QueryExpOperand(repr);
      exp2formula(loop, ir, r, f_and, freevars, op[0], e, 's', IR_COND_EQ,
                  false, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      exp2formula(loop, ir, r, f_and, freevars, op[1], e, 's', IR_COND_GT,
                  false, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);

      f_and = f_or->add_and();
      exp2formula(loop, ir, r, f_and, freevars, op[0], e, 's', IR_COND_EQ,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);
      exp2formula(loop, ir, r, f_and, freevars, op[1], e, 's', IR_COND_LT,
                  true, uninterpreted_symbols, uninterpreted_symbols_stringrepr,
                  index_variables);

      if (destroy)
        delete repr;
      break;
    }
    default:
      throw ir_exp_error("unrecognized conditional expression");
  }
}



//-----------------------------------------------------------------------------
// Generate iteration space constraints
//-----------------------------------------------------------------------------

// void add_loop_stride_constraints(Relation &r, F_And *f_root,
//                                  std::vector<Free_Var_Decl*> &freevars,
//                                  tree_for *tnf, char side) {

//   std::string name(tnf->index()->name());
//   int dim = 0;
//   for (;dim < r.n_set(); dim++)
//     if (r.set_var(dim+1)->name() == name)
//       break;

//   Relation bound = get_loop_bound(r, dim);

//   operand op = tnf->step_op();
//   if (!op.is_null()) {
//     if (op.is_immed()) {
//       immed im = op.immediate();
//       if (im.is_integer()) {
//         int c = im.integer();

//         if (c != 1 && c != -1)
//           add_loop_stride(r, bound, dim, c);
//       }
//       else
//         assert(0); // messy stride
//     }
//     else
//       assert(0);  // messy stride
//   }
// }




//-----------------------------------------------------------------------------
// Determine whether the loop (starting from 0) in the iteration space
// has only one iteration.
//-----------------------------------------------------------------------------
bool is_single_loop_iteration(const Relation &r, 
                              int level, 
                              const Relation &known) {
  int n = r.n_set();
  Relation r1;
  if(n > known.n_set()) { 
    r1 = Intersection(copy(r), Extend_Set(copy(known), n - known.n_set()));
  }
  else{
    r1 = Intersection(copy(known), Extend_Set(copy(r), known.n_set() - n));
    n = known.n_set();
  }
  
  
  Relation mapping(n, n);
  F_And *f_root = mapping.add_and();
  for (int i = 1; i <= level; i++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.input_var(i), 1);
    h.update_coef(mapping.output_var(i), -1);
  }
  r1 = Range(Restrict_Domain(mapping, r1));
  r1.simplify();
  
  Variable_ID v = r1.set_var(level);
  for (DNF_Iterator di(r1.query_DNF()); di; di++) {
    bool is_single = false;
    for (EQ_Iterator ei((*di)->EQs()); ei; ei++)
      if ((*ei).get_coef(v) != 0 && !(*ei).has_wildcards()) {
        is_single = true;
        break;
      }
    
    if (!is_single)
      return false;
  }
  
  return true;
}




bool is_single_iteration(const Relation &r, int dim) {
  assert(r.is_set());
  const int n = r.n_set();
  
  if (dim >= n)
    return true;
  
  Relation bound = get_loop_bound(r, dim);
  
  //   if (!bound.has_single_conjunct())
  //     return false;
  
  //   Conjunct *c = bound.query_DNF()->single_conjunct();
  
  for (DNF_Iterator di(bound.query_DNF()); di; di++) {
    bool is_single = false;
    for (EQ_Iterator ei((*di)->EQs()); ei; ei++)
      if (!(*ei).has_wildcards()) {
        is_single = true;
        break;
      }
    
    if (!is_single)
      return false;
  }
  
  return true;
  
  
  
  
  //   Relation r = copy(r_);
  //   const int n = r.n_set();
  
  //   if (dim >= n)
  //     return true;
  
  //   Relation bound = get_loop_bound(r, dim);
  //   bound = Approximate(bound);
  //   Conjunct *c = bound.query_DNF()->single_conjunct();
  
  //   return c->n_GEQs() == 0;
  
  
  
  
  
  //   Relation r = copy(r_);
  //   r.simplify();
  //   const int n = r.n_set();
  
  //   if (dim >= n)
  //     return true;
  
  //   for (DNF_Iterator i(r.query_DNF()); i; i++) {
  //     std::vector<bool> is_single(n);
  //     for (int j = 0; j < dim; j++)
  //       is_single[j] = true;
  //     for (int j = dim; j < n; j++)
  //       is_single[j] = false;
  
  //     bool found_new_single = true;
  //     while (found_new_single) {
  //       found_new_single = false;
  
  //       for (EQ_Iterator j = (*i)->EQs(); j; j++) {
  //         int saved_pos = -1;
  //         for (Constr_Vars_Iter k(*j); k; k++)
  //           if ((*k).var->kind() == Set_Var || (*k).var->kind() == Input_Var) {
  //             int pos = (*k).var->get_position() - 1;
  //             if (!is_single[pos])
  //               if (saved_pos == -1)
  //                 saved_pos = pos;
  //               else {
  //                 saved_pos = -1;
  //                 break;
  //               }
  //           }
  
  //         if (saved_pos != -1) {
  //           is_single[saved_pos] = true;
  //           found_new_single = true;
  //         }
  //       }
  
  //       if (is_single[dim])
  //         break;
  //     }
  
  //     if (!is_single[dim])
  //       return false;
  //   }
  
  //   return true;
}

//-----------------------------------------------------------------------------
// Set/get the value of a variable which is know to be constant.
//-----------------------------------------------------------------------------
void assign_const(Relation &r, int dim, int val) {
  const int n = r.n_out();
  
  Relation mapping(n, n);
  F_And *f_root = mapping.add_and();
  
  for (int i = 1; i <= n; i++) {
    if (i != dim+1) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(i), 1);
      h.update_coef(mapping.input_var(i), -1);
    }
    else {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.output_var(i), 1);
      h.update_const(-val);
    }
  }
  
  r = Composition(mapping, r);
}


int get_const(const Relation &r, int dim, Var_Kind type) {
  //  Relation rr = copy(r);
  Relation &rr = const_cast<Relation &>(r);
  
  Variable_ID v;
  switch (type) {
    // case Set_Var:
    //   v = rr.set_var(dim+1);
    //   break;
  case Input_Var:
    v = rr.input_var(dim+1);
    break;
  case Output_Var:
    v = rr.output_var(dim+1);
    break;
  default:
    throw std::invalid_argument("unsupported variable type");
  }
  
  for (DNF_Iterator di(rr.query_DNF()); di; di++)
    for (EQ_Iterator ei = (*di)->EQs(); ei; ei++)
      if ((*ei).is_const(v))
        return (*ei).get_const();
  
  throw std::runtime_error("cannot get variable's constant value");
}






//---------------------------------------------------------------------------
// Get the bound for a specific loop.
//---------------------------------------------------------------------------
Relation get_loop_bound(const Relation &r, int dim) {
  assert(r.is_set());
  const int n = r.n_set();
  
  //  Relation r1 = project_onto_levels(copy(r), dim+1, true);
  Relation mapping(n,n);
  F_And *f_root = mapping.add_and();
  for (int i = 1; i <= dim+1; i++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.input_var(i), 1);
    h.update_coef(mapping.output_var(i), -1);
  }
  Relation r1 = Range(Restrict_Domain(mapping, copy(r)));
  for (int i = 1; i <= n; i++)
    r1.name_set_var(i, const_cast<Relation &>(r).set_var(i)->name());
  r1.setup_names();
  Relation r2 = Project(copy(r1), dim+1, Set_Var);
  
  return Gist(r1, r2, 1);
}



Relation get_loop_bound(const Relation &r, int level, const Relation &known) {
  int n1 = r.n_set();
  int n = n1;
  Relation r1;
  if(n > known.n_set())
    r1 = Intersection(copy(r),
                      Extend_Set(copy(known), n - known.n_set()));
  else{
    r1 = Intersection(copy(known),
                      Extend_Set(copy(r), known.n_set() - n));
    n = known.n_set();
  }
  
  Relation mapping(n, n);
  F_And *f_root = mapping.add_and();
  for (int i = 1; i <= level; i++) {
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(mapping.input_var(i), 1);
    h.update_coef(mapping.output_var(i), -1);
  }
  r1 = Range(Restrict_Domain(mapping, r1));
  Relation r2 = Project(copy(r1), level, Set_Var);
  r1 = Gist(r1, r2, 1);
  
  for (int i = 1; i <= n1; i++)
    r1.name_set_var(i, const_cast<Relation &>(r).set_var(i)->name());
  r1.setup_names();
  
  return r1;
}



Relation get_max_loop_bound(const std::vector<Relation> &r, int dim) {
  if (r.size() == 0)
    return Relation::Null();
  
  const int n = r[0].n_set();
  Relation res(Relation::False(n));
  for (int i = 0; i < r.size(); i++) {
    Relation &t = const_cast<Relation &>(r[i]);
    if (t.is_satisfiable())
      res = Union(get_loop_bound(t, dim), res);
  }
  
  res.simplify();
  
  return res;
}

Relation get_min_loop_bound(const std::vector<Relation> &r, int dim) {
  if (r.size() == 0)
    return Relation::Null();
  
  const int n = r[0].n_set();
  Relation res(Relation::True(n));
  for (int i = 0; i < r.size(); i++) {
    Relation &t = const_cast<Relation &>(r[i]);
    if (t.is_satisfiable())
      res = Intersection(get_loop_bound(t, dim), res);
  }
  
  res.simplify();
  
  return res;
}

//-----------------------------------------------------------------------------
// Add strident to a loop.
// Issues:
// - Don't work with relations with multiple disjuncts.
// - Omega's dealing with max lower bound is awkward.
//-----------------------------------------------------------------------------
void add_loop_stride(Relation &r, const Relation &bound_, int dim, int stride) {
  F_And *f_root = r.and_with_and();
  Relation &bound = const_cast<Relation &>(bound_);
  for (DNF_Iterator di(bound.query_DNF()); di; di++) {
    F_Exists *f_exists = f_root->add_exists();
    Variable_ID e1 = f_exists->declare(tmp_e());
    Variable_ID e2 = f_exists->declare(tmp_e());
    F_And *f_and = f_exists->add_and();
    EQ_Handle stride_eq = f_and->add_EQ();
    stride_eq.update_coef(e1, 1);
    stride_eq.update_coef(e2, stride);
    if (!r.is_set())
      stride_eq.update_coef(r.output_var(dim+1), -1);
    else
      stride_eq.update_coef(r.set_var(dim+1), -1);
    F_Or *f_or = f_and->add_or();
    
    for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++) {
      if ((*gi).get_coef(bound.set_var(dim+1)) > 0) {
        // copy the lower bound constraint
        EQ_Handle h1 = f_or->add_and()->add_EQ();
        GEQ_Handle h2 = f_and->add_GEQ();
        for (Constr_Vars_Iter ci(*gi); ci; ci++) {
          switch ((*ci).var->kind()) {
            // case Set_Var:
          case Input_Var: 
            {
              int pos = (*ci).var->get_position();
              if (pos == dim + 1) {
                h1.update_coef(e1, (*ci).coef);
                h2.update_coef(e1, (*ci).coef);
              }
              else {
                if (!r.is_set()) {
                  h1.update_coef(r.output_var(pos), (*ci).coef);
                  h2.update_coef(r.output_var(pos), (*ci).coef);
                }
                else {
                  h1.update_coef(r.set_var(pos), (*ci).coef);
                  h2.update_coef(r.set_var(pos), (*ci).coef);
                }                
              }
              break;
            }
          case Global_Var: 
            {
              Global_Var_ID g = (*ci).var->get_global_var();
              h1.update_coef(r.get_local(g, (*ci).var->function_of()), (*ci).coef);
              h2.update_coef(r.get_local(g, (*ci).var->function_of()), (*ci).coef);
              break;
            }
          default:
            break;
          }
        }
        h1.update_const((*gi).get_const());
        h2.update_const((*gi).get_const());
      }
    }
  }
}


bool is_inner_loop_depend_on_level(const Relation &r, 
                                   int level,
                                   const Relation &known) {
  
  Relation r1;
  if(r.n_set() > known.n_set())
    r1 = Intersection(copy(r),
                      Extend_Set(copy(known), r.n_set() - known.n_set()));
  else
    r1 = Intersection(copy(known),
                      Extend_Set(copy(r), known.n_set() - r.n_set()));
  
  Relation r2 = copy(r1);
  for (int i = level+1; i <= r2.n_set(); i++)
    r2 = Project(r2, r2.set_var(i));
  r2.simplify(2, 4);
  Relation r3 = Gist(r1, r2);
  
  Variable_ID v = r3.set_var(level);
  for (DNF_Iterator di(r3.query_DNF()); di; di++) {
    for (EQ_Iterator ei = (*di)->EQs(); ei; ei++)
      if ((*ei).get_coef(v) != 0)
        return true;
    
    for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++)
      if ((*gi).get_coef(v) != 0)
        return true;
  }
  
  return false;
}


//-----------------------------------------------------------------------------
// Suppose loop dim is i. Replace i with i+adjustment in loop bounds.
// e.g. do i = 1, n
//        do j = i, n
// after call with dim = 0 and adjustment = 1:
//      do i = 1, n
//        do j = i+1, n
// -----------------------------------------------------------------------------
Relation adjust_loop_bound(const Relation &r, int level, int adjustment) {
  if (adjustment == 0)
    return copy(r);
  
  const int n = r.n_set();
  Relation r1 = copy(r);
  for (int i = level+1; i <= r1.n_set(); i++)
    r1 = Project(r1, r1.set_var(i));
  r1.simplify(2, 4);
  Relation r2 = Gist(copy(r), copy(r1));
  
  Relation mapping(n, n);
  F_And *f_root = mapping.add_and();
  for (int i = 1; i <= n; i++)
    if (i == level) {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.input_var(level), -1);
      h.update_coef(mapping.output_var(level), 1);
      h.update_const(static_cast<coef_t>(adjustment));
    }
    else {
      EQ_Handle h = f_root->add_EQ();
      h.update_coef(mapping.input_var(i), -1);
      h.update_coef(mapping.output_var(i), 1);
    }
  
  r2 = Range(Restrict_Domain(mapping, r2));
  r1 = Intersection(r1, r2);
  r1.simplify();
  
  for (int i = 1; i <= n; i++)
    r1.name_set_var(i, const_cast<Relation &>(r).set_var(i)->name());
  r1.setup_names();
  return r1;
}


// commented out on 07/14/2010
// void adjust_loop_bound(Relation &r, int dim, int adjustment, std::vector<Free_Var_Decl *> globals) {
//   assert(r.is_set());

//   if (adjustment == 0)
//     return;

//   const int n = r.n_set();
//   Tuple<std::string> name(n);
//   for (int i = 1; i <= n; i++)
//     name[i] = r.set_var(i)->name();

//   Relation r1 = project_onto_levels(copy(r), dim+1, true);
//   Relation r2 = Gist(copy(r), copy(r1));

//   // remove old bogus global variable conditions since we are going to
//   // update the value.
//   if (globals.size() > 0)
//     r1 = Gist(r1, project_onto_levels(copy(r), 0, true));

//   Relation r4 = Relation::True(n);

//     for (DNF_Iterator di(r2.query_DNF()); di; di++) {
//       for (EQ_Iterator ei = (*di)->EQs(); ei; ei++) {
//         EQ_Handle h = r4.and_with_EQ(*ei);

//         Variable_ID v = r2.set_var(dim+1);
//         coef_t c = (*ei).get_coef(v);
//         if (c != 0)
//           h.update_const(c*adjustment);

//         for (int i = 0; i < globals.size(); i++) {  
//           Variable_ID v = r2.get_local(globals[i]);
//           coef_t c = (*ei).get_coef(v);
//           if (c != 0)
//             h.update_const(c*adjustment);
//         }
//       }

//       for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++) {
//         GEQ_Handle h = r4.and_with_GEQ(*gi);

//         Variable_ID v = r2.set_var(dim+1);
//         coef_t c = (*gi).get_coef(v);
//         if (c != 0)
//           h.update_const(c*adjustment);

//         for (int i = 0; i < globals.size(); i++) {  
//           Variable_ID v = r2.get_local(globals[i]);
//           coef_t c = (*gi).get_coef(v);
//           if (c != 0)
//             h.update_const(c*adjustment);
//         }
//       }
//     }
//     r = Intersection(r1, r4);
// //   }
// //   else
// //     r = Intersection(r1, r2);

//   for (int i = 1; i <= n; i++)
//     r.name_set_var(i, name[i]);
//   r.setup_names();
// }


// void adjust_loop_bound(Relation &r, int dim, int adjustment) {
//   assert(r.is_set());
//   const int n = r.n_set();
//   Tuple<String> name(n);
//   for (int i = 1; i <= n; i++)
//     name[i] = r.set_var(i)->name();

//   Relation r1 = project_onto_levels(copy(r), dim+1, true);
//   Relation r2 = Gist(r, copy(r1));

//   Relation r3(n, n);
//   F_And *f_root = r3.add_and();
//   for (int i = 0; i < n; i++) {
//     EQ_Handle h = f_root->add_EQ();
//     h.update_coef(r3.output_var(i+1), 1);
//     h.update_coef(r3.input_var(i+1), -1);
//     if (i == dim)
//       h.update_const(adjustment);
//   }

//   r2 = Range(Restrict_Domain(r3, r2));
//   r = Intersection(r1, r2);

//   for (int i = 1; i <= n; i++)
//     r.name_set_var(i, name[i]);
//   r.setup_names();
// }  

// void adjust_loop_bound(Relation &r, int dim, Free_Var_Decl *global_var, int adjustment) {
//   assert(r.is_set());
//   const int n = r.n_set();
//   Tuple<String> name(n);
//   for (int i = 1; i <= n; i++)
//     name[i] = r.set_var(i)->name();

//   Relation r1 = project_onto_levels(copy(r), dim+1, true);
//   Relation r2 = Gist(r, copy(r1));

//   Relation r3(n);
//   Variable_ID v = r2.get_local(global_var);

//   for (DNF_Iterator di(r2.query_DNF()); di; di++) {
//     for (EQ_Iterator ei = (*di)->EQs(); ei; ei++) {
//       coef_t c = (*ei).get_coef(v);
//       EQ_Handle h = r3.and_with_EQ(*ei);
//       if (c != 0)
//         h.update_const(c*adjustment);
//     }
//     for (GEQ_Iterator gi = (*di)->GEQs(); gi; gi++) {
//       coef_t c = (*gi).get_coef(v);
//       GEQ_Handle h = r3.and_with_GEQ(*gi);
//       if (c != 0)
//         h.update_const(c*adjustment);
//     }
//   }

//   r = Intersection(r1, r3);
//   for (int i = 1; i <= n; i++)
//     r.name_set_var(i, name[i]);
//   r.setup_names();
// }



//------------------------------------------------------------------------------
// If the dimension has value posInfinity, the statement should be privatized
// at this dimension.
//------------------------------------------------------------------------------
// boolean is_private_statement(const Relation &r, int dim) {
//   int n;
//   if (r.is_set())
//     n = r.n_set();
//   else
//     n = r.n_out();

//   if (dim >= n)
//     return false;

//   try {
//     coef_t c;
//     if (r.is_set())
//       c = get_const(r, dim, Set_Var);
//     else
//       c = get_const(r, dim, Output_Var);
//     if (c == posInfinity)
//       return true;
//     else
//       return false;
//   }
//   catch (loop_error e){
//   }

//   return false;
// }



// // ----------------------------------------------------------------------------
// // Calculate v mod dividend based on equations inside relation r.
// // Return posInfinity if it is not a constant.
// // ----------------------------------------------------------------------------
// static coef_t mod_(const Relation &r_, Variable_ID v, int dividend, std::set<Variable_ID> &working_on) {
//   assert(dividend > 0);
//   if (v->kind() == Forall_Var || v->kind() == Exists_Var || v->kind() == Wildcard_Var)
//     return posInfinity;

//   working_on.insert(v);

//   Relation &r = const_cast<Relation &>(r_);
//   Conjunct *c = r.query_DNF()->single_conjunct();

//   for (EQ_Iterator ei(c->EQs()); ei; ei++) {
//     int coef = mod((*ei).get_coef(v), dividend);
//     if (coef != 1 && coef != dividend - 1 )
//       continue;

//     coef_t result = 0;
//     for (Constr_Vars_Iter cvi(*ei); cvi; cvi++)
//       if ((*cvi).var != v) {
//         int p = mod((*cvi).coef, dividend);

//         if (p == 0)
//           continue;

//         if (working_on.find((*cvi).var) != working_on.end()) {
//           result = posInfinity;
//           break;
//         }

//         coef_t q = mod_(r, (*cvi).var, dividend, working_on);
//         if (q == posInfinity) {
//           result = posInfinity;
//           break;
//         }
//         result += p * q;
//       }

//     if (result != posInfinity) {
//       result += (*ei).get_const();
//       if (coef == 1)
//         result = -result;
//       working_on.erase(v);

//       return mod(result, dividend);
//     }
//   }

//   working_on.erase(v);
//   return posInfinity;
// }


// coef_t mod(const Relation &r, Variable_ID v, int dividend) {
//   std::set<Variable_ID> working_on = std::set<Variable_ID>();

//   return mod_(r, v, dividend, working_on);
// }



//-----------------------------------------------------------------------------
// Generate mapping relation for permuation.
//-----------------------------------------------------------------------------
Relation permute_relation(const std::vector<int> &pi) {
  const int n = pi.size();
  
  Relation r(n, n);
  F_And *f_root = r.add_and();
  
  for (int i = 0; i < n; i++) {    
    EQ_Handle h = f_root->add_EQ();
    h.update_coef(r.output_var(i+1), 1);
    h.update_coef(r.input_var(pi[i]+1), -1);
  }
  
  return r;
}



//---------------------------------------------------------------------------
// Find the position index variable in a Relation by name.
//---------------------------------------------------------------------------
Variable_ID find_index(Relation &r, const std::string &s, char side) {
  // Omega quirks: assure the names are propagated inside the relation
  r.setup_names();
  if (r.is_set()) { // side == 's'
    for (int i = 1; i <= r.n_set(); i++) {
      std::string ss = r.set_var(i)->name();
      if (s == ss) {
        return r.set_var(i);
      } else if (s+"\'" == ss) {
        return r.set_var(i);
      }
      else if (s+"p" == ss) {    // Mahdi: 
        return r.set_var(i);
      }
    }
  }
  else if (side == 'w') {
    for (int i = 1; i <= r.n_inp(); i++) {
      std::string ss = r.input_var(i)->name();
      if (s == ss) {
        return r.input_var(i);
      }
    }
  }
  else { // side == 'r'
    for (int i = 1; i <= r.n_out(); i++) {
      std::string ss = r.output_var(i)->name();
      if (s+"\'" == ss) {
        return r.output_var(i);
      }
      else if (s+"p" == ss) {    // Mahdi: I added this else if, THE FIX FOR (') ISSUE for Gauss-Seidel example (only?)
        return r.output_var(i);
      }
    }
  }
  
  return NULL;
}

// EQ_Handle get_eq(const Relation &r, int dim, Var_Kind type) {
//   Variable_ID v;
//   switch (type) {
//   case Set_Var:
//     v = r.set_var(dim+1);
//     break;
//   case Input_Var:
//     v = r.input_var(dim+1);
//     break;
//   case Output_Var:
//     v = r.output_var(dim+1);
//     break;
//   default:
//     return NULL;
//   }
//   for (DNF_iterator di(r.query_DNF()); di; di++)
//     for (EQ_Iterator ei = (*di)->EQs(); ei; ei++)
//       if ((*ei).get_coef(v) != 0)
//         return (*ei);

//   return NULL;
// }


// std::Pair<Relation, Relation> split_loop(const Relation &r, const Relation &cond) {
//   Relation r1 = Intersection(copy(r), copy(cond));
//   Relation r2 = Intersection(copy(r), Complement(copy(cond)));

//   return std::Pair<Relation, Relation>(r1, r2);
// }



//----------------------------------------------------------------------------
//check if loop is normalized to zero
//----------------------------------------------------------------------------
bool lowerBoundIsZero(const omega::Relation &bound, int dim) {
  Relation &IS = const_cast<Relation &>(bound);
  Variable_ID v = IS.input_var(dim);
  bool found = false;
  for (DNF_Iterator di(IS.query_DNF()); di; di++) {
    bool is_single = false;
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++)
      if ((*gi).get_coef(v) >= 0 && !(*gi).is_const(v)
          && (*gi).get_const() != 0) {
        return false;
      } 
      else if ((*gi).get_coef(v) >= 0 && (*gi).is_const(v)
               && (*gi).get_const() == 0)
        found = true;
  }
  
  return found;
}



Relation replicate_IS_and_add_bound(const omega::Relation &R, int level,
                                    omega::Relation &bound) {
  
  if (!R.is_set())
    throw std::invalid_argument("Input R has to be a set not a relation!");
  
  Relation r(R.n_set());
  
  for (int i = 1; i <= R.n_set(); i++) {
    r.name_set_var(i + 1, const_cast<Relation &>(R).set_var(i)->name());
  }
  
  std::string new_var = bound.set_var(1)->name();
  
  r.name_set_var(level, new_var);
  
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++) {
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();
      
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
        case Input_Var:
          
          h.update_coef(r.input_var(v->get_position()),
                        cvi.curr_coef());
          
          break;
        case Wildcard_Var: 
          {
            Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                        f_exists, f_root, exists_mapping);
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
        case Global_Var: 
          {
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
  
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++) {
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++) {
      EQ_Handle h1 = f_root->add_EQ();
      
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
        case Input_Var:
          
          h1.update_coef(r.input_var(v->get_position()),
                         cvi.curr_coef());
          
          break;
        case Wildcard_Var: 
          {
            Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                        f_exists, f_root, exists_mapping);
            h1.update_coef(v2, cvi.curr_coef());
            break;
          }
        case Global_Var: 
          {
            Global_Var_ID g = v->get_global_var();
            Variable_ID v2;
            if (g->arity() == 0)
              v2 = r.get_local(g);
            else
              v2 = r.get_local(g, v->function_of());
            h1.update_coef(v2, cvi.curr_coef());
            break;
          }
        default:
          assert(false);
        }
      }
      h1.update_const((*gi).get_const());
    }
  }
  
  for (DNF_Iterator di(bound.query_DNF()); di; di++) {
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
        case Input_Var:
          
          h.update_coef(r.input_var(level), cvi.curr_coef());
          
          break;
        case Wildcard_Var: 
          {
            Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                        f_exists, f_root, exists_mapping);
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
        case Global_Var: 
          {
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
  
  for (DNF_Iterator di(bound.query_DNF()); di; di++) {
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++) {
      EQ_Handle h = f_root->add_EQ();
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
        case Input_Var:
          h.update_coef(r.input_var(level), cvi.curr_coef());
          
          break;
        case Wildcard_Var: 
          {
            Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                        f_exists, f_root, exists_mapping);
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
        case Global_Var: 
          {
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
  r.simplify();
  r.setup_names();
  return r;
}


//Return names of global vars with arity 0
std::set<std::string> get_global_vars(const omega::Relation &r) {
  std::set<std::string> vars;
  for (DNF_Iterator di(const_cast<Relation &>(r).query_DNF()); di; di++) {
    for (Constraint_Iterator gi((*di)->constraints()); gi; gi++) {
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {

        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
          case Global_Var: {
            Global_Var_ID g = v->get_global_var();
            Variable_ID v2;
            if (g->arity() == 0) {
              vars.insert(g->base_name());
            }
            break;
          }
          default:
            break;
        }
      }

    }
  }
  return vars;
}


// Mahdi: Adding this specifically for dependence extraction
//        This function basically renames tuple variables of old_relation
//        using tuple declaration of new_relation, and updates all constraints in the old_relation
//        and returns the updated relation. It can be used for doing something like following:
//        INPUTS:
//          old_relation: {[i,j]: col(j)=i && 0 < i < n}
//          new_relation: {[ip,jp]}
//        OUTPUT:
//                      : {[ip,jp]: col(jp)=ip && 0 < ip < n}
// Note: replace_set_var_as_another_set_var functions are suppose to do the same thing
//       but 1 tuple variable at a time, which is not efficient for dependence extractiion.
//       Also, those functions are not handling all kinds of variables.
Relation replace_set_vars(const omega::Relation &new_relation,
                                            const omega::Relation &old_relation) {
  Relation r = copy(new_relation);
  r.copy_names(new_relation);
  r.setup_names();
 
  Relation old = old_relation, new_r = new_relation;

  F_Exists *f_exists = r.and_with_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  // Just simply go over all the constraints and copy them over while changing the name of the tuple variables 
  for (DNF_Iterator di(const_cast<Relation &>(old_relation).query_DNF()); di; di++) {
    // Going over inequalities
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();
      h.update_const( (*gi).get_const() );             // Copy the constant
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        if ( v->kind() == Input_Var || v->kind() == Output_Var ) {

          h.update_coef(r.input_var( v->get_position() ), cvi.curr_coef());
        } else if ( v->kind() == Global_Var ) {

          Global_Var_ID g = v->get_global_var();
          Variable_ID v2;
          if (g->arity() == 0)  v2 = r.get_local(g);
          else v2 = r.get_local(g, v->function_of());
          h.update_coef(v2, cvi.curr_coef());
        } else if ( v->kind() == Wildcard_Var ) {

          Variable_ID v2 = 
          replicate_floor_definition( old_relation, v, r, f_exists, f_root,
                                                           exists_mapping);
          h.update_coef(v2, cvi.curr_coef());
        } else {
          throw omega_error("replace_set_var_as_another_set_var: Unknown Var!");
        }
      }
    }
    // Going over equalities
    for (EQ_Iterator ei((*di)->EQs()); ei; ei++) {
      EQ_Handle h = f_root->add_EQ();
      h.update_const( (*ei).get_const() );             // Copy the constant
      for (Constr_Vars_Iter cvi(*ei); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        if ( v->kind() == Input_Var || v->kind() == Output_Var ) {
          h.update_coef(r.input_var( v->get_position() ),
                              cvi.curr_coef());
        } else if ( v->kind() == Global_Var ) {
          Global_Var_ID g = v->get_global_var();
          Variable_ID v2;
          if (g->arity() == 0)  v2 = r.get_local(g);
          else v2 = r.get_local(g, v->function_of());
          h.update_coef(v2, cvi.curr_coef());
        } else if ( v->kind() == Wildcard_Var ) {

          Variable_ID v2 = 
          replicate_floor_definition( old_relation, v, r, f_exists, f_root,
                                                           exists_mapping);
          h.update_coef(v2, cvi.curr_coef());
        } else {
          throw omega_error("replace_set_var_as_another_set_var: Unknown Var!");
        }
      }
    }
  }

  return r;
}



// Replicates old_relation's bounds for set var at old_pos into new_relation at new_pos, but position's bounds must involve constants
//  only supports GEQs
//
Relation replace_set_var_as_another_set_var(const omega::Relation &new_relation,
                                            const omega::Relation &old_relation, int new_pos, int old_pos) {
  
  Relation r = copy(new_relation);
  r.copy_names(new_relation);
  r.setup_names();
  
  F_Exists *f_exists = r.and_with_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  
  for (DNF_Iterator di(const_cast<Relation &>(old_relation).query_DNF()); di;
       di++)
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();
      if (((*gi).get_coef(
                          const_cast<Relation &>(old_relation).set_var(old_pos)) != 0)
          && (*gi).is_const_except_for_global(
                                              const_cast<Relation &>(old_relation).set_var(
                                                                                           old_pos))) {
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
          case Input_Var: 
            {
              
              if (v->get_position() == old_pos)
                h.update_coef(r.input_var(new_pos),
                              cvi.curr_coef());
              else
                throw omega_error(
                                  "relation contains set vars other than that to be replicated!");
              break;
              
            }
          case Wildcard_Var: 
            {
              Variable_ID v2 = replicate_floor_definition(
                                                          old_relation, v, r, f_exists, f_root,
                                                          exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
          case Global_Var: 
            {
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
  return r;
  
}


Relation replace_set_var_as_another_set_var(const omega::Relation &new_relation,
                                            const omega::Relation &old_relation, int new_pos, int old_pos,
                                            std::map<int, int> &pos_mapping) {
  {
    copy(new_relation).print();
    copy(old_relation).print();
  }

  Relation r = copy(new_relation);
  r.copy_names(new_relation);
  r.setup_names();
  Relation alt(r.n_set());
  alt.copy_names(r);
  alt.setup_names();
  F_Exists *f_exists = r.and_with_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;

  Relation alt2(r.n_set());
  alt2.copy_names(r);
  alt2.setup_names();
  Relation r1 = copy(old_relation);
  r1.copy_names(old_relation);
  r1.setup_names();
  std::set<int> vars_to_project;

  bool ignore = false;
  for (DNF_Iterator di(const_cast<Relation &>(old_relation).query_DNF()); di;
       di++) {

    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      if ((*gi).get_coef(
          const_cast<Relation &>(old_relation).set_var(old_pos))
          != 0) {

        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
            case Input_Var: {

              if (v->get_position() > r.n_set()) {

                r1 = Project(r1, r1.set_var(v->get_position()));

                ignore = true;

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
              default:
                break;
            }
          }
        }
        /*&& (*gi).is_const_except_for_global(
         const_cast<Relation &>(old_relation).set_var(
         old_pos)))) */
        if (!ignore)
          alt.and_with_GEQ(*gi);
        else
          break;
      }
      if (ignore)
        break;
    }

    if (ignore)
      break;
    for (EQ_Iterator ei((*di)->EQs()); ei; ei++) {
      EQ_Handle h = f_root->add_EQ();
      if ((*ei).get_coef(
          const_cast<Relation &>(old_relation).set_var(old_pos)) != 0)
        /*&& (*ei).is_const_except_for_global(
         const_cast<Relation &>(old_relation).set_var(
         old_pos)))) */
      {

        for (Constr_Vars_Iter cvi(*ei); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
            case Input_Var: {

              if (v->get_position() > r.n_set())
                ignore = true;

              r1 = Project(r1, r1.set_var(v->get_position()));

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
            default:
              break;
          }
        }

        if (!ignore)
          alt.and_with_EQ(*ei);
        else
          break;
      }
      if (ignore)
        break;
    }
    if (ignore)
      break;
  }

  if (ignore) {
    ignore = false;
    for (DNF_Iterator di(const_cast<Relation &>(r1).query_DNF()); di;
         di++) {

      for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
        if ((*gi).get_coef(const_cast<Relation &>(r1).set_var(old_pos))
            != 0) {

          for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            switch (v->kind()) {
              case Input_Var: {

                if (v->get_position() > r.n_set()) {

                  //r1 = Project(r1, r1.set_var(v->get_position()));

                  ignore = true;

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
                default:
                  break;
              }
            }
          }
          /*&& (*gi).is_const_except_for_global(
           const_cast<Relation &>(old_relation).set_var(
           old_pos)))) */
          if (!ignore)
            alt2.and_with_GEQ(*gi);
          else
            break;
        }
        if (ignore)
          break;
      }

      if (ignore)
        break;
      for (EQ_Iterator ei((*di)->EQs()); ei; ei++) {
        EQ_Handle h = f_root->add_EQ();
        if ((*ei).get_coef(const_cast<Relation &>(r1).set_var(old_pos))
            != 0)
          /*&& (*ei).is_const_except_for_global(
           const_cast<Relation &>(old_relation).set_var(
           old_pos)))) */
        {

          for (Constr_Vars_Iter cvi(*ei); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            switch (v->kind()) {
              case Input_Var: {

                if (v->get_position() > r.n_set())
                  ignore = true;

                //r1 = Project(r1, r1.set_var(v->get_position()));

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
              default:
                break;
            }
          }

          if (!ignore)
            alt2.and_with_EQ(*ei);
          else
            break;
        }
        if (ignore)
          break;
      }
      if (ignore)
        break;
    }

  }

  for (DNF_Iterator di(const_cast<Relation &>(old_relation).query_DNF()); di;
       di++) {
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();
      if (((*gi).get_coef(
          const_cast<Relation &>(old_relation).set_var(old_pos)) != 0)) {
        //&& (*gi).is_const_except_for_global(
        //		const_cast<Relation &>(old_relation).set_var(
        //				old_pos))) {
        bool not_const = true;
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
            case Input_Var: {

              if (v->get_position() != old_pos) {
                bool found = false;
                int new_pos2 = -1;
                for (std::map<int, int>::iterator a =
                    pos_mapping.begin(); a != pos_mapping.end();
                     a++)
                  if(a->second == v->get_position()){
                    found = true;
                    new_pos2 = a->second;

                  }

                if(found){
                  if (new_pos2 > new_pos)
                    vars_to_project.insert(v->get_position());
                }
                else
                  vars_to_project.insert(v->get_position());
              }

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

            default:
              break;
          }
        }
        //h.update_const((*gi).get_const());
      }
    }
    for (EQ_Iterator ei((*di)->EQs()); ei; ei++) {
      EQ_Handle h = f_root->add_EQ();
      if (((*ei).get_coef(
          const_cast<Relation &>(old_relation).set_var(old_pos)) != 0)) {
        //&& (*ei).is_const_except_for_global(
        //		const_cast<Relation &>(old_relation).set_var(
        //				old_pos))) {
        for (Constr_Vars_Iter cvi(*ei); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
            case Input_Var: {
              if (v->get_position() != old_pos) {
                bool found = false;
                int new_pos2 = -1;
                for (std::map<int, int>::iterator a =
                    pos_mapping.begin(); a != pos_mapping.end();
                     a++)
                  if(a->second == v->get_position()){
                    found = true;
                    new_pos2 = a->second;

                  }

                if(found){
                  if (new_pos2 > new_pos)
                    vars_to_project.insert(v->get_position());
                }
                else
                  vars_to_project.insert(v->get_position());
              }

              break;

            }
            default:
              break;

          }
          //h.update_const((*ei).get_const());
        }

      }
    }
  }

  Relation temp = copy(old_relation);

  for (std::set<int>::reverse_iterator it = vars_to_project.rbegin();
       it != vars_to_project.rend(); it++) {
    temp = Project(temp, temp.set_var(*it));
    temp.simplify(2, 4);
  }

  for (DNF_Iterator di(const_cast<Relation &>(temp).query_DNF()); di; di++) {
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();
      if (((*gi).get_coef(const_cast<Relation &>(temp).set_var(old_pos))
           != 0)) {
        //&& (*gi).is_const_except_for_global(
        //		const_cast<Relation &>(old_relation).set_var(
        //				old_pos))) {
        bool not_const = true;
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
            case Input_Var: {

              if (v->get_position() == old_pos)
                h.update_coef(r.input_var(new_pos),
                              cvi.curr_coef());
              else if (v->get_position() > r.n_set()) {

                return alt2;
              } else
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
              Variable_ID v2 = replicate_floor_definition(temp, v, r,
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
      EQ_Handle h = f_root->add_EQ();
      if (((*ei).get_coef(const_cast<Relation &>(temp).set_var(old_pos))
           != 0)) {
        //&& (*ei).is_const_except_for_global(
        //		const_cast<Relation &>(old_relation).set_var(
        //				old_pos))) {
        for (Constr_Vars_Iter cvi(*ei); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
            case Input_Var: {

              if (v->get_position() == old_pos)
                h.update_coef(r.input_var(new_pos),
                              cvi.curr_coef());
              else if (v->get_position() > r.n_set())
                return alt2;
              else
                h.update_coef(r.input_var(v->get_position()),
                              cvi.curr_coef());
              //	else
              //		throw omega_error(
              //				"relation contains set vars other than that to be replicated!");
              break;

            }
            case Wildcard_Var: {
              Variable_ID v2 = replicate_floor_definition(temp, v, r,
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
  return r;

}


//-----------------------------------------------------------------------------
// Copy all relations from r and add new bound at position indicated by level.
// -----------------------------------------------------------------------------

Relation replicate_IS_and_add_at_pos(const omega::Relation &R, int level,
                                     omega::Relation &bound) {
  
  if (!R.is_set())
    throw std::invalid_argument("Input R has to be a set not a relation!");
  
  Relation r(R.n_set() + 1);
  
  for (int i = 1; i <= R.n_set(); i++) {
    if (i < level)
      r.name_set_var(i, const_cast<Relation &>(R).set_var(i)->name());
    else
      r.name_set_var(i + 1, const_cast<Relation &>(R).set_var(i)->name());
    
  }
  
  std::string new_var = bound.set_var(1)->name();
  
  r.name_set_var(level, new_var);
  
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  
  for (int i = 1; i <= R.n_set(); i++)
    for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++) {
      for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
        GEQ_Handle h = f_root->add_GEQ();
        if ((*gi).get_coef(const_cast<Relation &>(R).set_var(i)) != 0) {
          for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            switch (v->kind()) {
            case Input_Var: 
              {
                if (i < level)
                  h.update_coef(r.input_var(v->get_position()),
                                cvi.curr_coef());
                else
                  h.update_coef(
                                r.input_var(v->get_position() + 1),
                                cvi.curr_coef());
                break;
              }
            case Wildcard_Var: 
              {
                Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                            f_exists, f_root, exists_mapping);
                h.update_coef(v2, cvi.curr_coef());
                break;
              }
            case Global_Var: 
              {
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
    }
  
  for (int i = 1; i <= R.n_set(); i++)
    for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++) {
      for (EQ_Iterator gi((*di)->EQs()); gi; gi++) {
        EQ_Handle h1 = f_root->add_EQ();
        if ((*gi).get_coef(const_cast<Relation &>(R).set_var(i)) != 0) {
          for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            switch (v->kind()) {
            case Input_Var: 
              {
                if (i < level)
                  h1.update_coef(r.input_var(v->get_position()),
                                 cvi.curr_coef());
                else
                  h1.update_coef(
                                 r.input_var(v->get_position() + 1),
                                 cvi.curr_coef());
                break;
              }
            case Wildcard_Var: 
              {
                Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                            f_exists, f_root, exists_mapping);
                h1.update_coef(v2, cvi.curr_coef());
                break;
              }
            case Global_Var: 
              {
                Global_Var_ID g = v->get_global_var();
                Variable_ID v2;
                if (g->arity() == 0)
                  v2 = r.get_local(g);
                else
                  v2 = r.get_local(g, v->function_of());
                h1.update_coef(v2, cvi.curr_coef());
                break;
              }
            default:
              assert(false);
            }
          }
          h1.update_const((*gi).get_const());
        }
      }
    }
  
  for (DNF_Iterator di(bound.query_DNF()); di; di++) {
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
        case Input_Var: 
          {
            if (cvi.curr_var()->get_position() < level)
              h.update_coef(r.input_var(level), cvi.curr_coef());
            else
              h.update_coef(r.input_var(level), cvi.curr_coef());
            break;
          }
        case Wildcard_Var: 
          {
            Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                        f_exists, f_root, exists_mapping);
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
        case Global_Var: 
          {
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
        h.update_const((*gi).get_const());
      }
    }
  }
  
  for (DNF_Iterator di(bound.query_DNF()); di; di++) {
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++) {
      EQ_Handle h = f_root->add_EQ();
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
        case Input_Var: 
          {
            if (cvi.curr_var()->get_position() < level)
              h.update_coef(r.input_var(level), cvi.curr_coef());
            else
              h.update_coef(r.input_var(level), cvi.curr_coef());
            break;
          }
        case Wildcard_Var: 
          {
            Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                        f_exists, f_root, exists_mapping);
            h.update_coef(v2, cvi.curr_coef());
            break;
          }
        case Global_Var: 
          {
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
  r.simplify();
  r.setup_names();
  return r;
}




omega::Relation replace_set_var_as_Global(const omega::Relation &R, int pos,
                                          std::vector<omega::Relation> &bound) {
  
  if (!R.is_set())
    throw std::invalid_argument("Input R has to be a set not a relation!");
  
  Relation r(R.n_set());
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  int count = 1;
  for (int i = 1; i <= R.n_set(); i++) {
    
    if (i != pos) {
      r.name_set_var(i, const_cast<Relation &>(R).set_var(i)->name());
      
    }
    else
      r.name_set_var(i, "void");
  }
  
  Free_Var_Decl *repl = new Free_Var_Decl(
                                          const_cast<Relation &>(R).set_var(pos)->name());
  
  Variable_ID v3 = r.get_local(repl);
  
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++) {
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++) {
      EQ_Handle h1 = f_root->add_EQ();
      for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
        Variable_ID v = cvi.curr_var();
        switch (v->kind()) {
        case Input_Var: 
          {
            if (v->get_position() != pos)
              h1.update_coef(r.input_var(v->get_position()),
                             cvi.curr_coef());
            else
              
              h1.update_coef(v3, cvi.curr_coef());
            break;
          }
        case Wildcard_Var: 
          {
            Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                        f_exists, f_root, exists_mapping);
            h1.update_coef(v2, cvi.curr_coef());
            break;
          }
        case Global_Var: 
          {
            Global_Var_ID g = v->get_global_var();
            Variable_ID v2;
            if (g->arity() == 0)
              v2 = r.get_local(g);
            else
              v2 = r.get_local(g, v->function_of());
            h1.update_coef(v2, cvi.curr_coef());
            break;
          }
        default:
          assert(false);
        }
      }
      h1.update_const((*gi).get_const());
    }
  }
  
  for (int i = 0; i < bound.size(); i++)
    for (DNF_Iterator di(bound[i].query_DNF()); di; di++) {
      for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
        GEQ_Handle h = f_root->add_GEQ();
        if ((*gi).get_coef(bound[i].set_var(pos)) == 0) {
          for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
            Variable_ID v = cvi.curr_var();
            switch (v->kind()) {
            case Input_Var: 
              {
                //if (i < level)
                if (v->get_position() != pos)
                  h.update_coef(r.input_var(v->get_position()),
                                cvi.curr_coef());
                else
                  
                  h.update_coef(v3, cvi.curr_coef());
                break;
                
                //else
                //  h.update_coef(
                //      r.input_var(v->get_position() + 1),
                //      cvi.curr_coef());
                break;
              }
            case Wildcard_Var: 
              {
                Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                            f_exists, f_root, exists_mapping);
                h.update_coef(v2, cvi.curr_coef());
                break;
              }
            case Global_Var: 
              {
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
    }
  return r;
}





//-----------------------------------------------------------------------------
// Replace an input variable's constraints as an existential in  order
// to simplify other constraints in Relation
// -----------------------------------------------------------------------------
std::pair<Relation, bool> replace_set_var_as_existential(
                                                         const omega::Relation &R, int pos,
                                                         std::vector<omega::Relation> &bound) {
  
  if (!R.is_set())
    throw std::invalid_argument("Input R has to be a set not a relation!");
  
  Relation r(R.n_set());
  for (int i = 1; i <= R.n_set(); i++)
    r.name_set_var(i, const_cast<Relation &>(R).set_var(i)->name());
  
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  int coef_in_equality = 0;
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++)
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++)
      if (((*gi).get_coef(const_cast<Relation &>(R).set_var(pos)) != 0)
          && (!(*gi).has_wildcards()))
        if (coef_in_equality == 0)
          coef_in_equality = (*gi).get_coef(
                                            const_cast<Relation &>(R).set_var(pos));
        else
          return std::pair<Relation, bool>(copy(R), false);
  
  if (coef_in_equality < 0)
    coef_in_equality = -coef_in_equality;
  
  std::pair<EQ_Handle, Variable_ID> result = find_simplest_stride(
                                                                  const_cast<Relation &>(R), const_cast<Relation &>(R).set_var(pos));
  
  if (result.second == NULL && coef_in_equality != 1)
    return std::pair<Relation, bool>(copy(R), false);
  
  if (result.second != NULL) {
    if (result.first.get_coef(const_cast<Relation &>(R).set_var(pos)) != 1)
      return std::pair<Relation, bool>(copy(R), false);
    
    if (result.first.get_coef(result.second) != coef_in_equality)
      return std::pair<Relation, bool>(copy(R), false);
  }
  Variable_ID v3 = f_exists->declare();
  
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++) {
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++) {
      EQ_Handle h1 = f_root->add_EQ();
      if ((*gi).get_coef(const_cast<Relation &>(R).set_var(pos)) == 0
          || !(*gi).has_wildcards()) {
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
          case Input_Var: 
            {
              if (v->get_position() != pos)
                h1.update_coef(r.input_var(v->get_position()),
                               cvi.curr_coef());
              else
                
                h1.update_coef(v3, cvi.curr_coef());
              break;
            }
          case Wildcard_Var: 
            {
              Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                          f_exists, f_root, exists_mapping);
              h1.update_coef(v2, cvi.curr_coef());
              break;
            }
          case Global_Var: 
            {
              Global_Var_ID g = v->get_global_var();
              Variable_ID v2;
              if (g->arity() == 0)
                v2 = r.get_local(g);
              else
                v2 = r.get_local(g, v->function_of());
              h1.update_coef(v2, cvi.curr_coef());
              break;
            }
          default:
            assert(false);
          }
        }
        h1.update_const((*gi).get_const());
      }
    }
  }
  
  for (int i = 0; i < bound.size(); i++)
    for (DNF_Iterator di(bound[i].query_DNF()); di; di++) {
      for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
        GEQ_Handle h = f_root->add_GEQ();
        //if ((*gi).get_coef(const_cast<Relation &>(R).set_var(i)) != 0) {
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
          case Input_Var: 
            {
              //if (i < level)
              if (v->get_position() != pos)
                
                h.update_coef(r.set_var(v->get_position()),
                              cvi.curr_coef());
              else
                
                h.update_coef(v3, cvi.curr_coef());
              
              //else
              //  h.update_coef(
              //      r.input_var(v->get_position() + 1),
              //      cvi.curr_coef());
              break;
            }
          case Wildcard_Var: 
            {
              Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                          f_exists, f_root, exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
          case Global_Var: 
            {
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
        //}
      }
    }
  
  //for (int i = 1; i <= R.n_set(); i++)
  return std::pair<Relation, bool>(r, true);
}





//-----------------------------------------------------------------------------
// Copy all relations from r except those for set var v.
// And with GEQ given by g
// NOTE: This function only removes the relations involving v if they are simple relations
// involving only v but not complex relations that have v in other variables' constraints
// -----------------------------------------------------------------------------
Relation and_with_relation_and_replace_var(const Relation &R, Variable_ID v1,
                                           Relation &g) {
  if (!R.is_set())
    throw std::invalid_argument("Input R has to be a set not a relation!");
  
  Relation r(R.n_set());
  
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  EQ_Handle h = f_root->add_EQ();
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++) {
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++) {
      EQ_Handle h = f_root->add_EQ();
      if (!(*gi).is_const(v1) && !(*gi).is_const_except_for_global(v1)) {
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
          case Input_Var: 
            {
              h.update_coef(r.input_var(v->get_position()),
                            cvi.curr_coef());
              break;
            }
          case Wildcard_Var: 
            {
              Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                          f_exists, f_root, exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
          case Global_Var: 
            {
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
    
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++) {
      GEQ_Handle h = f_root->add_GEQ();
      if ((*gi).get_coef(v1) == 0) {
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
          case Input_Var: 
            {
              h.update_coef(r.input_var(v->get_position()),
                            cvi.curr_coef());
              break;
            }
          case Wildcard_Var: 
            {
              Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                          f_exists, f_root, exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
          case Global_Var: 
            {
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
  }
  for (DNF_Iterator di(const_cast<Relation &>(g).query_DNF()); di; di++)
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++)
      r.and_with_GEQ(*gi);
  
  for (DNF_Iterator di(const_cast<Relation &>(g).query_DNF()); di; di++)
    for (EQ_Iterator gi((*di)->EQs()); gi; gi++)
      r.and_with_EQ(*gi);
  
  r.simplify();
  r.copy_names(R);
  r.setup_names();
  return r;
}





omega::Relation extract_upper_bound(const Relation &R, Variable_ID v1) {
  if (!R.is_set())
    throw std::invalid_argument("Input R has to be a set not a relation!");
  
  Relation r(R.n_set());
  
  F_Exists *f_exists = r.add_and()->add_exists();
  F_And *f_root = f_exists->add_and();
  std::map<Variable_ID, Variable_ID> exists_mapping;
  GEQ_Handle h = f_root->add_GEQ();
  for (DNF_Iterator di(const_cast<Relation &>(R).query_DNF()); di; di++)
    for (GEQ_Iterator gi((*di)->GEQs()); gi; gi++)
      if ((*gi).get_coef(v1) < 0) {
        
        for (Constr_Vars_Iter cvi(*gi); cvi; cvi++) {
          Variable_ID v = cvi.curr_var();
          switch (v->kind()) {
          case Input_Var:
            h.update_coef(r.input_var(v->get_position()),
                          cvi.curr_coef());
            break;
          case Wildcard_Var: 
            {
              Variable_ID v2 = replicate_floor_definition(R, v, r,
                                                          f_exists, f_root, exists_mapping);
              h.update_coef(v2, cvi.curr_coef());
              break;
            }
          case Global_Var: 
            {
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
  
  r.simplify();
  
  return r;
  
}

/*CG_outputRepr * modified_output_subs_repr(CG_outputBuilder * ocg, const Relation &R, const EQ_Handle &h, Variable_ID v,const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
  std::map<std::string, std::vector<CG_outputRepr *> > unin){
  
  
  
  
  }
*/





CG_outputRepr * construct_int_floor(CG_outputBuilder * ocg, const Relation &R,
                                    const GEQ_Handle &h, Variable_ID v,
                                    const std::vector<std::pair<CG_outputRepr *, int> > &assigned_on_the_fly,
                                    std::map<std::string, std::vector<CG_outputRepr *> > unin) {
  
  std::set<Variable_ID> excluded_floor_vars;
  const_cast<Relation &>(R).setup_names(); // hack
  assert(v->kind() == Set_Var);
  
  int a = h.get_coef(v);
  
  CG_outputRepr *lhs = ocg->CreateIdent(v->name());
  excluded_floor_vars.insert(v);
  std::vector<std::pair<bool, GEQ_Handle> > result2;
  CG_outputRepr *repr = NULL;
  for (Constr_Vars_Iter cvi(h); cvi; cvi++)
    if (cvi.curr_var() != v) {
      CG_outputRepr *t;
      if (cvi.curr_var()->kind() == Wildcard_Var) {
        std::pair<bool, GEQ_Handle> result = find_floor_definition(R,
                                                                   cvi.curr_var(), excluded_floor_vars);
        if (!result.first) {
          delete repr;
          throw omega_error(
                            "Can't generate bound expression with wildcard not involved in floor definition");
        }
        
        try {
          t = output_inequality_repr(ocg, result.second,
                                     cvi.curr_var(), R, assigned_on_the_fly, unin,
                                     excluded_floor_vars);
        } catch (const std::exception &e) {
          delete repr;
          throw e;
        }
      } 
      else
        t = output_ident(ocg, R, cvi.curr_var(), assigned_on_the_fly,
                         unin);
      
      coef_t coef = cvi.curr_coef();
      if (a > 0) {
        if (coef > 0) {
          if (coef == 1)
            repr = ocg->CreateMinus(repr, t);
          else
            repr = ocg->CreateMinus(repr,
                                    ocg->CreateTimes(ocg->CreateInt(coef), t));
        } 
        else {
          if (coef == -1)
            repr = ocg->CreatePlus(repr, t);
          else
            repr = ocg->CreatePlus(repr,
                                   ocg->CreateTimes(ocg->CreateInt(-coef), t));
        }
      } 
      else {
        if (coef > 0) {
          if (coef == 1)
            repr = ocg->CreatePlus(repr, t);
          else
            repr = ocg->CreatePlus(repr,
                                   ocg->CreateTimes(ocg->CreateInt(coef), t));
        } 
        else {
          if (coef == -1)
            repr = ocg->CreateMinus(repr, t);
          else
            repr = ocg->CreateMinus(repr,
                                    ocg->CreateTimes(ocg->CreateInt(-coef), t));
        }
      }
    }
  coef_t c = h.get_const();
  if (c > 0) {
    if (a > 0)
      repr = ocg->CreateMinus(repr, ocg->CreateInt(c));
    else
      repr = ocg->CreatePlus(repr, ocg->CreateInt(c));
  } 
  else if (c < 0) {
    if (a > 0)
      repr = ocg->CreatePlus(repr, ocg->CreateInt(-c));
    else
      repr = ocg->CreateMinus(repr, ocg->CreateInt(-c));
  }
  
  if (abs(a) == 1)
    ocg->CreateAssignment(0, lhs, repr);
  else if (a > 0)
    return ocg->CreateAssignment(0, lhs,
                                 ocg->CreateIntegerCeil(repr, ocg->CreateInt(a)));
  else
    // a < 0
    return ocg->CreateAssignment(0, lhs,
                                 ocg->CreateIntegerFloor(repr, ocg->CreateInt(-a)));
  
}




