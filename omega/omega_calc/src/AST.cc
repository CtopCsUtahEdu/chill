/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2009-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   build relation from parsed input.

 Notes:

 History:
*****************************************************************************/

#include <omega_calc/AST.h>
#include <string.h>

using namespace omega;

Global_Declaration_Site *globalDecls; 
Declaration_Site *relationDecl = NULL; 
tupleDescriptor *currentTupleDescriptor;
std::map<omega::Const_String, Variable_Ref *> functionOfInput;
std::map<omega::Const_String, Variable_Ref *> functionOfOutput;

AST_constraints::AST_constraints(std::set<Exp *> *f, Rel_Op r, AST_constraints *o) {
  others = o;
  rel_op = r;
  first = f;
}

AST_constraints::AST_constraints(std::set<Exp *> *f, Rel_Op r, std::set<Exp *> *s) {
  others = new AST_constraints(s);
  rel_op = r;
  first = f;
}

AST_constraints::AST_constraints(std::set<Exp *> *f){
  others = 0;
  first = f;
}

AST_constraints::~AST_constraints() {
  for (std::set<Exp *>::iterator i = first->begin(); i != first->end(); i++)
    delete *i;
  delete first;
  delete others;
}

void AST_constraints::print() {
  for (std::set<Exp *>::iterator i = first->begin(); ;) {
    printf(coef_fmt, (*i)->constantTerm);
    for (std::map<Variable_Ref *, omega::coef_t>::iterator j = (*i)->coefs.begin(); j != (*i)->coefs.end(); j++)
      printf("+" coef_fmt "%s", (*j).second, static_cast<const char *>((*j).first->name));
    i++;
    if (i != first->end())
      printf(", ");
    else
      break;
  }
}


Exp::Exp(coef_t c) : coefs() {
  constantTerm = c;
}

Exp::Exp(Variable_Ref *v) : coefs() {
  assert(v != 0);
  constantTerm = 0;
  coefs[v] = 1;
}

Exp *negate (Exp *x) {
  x->constantTerm = -x->constantTerm;
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = x->coefs.begin(); i != x->coefs.end(); i++)
    (*i).second = -(*i).second;
  return x;
}

Exp *add (Exp *x, Exp *y) {
  x->constantTerm += y->constantTerm;
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = y->coefs.begin(); i != y->coefs.end(); i++)
    x->coefs[(*i).first] += (*i).second;
  delete y;
  return x;
}

Exp *subtract (Exp *x, Exp *y) {
  x->constantTerm -= y->constantTerm;
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = y->coefs.begin(); i != y->coefs.end(); i++)
    x->coefs[(*i).first] -= (*i).second;
  delete y;
  return x;
}

Exp *multiply (coef_t c, Exp *x) {
  x->constantTerm *= c;
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = x->coefs.begin(); i != x->coefs.end(); i++)
    (*i).second *= c;
  return x;
}

Exp *multiply (Exp *x, Exp *y) {
  bool found_nonzero = false;
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = x->coefs.begin(); i != x->coefs.end(); i++)
    if ((*i).second != 0)
      found_nonzero = true;
  if (!found_nonzero) {
    coef_t c = x->constantTerm;
    delete x;
    return multiply(c,y);
  }

  found_nonzero = false;
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = y->coefs.begin(); i != y->coefs.end(); i++)
    if ((*i).second != 0)
      found_nonzero = true;
  if (!found_nonzero) {
    coef_t c = y->constantTerm;
    delete y;
    return multiply(c,x);
  }
  
  delete x;
  delete y;
  throw std::runtime_error("invalid exp multiply");
}



Declaration_Site *current_Declaration_Site = 0;
       
Declaration_Site::Declaration_Site()  {
  previous = current_Declaration_Site;
  current_Declaration_Site = this;
}

Declaration_Site::Declaration_Site(std::set<char *> *v) {
  previous = current_Declaration_Site;
  current_Declaration_Site = this;
  for (std::set<char *>::iterator i = v->begin(); i != v->end(); i++)
    declarations.insert(new Variable_Ref(*i));
}

Declaration_Site::~Declaration_Site() {
  for (std::set<Variable_Ref *>::iterator i = declarations.begin(); i != declarations.end(); i++)
    delete *i;
}

Variable_Ref::Variable_Ref(char *s, int _arity, Argument_Tuple _of) {
  name = s;
  arity = _arity;
  of = _of;
  anonymous = !strncmp("In_",s,3) || !strncmp("Out_",s,4);
  char *t = s;
  while (*t != '\0') t++;
  t--;
  while (*t == '\'') t--;
  t++;
  *t = '\0';
  stripped_name = s;
  g = 0;
}

Variable_Ref::Variable_Ref(char *s) {
  name = s;
  arity = 0;
  anonymous = !strncmp("In_",s,3) || !strncmp("Out_",s,4);
  char *t = s;
  while (*t != '\0') t++;
  t--;
  while (*t == '\'') t--;
  t++;
  *t = '\0';
  stripped_name = s;
  g = 0;
}
 
Variable_Ref::Variable_Ref() {
  name = "#anonymous";
  arity = 0;
  anonymous = 1;
  stripped_name = name;
  g = 0;
}

Variable_Ref::~Variable_Ref() {
  assert(g == 0);
}

Variable_Ref *lookupScalar(char *s) {
  Declaration_Site *ds;
  for(ds = current_Declaration_Site; ds; ds = ds->previous)
    for (std::set<Variable_Ref *>::iterator i = ds->declarations.begin(); i != ds->declarations.end(); i++)
      if ((*i)->name == static_cast<Const_String>(s))
        return (*i);
  return NULL;
}
 
Declaration_Site *defined(char *s) {
  Declaration_Site *ds;
  for(ds = current_Declaration_Site; ds; ds = ds->previous)
    for (std::set<Variable_Ref *>::iterator i = ds->declarations.begin(); i != ds->declarations.end(); i++)
      if ((*i)->name == static_cast<Const_String>(s))
        return ds;
  return NULL;
}


void AST_Or::install(Formula *F) {
  if (F->node_type() != Op_Or) 
    F = F->add_or();
  left->install(F);
  right->install(F);
}

void AST_And::install(Formula *F) {
  if (F->node_type() != Op_And) F = F->add_and();
  left->install(F);
  right->install(F);
}

void AST_Not::install(Formula *F) {
  child->install(F->add_not());
}

void AST_exists::install(Formula *F) {
  F_Exists *G = F->add_exists();
  for (std::set<Variable_Ref *>::iterator i = declaredVariables->declarations.begin(); i != declaredVariables->declarations.end(); i++)
    (*i)->vid = G->declare((*i)->stripped_name);
  child->install(G);
}

void AST_forall::install(Formula *F) {
  F_Forall *G = F->add_forall();
  for (std::set<Variable_Ref *>::iterator i = declaredVariables->declarations.begin(); i != declaredVariables->declarations.end(); i++)
    (*i)->vid = G->declare((*i)->stripped_name);
  child->install(G);
}

void AST_constraints::install(Formula *F) {
  if (!others) return;
  F_And *f =  F->and_with();

  for (std::set<Exp *>::iterator i = first->begin(); i != first->end(); i++)
    for (std::set<Exp *>::iterator j = others->first->begin(); j != others->first->end(); j++)
      switch (rel_op) {
      case(lt) : install_gt(f, *j, *i); break;
      case(gt) : install_gt(f, *i, *j); break;
      case(leq) : install_geq(f, *j, *i); break;
      case(geq) : install_geq(f, *i, *j); break;
      case(eq) : install_eq(f, *i, *j); break;
      case(neq) : install_neq(f, *i, *j); break;
      default : assert(0);
      }
  others->install(f);
}
  
 
void install_neq(F_And *F, Exp *e1, Exp *e2) {
  F_Or *or_ = F->add_or();
  F_And *and1 = or_->add_and();
  F_And *and2 = or_->add_and();
  install_gt(and1,e1,e2);
  install_gt(and2,e2,e1);
};

void install_stride(F_And *F, strideConstraint *s) {
  Stride_Handle c = F->add_stride(s->step);
  c.update_const(s->e->constantTerm);
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = s->e->coefs.begin(); i != s->e->coefs.end(); i++)
    c.update_coef((*i).first->id(F->relation()), (*i).second);
  c.finalize();
}

void install_eq(F_And *F, Exp *e1, Exp *e2) {
  EQ_Handle c = F->add_EQ();
  c.update_const(e1->constantTerm);
  if (e2) c.update_const(-e2->constantTerm);
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = e1->coefs.begin(); i != e1->coefs.end(); i++)
    c.update_coef((*i).first->id(F->relation()), (*i).second);
  if (e2)
    for (std::map<Variable_Ref *, omega::coef_t>::iterator i = e2->coefs.begin(); i != e2->coefs.end(); i++)
      c.update_coef((*i).first->id(F->relation()), -(*i).second);
  c.finalize();
}
  
void install_geq(F_And *F, Exp *e1, Exp *e2) {
  GEQ_Handle c = F->add_GEQ();
  c.update_const(e1->constantTerm);
  if (e2) c.update_const(-e2->constantTerm);
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = e1->coefs.begin(); i != e1->coefs.end(); i++)
    c.update_coef((*i).first->id(F->relation()), (*i).second);
  if (e2)
    for (std::map<Variable_Ref *, omega::coef_t>::iterator i = e2->coefs.begin(); i != e2->coefs.end(); i++)
      c.update_coef((*i).first->id(F->relation()), -(*i).second);
  c.finalize();
}

void install_gt(F_And *F, Exp *e1, Exp *e2) {
  GEQ_Handle c = F->add_GEQ();
  c.update_const(-1);
  c.update_const(e1->constantTerm);
  if (e2) c.update_const(-e2->constantTerm);
  for (std::map<Variable_Ref *, omega::coef_t>::iterator i = e1->coefs.begin(); i != e1->coefs.end(); i++)
    c.update_coef((*i).first->id(F->relation()), (*i).second);
  if (e2)
    for (std::map<Variable_Ref *, omega::coef_t>::iterator i = e2->coefs.begin(); i != e2->coefs.end(); i++)
      c.update_coef((*i).first->id(F->relation()), -(*i).second);
  c.finalize();
}
  

Global_Declaration_Site::~Global_Declaration_Site() {
/*   // Take care of global variables - since we do that kludge */
/*   // of declaring globals twice if arity > 0, we must take care */
/*   // not to just delete each global once per declaration. */

/*   // Actually, we can't free these, since Relations containing references to */
/*  // this may get freed later */
/*   foreach(v,Variable_Ref*,this->declarations,v->g=0); */
/*   //Set<Free_Var_Decl *> globals; */
/*   //foreach(v,Variable_Ref*,this->declarations,(globals.insert(v->g),v->g=0)); */
/*   //foreach(g,Free_Var_Decl*,globals,delete g); */

  // Only delete global variables here. --chun 5/28/2008
  for (std::set<Variable_Ref *>::iterator i= declarations.begin(); i != declarations.end(); i++) {
    if ((*i)->g != 0) {
      if ((*i)->arity != 0) { // functional symbols
        // only delete once from a pair of "(in)" and "(out)" variables
        const char *name = static_cast<const char *>((*i)->name);
        const char *s = "(in)";
        bool match = true;
        for (size_t p = strlen(name)-4, q = 0; p < strlen(name); p++, q++)
          if (s[q] != name[p]) {
            match = false;
            break;
          }
        if (match)
          delete (*i)->g;
      }
      else // not functions
        delete (*i)->g;
      
      (*i)->g = 0;
    }
  }
}

Variable_Ref * Global_Declaration_Site::extend(char *s) {
  Variable_Ref *r  = new Variable_Ref(s);
  r->g = new Free_Var_Decl(r->stripped_name);
  declarations.insert(r);
  return r;
}
 
void Global_Declaration_Site::extend_both_tuples(char *s, int arity) {
  if (arity == 0)
    extend(s);
  else {
    assert(arity > 0);

    char s1[strlen(s)+5], s2[strlen(s)+6];
    strcpy(s1,s); strcat(s1,"(in)");
    strcpy(s2,s); strcat(s2,"(out)");
    Const_String name = s;

    Variable_Ref *r1 = new Variable_Ref(s1, arity, Input_Tuple);
    Variable_Ref *r2 = new Variable_Ref(s2, arity, Output_Tuple);
    r1->g = r2->g = new Free_Var_Decl(s,arity);
 
    functionOfInput[name] = r1;
    functionOfOutput[name] = r2;

    declarations.insert(r1);
    declarations.insert(r2);
  }
}
 

void resetGlobals() {
  for (std::set<Variable_Ref *>::iterator i = globalDecls->declarations.begin(); i != globalDecls->declarations.end(); i++)
    (*i)->vid = 0;
}


Variable_Ref *Declaration_Site::extend(char *s) {
  Variable_Ref *r  = new Variable_Ref(s);
  declarations.insert(r);
  return r;
}

Variable_Ref *Declaration_Site::extend(char *s, Argument_Tuple of, int pos) {
  Variable_Ref *r  = new Variable_Ref(s);
  declarations.insert(r);
  r->of = of;
  r->pos = pos;
  return r;
}

Variable_Ref * Declaration_Site::extend() {
  Variable_Ref *r  = new Variable_Ref();
  declarations.insert(r);
  return r;
}
 
void tupleDescriptor::extend(char *s) {
  Variable_Ref *r = relationDecl->extend(s);
  size++;
  vars.push_back(r);
  assert(size == vars.size());
}

void tupleDescriptor::extend(char *s, Argument_Tuple of, int pos) {
  Variable_Ref *r  = relationDecl->extend(s, of, pos);
  size++;
  vars.push_back(r);
  assert(size == vars.size());
}

void tupleDescriptor::extend(Exp *e) {
  Variable_Ref *r  = relationDecl->extend();
  size++;
  vars.push_back(r);
  assert(size == vars.size());
  Exp *eq = subtract(e, new Exp(r));
  eq_constraints.insert(eq); 
}

void tupleDescriptor::extend(char *s, Exp *e) {
  Variable_Ref *r  = relationDecl->extend(s);
  size++;
  vars.push_back(r);
  assert(size == vars.size());
  Exp *eq = subtract(e, new Exp(r));
  eq_constraints.insert(eq); 
}

void tupleDescriptor::extend() {
  Variable_Ref *r  = relationDecl->extend();
  size++;
  vars.push_back(r);
  assert(size == vars.size());
}
void tupleDescriptor::extend(Exp *lb,Exp *ub) {
  Variable_Ref *r  = relationDecl->extend();
  size++;
  vars.push_back(r);
  assert(size == vars.size());
  Exp *lb_exp = subtract(new Exp(r), lb);
  geq_constraints.insert(lb_exp); 
  Exp *ub_exp = subtract(ub, new Exp(r));
  geq_constraints.insert(ub_exp); 
}
void tupleDescriptor::extend(Exp *lb,Exp *ub, coef_t stride) {
  Variable_Ref *r  = relationDecl->extend();
  size++;
  vars.push_back(r);
  Exp *lb_exp = subtract(new Exp(r), new Exp(*lb));
  geq_constraints.insert(lb_exp); 
  Exp *ub_exp = subtract(ub, new Exp(r));
  geq_constraints.insert(ub_exp); 
  strideConstraint *s = new strideConstraint;
  s->e = subtract(lb, new Exp(r));
  s->step = stride;
  stride_constraints.insert(s); 
}
