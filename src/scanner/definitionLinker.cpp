//
// Created by ztuowen on 2/19/17.
//

#include "scanner/definitionLinker.h"
#include <map>
#include <vector>
#include <string>

using namespace chill::scanner;

typedef vector<std::map<std::string, chillAST_node *> > declmap;

void DefinitionLinker::exec(chillAST_node *n) {
  declmap decls(1);
  run(n, (void *) &decls);
}

void DefinitionLinker::runS(chillAST_TypedefDecl *n, void *o) {
  declmap *decl = (declmap *) o;
  decl->back()[std::string(n->newtype)] = n;
  if (n->rd)
  for (auto s : n->rd->getSubparts())
    run(s,o);
}

void DefinitionLinker::runS(chillAST_RecordDecl *n, void *o) {
  declmap *decl = (declmap *) o;
  decl->back()[std::string(n->getName())] = n;
  for (auto s : n->getSubparts())
  run(s,o);
}

void DefinitionLinker::runS(chillAST_FunctionDecl *n, void *o) {
  declmap *decl = (declmap *) o;
  decl->back()[std::string(n->functionName)] = n;
  decl->emplace_back();
  auto parms = n->getParameterSymbolTable();
  for (auto it = parms->begin(); it != parms->end(); ++it)
    run(*it, o);
  for (int i = 0; i < n->getNumChildren(); ++i)
    run(n->getChild(i), o);
  decl->pop_back();
}

void DefinitionLinker::runS(chillAST_CompoundStmt *n, void *o) {
  declmap *decl = (declmap *) o;
  decl->emplace_back();
  for (int i = 0; i < n->getNumChildren(); ++i)
    run(n->getChild(i), o);
  decl->pop_back();
}

void DefinitionLinker::runS(chillAST_VarDecl *n, void *o) {
  declmap *decl = (declmap *) o;
  decl->back()[std::string(n->varname)] = n;
  auto it = decl->end();
  if (n->vardef)
    run(n->vardef, o);
  char * vartype;
  if ( !strncmp( n->vartype, "struct ", 7))
    vartype = &(n->vartype[7]);
  else
    vartype = n->vartype;
  if (!n->vardef && !n->typedefinition) {
    do {
      --it;
      auto ret = (*it).find(vartype);
      if (!(ret == (*it).end())) {
        if (static_cast<chillAST_RecordDecl *>(ret->second))
          n->vardef = (chillAST_RecordDecl *) ret->second;
        return;
      }
    } while (it != decl->begin());
    debug_printf("Looking for def %s\n", vartype);
  }
}

void DefinitionLinker::runS(chillAST_DeclRefExpr *n, void *o) {
  declmap *decl = (declmap *) o;
  auto it = decl->end();
  do {
    --it;
    auto ret = (*it).find(n->declarationName);
    if (!(ret == (*it).end())) {
      n->decl = ret->second;
      return;
    }
  } while (it != decl->begin());
  chill_error_printf("declaration name %s unknown\n", n->declarationName);
}
