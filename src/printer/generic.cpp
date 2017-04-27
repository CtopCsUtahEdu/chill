//
// Created by ztuowen on 9/24/16.
//

#include "printer/generic.h"
#include "printer/data.h"
#include "scanner.h"
#include <sstream>

using namespace chill::printer;

void GenericPrinter::run(chillAST_node *n, std::string indent, std::ostream &o) {
  chill::Scanner<std::string, std::ostream&>::run(n, indent, o);
  o.flush();
}

void GenericPrinter::print(chillAST_node *n, std::string indent, std::ostream &o) {
  run(n,indent, o);
}

std::string GenericPrinter::print(chillAST_node *n, std::string indent) {
  std::ostringstream os;
  print(n, indent, os);
  return os.str();
}

void GenericPrinter::errorRunS(chillAST_node *n, std::string indent, std::ostream &o) {
  chill_error_printf("Unhandled case in printer: %s\n", n->getTypeString());
}