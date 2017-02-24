//
// Created by ztuowen on 2/21/17.
//

#ifndef CHILL_PRINTER_DATA_H
#define CHILL_PRINTER_DATA_H

#include <string>
#include <ostream>

namespace chill {
  namespace printer {
    struct pParam{
      std::string indent;
      std::ostream &o;
      pParam(std::string ind, std::ostream &o):indent(ind),o(o) {}
    };
  }
}

#endif //CHILL_PRINTER_DATA_H
