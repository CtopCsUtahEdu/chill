//
// Created by ztuowen on 9/24/16.
//

#ifndef CHILL_PRINTER_H_H
#define CHILL_PRINTER_H_H

#include "scanner.h"
#include "chill_ast.hh"
#include <string>
#include <ostream>

namespace chill {
  namespace printer {
    /*!
     * \brief this is a generic AST printSer that prints the code out to a C-family like syntax
     */
    class GenericPrinter : public Scanner<std::string, std::ostream&> {
    protected:
      std::string identSpace;

      //! default error output when encountering not recognized error code
      /*!
       * @param ident
       * @param n
       * @param o
       */
      virtual void errorRunS(chillAST_node *n, std::string indent, std::ostream &o);

    public:
      GenericPrinter() { identSpace = "  "; }
      void run(chillAST_node *n, std::string indent, std::ostream &o);
      //! Set the indentation for print
      /*!
       * Some subclass has indentation unused, like Dump. Also, only spaces is supported,
       * so it is a number of the spaces in the indentaion.
       * @param numspaces number of spaces for the indentation
       */
      void setIndentSpace(int numspaces) {
        identSpace = "";
        for (int i = 0; i < numspaces; ++i)
          identSpace += "  ";
      }
      //! Print the AST to string stream, multiplexer
      /*!
       * @param indent indentation of the node
       * @param n the chillAST_Node
       * @param o the string stream
       */
      virtual void print(chillAST_node *n, std::string indent, std::ostream &o);
      //! Print the AST to string, overload the printS function
      /*!
       * @param indent indentation of the node
       * @param n the chillAST_Node
       * @return a string of the corresponding code
       */
      virtual std::string print(chillAST_node *n, std::string indent);
      //! Print the AST to stdout
      /*!
       * @param indent indentation of the node, the one inherited from the parent
       * @param n the chillAST_Node
       */
      virtual void printOut(chillAST_node *n, std::string indent) {
        print(n, indent, std::cout);
      }
      //! Print the AST to stdErr
      /*!
       * @param indent indentation of the node
       * @param n the chillAST_Node
       */
      virtual void printErr(chillAST_node *n, std::string indent) {
        print(n, indent, std::cerr);
      }
    };
  }
}

#endif //CHILL_PRINTER_H_H
