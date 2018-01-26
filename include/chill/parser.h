//
// Created by Tuowen Zhao on 7/2/17.
//

#ifndef CHILL_PARSER_H
#define CHILL_PARSER_H

#include "chill_ast.hh"

extern vector<chillAST_VarDecl *> VariableDeclarations;
extern vector<chillAST_FunctionDecl *> FunctionDeclarations;

namespace chill {
  class Parser {
    /*!
     * \brief Parser uses frontend to parse the source file and translates them into chillAST
     */
  public:
    //! the parsed AST
    chillAST_SourceFile *entire_file_AST;

    Parser() : entire_file_AST(NULL) {}
    virtual ~Parser() = default;

    /*!
     * @brief Parse a new file
     * @param procname Needed to find the right file containing the function
     */
    virtual void parse(std::string filename, std::string procname) = 0;
  };
}

#endif //CHILL_PARSER_H
