#ifndef _MYFLEX_H
#define _MYFLEX_H

#ifndef yyFlexLexerOnce
#include <FlexLexer.h>
#endif
#include <iostream>
#include <string>
#include <vector>

class myFlexLexer: public yyFlexLexer {
protected:
  std::string cur_line;
  int cur_pos;
  std::vector<std::string> history;
  int first_history_pos;
  int last_history_pos;
  std::vector<std::string> key_seqs;
  
public:
  myFlexLexer(std::istream *arg_yyin = NULL, std::ostream *arg_yyout = NULL);
  ~myFlexLexer() {}
protected:
  int LexerInput(char *buf, int max_size);
};

#endif
