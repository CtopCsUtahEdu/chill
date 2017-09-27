%option prefix="expr"

%{
// some C++ code
#include "chill_run_util.hh"
#include "parse_expr.tab.hh"
%}

%option noyywrap

%%
[ \t]+                  /*ignore*/
\n                      /*ignore*/
L[0-9]+                 { exprlval.val = atoi(&yytext[1]); return LEVEL; }
[0-9]+                  { exprlval.val = atoi(yytext); return NUMBER; }
\<\=                    return LE;
\>\=                    return GE;
\=(\=)?                 return EQ;
[a-zA-Z_][a-zA-Z_0-9]*  {
                           exprlval.str_val = new char[yyleng+1];
                           strcpy(exprlval.str_val, yytext);
                           return VARIABLE;
                         }
.                        return (int)yytext[0];
%%

