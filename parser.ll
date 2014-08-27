/*****************************************************************************
 Copyright (C) 2008 University of Southern California. 
 All Rights Reserved.

 Purpose:
   CHiLL script lexical analysis

 Update history:
   created by Chun Chen, Jan 2008
*****************************************************************************/

%{
#include <stdio.h>
#include <string.h>
#include <vector>
#include <map>
#include "parser.tab.hh"

extern std::map<std::string, int> parameter_tab;
extern bool is_interactive;
extern const char *PROMPT_STRING;
%}

%s LINE COMMENT FILE_NAME PROCEDURE_NAME
%option yylineno
%option noyywrap

%%
#                      BEGIN(COMMENT);
<COMMENT>.*            /* comment */
source                 BEGIN(FILE_NAME); return SOURCE;
<FILE_NAME>[^ \t\n:#]+ yylval.name = new char[yyleng+1]; strcpy(yylval.name, yytext); return FILENAME;
procedure              BEGIN(LINE); return PROCEDURE;
loop                   BEGIN(LINE); return LOOP;
format                 BEGIN(FILE_NAME); return FORMAT;
original               BEGIN(LINE); return ORIGINAL;
permute                BEGIN(LINE); return PERMUTE;
pragma                 BEGIN(LINE); return PRAGMA;
prefetch               BEGIN(LINE); return PREFETCH;
tile                   BEGIN(LINE); return TILE;
datacopy               BEGIN(LINE); return DATACOPY;
datacopy_privatized    BEGIN(LINE); return DATACOPY_PRIVATIZED;
unroll                 BEGIN(LINE); return UNROLL;
unroll_extra           BEGIN(LINE); return UNROLL_EXTRA;
split                  BEGIN(LINE); return SPLIT;
nonsingular            BEGIN(LINE); return NONSINGULAR;
print                  BEGIN(LINE); return PRINT;
dep                    BEGIN(LINE); return PRINT_DEP;
code                   BEGIN(LINE); return PRINT_CODE;
space                  BEGIN(LINE); return PRINT_IS;                     
exit                   BEGIN(LINE); return EXIT;
known                  BEGIN(LINE); return KNOWN;
strided                BEGIN(LINE); return STRIDED;
counted                BEGIN(LINE); return COUNTED;
num_statement          BEGIN(LINE); return NUM_STATEMENT;
ceil                   BEGIN(LINE); return CEIL;
floor                  BEGIN(LINE); return FLOOR;
true                   BEGIN(LINE); yylval.bool_val = true; return TRUEORFALSE;
false                  BEGIN(LINE); yylval.bool_val = false; return TRUEORFALSE;
skew                   BEGIN(LINE); return SKEW;
shift                  BEGIN(LINE); return SHIFT;
scale                  BEGIN(LINE); return SCALE;
reverse                BEGIN(LINE); return REVERSE;
shift_to               BEGIN(LINE); return SHIFT_TO;
fuse                   BEGIN(LINE); return FUSE;
peel                   BEGIN(LINE); return PEEL;
distribute             BEGIN(LINE); return DISTRIBUTE;
remove_dep             BEGIN(LINE); return REMOVE_DEP;
structure              BEGIN(LINE); return PRINT_STRUCTURE;
[ \t]+                 /* ignore whitespaces */
\n                     BEGIN(INITIAL); return (int)yytext[0];
L[0-9]+                yylval.val = atoi(&yytext[1]); return LEVEL;
[a-zA-Z_][a-zA-Z_0-9]* {
                         BEGIN(LINE);
                         yylval.name = new char[yyleng+1];
                         strcpy(yylval.name, yytext);
                         return VARIABLE;
                       }
\"(\\.|[^\\"])*\"      {
                         BEGIN(LINE);
                         std::string str = std::string(yytext);
                         yylval.name = new char[yyleng-1];
                         str = str.substr(1,yyleng-2);
                         strcpy(yylval.name, str.c_str());
                         return STRING;
                       }
[0-9]+                 yylval.val = atoi(yytext); return NUMBER;
\>\=                   return GE;
\<\=                   return LE;
\!\=                   return NE;
\=\=                   return EQ;
.                      return (int)yytext[0];
<LINE><<EOF>>          BEGIN(INITIAL); unput('\n');

%%


