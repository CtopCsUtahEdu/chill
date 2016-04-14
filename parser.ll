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
#define ONCE
#include "parser.hh"
int zzlex(){return lexer.yylex();}

extern std::map<std::string, int> parameter_tab;
extern bool is_interactive;
extern const char *PROMPT_STRING;
%}

%s LINE COMMENT FILE_NAME PROCEDURE_NAME
%option yylineno
%option noyywrap
%option prefix="zz"
%%
#                      BEGIN(COMMENT);
<COMMENT>.*            /* comment */
source                 BEGIN(FILE_NAME); return SOURCE;
dest                   BEGIN(FILE_NAME); return DEST;
<FILE_NAME>[^ \t\n:#]+ zzlval.name = new char[yyleng+1]; strcpy(zzlval.name, yytext); return FILENAME;
procedure              BEGIN(LINE); return PROCEDURE;
loop                   BEGIN(LINE); return LOOP;
find_stencil_shape     BEGIN(LINE); return FIND_STENCIL_SHAPE;
stencil_temp           BEGIN(LINE); return STENCIL_TEMP;
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
true                   BEGIN(LINE); zzlval.bool_val = true; return TRUEORFALSE;
false                  BEGIN(LINE); zzlval.bool_val = false; return TRUEORFALSE;
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
reduce                 BEGIN(LINE); return REDUCE;  /* Added by Manu */ 
scalar_expand          BEGIN(LINE); return SCALAR_EXPAND;  /* Added by Manu */
split_with_alignment   BEGIN(LINE); return SPLIT_WITH_ALIGNMENT; /* Added by Anand */
coalesce               BEGIN(LINE); return FLATTEN; 
normalize              BEGIN(LINE); return NORMALIZE;
ELLify                 BEGIN(LINE); return ELLIFY; 
compact                BEGIN(LINE); return COMPACT; 
make_dense         BEGIN(LINE); return MAKE_DENSE;  
set_array_size         BEGIN(LINE); return SET_ARRAY_SIZE;
add_ghosts	       BEGIN(LINE); return ADD_GHOSTS;

[ \t]+                 /* ignore whitespaces */
\n                     BEGIN(INITIAL); return (int)yytext[0];
L[0-9]+                zzlval.val = atoi(&yytext[1]); return LEVEL;
[a-zA-Z_][a-zA-Z_0-9]* {
                         BEGIN(LINE);
                         zzlval.name = new char[yyleng+1];
                         strcpy(zzlval.name, yytext);
                         return VARIABLE;
                       }
[0-9]+                 zzlval.val = atoi(yytext); return NUMBER;
\>\=                   return GE;
\<\=                   return LE;
\!\=                   return NE;
\=\=                   return EQ;
.                      return (int)yytext[0];
<LINE><<EOF>>          BEGIN(INITIAL); unput('\n');

%%


