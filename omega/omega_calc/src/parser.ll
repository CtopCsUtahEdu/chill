/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   lex parser for calculator.

 Notes:

 History:
   02/04/11 migrate to flex c++ mode, Chun Chen
*****************************************************************************/

%{
#include <stdio.h>
#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <omega_calc/AST.h>
#include <basic/Dynamic_Array.h>
#include "parser.tab.hh"
#include <omega_calc/myflex.h>

myFlexLexer mylexer;
bool is_interactive;
const char *PROMPT_STRING = ">>>";

#define BUFFER scanBuf += yytext
std::string scanBuf;
std::string err_msg;
  
extern bool need_coef;

void yyerror(const std::string &s);
void flushScanBuffer();

%}

%s LATEX INCLUDE COMMENT
%option yylineno
%option noyywrap

%% 

"<<"                  { BUFFER; BEGIN(INCLUDE); }
<INCLUDE>[^>\n ]+">>" {
  BUFFER;
  scanBuf += "\n";
  flushScanBuffer();

  if (is_interactive) {
    std::cout << "file include disabled in interactive mode\n";
  }
  else {    
    char *s = yytext;
    while (*s != '>') s++;
    *s = '\0';
    std::ifstream *ifs = new std::ifstream(yytext, std::ifstream::in);
    if (!ifs->is_open()) {
      fprintf(stderr, "Can't open file %s\n", yytext);
    }
    else {
      yy_buffer_state *bs = mylexer.yy_create_buffer(ifs, 8092);
      mylexer.yypush_buffer_state(bs);
    }
  }
  BEGIN(INITIAL);
}
<INCLUDE>[ \n] {
  std::cout << "Error in include syntax\n";
  std::cout << "Use <<fname>> to include the file named fname\n";
  BEGIN(INITIAL);
  if(is_interactive) {
    std::cout << PROMPT_STRING << ' ';
    std::cout.flush();
  }
}





<LATEX>"\\ "  { BUFFER; }
[ \t]+        { BUFFER; }
#             { BUFFER; BEGIN(COMMENT); }
<COMMENT>.*   { BUFFER; }
<LATEX>"\$\$\n" { BUFFER; BEGIN(INITIAL); }
<LATEX>"\$\$" { BUFFER; BEGIN(INITIAL); }
"\$\$"        { BUFFER; BEGIN(LATEX); }
<LATEX>"\\n"   { BUFFER; }
<LATEX>"\\t"  { BUFFER; }
<LATEX>"\\!"  { BUFFER; }
<LATEX>"\\\\" { BUFFER; }
<LATEX>"\n"   { BUFFER; }


\n {
  BUFFER;
  BEGIN(INITIAL);
  if(is_interactive) {
    std::cout << PROMPT_STRING << ' ';
    std::cout.flush();
  }
}





"{"                    { BUFFER;  return OPEN_BRACE; }
<LATEX>"\\{"           { BUFFER;  return OPEN_BRACE; }
"}"                    { BUFFER;  return CLOSE_BRACE; }
<LATEX>"\\}"           { BUFFER;  return CLOSE_BRACE; }
"approximate"          { BUFFER;  return APPROX; }
"union"                { BUFFER;  return UNION; }
<LATEX>"\\cup"         { BUFFER;  return UNION; }
"intersection"         { BUFFER;  return INTERSECTION; }
<LATEX>"\\cap"         { BUFFER;  return INTERSECTION; }
"without_simplify"     { BUFFER;  return NO_SIMPLIFY; }
"symbolic"             { BUFFER;  return SYMBOLIC; }
"sym"                  { BUFFER;  return SYMBOLIC; }
<LATEX>"\\mid"         { BUFFER;  return VERTICAL_BAR; }
<LATEX>"|"             { BUFFER;  return VERTICAL_BAR; }
<LATEX>"\\st"          { BUFFER;  return SUCH_THAT; }
"s.t."                 { BUFFER;  return SUCH_THAT; }
"inverse"              { BUFFER;  return INVERSE; }
"complement"           { BUFFER;  return COMPLEMENT; }
<LATEX>"\\circ"        { BUFFER;  return COMPOSE; }
"compose"              { BUFFER;  return COMPOSE; }
"difference"           { BUFFER;  return DIFFERENCE; }
"diffToRel"            { BUFFER;  return DIFFERENCE_TO_RELATION; }
"project away symbols" { BUFFER;  return PROJECT_AWAY_SYMBOLS; }
"project_away_symbols" { BUFFER;  return PROJECT_AWAY_SYMBOLS; }
"projectAwaySymbols"   { BUFFER;  return PROJECT_AWAY_SYMBOLS; }
"project on symbols"   { BUFFER;  return PROJECT_ON_SYMBOLS; }
"project_on_symbols"   { BUFFER;  return PROJECT_ON_SYMBOLS; }
"projectOnSymbols"     { BUFFER;  return PROJECT_ON_SYMBOLS; }
<LATEX>"\\join"        { BUFFER;  return JOIN; }
"\."                   { BUFFER;  return JOIN; }
"join"                 { BUFFER;  return JOIN; }
"domain"               { BUFFER;  return DOMAIN; }
"time"                 { BUFFER; return TIME; }
"timeclosure"          { BUFFER; return TIMECLOSURE; }
"range"                { BUFFER;  return RANGE; }
<LATEX>"\\forall"      { BUFFER;  return FORALL; }
"forall"               { BUFFER;  return FORALL; }
<LATEX>"\\exists"      { BUFFER;  return EXISTS; }
"exists"               { BUFFER;  return EXISTS; }

"Venn"                 { BUFFER; return VENN; }
"ConvexRepresentation" { BUFFER; return CONVEX_REPRESENTATION; }
"ConvexCombination"    { BUFFER; return CONVEX_COMBINATION; }
"PositiveCombination"  { BUFFER; return POSITIVE_COMBINATION; }
"LinearCombination"    { BUFFER; return LINEAR_COMBINATION; }
"AffineCombination"    { BUFFER; return AFFINE_COMBINATION; }
"RectHull"             { /*deprecated*/ BUFFER; return RECT_HULL; }
"SimpleHull"           { BUFFER; return SIMPLE_HULL; }
"ConvexHull"           { BUFFER; return CONVEX_HULL; }
"DecoupledConvexHull"  { BUFFER; return DECOUPLED_CONVEX_HULL; }
"AffineHull"           { BUFFER; return AFFINE_HULL; }
"ConicHull"            { BUFFER; return CONIC_HULL; }
"LinearHull"           { BUFFER; return LINEAR_HULL; }
"PairwiseCheck"        { /*deprecated*/ BUFFER; return PAIRWISE_CHECK; }
"ConvexCheck"          { /*deprecated*/ BUFFER; return CONVEX_CHECK; }
"QuickHull"            { /*deprecated*/ BUFFER; return QUICK_HULL; }
"Hull"                 { BUFFER; return HULL; }
"farkas"               { BUFFER;  return FARKAS; }
"decoupledfarkas"      { BUFFER;  return DECOUPLED_FARKAS; }
"decoupled-farkas"     { BUFFER;  return DECOUPLED_FARKAS; }
"decoupled_farkas"     { BUFFER;  return DECOUPLED_FARKAS; }

"minimize"             { BUFFER;  return MINIMIZE; }
"maximize"             { BUFFER;  return MAXIMIZE; }
"minimize-range"       { BUFFER;  return MINIMIZE_RANGE; }
"maximize-range"       { BUFFER;  return MAXIMIZE_RANGE; }
"minimizerange"        { BUFFER;  return MINIMIZE_RANGE; }
"maximizerange"        { BUFFER;  return MAXIMIZE_RANGE; }
"minimize-domain"      { BUFFER;  return MINIMIZE_DOMAIN; }
"maximize-domain"      { BUFFER;  return MAXIMIZE_DOMAIN; }
"minimizedomain"       { BUFFER;  return MINIMIZE_DOMAIN; }
"maximizedomain"       { BUFFER;  return MAXIMIZE_DOMAIN; }
"gist"                 { BUFFER;  return GIST; }
"given"                { BUFFER;  return GIVEN; }
"within"               { BUFFER;  return WITHIN; }
"subset"               { BUFFER;  return SUBSET; }
"codegen"              { BUFFER;  return CODEGEN; }
"upper_bound"          { BUFFER;  return MAKE_UPPER_BOUND; }
"lower_bound"          { BUFFER;  return MAKE_LOWER_BOUND; }
"supersetof"           { BUFFER;  return SUPERSETOF;}
"subsetof"             { BUFFER;  return SUBSETOF;}
"sym_example"          { BUFFER;  return SYM_SAMPLE;}
"example"              { BUFFER;  return SAMPLE;}
"carried_by"           { BUFFER;  return CARRIED_BY;}
"reachable"            { BUFFER;  return REACHABLE_FROM; }
"reachable of"         { BUFFER;  return REACHABLE_OF; }
"restrict_domain"      { BUFFER;  return RESTRICT_DOMAIN; }
"restrictDomain"       { BUFFER;  return RESTRICT_DOMAIN; }
"\\"                   { BUFFER;  return RESTRICT_DOMAIN; }
"restrict_range"       { BUFFER;  return RESTRICT_RANGE; }
"restrictRange"        { BUFFER;  return RESTRICT_RANGE; }
"assertUnsatisfiable"  { BUFFER;  return ASSERT_UNSAT; }
"assert_unsatisfiable" { BUFFER;  return ASSERT_UNSAT; }

"/"                   { BUFFER; return RESTRICT_RANGE; }
"&"                   { BUFFER; return AND; }
"|"                   { BUFFER; return OR; }
"&&"                  { BUFFER; return AND; }
"||"                  { BUFFER; return OR; }
"and"                 { BUFFER; return AND; }
"or"                  { BUFFER; return OR; }
<LATEX>"\\land"       { BUFFER; return AND; }
<LATEX>"\\lor"        { BUFFER; return OR; }
"!"                   { BUFFER; return NOT; }
"not"                 { BUFFER; return NOT; }
<LATEX>"\\neg"        { BUFFER; return NOT; }
":="                  { BUFFER; return IS_ASSIGNED; }
"->"                  { BUFFER; return GOES_TO; }
"in"                  { BUFFER; return IN; }
<LATEX>"\\rightarrow" { BUFFER; return GOES_TO; }
"<="                  { BUFFER; yylval.REL_OPERATOR = leq; return REL_OP; }
<LATEX>"\\leq"        { BUFFER; yylval.REL_OPERATOR = leq; return REL_OP; }
<LATEX>"\\le"         { BUFFER; yylval.REL_OPERATOR = leq; return REL_OP; }
">="                  { BUFFER; yylval.REL_OPERATOR = geq; return REL_OP; }
<LATEX>"\\geq"        { BUFFER; yylval.REL_OPERATOR = geq; return REL_OP; }
<LATEX>"\\ge"         { BUFFER; yylval.REL_OPERATOR = geq; return REL_OP; }
"!="                  { BUFFER; yylval.REL_OPERATOR = neq; return REL_OP; }
<LATEX>"\\neq"        { BUFFER; yylval.REL_OPERATOR = neq; return REL_OP; }
"<"                   { BUFFER; yylval.REL_OPERATOR = lt; return REL_OP; }
">"                   { BUFFER; yylval.REL_OPERATOR = gt; return REL_OP; }
"="                   { BUFFER; yylval.REL_OPERATOR = eq; return REL_OP; }
"=="                  { BUFFER; yylval.REL_OPERATOR = eq; return REL_OP; }

[A-Za-z_][A-Za-z0-9_]*[\']* {
  BUFFER;
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  return VAR;
}
[A-Za-z][A-Za-z0-9_]*"(In)" {
  BUFFER;
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-3] = 'i';  // lowercase
  yylval.VAR_NAME[yyleng-2] = 'n';
  return VAR;
}
[A-Za-z][A-Za-z0-9_]*"(Set)" {
  BUFFER;
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-4] = 'i';  // Change to "in"
  yylval.VAR_NAME[yyleng-3] = 'n';  // Be afraid
  yylval.VAR_NAME[yyleng-2] = ')';
  yylval.VAR_NAME[yyleng-1] = '\0';
  return VAR;
}
[A-Za-z][A-Za-z0-9_]*"(Out)" {
  BUFFER;
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-4] = 'o';  // lowercase
  yylval.VAR_NAME[yyleng-3] = 'u';
  yylval.VAR_NAME[yyleng-2] = 't';
  return VAR;
}
<LATEX>"\\"[A-Za-z][A-Za-z0-9_]* {
  BUFFER;  
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  return VAR;
 }
<LATEX>"\\"[A-Za-z][A-Za-z0-9_]*"(In)" {
  BUFFER;
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-3] = 'i';  // lowercase
  yylval.VAR_NAME[yyleng-2] = 'n';
  return VAR;
 }
<LATEX>"\\"[A-Za-z][A-Za-z0-9_]*"(Set)" {
  BUFFER;
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-4] = 'i';  // Change to "in"
  yylval.VAR_NAME[yyleng-3] = 'n';  // Be afraid
  yylval.VAR_NAME[yyleng-2] = ')';
  yylval.VAR_NAME[yyleng-1] = '\0';
  return VAR;
 }
<LATEX>"\\"[A-Za-z][A-Za-z0-9_]*"(Out)" {
  BUFFER;
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-4] = 'o';  // lowercase
  yylval.VAR_NAME[yyleng-3] = 'u';
  yylval.VAR_NAME[yyleng-2] = 't';
  return VAR;
 }

[0-9]+ { BUFFER;
  if (need_coef) {
    sscanf(yytext, coef_fmt, &yylval.COEF_VALUE);
    return COEF;   
  }
  else {
    yylval.INT_VALUE = atoi(yytext);
    return INT;
  }
}

\"[^\"]*\" { BUFFER;
  yytext[yyleng-1]='\0';
  yylval.STRING_VALUE = new std::string(yytext+1);
  return STRING;
}


<<EOF>> {
  mylexer.yypop_buffer_state();
  if (!YY_CURRENT_BUFFER) {
    flushScanBuffer();
    return YY_NULL;
  }
}

.        { BUFFER; return yytext[0]; }


%%

void flushScanBuffer() {
  if (scanBuf.size() == 0)
    return;
  if (!is_interactive) {
    size_t prev_pos = 0;
    if (scanBuf[0] == '\n')
      prev_pos = 1;
    for (size_t pos = prev_pos; pos <= scanBuf.size(); pos++) {
      if (pos == scanBuf.size() || scanBuf[pos] == '\n') {
        std::cout << PROMPT_STRING << " " << scanBuf.substr(prev_pos, pos-prev_pos) << std::endl;
        prev_pos = pos+1;
      }
    }
  }

  scanBuf.clear();
}
