%{
#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>
#include <omega_calc/AST.h>
#include <basic/Dynamic_Array.h>

using namespace omega;
#include "y.tab.h"

#define BUFFER scanBuf += yytext
std::string scanBuf;
std::string err_msg;
  
extern "C" int yywrap() {return 1;};


#define MAX_INCLUDE_DEPTH 10
YY_BUFFER_STATE include_stack[MAX_INCLUDE_DEPTH];
int include_stack_ptr = 0;
int comment_caller;

extern bool is_interactive;
extern const char *PROMPT_STRING;
extern bool need_coef;

// void yyerror(const char *s);
void yyerror(const std::string &s);
void flushScanBuffer();

%}

%s LATEX INCLUDE COMMENT
%option yylineno

%% 

"<<"                  { BUFFER; BEGIN(INCLUDE); }
<INCLUDE>[^>\n ]+">>" {
  BUFFER;
  scanBuf += "\n";
  flushScanBuffer();

  if (include_stack_ptr >= MAX_INCLUDE_DEPTH) {
    fprintf(stderr, "File including nested too deeply, abort");
    exit(1);
  }
  
  char *s = yytext;
  while (*s != '>') s++;
  *s = '\0';
  FILE *f = fopen(yytext, "r");
  if (!f) {
    fprintf(stderr, "Can't open file %s\n", yytext);
  }
  else {
    include_stack[include_stack_ptr++] = YY_CURRENT_BUFFER;
    yyin = f;
    yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE));
  }
  BEGIN(INITIAL);
}
<INCLUDE>[ \n] {
  fprintf(stderr,"Error in include syntax\n");
  fprintf(stderr,"Use <<fname>> to include the file named fname\n");
  BEGIN(INITIAL);
  if(is_interactive && include_stack_ptr == 0)
    printf("%s ", PROMPT_STRING);
}

<COMMENT>\n {
  BUFFER;
  BEGIN(comment_caller);
  if(is_interactive && include_stack_ptr == 0)
    printf("%s ", PROMPT_STRING);
}

\n {
  BUFFER;
  if(is_interactive && include_stack_ptr == 0)
    printf("%s ", PROMPT_STRING);
}

<LATEX>"\\ "  { BUFFER; }
[ \t]+        { BUFFER; }
#             { BUFFER; comment_caller = YY_START; BEGIN(COMMENT); }
<COMMENT>.*   { BUFFER; flushScanBuffer(); }
<LATEX>"\$\$" { BUFFER; BEGIN(INITIAL); }
"\$\$"        { BUFFER; BEGIN(LATEX); }
<LATEX>"\\t"  { BUFFER; }
<LATEX>"\\!"  { BUFFER; }
<LATEX>"\\\\" { BUFFER; }

"{"                    { BUFFER;  return OPEN_BRACE; }
<LATEX>"\\{"           { BUFFER;  return OPEN_BRACE; }
"}"                    { BUFFER;  return CLOSE_BRACE; }
<LATEX>"\\}"           { BUFFER;  return CLOSE_BRACE; }
"approximate"          { BUFFER;  return APPROX; }
"union"                { BUFFER;  return UNION; }
<LATEX>"\\cup"         { BUFFER;  return UNION; }
"intersection"         { BUFFER;  return INTERSECTION; }
<LATEX>"\\cap"         { BUFFER;  return INTERSECTION; }
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
"RectHull"             { BUFFER; return RECT_HULL; }
"ConvexHull"           { BUFFER; return CONVEX_HULL; }
"DecoupledConvexHull"  { BUFFER; return DECOUPLED_CONVEX_HULL; }
"AffineHull"           { BUFFER; return AFFINE_HULL; }
"ConicHull"            { BUFFER; return CONIC_HULL; }
"LinearHull"           { BUFFER; return LINEAR_HULL; }
"PairwiseCheck"        { /*deprecated*/ BUFFER; return PAIRWISE_CHECK; }
"ConvexCheck"          { /*deprecated*/ BUFFER; return CONVEX_CHECK; }
"QuickHull"            { /*deprecated*/ BUFFER; return QUICK_HULL; }
"hull"                 { BUFFER; return HULL; }

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
"farkas"               { BUFFER;  return FARKAS; }
"decoupledfarkas"      { BUFFER;  return DECOUPLED_FARKAS; }
"decoupled-farkas"     { BUFFER;  return DECOUPLED_FARKAS; }
"decoupled_farkas"     { BUFFER;  return DECOUPLED_FARKAS; }
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
<LATEX>"\\"            { yyerror("Can't use \\ for restrict_domain in Tex mode"); }
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

[A-Za-z][A-Za-z0-9_]*[\']* {
  BUFFER;
  if (yyleng > 19) yyerror("Identifier too long");
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  return VAR;
}
[A-Za-z][A-Za-z0-9_]*"(in)" {
  BUFFER;
  if (yyleng > 19) yyerror("Identifier too long");
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-3] = 'i';  // lowercase
  yylval.VAR_NAME[yyleng-2] = 'n';
  return VAR;
}
[A-Za-z][A-Za-z0-9_]*"(set)" {
  BUFFER;
  if (yyleng > 19) yyerror("Identifier too long");
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-4] = 'i';  // Change to "in"
  yylval.VAR_NAME[yyleng-3] = 'n';  // Be afraid
  yylval.VAR_NAME[yyleng-2] = ')';
  yylval.VAR_NAME[yyleng-1] = '\0';
  return VAR;
}
[A-Za-z][A-Za-z0-9_]*"(out)" {
  BUFFER;
  if (yyleng > 19) yyerror("Identifier too long");
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-4] = 'o';  // lowercase
  yylval.VAR_NAME[yyleng-3] = 'u';
  yylval.VAR_NAME[yyleng-2] = 't';
  return VAR;
}
<LATEX>"\\"[A-Za-z][A-Za-z0-9_]* {
  BUFFER;  
  if (yyleng > 19) yyerror("Identifier too long");
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  return VAR;
 }
<LATEX>"\\"[A-Za-z][A-Za-z0-9_]*"(in)" {
  BUFFER;
  if (yyleng > 19) yyerror("Identifier too long");
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-3] = 'i';  // lowercase
  yylval.VAR_NAME[yyleng-2] = 'n';
  return VAR;
 }
<LATEX>"\\"[A-Za-z][A-Za-z0-9_]*"(set)" {
  BUFFER;
  if (yyleng > 19) yyerror("Identifier too long");
  yylval.VAR_NAME = new char[yyleng+1];
  strcpy(yylval.VAR_NAME,yytext);
  yylval.VAR_NAME[yyleng-4] = 'i';  // Change to "in"
  yylval.VAR_NAME[yyleng-3] = 'n';  // Be afraid
  yylval.VAR_NAME[yyleng-2] = ')';
  yylval.VAR_NAME[yyleng-1] = '\0';
  return VAR;
 }
<LATEX>"\\"[A-Za-z][A-Za-z0-9_]*"(out)" {
  BUFFER;
  if (yyleng > 19) yyerror("Identifier too long");
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
  yytext[strlen(yytext)-1]='\0';
  yylval.STRING_VALUE = new std::string(yytext+1);
  return STRING;
}


<<EOF>> {
  if (--include_stack_ptr < 0) {
    flushScanBuffer();
    return yytext[0]; 
  }
  yy_delete_buffer(YY_CURRENT_BUFFER);
  yy_switch_to_buffer(include_stack[include_stack_ptr]);
}

.        { BUFFER; return yytext[0]; }


%%

void yyerror(const std::string &s) {
  std::stringstream ss;
  if (is_interactive && include_stack_ptr == 0)
    ss << s << "\n";
  else
    ss << s << " at line " << yylineno << "\n";
  err_msg = ss.str();
}


void flushScanBuffer() {
  if (scanBuf.size() == 0)
    return;
  if (!is_interactive || include_stack_ptr > 0) {
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
