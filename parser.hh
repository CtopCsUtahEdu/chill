#ifndef ONCE
#define yyFlexLexer zzFlexLexer
#include <FlexLexer.h>

//#include <FlexLexer.h>

yyFlexLexer lexer;

#else

extern yyFlexLexer lexer;

#endif


//int zzlex(){ return lexer.yylex();}
