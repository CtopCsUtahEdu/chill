/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   yacc parser for calculator.

 Notes:

 History:
   02/04/11 work with flex c++ mode, Chun Chen
*****************************************************************************/

%{
//#define YYDEBUG 1
#include <basic/Dynamic_Array.h>
#include <basic/Iterator.h>
#include <omega_calc/AST.h>
#include <omega/hull.h>
#include <omega/closure.h>
#include <omega/reach.h>
#include <string>
#include <iostream>
#include <fstream>
#include "parser.tab.hh"
#include <omega_calc/myflex.h>
//#include <stdio.h>

#if defined __USE_POSIX
#include <unistd.h>
#elif defined  __WIN32
#include <io.h>
#endif


#ifndef WIN32
#include <sys/time.h>
#include <sys/resource.h>
#endif
#if !defined(OMIT_GETRUSAGE)
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

#if !defined(OMIT_GETRUSAGE)
#ifdef __sparc__
extern "C" int getrusage (int, struct rusage*);
#endif


  

struct rusage start_time;
bool anyTimingDone = false;

void start_clock( void ) {
  getrusage(RUSAGE_SELF, &start_time);
}

int clock_diff( void ) {
  struct rusage current_time;
  getrusage(RUSAGE_SELF, &current_time);
  return (current_time.ru_utime.tv_sec -start_time.ru_utime.tv_sec)*1000000 + (current_time.ru_utime.tv_usec-start_time.ru_utime.tv_usec);
}
#endif


#ifdef BUILD_CODEGEN  
#include <codegen.h>
#endif

extern myFlexLexer mylexer;
#define yylex mylexer.yylex

  


int omega_calc_debug = 0;

extern bool is_interactive;
extern const char *PROMPT_STRING;
bool simplify = true;
using namespace omega;

extern std::string err_msg;

bool need_coef;

namespace {
  int redundant_conj_level = 2;  // default maximum 2
  int redundant_constr_level = 4;  // default maximum 4
}

std::map<std::string, Relation *> relationMap;
int argCount = 0;
int tuplePos = 0;
Argument_Tuple currentTuple = Input_Tuple;

Relation LexForward(int n);
reachable_information *reachable_info;

void yyerror(const std::string &s);
void flushScanBuffer();

%}

%union {
  int INT_VALUE;
  omega::coef_t COEF_VALUE;
  Rel_Op REL_OPERATOR;
  char *VAR_NAME;
  std::set<char *> *VAR_LIST;
  Exp *EXP;
  std::set<Exp *> *EXP_LIST;
  AST *ASTP;
  omega::Argument_Tuple ARGUMENT_TUPLE;
  AST_constraints *ASTCP;
  Declaration_Site *DECLARATION_SITE;
  omega::Relation *RELATION;
  tupleDescriptor *TUPLE_DESCRIPTOR;
  std::pair<std::vector<omega::Relation>, std::vector<omega::Relation> > *REL_TUPLE_PAIR;
  omega::Dynamic_Array1<omega::Relation> * RELATION_ARRAY_1D;
  std::string *STRING_VALUE;
}

%token <VAR_NAME> VAR 
%token <INT_VALUE> INT
%token <COEF_VALUE> COEF
%token <STRING_VALUE> STRING
%token OPEN_BRACE CLOSE_BRACE
%token SYMBOLIC NO_SIMPLIFY
%token OR AND NOT
%token ST APPROX
%token IS_ASSIGNED
%token FORALL EXISTS
%token DOMAIN RANGE
%token DIFFERENCE DIFFERENCE_TO_RELATION
%token GIST GIVEN HULL WITHIN MAXIMIZE MINIMIZE 
%token AFFINE_HULL VENN CONVEX_COMBINATION POSITIVE_COMBINATION LINEAR_COMBINATION AFFINE_COMBINATION CONVEX_HULL CONIC_HULL LINEAR_HULL QUICK_HULL PAIRWISE_CHECK CONVEX_CHECK CONVEX_REPRESENTATION RECT_HULL SIMPLE_HULL DECOUPLED_CONVEX_HULL
%token MAXIMIZE_RANGE MINIMIZE_RANGE
%token MAXIMIZE_DOMAIN MINIMIZE_DOMAIN
%token LEQ GEQ NEQ
%token GOES_TO
%token COMPOSE JOIN INVERSE COMPLEMENT IN CARRIED_BY TIME TIMECLOSURE
%token UNION INTERSECTION
%token VERTICAL_BAR SUCH_THAT
%token SUBSET CODEGEN DECOUPLED_FARKAS FARKAS
%token MAKE_UPPER_BOUND MAKE_LOWER_BOUND
%token <REL_OPERATOR> REL_OP
%token RESTRICT_DOMAIN RESTRICT_RANGE
%token SUPERSETOF SUBSETOF SAMPLE SYM_SAMPLE
%token PROJECT_AWAY_SYMBOLS PROJECT_ON_SYMBOLS REACHABLE_FROM REACHABLE_OF
%token ASSERT_UNSAT
%token PARSE_EXPRESSION PARSE_FORMULA PARSE_RELATION

%type <EXP> exp simpleExp 
%type <EXP_LIST> expList 
%type <VAR_LIST> varList
%type <ARGUMENT_TUPLE> argumentList 
%type <ASTP> formula optionalFormula
%type <ASTCP> constraintChain
%type <TUPLE_DESCRIPTOR> tupleDeclaration
%type <DECLARATION_SITE> varDecl varDeclOptBrackets
%type <RELATION> relation builtRelation context
%type <RELATION> reachable_of
%type <REL_TUPLE_PAIR> relPairList
%type <RELATION_ARRAY_1D> reachable

%destructor {delete []$$;} VAR
%destructor {delete $$;} STRING
%destructor {delete $$;} relation builtRelation tupleDeclaration formula optionalFormula context reachable_of constraintChain varDecl varDeclOptBrackets relPairList reachable
%destructor {delete $$;} exp simpleExp
%destructor {
  for (std::set<Exp *>::iterator i = $$->begin(); i != $$->end(); i++)
    delete *i;
  delete $$;
 } expList;
%destructor {
  for (std::set<char *>::iterator i = $$->begin(); i != $$->end(); i++)
    delete []*i;
  delete $$;
 } varList;

%nonassoc ASSERT_UNSAT
%left UNION p1 '+' '-'
%nonassoc  SUPERSETOF SUBSETOF
%left p2 RESTRICT_DOMAIN RESTRICT_RANGE
%left INTERSECTION p3 '*' '@' 
%left p4
%left OR p5
%left AND p6 
%left COMPOSE JOIN CARRIED_BY
%right NOT APPROX DOMAIN RANGE HULL PROJECT_AWAY_SYMBOLS PROJECT_ON_SYMBOLS DIFFERENCE DIFFERENCE_TO_RELATION INVERSE COMPLEMENT FARKAS SAMPLE SYM_SAMPLE MAKE_UPPER_BOUND MAKE_LOWER_BOUND p7
%left p8
%nonassoc GIVEN
%left p9
%left '(' p10

%%

inputSequence : /*empty*/
              | inputSequence { assert( current_Declaration_Site == globalDecls);}
                inputItem;
;

inputItem : ';' /*empty*/
          | NO_SIMPLIFY ';'{
             simplify = false; 
          } 
          | error ';' {
            flushScanBuffer();
            std::cout << err_msg;
            err_msg.clear();
            current_Declaration_Site = globalDecls;
            need_coef = false;
            std::cout << "...skipping to statement end..." << std::endl;
            delete relationDecl;
            relationDecl = NULL;
          }
          | SYMBOLIC globVarList ';' {flushScanBuffer();}
          | VAR IS_ASSIGNED relation ';' {
            flushScanBuffer();
            try {
              if(simplify)
              $3->simplify(redundant_conj_level, redundant_constr_level);
              else
              $3->simplify();
              Relation *r = relationMap[std::string($1)];
              if (r != NULL) delete r;
              relationMap[std::string($1)] = $3;
            }
             catch(const std::exception &e){
                  std::cout << e.what() << std::endl; 
             } 
           
            delete []$1;
          }
          | relation ';' {
            flushScanBuffer();
            if(simplify)
            $1->simplify(redundant_conj_level, redundant_constr_level);
            else
            $1->simplify();
            $1->print_with_subs(stdout); 
            delete $1;
          }
          | TIME relation ';' {
#if defined(OMIT_GETRUSAGE)
            printf("'time' requires getrusage, but the omega calclator was compiled with OMIT_GETRUSAGE set!\n");
#else
            flushScanBuffer();
            printf("\n");
            int t;
            Relation R;
            bool SKIP_FULL_CHECK = getenv("OC_TIMING_SKIP_FULL_CHECK");
            ($2)->and_with_GEQ();
            start_clock();
            for (t=1;t<=100;t++) {
              R = *$2;
              R.finalize();
            }
            int copyTime = clock_diff();
            start_clock();
            for (t=1;t<=100;t++) {
              R = *$2;
              R.finalize();
              R.simplify();  /* default simplification effort */
            }
            int simplifyTime = clock_diff() -copyTime;
            Relation R2;
            if (!SKIP_FULL_CHECK) {
              start_clock();
              for (t=1;t<=100;t++) {
                R2 = *$2;
                R2.finalize();
                R2.simplify(2,4); /* maximal simplification effort */
              }
            }
            int excessiveTime = clock_diff() - copyTime;
            printf("Times (in microseconds): \n");
            printf("%5d us to copy original set of constraints\n",copyTime/100);
            printf("%5d us to do the default amount of simplification, obtaining: \n\t", simplifyTime/100);
            R.print_with_subs(stdout); 
            printf("\n"); 
            if (!SKIP_FULL_CHECK) {
              printf("%5d us to do the maximum (i.e., excessive) amount of simplification, obtaining: \n\t", excessiveTime/100);
              R2.print_with_subs(stdout); 
              printf("\n");
            }
            if (!anyTimingDone) {
              bool warn = false;
#ifndef SPEED 
              warn =true;
#endif
#ifndef NDEBUG
              warn = true;
#endif
              if (warn) {
                printf("WARNING: The Omega calculator was compiled with options that force\n");
                printf("it to perform additional consistency and error checks\n");
                printf("that may slow it down substantially\n");
                printf("\n");
              }
              printf("NOTE: These times relect the time of the current _implementation_\n");
              printf("of our algorithms. Performance bugs do exist. If you intend to publish or \n");
              printf("report on the performance on the Omega test, we respectfully but strongly \n");
              printf("request that send your test cases to us to allow us to determine if the \n");
              printf("times are appropriate, and if the way you are using the Omega library to \n"); 
              printf("solve your problem is the most effective way.\n");
              printf("\n");

              printf("Also, please be aware that over the past two years, we have focused our \n");
              printf("efforts on the expressive power of the Omega library, sometimes at the\n");
              printf("expensive of raw speed. Our original implementation of the Omega test\n");
              printf("was substantially faster on the limited domain it handled.\n");
              printf("\n");
              printf("  Thanks, \n");
              printf("  the Omega Team \n");
            }
            anyTimingDone = true;
            delete $2;
#endif
          }
          | TIMECLOSURE relation ';' {
#if defined(OMIT_GETRUSAGE)
            printf("'timeclosure' requires getrusage, but the omega calclator was compiled with OMIT_GETRUSAGE set!\n");
#else
            flushScanBuffer();
            try {
              int t;
              Relation R;
              ($2)->and_with_GEQ();
              start_clock();
              for (t=1;t<=100;t++) {
                R = *$2;
                R.finalize();
              }
              int copyTime = clock_diff();
              start_clock();
              for (t=1;t<=100;t++) {
                R = *$2;
                R.finalize();
                R.simplify();
              }
              int simplifyTime = clock_diff() -copyTime;
              Relation Rclosed;
              start_clock();
              for (t=1;t<=100;t++) {
                Rclosed = *$2;
                Rclosed.finalize();
                Rclosed = TransitiveClosure(Rclosed, 1,Relation::Null());
              }
              int closureTime = clock_diff() - copyTime;
              Relation R2;
              start_clock();
              for (t=1;t<=100;t++) {
                R2 = *$2;
                R2.finalize();
                R2.simplify(2,4);
              }
              int excessiveTime = clock_diff() - copyTime;
              printf("Times (in microseconds): \n");
              printf("%5d us to copy original set of constraints\n",copyTime/100);
              printf("%5d us to do the default amount of simplification, obtaining: \n\t", simplifyTime/100);
              R.print_with_subs(stdout); 
              printf("\n"); 
              printf("%5d us to do the maximum (i.e., excessive) amount of simplification, obtaining: \n\t", excessiveTime/100);
              R2.print_with_subs(stdout); 
              printf("%5d us to do the transitive closure, obtaining: \n\t", closureTime/100);
              Rclosed.print_with_subs(stdout);
              printf("\n");
              if (!anyTimingDone) {
                bool warn = false;
#ifndef SPEED 
                warn =true;
#endif
#ifndef NDEBUG
                warn = true;
#endif
                if (warn) {
                  printf("WARNING: The Omega calculator was compiled with options that force\n");
                  printf("it to perform additional consistency and error checks\n");
                  printf("that may slow it down substantially\n");
                  printf("\n");
                }
                printf("NOTE: These times relect the time of the current _implementation_\n");
                printf("of our algorithms. Performance bugs do exist. If you intend to publish or \n");
                printf("report on the performance on the Omega test, we respectfully but strongly \n");
                printf("request that send your test cases to us to allow us to determine if the \n");
                printf("times are appropriate, and if the way you are using the Omega library to \n"); 
                printf("solve your problem is the most effective way.\n");
                printf("\n");
              
                printf("Also, please be aware that over the past two years, we have focused our \n");
                printf("efforts on the expressive power of the Omega library, sometimes at the\n");
                printf("expensive of raw speed. Our original implementation of the Omega test\n");
                printf("was substantially faster on the limited domain it handled.\n");
                printf("\n");
                printf("  Thanks, \n");
                printf("  the Omega Team \n");
              }
              anyTimingDone = true;
            }
            catch (const std::exception &e) {
              std::cout << e.what() << std::endl;
            }
            delete $2;
#endif
          }
          | relation SUBSET relation ';' {
            flushScanBuffer();
            try {
              if (Must_Be_Subset(copy(*$1), copy(*$3)))
                std::cout << "True" << std::endl;
              else if (Might_Be_Subset(copy(*$1), copy(*$3)))
                std::cout << "Possible" << std::endl;
              else
                std::cout << "False" << std::endl;
            }
            catch (const std::exception &e) {
              std::cout << e.what() << std::endl;
            }
            delete $1;
            delete $3;
          } 
          | CODEGEN relPairList context';' {
            flushScanBuffer();
#ifdef BUILD_CODEGEN
            try {
              CodeGen cg($2->first, $2->second, *$3);
              CG_result *cgr = cg.buildAST();
              if (cgr != NULL) {
                std::string s = cgr->printString();
                std::cout << s << std::endl;
                delete cgr;
              }
              else
                std::cout << "/* empty */" << std::endl;
            }
            catch (const std::exception &e) {
              std::cout << e.what() << std::endl;
            }
#else
            std::cout << "CodeGen package not built" << std::endl;
#endif
            delete $3;
            delete $2;
          }
          | CODEGEN INT relPairList context';' {
            flushScanBuffer();
#ifdef BUILD_CODEGEN
            try {
              CodeGen cg($3->first, $3->second, *$4);
              CG_result *cgr = cg.buildAST($2);
              if (cgr != NULL) {
                std::string s = cgr->printString();
                std::cout << s << std::endl;
                delete cgr;
              }
              else
                std::cout << "/* empty */" << std::endl;
            }
            catch (const std::exception &e) {
              std::cout << e.what() << std::endl;
            }
#else
            std::cout << "CodeGen package not built" << std::endl;
#endif
            delete $4;
            delete $3;
          }
          | reachable ';' {
            flushScanBuffer();
            Dynamic_Array1<Relation> &final = *$1;
            bool any_sat = false;
            int i,n_nodes = reachable_info->node_names.size();
            for(i = 1; i <= n_nodes; i++)
              if(final[i].is_upper_bound_satisfiable()) {
                any_sat = true;
                std::cout << "Node " << reachable_info->node_names[i] << ": ";
                final[i].print_with_subs(stdout);
              }
            if(!any_sat)
              std::cout << "No nodes reachable.\n";
            delete $1;
            delete reachable_info;
          }
;


context : {$$ = new Relation(); *$$ = Relation::Null();}
        | GIVEN relation {$$ = $2; }
;

relPairList : relPairList ',' relation ':' relation {
              try {
                $1->first.push_back(*$3);
                $1->second.push_back(*$5);
              }
              catch (const std::exception &e) {
                delete $1;
                delete $3;
                delete $5;
                yyerror(e.what());
                YYERROR;
              }
              delete $3;
              delete $5;
              $$ = $1;
            }
            | relPairList ',' relation {
              try {
                $1->first.push_back(Identity($3->n_set()));
                $1->second.push_back(*$3);
              }
              catch (const std::exception &e) {
                delete $1;
                delete $3;
                yyerror(e.what());
                YYERROR;
              }
              delete $3;
              $$ = $1;
            }
            | relation ':' relation {
              std::pair<std::vector<Relation>, std::vector<Relation> > *rtp = new std::pair<std::vector<Relation>, std::vector<Relation> >();
              try {
                rtp->first.push_back(*$1);
                rtp->second.push_back(*$3);
              }
              catch (const std::exception &e) {
                delete rtp;
                delete $1;
                delete $3;
                yyerror(e.what());
                YYERROR;
              }
              delete $1;
              delete $3;
              $$ = rtp;
            }
            | relation {
              std::pair<std::vector<Relation>, std::vector<Relation> > *rtp = new std::pair<std::vector<Relation>, std::vector<Relation> >();
              try {
                rtp->first.push_back(Identity($1->n_set()));
                rtp->second.push_back(*$1);
              }
              catch (const std::exception &e) {
                delete rtp;
                delete $1;
                yyerror(e.what());
                YYERROR;
              }
              delete $1;
              $$ = rtp;
            }
;

relation : OPEN_BRACE {need_coef = true; relationDecl = new Declaration_Site();}
           builtRelation CLOSE_BRACE {
           need_coef = false;
           $$ = $3; 
           current_Declaration_Site = globalDecls;
           delete relationDecl;
           relationDecl = NULL;
         }
         | VAR {
           Relation *r = relationMap[std::string($1)];
           if (r == NULL) {
             yyerror(std::string("relation ") + to_string($1) + std::string(" not declared"));
             delete []$1;
             YYERROR;
           }
           $$ = new Relation(*r);
           delete []$1;
         }
         | '(' relation ')' {$$ = $2;}
         | relation '+' %prec p9 {
           $$ = new Relation();
           try {
             *$$ = TransitiveClosure(*$1, 1, Relation::Null());
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
         }
         | relation '*' %prec p9 {
           $$ = new Relation();
           try {
             int vars = $1->n_inp();
             *$$ = Union(Identity(vars), TransitiveClosure(*$1, 1, Relation::Null()));
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             yyerror(e.what());
             YYERROR;
           }           
           delete $1;
         }
         | relation '+' WITHIN relation %prec p9 {
           $$ = new Relation();
           try {
             *$$= TransitiveClosure(*$1, 1, *$4);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $4;
             yyerror(e.what());
             YYERROR;
           }           
           delete $1;
           delete $4;
         }
        | relation '^' '@' %prec p8 {
           $$ = new Relation();
           try {
             *$$ = ApproxClosure(*$1); 
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
         }
         | relation '^' '+' %prec p8 {
           $$ = new Relation();
           try {
             *$$ = calculateTransitiveClosure(*$1);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
         }
         | MINIMIZE_RANGE relation %prec p8 {
           $$ = new Relation();
           try {
             Relation o(*$2);
             Relation r(*$2);
             r = Join(r,LexForward($2->n_out()));
             *$$ = Difference(o,r);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }           
           delete $2;
         }
         | MAXIMIZE_RANGE relation %prec p8 {
           $$ = new Relation();
           try {
             Relation o(*$2);
             Relation r(*$2);
             r = Join(r,Inverse(LexForward($2->n_out())));
             *$$ = Difference(o,r);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | MINIMIZE_DOMAIN relation %prec p8 {
           $$ = new Relation();
           try {
             Relation o(*$2);
             Relation r(*$2);
             r = Join(LexForward($2->n_inp()),r);
             *$$ = Difference(o,r);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | MAXIMIZE_DOMAIN relation %prec p8 {
           $$ = new Relation();
           try {
             Relation o(*$2);
             Relation r(*$2);
             r = Join(Inverse(LexForward($2->n_inp())),r);
             *$$ = Difference(o,r);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | MAXIMIZE relation %prec p8 {
           $$ = new Relation();
           try {
             Relation c(*$2);
             Relation r(*$2);
             *$$ = Cross_Product(Relation(*$2),c);
             *$$ = Difference(r,Domain(Intersection(*$$,LexForward($$->n_inp()))));
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | MINIMIZE relation %prec p8 {
           $$ = new Relation();
           try {
             Relation c(*$2);
             Relation r(*$2);
             *$$ = Cross_Product(Relation(*$2),c);
             *$$ = Difference(r,Range(Intersection(*$$,LexForward($$->n_inp()))));
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;       
         }
         | FARKAS relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Farkas(*$2, Basic_Farkas);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | DECOUPLED_FARKAS relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Farkas(*$2, Decoupled_Farkas);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | relation '@' %prec p9 {
           $$ = new Relation();
           try {
             *$$ = ConicClosure(*$1);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             yyerror(e.what());
             YYERROR;
           }             
           delete $1;
         }
         | PROJECT_AWAY_SYMBOLS relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Project_Sym(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | PROJECT_ON_SYMBOLS relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Project_On_Sym(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | DIFFERENCE relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Deltas(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | DIFFERENCE_TO_RELATION relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = DeltasToRelation(*$2,$2->n_set(),$2->n_set());
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | DOMAIN relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Domain(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | VENN relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = VennDiagramForm(*$2,Relation::True(*$2));
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | VENN relation GIVEN  relation  %prec p8 {
           $$ = new Relation();
           try {
             *$$ = VennDiagramForm(*$2,*$4);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             delete $4;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
           delete $4;
         }
         | CONVEX_HULL relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = ConvexHull(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | DECOUPLED_CONVEX_HULL relation  %prec p8 {
           $$ = new Relation();
           try {
             *$$ = DecoupledConvexHull(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | POSITIVE_COMBINATION relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Farkas(*$2,Positive_Combination_Farkas);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | LINEAR_COMBINATION relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Farkas(*$2,Linear_Combination_Farkas);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | AFFINE_COMBINATION relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Farkas(*$2,Affine_Combination_Farkas);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | CONVEX_COMBINATION relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Farkas(*$2,Convex_Combination_Farkas);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }           
           delete $2;
         }
         | PAIRWISE_CHECK relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = CheckForConvexRepresentation(CheckForConvexPairs(*$2));
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | CONVEX_CHECK relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = CheckForConvexRepresentation(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | CONVEX_REPRESENTATION relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = ConvexRepresentation(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | AFFINE_HULL relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = AffineHull(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | CONIC_HULL relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = ConicHull(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | LINEAR_HULL relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = LinearHull(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | QUICK_HULL relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = QuickHull(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | RECT_HULL relation %prec p8 {
           $$ = new Relation();
           try {
             *$$ = RectHull(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | SIMPLE_HULL relation %prec p8 {
           $$ = new Relation();
           try {
             *$$ = SimpleHull(*$2, true, true);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | HULL relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Hull(*$2,true,1,Relation::Null());
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | HULL relation GIVEN relation  %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Hull(*$2,true,1,*$4);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             delete $4;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
           delete $4;
         }
         | APPROX relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Approximate(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | RANGE relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Range(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | INVERSE relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Inverse(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | COMPLEMENT relation   %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Complement(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | GIST relation GIVEN relation %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Gist(*$2,*$4,1);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             delete $4;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
           delete $4;
         }
         | relation '(' relation ')' {
           $$ = new Relation();
           try {
             *$$ = Composition(*$1,*$3);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $3;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
           delete $3;
         }
         | relation COMPOSE relation {
           $$ = new Relation();
           try {
             *$$ = Composition(*$1,*$3);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $3;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
           delete $3;
         }
         | relation CARRIED_BY INT {
           $$ = new Relation();
           try {
             *$$ = After(*$1,$3,$3);
             (*$$).prefix_print(stdout);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
         }
         | relation JOIN relation {
           $$ = new Relation();
           try {
             *$$ = Composition(*$3,*$1);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $3;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
           delete $3;
         }
         | relation RESTRICT_RANGE relation {
           $$ = new Relation();
           try {
             *$$ = Restrict_Range(*$1,*$3);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $3;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
           delete $3;
         }
         | relation RESTRICT_DOMAIN relation {
           $$ = new Relation();
           try {
             *$$ = Restrict_Domain(*$1,*$3);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $3;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
           delete $3;
         }
         | relation INTERSECTION relation {
           $$ = new Relation();
           try {
             *$$ = Intersection(*$1,*$3);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $3;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
           delete $3;
         }
         | relation '-' relation %prec INTERSECTION {
           $$ = new Relation();
           try {
             *$$ = Difference(*$1,*$3);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $3;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
           delete $3;
         }
         | relation UNION relation {
           $$ = new Relation();
           try {
             *$$ = Union(*$1,*$3);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $3;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
           delete $3;
         }
         | relation '*' relation {
           $$ = new Relation();
           try {
             *$$ = Cross_Product(*$1,*$3);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $1;
             delete $3;
             yyerror(e.what());
             YYERROR;
           }
           delete $1;
           delete $3;
         }
         | SUPERSETOF relation {
           $$ = new Relation();
           try {
             *$$ = Union(*$2, Relation::Unknown(*$2));
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         } 
         | SUBSETOF relation {
           $$ = new Relation();
           try {
             *$$ = Intersection(*$2, Relation::Unknown(*$2));
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | MAKE_UPPER_BOUND relation %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Upper_Bound(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         } 
         | MAKE_LOWER_BOUND relation %prec p8 {
           $$ = new Relation();
           try {
             *$$ = Lower_Bound(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | SAMPLE relation {
           $$ = new Relation();
           try {
             *$$ = Sample_Solution(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | SYM_SAMPLE relation {
           $$ = new Relation();
           try {
             *$$ = Symbolic_Solution(*$2);
           }
           catch (const std::exception &e) {
             delete $$;
             delete $2;
             yyerror(e.what());
             YYERROR;
           }
           delete $2;
         }
         | reachable_of { $$ = $1; }
         | ASSERT_UNSAT relation {
           if (($2)->is_satisfiable()) {
             fprintf(stderr,"assert_unsatisfiable failed on ");
             ($2)->print_with_subs(stderr);
             exit(1);
           }
           $$=$2;
         }
;

builtRelation :  tupleDeclaration GOES_TO {currentTuple = Output_Tuple;} 
                 tupleDeclaration {currentTuple = Input_Tuple;} optionalFormula {
                 Relation * r = new Relation($1->size,$4->size);
                 resetGlobals();
                 F_And *f = r->add_and();
                 for(int i = 1; i <= $1->size; i++) {
                   $1->vars[i-1]->vid = r->input_var(i);
                   if (!$1->vars[i-1]->anonymous) 
                     r->name_input_var(i, $1->vars[i-1]->stripped_name);
                 }
                 for(int i = 1; i <= $4->size; i++) {
                   $4->vars[i-1]->vid = r->output_var(i);
                   if (!$4->vars[i-1]->anonymous) 
                     r->name_output_var(i, $4->vars[i-1]->stripped_name);
                 }
                 r->setup_names();
                 for (std::set<Exp *>::iterator i = $1->eq_constraints.begin(); i != $1->eq_constraints.end(); i++)
                   install_eq(f, *i, 0);
                 for (std::set<Exp *>::iterator i = $1->geq_constraints.begin(); i != $1->geq_constraints.end(); i++)
                   install_geq(f, *i, 0);
                 for (std::set<strideConstraint *>::iterator i = $1->stride_constraints.begin(); i != $1->stride_constraints.end(); i++)
                   install_stride(f, *i);
                 for (std::set<Exp *>::iterator i = $4->eq_constraints.begin(); i != $4->eq_constraints.end(); i++)
                   install_eq(f, *i, 0);
                 for (std::set<Exp *>::iterator i = $4->geq_constraints.begin(); i != $4->geq_constraints.end(); i++)
                   install_geq(f, *i, 0);
                 for (std::set<strideConstraint *>::iterator i = $4->stride_constraints.begin(); i != $4->stride_constraints.end(); i++)
                   install_stride(f, *i);
                 if ($6) $6->install(f);
                 delete $1;
                 delete $4;
                 delete $6;
                 $$ = r;
               }
               | tupleDeclaration optionalFormula {
                 Relation * r = new Relation($1->size);
                 resetGlobals();
                 F_And *f = r->add_and();
                 for(int i = 1; i <= $1->size; i++) {
                   $1->vars[i-1]->vid = r->set_var(i);
                   if (!$1->vars[i-1]->anonymous) 
                     r->name_set_var(i, $1->vars[i-1]->stripped_name);
                 }
                 r->setup_names();
                 for (std::set<Exp *>::iterator i = $1->eq_constraints.begin(); i != $1->eq_constraints.end(); i++)
                   install_eq(f, *i, 0);
                 for (std::set<Exp *>::iterator i = $1->geq_constraints.begin(); i != $1->geq_constraints.end(); i++)
                   install_geq(f, *i, 0);
                 for (std::set<strideConstraint *>::iterator i = $1->stride_constraints.begin(); i != $1->stride_constraints.end(); i++)
                   install_stride(f, *i);
                 if ($2) $2->install(f);
                 delete $1;
                 delete $2;
                 $$ = r;
               }
               | formula {
                 Relation * r = new Relation(0,0);
                 F_And *f = r->add_and();
                 $1->install(f);
                 delete $1;
                 $$ = r;
               }
;

optionalFormula : formula_sep formula {$$ = $2;}
                | {$$ = 0;}  
;
 
formula_sep : ':'
            | VERTICAL_BAR
            | SUCH_THAT
;

tupleDeclaration : {currentTupleDescriptor = new tupleDescriptor; tuplePos = 1;}
                   '[' optionalTupleVarList ']'
                   {$$ = currentTupleDescriptor; tuplePos = 0;}
;

optionalTupleVarList : /* empty */
                     | tupleVar 
                     | optionalTupleVarList ',' tupleVar 
;

tupleVar : VAR %prec p10 {
           Declaration_Site *ds = defined($1);
           if (!ds)
             currentTupleDescriptor->extend($1,currentTuple,tuplePos);
           else {
             Variable_Ref *v = lookupScalar($1);
             if (v == NULL) {
               yyerror(std::string("cannot find declaration for variable ") + to_string($1));
               delete []$1;
               YYERROR;
             }
             if (ds != globalDecls)
               currentTupleDescriptor->extend($1, new Exp(v));
             else
               currentTupleDescriptor->extend(new Exp(v));
           }
           tuplePos++;
           delete []$1;
         }
         | '*' {currentTupleDescriptor->extend(); tuplePos++;}
         | exp %prec p1 {
             currentTupleDescriptor->extend($1);
             tuplePos++;
         }
         | exp ':' exp %prec p1 {
             currentTupleDescriptor->extend($1,$3);
             tuplePos++;
         }
         | exp ':' exp ':' COEF %prec p1 {
             currentTupleDescriptor->extend($1,$3,$5);
             tuplePos++;
         }
;

varList : varList ',' VAR {$$ = $1; $$->insert($3); $3 = NULL;}
         | VAR {$$ = new std::set<char *>(); $$->insert($1); $1 = NULL;}
;

varDecl : varList {
          $$ = current_Declaration_Site = new Declaration_Site($1);
          for (std::set<char *>::iterator i = $1->begin(); i != $1->end(); i++)
            delete [](*i);
          delete $1;
        }
;

varDeclOptBrackets : varDecl {$$ = $1;}
                   |'[' varDecl ']' {$$ = $2;}
;

globVarList : globVarList ',' globVar
            | globVar
;

globVar : VAR '(' INT ')' {globalDecls->extend_both_tuples($1, $3); delete []$1;}
        | VAR {
          globalDecls->extend($1);
          delete []$1;
        }
;

formula : formula AND formula {$$ = new AST_And($1,$3);}
        | formula OR formula {$$ = new AST_Or($1,$3);}
        | constraintChain {$$ = $1;}
        | '(' formula ')' {$$ = $2;}
        | NOT formula {$$ = new AST_Not($2);}
        | start_exists varDeclOptBrackets exists_sep formula end_quant {$$ = new AST_exists($2,$4);}
        | start_forall varDeclOptBrackets forall_sep formula end_quant {$$ = new AST_forall($2,$4);}
;

start_exists : '(' EXISTS
             | EXISTS '('
;

exists_sep : ':'
           | VERTICAL_BAR
           | SUCH_THAT
;

start_forall : '(' FORALL
             | FORALL '('
;

forall_sep : ':'
;

end_quant : ')' {popScope();}
;

expList : exp ',' expList {$$ = $3; $$->insert($1);}
        | exp {$$ = new std::set<Exp *>(); $$->insert($1);}
;

constraintChain : expList REL_OP expList {$$ = new AST_constraints($1,$2,$3);}
                | expList REL_OP constraintChain {$$ = new AST_constraints($1,$2,$3);}
;

simpleExp : VAR %prec p9 {
            Variable_Ref *v = lookupScalar($1);
            if (v == NULL) {
              yyerror(std::string("cannot find declaration for variable ") + to_string($1));
              delete []$1;
              YYERROR;
            }
            $$ = new Exp(v);
            delete []$1;
          }
          | VAR '(' {argCount = 1;}  argumentList ')' %prec p9 {
            Variable_Ref *v;
            if ($4 == Input_Tuple)
              v = functionOfInput[$1];
            else
              v = functionOfOutput[$1];
            if (v == NULL) {
              yyerror(std::string("Function ") + to_string($1) + std::string(" not declared"));
              delete []$1;
              YYERROR;
            }
            $$ = new Exp(v);
            delete []$1;
          }
          | '(' exp ')'  { $$ = $2; }
;

argumentList : argumentList ',' VAR {
               Variable_Ref *v = lookupScalar($3);
               if (v == NULL) {
                 yyerror(std::string("cannot find declaration for variable ") + to_string($1));
                 delete []$3;
                 YYERROR;
               }
               if (v->pos != argCount || v->of != $1 || (v->of != Input_Tuple && v->of != Output_Tuple)) {
                 yyerror("arguments to function must be prefix of input or output tuple");
                 delete []$3;
                 YYERROR;
               }
               $$ = v->of;
               argCount++;
               delete []$3;
             }
             | VAR {
               Variable_Ref *v = lookupScalar($1);
               if (v == NULL) {
                 yyerror(std::string("cannot find declaration for variable ") + to_string($1));
                 delete []$1;
                 YYERROR;
               }
               if (v->pos != argCount || (v->of != Input_Tuple && v->of != Output_Tuple)) {
                 yyerror("arguments to function must be prefix of input or output tuple");
                 delete []$1;
                 YYERROR;
               }
               $$ = v->of;
               argCount++;
               delete []$1;
             }
;

exp : COEF {$$ = new Exp($1);}
    | COEF simpleExp  %prec '*' {$$ = multiply($1,$2);}
    | simpleExp {$$ = $1; }
    | '-' exp %prec '*' {$$ = negate($2);}
    | exp '+' exp {$$ = add($1,$3);}
    | exp '-' exp {$$ = subtract($1,$3);}
    | exp '*' exp {
      try {
        $$ = multiply($1,$3);
      }
      catch (const std::exception &e) {
        yyerror(e.what());
        YYERROR;
      }
    }
;


reachable : REACHABLE_FROM nodeNameList nodeSpecificationList {
            Dynamic_Array1<Relation> *final = Reachable_Nodes(reachable_info);
            $$ = final;
          }
;

reachable_of : REACHABLE_OF VAR IN nodeNameList nodeSpecificationList {
               Dynamic_Array1<Relation> *final = Reachable_Nodes(reachable_info);
               int index = reachable_info->node_names.index(std::string($2));
               if (index == 0) {
                 yyerror(std::string("no such node ") + to_string($2));
                 delete []$2;
                 delete final;
                 delete reachable_info;
                 YYERROR;
               }
               $$ = new Relation; 
               *$$ = (*final)[index];
               delete final;
               delete reachable_info;
               delete []$2;
             }
;

nodeNameList : '(' realNodeNameList ')' {
               int sz = reachable_info->node_names.size();
               reachable_info->node_arity.reallocate(sz);
               reachable_info->transitions.resize(sz+1,sz+1);
               reachable_info->start_nodes.resize(sz+1);
             }
;

realNodeNameList : realNodeNameList ',' VAR {
                   reachable_info->node_names.append(std::string($3));
                   delete []$3;
                 }
                 | VAR {
                   reachable_info = new reachable_information;
                   reachable_info->node_names.append(std::string($1));
                   delete []$1;
                 }
;


nodeSpecificationList : OPEN_BRACE realNodeSpecificationList CLOSE_BRACE {  
                        int i,j;
                        int n_nodes = reachable_info->node_names.size();
                        Tuple<int> &arity = reachable_info->node_arity;
                        Dynamic_Array2<Relation> &transitions = reachable_info->transitions;

                        /* fixup unspecified transitions to be false */
                        /* find arity */
                        for(i = 1; i <= n_nodes; i++) arity[i] = -1;
                        for(i = 1; i <= n_nodes; i++)
                          for(j = 1; j <= n_nodes; j++)
                            if(! transitions[i][j].is_null()) {
                              int in_arity = transitions[i][j].n_inp();
                              int out_arity = transitions[i][j].n_out();
                              if(arity[i] < 0) arity[i] = in_arity;
                              if(arity[j] < 0) arity[j] = out_arity;
                              if(in_arity != arity[i] || out_arity != arity[j]) {
                                yyerror(std::string("arity mismatch in node transition: ") + to_string(reachable_info->node_names[i]) + std::string(" -> ") + to_string(reachable_info->node_names[j]));
                                delete reachable_info;
                                YYERROR;
                              }
                            }
                        for(i = 1; i <= n_nodes; i++) 
                          if(arity[i] < 0) arity[i] = 0;
                        /* Fill in false relations */
                        for(i = 1; i <= n_nodes; i++)
                          for(j = 1; j <= n_nodes; j++)
                            if(transitions[i][j].is_null())
                              transitions[i][j] = Relation::False(arity[i],arity[j]);

                        /* fixup unused start node positions */
                        Dynamic_Array1<Relation> &nodes = reachable_info->start_nodes;
                        for(i = 1; i <= n_nodes; i++) 
                          if(nodes[i].is_null()) 
                            nodes[i] = Relation::False(arity[i]);
                          else
                            if(nodes[i].n_set() != arity[i]){
                              yyerror(std::string("arity mismatch in start node ") + to_string(reachable_info->node_names[i]));
                              delete reachable_info;
                              YYERROR;
                            }
                   }
;

realNodeSpecificationList : realNodeSpecificationList ',' VAR ':' relation {
                            int n_nodes = reachable_info->node_names.size();
                            int index = reachable_info->node_names.index($3);
                            if (!(index > 0 && index <= n_nodes)) {
                              yyerror(std::string("no such node ")+to_string($3));
                              delete $5;
                              delete []$3;
                              delete reachable_info;
                              YYERROR;
                            }
                            reachable_info->start_nodes[index] = *$5;
                            delete $5;
                            delete []$3;
                          }
                          | realNodeSpecificationList ',' VAR GOES_TO VAR ':' relation {
                            int n_nodes = reachable_info->node_names.size();
                            int from_index = reachable_info->node_names.index($3);
                            if (!(from_index > 0 && from_index <= n_nodes)) {
                              yyerror(std::string("no such node ")+to_string($3));
                              delete $7;
                              delete []$3;
                              delete []$5;
                              delete reachable_info;
                              YYERROR;
                            }
                            int to_index = reachable_info->node_names.index($5);
                            if (!(to_index > 0 && to_index <= n_nodes)) {
                              yyerror(std::string("no such node ")+to_string($5));
                              delete $7;
                              delete []$3;
                              delete []$5;
                              delete reachable_info;
                              YYERROR;
                            }
                            reachable_info->transitions[from_index][to_index] = *$7;
                            delete $7;
                            delete []$3;
                            delete []$5;
                          }
                          | VAR GOES_TO VAR ':' relation {
                            int n_nodes = reachable_info->node_names.size();
                            int from_index = reachable_info->node_names.index($1);
                            if (!(from_index > 0 && from_index <= n_nodes)) {
                              yyerror(std::string("no such node ")+to_string($1));
                              delete $5;
                              delete []$1;
                              delete []$3;
                              delete reachable_info;
                              YYERROR;
                            }
                            int to_index = reachable_info->node_names.index($3);
                            if (!(to_index > 0 && to_index <= n_nodes)) {
                              yyerror(std::string("no such node ")+to_string($3));
                              delete $5;
                              delete []$1;
                              delete []$3;
                              delete reachable_info;
                              YYERROR;
                            }
                            reachable_info->transitions[from_index][to_index] = *$5;
                            delete $5;
                            delete []$1;
                            delete []$3;
                          }
                          | VAR ':' relation {
                            int n_nodes = reachable_info->node_names.size();
                            int index = reachable_info->node_names.index($1);
                            if (!(index > 0 && index <= n_nodes)) {
                              yyerror(std::string("no such node ")+to_string($1));
                              delete $3;
                              delete []$1;
                              delete reachable_info;
                              YYERROR;
                            }
                            reachable_info->start_nodes[index] = *$3;
                            delete $3;
                            delete []$1;
                          }
;

%%

void yyerror(const std::string &s) {
  std::stringstream ss;
  if (is_interactive)
    ss << s << "\n";
  else
    ss << s << " at line " << mylexer.lineno() << "\n";
  err_msg = ss.str();
}


int main(int argc, char **argv) {
  if (argc > 2){
    fprintf(stderr, "Usage: %s [script_file]\n", argv[0]);
    exit(1);
  }

  if (argc == 2) {
    std::ifstream *ifs = new std::ifstream;
    ifs->open(argv[1], std::ifstream::in);
    if (!ifs->is_open()) {
        fprintf(stderr, "can't open input file %s\n", argv[1]);
        exit(1);
    }
    yy_buffer_state *bs = mylexer.yy_create_buffer(ifs, 8092);
    mylexer.yypush_buffer_state(bs);
  }

  //yydebug = 1;
  is_interactive = false;
  if (argc == 1) {
#if defined __USE_POSIX  
    if (isatty((int)fileno(stdin)))
      is_interactive = true;
#elif defined  __WIN32
    if (_isatty(_fileno(stdin)))
      is_interactive = true;
#endif
  }

  if (is_interactive) {
#ifdef BUILD_CODEGEN
    std::cout << "Omega+ and CodeGen+ ";
#else
    std::cout << "Omega+ ";
#endif
    std::cout << "v2.2.3 (built on " OC_BUILD_DATE ")" << std::endl;
    std::cout << "Copyright (C) 1994-2000 the Omega Project Team" << std::endl;
    std::cout << "Copyright (C) 2005-2011 Chun Chen" << std::endl;
    std::cout << "Copyright (C) 2011-2012 University of Utah" << std::endl;
    std::cout << PROMPT_STRING << ' ';
    std::cout.flush();
  }

  need_coef = false;  
  current_Declaration_Site = globalDecls = new Global_Declaration_Site();

  if (yyparse() != 0) {
    if (!is_interactive)
      std::cout << "syntax error at the end of the file, missing ';'" << std::endl;
    else
      std::cout << std::endl;
    delete relationDecl;
    relationDecl = NULL;
  }
  else {
    if (is_interactive)
      std::cout << std::endl;
  }

  for (std::map<std::string, Relation *>::iterator i = relationMap.begin(); i != relationMap.end(); i++)
    delete (*i).second;
  delete globalDecls;  
  
  return 0;
}

Relation LexForward(int n) {
  Relation r(n,n);
  F_Or *f = r.add_or();
  for (int i=1; i <= n; i++) {
    F_And *g = f->add_and();
    for(int j=1;j<i;j++) {
      EQ_Handle e = g->add_EQ();
      e.update_coef(r.input_var(j),-1);
      e.update_coef(r.output_var(j),1);
      e.finalize();
    }
    GEQ_Handle e = g->add_GEQ();
    e.update_coef(r.input_var(i),-1);
    e.update_coef(r.output_var(i),1);
    e.update_const(-1);
    e.finalize();
  }
  r.finalize();
  return r;
}
