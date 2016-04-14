/*****************************************************************************
 Copyright (C) 2008 University of Southern California.
 Copyright (C) 2009-2010 University of Utah.
 All Rights Reserved.

 Purpose:
   CHiLL script yacc parser

 Notes:

 History:
   01/2008 created by Chun Chen
*****************************************************************************/

%{
/* Substitute the variable and function names.  */
#define yyparse         zzparse
#define yylex           zzlex
#define yyerror         zzerror
#define yylval          zzlval
#define yychar          zzchar
#define yydebug         zzdebug
#define yynerrs         zznerrs


#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include "parser.hh"


#include "parser.tab.hh"

#include <omega.h>
#include "ir_code.hh"
#include "loop.hh"

// should these be mutually exclusive?
#ifdef FRONTEND_ROSE
#include "ir_rose.hh"
#endif

#ifdef FRONTEND_CLANG
#include "ir_clang.hh"
#endif 





using namespace omega;
  
extern int yydebug;

void yyerror(const char *);
int yylex();  
extern int yylex();   // ?? 
// ?? yyFlexLexer lexer;

namespace {
  enum COMPILER_FRONT_END {FE_NULL, FE_SUIF, FE_ROSE, FE_CLANG, FE_GNU}; // a historical summary. SUIF is NO LONGER VALID
  COMPILER_FRONT_END frontend_compiler = FE_NULL;

  char *source_filename = NULL;
  char *dest_filename = NULL;

  #if defined(FRONTEND_ROSE) || defined(FRONTEND_CLANG) 
  char* procedure_name = NULL; // used by Rose and Clang
  #endif

  int loop_num_start, loop_num_end;
  Loop *myloop = NULL;          // the one loop we're currently modifying
}

#define PRINT_ERR_LINENO {if (is_interactive) fprintf(stderr, "\n"); else fprintf(stderr, " at line %d\n", lexer.lineno()-1);}

std::map<std::string, int> parameter_tab;
bool is_interactive;
const char *PROMPT_STRING = ">>>";

IR_Code *ir_code = NULL;
std::vector<IR_Control *> ir_controls;
std::vector<int> loops;
%}

%union {
  int val;
  float fval;
  bool bool_val;
  char *name;
  std::vector<int> *vec;
  std::vector<std::string> *string_vec;
  std::vector<std::vector<int> > *mat;
  std::map<std::string, int> *tab;
  std::vector<std::map<std::string, int> > *tab_lst;
  std::pair<std::vector<std::map<std::string, int> >, std::map<std::string, int> > *eq_term_pair;
}

%token <val> NUMBER LEVEL
%token <bool_val> TRUEORFALSE
%token <name> FILENAME PROCEDURENAME VARIABLE FREEVAR STRING
%token SOURCE DEST PROCEDURE FORMAT LOOP PERMUTE ORIGINAL TILE UNROLL 
%token FIND_STENCIL_SHAPE STENCIL_TEMP
%token SPLIT UNROLL_EXTRA REDUCE SPLIT_WITH_ALIGNMENT
%token PRAGMA PREFETCH 
%token DATACOPY DATACOPY_PRIVATIZED
%token FLATTEN SCALAR_EXPAND NORMALIZE ELLIFY COMPACT MAKE_DENSE SET_ARRAY_SIZE
%token NONSINGULAR EXIT KNOWN SKEW SHIFT SHIFT_TO 
%token FUSE DISTRIBUTE REMOVE_DEP SCALE REVERSE PEEL
%token STRIDED COUNTED NUM_STATEMENT CEIL FLOOR
%token PRINT PRINT_CODE PRINT_DEP PRINT_IS PRINT_STRUCTURE ADD_GHOSTS
%token NE LE GE EQ

%type <vec> vector vector_number
%type <string_vec> vector_string string_vector
/* TODO: %type <eq_term_pair> cond_term cond */
%type <tab> cond_term
%type <tab_lst> cond
%type <mat> matrix matrix_part
%type <val> expr
%type <fval> float_expr

%destructor {delete []$$; } FILENAME VARIABLE FREEVAR
%destructor {delete $$; } vector vector_number cond_term cond matrix matrix_part

%left '>' '<' NE LE GE
%left '+' '-'
%left '*' '/'
%left '%'
%left UMINUS


%%
script : /* empty */
       | script command
;


vector : '[' vector_number ']' {$$ = $2;}
;

vector_number : {$$ = new std::vector<int>();}
              | expr {$$ = new std::vector<int>(); $$->push_back($1);}
              | vector_number ',' expr {$$ = $1; $$->push_back($3);}
;

vector_string : '[' string_vector ']' {$$ = $2;}
;

string_vector : {$$ = new std::vector<std::string>();}
              | VARIABLE {$$ = new std::vector<std::string>(); $$->push_back($1);}
              | string_vector ',' VARIABLE {$$ = $1; $$->push_back($3);}
;

matrix: '[' matrix_part ']' {$$ = $2;}

matrix_part : vector {$$ = new std::vector<std::vector<int> >(); $$->push_back(*$1); delete $1;}
            | matrix_part ',' vector {$$ = $1; $$->push_back(*$3); delete $3;}

expr : NUMBER {$$ = $1;}
     | VARIABLE {
       std::map<std::string, int>::iterator it = parameter_tab.find(std::string($1));
       if (it != parameter_tab.end()) {
         $$ = it->second;
         delete []$1;
       }
       else {
         if (is_interactive)
           fprintf(stderr, "variable \"%s\" undefined\n", $1);
         else
           fprintf(stderr, "variable \"%s\" undefined at line %d\n", $1, lexer.lineno());
         delete []$1;
         if (!is_interactive)
           exit(2);
       }
     }
     | NUM_STATEMENT '(' ')' {
       if (myloop == NULL)
         $$ = 0;
       else
         $$ = myloop->num_statement();
     }
     | CEIL '(' float_expr ')' {
       $$ = ceil($3);
     }
     | FLOOR '(' float_expr ')' {
       $$ = floor($3);
     }     
     | '(' expr ')' {$$ = $2;}
     | expr '-' expr {$$ = $1-$3;}
     | expr '+' expr {$$ = $1+$3;}
     | expr '*' expr {$$ = $1*$3;}
     | expr '/' expr {$$ = $1/$3;}
     | '-' expr %prec UMINUS {$$ = -$2;}
;

float_expr : NUMBER {$$ = $1;}
           | VARIABLE {
             std::map<std::string, int>::iterator it = parameter_tab.find(std::string($1));
             if (it != parameter_tab.end()) {
               $$ = it->second;
               delete []$1;
             }
             else {
               if (is_interactive)
                 fprintf(stderr, "variable \"%s\" undefined\n", $1);
               else
                 fprintf(stderr, "variable \"%s\" undefined at line %d\n", $1, lexer.lineno());
               delete []$1;
               if (!is_interactive)
                 exit(2);
             }
           }
           | NUM_STATEMENT '(' ')' {
             if (myloop == NULL)
               $$ = 0;
             else
               $$ = myloop->num_statement();
           }
           | CEIL '(' float_expr ')' {
             $$ = ceil($3);
           }
           | FLOOR '(' float_expr ')' {
             $$ = floor($3);
           }     
           | '(' float_expr ')' {$$ = $2;}
           | float_expr '-' float_expr {$$ = $1-$3;}
           | float_expr '+' float_expr {$$ = $1+$3;}
           | float_expr '*' float_expr {$$ = $1*$3;}
           | float_expr '/' float_expr {$$ = $1/$3;}
           | '-' float_expr %prec UMINUS {$$ = -$2;}
;


cond : cond_term GE cond_term {
       for (std::map<std::string, int>::iterator it = $3->begin(); it != $3->end(); it++)
         (*$1)[it->first] -= it->second;
       $$ = new std::vector<std::map<std::string, int> >();
       $$->push_back(*$1);
       delete $1;
       delete $3;
     }
     | cond_term '>' cond_term {
       for (std::map<std::string, int>::iterator it = $3->begin(); it != $3->end(); it++)
         (*$1)[it->first] -= it->second;
       $$ = new std::vector<std::map<std::string, int> >();
       (*$1)[to_string(0)] -= 1;
       $$->push_back(*$1);
       delete $1;
       delete $3;
     }
     | cond_term LE cond_term {
       for (std::map<std::string, int>::iterator it = $1->begin(); it != $1->end(); it++)
         (*$3)[it->first] -= it->second;
       $$ = new std::vector<std::map<std::string, int> >();
       $$->push_back(*$3);
       delete $1;
       delete $3;
     }
     | cond_term '<' cond_term {
       for (std::map<std::string, int>::iterator it = $1->begin(); it != $1->end(); it++)
         (*$3)[it->first] -= it->second;
       $$ = new std::vector<std::map<std::string, int> >();
       (*$3)[to_string(0)] -= 1;
       $$->push_back(*$3);
       delete $1;
       delete $3;
     }
     | cond_term EQ cond_term {
       for (std::map<std::string, int>::iterator it = $3->begin(); it != $3->end(); it++)
         (*$1)[it->first] -= it->second;
       $$ = new std::vector<std::map<std::string, int> >();
       $$->push_back(*$1);
       for (std::map<std::string, int>::iterator it = $1->begin(); it != $1->end(); it++)
         it->second = -it->second;
       $$->push_back(*$1);
       delete $1;
       delete $3;
     }
;

cond_term : NUMBER {$$ = new std::map<std::string, int>(); (*$$)[to_string(0)] = $1;}
          | LEVEL {$$ = new std::map<std::string, int>(); (*$$)[to_string($1)] = 1;}
          | VARIABLE {
            $$ = new std::map<std::string, int>();

            std::map<std::string, int>::iterator it = parameter_tab.find(std::string($1));
            if (it != parameter_tab.end())
              (*$$)[to_string(0)] = it->second;
            else
             (*$$)[std::string($1)] = 1;

            delete []$1;
          }
          | '(' cond_term ')' {$$ = $2;}
          | cond_term '-' cond_term {
            for (std::map<std::string, int>::iterator it = $3->begin(); it != $3->end(); it++)
              (*$1)[it->first] -= it->second;
            $$ = $1;
            delete $3;
          }
          | cond_term '+' cond_term {
            for (std::map<std::string, int>::iterator it = $3->begin(); it != $3->end(); it++)
              (*$1)[it->first] += it->second;
            $$ = $1;
            delete $3;
          }
          | cond_term '*' cond_term {
            (*$1)[to_string(0)] += 0;
            (*$3)[to_string(0)] += 0;
            if ($1->size() == 1) {
              int t = (*$1)[to_string(0)];
              for (std::map<std::string, int>::iterator it = $3->begin(); it != $3->end(); it++)
                it->second *= t;
              $$ = $3;
              delete $1;
            }
            else if ($3->size() == 1) {
              int t = (*$3)[to_string(0)];
              for (std::map<std::string, int>::iterator it = $1->begin(); it != $1->end(); it++)
                it->second *= t;
              $$ = $1;
              delete $3;
            }
            else {
              if (is_interactive)
                fprintf(stderr, "require Presburger formula\n");
              else
                fprintf(stderr, "require Presburger formula at line %d\n", lexer.lineno());
              delete $1;
              delete $3;
              exit(2);
            }
          }
          | '-' cond_term %prec UMINUS {
            for (std::map<std::string, int>::iterator it = $2->begin(); it != $2->end(); it++)
              it->second = -(it->second);
            $$ = $2;
          }              
;

command : '\n' { if (is_interactive) printf("%s ", PROMPT_STRING); }
        | error '\n' { if (!is_interactive) exit(2); else printf("%s ", PROMPT_STRING); }
        | SOURCE ':' FILENAME '\n' {
          if (source_filename != NULL) {
            fprintf(stderr, "only one file can be handled in a script");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          //fprintf(stderr, "source: %s\n", std::string($3).c_str()); 
          source_filename = $3;
          if (is_interactive)
            printf("%s ", PROMPT_STRING);
        }
        | DEST ':' FILENAME '\n' {
          if (dest_filename != NULL) {
            fprintf(stderr, "only one destination file can be handled in a script");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          //fprintf(stderr, "dest: %s\n", std::string($3).c_str()); 
          dest_filename = $3;
          if (is_interactive)
            printf("%s ", PROMPT_STRING);
        }
        | PROCEDURE ':' VARIABLE '\n' {

          #if defined(FRONTEND_ROSE) || defined(FRONTEND_CLANG)

          if (procedure_name != NULL) {
            fprintf(stderr, "only one procedure can be handled in a script");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          procedure_name = $3;
          //fprintf(stderr, "procedure is %s\n", procedure_name); 

          if (is_interactive)
            printf("%s ", PROMPT_STRING);
          #else
            fprintf(stderr, "Please configure IR type to ROSE or CLANG!!: procedure name for ROSE or CLANG/LLVM");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);   
          #endif         
        }


        | PROCEDURE ':' NUMBER '\n' { //  this works only for SUIF (no longer supported)

        #if defined(FRONTEND_ROSE) || defined(FRONTEND_CLANG) 
            fprintf(stderr, "Please specify procedure's name and not number!!");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
        #else
            fprintf(stderr, "Please configure FRONTEND to ROSE or CLANG!!");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);   
        #endif
        }



        | FORMAT ':' FILENAME '\n' {  // TODO  this is really front end, not IR format
           fprintf(stderr, "format: %s\n",  std::string($3).c_str());
         if (frontend_compiler != FE_NULL) {
            fprintf(stderr, "compiler intermediate format already specified");
            PRINT_ERR_LINENO;
            delete []$3;
            if (!is_interactive)
              exit(2);
          }
          else {
            //fprintf(stderr, "format %s\n", std::string($3).c_str()); 
            if(std::string($3) == "rose" || std::string($3) == "ROSE") {
              frontend_compiler = FE_ROSE;
              delete []$3;
            }   
            else if(std::string($3) == "clang" || std::string($3) == "CLANG") {
              frontend_compiler = FE_CLANG;
              delete []$3;
            }   
            else {
              fprintf(stderr, "unrecognized front end compiler. pick rose or clang\n");
              PRINT_ERR_LINENO;
              delete []$3;
              if (!is_interactive)
                exit(2);
            }
          }
          if (is_interactive)
            printf("%s ", PROMPT_STRING);
        }

        | LOOP ':' NUMBER '\n' {
          if (source_filename == NULL) {
            fprintf(stderr, "source file not set when initializing the loop");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          else {
          if (ir_code == NULL) {
            //fprintf(stderr, "LOOP ':' NUMBER   parse the file because we haven't yet\n"); 
            /* parse the file because we haven't yet */
            #if defined(FRONTEND_ROSE) || defined(FRONTEND_CLANG) 
              if (procedure_name == NULL)
                procedure_name = "main";
            #else
              fprintf(stderr, "Please configure FRONTEND to ROSE or CLANG!!");
              PRINT_ERR_LINENO;
              if (!is_interactive)
                exit(2);   
            #endif

            switch (frontend_compiler) {
              case FE_CLANG:
              #ifdef FRONTEND_CLANG 
                 fprintf(stderr, "clang parsing the file\n"); 
                ir_code = new IR_clangCode(source_filename, procedure_name);
              #else
                fprintf(stderr, "CLANG front end not installed");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
              #endif 
                break;

              case FE_ROSE:
              //fprintf(stderr, "FE_ROSE\n"); 
              #ifdef FRONTEND_ROSE
                 //fprintf(stderr, "LOOP  ir_code = new IR_roseCode(source_filename, procedure_name);\n"); 
                 ir_code = new IR_roseCode(source_filename, procedure_name, dest_filename);
                 //fprintf(stderr, "LOOP RETURN ir_code = new IR_roseCode(source_filename, procedure_name);\n"); 

              #else
                fprintf(stderr, "ROSE IR not installed");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
              #endif
                break; 

              case FE_NULL:
                fprintf(stderr, "compiler IR format not specified");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
                break;
              }
            
            /* get the code associated with the procedure */
            //fprintf(stderr, "parser.yy L504  yyparse  block = ir_code->GetCode();\n"); 
            IR_Block *irblock = ir_code->GetCode();  // block that is / enclosing the FUNCTION DEFINITION ??
            //fprintf(stderr, "parser.yy L506 irblock from getcode is %p\n", irblock);
            //IR_roseblock *rb = (IR_roseblock *)irblock;
            //fprintf(stderr, "cheating ast is %p\n", rb->chillAST); 

            ir_controls = ir_code->FindOneLevelControlStructure(irblock);
            for (int i = 0; i < ir_controls.size(); i++) { 
              if (ir_controls[i]->type() == IR_CONTROL_LOOP)
                loops.push_back(i);
            }
            delete irblock;
            }
            //fprintf(stderr, "(parser.yy) I found %ld loops in the procedure\n",  loops.size()); 

            if (myloop != NULL && myloop->isInitialized()) {
              //fprintf(stderr, "there is already a myloop and it is initialized\n"); 
              fprintf(stderr, "have to update <clang/rose/gcc> Intermediate Representation before changing the next loop\n"); 
              /* we are processing multiple loops. replace the <clang/rose/gcc> IR
                 for loops we've already changed (?) */

              if (loop_num_start == loop_num_end) {
                /* we changed one loop before (?) */
                //fprintf(stderr, "replacing code for a single loop in <clang/rose/gcc> IR\n"); 
                ir_code->ReplaceCode(ir_controls[loops[loop_num_start]], myloop->getCode());
                ir_controls[loops[loop_num_start]] = NULL;
              }
              else {
                /* we changed multiple loops before (?) */
                //fprintf(stderr, "replacing code for multiple loops in <clang/rose/gcc> IR\n"); 
                std::vector<IR_Control *> parm;
                for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++)
                  parm.push_back(ir_controls[i]);
                IR_Block *block = ir_code->MergeNeighboringControlStructures(parm);
                ir_code->ReplaceCode(block, myloop->getCode());
                for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++) {
                  delete ir_controls[i];
                  ir_controls[i] = NULL;
                }
              } 
              //fprintf(stderr, "deleting old myloop\n"); 
              delete myloop;
              myloop = NULL; 
            }
            loop_num_start = loop_num_end = $3;
            if (loop_num_start >= loops.size()) {
              fprintf(stderr, "loop %d does not exist", loop_num_start);
              PRINT_ERR_LINENO;
              if (!is_interactive)
                exit(2);
            }
            if (ir_controls[loops[loop_num_start]] == NULL) {
              fprintf(stderr, "loop %d has already been transformed", loop_num_start);
              PRINT_ERR_LINENO;
              if (!is_interactive)
                exit(2);
            }

            //fprintf(stderr, "\nparse.yy  L 505 making a new myloop loop num start %d\n",loop_num_start ) ;
            //fprintf(stderr, "\n***                                                   ROSE (parser.yy) making a new myloop\n"); 
            myloop = new Loop(ir_controls[loops[loop_num_start]]);
            //fprintf(stderr, "\nparse.yy  L 559, made a new loop\n"); 

            //fprintf(stderr, "\n(start dump of loop I'm about to change )\n"); 
            // myloop->dump(); fflush(stdout); fprintf(stderr, "(end dump)\n\n(start printcode)\n");
            //// this dies TODO myloop->printCode(); fflush(stdout);
            //fprintf(stderr, "(end printcode)\n\n(internal loop strucuure)\n");
            //myloop->print_internal_loop_structure(); fflush(stdout);
            //fprintf(stderr, "\n(iteration space)\n"); 
            //myloop->printIterationSpace();  fflush(stdout);
            //fprintf(stderr, "\n"); 
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | LOOP ':' NUMBER '-' NUMBER '\n' {
          /* modify a range of loops */
          //fprintf(stderr, "loop: a-b\n");

          if (source_filename == NULL) {
            fprintf(stderr, "source file not set when initializing the loop");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          else {
            if (ir_code == NULL) { 

            if (procedure_name == NULL)
              procedure_name = "main";
        
            switch (frontend_compiler) {
              case FE_ROSE:
              #ifdef FRONTEND_ROSE
                 ir_code = new IR_roseCode(source_filename, procedure_name, dest_filename);
              #else
                fprintf(stderr, "ROSE IR not installed");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
              #endif
                break;

              // todo case FE_CLANG

              case FE_NULL:
                fprintf(stderr, "compiler IR format not specified");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
                break;
              }
            
           
 
              IR_Block *block = ir_code->GetCode();
              //fprintf(stderr, "calling FindOneLevelControlStructure  for loop: a-b\n"); 
              ir_controls = ir_code->FindOneLevelControlStructure(block);
              for (int i = 0; i < ir_controls.size(); i++)
                if (ir_controls[i]->type() == IR_CONTROL_LOOP)
                  loops.push_back(i);
              delete block;
           }                          
              if (myloop != NULL && myloop->isInitialized()) {
                if (loop_num_start == loop_num_end) {
                  ir_code->ReplaceCode(ir_controls[loops[loop_num_start]], myloop->getCode());
                  ir_controls[loops[loop_num_start]] = NULL;
                }
                else {
                  std::vector<IR_Control *> parm;
                  for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++)
                    parm.push_back(ir_controls[i]);
                  IR_Block *block = ir_code->MergeNeighboringControlStructures(parm);
                  ir_code->ReplaceCode(block, myloop->getCode());
                  for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++) {
                    delete ir_controls[i];
                    ir_controls[i] = NULL;
                  }
                }
                delete myloop;
              }
              loop_num_start = $3;
              loop_num_end = $5;
              if ($5 < $3) {
                fprintf(stderr, "the last loop must be after the start loop");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
              }              
              if (loop_num_end >= loops.size()) {
                fprintf(stderr, "loop %d does not exist", loop_num_end);
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
              }
              std::vector<IR_Control *> parm;
              for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++) {
                if (ir_controls[i] == NULL) {
                  fprintf(stderr, "loop has already been processed");
                  PRINT_ERR_LINENO;
                  if (!is_interactive)
                    exit(2);
                }
                parm.push_back(ir_controls[i]);
              }
              IR_Block *block = ir_code->MergeNeighboringControlStructures(parm);
              myloop = new Loop(block);
              delete block;
            
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | PRINT '\n' {
          //printf("\n*** parser: print\n"); fflush(stdout); 
          if (myloop == NULL) {
            fprintf(stderr, "loop not initialized");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          else {
            myloop->printCode();
          }
          if (is_interactive) printf("%s ", PROMPT_STRING); else printf("\n");
        }
        | PRINT PRINT_CODE '\n' {
          if (myloop == NULL) {
            fprintf(stderr, "loop not initialized");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          else {
            if (!is_interactive) {
              if (loop_num_start != loop_num_end)
                std::cout << "/* procedure :" << procedure_name << " loop #" << loop_num_start << "-" << loop_num_end << " */" << std::endl;
              else
                std::cout << "/* procedure :" << procedure_name << " loop #" << loop_num_start << " */" << std::endl;

           } 

            myloop->printCode();
          }
          if (is_interactive) printf("%s ", PROMPT_STRING); else printf("\n");
        }
        | PRINT PRINT_DEP '\n' {
          if (myloop == NULL) {
            fprintf(stderr, "loop not initialized");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              YYABORT;
          }
          else {
            myloop->printDependenceGraph();
          }
          if (is_interactive) printf("%s ", PROMPT_STRING); else printf("\n");
        }
        | PRINT PRINT_IS '\n' {
          if (myloop == NULL) {
            fprintf(stderr, "loop not initialized");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              YYABORT;
          }
          else {
            myloop->printIterationSpace();
          }
          if (is_interactive) printf("%s ", PROMPT_STRING); else printf("\n");
        }
        | PRINT PRINT_STRUCTURE '\n' {
          if (myloop == NULL) {
            fprintf(stderr, "loop not initialized");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          else {
            myloop->print_internal_loop_structure();
          }
          if (is_interactive) printf("%s ", PROMPT_STRING); else printf("\n");
        }
        | PRINT expr '\n' {
/*          if (parameter_tab.find(std::string($2)) == parameter_tab.end()) {
            fprintf(stderr, "cannot print undefined variable %s\n", $2);
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          std::cout << parameter_tab[std::string($2)] << std::endl;
*/
          std::cout << $2 << std::endl;
          if (is_interactive) printf("%s ", PROMPT_STRING); else printf("\n");
        }
        | EXIT '\n' { return(0); }
        | VARIABLE '=' expr '\n' {
          parameter_tab[std::string($1)] = $3;
          delete []$1;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }


        | KNOWN '(' cond ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            
            //fprintf(stderr, "KNOWN\n"); 
            int num_dim = myloop->known.n_set();
            //fprintf(stderr, "num_dim %d\n", num_dim); 

            Relation rel(num_dim);
            F_And *f_root = rel.add_and();
            for (int j = 0; j < $3->size(); j++) {
              GEQ_Handle h = f_root->add_GEQ();
              for (std::map<std::string, int>::iterator it = (*$3)[j].begin(); it != (*$3)[j].end(); it++) {
                try {
                  int dim = from_string<int>(it->first);
                  if (dim == 0)
                    h.update_const(it->second);
                  else
                    throw std::invalid_argument("only symbolic variables are allowed in known condition");
                }
                catch (std::ios::failure e) {
                  Free_Var_Decl *g = NULL;
                  for (unsigned i = 0; i < myloop->freevar.size(); i++) {
                    std::string name = myloop->freevar[i]->base_name();
                    if (name == it->first) {
                      g = myloop->freevar[i];
                      break;
                    }
                  }
                  if (g == NULL) { 
                    fprintf(stderr, "parser,  g NULL\n"); 
                    throw std::invalid_argument("symbolic variable " + it->first + " not found");
                    }
                  else
                    h.update_coef(rel.get_local(g), it->second);
                }
              }
            }
            myloop->addKnown(rel);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | REMOVE_DEP '(' NUMBER ',' NUMBER ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->removeDependence($3, $5);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              YYABORT;
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        } 
        | ORIGINAL '(' ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->original();
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | PERMUTE '(' vector ')' '\n' {
         //fprintf(stderr, "\n\n\n*** PERMUTE\n"); 
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->permute(*$3);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | PERMUTE '(' expr ',' NUMBER ',' vector ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->permute($3, $5, *$7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $7;
              exit(2);
            }
          }
          delete $7;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | PERMUTE '(' vector ',' vector ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            std::set<int> active;
            for (int i = 0; i < (*$3).size(); i++)
              active.insert((*$3)[i]);
            
            myloop->permute(active, *$5);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              delete $5;
              exit(2);
            }
          }
          delete $3;
          delete $5;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | PRAGMA '(' NUMBER ',' NUMBER ',' STRING ')' '\n' {
          myloop->pragma($3,$5,$7);
        }
        | PREFETCH '(' NUMBER ',' NUMBER ',' STRING ',' expr ')' '\n' {
         myloop->prefetch($3, $5, $7, $9);
        }
        | TILE '(' expr ',' NUMBER ',' expr ')' '\n' {
          //fprintf(stderr, "TILE (3)\n"); 
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->tile($3,$5,$7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }


        | FIND_STENCIL_SHAPE '(' NUMBER ')' { 
          printf("\n*** parser: find stencil shape\n"); fflush(stdout); 
          myloop->find_stencil_shape( $3 ); 
        }


        | STENCIL_TEMP '(' NUMBER ')' { 
          printf("\n*** parser: protonu's stencil\n"); fflush(stdout); 
          myloop->stencilASEPadded( $3 ); 
          printf("\n*** parser: protonu's stencil DONE\n"); fflush(stdout); 
        }


        | FLATTEN '(' NUMBER  ',' VARIABLE ',' vector ',' VARIABLE ')'  '\n' {
          #if 0
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
             std::vector<int> loop_levels;
            for (int i = 0; i < (*$7).size(); i++)
              loop_levels.push_back((*$7)[i]);

            const char* index= $5; 
            std::string str1(index);
            const char* inspector= $9; 
            std::string str2(inspector);
            myloop->flatten($3,str1, loop_levels,str2);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
          #endif
        }

        | COMPACT '(' NUMBER  ',' NUMBER ',' VARIABLE ',' NUMBER ',' VARIABLE')'  '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
              
            myloop->compact($3,$5, $7, $9,$11);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | MAKE_DENSE '(' NUMBER  ',' NUMBER ',' VARIABLE')'  '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            printf("parser: make_dense(number, number, variable)\n"); fflush(stdout); 
            myloop->make_dense($3,$5,$7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }

        | SET_ARRAY_SIZE '(' VARIABLE  ',' NUMBER ')'  '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->set_array_size($3,$5);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }

/*
        | ELLIFY '(' NUMBER  ',' vector_string ',' NUMBER')'  '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            std::vector<std::string> arrays;
            for (int i = 0; i < (*$5).size(); i++)
              arrays.push_back((*$5)[i]);

            myloop->ELLify($3, arrays,$7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }

        | ELLIFY '(' NUMBER  ',' vector_string ',' NUMBER ',' TRUEORFALSE ',' VARIABLE')'  '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            std::vector<std::string> arrays;
            for (int i = 0; i < (*$5).size(); i++)
              arrays.push_back((*$5)[i]);

            myloop->ELLify($3, arrays,$7,$9,$11);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
*/
        | NORMALIZE '(' NUMBER  ',' NUMBER ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            myloop->normalize($3, $5);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }

        | TILE '(' expr',' NUMBER ',' expr ',' NUMBER ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->tile($3,$5,$7,$9);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | TILE '(' expr ',' NUMBER ',' expr ',' NUMBER ',' STRIDED ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->tile($3,$5,$7,$9,StridedTile);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | TILE '(' expr ',' NUMBER ',' expr ',' NUMBER ',' STRIDED ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->tile($3,$5,$7,$9,StridedTile,$13);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | TILE '(' expr ',' NUMBER ',' expr ',' NUMBER ',' STRIDED ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->tile($3,$5,$7,$9,StridedTile,$13,$15);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | TILE '(' expr ',' NUMBER ',' expr ',' NUMBER ',' COUNTED ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->tile($3,$5,$7,$9,CountedTile);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | TILE '(' expr ',' NUMBER ',' expr ',' NUMBER ',' COUNTED ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->tile($3,$5,$7,$9,CountedTile,$13);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | TILE '(' expr ',' NUMBER ',' expr ',' NUMBER ',' COUNTED ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->tile($3,$5,$7,$9,CountedTile,$13,$15);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' matrix ',' NUMBER ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            
            std::vector<std::pair<int, std::vector<int> > > array_ref_nums((*$3).size());
            for (int i = 0; i < (*$3).size(); i++) {
              if ((*$3)[i].size() <= 1)
                throw std::invalid_argument("statement missing in the first parameter");
              array_ref_nums[i].first = (*$3)[i][0];
              for (int j = 1; j < (*$3)[i].size(); j++)
                array_ref_nums[i].second.push_back((*$3)[i][j]);
            }
            myloop->datacopy(array_ref_nums,$5);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' matrix ',' NUMBER ',' TRUEORFALSE ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            
            std::vector<std::pair<int, std::vector<int> > > array_ref_nums((*$3).size());
            for (int i = 0; i < (*$3).size(); i++) {
              if ((*$3)[i].size() <= 1)
                throw std::invalid_argument("statement missing in the first parameter");
              array_ref_nums[i].first = (*$3)[i][0];
              for (int j = 1; j < (*$3)[i].size(); j++)
                array_ref_nums[i].second.push_back((*$3)[i][j]);
            }
            myloop->datacopy(array_ref_nums,$5,$7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' matrix ',' NUMBER ',' TRUEORFALSE ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            
            std::vector<std::pair<int, std::vector<int> > > array_ref_nums((*$3).size());
            for (int i = 0; i < (*$3).size(); i++) {
              if ((*$3)[i].size() <= 1)
                throw std::invalid_argument("statement missing in the first parameter");
              array_ref_nums[i].first = (*$3)[i][0];
              for (int j = 1; j < (*$3)[i].size(); j++)
                array_ref_nums[i].second.push_back((*$3)[i][j]);
            }
            myloop->datacopy(array_ref_nums,$5,$7,$9);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' matrix ',' NUMBER ',' TRUEORFALSE ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            
            std::vector<std::pair<int, std::vector<int> > > array_ref_nums((*$3).size());
            for (int i = 0; i < (*$3).size(); i++) {
              if ((*$3)[i].size() <= 1)
                throw std::invalid_argument("statement missing in the first parameter");
              array_ref_nums[i].first = (*$3)[i][0];
              for (int j = 1; j < (*$3)[i].size(); j++)
                array_ref_nums[i].second.push_back((*$3)[i][j]);
            }
            myloop->datacopy(array_ref_nums,$5,$7,$9,$11);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' matrix ',' NUMBER ',' TRUEORFALSE ',' expr ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            
            std::vector<std::pair<int, std::vector<int> > > array_ref_nums((*$3).size());
            for (int i = 0; i < (*$3).size(); i++) {
              if ((*$3)[i].size() <= 1)
                throw std::invalid_argument("statement missing in the first parameter");
              array_ref_nums[i].first = (*$3)[i][0];
              for (int j = 1; j < (*$3)[i].size(); j++)
                array_ref_nums[i].second.push_back((*$3)[i][j]);
            }
            myloop->datacopy(array_ref_nums,$5,$7,$9,$11,$13);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' expr ',' NUMBER ',' VARIABLE ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            
            myloop->datacopy($3,$5,$7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete []$7;
              exit(2);
            }
          }
          delete []$7;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' expr ',' NUMBER ',' VARIABLE ',' TRUEORFALSE ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            
            myloop->datacopy($3,$5,$7,$9);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete []$7;
              exit(2);
            }
          }
          delete []$7;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' expr ',' NUMBER ',' VARIABLE ',' TRUEORFALSE ',' expr ')' '\n' {
          //fprintf(stderr, "\n\n\nDATACOPY (5)\n"); 
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->datacopy($3,$5,$7,$9,$11);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete []$7;
              exit(2);
            }
          }
          delete []$7;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' expr ',' NUMBER ',' VARIABLE ',' TRUEORFALSE ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->datacopy($3,$5,$7,$9,$11,$13);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete []$7;
              exit(2);
            }
          }
          delete []$7;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY '(' expr ',' NUMBER ',' VARIABLE ',' TRUEORFALSE ',' expr ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->datacopy($3,$5,$7,$9,$11,$13,$15);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete []$7;
              exit(2);
            }
          }
          delete []$7;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }        
        | DATACOPY_PRIVATIZED '(' matrix ',' NUMBER ',' vector ',' TRUEORFALSE ',' expr ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            
            std::vector<std::pair<int, std::vector<int> > > array_ref_nums((*$3).size());
            for (int i = 0; i < (*$3).size(); i++) {
              if ((*$3)[i].size() <= 1)
                throw std::invalid_argument("statement missing in the first parameter");
              array_ref_nums[i].first = (*$3)[i][0];
              for (int j = 1; j < (*$3)[i].size(); j++)
                array_ref_nums[i].second.push_back((*$3)[i][j]);
            }
            myloop->datacopy_privatized(array_ref_nums,$5,*$7,$9,$11,$13,$15);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              delete $7;
              exit(2);
            }
          }
          delete $3;
          delete $7;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DATACOPY_PRIVATIZED '(' expr ',' NUMBER ',' VARIABLE ',' vector ',' TRUEORFALSE ',' expr ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->datacopy_privatized($3,$5,$7,*$9,$11,$13,$15,$17);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete []$7;
              delete $9;
              exit(2);
            }
          }
          delete []$7;
          delete $9;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }        


	| ADD_GHOSTS '(' vector ',' NUMBER ',' NUMBER ',' NUMBER ')' '\n'{

       	  try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
      	   std::vector<int> stmt_nums;
            for (int i = 0; i < (*$3).size(); i++)
              stmt_nums.push_back((*$3)[i]);

            //myloop->generate_ghostcells($3,$5,$7);
            myloop->generate_ghostcells_v2(stmt_nums,$5,$7, $9);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);

	}




        | UNROLL '(' expr ',' NUMBER ',' expr ')' '\n' {
          //fprintf(stderr, "\n\n                             parser 1          unroll( a,b,c )\n"); 
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->unroll($3,$5,$7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | UNROLL '(' expr ',' NUMBER ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

          //fprintf(stderr, "\n\n                                 parser 2      unroll( a,b,c )\n"); 
            myloop->unroll($3,$5,$7,std::vector< std::vector<std::string> >(),  $9);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | UNROLL_EXTRA '(' expr ',' NUMBER ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

          //fprintf(stderr, "\n\n                                 parser 3      unroll( a,b,c )\n"); 
            myloop->unroll_extra($3,$5,$7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | UNROLL_EXTRA '(' expr ',' NUMBER ',' expr ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

          //fprintf(stderr, "\n\n                                 parser 4      unroll( a,b,c )\n"); 
            myloop->unroll_extra($3,$5,$7,$9);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | SPLIT '(' expr ',' NUMBER ',' cond ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");
            if ($3 < 0 || $3 >= myloop->num_statement())
              throw std::invalid_argument("invalid statement " + to_string($3));
            int num_dim = myloop->stmt[$3].xform.n_out();
            
            Relation rel((num_dim-1)/2);
            F_And *f_root = rel.add_and();
            for (int j = 0; j < $7->size(); j++) {
              GEQ_Handle h = f_root->add_GEQ();
              for (std::map<std::string, int>::iterator it = (*$7)[j].begin(); it != (*$7)[j].end(); it++) {
                try {
                  int dim = from_string<int>(it->first);
                  if (dim == 0)
                    h.update_const(it->second);
                  else {
                    if (dim > (num_dim-1)/2)
                      throw std::invalid_argument("invalid loop level " + to_string(dim) + " in split condition");
                    h.update_coef(rel.set_var(dim), it->second);
                  }
                }
                catch (std::ios::failure e) {
                  Free_Var_Decl *g = NULL;
                  for (unsigned i = 0; i < myloop->freevar.size(); i++) {
                    std::string name = myloop->freevar[i]->base_name();
                    if (name == it->first) {
                      g = myloop->freevar[i];
                      break;
                    }
                  }
                  if (g == NULL)
                    throw std::invalid_argument("unrecognized variable " + to_string(it->first.c_str()));
                  h.update_coef(rel.get_local(g), it->second);
                }
              }
            }
            myloop->split($3,$5,rel);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $7;
              exit(2);
            }
          }
          delete $7;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | NONSINGULAR '(' matrix ')' '\n' {
          try {
            myloop->nonsingular(*$3);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | SKEW '(' vector ',' NUMBER ',' vector ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            std::set<int> stmt_nums;
            for (int i = 0; i < (*$3).size(); i++)
              stmt_nums.insert((*$3)[i]);
            myloop->skew(stmt_nums, $5, *$7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              delete $7;
              exit(2);
            }
          }
          delete $3;
          delete $7;
          if (is_interactive) printf("%s ", PROMPT_STRING);
          }
          | SCALE '(' vector ',' NUMBER ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            std::set<int> stmt_nums;
            for (int i = 0; i < (*$3).size(); i++)
              stmt_nums.insert((*$3)[i]);
            myloop->scale(stmt_nums, $5, $7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | REVERSE '(' vector ',' NUMBER ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            std::set<int> stmt_nums;
            for (int i = 0; i < (*$3).size(); i++)
              stmt_nums.insert((*$3)[i]);
            myloop->reverse(stmt_nums, $5);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
             exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | SHIFT '(' vector ',' NUMBER ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            std::set<int> stmt_nums;
            for (int i = 0; i < (*$3).size(); i++)
              stmt_nums.insert((*$3)[i]);

            myloop->shift(stmt_nums, $5, $7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | SHIFT_TO '(' expr ',' NUMBER ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->shift_to($3, $5, $7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              exit(2);
            }
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        } 
        | PEEL '(' NUMBER ',' NUMBER ',' expr ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->peel($3, $5, $7);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              exit(2);
            }
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | PEEL '(' NUMBER ',' NUMBER ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            myloop->peel($3, $5);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              exit(2);
            }
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }         
        | FUSE '(' vector ',' NUMBER ')' '\n' {
          try {
            if (myloop == NULL)
              throw std::runtime_error("loop not initialized");

            std::set<int> stmt_nums;
            for (int i = 0; i < (*$3).size(); i++)
              stmt_nums.insert((*$3)[i]);

            myloop->fuse(stmt_nums, $5);
          }
          catch (const std::exception &e) {
            fprintf(stderr, e.what());
            PRINT_ERR_LINENO;
            if (!is_interactive) {
              delete $3;
              exit(2);
            }
          }
          delete $3;
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | DISTRIBUTE '(' vector ',' NUMBER ')' '\n' {
          if (myloop == NULL) {
            fprintf(stderr, "loop not initialized");
            PRINT_ERR_LINENO;
            delete $3;
            if (!is_interactive)
              exit(2);
          }
          else {
            std::set<int> stmt_nums;
            for (int i = 0; i < (*$3).size(); i++)
              stmt_nums.insert((*$3)[i]);
            delete $3;
            try {
              myloop->distribute(stmt_nums, $5);
            }
            catch (const std::exception &e) {
              fprintf(stderr, e.what());
              PRINT_ERR_LINENO;
              if (!is_interactive)
                exit(2);
            }
            if (is_interactive) printf("%s ", PROMPT_STRING);
          }
        }
;

%%


void yyerror(const char *str) {
  int err_lineno = lexer.lineno();
  if (lexer.YYText()[0] == '\n')
    err_lineno--;

  if (is_interactive)
    fprintf(stderr, "%s\n", str);
  else 
    fprintf(stderr, "%s at line %d\n", str, err_lineno);
}

int main(int argc, char *argv[]) {
  yydebug = 0;

  if (argc > 2) {
    fprintf(stderr, "Usage: %s [script_file]\n", argv[0]);
    exit(-1);
  }

  std::ifstream script;
  if (argc == 2) {
    script.open(argv[1]);
    if (!script.is_open()) {
      printf("can't open script file \"%s\"\n", argv[1]);
      exit(-1);
    }  
    lexer.switch_streams(&script, &std::cout);
  }

  if (argc == 1 && isatty((int)fileno(stdin))) {
    is_interactive = true;
    printf("CHiLL v0.2.0 (built on %s)\n", CHILL_BUILD_DATE);
    printf("Copyright (C) 2008 University of Southern California\n");
    printf("Copyright (C) 2009-2012 University of Utah\n");
    printf("%s ", PROMPT_STRING);
  }
  else
    is_interactive = false;
    
  ir_code = NULL;
  initializeOmega();

  if (yyparse() == 0) {
    if (!is_interactive)
      fprintf(stderr, "script success!\n");
    else
      printf("\n");

      
    if (ir_code != NULL && myloop != NULL && myloop->isInitialized()) {
      fprintf(stderr, "\nparser.yy almost done\n"); 
      if (loop_num_start == loop_num_end) {
        fprintf(stderr, "1 loop?   (loop_num_start == loop_num_end)\n"); 

        fprintf(stderr, "oldrepr = myloop->getCode()\n");
        omega::CG_outputRepr *oldrepr = myloop->getCode();

        fprintf(stderr, "ir_code->ReplaceCode()\n"); 
        ir_code->ReplaceCode(ir_controls[loops[loop_num_start]], oldrepr);
        ir_controls[loops[loop_num_start]] = NULL;
      }
      else {
        fprintf(stderr, "multiple loops?   (loop_num_start != loop_num_end)\n"); 
        std::vector<IR_Control *> parm;
        for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++)
          parm.push_back(ir_controls[i]);

        fprintf(stderr, "ir_code->MergeNeighboringControlStructures(parm)\n"); 
        IR_Block *block = ir_code->MergeNeighboringControlStructures(parm);
        ir_code->ReplaceCode(block, myloop->getCode());
        for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++) {
          delete ir_controls[i];
          ir_controls[i] = NULL;
        }
      } 
    }
  }

  delete myloop;
  for (int i = 0; i < ir_controls.size(); i++)
    delete ir_controls[i];

  #ifdef FRONTEND_ROSE
  ((IR_roseCode*)(ir_code))->finalizeRose();
  #endif

  fprintf(stderr, "parser.yy  delete ir_code;\n"); 
  delete ir_code;
  delete []source_filename;
  delete []dest_filename;
}
