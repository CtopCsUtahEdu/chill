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
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <FlexLexer.h>
#include "parser.tab.hh"

#include <omega.h>
#include "ir_code.hh"
#include "loop.hh"

#ifdef BUILD_ROSE
#include "ir_rose.hh"
#elif BUILD_SUIF
#include "ir_suif.hh"
#endif


using namespace omega;
  
extern int yydebug;

void yyerror(const char *);
int yylex();  
yyFlexLexer lexer;

namespace {
  enum COMPILER_IR_TYPE {CIT_NULL, CIT_SUIF, CIT_ROSE};
  char *source_filename = NULL;
  COMPILER_IR_TYPE cit_name = CIT_NULL;
  #ifdef BUILD_ROSE
  char* procedure_name = NULL;
  #elif BUILD_SUIF
  int procedure_number = -1;
  #endif
   
  int loop_num_start, loop_num_end;
  Loop *myloop = NULL;
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
  std::vector<std::vector<int> > *mat;
  std::map<std::string, int> *tab;
  std::vector<std::map<std::string, int> > *tab_lst;
  std::pair<std::vector<std::map<std::string, int> >, std::map<std::string, int> > *eq_term_pair;
}

%token <val> NUMBER LEVEL
%token <bool_val> TRUEORFALSE
%token <name> FILENAME PROCEDURENAME VARIABLE FREEVAR STRING
%token SOURCE PROCEDURE FORMAT LOOP PERMUTE ORIGINAL TILE UNROLL SPLIT UNROLL_EXTRA PRAGMA PREFETCH
%token DATACOPY DATACOPY_PRIVATIZED
%token NONSINGULAR EXIT KNOWN SKEW SHIFT SHIFT_TO FUSE DISTRIBUTE REMOVE_DEP SCALE REVERSE PEEL
%token STRIDED COUNTED NUM_STATEMENT CEIL FLOOR
%token PRINT PRINT_CODE PRINT_DEP PRINT_IS PRINT_STRUCTURE
%token NE LE GE EQ

%type <vec> vector vector_number
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
            fprintf(stderr, "only one file can be handle in a script");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          source_filename = $3;
          if (is_interactive)
            printf("%s ", PROMPT_STRING);
        }
        | PROCEDURE ':' VARIABLE '\n' {

          #ifdef BUILD_ROSE

          if (procedure_name != NULL) {
            fprintf(stderr, "only one procedure can be handled in a script");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          procedure_name = $3;
          if (is_interactive)
            printf("%s ", PROMPT_STRING);
          #elif BUILD_SUIF
            fprintf(stderr, "Please specify procedure number and not name!!");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          #else
            fprintf(stderr, "Please configure IR type to ROSE or SUIF!!: Procedure number for SUIF and procedure name for ROSE");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);   
          #endif         
        }
        | PROCEDURE ':' NUMBER '\n' {

        #ifdef BUILD_ROSE
            fprintf(stderr, "Please specify procedure's name and not number!!");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
        
        #elif BUILD_SUIF      
          if (procedure_number != -1) {
            fprintf(stderr, "only one procedure can be handled in a script");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          procedure_number = $3;
          if (is_interactive)
            printf("%s ", PROMPT_STRING);
          #else
            fprintf(stderr, "Please configure IR type to ROSE or SUIF: Procedure number for suif and procedure name for rose!!");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);   
        #endif
        }
        | FORMAT ':' FILENAME '\n' {
          if (cit_name != CIT_NULL) {
            fprintf(stderr, "compiler intermediate format already specified");
            PRINT_ERR_LINENO;
            delete []$3;
            if (!is_interactive)
              exit(2);
          }
          else {
            
            if (std::string($3) == "suif" || std::string($3) == "SUIF") {
              cit_name = CIT_SUIF;
              delete []$3;
            }
            else if(std::string($3) == "rose" || std::string($3) == "ROSE") {
              cit_name = CIT_ROSE;
              delete []$3;
            }   
            else {
              fprintf(stderr, "unrecognized IR format");
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
           #ifdef BUILD_ROSE  
            if (procedure_name == NULL)
              procedure_name = "main";
           #elif BUILD_SUIF   
            if (procedure_number == -1)
              procedure_number = 0;   
            #endif       
           
              switch (cit_name) {
              #ifndef BUILD_ROSE
              case CIT_SUIF:
              #ifdef BUILD_SUIF
                ir_code = new IR_suifCode(source_filename, procedure_number);
              #else
                fprintf(stderr, "SUIF IR not installed");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
              #endif
                break;
              #endif
              case CIT_ROSE:
              #ifdef BUILD_ROSE
                 ir_code = new IR_roseCode(source_filename, procedure_name);
              #else
                fprintf(stderr, "ROSE IR not installed");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
              #endif
                break; 
              case CIT_NULL:
                fprintf(stderr, "compiler IR format not specified");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
                break;
              }
            
            IR_Block *block = ir_code->GetCode();
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
            loop_num_start = loop_num_end = $3;
            if (loop_num_start >= loops.size()) {
              fprintf(stderr, "loop %d does not exist", loop_num_start);
              PRINT_ERR_LINENO;
              if (!is_interactive)
                exit(2);
            }
            if (ir_controls[loops[loop_num_start]] == NULL) {
              fprintf(stderr, "loop %d has already be transformed", loop_num_start);
              PRINT_ERR_LINENO;
              if (!is_interactive)
                exit(2);
            }
            myloop = new Loop(ir_controls[loops[loop_num_start]]);
          }
          if (is_interactive) printf("%s ", PROMPT_STRING);
        }
        | LOOP ':' NUMBER '-' NUMBER '\n' {
          if (source_filename == NULL) {
            fprintf(stderr, "source file not set when initializing the loop");
            PRINT_ERR_LINENO;
            if (!is_interactive)
              exit(2);
          }
          else {
            if (ir_code == NULL) { 
            #ifdef BUILD_ROSE
            if (procedure_name == NULL)
              procedure_name = "main";
            #elif BUILD_SUIF
            if (procedure_number == -1)
              procedure_number = 0;
            #endif            
        
              switch (cit_name) {
              #ifndef BUILD_ROSE
              case CIT_SUIF:
              #ifdef BUILD_SUIF
                ir_code = new IR_suifCode(source_filename, procedure_number);
              #else
                fprintf(stderr, "SUIF IR not installed");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
              #endif
                break;
              #endif
              case CIT_ROSE:
              #ifdef BUILD_ROSE
                 ir_code = new IR_roseCode(source_filename, procedure_name);
              #else
                fprintf(stderr, "ROSE IR not installed");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
              #endif
                break;
              case CIT_NULL:
                fprintf(stderr, "compiler IR format not specified");
                PRINT_ERR_LINENO;
                if (!is_interactive)
                  exit(2);
                break;
              }
            
           
 
              IR_Block *block = ir_code->GetCode();
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
            
            int num_dim = myloop->known.n_set();
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
                  if (g == NULL)
                    throw std::invalid_argument("symbolic variable " + it->first + " not found");
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
        | UNROLL '(' expr ',' NUMBER ',' expr ')' '\n' {
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

inline int yylex() { return lexer.yylex();}

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
    }
  }

  delete myloop;
  for (int i = 0; i < ir_controls.size(); i++)
    delete ir_controls[i];
  #ifdef BUILD_ROSE
  ((IR_roseCode*)(ir_code))->finalizeRose();
  #endif
  delete ir_code;
  delete []source_filename;
}
