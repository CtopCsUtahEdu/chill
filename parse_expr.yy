%define api.prefix {expr}

%{
#include "chill_run_util.hh"
#include "parse_expr.ll.hh"
#include "chill_io.hh"

extern int exprdebug;

int exprlex(void);
void exprerror(const char*);
int exprparse(simap_vec_t** rel);

static simap_vec_t* return_rel; // used as the return value for yyparse

%}

%union {
  int val;
  char* str_val;
  simap_t* cond_item;
  simap_vec_t* cond;
}

%token <val> NUMBER
%token <val> LEVEL
%token <str_val> VARIABLE

%left LE GE EQ '<' '>'
%left '-' '+' '*' '/'

/*the final output from this language should be an Omega Relation object*/
%type <cond> cond prog
%type <cond_item> expr add_expr mul_expr neg_expr

%%
prog : cond                      { return_rel = make_prog($1); }
;

cond : expr '>' expr             { $$ = make_cond_gt($1, $3); }
     | expr '<' expr             { $$ = make_cond_lt($1, $3); }
     | expr GE expr              { $$ = make_cond_ge($1, $3); }
     | expr LE expr              { $$ = make_cond_le($1, $3); }
     | expr EQ expr              { $$ = make_cond_eq($1, $3); }
;

expr : add_expr                  { $$ = $1; }
;

add_expr : add_expr '+' mul_expr { $$ = make_cond_item_add($1,$3); }
         | add_expr '-' mul_expr { $$ = make_cond_item_sub($1,$3); }
         | mul_expr              { $$ = $1; }
;

mul_expr : mul_expr '*' neg_expr { $$ = make_cond_item_mul($1,$3); }
         | neg_expr              { $$ = $1; }
;

neg_expr : '-' neg_expr          { $$ = make_cond_item_neg($2); }
         | '(' expr ')'          { $$ = $2; }
         | NUMBER                { $$ = make_cond_item_number($1); }
         | LEVEL                 { $$ = make_cond_item_level($1); }
         | VARIABLE              { $$ = make_cond_item_variable($1); }
;
%%

void exprerror(const char* msg) {
  debug_fprintf(stderr, "Parse error: %s", msg);
}

simap_vec_t* parse_relation_vector(const char* expr) {
  exprdebug=0;
  YY_BUFFER_STATE state;
  
  //if(yylex_init()) {
  //   TODO: error out or something
  //}
  
  state = expr_scan_string(expr);
  
  if(exprparse()) {
    // TODO: error out or something
  }
  
  expr_delete_buffer(state);
  exprlex_destroy();
  return return_rel;
}

