//#define PYTHON 1 
// uncomment above line to have python interpret something.lua transformation files
// TODO: put in Makefile?

#include "chill_io.hh"


// this is a little messy. the Makefile should be able to define one or the other
#ifndef PYTHON
#ifndef LUA
#define LUA
#endif
#endif

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <argp.h>

#include "chill_env.hh"

#include "loop.hh"
#include <omega.h>
#include "ir_code.hh"

#ifdef CUDACHILL

#ifdef FRONTEND_ROSE
#include "loop_cuda_chill.hh"
#include "ir_cudarose.hh"
#endif

typedef LoopCuda loop_t;
#else // not defined(CUDACHILL)

#ifdef FRONTEND_ROSE
#include "ir_rose.hh"
#endif

typedef Loop loop_t;
#endif // not defined(CUDACHILL)

#ifdef LUA
#define lua_c //Get the configuration defines for doing an interactive shell
#include <lua.hpp> //All lua includes wrapped in extern "C"
#include "chill_env.hh" // Lua wrapper functions for CHiLL
#elif defined(PYTHON)
#include "chillmodule.hh" // Python wrapper functions for CHiLL
#endif

//---
// CHiLL globals 
//---
loop_t *myloop = NULL;  // was Loop !! 
IR_Code *ir_code = NULL;
bool repl_stop = false;
bool is_interactive = false;

std::vector<IR_Control *> ir_controls;
std::vector<int> loops;

// this whole section belongs somewhere else
#ifdef LUA
//---
// Interactive mode functions, directly copied out of lua.c
//---
// The Lua interpreter state
static lua_State *globalL = NULL;
static const char *progname = "CHiLL";

static void lstop (lua_State *L, lua_Debug *ar) {
  (void)ar;  /* unused arg. */
  lua_sethook(L, NULL, 0, 0);
  luaL_error(L, "interrupted!");
}


static void laction (int i) {
  signal(i, SIG_DFL); /* if another SIGINT happens before lstop,
                         terminate process (default action) */
  lua_sethook(globalL, lstop, LUA_MASKCALL | LUA_MASKRET | LUA_MASKCOUNT, 1);
}


static void l_message (const char *pname, const char *msg) {
  if (pname) debug_fprintf(stderr, "%s: ", pname);
  debug_fprintf(stderr, "%s\n", msg);
}


static int report (lua_State *L, int status) {
  if (status && !lua_isnil(L, -1)) {
    const char *msg = lua_tostring(L, -1);
    if (msg == NULL) msg = "(error object is not a string)";
    l_message(progname, msg);
    lua_pop(L, 1);
  }
  return status;
}


static int traceback (lua_State *L) {
  if (!lua_isstring(L, 1))  /* 'message' not a string? */
    return 1;  /* keep it intact */
  lua_getfield(L, LUA_GLOBALSINDEX, "debug");
  if (!lua_istable(L, -1)) {
    lua_pop(L, 1);
    return 1;
  }
  lua_getfield(L, -1, "traceback");
  if (!lua_isfunction(L, -1)) {
    lua_pop(L, 2);
    return 1;
  }
  lua_pushvalue(L, 1);  /* pass error message */
  lua_pushinteger(L, 2);  /* skip this function and traceback */
  lua_call(L, 2, 1);  /* call debug.traceback */
  return 1;
}


static int docall (lua_State *L, int narg, int clear) {
  debug_fprintf(stderr, "\ndocall() narg %d  clear %d\n", narg, clear); 
  int status;
  int base = lua_gettop(L) - narg;  /* function index */
  lua_pushcfunction(L, traceback);  /* push traceback function */
  lua_insert(L, base);  /* put it under chunk and args */
  signal(SIGINT, laction);
  
  debug_fprintf(stderr, "status = lua_pcall(L, narg, (clear ? 0 : LUA_MULTRET), base);\n"); 
  
  debug_fprintf(stderr, "chill_run.cc L122\n"); 

  status = lua_pcall(L, narg, (clear ? 0 : LUA_MULTRET), base);
  debug_fprintf(stderr, "docall will return status %d\n", status); 

  signal(SIGINT, SIG_DFL);
  lua_remove(L, base);  /* remove traceback function */
  /* force a complete garbage collection in case of errors */
  if (status != 0) lua_gc(L, LUA_GCCOLLECT, 0);
  return status;
}

static int dofile (lua_State *L, const char *name) {
  // FORMERLY int status = luaL_loadfile(L, name) || docall(L, 0, 1);

  debug_fprintf(stderr,"dofile %s\n", name);
  int status =  luaL_loadfile(L, name);
  debug_fprintf(stderr, "loadfile stat %d\n", status); 
  
  if (status == 0) { 
    debug_fprintf(stderr, "calling docall()\n");
    status = docall(L, 0, 1);
    debug_fprintf(stderr, "docall status %d\n", status); 
  }
  return report(L, status);
}

static const char *get_prompt (lua_State *L, int firstline) {
  const char *p;
  lua_getfield(L, LUA_GLOBALSINDEX, firstline ? "_PROMPT" : "_PROMPT2");
  p = lua_tostring(L, -1);
  if (p == NULL) p = (firstline ? LUA_PROMPT : LUA_PROMPT2);
  lua_pop(L, 1);  /* remove global */
  return p;
}


static int incomplete (lua_State *L, int status) {
  if (status == LUA_ERRSYNTAX) {
    size_t lmsg;
    const char *msg = lua_tolstring(L, -1, &lmsg);
    const char *tp = msg + lmsg - (sizeof(LUA_QL("<eof>")) - 1);
    if (strstr(msg, LUA_QL("<eof>")) == tp) {
      lua_pop(L, 1);
      return 1;
    }
  }
  return 0;  /* else... */
}


static int pushline (lua_State *L, int firstline) {
  char buffer[LUA_MAXINPUT];
  char *b = buffer;
  size_t l;
  const char *prmt = get_prompt(L, firstline);
  if (lua_readline(L, b, prmt) == 0)
    return 0;  /* no input */
  l = strlen(b);
  if (l > 0 && b[l-1] == '\n')  /* line ends with newline? */
    b[l-1] = '\0';  /* remove it */
  if (firstline && b[0] == '=')  /* first line starts with `=' ? */
    lua_pushfstring(L, "return %s", b+1);  /* change it to `return' */
  else
    lua_pushstring(L, b);
  lua_freeline(L, b);
  return 1;
}


static int loadline (lua_State *L) {
  int status;
  lua_settop(L, 0);
  if (!pushline(L, 1))
    return -1;  /* no input */
  for (;;) {  /* repeat until gets a complete line */
    status = luaL_loadbuffer(L, lua_tostring(L, 1), lua_strlen(L, 1), "=stdin");
    if (!incomplete(L, status)) break;  /* cannot try to add lines? */
    if (!pushline(L, 0))  /* no more input? */
      return -1;
    lua_pushliteral(L, "\n");  /* add a new line... */
    lua_insert(L, -2);  /* ...between the two lines */
    lua_concat(L, 3);  /* join them */
  }
  lua_saveline(L, 1);
  lua_remove(L, 1);  /* remove line */
  return status;
}


static void dotty (lua_State *L) {
  int status;
  const char *oldprogname = progname;
  progname = NULL;
  while ((status = loadline(L)) != -1) {
    if (status == 0) status = docall(L, 0, 0);
    report(L, status);
    if(repl_stop)
      break;
    if (status == 0 && lua_gettop(L) > 0) {  /* any result to print? */
      lua_getglobal(L, "print");
      lua_insert(L, 1);
      if (lua_pcall(L, lua_gettop(L)-1, 0, 0) != 0)
        l_message(progname, lua_pushfstring(L,
                                            "error calling " LUA_QL("print") " (%s)",
                                            lua_tostring(L, -1)));
    }
  }
  lua_settop(L, 0);  /* clear stack */
  fputs("\n", stdout);
  fflush(stdout);
  progname = oldprogname;
}
#endif

//---
//---

// Argument variables
static bool   arg_is_interactive  = 1;
static char*  arg_script_filename = NULL;
static char*  arg_output_filename = NULL;

// Argument parser function
static error_t parse_chill_arg(int key, char* value, argp_state* state) {
  switch(key) {
  case ARGP_KEY_ARG:
    if(state->arg_num == 0) {
      arg_script_filename = value;
      arg_is_interactive = 0;
    }
    else {
      return ARGP_ERR_UNKNOWN;
    }
  case 'D':
    if(value != NULL) {
      debug_enable(true);
      debug_define(value);
    }
    else {
      debug_enable(true);
    }
    break;
  case 'o':
    arg_output_filename = value;
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static const argp_option chill_options[] = {
  //long-name,  char-name,  value-name,         flags,                  doc,                        group
  {"debug",     'D',        "debug-symbol",     OPTION_ARG_OPTIONAL,    "Define a debug symbol.",   0},
  {"outfile",   'o',        "output-filename",  0,                      "Set output filename.",     0},
  {NULL,        NULL,       NULL,               NULL,                   NULL,                       0}
};

static const argp chill_args {
  chill_options,                // options
  &parse_chill_arg,             // parse function
  NULL,
  NULL,
  NULL,
  NULL,
  NULL
};

//---
// CHiLL program main
// Initialize state and run script or interactive mode
//---
int main( int argc, char* argv[] )
{
  // parse program arguments
  argp_parse(&chill_args, argc, argv, ARGP_IN_ORDER, NULL, NULL);
  
  int fail = 0;
  
#ifdef PYTHON
  // PYTHON version  (python interprets the transformation file )            
  
  //Operate on a single global IR_Code and Loop instance
  ir_code = NULL;
  myloop = NULL;
  omega::initializeOmega();
  debug_fprintf(stderr, "Omega initialized\n");
  
  // Create PYTHON interpreter
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(argv[0]);
  
  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();
  
  /* Add a static module */
  initchill();
  
  if (!arg_is_interactive) {
    /* Run Python interpreter */
    FILE* f = fopen(arg_script_filename, "r");
    if(!f){
      chill_printf("can't open script file \"%s\"\n", argv[1]);
      exit(-1);
    }
    PyRun_SimpleFile(f, arg_script_filename);
    fclose(f);
  }
  if (arg_is_interactive) {
    //---
    // Run a CHiLL interpreter
    //---
    chill_printf("CHiLL v0.2.3 (built on %s)\n", CHILL_BUILD_DATE);
    chill_printf("Copyright (C) 2008 University of Southern California\n");
    chill_printf("Copyright (C) 2009-2016 University of Utah\n");
    //is_interactive = true; // let the lua interpreter know.
    fflush(stdout);
    // TODO: read lines of python code.
    //Not sure if we should set fail from interactive mode
    chill_printf("CHiLL ending...\n");
  }
  //END python setup
#endif
#ifdef LUA
  
  //Create interpreter
  lua_State* L = lua_open();
  globalL = L;
  
  //Initialize the std libs
  luaL_openlibs(L);
  
  //Initialize globals
  register_globals(L);
  
  //Register CHiLL functions
  register_functions(L);
  
  if (!arg_is_interactive) {
    //---
    // Run a CHiLL script from a file
    //---
    
    //Check that the file can be opened
    FILE* f = fopen(arg_script_filename, "r");
    if(!f){
      printf("can't open script file \"%s\"\n", argv[1]);
      exit(-1);
    }
    fclose(f);
    
    debug_fprintf(stderr, "\n*********************evaluating file '%s'\n", arg_script_filename); 
    
    //Evaluate the file
    fail = dofile(L, argv[1]);
    if(!fail){
      debug_fprintf(stderr, "script success!\n");
    }
  }
  if (arg_is_interactive && isatty((int)fileno(stdin))) {
    //---
    // Run a CHiLL interpreter
    //---
    chill_printf("CHiLL v0.2.3 (built on %s)\n", CHILL_BUILD_DATE);
    chill_printf("Copyright (C) 2008 University of Southern California\n");
    chill_printf("Copyright (C) 2009-2016 University of Utah\n");

    is_interactive = true; // let the lua interpreter know.
    dotty(L);
    //Not sure if we should set fail from interactive mode
    chill_printf("CHiLL ending...\n");
  }
#endif
  
  debug_fprintf(stderr, "BIG IF\n"); 
  debug_fprintf(stderr, "fail %d\n", fail);
  
  if (!fail && ir_code != NULL && myloop != NULL && myloop->stmt.size() != 0 && !myloop->stmt[0].xform.is_null()) {
    debug_fprintf(stderr, "big if true\n"); 
#ifdef CUDACHILL
    debug_fprintf(stderr, "CUDACHILL IS DEFINED\n"); 
    int lnum;
    #ifdef PYTHON
    lnum = 0;
    #else
    lnum = get_loop_num( L );
    #endif
    #ifdef FRONTEND_ROSE
    debug_fprintf(stderr, "calling commit_loop()\n"); 
    ((IR_cudaroseCode *)(ir_code))->commit_loop(myloop, lnum);
    ((IR_roseCode*)(ir_code))->finalizeRose();
    #endif
#else
    debug_fprintf(stderr, "CUDACHILL IS NOT DEFINED\n"); 
    int lnum_start;
    int lnum_end;
    #ifdef PYTHON
    lnum_start = get_loop_num_start();
    lnum_end = get_loop_num_end();
    debug_fprintf(stderr, "calling ROSE code gen?    loop num %d - %d\n", lnum_start, lnum_end);
    #else
    lnum_start = get_loop_num_start(L);
    lnum_end = get_loop_num_end(L);
    debug_fprintf(stderr, "calling ROSE code gen?    loop num %d - %d\n", lnum_start, lnum_end);
    #endif
    
    finalize_loop(lnum_start, lnum_end);
    #ifdef FRONTEND_ROSE
    ((IR_roseCode*)(ir_code))->finalizeRose();
    #endif
#endif
    delete ir_code;
  }
  else     debug_fprintf(stderr, "big if FALSE\n"); 

#ifdef PYTHON
  Py_Finalize();
#endif
#ifdef LUA
  lua_close(L);
#endif
  return 0;
}
