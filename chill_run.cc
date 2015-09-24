#include "chilldebug.h"

// this is a little messy. the Makefile should be able to define one or the other
#ifndef PYTHON
#ifndef LUA
#define LUA
#endif
#endif

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chill_env.hh"

#include "loop.hh"
#include <omega.h>
#include "ir_code.hh"

#ifdef CUDACHILL

#ifdef BUILD_ROSE
#include "loop_cuda_rose.hh"
#include "ir_cudarose.hh"
#elif BUILD_SUIF
#include "loop_cuda.hh"
#include "ir_cudasuif.hh"
#endif

#else

#ifdef BUILD_ROSE
#include "ir_rose.hh"
#elif BUILD_SUIF
#include "ir_suif.hh"
#endif

#endif

#ifdef LUA
#define lua_c //Get the configuration defines for doing an interactive shell
#include <lua.hpp> //All lua includes wrapped in extern "C"
#include "chill_env.hh" // Lua wrapper functions for CHiLL
#elif PYTHON
#include "chillmodule.hh" // Python wrapper functions for CHiLL
#endif

//---
// CHiLL globals
//---
Loop *myloop = NULL;
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
  if (pname) fprintf(stderr, "%s: ", pname);
  fprintf(stderr, "%s\n", msg);
  fflush(stderr); // ? does this do anything ?
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
  DEBUG_PRINT("\ndocall()\n"); 
  int status;
  int base = lua_gettop(L) - narg;  /* function index */
  lua_pushcfunction(L, traceback);  /* push traceback function */
  lua_insert(L, base);  /* put it under chunk and args */
  signal(SIGINT, laction);
  
  DEBUG_PRINT("status = lua_pcall(L, narg, (clear ? 0 : LUA_MULTRET), base);\n"); 
  
  status = lua_pcall(L, narg, (clear ? 0 : LUA_MULTRET), base);
  signal(SIGINT, SIG_DFL);
  lua_remove(L, base);  /* remove traceback function */
  /* force a complete garbage collection in case of errors */
  if (status != 0) lua_gc(L, LUA_GCCOLLECT, 0);
  return status;
}

static int dofile (lua_State *L, const char *name) {
  int status = luaL_loadfile(L, name) || docall(L, 0, 1);
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

//---
// CHiLL program main
// Initialize state and run script or interactive mode
//---
int main( int argc, char* argv[] )
{
  DEBUG_PRINT("%s  main()\n", argv[0]);
  if (argc > 2) {
    fprintf(stderr, "Usage: %s [script_file]\n", argv[0]);
    exit(-1);
  }
  
  int fail = 0;
  
#ifdef PYTHON
  // Create PYTHON interpreter
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(argv[0]);
  
  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();
  
  /* Add a static module */
  initchill();
  
  if (argc == 2) {
/*  #ifdef CUDACHILL --- This code is for translating lua to python before interprating. ---
    //DEBUG_PRINT("\ncalling python\n");
    // file interpretlua.py  has routines to read the lua transformation file
    PyRun_SimpleString("from interpretlua import *");
    //DEBUG_PRINT("DONE calling python import of functions\n\n");
    char pythoncommand[800];
    sprintf(pythoncommand, "\n\ndopytransform(\"%s\")\0", argv[1]);
    //DEBUG_PRINT("in C, running python command '%s'\n", pythoncommand);
    
    PyRun_SimpleString( pythoncommand );
    #else*/
    FILE* f = fopen(argv[1], "r");
    if(!f){
      printf("can't open script file \"%s\"\n", argv[1]);
      exit(-1);
    }
    PyRun_SimpleFile(f, argv[1]);
    fclose(f);
  }
  if (argc == 1) {
    //---
    // Run a CHiLL interpreter
    //---
    printf("CHiLL v0.2.1 (built on %s)\n", CHILL_BUILD_DATE);
    printf("Copyright (C) 2008 University of Southern California\n");
    printf("Copyright (C) 2009-2012 University of Utah\n");
    //is_interactive = true; // let the lua interpreter know.
    fflush(stdout);
    // TODO: read lines of python code.
    //Not sure if we should set fail from interactive mode
    printf("CHiLL ending...\n");
    fflush(stdout);
  }

  //printf("DONE with PyRun_SimpleString()\n");
//  #endif --- endif for CUDACHILL ---
#endif
  //END python setup
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
  
  if (argc == 2) {
    //---
    // Run a CHiLL script from a file
    //---
    
    //Check that the file can be opened
    FILE* f = fopen(argv[1],"r");
    if(!f){
      printf("can't open script file \"%s\"\n", argv[1]);
      exit(-1);
    }
    fclose(f);
    
    DEBUG_PRINT("\n*********************evaluating file '%s'\n", argv[1]); 
    
    //Evaluate the file
    fail = dofile(L, argv[1]);
    if(!fail){
      fprintf(stderr, "script success!\n");
    }
  }
  if (argc == 1 && isatty((int)fileno(stdin))) {
    //---
    // Run a CHiLL interpreter
    //---
    printf("CUDA-CHiLL v0.2.1 (built on %s)\n", CHILL_BUILD_DATE);
    printf("Copyright (C) 2008 University of Southern California\n");
    printf("Copyright (C) 2009-2012 University of Utah\n");
    is_interactive = true; // let the lua interpreter know.
    fflush(stdout);
    dotty(L);
    //Not sure if we should set fail from interactive mode
    printf("CUDA-CHiLL ending...\n");
    fflush(stdout);
  }
#endif
  
  
  if (!fail && ir_code != NULL && myloop != NULL && myloop->stmt.size() != 0 && !myloop->stmt[0].xform.is_null()) {
#ifdef CUDACHILL
    int lnum;
    #ifdef PYTHON
    lnum = 0;
    #else
    lnum = get_loop_num( L );
    #endif
    #ifdef BUILD_ROSE
    ((IR_cudaroseCode *)(ir_code))->commit_loop(myloop, lnum);
    #elif BUILD_SUIF
    ((IR_cudasuifCode *)(ir_code))->commit_loop(myloop, lnum);
    #endif
#else
    int lnum_start;
    int lnum_end;
    #ifdef PYTHON
    lnum_start = get_loop_num_start();
    lnum_end = get_loop_num_end();
    DEBUG_PRINT("calling ROSE code gen?    loop num %d\n", lnum);
    #else
    lnum_start = get_loop_num_start(L);
    lnum_end = get_loop_num_end(L);
    DEBUG_PRINT("calling ROSE code gen?    loop num %d - %d\n", lnum_start, lnum_end);
    #endif
#endif
    #ifdef BUILD_ROSE
    finalize_loop(lnum_start, lnum_end);
    //((IR_roseCode*)(ir_cide))->commit_loop(myloop, lnum);
    ((IR_roseCode*)(ir_code))->finalizeRose();
    //#elif BUILD_SUIF
    //((IR_suifCode*)(ir_code))->commit_loop(myloop, lnum);
    #endif
    delete ir_code;
  }
#ifdef PYTHON
  Py_Finalize();
#endif
#ifdef LUA
  lua_close(L);
#endif
  return 0;
}
