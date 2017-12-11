#include "chill_io.hh"

#include <cstdio>
#include <argp.h>

#include "chillmodule.hh" // Python wrapper functions for CHiLL

// Argument variables
static bool   arg_is_interactive  = true;
static char*  arg_script_filename = nullptr;
static char*  arg_output_filename = nullptr;

// Argument parser function
static error_t parse_chill_arg(int key, char* value, argp_state* state) {
  switch(key) {
  case ARGP_KEY_ARG:
    if(state->arg_num == 0) {
      arg_script_filename = value;
      arg_is_interactive = false;
    }
    else {
      return ARGP_ERR_UNKNOWN;
    }
    break;
  case 'D':
    if(value != nullptr) {
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
    chill_printf("CHiLL v%s (built on %s)\n", CHILL_BUILD_VERSION, CHILL_BUILD_DATE);
    chill_printf("Copyright (C) 2008 University of Southern California\n");
    chill_printf("Copyright (C) 2009-2017 University of Utah\n");
    //is_interactive = true; // let the lua interpreter know.
    fflush(stdout);
    is_interactive = true;
    PyRun_InteractiveLoop(stdin, "-");
    //Not sure if we should set fail from interactive mode
    chill_printf("CHiLL ending...\n");
  }
  //END python setup

  debug_fprintf(stderr, "BIG IF\n"); 

  if (ir_code != NULL && myloop != NULL && myloop->stmt.size() != 0 && !myloop->stmt[0].xform.is_null()) {
    debug_fprintf(stderr, "big if true\n"); 
#ifdef CUDACHILL
    debug_fprintf(stderr, "CUDACHILL IS DEFINED\n"); 
    ((IR_cudaChillCode*)ir_code)->commit_loop(myloop, 0);
#else
    debug_fprintf(stderr, "CUDACHILL IS NOT DEFINED\n"); 
    int lnum_start;
    int lnum_end;
    lnum_start = get_loop_num_start();
    lnum_end = get_loop_num_end();

    finalize_loop(lnum_start, lnum_end);
#endif
    delete ir_code;
  }
  else     debug_fprintf(stderr, "big if FALSE\n");

  Py_Finalize();

  return 0;
}
