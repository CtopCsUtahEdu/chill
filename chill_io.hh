#ifndef CHILL_IO_HH
#define CHILL_IO_HH

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>

// ----------------------------------------- //
// Stream output (for debugging, and stdout) //
// ----------------------------------------- //

// Normal chill output
#define chill_fprintf(f, ...)               do { fprintf(f, __VA_ARGS__); fflush(f); } while (0)
#define chill_printf(...)                   do { printf(__VA_ARGS__); fflush(stdout); } while (0)

// User error messages
#define chill_error_fprintf(f, ...)         chill_fprintf(f, __VA_ARGS__)
#define chill_error_printf(...)             chill_fprintf(stderr, __VA_ARGS__)

#ifdef DEBUGCHILL

void debug_enable(bool enable = true);
void debug_define(char* symbol);
bool debug_isenabled();
bool debug_isdefined(char* symbol);

#define debug_fprintf(f, ...)               do { if(debug_isenabled()) { chill_fprintf(f, __VA_ARGS__); } } while(0)
#define debug_printf(...)                   do { if(debug_isenabled()) { chill_printf(__VA_ARGS__); } } while (0)
#define debug_cond_fprintf(s, f, ...)       do { if(debug_isdefined(s)) { debug_fprintf(f, __VA_ARGS__); } } while (0)
#define debug_cond_printf(s, ...)           do { if(debug_isdefined(s)) { debug_printf(__VA_ARGS__); } } while (0)

#else

#define debug_enable(...)                   do {} while (0)
#define debug_define(...)                   do {} while (0)
#define debug_fprintf(...)                  do {} while (0)
#define debug_printf(...)                   do {} while (0)

#endif


#endif