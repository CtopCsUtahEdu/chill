#ifndef CHILL_IO_HH
#define CHILL_IO_HH

#include <stdlib.h>

// ----------------------------------------- //
// Stream output (for debugging, and stdout) //
// ----------------------------------------- //

// Normal chill output
#define chill_fprintf(f, ...)               { fprintf(f, __VA_ARGS__); fflush(f); }
#define chill_printf(...)                   { printf(__VA_ARGS__); fflush(stdout); }

// User error messages
#define chill_error_fprintf(f, ...)         chill_fprintf(f, __VA_ARGS__)
#define chill_error_printf(...)             chill_fprintf(stderr, __VA_ARGS__)

#ifdef DEBUGCHILL

void debug_enable(bool enable = true);
void debug_define(char* symbol);
bool debug_isenabled();
bool debug_isdefined(char* symbol);

#define debug_fprintf(f, ...)               if(debug_isenabled()) { chill_fprintf(f, __VA_ARGS__); }
#define debug_printf(...)                   if(debug_isenabled()) { chill_printf(__VA_ARGS__); }
#define debug_cond_fprintf(s, f, ...)       if(debug_isdefined(s)) { debug_fprintf(f, __VA_ARGS__); }
#define debug_cond_printf(s, ...)           if(debug_isdefined(s)) { debug_printf(__VA_ARGS__); }

#else

#define debug_enable(...)
#define debug_define(...)
#define debug_fprintf(...)
#define debug_printf(...)

#endif


#endif
