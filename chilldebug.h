
// a central place to turn on debugging messages

// enable the next line to get lots of output 
//#define DEBUGCHILL

#ifdef DEBUGCHILL 
#define DEBUG_PRINT(args...) fprintf(stderr, args )
#else
#define DEBUG_PRINT(args...)    /* Don't do anything  */
#endif
