#ifndef BASIC_CHILLMODULE_HH
#define BASIC_CHILLMODULE_HH
// TODO    Python.h defines these and something else does too 
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>

// a C routine that will be called from python
//static PyObject * chill_print_code(PyObject *self, PyObject *args);

//static PyMethodDef ChillMethods[] ; 

#ifndef CUDACHILL
void finalize_loop(int loop_num_start, int loop_num_end);
int get_loop_num_start();
int get_loop_num_end();
#endif

PyMODINIT_FUNC initchill() ;   // pass C methods to python
#endif
