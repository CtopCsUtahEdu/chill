#ifndef BASIC_CHILLMODULE_HH
#define BASIC_CHILLMODULE_HH
// TODO    Python.h defines these and something else does too 
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

/*!
 * \file
 * \brief chill interface to python
 */

#include <Python.h>
#include <tuple>

// a C routine that will be called from python
//static PyObject * chill_print_code(PyObject *self, PyObject *args);

//static PyMethodDef ChillMethods[] ; 

#ifdef CUDACHILL
#include "loop_cuda_chill.hh"
#include "ir_cudachill.hh"

typedef LoopCuda loop_t;
#else // not defined(CUDACHILL)
#include "loop.hh"
void finalize_loop(int loop_num_start, int loop_num_end);
int get_loop_num_start();
int get_loop_num_end();

typedef Loop loop_t;
#endif // not defined(CUDACHILL)

//! pass C methods to python
PyMODINIT_FUNC initchill() ;   // pass C methods to python

extern loop_t *myloop;
extern IR_Code *ir_code;
extern bool is_interactive;

#endif
