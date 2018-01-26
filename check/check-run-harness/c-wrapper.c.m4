define(`__define',  `#define   $1')
define(`__include', `#include "$1"')
define(`__undef',   `#undef    $1')

__include(  original_header)

__define(   proc_name           proc_name`'_original)
__include(  original_source)
__undef(    proc_name)

__define(   proc_name           proc_name`'_generated)
__include(  generated_source)
__undef(    proc_name)

undefine(`__define')
undefine(`__include')
undefine(`__undef')

