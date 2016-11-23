SUBDIRS = omega/omega_lib/obj omega/code_gen/obj
build_date = "\"`date +%m/%d/%Y`\""

AM_CFLAGS=-g
AM_CXXFLAGS=-g
EXTRA_DIST=

## Core Libraries ##
core_libs     = -lcodegen -lomega
core_libs    += -lm -lrose -lrt -lutil -ldl
core_libs    += -lboost_date_time -lboost_filesystem -lboost_program_options
core_libs    += -lboost_regex -lboost_system -lboost_wave -lboost_iostreams

core_libs    += -liegenlib -lisl -lgmp

core_libdirs  = -Lomega/code_gen/obj -Lomega/omega_lib/obj
core_libdirs += -L$(ROSEHOME)/lib -L$(BOOSTHOME)/lib

core_libdirs += -L$(IEGENHOME)/lib
core_libdirs += -L$(IEGENHOME)/../lib/installed/lib



## Core Includes ##
core_includes  = -I$(srcdir)/omega/include
core_includes += -I$(srcdir)/omega/basic/include
core_includes += -I$(srcdir)/omega/code_gen/include
core_includes += -I$(srcdir)/omega/omega_lib/include
core_includes += -I$(ROSEHOME)/include/rose
core_includes += -I$(BOOSTHOME)/include

core_includes += -I$(IEGENHOME)/include/iegenlib
core_includes += -I$(IEGENHOME)/../lib/installed/include


## Source ##
# Core #
core_src  = calculator_eval.cc
core_src += chill_io.cc
core_src += chill_ast.cc
core_src += dep.cc
core_src += irtools.cc
core_src += loop.cc
core_src += loop_basic.cc
core_src += loop_datacopy.cc
core_src += loop_extra.cc
core_src += loop_ghostcells.cc
core_src += loop_stencil.cc
core_src += loop_tile.cc
core_src += loop_unroll.cc
core_src += omegatools.cc
core_src += stencil.cc

# chill #
# cudachill #
cudachill_src  = mem_mapping_utils.cc
cudachill_src += loop_cuda.cc
cudachill_src += loop_cuda_chill.cc
cudachill_src += loop_cuda_rose.cc

# IR #
#  chill
ir_chill_src      = ir_chill.cc
ir_chill_src     += ir_rose.cc
ir_chill_src     += ir_rose_utils.cc

#  cudachill
ir_cudachill_src  = ir_chill.cc
ir_cudachill_src += ir_rose.cc
ir_cudachill_src += ir_rose_utils.cc
ir_cudachill_src += ir_cudarose.cc
ir_cudachill_src += ir_cuda_rose_utils.cc

# runner (interface) #
#  cc
chill_script_runner_src     = 
chill_python_runner_src     = chill_run.cc chill_run_util.cc chillmodule.cc
chill_lua_runner_src        = chill_run.cc chill_run_util.cc chill_env.cc
#  yacc
chill_script_yacc_src       = lex.zz.cc parser.tab.cc
chill_lua_yacc_src          = parse_expr.yy.cc parse_expr.tab.cc
chill_python_yacc_src       = parse_expr.yy.cc parse_expr.tab.cc


## Debug ##
if DEBUG_OPT
debug_flags					= -DDEBUGCHILL
else
debug_flags					= -DNO_CHILL_DEBUG
endif


## Interface ##
# Libs/Source/Flags #
if PYTHON_OPT
runner_libs           = -lpython$(PYTHON_VERSION)
runner_libdirs        = -L$(PYTHON_LIBDIR)
runner_includes       = -I$(PYTHON_INCDIR)
chill_runner_src      = $(chill_python_runner_src) $(chill_python_yacc_src)
cudachill_runner_src  = $(chill_python_runner_src) $(chill_python_yacc_src)
yacc_src              = $(chill_python_yacc_src)
runner_flags          = -DCHILL_BUILD_DATE=$(build_date)
runner_flags         += -DCHILL_BUILD_VERSION=$(CHILL_BUILD_VERSION)
runner_flags         += -DPYTHON
else
if LUA_OPT
runner_libs           = -llua -lpthread -ldl
runner_libdirs        = -L$(LUAHOME)/lib
runner_includes       = -I$(LUAHOME)/include
chill_runner_src      = $(chill_lua_runner_src) $(chill_lua_yacc_src)
cudachill_runner_src  = $(chill_lua_runner_src) $(chill_lua_yacc_src)
yacc_src              = $(chill_lua_yacc_src)
runner_flags          = -DCHILL_BUILD_DATE=$(build_date)
runner_flags         += -DCHILL_BUILD_VERSION=$(CHILL_BUILD_VERSION)
runner_flags         += -DLUA
else
if SCRIPT_OPT
runner_libs           =
runner_libdirs        =
runner_includes       =
chill_runner_src      = $(chill_script_runner_src) $(chill_script_yacc_src)
yacc_src              = $(chill_script_yacc_src)
runner_flags          = -DCHILL_BUILD_DATE=$(build_date)
runner_flags         += -DCHILL_BUILD_VERSION=$(CHILL_BUILD_VERSION)
endif
endif
endif



## Yacc ##
# config
BUILT_SOURCES=$(yacc_src)
AM_YFLAGS=-d -t

# rules
EXTRA_DIST += lex.zz.cc
lex.zz.cc: $(srcdir)/parser.ll parser.tab.cc
	flex++ $(srcdir)/parser.ll

EXTRA_DIST += parser.tab.cc
parser.tab.cc: $(srcdir)/parser.yy
	bison -p zz -t -d $<

EXTRA_DIST += parse_expr.yy.cc
parse_expr.yy.cc: parse_expr.tab.cc $(srcdir)/parse_expr.ll
	flex -o parse_expr.yy.cc $(srcdir)/parse_expr.ll

EXTRA_DIST += parse_expr.tab.yy
parse_expr.tab.cc: $(srcdir)/parse_expr.yy
	bison -t -d $(srcdir)/parse_expr.yy



## Common ##
common_cppflags         = $(core_includes) $(runner_includes)
common_cppflags        += $(runner_flags) $(debug_flags)
common_cppflags        += -DBUILD_ROSE -DFRONTEND_ROSE
common_libs             = $(core_libs) $(runner_libs)
common_libdirs          = $(core_libdirs) $(runner_libdirs)

## Binaries ##
if CUDACHILL_OPT
TESTS=
bin_PROGRAMS=cuda-chill
else
TESTS=
bin_PROGRAMS=chill
endif

cuda_chill_CPPFLAGS     =-DCUDACHILL $(common_cppflags)
cuda_chill_LDADD        =$(common_libs)
cuda_chill_LDFLAGS      =$(common_libdirs)
cuda_chill_SOURCES      =$(core_src) $(cudachill_runner_src) $(cudachill_src) $(ir_cudachill_src)

chill_CPPFLAGS          =$(common_cppflags) $(chill_runner_cppflags)
chill_LDADD             =$(common_libs)
chill_LDFLAGS           =$(common_libdirs)
chill_SOURCES           =$(core_src) $(chill_runner_src) $(chill_src) $(ir_chill_src)

