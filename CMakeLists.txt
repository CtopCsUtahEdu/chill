cmake_minimum_required(VERSION 2.8)
project(chill)
set(CHILL_VERSION 0.2.1) # 2.8 doesn't support project version clause

if(NOT DEFINED ROSEHOME)
    message( FATAL_ERROR "ROSEHOME is not set, try use -DROSEHOME" )
endif()
if (NOT DEFINED BOOSTHOME)
    message( FATAL_ERROR "BOOSTHOME is not set, try use -DBOOSTHOME" )
endif()
if (NOT DEFINED IEGENHOME)
    message( FATAL_ERROR "IEGENHOME is not set, try use -DIEGENHOME" )
endif()

find_package(PythonLibs 2.7 REQUIRED)

string(TIMESTAMP build_date "\\\"%m/%d/%Y\\\"")

set(CORE_LIBS m rose rt util omega_basic omega_lib codegen codegen_rose codegen_chill dl parseRel
    boost_date_time boost_filesystem boost_program_options iegenlib isl
    boost_regex boost_system boost_wave boost_iostreams)

set(CORE_SRC
    dep.cc
    irtools.cc
    loop.cc
    loop_basic.cc
    loop_datacopy.cc
    loop_extra.cc
    loop_tile.cc
    loop_unroll.cc
    omegatools.cc
    stencil.cc
    calculator_eval.cc
    )

set(IR_CHILL_SRC
    ir_chill.cc
    ir_rose.cc
    ir_rose_utils.cc
    )

set(PYTHON_SRC
    chillmodule.cc
    )

set(CHILL_AST_SRC
    chill_ast.cc
    )

set(CUDACHILL_SRC
    mem_mapping_utils.cc
    loop_cuda.cc
    loop_cuda_chill.cc
    ir_cudarose.cc
    ir_cuda_rose_utils.cc
    )

set(COMMON_FLAGS "-DCHILL_BUILD_DATE=\"${build_date}\" -DCHILL_BUILD_VERSION=\"\\\"${CHILL_VERSION}\\\"\"")

set(CMAKE_CXX_FLAGS "-DPYTHON -DFRONTEND_ROSE -D_GLIBCXX_USE_CXX11_ABI=0 ${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")

if (DEFINED OMEGAHOME)
    link_directories(${OMEGAHOME}/lib)
    set(OMEGA_INC ${OMEGAHOME}/include)
else()
    set(OMEGA_INC
        ${CMAKE_CURRENT_SOURCE_DIR}/omega/omega_lib/include
        ${CMAKE_CURRENT_SOURCE_DIR}/omega/basic/include
        ${CMAKE_CURRENT_SOURCE_DIR}/omega/code_gen/include
        ${CMAKE_CURRENT_SOURCE_DIR}/omega/code_gen/include/code_gen
        )
endif()

set(CHILL_INC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(BISON)
find_package(FLEX)

FLEX_TARGET(ExprScanner parse_expr.ll ${CMAKE_CURRENT_BINARY_DIR}/parse_expr.yy.cc COMPILE_FLAGS
    "--header-file=${CMAKE_CURRENT_BINARY_DIR}/parse_expr.ll.hh") # Hack to avoid generating header in root
BISON_TARGET(ExprParser parse_expr.yy ${CMAKE_CURRENT_BINARY_DIR}/parse_expr.tab.cc COMPILE_FLAGS "-t -d")
ADD_FLEX_BISON_DEPENDENCY(ExprScanner ExprParser)

include_directories(
    ${CHILL_INC}
    ${CMAKE_CURRENT_BINARY_DIR}
    )

add_library(parseRel
    chill_run_util.cc
    ${FLEX_ExprScanner_OUTPUTS}
    ${BISON_ExprParser_OUTPUTS}
    )

link_directories(${ROSEHOME}/lib ${BOOSTOME}/lib ${IEGENHOME}/lib)

include_directories(
    ${CHILL_INC}
    ${OMEGA_INC}
    ${ROSEHOME}/include
    ${ROSEHOME}/include/rose
    ${BOOSTHOME}/include
    ${IEGENHOME}/include/iegenlib
    ${PYTHON_INCLUDE_DIRS})

add_executable(chill
        chill_run.cc
        ${CORE_SRC}
        ${PYTHON_SRC}
        ${IR_CHILL_SRC}
        ${CHILL_AST_SRC}
        )

target_link_libraries(chill
        ${CORE_LIBS}
        ${PYTHON_LIBRARY})

add_dependencies(chill parseRel omega_basic omega_lib codegen codegen_rose codegen_chill )

install(TARGETS chill
        RUNTIME DESTINATION bin)

add_executable(cuda-chill
        chill_run.cc
        ${CORE_SRC}
        ${PYTHON_SRC}
        ${IR_CHILL_SRC}
        ${CHILL_AST_SRC}
        ${CUDACHILL_SRC}
        )

target_link_libraries(cuda-chill
        ${CORE_LIBS}
        ${PYTHON_LIBRARY})

add_dependencies(cuda-chill parseRel omega_basic omega_lib codegen codegen_rose codegen_chill )

target_compile_definitions(cuda-chill
        PRIVATE
        CUDACHILL)

install(TARGETS cuda-chill
        RUNTIME DESTINATION bin)

add_subdirectory(omega/basic)
add_subdirectory(omega/omega_lib)
add_subdirectory(omega/code_gen)

add_subdirectory(doc)
