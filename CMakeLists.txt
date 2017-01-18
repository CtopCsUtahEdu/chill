cmake_minimum_required(VERSION 2.8)
project(chill)
file(STRINGS ${CMAKE_SOURCE_DIR}/CHILL_VERSION CHILL_VERSION) # 2.8 doesn't support project version clause

message(STATUS "Configuring for chill ${CHILL_VERSION}")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(GTest)

if (DEFINED ENV{CHILLENV})
    set(CHILLENV $ENV{CHILLENV})
endif()

find_package(PythonLibs 2.7 REQUIRED)
find_package(Rose REQUIRED)
find_package(RoseBoost REQUIRED)
find_package(IEGen REQUIRED)

string(TIMESTAMP build_date "\\\"%m/%d/%Y\\\"")

set(CORE_LIBS chill_core chill_ir chill_ast chill_io
    m ${Rose_LIBS} rt util omega_basic omega_lib codegen codegen_rose codegen_chill dl parseRel
    ${IEGen_LIBS} isl ${RoseBoost_LIBS})

add_library(chill_io chill_io.cc)

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

add_library(chill_core ${CORE_SRC})

set(IR_CHILL_SRC
    ir_chill.cc
    ir_rose.cc
    ir_rose_utils.cc
    )

add_library(chill_ir ${IR_CHILL_SRC})

set(PYTHON_SRC
    chillmodule.cc
    )

set(CHILL_AST_SRC
    chill_ast.cc
    )

add_library(chill_ast ${CHILL_AST_SRC})

set(CUDACHILL_SRC
    mem_mapping_utils.cc
    loop_cuda.cc
    loop_cuda_chill.cc
    ir_cudarose.cc
    ir_cuda_rose_utils.cc
    )

add_library(chill_cuda ${CUDACHILL_SRC})

add_library(chill_printer printer/src/generic.cpp
                          printer/src/dump.cpp
                          printer/src/cfamily.cpp)

set(COMMON_FLAGS "-DCHILL_BUILD_DATE=\"${build_date}\" -DCHILL_BUILD_VERSION=\"\\\"${CHILL_VERSION}\\\"\"")

set(CMAKE_CXX_FLAGS "-DPYTHON -DFRONTEND_ROSE ${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")

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

set(CHILL_INC ${CMAKE_CURRENT_SOURCE_DIR}
              ${CMAKE_CURRENT_SOURCE_DIR}/printer/include)

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

include_directories(
    ${CHILL_INC}
    ${OMEGA_INC}
    ${Rose_INCLUDE_DIRS}
    ${RoseBoost_INCLUDE_DIRS}
    ${IEGen_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS})

set(chill-libs
    parseRel omega_basic omega_lib codegen codegen_rose codegen_chill
    chill_core chill_ir chill_ast chill_printer chill_io)

add_executable(chill
        chill_run.cc
        ${PYTHON_SRC}
        )

target_link_libraries(chill
        ${CORE_LIBS}
        ${PYTHON_LIBRARY}
        )

add_dependencies(chill ${chill-libs})

install(TARGETS chill
        RUNTIME DESTINATION bin)

add_executable(cuda-chill
        chill_run.cc
        ${PYTHON_SRC}
        )

target_link_libraries(cuda-chill
        chill_cuda
        ${CORE_LIBS}
        ${PYTHON_LIBRARY})

add_dependencies(cuda-chill ${chill-libs} chill_cuda)

target_compile_definitions(cuda-chill
        PRIVATE
        CUDACHILL)

install(TARGETS cuda-chill
        RUNTIME DESTINATION bin)

add_subdirectory(omega/basic)
add_subdirectory(omega/omega_lib)
add_subdirectory(omega/code_gen)

add_subdirectory(docs)
add_subdirectory(unittests)

# setup tests

set(TD_PATH ${CMAKE_BINARY_DIR}/test-chill/Testdriver.py)
set(EE_PATH ${CMAKE_BINARY_DIR}/tests/e2e)

add_custom_command(OUTPUT ${TD_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test-chill ${CMAKE_BINARY_DIR}/test-chill
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/examples ${CMAKE_BINARY_DIR}/tests/e2e
    COMMAND ${CMAKE_BINARY_DIR}/test-chill/make_chill_testcases.sh tests/e2e/chill/testcases tests/e2e/chill/testcases/RIGHTANSWERS
    DEPENDS UnitTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

add_custom_target(check-all
    COMMAND ${CMAKE_COMMAND} -E env "SRCDIR=${CMAKE_BINARY_DIR}" ${TD_PATH} ${CMAKE_BINARY_DIR}/tests/unittests ${CMAKE_BINARY_DIR}/test-chill
    DEPENDS ${TD_PATH} chill
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

