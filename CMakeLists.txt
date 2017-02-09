cmake_minimum_required(VERSION 3.1)

project(chill)
file(STRINGS ${CMAKE_SOURCE_DIR}/CHILL_VERSION CHILL_VERSION)

set (CMAKE_CXX_STANDARD 11)

message(STATUS "Configuring for chill ${CHILL_VERSION}")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(GTest)
include(CHiLLModule)

if (DEFINED ENV{CHILLENV})
    set(CHILLENV $ENV{CHILLENV})
endif()

if (NOT DEFINED FRONTEND)
    set(FRONTEND "Rose")
endif()

find_package(PythonLibs 2.7 REQUIRED)
find_package(Rose REQUIRED)
find_package(RoseBoost REQUIRED)
find_package(IEGen REQUIRED)

message(STATUS "Using frontend option \"${FRONTEND}\"")
if (${FRONTEND} STREQUAL "Rose")
    set(FRONTEND_DEFINITIONS FRONTEND_ROSE)

    add_chill_module(chill_ir
            ir_chill.cc
            ir_rose.cc
            ir_rose_utils.cc
            )

    chill_module_link_libraries(chill_ir
            ${Rose_LIBS} ${RoseBoost_LIBS}
            )
elseif (${FRONTEND} STREQUAL "Clang")
    set(FRONTEND_DEFINITIONS FRONTEND_CLANG)
    find_package(LLVM REQUIRED CONFIG)
    find_package(Clang REQUIRED)
    add_definitions(${LLVM_DEFINITIONS})

    llvm_map_components_to_libnames(llvm_libs all)

    add_chill_module(chill_ir
            ir_chill.cc
            ir_clang.cc
            ir_rose.cc
            ir_rose_utils.cc
            )

    chill_module_link_libraries(chill_ir
            ${Rose_LIBS} ${RoseBoost_LIBS} ${llvm_libs} ${CLANG_LIBS}
            )

    set(FRONTEND_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Unrecognized frontend option ${FRONTEND}")
endif()

string(TIMESTAMP build_date "\\\"%m/%d/%Y\\\"")

set(CORE_LIBS
    m rt util omega_basic omega_lib ${FRONTEND_LIBS} codegen codegen_chill dl
    ${IEGen_LIBS})

add_chill_module(chill_io 
    chill_io.cc
    )

add_chill_module(chill_core
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

set(PYTHON_SRC
    chillmodule.cc
    )

add_chill_module(chill_ast
    chill_ast.cc
    )

add_chill_module(chill_cuda
    mem_mapping_utils.cc
    loop_cuda.cc
    loop_cuda_chill.cc
    ir_cudarose.cc
    ir_cuda_rose_utils.cc
    )

target_include_directories(chill_cuda
    PRIVATE ${Rose_INCLUDE_DIRS})

add_chill_module(chill_printer
    printer/src/generic.cpp
    printer/src/dump.cpp
    printer/src/cfamily.cpp
    )

set(COMMON_FLAGS "-DCHILL_BUILD_DATE=\"${build_date}\" -DCHILL_BUILD_VERSION=\"\\\"${CHILL_VERSION}\\\"\"")

set(CMAKE_CXX_FLAGS "-DPYTHON ${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")

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

add_chill_module(parseRel
    chill_run_util.cc
    ${FLEX_ExprScanner_OUTPUTS}
    ${BISON_ExprParser_OUTPUTS}
    )

include_directories(
    ${CHILL_INC}
    ${OMEGA_INC}
    ${RoseBoost_INCLUDE_DIRS}
    ${Rose_INCLUDE_DIRS}
    ${IEGen_INCLUDE_DIRS}
    ${FRONTEND_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS})

set(chill-libs
    parseRel omega_basic omega_lib codegen codegen_chill
    chill_core chill_ir chill_ast chill_printer chill_io)

add_executable(chill
    chill_run.cc
    ${PYTHON_SRC}
    )

chill_link_libraries(chill
        chill_io chill_ir chill_ast chill_core parseRel
        ${CORE_LIBS}
        ${PYTHON_LIBRARY}
        )

add_dependencies(chill ${chill-libs} chill_cuda)

target_compile_definitions(chill
        PRIVATE
        ${FRONTEND_DEFINITIONS})

install(TARGETS chill
        RUNTIME DESTINATION bin)

add_executable(cuda-chill
        chill_run.cc
        ${PYTHON_SRC}
        )

chill_link_libraries(cuda-chill
        chill_cuda chill_io chill_ir chill_ast chill_core parseRel
        ${CORE_LIBS}
        ${PYTHON_LIBRARY})

add_dependencies(cuda-chill ${chill-libs} chill_cuda)

target_compile_definitions(cuda-chill
        PRIVATE
        CUDACHILL
        FRONTEND_ROSE)

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

