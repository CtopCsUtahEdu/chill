add_custom_target(UnitTests)

include(ExternalProject)
ExternalProject_Add(googletest
    URL https://github.com/google/googletest/archive/release-1.8.0.zip
    URL_MD5 "adfafc8512ab65fd3cf7955ef0100ff5"
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/gtest;-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    LOG_CONFIGURE 1
    LOG_INSTALL 1
)

set_target_properties(googletest PROPERTIES EXCLUDE_FROM_ALL TRUE)
set(GTEST_ROOT ${CMAKE_BINARY_DIR}/gtest)

function(add_unittest testname)
    include_directories(${GTEST_ROOT}/include)
    link_directories(${GTEST_ROOT}/lib)
    add_executable(${testname} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${testname} ${UTEST_MODULES} chill_io gtest_main gtest pthread)
    set_target_properties(${testname}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/unittests"
    )
    add_dependencies(${testname} googletest ${UTEST_MODULES} chill_io)
    add_dependencies(UnitTests ${testname})
endfunction()
