set(CG_SRC
    src/codegen.cc
    src/CG.cc
    src/CG_stringBuilder.cc
    src/CG_stringRepr.cc
    src/CG_utils.cc
    )

set(CODEGEN_ROSE_SRC
    src/CG_roseBuilder.cc
    src/CG_roseRepr.cc
    src/rose_attributes.cc
    )

set(CODEGEN_CHILL_SRC
    src/CG_chillBuilder.cc
    src/CG_chillRepr.cc
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings")

include_directories(
    include
    ${CHILL_INC}
    ${OMEGAROOT}/include
    )

add_library(codegen
    ${CG_SRC}
    )

install(TARGETS codegen
    ARCHIVE DESTINATION lib)
install(DIRECTORY include
    DESTINATION .)

include_directories(
    include
    ${CHILL_INC}
    ${OMEGA_INC}
    ${ROSEHOME}/include
    ${ROSEHOME}/include/rose
    ${BOOSTHOME}/include
    )

add_library(codegen_chill ${CODEGEN_CHILL_SRC})

install(TARGETS codegen_chill
        DESTINATION lib)

include_directories(
    include
    ${CHILL_INC}
    ${OMEGA_INC}
    )

add_library(codegen_rose ${CODEGEN_ROSE_SRC})

install(TARGETS codegen_rose
        DESTINATION lib)

