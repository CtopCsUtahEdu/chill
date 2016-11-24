set(OMEGAROOT ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

set(BASIC_SRC
    src/ConstString.cc
    src/Link.cc
    )

include_directories(
    include
    )

add_library(omega_basic
    ${BASIC_SRC}
    )

install(TARGETS omega_basic
    ARCHIVE DESTINATION lib)

install(DIRECTORY include
    DESTINATION .)
