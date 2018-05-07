message(STATUS "Using frontend option \"${FRONTEND}\"")
if (${FRONTEND} STREQUAL "Rose")
    set(FRONTEND_DEFINITIONS "FRONTEND_ROSE" CACHE INTERNAL "Frontend options")

    add_chill_module(chill_parser
            parser.cpp
            rose.cpp)

    chill_module_link_libraries(chill_parser
            ${Rose_LIBS} ${RoseBoost_LIBS}
            )

    target_include_directories(chill_parser
            PRIVATE ${Rose_INCLUDE_DIRS})
elseif (${FRONTEND} STREQUAL "Clang")
    set(FRONTEND_DEFINITIONS "FRONTEND_CLANG" CACHE INTERNAL "Frontend options")

    add_definitions(${LLVM_DEFINITIONS})

    if (${LLVM_VERSION_MAJOR} GREATER 5)
        set(llvm_libs LLVM)
    else()
        llvm_map_components_to_libnames(llvm_libs bitreader mcparser option profiledata)
    endif()

    add_chill_module(chill_parser
            parser.cpp
            clang.cpp
            )

    chill_module_link_libraries(chill_parser
            ${CLANG_LIBS}
            ${llvm_libs}
            tinfo
            )

    target_include_directories(chill_parser
            PRIVATE ${LLVM_INCLUDE_DIRS})
endif ()
