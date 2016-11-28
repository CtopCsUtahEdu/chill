set(PROJECT_INC_DIRS "${PROJECT_SOURCE_DIR}/chill/include ${PROJECT_SOURCE_DIR}/omegalib/basic/include ${PROJECT_SOURCE_DIR}/omegalib/omega_lib/include ${PROJECT_SOURCE_DIR}/omegalib/code_gen/include ${ROSEHOME}/include ${ROSEHOME}/include/rose ${BOOSTHOME}/include")

find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
endif()
