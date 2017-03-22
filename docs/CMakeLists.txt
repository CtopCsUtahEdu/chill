# Generating html doc from doxygen output
set(PROJECT_INC_DIRS "${PROJECT_SOURCE_DIR}/chill/include ${PROJECT_SOURCE_DIR}/omegalib/basic/include ${PROJECT_SOURCE_DIR}/omegalib/omega_lib/include ${PROJECT_SOURCE_DIR}/omegalib/code_gen/include ${ROSEHOME}/include ${ROSEHOME}/include/rose ${BOOSTHOME}/include")

find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doxydoc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
endif()

#Generating manual from *.texi files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/manual DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_program(texi2pdf texi2pdf)

if(texi2pdf)
    add_custom_target(man-pdf
        ${texi2pdf} CHiLL.texi -t "@set CHILLVERSION ${CHILL_VERSION}" --output=${CMAKE_CURRENT_BINARY_DIR}/chill.pdf
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/manual"
        COMMENT "Generating manual (pdf) (with texi2pdf)" VERBATIM)
else()
    message (STATUS "texi2pdf not found, pdf manual can not be built")
endif()

find_program(makeinfo makeinfo)

if(makeinfo)
    add_custom_target(man-info
        ${makeinfo} CHiLL.texi -D "CHILLVERSION ${CHILLVERSION}" --output=${CMAKE_CURRENT_BINARY_DIR}/chill.info
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/manual"
        COMMENT "Generating manual (info) (with makeinfo)" VERBATIM)
else()
    message (STATUS "makeinfo not found, info manual can not be built")
endif()
