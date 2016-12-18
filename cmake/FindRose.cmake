# Find Rose
# It uses ROSEHOME or generic CHILLENV to determine where to serch for Rose
# It defines the following variables
# Rose_FOUND        - True if Rose found.
# Rose_INCLUDE_DIRS - where to find Rose include files
# Rose_LIBS         - list of Rose libs
# Rose_BOOST_LIBS   - list of boost libs required by rose

if (DEFINED ENV{ROSEHOME})
    set(ROSEHOME $ENV{ROSEHOME})
    message(STATUS "ROSEHOME is set to ${ROSEHOME}")
endif()

find_path(Rose_INCLUDE_DIR rose/rose.h
    HINTS ${ROSEHOME} ${CHILLENV}
    PATHS /usr
    PATH_SUFFIXES include) # This ONLY includes the include dir

MACRO(FIND_AND_ADD_Rose_LIB _libname_)
    find_library(Rose_${_libname_}_LIB ${_libname_}
        HINTS ${ROSEHOME} ${CHILLENV}
        PATHS /usr
        PATH_SUFFIXES lib)
    if (Rose_${_libname_}_LIB)
        set(Rose_LIBS ${Rose_LIBS} ${Rose_${_libname_}_LIB})
    endif (Rose_${_libname_}_LIB)
ENDMACRO(FIND_AND_ADD_Rose_LIB)

FIND_AND_ADD_Rose_LIB(rose)

if (Rose_LIBS AND Rose_INCLUDE_DIR)
    MESSAGE(STATUS "Rose libs: " ${Rose_LIBS})
    set(Rose_FOUND TRUE)
    set(Rose_INCLUDE_DIRS
        ${Rose_INCLUDE_DIR}
        ${Rose_INCLUDE_DIR}/rose)
endif (Rose_LIBS AND Rose_INCLUDE_DIR)

if (Rose_FOUND)
    message(STATUS "Found Rose: ${Rose_INCLUDE_DIRS}")
else (Rose_FOUND)
    if (Rose_FIND_REQUIRED)
        message(FATAL_ERROR "Could NOT find Rose, you may set ROSEHOME to the appropriate path.")
    endif (Rose_FIND_REQUIRED)
endif (Rose_FOUND)

