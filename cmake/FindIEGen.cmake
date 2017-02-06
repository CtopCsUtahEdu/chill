# Find IEGen
# It uses IEGENHOME or CHILLENV to determine where to find IEGen
# It defines the following variables
# IEGen_FOUND        - True if Rose found.
# IEGen_INCLUDE_DIRS - where to find Rose include files
# IEGen_LIBS         - list of Rose libs
# IEGen_BOOST_LIBS   - list of boost libs required by rose

if (DEFINED ENV{IEGENHOME})
    set(IEGENHOME $ENV{IEGENHOME})
    message(STATUS "IEGENHOME is set to ${IEGENHOME}")
endif()

# This ONLY includes the include dir
find_path(IEGen_INCLUDE_DIR iegenlib/iegenlib.h
    HINTS ${IEGENHOME} ${CHILLENV}
    PATHS /usr 
    PATH_SUFFIXES include)

MACRO(FIND_AND_ADD_IEGen_LIB _libname_)
    find_library(IEGen_${_libname_}_LIB ${_libname_}
        HINTS ${IEGENHOME} ${CHILLENV}
        PATHS /usr
        PATH_SUFFIXES lib)
    if (IEGen_${_libname_}_LIB)
        set(IEGen_LIBS ${IEGen_LIBS} ${IEGen_${_libname_}_LIB})
    endif (IEGen_${_libname_}_LIB)
ENDMACRO(FIND_AND_ADD_IEGen_LIB)

FIND_AND_ADD_IEGen_LIB(iegenlib)
FIND_AND_ADD_IEGen_LIB(isl)
FIND_AND_ADD_IEGen_LIB(gmp)

if (IEGen_LIBS AND IEGen_INCLUDE_DIR)
    MESSAGE(STATUS "IEGen libs: " ${IEGen_LIBS})
    set(IEGen_FOUND TRUE)
    set(IEGen_INCLUDE_DIRS ${IEGen_INCLUDE_DIR} ${IEGen_INCLUDE_DIR}/iegenlib)
endif (IEGen_LIBS AND IEGen_INCLUDE_DIR)

if (IEGen_FOUND)
    message(STATUS "Found IEGen: ${IEGen_INCLUDE_DIRS}")
else (IEGen_FOUND)
    if (IEGen_FIND_REQUIRED)
        message(FATAL_ERROR "Could NOT find IEGen, you may set IEGENHOME to the appropriate path.")
    endif (IEGen_FIND_REQUIRED)
endif (IEGen_FOUND)

