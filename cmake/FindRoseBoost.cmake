# Find Boost for Rose
# It uses BOOSTHOME or CHILLENV to determine where to find boost
# It defines the following variables
# RoseBoost_FOUND        - True if Rose found.
# RoseBoost_INCLUDE_DIRS - where to find Rose include files
# RoseBoost_LIBS         - list of Rose libs
# RoseBoost_BOOST_LIBS   - list of boost libs required by rose

if (DEFINED ENV{BOOSTHOME})
    set(BOOSTHOME $ENV{BOOSTHOME})
    message(STATUS "BOOSTHOME is set to ${BOOSTHOME}")
endif()

find_path(RoseBoost_INCLUDE_DIR boost/thread.hpp
    HINTS ${BOOSTHOME} ${CHILLENV}
    PATHS /usr
    PATH_SUFFIXES include) # This ONLY includes the include dir

MACRO(FIND_AND_ADD_RoseBoost_LIB _libname_)
    find_library(RoseBoost_${_libname_}_LIB ${_libname_}
        HINTS ${BOOSTHOME} ${CHILLENV}
        PATHS /usr
        PATH_SUFFIXES lib)
    if (RoseBoost_${_libname_}_LIB)
        set(RoseBoost_LIBS ${RoseBoost_LIBS} ${RoseBoost_${_libname_}_LIB})
    endif (RoseBoost_${_libname_}_LIB)
ENDMACRO(FIND_AND_ADD_RoseBoost_LIB)

FIND_AND_ADD_RoseBoost_LIB(boost_date_time)
FIND_AND_ADD_RoseBoost_LIB(boost_filesystem)
FIND_AND_ADD_RoseBoost_LIB(boost_program_options)
FIND_AND_ADD_RoseBoost_LIB(boost_regex)
FIND_AND_ADD_RoseBoost_LIB(boost_system)
FIND_AND_ADD_RoseBoost_LIB(boost_wave)
FIND_AND_ADD_RoseBoost_LIB(boost_iostreams)

if (RoseBoost_LIBS AND RoseBoost_INCLUDE_DIR)
    MESSAGE(STATUS "Boost libs: " ${RoseBoost_LIBS})
    set(RoseBoost_FOUND TRUE)
    set(RoseBoost_INCLUDE_DIRS
        ${RoseBoost_INCLUDE_DIR})
endif (RoseBoost_LIBS AND RoseBoost_INCLUDE_DIR)

if (RoseBoost_FOUND)
    message(STATUS "Found Boost: ${RoseBoost_INCLUDE_DIRS}")
else (RoseBoost_FOUND)
    if (RoseBoost_FIND_REQUIRED)
        message(FATAL_ERROR "Could NOT find Boost, you may set BOOSTHOME to the appropriate path.")
    endif (RoseBoost_FIND_REQUIRED)
endif (RoseBoost_FOUND)

