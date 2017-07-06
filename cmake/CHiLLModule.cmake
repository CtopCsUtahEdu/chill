# setup a CHiLL module that is:
#   A relative independent part of CHiLL
# Usage: add_chill_module(<module_name> <sourcefiles>)
function(add_chill_module mname)
    add_library(${mname} ${ARGN})
    set(${mname}_IS_CHILL_MODULE YES CACHE INTERNAL "INTERNAL: if it is a chill module")
endfunction()

# Specify what EXTERNAL libraries that a module depends upon
# Usage: chill_module_link_libraries(<module_name> <external_libraries>)
function(chill_module_link_libraries mname)
    set(${mname}_LINK_LIBRARIES ${ARGN} CACHE INTERNAL "INTERNAL: the libraries to link")
endfunction()

# Link libraries to a CHiLL executable, if libraries linked are CHiLL's 
#   it will automatically link their external dependencies
# Usage: chill_link_libraries(<executable> <libraries>)
function(chill_link_libraries exename)
    set(LINKED_LIBRARIES)
    foreach(mname ${ARGN})
        set(LINKED_LIBRARIES ${LINKED_LIBRARIES} ${${mname}_LINK_LIBRARIES})
        if (DEFINED ${mname}_IS_CHILL_MODULE)
            add_dependencies(${exename} ${mname})
        endif()
    endforeach(mname)
    set(LINKED_LIBRARIES ${ARGN} ${LINKED_LIBRARIES})
    if (LINKED_LIBRARIES)
        list(REMOVE_DUPLICATES LINKED_LIBRARIES)
    endif()
    target_link_libraries(${exename} ${LINKED_LIBRARIES})
endfunction()
