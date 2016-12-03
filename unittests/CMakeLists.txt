add_subdirectory(basic_tests)

set(UTD_PATH ${CMAKE_BINARY_DIR}/tests/UTdriver.py)

add_custom_command(OUTPUT ${UTD_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/UTdriver.py ${UTD_PATH}
    DEPENDS UnitTests
    )

add_custom_target(check-all
    COMMAND ${UTD_PATH} ${CMAKE_BINARY_DIR}/tests/unittests
    DEPENDS ${UTD_PATH}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )

