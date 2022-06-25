if (DEFINED IN_DOCKER)
    find_package(GTest REQUIRED)
endif()

add_subdirectory(memoc)
add_subdirectory(computoc)

