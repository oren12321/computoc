add_library(math INTERFACE)
target_include_directories(math INTERFACE .)

set_property(TARGET math PROPERTY CXX_STANDARD 20)

if (WIN32)
    file(GLOB_RECURSE MATH_SRC CONFIGURE_DEPENDS "*.h")
    add_library(math_ ${MATH_SRC})
    set_target_properties(math_ PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(math_ math)
endif()

add_subdirectory(memoc)

