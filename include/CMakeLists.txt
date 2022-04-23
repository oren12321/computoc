add_library(math INTERFACE)
target_include_directories(math INTERFACE .)

if (WIN32)
    file(GLOB_RECURSE MATH_SRC CONFIGURE_DEPENDS "*.h")
    add_custom_target(math_ SOURCES ${MATH_SRC})
endif()

set_property(TARGET math PROPERTY CXX_STANDARD 20)

