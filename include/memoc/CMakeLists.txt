add_library(memoc INTERFACE)
target_include_directories(memoc INTERFACE ${CMAKE_SOURCE_DIR}/include)

set_property(TARGET memoc PROPERTY CXX_STANDARD 20)

if (WIN32)
    file(GLOB_RECURSE MEMOC_SRC CONFIGURE_DEPENDS "*.h")
    add_library(memoc_ ${MEMOC_SRC})
    set_target_properties(memoc_ PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(memoc_ memoc)
    set_property(TARGET memoc_ PROPERTY CXX_STANDARD 20)
endif()

