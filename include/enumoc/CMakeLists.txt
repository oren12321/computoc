add_library(enumoc INTERFACE)
target_include_directories(enumoc INTERFACE ${CMAKE_SOURCE_DIR}/include)

set_property(TARGET enumoc PROPERTY CXX_STANDARD 20)

if (WIN32)
    file(GLOB_RECURSE ENUMOC_SRC CONFIGURE_DEPENDS "*.h")
    add_library(enumoc_ ${ENUMOC_SRC})
    set_target_properties(enumoc_ PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(enumoc_ enumoc)
    set_property(TARGET enumoc_ PROPERTY CXX_STANDARD 20)
endif()

