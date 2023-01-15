add_library(erroc INTERFACE)
target_include_directories(erroc INTERFACE ${CMAKE_SOURCE_DIR}/include)

set_property(TARGET erroc PROPERTY CXX_STANDARD 20)

if (WIN32)
    file(GLOB_RECURSE ERROC_SRC CONFIGURE_DEPENDS "*.h")
    add_library(erroc_ ${ERROC_SRC})
    set_target_properties(erroc_ PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(erroc_ erroc)
    set_property(TARGET erroc_ PROPERTY CXX_STANDARD 20)
endif()

