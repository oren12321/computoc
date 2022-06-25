add_library(computoc INTERFACE)
target_include_directories(computoc INTERFACE ${CMAKE_SOURCE_DIR}/include)

set_property(TARGET computoc PROPERTY CXX_STANDARD 20)

if (WIN32)
    file(GLOB_RECURSE COMPUTOC_CSRC CONFIGURE_DEPENDS "*.h")
    add_library(computoc_ ${COMPUTOC_CSRC})
    set_target_properties(computoc_ PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(computoc_ computoc)
    set_property(TARGET computoc_ PROPERTY CXX_STANDARD 20)
endif()
