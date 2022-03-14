add_library(la INTERFACE)

target_include_directories(la INTERFACE .)
set_property(TARGET la PROPERTY CXX_STANDARD 20)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION /usr/local/include)

