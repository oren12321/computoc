add_library(math INTERFACE)

target_include_directories(math INTERFACE .)
set_property(TARGET math PROPERTY CXX_STANDARD 20)

