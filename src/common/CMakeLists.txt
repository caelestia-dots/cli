add_library(caelestia-common INTERFACE
    target.hpp
)
add_include(caelestia-common INTERFACE)
target_link_libraries(caelestia-common INTERFACE caelestia-parser)
