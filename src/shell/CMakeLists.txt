add_library(caelestia-shell STATIC
    shellnode.cpp
    shellimpl.cpp
)
add_include(caelestia-shell)
target_link_libraries(caelestia-shell PUBLIC caelestia-parser)
