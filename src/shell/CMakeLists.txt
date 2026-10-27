add_library(caelestia-shell STATIC
    shell.cpp
    shell.hpp
)
target_link_libraries(caelestia-shell PUBLIC caelestia-parser)
