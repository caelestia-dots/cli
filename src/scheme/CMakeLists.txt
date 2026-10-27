add_library(caelestia-scheme STATIC
    scheme.cpp
    scheme.hpp
)
target_link_libraries(caelestia-scheme PUBLIC caelestia-parser PRIVATE caelestia::caelestia)
