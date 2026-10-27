add_library(caelestia-scheme STATIC
    schemenode.cpp
    schemeimpl.cpp
)
add_include(caelestia-scheme)
target_link_libraries(caelestia-scheme PUBLIC caelestia-parser PRIVATE caelestia::caelestia)
