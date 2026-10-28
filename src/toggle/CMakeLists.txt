add_library(caelestia-toggle STATIC
    toggle.cpp
)
add_include(caelestia-toggle)
target_link_libraries(caelestia-toggle PUBLIC caelestia-parser)
