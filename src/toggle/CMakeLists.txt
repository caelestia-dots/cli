add_library(caelestia-toggle STATIC
    toggle.cpp
    toggle.hpp
)
target_link_libraries(caelestia-toggle PUBLIC caelestia-parser)
