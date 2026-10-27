add_library(caelestia-screenshot STATIC
    screenshot.cpp
    screenshot.hpp
)
target_link_libraries(caelestia-screenshot PUBLIC caelestia-parser caelestia-common)
