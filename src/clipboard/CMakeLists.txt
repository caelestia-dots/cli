add_library(caelestia-clipboard STATIC
    clipboard.cpp
    clipboard.hpp
)
target_link_libraries(caelestia-clipboard PUBLIC caelestia-parser)
