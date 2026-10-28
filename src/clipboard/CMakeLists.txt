add_library(caelestia-clipboard STATIC
    clipnode.cpp
    clipimpl.cpp
)
add_include(caelestia-clipboard)
target_link_libraries(caelestia-clipboard PUBLIC caelestia-parser)
