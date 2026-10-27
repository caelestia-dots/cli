add_library(caelestia-parser STATIC
    parser.cpp
    node.hpp
    spec.hpp
)
add_include(caelestia-parser)
target_link_libraries(caelestia-parser PUBLIC Qt::Core)
