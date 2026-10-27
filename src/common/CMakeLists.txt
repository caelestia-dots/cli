add_library(caelestia-common INTERFACE
    ansi.hpp
    target.hpp
)
add_include(caelestia-common INTERFACE)
target_link_libraries(caelestia-common INTERFACE Qt::Core)
