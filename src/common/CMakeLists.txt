add_library(caelestia-common STATIC
    ansi.hpp
    notify.cpp
    target.hpp
)
add_include(caelestia-common)
target_link_libraries(caelestia-common PUBLIC caelestia-parser Qt::Core Qt::DBus)
