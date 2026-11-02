add_library(caelestia-common STATIC
    ansi.hpp
    notify.cpp
    target.hpp
)
add_include(caelestia-common)
target_link_libraries(caelestia-common PUBLIC Qt::Core Qt::DBus)
