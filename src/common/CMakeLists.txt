add_library(caelestia-common STATIC
    ansi.hpp
    hypr.cpp
    notify.cpp
    target.hpp
)
add_include(caelestia-common)
target_link_libraries(caelestia-common PUBLIC Qt::Core Qt::Network Qt::DBus)
