add_library(caelestia-install STATIC
    install.cpp
    install.hpp
)
target_link_libraries(caelestia-install PUBLIC caelestia-parser)
