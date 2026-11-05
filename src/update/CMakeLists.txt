add_library(caelestia-update STATIC
    update.cpp
    update.hpp
)
target_link_libraries(caelestia-update PUBLIC caelestia-parser)
