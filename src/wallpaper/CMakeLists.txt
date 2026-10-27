add_library(caelestia-wallpaper STATIC
    wallpaper.cpp
    wallpaper.hpp
)
target_link_libraries(caelestia-wallpaper PUBLIC caelestia-parser)
