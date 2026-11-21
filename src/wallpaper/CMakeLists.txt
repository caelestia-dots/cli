add_library(caelestia-wallpaper STATIC
    wallnode.cpp
    wallimpl.cpp
)
add_include(caelestia-wallpaper)
target_link_libraries(caelestia-wallpaper PUBLIC caelestia-parser PRIVATE caelestia::caelestia)
