add_library(caelestia-wallpaper STATIC
    wallnode.cpp
    wallimpl.cpp
)
target_link_libraries(caelestia-wallpaper PUBLIC caelestia-parser PRIVATE caelestia::caelestia)
