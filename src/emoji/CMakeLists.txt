add_library(caelestia-emoji STATIC
    emoji.cpp
    emoji.hpp
)
target_link_libraries(caelestia-emoji PUBLIC caelestia-parser)

if(CAELESTIA_DEV_COMMANDS)
    target_compile_definitions(caelestia-emoji PRIVATE CAELESTIA_DEV_COMMANDS)
endif()
