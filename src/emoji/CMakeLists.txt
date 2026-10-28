add_library(caelestia-emoji STATIC
    emojinode.cpp
    emojiimpl.cpp
)
add_include(caelestia-emoji)
target_link_libraries(caelestia-emoji PUBLIC caelestia-parser)

qt_add_resources(caelestia-emoji "emojis"
    PREFIX "/emoji"
    BASE data
    FILES data/emojis.txt
)

if(CAELESTIA_DEV_COMMANDS)
    target_compile_definitions(caelestia-emoji PRIVATE
        CAELESTIA_DEV_COMMANDS
        CAELESTIA_EMOJI_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/emojis.txt"
    )
endif()
