create_test(emoji list Qt::Gui)

if(CAELESTIA_DEV_COMMANDS)
    create_test(emoji fetch caelestia::caelestia)
    target_compile_definitions(test-emoji-fetch PRIVATE CAELESTIA_DEV_COMMANDS)
endif()
