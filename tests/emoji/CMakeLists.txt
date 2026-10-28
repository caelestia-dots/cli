create_test(emoji list)

if(CAELESTIA_DEV_COMMANDS)
    create_test(emoji fetch)
    target_compile_definitions(test-emoji-fetch PRIVATE CAELESTIA_DEV_COMMANDS)
endif()
