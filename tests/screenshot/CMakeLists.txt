set(fake_bus "${PROJECT_SOURCE_DIR}/tests/common/fakebus.hpp")
create_test(screenshot edit "${fake_bus}")
create_test(screenshot local "${fake_bus}")
create_test(screenshot save "${fake_bus}")
create_test(screenshot helper "${fake_bus}")
foreach(name edit save)
    target_compile_definitions(test-screenshot-${name} PRIVATE
        CAELESTIA_SCREENSHOT_HELPER_BIN="${CMAKE_INSTALL_FULL_LIBDIR}/caelestia/bin/caelestia-screenshot-helper"
    )
endforeach()
target_compile_definitions(test-screenshot-helper PRIVATE SCREENSHOT_HELPER_BIN="$<TARGET_FILE:caelestia-screenshot-helper>")
