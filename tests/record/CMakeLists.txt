set(fake_bus "${PROJECT_SOURCE_DIR}/tests/common/fakebus.hpp")
create_test(record start "${fake_bus}")
create_test(record stop "${fake_bus}")
create_test(record open_in_file_manager "${fake_bus}")
target_compile_definitions(test-record-stop PRIVATE
    CAELESTIA_OPEN_IN_FILE_MANAGER_BIN="${CMAKE_INSTALL_FULL_LIBDIR}/caelestia/bin/caelestia-open-in-file-manager"
)
target_compile_definitions(test-record-open_in_file_manager PRIVATE OPEN_IN_FILE_MANAGER_BIN="$<TARGET_FILE:caelestia-open-in-file-manager>")
