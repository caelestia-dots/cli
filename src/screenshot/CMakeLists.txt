add_library(caelestia-screenshot STATIC
    notify.cpp
    notify.hpp
    screenshot.cpp
    screenshot.hpp
)
add_include(caelestia-screenshot)
target_link_libraries(caelestia-screenshot PUBLIC caelestia-parser caelestia-common Qt::Gui PRIVATE caelestia::caelestia)

set(helper_dir "${CMAKE_INSTALL_FULL_LIBDIR}/caelestia/bin")

qt_add_executable(caelestia-screenshot-helper helper.cpp)
target_link_libraries(caelestia-screenshot-helper PRIVATE caelestia-screenshot caelestia::caelestia)
install(TARGETS caelestia-screenshot-helper DESTINATION "${helper_dir}")

target_compile_definitions(caelestia-screenshot PRIVATE
    CAELESTIA_SCREENSHOT_HELPER_BIN="${helper_dir}/caelestia-screenshot-helper"
)
