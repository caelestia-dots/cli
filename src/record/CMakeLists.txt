add_library(caelestia-record STATIC
    notify.cpp
    recordnode.cpp
    recordimpl.cpp
    session.cpp
)
add_include(caelestia-record)
target_link_libraries(caelestia-record PUBLIC caelestia-parser caelestia-common PRIVATE caelestia::caelestia)

set(helper_dir "${CMAKE_INSTALL_FULL_LIBDIR}/caelestia/bin")

qt_add_executable(caelestia-open-in-file-manager helper.cpp)
target_link_libraries(caelestia-open-in-file-manager PRIVATE Qt::Core Qt::DBus)
install(TARGETS caelestia-open-in-file-manager DESTINATION "${helper_dir}")

target_compile_definitions(caelestia-record PRIVATE
    CAELESTIA_OPEN_IN_FILE_MANAGER_BIN="${helper_dir}/caelestia-open-in-file-manager"
)
