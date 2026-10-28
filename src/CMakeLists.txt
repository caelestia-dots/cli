function(add_include target)
    set(include_dir "${CMAKE_CURRENT_BINARY_DIR}/include")
    file(MAKE_DIRECTORY "${include_dir}")
    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME dir_name)
    file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}" "${include_dir}/${dir_name}" SYMBOLIC)
    target_include_directories(${target} PUBLIC "${include_dir}")
endfunction()

add_subdirectory(common)
add_subdirectory(parser)
add_subdirectory(clipboard)
add_subdirectory(emoji)
add_subdirectory(install)
add_subdirectory(record)
add_subdirectory(scheme)
add_subdirectory(screenshot)
add_subdirectory(shell)
add_subdirectory(toggle)
add_subdirectory(update)
add_subdirectory(wallpaper)

qt_add_executable(caelestia
    main.cpp
)

target_link_libraries(caelestia PRIVATE
    Qt::Core
    caelestia::caelestia
    caelestia-parser
    caelestia-clipboard
    caelestia-emoji
    caelestia-install
    caelestia-record
    caelestia-scheme
    caelestia-screenshot
    caelestia-shell
    caelestia-toggle
    caelestia-update
    caelestia-wallpaper
)

install(TARGETS caelestia DESTINATION "${CMAKE_INSTALL_BINDIR}")
