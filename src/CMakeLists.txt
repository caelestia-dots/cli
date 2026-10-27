function(add_include target)
    set(include_dir "${CMAKE_CURRENT_BINARY_DIR}/include")
    file(MAKE_DIRECTORY "${include_dir}")
    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME dir_name)
    file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}" "${include_dir}/${dir_name}" SYMBOLIC)

    set(scope PUBLIC)
    if(${ARGC} GREATER 1)
        set(scope ${ARGV1})
    endif()
    target_include_directories(${target} ${scope} "${include_dir}")
endfunction()

add_subdirectory(parser)
qt_add_executable(caelestia
    main.cpp
)

target_link_libraries(caelestia PRIVATE
    Qt::Core
    caelestia::caelestia
)

install(TARGETS caelestia DESTINATION "${CMAKE_INSTALL_BINDIR}")
