qt_add_executable(caelestia
    main.cpp
)

target_compile_definitions(caelestia PRIVATE
    CAELESTIA_VERSION="${VERSION}"
    CAELESTIA_GIT_REVISION="${GIT_REVISION}"
    CAELESTIA_DISTRIBUTOR="${DISTRIBUTOR}"
)

target_link_libraries(caelestia PRIVATE
    Qt::Core
    caelestia::caelestia
)

install(TARGETS caelestia DESTINATION "${CMAKE_INSTALL_BINDIR}")
