cmake_minimum_required(VERSION 3.25)

project(caelestia-cli LANGUAGES CXX)

find_package(caelestia REQUIRED)
caelestia_setup()
caelestia_setup_warnings()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-missing-designated-field-initializers)
endif()

find_package(Qt6 REQUIRED COMPONENTS Core)
qt_standard_project_setup()

option(BUILD_TESTING "Build tests" OFF)
option(CAELESTIA_DEV_COMMANDS "Enable commands intended for CI/dev builds only" OFF)

add_subdirectory(src)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

install(FILES completions/caelestia.fish
    DESTINATION "${CMAKE_INSTALL_DATADIR}/fish/vendor_completions.d"
)
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/caelestia-cli")
