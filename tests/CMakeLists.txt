find_package(Qt6 REQUIRED COMPONENTS Test)

function(create_test module name)
    set(target_name "test-${module}-${name}")
    set(exec_name "test_${module}_${name}")

    add_executable(${target_name} "${name}_test.cpp")
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${exec_name})
    target_include_directories(${target_name} PRIVATE "${PROJECT_SOURCE_DIR}/tests")
    target_link_libraries(${target_name} PRIVATE caelestia-${module} ${ARGN} Qt::Test)
    add_test(NAME "${module}_${name}" COMMAND ${exec_name})
endfunction()

add_subdirectory(parser)
add_subdirectory(scheme)
add_subdirectory(wallpaper)
