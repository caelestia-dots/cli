add_library(caelestia-record STATIC
    record.cpp
    record.hpp
)
target_link_libraries(caelestia-record PUBLIC caelestia-parser caelestia-common)
