cmake_minimum_required(VERSION 3.16)

project(moda VERSION 1.0.13 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(moda/sources.cmake)

add_library(moda STATIC ${ALL_SOURCES})

set_target_properties(moda PROPERTIES
    DEBUG_POSTFIX d
)

# Public include folder
target_include_directories(moda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install the library
install(TARGETS moda
    EXPORT modaTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers only
install(DIRECTORY ${CMAKE_SOURCE_DIR}/moda/
    DESTINATION include/moda
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# --- PACKAGE CONFIGURATION ---
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/modaConfig.cmake
    INSTALL_DESTINATION lib/cmake/moda
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/modaConfig.cmake
    DESTINATION lib/cmake/moda
)

install(
    EXPORT modaTargets
    FILE modaTargets.cmake
    NAMESPACE moda::
    DESTINATION lib/cmake/moda
)