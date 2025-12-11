# Minimum CMake version requirement
cmake_minimum_required(VERSION 3.16)

# Project definition
project(moda VERSION 1.0.13 LANGUAGES CXX)

# Standard vcpkg settings for modern CMake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Includes a separate file for finding and configuring Boost and GMP dependencies
# include(cmake/dependencies.cmake)

# Includes the source code logic (likely where the add_library or add_executable is)
include(moda/sources.cmake) 

# Sets up the target (executable)
add_library(moda STATIC ${ALL_SOURCES})
set_target_properties(moda PROPERTIES
    DEBUG_POSTFIX d
)
# ... (Target settings for include directories, linking dependencies like Boost, fmt, and GMP)
# ...

# ----------------------------------------------------
# VCPKG SUPPORT
# ----------------------------------------------------
# Includes a sample file intended for vcpkg use. This is often where
# vcpkg_install_cmake is called in an official vcpkg port, but here
# it looks like a sample for users.
# include(vcpkg-sample.cmake)

# Installs the executable and headers
install(TARGETS moda
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/moda
    DESTINATION include
)