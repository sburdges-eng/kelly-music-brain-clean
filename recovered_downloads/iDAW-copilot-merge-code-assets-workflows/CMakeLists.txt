cmake_minimum_required(VERSION 3.20)

project(PentaCore VERSION 1.0.0 LANGUAGES CXX)

# C++20 standard for modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(PENTA_BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" ON)
option(PENTA_BUILD_JUCE_PLUGIN "Build JUCE VST3/AU plugins" ON)
option(PENTA_BUILD_TESTS "Build unit tests" ON)
option(PENTA_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(PENTA_ENABLE_LTO "Enable Link-Time Optimization" OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(PENTA_ENABLE_LTO)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
    if(PENTA_ENABLE_SIMD)
        if(MSVC)
            add_compile_options(/arch:AVX2)
        else()
            add_compile_options(-mavx2 -mfma)
        endif()
    endif()
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# External dependencies
add_subdirectory(external)

# Core library
add_subdirectory(src)

# Python bindings
if(PENTA_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()

# JUCE plugins
if(PENTA_BUILD_JUCE_PLUGIN)
    add_subdirectory(plugins)
endif()

# Tests
if(PENTA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(DIRECTORY include/ DESTINATION include)
install(TARGETS penta_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
