# External dependencies management

include(FetchContent)

# pybind11 for Python bindings
if(PENTA_BUILD_PYTHON_BINDINGS)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# JUCE framework for audio plugins
if(PENTA_BUILD_JUCE_PLUGIN)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 7.0.9
    )
    set(JUCE_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
    set(JUCE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(JUCE)
endif()

# Create stub oscpack library (actual implementation deferred to Week 6)
add_library(oscpack INTERFACE)
target_include_directories(oscpack INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/oscpack-stub
)

# Create stub readerwriterqueue library  
add_library(readerwriterqueue INTERFACE)
target_include_directories(readerwriterqueue INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/readerwriterqueue-stub
)

# Google Test for unit testing
if(PENTA_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()
