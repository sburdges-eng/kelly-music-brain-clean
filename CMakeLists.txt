cmake_minimum_required(VERSION 3.27)
project(Kelly VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_DESKTOP "Build desktop GUI application" ON)
option(BUILD_PLUGINS "Build VST3 and CLAP plugins" ON)
option(BUILD_TESTS "Build tests" OFF)
option(ENABLE_TRACY "Enable Tracy profiling" OFF)
option(ENABLE_RTNEURAL "Enable RTNeural for ML inference" OFF)
option(ENABLE_ONNX_RUNTIME "Enable ONNX Runtime for ML inference" OFF)

# CMake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# macOS SDK 26.2+ compatibility fix for __CLOCK_AVAILABILITY
# The SDK header _time.h uses __CLOCK_AVAILABILITY in enum definitions, which requires
# __OSX_AVAILABLE to be defined. We ensure deployment target is set so Availability.h works.
# Fix is implemented in external/JUCE/modules/juce_core/system/juce_StandardHeader.h
# These settings MUST be configured before any targets are created to apply effectively.
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum macOS version")
    # Ensure availability macros are available before system headers
    # Use numeric value (101200 = 10.12 * 10000) since macro constants aren't available at CMake time
    # MAX_ALLOWED defaults to SDK version if not set, so we only set MIN_REQUIRED
    # Note: CMAKE_OSX_DEPLOYMENT_TARGET also sets this automatically, but we set it explicitly
    # to ensure it's available before JUCE headers are processed
    #
    # For SDK 26.2+, we define simple (non-function) availability macros to empty.
    # Function-style macros (__API_AVAILABLE, etc.) are defined in juce_StandardHeader.h
    # because CMake can't pass function-style preprocessor definitions on the command line.
    add_compile_definitions(
        MAC_OS_X_VERSION_MIN_REQUIRED=101200
        __CLOCK_AVAILABILITY=
        __WATCHOS_PROHIBITED=
        __TVOS_PROHIBITED=
        __IOS_PROHIBITED=
    )
endif()
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)

# JUCE setup
add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)

# Readerwriterqueue (lock-free queue) for RT-safe OSC
# Note: This is a header-only library with no CMakeLists.txt, so we must
# explicitly create an INTERFACE target and ensure SOURCE_DIR is populated.
include(FetchContent)
FetchContent_Declare(
    readerwriterqueue
    GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
    GIT_TAG v1.0.6
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(readerwriterqueue)

# Explicitly populate SOURCE_DIR for header-only library without CMakeLists.txt
FetchContent_GetProperties(readerwriterqueue)

if(NOT TARGET readerwriterqueue)
    add_library(readerwriterqueue INTERFACE)
    target_include_directories(readerwriterqueue INTERFACE
        ${readerwriterqueue_SOURCE_DIR}
    )
endif()

# ============================================================================
# Penta-Core Library (C++ RT engines)
# ============================================================================
add_subdirectory(src_penta-core)

# ============================================================================
# Python Bindings (Optional)
# ============================================================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings for penta-core" ON)

if(BUILD_PYTHON_BINDINGS)
    # Find pybind11
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found, building Python bindings")
        add_subdirectory(bindings)
    else()
        message(STATUS "pybind11 not found. Python bindings will not be built.")
        message(STATUS "  Install via: pip install pybind11")
    endif()
endif()

# ============================================================================
# ML Dependencies (Optional)
# ============================================================================

# RTNeural - Header-only neural network library for real-time inference
if(ENABLE_RTNEURAL)
    find_package(RTNeural QUIET)
    if(RTNeural_FOUND)
        message(STATUS "RTNeural enabled: ${RTNeural_INCLUDE_DIRS}")
    else()
        # Try to fetch RTNeural if not found
        message(STATUS "RTNeural not found locally, fetching from GitHub...")
        FetchContent_Declare(
            rtneural
            GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
            GIT_TAG main
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(rtneural)
        set(RTNeural_FOUND TRUE)
        set(RTNeural_INCLUDE_DIRS ${rtneural_SOURCE_DIR})
    endif()
endif()

# ONNX Runtime - Cross-platform inference engine
if(ENABLE_ONNX_RUNTIME)
    find_package(ONNXRuntime QUIET)
    if(ONNXRuntime_FOUND)
        message(STATUS "ONNX Runtime enabled: ${ONNXRuntime_LIBRARIES}")
    else()
        message(STATUS "ONNX Runtime not found. ONNX models will not be available.")
        message(STATUS "  Install ONNX Runtime or set ONNXRuntime_ROOT")
    endif()
endif()

# ============================================================================
# Main Kelly library (shared core)
# ============================================================================
file(GLOB_RECURSE KELLY_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm
)

# Exclude app entry point and plugin sources from the core lib
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/gui/main.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/plugin/")
# NOTE: Previous line 31 had "list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/")"
# which would exclude ALL source files (since all paths contain "/src/"), making
# KELLY_CORE_SOURCES empty. This has been removed. If macro conflicts exist,
# they should be fixed by excluding specific problematic files/directories rather
# than excluding everything.

add_library(KellyCore STATIC ${KELLY_CORE_SOURCES})

# Ensure ObjC++ file (for biometrics) is treated correctly if present
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/biometric/BiometricInput.mm
    PROPERTIES COMPILE_LANGUAGES OBJCXX
)

target_include_directories(KellyCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(KellyCore PUBLIC
    Qt6::Core
    Qt6::Widgets
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_processors
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_dsp
    juce::juce_osc
    readerwriterqueue
)

# ML Dependencies
if(ENABLE_RTNEURAL AND RTNeural_FOUND)
    target_include_directories(KellyCore PUBLIC ${RTNeural_INCLUDE_DIRS})
    target_compile_definitions(KellyCore PUBLIC ENABLE_RTNEURAL)
    message(STATUS "KellyCore: RTNeural inference enabled")
endif()

if(ENABLE_ONNX_RUNTIME AND ONNXRuntime_FOUND)
    target_include_directories(KellyCore PUBLIC ${ONNXRuntime_INCLUDE_DIRS})
    target_link_libraries(KellyCore PUBLIC ${ONNXRuntime_LIBRARIES})
    target_compile_definitions(KellyCore PUBLIC ENABLE_ONNX_RUNTIME)
    message(STATUS "KellyCore: ONNX Runtime inference enabled")
endif()

target_compile_options(KellyCore PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# Kelly GUI Application
if(BUILD_DESKTOP)
    add_executable(KellyApp
        src/gui/main.cpp
        src/gui/main_window.cpp
    )

    set_target_properties(KellyApp PROPERTIES
        AUTOMOC ON
    )

    target_link_libraries(KellyApp PRIVATE
        KellyCore
        Qt6::Core
        Qt6::Widgets
    )
endif()

# Plugin builds
if(BUILD_PLUGINS)
    # VST3/CLAP plugin
    juce_add_plugin(KellyPlugin
        COMPANY_NAME "Kelly"
        PLUGIN_MANUFACTURER_CODE Klly
        PLUGIN_CODE Klp1
        FORMATS VST3 CLAP
        PRODUCT_NAME "Kelly Emotion Processor"
        VST3_CATEGORIES Fx Synth
        CLAP_ID com.kelly.emotion-processor
    )

    target_sources(KellyPlugin PRIVATE
        src/plugin/PluginProcessor.cpp
        src/plugin/PluginEditor.cpp
        src/plugin/PluginState.cpp
    )

    target_link_libraries(KellyPlugin PRIVATE
        KellyCore
        juce::juce_audio_plugin_client
    )
endif()

# Tests
if(BUILD_TESTS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/Catch2/CMakeLists.txt)
        enable_testing()
        add_subdirectory(external/Catch2 EXCLUDE_FROM_ALL)
        add_executable(KellyTests
            tests/cpp/test_emotion_engine.cpp
            tests/cpp/test_midi_pipeline.cpp
            tests/cpp/test_chord_diagnostics.cpp
            tests/cpp/test_ml_pipeline.cpp
        )
        target_link_libraries(KellyTests PRIVATE
            KellyCore
            Catch2::Catch2WithMain
        )
        include(CTest)
        include(Catch2)
        catch_discover_tests(KellyTests)
    else()
        message(WARNING "Catch2 not found; skipping tests. Set BUILD_TESTS=OFF or add external/Catch2.")
    endif()
endif()

# Tracy profiling
if(ENABLE_TRACY)
    add_subdirectory(external/tracy EXCLUDE_FROM_ALL)
    target_link_libraries(KellyCore PUBLIC TracyClient)
    target_compile_definitions(KellyCore PUBLIC TRACY_ENABLE)
endif()

# Installation
set(KELLY_INSTALL_TARGETS KellyCore)
if(BUILD_DESKTOP)
    list(APPEND KELLY_INSTALL_TARGETS KellyApp)
endif()

install(TARGETS ${KELLY_INSTALL_TARGETS}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/
    DESTINATION include/kelly
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install ML models
install(DIRECTORY models/
    DESTINATION share/kelly/models
    FILES_MATCHING 
    PATTERN "*.json"
    PATTERN "*.onnx"
)

# Install Python ML module (optional)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/python/penta_core)
    install(DIRECTORY python/penta_core
        DESTINATION share/kelly/python
        FILES_MATCHING PATTERN "*.py"
    )
endif()
