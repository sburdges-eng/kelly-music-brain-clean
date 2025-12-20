cmake_minimum_required(VERSION 3.27)
project(Kelly VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_PLUGINS "Build VST3 and CLAP plugins" ON)
option(BUILD_TESTS "Build tests" OFF)
option(ENABLE_TRACY "Enable Tracy profiling" OFF)

# Find packages
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)

# JUCE setup
add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)

# Main Kelly library (shared core)
file(GLOB_RECURSE KELLY_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm
)

# Exclude app entry point and plugin sources from the core lib
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/gui/main.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/plugin/")
# Exclude sources that depend on missing daiw/audio headers for now
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/audio/AudioFile.cpp$")

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
)

target_compile_options(KellyCore PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# Kelly GUI Application
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
install(TARGETS KellyApp KellyCore
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/
    DESTINATION include/kelly
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
