# Core C++ library

# macOS SDK compatibility fix - Define function-style availability macros
# These must be defined before any system headers are included
# We define them as function-style macros (with ...) that expand to nothing
if(APPLE)
    # Note: CMAKE cannot pass function-style macros via -D, so we rely on
    # JUCE's juce_StandardHeader.h which is included first via juce_core
    # The main CMakeLists.txt sets MAC_OS_X_VERSION_MIN_REQUIRED which helps
endif()

# Readerwriterqueue: header-only library with no CMakeLists.txt
# Must explicitly create INTERFACE target and ensure SOURCE_DIR is populated
include(FetchContent)
if(NOT TARGET readerwriterqueue)
    FetchContent_Declare(
        readerwriterqueue
        GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
        GIT_TAG v1.0.6
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(readerwriterqueue)
    
    # Explicitly populate SOURCE_DIR for header-only library
    FetchContent_GetProperties(readerwriterqueue)

    if(NOT TARGET readerwriterqueue)
        add_library(readerwriterqueue INTERFACE)
        target_include_directories(readerwriterqueue INTERFACE
            ${readerwriterqueue_SOURCE_DIR}
        )
    endif()
endif()

set(PENTA_CORE_SOURCES
    # Common utilities
    common/RTMemoryPool.cpp
    common/RTLogger.cpp
    
    # Harmony engine
    harmony/ChordAnalyzer.cpp
    harmony/ChordAnalyzerSIMD.cpp
    harmony/ScaleDetector.cpp
    harmony/VoiceLeading.cpp
    harmony/HarmonyEngine.cpp
    
    # Groove analysis
    groove/OnsetDetector.cpp
    groove/TempoEstimator.cpp
    groove/RhythmQuantizer.cpp
    groove/GrooveEngine.cpp
    
    # Diagnostics
    diagnostics/PerformanceMonitor.cpp
    diagnostics/AudioAnalyzer.cpp
    diagnostics/DiagnosticsEngine.cpp
    
    # OSC communication
    osc/OSCServer.cpp
    osc/OSCClient.cpp
        osc/OSCMessage.cpp
        osc/OSCClient.cpp
        osc/OSCHub.cpp
        osc/RTMessageQueue.cpp

    # ML interface
    ml/MLInterface.cpp

    # Mixer engine
    mixer/MixerEngine.cpp
)

set(PENTA_CORE_HEADERS
    ${PROJECT_SOURCE_DIR}/include/penta/common/RTMemoryPool.h
    ${PROJECT_SOURCE_DIR}/include/penta/common/RTLogger.h
    ${PROJECT_SOURCE_DIR}/include/penta/common/RTTypes.h
    
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/ChordAnalyzer.h
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/ScaleDetector.h
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/VoiceLeading.h
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/HarmonyEngine.h
    
    ${PROJECT_SOURCE_DIR}/include/penta/groove/OnsetDetector.h
    ${PROJECT_SOURCE_DIR}/include/penta/groove/TempoEstimator.h
    ${PROJECT_SOURCE_DIR}/include/penta/groove/RhythmQuantizer.h
    ${PROJECT_SOURCE_DIR}/include/penta/groove/GrooveEngine.h
    
    ${PROJECT_SOURCE_DIR}/include/penta/diagnostics/PerformanceMonitor.h
    ${PROJECT_SOURCE_DIR}/include/penta/diagnostics/AudioAnalyzer.h
    ${PROJECT_SOURCE_DIR}/include/penta/diagnostics/DiagnosticsEngine.h
    
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCServer.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCClient.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCMessage.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/RTMessageQueue.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCHub.h

    ${PROJECT_SOURCE_DIR}/include/penta/mixer/MixerEngine.h
    ${PROJECT_SOURCE_DIR}/include/penta/ml/MLInterface.h
)

add_library(penta_core STATIC
    ${PENTA_CORE_SOURCES}
    ${PENTA_CORE_HEADERS}
)

set_target_properties(penta_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(penta_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(penta_core PUBLIC
    readerwriterqueue
    juce::juce_core
    juce::juce_audio_devices
    juce::juce_dsp
    juce::juce_osc
)

# Enable AVX2 SIMD optimizations if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(penta_core PRIVATE -mavx2 -mpopcnt)
    message(STATUS "AVX2 SIMD optimizations enabled")
else()
    message(STATUS "AVX2 not available, using scalar fallback")
endif()

# Platform-specific threading
if(UNIX)
    target_link_libraries(penta_core PUBLIC pthread)
endif()
