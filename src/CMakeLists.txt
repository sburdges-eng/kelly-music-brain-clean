# Core C++ library

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
    osc/OSCHub.cpp
    osc/RTMessageQueue.cpp
)

# =============================================================================
# MidiKompanion Plugin Sources (Agent 1, 3, 4 ownership)
# =============================================================================

set(MIDIKOMPANION_SOURCES
    # Project management (Agent 1)
    project/ProjectManager.cpp
    
    # MIDI export (Agent 1)
    midi/MidiExporter.cpp
    
    # ML infrastructure (Agent 3)
    ml/ONNXInference.cpp
    ml/NodeMLMapper.cpp
    ml/MultiModelProcessor.cpp
    
    # Audio analysis (Agent 4)
    audio/AudioAnalyzer.cpp
    
    # Plugin (Agent 1 + Agent 5)
    plugin/plugin_processor.cpp
    plugin/plugin_editor.cpp
)

set(MIDIKOMPANION_HEADERS
    # Project management
    project/ProjectManager.h
    
    # MIDI export
    midi/MidiExporter.h
    
    # ML infrastructure
    ml/ONNXInference.h
    ml/NodeMLMapper.h
    ml/MultiModelProcessor.h
    
    # Audio analysis
    audio/AudioAnalyzer.h
    
    # Plugin
    plugin/plugin_processor.h
    plugin/plugin_editor.h
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
    oscpack
    readerwriterqueue
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

# =============================================================================
# MidiKompanion Library (JUCE-based)
# =============================================================================

# Only build if JUCE is available
if(TARGET juce::juce_audio_processors)
    add_library(midikompanion_lib STATIC
        ${MIDIKOMPANION_SOURCES}
        ${MIDIKOMPANION_HEADERS}
    )
    
    target_include_directories(midikompanion_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
    )
    
    target_link_libraries(midikompanion_lib PUBLIC
        juce::juce_audio_processors
        juce::juce_audio_basics
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
        juce::juce_gui_basics
    )
    
    # ONNX Runtime (optional - for ML inference)
    # Agent 3: Add ONNX Runtime dependency when available
    option(MIDIKOMPANION_HAS_ONNX "Enable ONNX Runtime for ML inference" OFF)
    
    if(MIDIKOMPANION_HAS_ONNX)
        find_package(onnxruntime QUIET)
        if(onnxruntime_FOUND)
            target_link_libraries(midikompanion_lib PUBLIC onnxruntime::onnxruntime)
            target_compile_definitions(midikompanion_lib PUBLIC MIDIKOMPANION_HAS_ONNX=1)
            message(STATUS "ONNX Runtime enabled for MidiKompanion")
        else()
            message(WARNING "ONNX Runtime not found - ML features will use stub mode")
            target_compile_definitions(midikompanion_lib PUBLIC MIDIKOMPANION_HAS_ONNX=0)
        endif()
    else()
        target_compile_definitions(midikompanion_lib PUBLIC MIDIKOMPANION_HAS_ONNX=0)
        message(STATUS "ONNX Runtime disabled - ML features will use stub mode")
    endif()
    
    set_target_properties(midikompanion_lib PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()
