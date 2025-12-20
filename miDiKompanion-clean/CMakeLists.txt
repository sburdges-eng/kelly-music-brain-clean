cmake_minimum_required(VERSION 3.22)
project(KellyMidiCompanion VERSION 2.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE - Use local copy if available, otherwise fetch from GitHub
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/JUCE")
    set(JUCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/JUCE")
    add_subdirectory(${JUCE_DIR} ${CMAKE_BINARY_DIR}/JUCE EXCLUDE_FROM_ALL)
    message(STATUS "Using local JUCE from external/JUCE")
else()
    include(FetchContent)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.4
    )
    FetchContent_MakeAvailable(JUCE)
    message(STATUS "Fetching JUCE from GitHub")
endif()

# Plugin target
juce_add_plugin(KellyMidiCompanion
    COMPANY_NAME "Kelly Project"
    PLUGIN_MANUFACTURER_CODE Klly
    PLUGIN_CODE Kmcp

    # Plugin formats
    FORMATS AU VST3 Standalone

    # This is a MIDI effect, not an audio processor
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT TRUE

    PRODUCT_NAME "Kelly MIDI Companion"
    PLUGIN_DESCRIPTION "Therapeutic MIDI generation through emotion mapping. Transforms emotional states (valence, arousal, intensity) into musical MIDI patterns using a 216-node emotion thesaurus and intentional rule-breaking system."

    # Bundle IDs
    BUNDLE_ID "com.kellyproject.midicompanion"
    AU_MAIN_TYPE "kAudioUnitType_MIDIProcessor"
)

# Source files
target_sources(KellyMidiCompanion
    PRIVATE
        # Plugin core
        src/plugin/PluginProcessor.cpp
        src/plugin/PluginEditor.cpp
        src/plugin/PluginState.cpp

        # Emotion engine
        src/engine/EmotionThesaurus.cpp
        src/engine/EmotionThesaurusLoader.cpp
        src/engine/WoundProcessor.cpp
        src/engine/RuleBreakEngine.cpp
        src/engine/IntentPipeline.cpp
        src/engine/KellyBrain.cpp
        src/engine/VADCalculator.cpp
        src/engine/QuantumEmotionalField.cpp
        src/engine/EmotionToMusicMapper.cpp

        # MIDI generation
        src/midi/ChordGenerator.cpp
        src/midi/GrooveEngine.cpp
        src/midi/MidiBuilder.cpp
        src/midi/MidiGenerator.cpp
        src/midi/InstrumentSelector.cpp

        # Common utilities
        src/common/EQPresetManager.cpp
        src/common/PathResolver.cpp

        # Project management
        src/project/ProjectManager.cpp

        # UI (VERSION 3.0.00 components)
        src/ui/EmotionWorkstation.cpp
        src/ui/GenerateButton.cpp
        src/ui/KellyLookAndFeel.cpp
        src/ui/EmotionWheel.cpp
        src/ui/AIGenerationDialog.cpp
        src/ui/ChordDisplay.cpp
        src/ui/EmotionRadar.cpp
        src/ui/MusicTheoryPanel.cpp
        src/ui/PianoRollPreview.cpp
        src/ui/TooltipComponent.cpp
        src/ui/WorkstationPanel.cpp
        src/ui/LyricDisplay.cpp
        src/ui/VocalControlPanel.cpp

        # Algorithm engines (VERSION 3.0.00)
        src/engines/ArrangementEngine.cpp
        src/engines/BassEngine.cpp
        src/engines/CounterMelodyEngine.cpp
        src/engines/DynamicsEngine.cpp
        src/engines/FillEngine.cpp
        src/engines/MelodyEngine.cpp
        src/engines/PadEngine.cpp
        src/engines/RhythmEngine.cpp
        src/engines/StringEngine.cpp
        src/engines/TensionEngine.cpp
        src/engines/TransitionEngine.cpp
        src/engines/VariationEngine.cpp
        src/engines/VoiceLeading.cpp
        src/engines/GrooveEngine.cpp
        src/engines/DrumGrooveEngine.cpp
        # Note: src/engines/GrooveEngine.cpp implements GroovePatternEngine (drum patterns)
        # src/midi/GrooveEngine.cpp implements GrooveEngine (MIDI note processing)
        # Both are needed - different namespaces prevent conflict

        # Voice synthesis (v2.0)
        src/voice/VoiceSynthesizer.cpp
        src/voice/VocoderEngine.cpp
        src/voice/LyricGenerator.cpp
        src/voice/PhonemeConverter.cpp
        src/voice/PitchPhonemeAligner.cpp
        src/voice/ExpressionEngine.cpp
        src/voice/ProsodyAnalyzer.cpp
        src/voice/RhymeEngine.cpp
        src/voice/LyriSync.cpp

        # Biometric input (v2.0)
        src/biometric/BiometricInput.cpp

        # ML Inference (v2.0+)
        src/ml/RTNeuralProcessor.cpp
        src/ml/InferenceThreadManager.cpp
        src/ml/MLFeatureExtractor.cpp
        src/ml/MIDITokenizer.cpp
        src/ml/DDSPProcessor.cpp
        src/ml/MultiModelProcessor.cpp
        src/ml/MLBridge.cpp
)

# Include directories
target_include_directories(KellyMidiCompanion
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# JUCE modules
target_compile_definitions(KellyMidiCompanion
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(KellyMidiCompanion
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# =============================================================================
# RTNeural - Real-time neural network inference library
# =============================================================================
option(ENABLE_RTNEURAL "Enable RTNeural library for ML inference" ON)

if(ENABLE_RTNEURAL)
    # Fetch RTNeural if not available locally
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/RTNeural")
        set(RTNEURAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/RTNeural")
        add_subdirectory(${RTNEURAL_DIR} ${CMAKE_BINARY_DIR}/RTNeural EXCLUDE_FROM_ALL)
        message(STATUS "Using local RTNeural from external/RTNeural")
    else()
        # Auto-fetch RTNeural from GitHub if not found locally
        include(FetchContent)
        FetchContent_Declare(
            RTNeural
            GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
            GIT_TAG main  # Use main branch (stable)
        )
        FetchContent_MakeAvailable(RTNeural)
        message(STATUS "✅ Auto-fetching RTNeural from GitHub (main branch)")
        message(STATUS "   To use local copy: git clone https://github.com/jatinchowdhury18/RTNeural.git external/RTNeural")
    endif()

    # Add compile definition
    target_compile_definitions(KellyMidiCompanion PRIVATE ENABLE_RTNEURAL=1)
    message(STATUS "RTNeural enabled - ML inference available")
else()
    message(STATUS "RTNeural disabled - ML inference unavailable")
endif()

# Link RTNeural if enabled
# Note: RTNeural is header-only, but we link the target to get include directories
if(ENABLE_RTNEURAL)
    # RTNeural exports different target names depending on version
    # Try RTNeural first, fallback to RTNeural::RTNeural if needed
    if(TARGET RTNeural)
        target_link_libraries(KellyMidiCompanion PRIVATE RTNeural)
    elseif(TARGET RTNeural::RTNeural)
        target_link_libraries(KellyMidiCompanion PRIVATE RTNeural::RTNeural)
    endif()
    message(STATUS "RTNeural library linked to KellyMidiCompanion")
endif()

# Note: Code signing is handled by build_and_install.sh script
# We don't sign during build to avoid Gatekeeper issues
# The install script properly removes quarantine and signs after installation

# =============================================================================
# Python-C++ Bridge (Optional - requires pybind11)
# =============================================================================

option(BUILD_PYTHON_BRIDGE "Build Python-C++ bridge module" OFF)

if(BUILD_PYTHON_BRIDGE)
    # Find Python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # Use local pybind11 if available, otherwise fetch from GitHub
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11")
        set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11")
        add_subdirectory(${pybind11_DIR})
        message(STATUS "Using local pybind11 from external/pybind11")
    else()
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
        message(STATUS "Fetching pybind11 from GitHub")
    endif()

    # Python module target
    # Note: Excluding files that require JUCE (EmotionThesaurusLoader, PathResolver, EQPresetManager)
    # The bridge uses EmotionThesaurus which has embedded defaults, so file loading isn't needed
    pybind11_add_module(kelly_bridge
        src/bridge/kelly_bridge.cpp

        # Emotion engine (JUCE-free components)
        src/engine/EmotionThesaurus.cpp
        src/engine/WoundProcessor.cpp
        src/engine/RuleBreakEngine.cpp
        src/engine/IntentPipeline.cpp

        # Note: MIDI generation files excluded - bridge focuses on emotion processing
        # Users can generate MIDI via the main plugin or Python ML framework
    )

    # Include directories
    target_include_directories(kelly_bridge
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_INCLUDE_DIRS}
    )

    # Link Python only - bridge doesn't need JUCE
    # EmotionThesaurus has embedded defaults, so file loading isn't required
    target_link_libraries(kelly_bridge
        PRIVATE
            ${Python3_LIBRARIES}
    )

    # Compile definitions
    # Define KELLY_BRIDGE_NO_JUCE to disable JUCE-dependent features
    target_compile_definitions(kelly_bridge
        PRIVATE
            VERSION_INFO="${PROJECT_VERSION}"
            KELLY_BRIDGE_NO_JUCE=1
    )

    # Set output directory to python/ directory
    set_target_properties(kelly_bridge PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        OUTPUT_NAME kelly_bridge
    )

    message(STATUS "Python-C++ bridge enabled")
    message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "  Python version: ${Python3_VERSION}")
    message(STATUS "  Module will be built to: ${CMAKE_CURRENT_SOURCE_DIR}/python")
endif()

# =============================================================================
# Unit Tests (Optional - requires Google Test)
# =============================================================================

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    add_subdirectory(tests)
    message(STATUS "Unit tests enabled")
    message(STATUS "  Run tests with: ctest or ./tests/KellyTests")
endif()
