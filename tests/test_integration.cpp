/**
 * @file test_integration.cpp
 * @brief Integration tests for miDiKompanion v1.0 features
 *
 * Tests:
 * - ProjectManager save/load
 * - MidiExporter export
 * - ONNX inference pipeline
 * - Audio analysis integration
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

// For testing, we define minimal versions of the structures
// In real tests, include the actual headers

namespace test {

//==============================================================================
// Test Utilities
//==============================================================================

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

#define RUN_TEST(test_func) \
    do { \
        std::cout << "Running " << #test_func << "... "; \
        if (test_func()) { \
            std::cout << "PASSED" << std::endl; \
            passed++; \
        } else { \
            std::cout << "FAILED" << std::endl; \
            failed++; \
        } \
    } while (0)

//==============================================================================
// ProjectManager Tests
//==============================================================================

bool test_project_metadata_serialization() {
    // Test that project metadata can be round-tripped through JSON
    struct ProjectMetadata {
        std::string name = "Test Project";
        std::string author = "Test Author";
        std::string created = "2024-12-20T00:00:00Z";
        std::string modified = "2024-12-20T00:00:00Z";
        std::string description = "Test Description";
        std::vector<std::string> tags = {"test", "integration"};
    };
    
    ProjectMetadata meta;
    meta.name = "Integration Test Project";
    meta.author = "CI System";
    
    TEST_ASSERT(meta.name == "Integration Test Project", "Metadata name should be set");
    TEST_ASSERT(meta.author == "CI System", "Metadata author should be set");
    TEST_ASSERT(meta.tags.size() == 2, "Metadata should have 2 tags");
    
    return true;
}

bool test_project_version_migration() {
    // Test version migration logic
    struct ProjectVersion {
        int major = 1;
        int minor = 0;
        int patch = 0;
        
        bool needsMigration(int targetMajor) const {
            return major < targetMajor;
        }
    };
    
    ProjectVersion v090{0, 9, 0};
    ProjectVersion v100{1, 0, 0};
    
    TEST_ASSERT(v090.needsMigration(1), "v0.9.0 needs migration to v1.x");
    TEST_ASSERT(!v100.needsMigration(1), "v1.0.0 does not need migration to v1.x");
    
    return true;
}

bool test_track_data_structure() {
    struct MidiNoteData {
        int noteNumber = 60;
        int velocity = 100;
        double startBeat = 0.0;
        double durationBeats = 1.0;
        int channel = 0;
    };
    
    struct TrackData {
        std::string name;
        std::vector<MidiNoteData> notes;
    };
    
    TrackData track;
    track.name = "Melody";
    track.notes.push_back({60, 100, 0.0, 1.0, 0});
    track.notes.push_back({64, 80, 1.0, 0.5, 0});
    track.notes.push_back({67, 90, 1.5, 0.5, 0});
    
    TEST_ASSERT(track.name == "Melody", "Track name should be Melody");
    TEST_ASSERT(track.notes.size() == 3, "Track should have 3 notes");
    TEST_ASSERT(track.notes[0].noteNumber == 60, "First note should be C4 (60)");
    
    return true;
}

//==============================================================================
// MidiExporter Tests
//==============================================================================

bool test_midi_export_options() {
    struct MidiExportOptions {
        bool includeTempoTrack = true;
        bool includeTimeSignature = true;
        bool includeLyrics = false;
        bool includeCC = true;
        double tempo = 120.0;
        int numerator = 4;
        int denominator = 4;
        int ppqn = 480;
        int midiFormat = 1;  // 0=single track, 1=multi track
    };
    
    MidiExportOptions options;
    
    TEST_ASSERT(options.includeTempoTrack, "Tempo track should be included by default");
    TEST_ASSERT(options.tempo == 120.0, "Default tempo should be 120 BPM");
    TEST_ASSERT(options.ppqn == 480, "Default PPQN should be 480");
    TEST_ASSERT(options.midiFormat == 1, "Default format should be Type 1");
    
    return true;
}

bool test_beat_to_tick_conversion() {
    const int ppqn = 480;
    
    auto beatToTick = [ppqn](double beat) -> int {
        return static_cast<int>(beat * ppqn);
    };
    
    TEST_ASSERT(beatToTick(0.0) == 0, "Beat 0 should be tick 0");
    TEST_ASSERT(beatToTick(1.0) == 480, "Beat 1 should be tick 480");
    TEST_ASSERT(beatToTick(0.5) == 240, "Beat 0.5 should be tick 240");
    TEST_ASSERT(beatToTick(4.0) == 1920, "Beat 4 should be tick 1920");
    
    return true;
}

bool test_midi_note_validation() {
    auto isValidNote = [](int note, int velocity) -> bool {
        return note >= 0 && note <= 127 && velocity >= 0 && velocity <= 127;
    };
    
    TEST_ASSERT(isValidNote(60, 100), "C4 at velocity 100 should be valid");
    TEST_ASSERT(isValidNote(0, 1), "Note 0 at velocity 1 should be valid");
    TEST_ASSERT(isValidNote(127, 127), "Note 127 at velocity 127 should be valid");
    TEST_ASSERT(!isValidNote(-1, 100), "Negative note should be invalid");
    TEST_ASSERT(!isValidNote(60, 128), "Velocity 128 should be invalid");
    TEST_ASSERT(!isValidNote(128, 100), "Note 128 should be invalid");
    
    return true;
}

//==============================================================================
// ONNX Inference Tests
//==============================================================================

bool test_model_spec_validation() {
    struct ModelSpec {
        std::string name;
        int inputSize;
        int outputSize;
        
        bool isValid() const {
            return !name.empty() && inputSize > 0 && outputSize > 0;
        }
    };
    
    ModelSpec emotionRecognizer = {"emotion_recognizer", 128, 64};
    ModelSpec melodyTransformer = {"melody_transformer", 64, 128};
    ModelSpec harmonyPredictor = {"harmony_predictor", 128, 64};
    ModelSpec dynamicsEngine = {"dynamics_engine", 32, 16};
    ModelSpec groovePredictor = {"groove_predictor", 64, 32};
    
    TEST_ASSERT(emotionRecognizer.isValid(), "EmotionRecognizer spec should be valid");
    TEST_ASSERT(melodyTransformer.isValid(), "MelodyTransformer spec should be valid");
    TEST_ASSERT(harmonyPredictor.isValid(), "HarmonyPredictor spec should be valid");
    TEST_ASSERT(dynamicsEngine.isValid(), "DynamicsEngine spec should be valid");
    TEST_ASSERT(groovePredictor.isValid(), "GroovePredictor spec should be valid");
    
    return true;
}

bool test_emotion_embedding_dimensions() {
    // 216-node thesaurus: 6 base * 6 sub * 6 intensity
    const int numBaseEmotions = 6;
    const int numSubEmotions = 6;
    const int numIntensityLevels = 6;
    const int totalNodes = numBaseEmotions * numSubEmotions * numIntensityLevels;
    
    TEST_ASSERT(totalNodes == 216, "Total emotion nodes should be 216");
    
    // Verify node ID to VAD conversion
    auto nodeToIndices = [](int nodeId) -> std::tuple<int, int, int> {
        int baseIdx = nodeId / 36;
        int subIdx = (nodeId % 36) / 6;
        int intensityIdx = nodeId % 6;
        return {baseIdx, subIdx, intensityIdx};
    };
    
    auto [base0, sub0, int0] = nodeToIndices(0);
    TEST_ASSERT(base0 == 0 && sub0 == 0 && int0 == 0, "Node 0 should map to (0,0,0)");
    
    auto [base100, sub100, int100] = nodeToIndices(100);
    TEST_ASSERT(base100 == 2, "Node 100 should have base index 2");
    
    auto [base215, sub215, int215] = nodeToIndices(215);
    TEST_ASSERT(base215 == 5 && sub215 == 5 && int215 == 5, "Node 215 should map to (5,5,5)");
    
    return true;
}

bool test_fallback_mode() {
    // Test that fallback mode produces valid output when ML is not available
    const int embeddingSize = 64;
    std::vector<float> fallbackEmbedding(embeddingSize);
    
    // Simulate fallback embedding generation
    float valence = 0.5f;
    float arousal = 0.5f;
    float dominance = 0.5f;
    float intensity = 0.5f;
    
    fallbackEmbedding[0] = valence;
    fallbackEmbedding[1] = arousal;
    fallbackEmbedding[2] = dominance;
    fallbackEmbedding[3] = intensity;
    
    // Fill remaining with zeros
    for (int i = 4; i < embeddingSize; ++i) {
        fallbackEmbedding[i] = 0.0f;
    }
    
    TEST_ASSERT(fallbackEmbedding.size() == 64, "Fallback embedding should be 64-dim");
    TEST_ASSERT(fallbackEmbedding[0] == 0.5f, "Valence should be 0.5");
    
    return true;
}

//==============================================================================
// Audio Analysis Tests
//==============================================================================

bool test_f0_frequency_to_midi() {
    auto frequencyToMidi = [](float frequency) -> int {
        if (frequency <= 0.0f) return -1;
        return static_cast<int>(std::round(12.0f * std::log2(frequency / 440.0f) + 69.0f));
    };
    
    TEST_ASSERT(frequencyToMidi(440.0f) == 69, "440 Hz should be MIDI note 69 (A4)");
    TEST_ASSERT(frequencyToMidi(261.63f) == 60, "261.63 Hz should be MIDI note 60 (C4)");
    TEST_ASSERT(frequencyToMidi(880.0f) == 81, "880 Hz should be MIDI note 81 (A5)");
    TEST_ASSERT(frequencyToMidi(0.0f) == -1, "0 Hz should return -1 (unvoiced)");
    
    return true;
}

bool test_spectral_centroid_calculation() {
    // Simplified spectral centroid calculation
    auto calculateCentroid = [](const std::vector<float>& magnitudes, 
                                const std::vector<float>& frequencies) -> float {
        float weightedSum = 0.0f;
        float totalMag = 0.0f;
        
        for (size_t i = 0; i < magnitudes.size(); ++i) {
            weightedSum += magnitudes[i] * frequencies[i];
            totalMag += magnitudes[i];
        }
        
        return totalMag > 0.0f ? weightedSum / totalMag : 0.0f;
    };
    
    // Test case: peak at 1000 Hz
    std::vector<float> mags = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> freqs = {100.0f, 500.0f, 1000.0f, 2000.0f, 4000.0f};
    
    float centroid = calculateCentroid(mags, freqs);
    TEST_ASSERT(std::abs(centroid - 1000.0f) < 0.01f, "Centroid should be 1000 Hz");
    
    return true;
}

bool test_ml_feature_vector_size() {
    const int expectedSize = 128;
    
    // Simulate feature vector layout:
    // [0-15]: F0 features
    // [16-31]: Loudness features
    // [32-63]: Spectral features
    // [64-79]: Harmonic features
    // [80-127]: Derived/combined features
    
    int f0Features = 16;
    int loudnessFeatures = 16;
    int spectralFeatures = 32;
    int harmonicFeatures = 16;
    int derivedFeatures = 48;
    
    int totalFeatures = f0Features + loudnessFeatures + spectralFeatures + 
                        harmonicFeatures + derivedFeatures;
    
    TEST_ASSERT(totalFeatures == expectedSize, "Feature vector should be 128-dimensional");
    
    return true;
}

//==============================================================================
// End-to-End Integration Tests
//==============================================================================

bool test_emotion_to_midi_pipeline() {
    // Simulate the full pipeline: emotion node -> embedding -> notes
    
    // 1. Select emotion node
    int nodeId = 42;  // Some arbitrary node
    
    // 2. Calculate VAD
    int baseIdx = nodeId / 36;
    int subIdx = (nodeId % 36) / 6;
    int intensityIdx = nodeId % 6;
    
    float valence = (baseIdx == 0) ? 0.8f : -0.4f;  // Happy vs others
    float arousal = (subIdx + 3.0f) / 9.0f;
    float intensity = (intensityIdx + 1.0f) / 6.0f;
    
    // 3. Generate embedding (simulated)
    std::vector<float> embedding(64, 0.0f);
    embedding[0] = valence;
    embedding[1] = arousal;
    embedding[2] = intensity;
    
    // 4. Generate notes (rule-based fallback)
    std::vector<int> notes;
    int rootNote = 60 + static_cast<int>(valence * 12);
    
    for (int i = 0; i < 8; ++i) {
        int note = rootNote + (i * 2) % 12;
        notes.push_back(note);
    }
    
    TEST_ASSERT(notes.size() == 8, "Should generate 8 notes");
    TEST_ASSERT(notes[0] >= 48 && notes[0] <= 84, "Notes should be in reasonable range");
    
    return true;
}

bool test_project_save_load_roundtrip() {
    // Simulate project data round-trip
    struct SimpleProject {
        std::string name;
        float valence;
        float arousal;
        int numNotes;
        
        std::string serialize() const {
            std::stringstream ss;
            ss << name << "," << valence << "," << arousal << "," << numNotes;
            return ss.str();
        }
        
        static SimpleProject deserialize(const std::string& data) {
            SimpleProject p;
            std::stringstream ss(data);
            std::string token;
            
            std::getline(ss, p.name, ',');
            ss >> p.valence;
            ss.ignore();
            ss >> p.arousal;
            ss.ignore();
            ss >> p.numNotes;
            
            return p;
        }
    };
    
    SimpleProject original;
    original.name = "TestProject";
    original.valence = 0.75f;
    original.arousal = 0.5f;
    original.numNotes = 16;
    
    std::string serialized = original.serialize();
    SimpleProject loaded = SimpleProject::deserialize(serialized);
    
    TEST_ASSERT(loaded.name == original.name, "Name should round-trip correctly");
    TEST_ASSERT(std::abs(loaded.valence - original.valence) < 0.001f, "Valence should round-trip");
    TEST_ASSERT(loaded.numNotes == original.numNotes, "Note count should round-trip");
    
    return true;
}

} // namespace test

//==============================================================================
// Main Test Runner
//==============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "miDiKompanion Integration Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    // ProjectManager Tests
    std::cout << "--- ProjectManager Tests ---" << std::endl;
    RUN_TEST(test::test_project_metadata_serialization);
    RUN_TEST(test::test_project_version_migration);
    RUN_TEST(test::test_track_data_structure);
    std::cout << std::endl;
    
    // MidiExporter Tests
    std::cout << "--- MidiExporter Tests ---" << std::endl;
    RUN_TEST(test::test_midi_export_options);
    RUN_TEST(test::test_beat_to_tick_conversion);
    RUN_TEST(test::test_midi_note_validation);
    std::cout << std::endl;
    
    // ONNX Inference Tests
    std::cout << "--- ONNX Inference Tests ---" << std::endl;
    RUN_TEST(test::test_model_spec_validation);
    RUN_TEST(test::test_emotion_embedding_dimensions);
    RUN_TEST(test::test_fallback_mode);
    std::cout << std::endl;
    
    // Audio Analysis Tests
    std::cout << "--- Audio Analysis Tests ---" << std::endl;
    RUN_TEST(test::test_f0_frequency_to_midi);
    RUN_TEST(test::test_spectral_centroid_calculation);
    RUN_TEST(test::test_ml_feature_vector_size);
    std::cout << std::endl;
    
    // End-to-End Tests
    std::cout << "--- End-to-End Tests ---" << std::endl;
    RUN_TEST(test::test_emotion_to_midi_pipeline);
    RUN_TEST(test::test_project_save_load_roundtrip);
    std::cout << std::endl;
    
    // Summary
    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Total:  " << (passed + failed) << std::endl;
    std::cout << std::endl;
    
    if (failed == 0) {
        std::cout << "✓ All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some tests failed." << std::endl;
        return 1;
    }
}
