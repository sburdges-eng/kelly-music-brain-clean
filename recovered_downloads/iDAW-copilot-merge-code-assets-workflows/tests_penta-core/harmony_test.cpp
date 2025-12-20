#include <gtest/gtest.h>
#include <chrono>
#include "penta/harmony/HarmonyEngine.h"
#include "penta/harmony/ChordAnalyzer.h"
#include "penta/harmony/ScaleDetector.h"
#include "penta/harmony/VoiceLeading.h"

using namespace penta;
using namespace penta::harmony;

// ========== ChordAnalyzer Tests ==========

class ChordAnalyzerTest : public ::testing::Test {
protected:
    void SetUp() override {
        analyzer = std::make_unique<ChordAnalyzer>();
    }
    
    std::unique_ptr<ChordAnalyzer> analyzer;
};

TEST_F(ChordAnalyzerTest, RecognizesCMajorTriad) {
    std::array<bool, 12> cMajor = {
        true,  false, false, false, true,  false, // C - D - E
        false, true,  false, false, false, false  // - G - - - -
    };
    
    Chord result = analyzer->analyze(cMajor);
    
    EXPECT_EQ(result.root, 0);        // C
    EXPECT_EQ(result.quality, 0);     // Major
    EXPECT_GT(result.confidence, 0.9f);
}

TEST_F(ChordAnalyzerTest, RecognizesDominantSeventh) {
    std::array<bool, 12> cDom7 = {
        true,  false, false, false, true,  false, // C - - - E -
        false, true,  false, false, true,  false  // - G - - Bb -
    };
    
    Chord result = analyzer->analyze(cDom7);
    
    EXPECT_EQ(result.root, 0);        // C
    EXPECT_EQ(result.quality, 4);     // Dom7
    EXPECT_GT(result.confidence, 0.85f);
}

TEST_F(ChordAnalyzerTest, EmptyInputReturnsZeroConfidence) {
    std::array<bool, 12> empty = {};
    
    Chord result = analyzer->analyze(empty);
    
    EXPECT_EQ(result.confidence, 0.0f);
}

TEST_F(ChordAnalyzerTest, SIMDMatchesScalar) {
    std::array<bool, 12> cMajor = {
        true,  false, false, false, true,  false,
        false, true,  false, false, false, false
    };
    
    Chord scalarResult = analyzer->analyze(cMajor);
    Chord simdResult = analyzer->analyzeSIMD(cMajor);
    
    EXPECT_EQ(scalarResult.root, simdResult.root);
    EXPECT_EQ(scalarResult.quality, simdResult.quality);
    EXPECT_NEAR(scalarResult.confidence, simdResult.confidence, 0.01f);
}

// ========== Performance Benchmarks ==========

class PerformanceBenchmark : public ::testing::Test {
protected:
    ChordAnalyzer analyzer;
    std::array<bool, 12> testPattern = {
        true, false, false, false, true, false,
        false, true, false, false, true, false
    };
};

TEST_F(PerformanceBenchmark, ChordAnalysisUnder50Microseconds) {
    constexpr int iterations = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        volatile Chord result = analyzer.analyze(testPattern);
        (void)result;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgMicros = static_cast<double>(duration.count()) / iterations;
    
    std::cout << "Average chord analysis time: " << avgMicros << " μs\n";
    
    EXPECT_LT(avgMicros, 50.0);  // Target: <50μs per analysis
}

TEST_F(PerformanceBenchmark, SIMDFasterThanScalar) {
    constexpr int iterations = 10000;
    
    // Scalar version
    auto scalarStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        volatile Chord result = analyzer.analyze(testPattern);
        (void)result;
    }
    auto scalarEnd = std::chrono::high_resolution_clock::now();
    
    // SIMD version
    auto simdStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        volatile Chord result = analyzer.analyzeSIMD(testPattern);
        (void)result;
    }
    auto simdEnd = std::chrono::high_resolution_clock::now();
    
    auto scalarDuration = std::chrono::duration_cast<std::chrono::microseconds>(scalarEnd - scalarStart);
    auto simdDuration = std::chrono::duration_cast<std::chrono::microseconds>(simdEnd - simdStart);
    
    double speedup = static_cast<double>(scalarDuration.count()) / simdDuration.count();
    
    std::cout << "SIMD speedup: " << speedup << "x\n";
    
#ifdef __AVX2__
    EXPECT_GT(speedup, 1.5);  // Expect at least 1.5x speedup with AVX2
#else
    EXPECT_NEAR(speedup, 1.0, 0.1);  // Should be same without SIMD
#endif
}

// ========== Original Tests ==========

class HarmonyEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        HarmonyEngine::Config config;
        config.sampleRate = 48000.0;
        config.confidenceThreshold = 0.5f;
        engine = std::make_unique<HarmonyEngine>(config);
    }
    
    std::unique_ptr<HarmonyEngine> engine;
};

TEST_F(HarmonyEngineTest, DetectsCMajorChord) {
    // Create C major triad notes
    std::vector<Note> notes = {
        Note{60, 80},  // C4
        Note{64, 75},  // E4
        Note{67, 70},  // G4
    };
    
    engine->processNotes(notes.data(), notes.size());
    
    const auto& chord = engine->getCurrentChord();
    
    EXPECT_EQ(chord.root, 0);  // C
    EXPECT_GT(chord.confidence, 0.5f);
}

TEST_F(HarmonyEngineTest, DetectsDMinorChord) {
    std::vector<Note> notes = {
        Note{62, 80},  // D4
        Note{65, 75},  // F4
        Note{69, 70},  // A4
    };
    
    engine->processNotes(notes.data(), notes.size());
    
    const auto& chord = engine->getCurrentChord();
    
    EXPECT_EQ(chord.root, 2);  // D
    // Quality 1 should be minor
}

TEST(ChordAnalyzerTest, AnalyzesPitchClassSet) {
    ChordAnalyzer analyzer;
    
    // C major pitch class set
    std::array<bool, 12> pitchClasses = {
        true, false, false, false, true,   // C, E
        false, false, true, false, false,  // G
        false, false
    };
    
    auto chord = analyzer.analyze(pitchClasses);
    
    EXPECT_EQ(chord.root, 0);  // C
    EXPECT_GT(chord.confidence, 0.0f);
}

TEST(ScaleDetectorTest, DetectsCMajorScale) {
    ScaleDetector detector;
    
    // C major scale pitch classes
    std::array<bool, 12> pitchClasses = {
        true, false, true, false, true,    // C, D, E
        true, false, true, false, true,    // F, G, A
        false, true                         // B
    };
    
    auto scale = detector.analyze(pitchClasses);
    
    EXPECT_EQ(scale.tonic, 0);  // C
    EXPECT_GT(scale.confidence, 0.0f);
}
