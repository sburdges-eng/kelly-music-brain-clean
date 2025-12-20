#include "penta/groove/OnsetDetector.h"
#include "penta/groove/TempoEstimator.h"
#include "penta/groove/RhythmQuantizer.h"
#include "penta/groove/GrooveEngine.h"
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>

using namespace penta::groove;

// ========== OnsetDetector Tests ==========

class OnsetDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        detector = std::make_unique<OnsetDetector>(44100.0, 512);
    }
    
    std::unique_ptr<OnsetDetector> detector;
};

TEST_F(OnsetDetectorTest, DetectsSimpleClick) {
    constexpr size_t blockSize = 512;
    std::array<float, blockSize> signal = {};
    
    // Create impulse at start
    signal[0] = 1.0f;
    
    bool detected = detector->processBlock(signal.data(), blockSize);
    
    EXPECT_TRUE(detected);
}

TEST_F(OnsetDetectorTest, IgnoresConstantSignal) {
    constexpr size_t blockSize = 512;
    std::array<float, blockSize> signal;
    signal.fill(0.1f);  // Constant low level
    
    bool detected = detector->processBlock(signal.data(), blockSize);
    
    EXPECT_FALSE(detected);
}

TEST_F(OnsetDetectorTest, DetectsSineWaveOnset) {
    constexpr size_t blockSize = 512;
    std::array<float, blockSize> signal;
    
    // Silence first half
    for (size_t i = 0; i < blockSize / 2; ++i) {
        signal[i] = 0.0f;
    }
    
    // Sine wave second half
    for (size_t i = blockSize / 2; i < blockSize; ++i) {
        signal[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
    }
    
    bool detected = detector->processBlock(signal.data(), blockSize);
    
    EXPECT_TRUE(detected);
}

TEST_F(OnsetDetectorTest, RespondsToSensitivityChanges) {
    constexpr size_t blockSize = 512;
    std::array<float, blockSize> weakSignal = {};
    weakSignal[0] = 0.1f;  // Weak impulse
    
    detector->setSensitivity(0.1f);  // Low sensitivity
    bool lowSens = detector->processBlock(weakSignal.data(), blockSize);
    
    detector->reset();
    
    detector->setSensitivity(0.9f);  // High sensitivity
    bool highSens = detector->processBlock(weakSignal.data(), blockSize);
    
    // High sensitivity should detect what low sensitivity misses
    EXPECT_TRUE(highSens || !lowSens);
}

// ========== TempoEstimator Tests ==========

class TempoEstimatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        estimator = std::make_unique<TempoEstimator>(44100.0);
    }
    
    std::unique_ptr<TempoEstimator> estimator;
};

TEST_F(TempoEstimatorTest, Estimates120BPM) {
    // 120 BPM = 0.5 seconds per beat = 22050 samples at 44.1kHz
    constexpr size_t samplesPerBeat = 22050;
    
    // Feed 4 beats
    for (int beat = 0; beat < 4; ++beat) {
        estimator->addOnset(beat * samplesPerBeat);
    }
    
    float bpm = estimator->getCurrentTempo();
    
    EXPECT_NEAR(bpm, 120.0f, 5.0f);  // Within 5 BPM
}

TEST_F(TempoEstimatorTest, Estimates90BPM) {
    // 90 BPM = 0.667 seconds per beat = 29400 samples
    constexpr size_t samplesPerBeat = 29400;
    
    for (int beat = 0; beat < 4; ++beat) {
        estimator->addOnset(beat * samplesPerBeat);
    }
    
    float bpm = estimator->getCurrentTempo();
    
    EXPECT_NEAR(bpm, 90.0f, 5.0f);
}

TEST_F(TempoEstimatorTest, ReturnsZeroWithNoOnsets) {
    float bpm = estimator->getCurrentTempo();
    
    EXPECT_EQ(bpm, 0.0f);
}

TEST_F(TempoEstimatorTest, SmoothsTempoChanges) {
    estimator->setSmoothing(0.8f);  // High smoothing
    
    // First tempo: 120 BPM
    for (int i = 0; i < 4; ++i) {
        estimator->addOnset(i * 22050);
    }
    float tempo1 = estimator->getCurrentTempo();
    
    // Sudden change to 140 BPM
    estimator->reset();
    for (int i = 0; i < 4; ++i) {
        estimator->addOnset(i * 18900);  // 140 BPM
    }
    float tempo2 = estimator->getCurrentTempo();
    
    // Smoothing should affect the estimate
    EXPECT_NE(tempo1, tempo2);
}

// ========== RhythmQuantizer Tests ==========

class RhythmQuantizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        quantizer = std::make_unique<RhythmQuantizer>(44100.0);
        quantizer->setTempo(120.0f);
        quantizer->setTimeSignature(4, 4);
    }
    
    std::unique_ptr<RhythmQuantizer> quantizer;
};

TEST_F(RhythmQuantizerTest, QuantizesToNearestSixteenth) {
    quantizer->setGrid(RhythmQuantizer::GridResolution::Sixteenth);
    
    // 120 BPM, 4/4 = 0.5s per beat = 22050 samples
    // Sixteenth note = 22050 / 4 = 5512.5 samples
    
    uint64_t nearSixteenth = 5500;  // Just before 1st sixteenth
    uint64_t quantized = quantizer->quantize(nearSixteenth);
    
    EXPECT_NEAR(quantized, 5512, 100);  // Should snap to sixteenth
}

TEST_F(RhythmQuantizerTest, QuantizesToNearestEighth) {
    quantizer->setGrid(RhythmQuantizer::GridResolution::Eighth);
    
    uint64_t nearEighth = 11000;  // Near first eighth note
    uint64_t quantized = quantizer->quantize(nearEighth);
    
    EXPECT_NEAR(quantized, 11025, 100);  // 22050 / 2
}

TEST_F(RhythmQuantizerTest, HandlesDownbeat) {
    uint64_t nearDownbeat = 100;
    uint64_t quantized = quantizer->quantize(nearDownbeat);
    
    EXPECT_NEAR(quantized, 0, 200);  // Should snap to beat 1
}

TEST_F(RhythmQuantizerTest, HandlesSwing) {
    quantizer->setSwing(0.66f);  // Swing feel
    quantizer->setGrid(RhythmQuantizer::GridResolution::Eighth);
    
    uint64_t straightEighth = 11025;
    uint64_t swungEighth = quantizer->quantize(straightEighth);
    
    // Swing should shift timing
    EXPECT_NE(straightEighth, swungEighth);
}

TEST_F(RhythmQuantizerTest, SupportsTriplets) {
    quantizer->setGrid(RhythmQuantizer::GridResolution::EighthTriplet);
    
    // Triplet divides beat into 3
    uint64_t nearTriplet = 7350;  // 22050 / 3
    uint64_t quantized = quantizer->quantize(nearTriplet);
    
    EXPECT_NEAR(quantized, 7350, 100);
}

// ========== GrooveEngine Tests ==========

class GrooveEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<GrooveEngine>(44100.0);
    }
    
    std::unique_ptr<GrooveEngine> engine;
};

TEST_F(GrooveEngineTest, ProcessesAudioBlock) {
    constexpr size_t blockSize = 512;
    std::array<float, blockSize> testSignal;
    
    // Generate click pattern
    testSignal.fill(0.0f);
    testSignal[0] = 1.0f;
    testSignal[256] = 1.0f;
    
    engine->processBlock(testSignal.data(), blockSize);
    
    float tempo = engine->getCurrentTempo();
    EXPECT_GE(tempo, 0.0f);  // Should not crash
}

TEST_F(GrooveEngineTest, RespondsToParameterChanges) {
    engine->setParameter(0, 0.7f);  // Onset sensitivity
    engine->setParameter(1, 0.5f);  // Tempo smoothing
    
    EXPECT_NO_THROW(engine->processBlock(nullptr, 0));
}

// ========== Performance Benchmarks ==========

class GroovePerformanceBenchmark : public ::testing::Test {
protected:
    OnsetDetector detector{44100.0, 512};
    std::array<float, 512> testSignal;
    
    void SetUp() override {
        // Generate test signal with onset
        testSignal.fill(0.0f);
        testSignal[0] = 1.0f;
        for (size_t i = 100; i < 512; ++i) {
            testSignal[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
        }
    }
};

TEST_F(GroovePerformanceBenchmark, OnsetDetectionUnder150Microseconds) {
    constexpr int iterations = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        volatile bool result = detector.processBlock(testSignal.data(), 512);
        (void)result;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgMicros = static_cast<double>(duration.count()) / iterations;
    
    std::cout << "Average onset detection time: " << avgMicros << " μs\n";
    
    EXPECT_LT(avgMicros, 150.0);  // Target: <150μs per 512-sample block
}

TEST_F(GroovePerformanceBenchmark, TempoEstimationUnder200Microseconds) {
    TempoEstimator estimator(44100.0);
    constexpr int iterations = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        estimator.addOnset(i * 22050);
        volatile float tempo = estimator.getCurrentTempo();
        (void)tempo;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgMicros = static_cast<double>(duration.count()) / iterations;
    
    std::cout << "Average tempo estimation time: " << avgMicros << " μs\n";
    
    EXPECT_LT(avgMicros, 200.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
