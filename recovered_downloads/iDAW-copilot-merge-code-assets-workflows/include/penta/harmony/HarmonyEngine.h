#pragma once

#include "penta/common/RTTypes.h"
#include "penta/harmony/ChordAnalyzer.h"
#include "penta/harmony/ScaleDetector.h"
#include "penta/harmony/VoiceLeading.h"
#include <memory>
#include <vector>

namespace penta::harmony {

/**
 * Main harmony analysis engine
 * Coordinates chord analysis, scale detection, and voice leading
 */
class HarmonyEngine {
public:
    struct Config {
        double sampleRate;
        size_t analysisWindowSize;
        bool enableVoiceLeading;
        bool enableScaleDetection;
        float confidenceThreshold;
        
        Config()
            : sampleRate(kDefaultSampleRate)
            , analysisWindowSize(2048)
            , enableVoiceLeading(true)
            , enableScaleDetection(true)
            , confidenceThreshold(0.5f)
        {}
    };
    
    explicit HarmonyEngine(const Config& config = Config{});
    ~HarmonyEngine();
    
    // Non-copyable, movable
    HarmonyEngine(const HarmonyEngine&) = delete;
    HarmonyEngine& operator=(const HarmonyEngine&) = delete;
    HarmonyEngine(HarmonyEngine&&) noexcept = default;
    HarmonyEngine& operator=(HarmonyEngine&&) noexcept = default;
    
    // RT-safe: Analyze incoming MIDI notes
    void processNotes(const Note* notes, size_t count) noexcept;
    
    // RT-safe: Get current harmonic state
    const Chord& getCurrentChord() const noexcept { return currentChord_; }
    const Scale& getCurrentScale() const noexcept { return currentScale_; }
    
    // RT-safe: Get voice leading suggestions
    std::vector<Note> suggestVoiceLeading(
        const Chord& targetChord,
        const std::vector<Note>& currentVoices
    ) const noexcept;
    
    // Non-RT: Update configuration
    void updateConfig(const Config& config);
    
    // Non-RT: Get analysis history
    std::vector<Chord> getChordHistory(size_t maxCount = 100) const;
    std::vector<Scale> getScaleHistory(size_t maxCount = 100) const;
    
private:
    void updateChordAnalysis() noexcept;
    void updateScaleDetection() noexcept;
    
    Config config_;
    
    std::unique_ptr<ChordAnalyzer> chordAnalyzer_;
    std::unique_ptr<ScaleDetector> scaleDetector_;
    std::unique_ptr<VoiceLeading> voiceLeading_;
    
    Chord currentChord_;
    Scale currentScale_;
    
    std::array<uint8_t, 128> activeNotes_; // Note velocity (0 = off)
    std::array<bool, 12> pitchClassSet_;   // Current pitch classes
};

} // namespace penta::harmony
