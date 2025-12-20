#pragma once

#include "penta/harmony/ChordAnalyzer.h"
#include "penta/harmony/ScaleDetector.h"
#include "penta/harmony/VoiceLeading.h"
#include "penta/common/RTTypes.h"

#include <array>
#include <memory>
#include <vector>

namespace penta::harmony {

// Forward declarations
struct Config;
struct Chord;
struct Scale;
struct Note;

/**
 * HarmonyEngine - Integrated harmony analysis coordinator
 * 
 * Provides real-time chord detection, scale analysis, and voice leading suggestions.
 */
class HarmonyEngine {
public:
    explicit HarmonyEngine(const Config& config);
    ~HarmonyEngine();

    // Process MIDI notes for harmony analysis
    void processNotes(const Note* notes, size_t count) noexcept;

    // Get current analysis results
    Chord getCurrentChord() const noexcept { return currentChord_; }
    Scale getCurrentScale() const noexcept { return currentScale_; }

    // Voice leading suggestions
    std::vector<Note> suggestVoiceLeading(
        const Chord& targetChord,
        const std::vector<Note>& currentVoices
    ) const noexcept;

    // Configuration
    void updateConfig(const Config& config);

    // History tracking
    std::vector<Chord> getChordHistory(size_t maxCount) const;
    std::vector<Scale> getScaleHistory(size_t maxCount) const;

private:
    void updateChordAnalysis() noexcept;
    void updateScaleDetection() noexcept;

    Config config_;
    std::unique_ptr<ChordAnalyzer> chordAnalyzer_;
    std::unique_ptr<ScaleDetector> scaleDetector_;
    std::unique_ptr<VoiceLeading> voiceLeading_;

    std::array<uint8_t, 128> activeNotes_;  // MIDI note velocities
    std::array<bool, 12> pitchClassSet_;   // Active pitch classes

    Chord currentChord_;
    Scale currentScale_;
};

} // namespace penta::harmony
