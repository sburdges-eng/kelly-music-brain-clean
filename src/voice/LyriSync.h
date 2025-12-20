#pragma once

#include "voice/LyricTypes.h"
#include "voice/VoiceSynthesizer.h"
#include <vector>
#include <string>

namespace kelly {

/**
 * LyriSync - Synchronizes lyrics with vocal melody for display/playback.
 *
 * This class handles:
 * - Synchronizing lyrics with vocal melody timing
 * - Handling timing alignment
 * - Supporting real-time updates
 * - Syllable timing calculation
 */
class LyriSync {
public:
    /**
     * Synchronized lyric item with timing information.
     */
    struct SyncItem {
        std::string text;           // Text for this item
        double startBeat;           // Start position in beats
        double duration;            // Duration in beats
        bool isHighlighted;         // Whether this item should be highlighted

        SyncItem() : startBeat(0.0), duration(0.0), isHighlighted(false) {}
    };

    /**
     * Synchronization result with timed lyric items.
     */
    struct SyncResult {
        std::vector<SyncItem> items;  // Synchronized items (words/syllables)
        double totalDuration;         // Total duration in beats
    };

    LyriSync();
    ~LyriSync() = default;

    /**
     * Synchronize lyrics with vocal notes.
     * @param lyrics Lyric structure
     * @param vocalNotes Vocal notes with timing
     * @param bpm Beats per minute
     * @return Synchronization result
     */
    SyncResult synchronize(
        const LyricStructure& lyrics,
        const std::vector<VoiceSynthesizer::VocalNote>& vocalNotes,
        float bpm = 120.0f
    );

    /**
     * Get current sync item at a given beat position.
     * @param syncResult Synchronization result
     * @param currentBeat Current beat position
     * @return Index of current sync item, or -1 if none
     */
    int getCurrentItem(const SyncResult& syncResult, double currentBeat) const;

    /**
     * Calculate syllable durations from tempo.
     * @param syllables Syllables to time
     * @param totalDuration Total duration in beats
     * @return Vector of durations for each syllable
     */
    std::vector<double> calculateSyllableDurations(
        const std::vector<Syllable>& syllables,
        double totalDuration
    ) const;

    /**
     * Handle word stress timing (stressed syllables get more time).
     * @param syllables Syllables with stress information
     * @param baseDuration Base duration per syllable
     * @return Vector of durations adjusted for stress
     */
    std::vector<double> applyStressTiming(
        const std::vector<Syllable>& syllables,
        double baseDuration
    ) const;

    /**
     * Support rubato (tempo variation).
     * @param baseDuration Base duration
     * @param rubatoAmount Rubato amount (0.0 = no rubato, 1.0 = full rubato)
     * @param position Position in phrase (0.0 to 1.0)
     * @return Adjusted duration
     */
    double applyRubato(double baseDuration, float rubatoAmount, float position) const;

    /**
     * Set BPM for timing calculations.
     */
    void setBPM(float bpm) { bpm_ = bpm; }

private:
    float bpm_ = 120.0f;
};

} // namespace kelly
