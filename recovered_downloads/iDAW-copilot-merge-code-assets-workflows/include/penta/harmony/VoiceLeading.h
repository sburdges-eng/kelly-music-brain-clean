#pragma once

#include "penta/common/RTTypes.h"
#include <vector>

namespace penta::harmony {

/**
 * Voice leading optimizer using minimal motion principles
 * Implements smooth voice transitions between chords
 */
class VoiceLeading {
public:
    struct Config {
        float maxVoiceDistance;  // Max semitones to move
        float parallelPenalty;    // Penalty for parallel motion
        float contraryBonus;      // Bonus for contrary motion
        bool allowVoiceCrossing;
        
        Config()
            : maxVoiceDistance(12.0f)
            , parallelPenalty(5.0f)
            , contraryBonus(2.0f)
            , allowVoiceCrossing(false)
        {}
    };
    
    explicit VoiceLeading(const Config& config = Config{});
    ~VoiceLeading() = default;
    
    // RT-safe: Find optimal voice leading from current to target chord
    std::vector<Note> findOptimalVoicing(
        const Chord& targetChord,
        const std::vector<Note>& currentVoices,
        uint8_t targetOctave = 4
    ) const noexcept;
    
    // RT-safe: Calculate voice leading cost
    float calculateCost(
        const std::vector<Note>& from,
        const std::vector<Note>& to
    ) const noexcept;
    
    // Configuration
    void updateConfig(const Config& config) noexcept;
    
private:
    struct VoicingCandidate {
        std::vector<Note> voices;
        float cost;
        
        bool operator<(const VoicingCandidate& other) const {
            return cost < other.cost;
        }
    };
    
    void generateVoicingCandidates(
        const Chord& chord,
        uint8_t octave,
        std::vector<VoicingCandidate>& candidates
    ) const noexcept;
    
    float calculateMotionCost(
        uint8_t fromPitch,
        uint8_t toPitch
    ) const noexcept;
    
    Config config_;
};

} // namespace penta::harmony
