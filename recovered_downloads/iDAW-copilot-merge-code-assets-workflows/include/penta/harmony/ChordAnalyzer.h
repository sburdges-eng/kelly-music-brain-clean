#pragma once

#include "penta/common/RTTypes.h"
#include <array>

namespace penta::harmony {

/**
 * Real-time chord analysis using pitch class sets
 * Identifies chord quality, root, and inversions
 */
class ChordAnalyzer {
public:
    ChordAnalyzer();
    ~ChordAnalyzer() = default;
    
    // RT-safe: Analyze pitch class set and return chord
    Chord analyze(const std::array<bool, 12>& pitchClassSet) noexcept;
    
    // RT-safe: Update with new pitch class set
    void update(const std::array<bool, 12>& pitchClassSet) noexcept;
    
    // RT-safe: Get current best chord match
    const Chord& getCurrentChord() const noexcept { return currentChord_; }
    
    // SIMD-optimized analysis (AVX2 when available, scalar fallback otherwise)
    Chord analyzeSIMD(const std::array<bool, 12>& pitchClassSet) noexcept;
    
    // Configuration
    void setConfidenceThreshold(float threshold) noexcept;
    void setTemporalSmoothing(float factor) noexcept; // 0.0-1.0
    
private:
    struct ChordTemplate {
        std::array<bool, 12> pattern;
        uint8_t quality;
        const char* name;
    };
    
    float scoreAgainstTemplate(
        const std::array<bool, 12>& pitchClassSet,
        const ChordTemplate& template_,
        uint8_t root
    ) const noexcept;
    
    void findBestMatch(
        const std::array<bool, 12>& pitchClassSet,
        Chord& outChord
    ) noexcept;
    
    // SIMD-optimized implementations
    float scoreAgainstTemplateSIMD(
        const std::array<bool, 12>& pitchClassSet,
        const ChordTemplate& template_,
        uint8_t root
    ) const noexcept;
    
    void findBestMatchSIMD(
        const std::array<bool, 12>& pitchClassSet,
        Chord& outChord
    ) noexcept;
    
    static const std::array<ChordTemplate, 32> kChordTemplates;
    
    Chord currentChord_;
    Chord previousChord_;
    float confidenceThreshold_;
    float temporalSmoothing_;
};

} // namespace penta::harmony
