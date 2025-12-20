#include "penta/harmony/ChordAnalyzer.h"
#include <algorithm>
#include <cmath>

// SIMD intrinsics (AVX2)
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace penta::harmony {

// Comprehensive chord template database (30+ chord types)
const std::array<ChordAnalyzer::ChordTemplate, 32> ChordAnalyzer::kChordTemplates = {{
    // Basic Triads (0-3)
    {{true, false, false, false, true, false, false, true, false, false, false, false}, 0, "Major"},        // C E G
    {{true, false, false, true, false, false, false, true, false, false, false, false}, 1, "Minor"},        // C Eb G
    {{true, false, false, true, false, false, true, false, false, false, false, false}, 2, "Dim"},          // C Eb Gb
    {{true, false, false, false, true, false, false, false, true, false, false, false}, 3, "Aug"},          // C E G#
    
    // Seventh Chords (4-9)
    {{true, false, false, false, true, false, false, true, false, false, true, false}, 4, "Dom7"},          // C E G Bb
    {{true, false, false, false, true, false, false, true, false, false, false, true}, 5, "Maj7"},          // C E G B
    {{true, false, false, true, false, false, false, true, false, false, true, false}, 6, "Min7"},          // C Eb G Bb
    {{true, false, false, true, false, false, true, false, false, false, true, false}, 7, "HalfDim7"},      // C Eb Gb Bb (m7b5)
    {{true, false, false, true, false, false, true, false, false, true, false, false}, 8, "Dim7"},          // C Eb Gb Bbb
    {{true, false, false, true, false, false, false, true, false, false, false, true}, 9, "MinMaj7"},       // C Eb G B
    
    // Extended Chords (10-15)
    {{true, false, true, false, true, false, false, true, false, false, true, false}, 10, "Dom9"},          // C E G Bb D
    {{true, false, true, false, true, false, false, true, false, false, false, true}, 11, "Maj9"},          // C E G B D
    {{true, false, true, true, false, false, false, true, false, false, true, false}, 12, "Min9"},          // C Eb G Bb D
    {{true, false, true, false, true, false, false, true, false, false, true, true}, 13, "Dom11"},          // C E G Bb D F
    {{true, false, true, false, true, false, false, true, false, true, true, false}, 14, "Dom13"},          // C E G Bb D A
    {{true, false, true, false, true, false, false, true, false, false, false, true}, 15, "Maj9"},          // C E G B D
    
    // Suspended Chords (16-19)
    {{true, false, true, false, false, false, false, true, false, false, false, false}, 16, "Sus2"},        // C D G
    {{true, false, false, false, false, true, false, true, false, false, false, false}, 17, "Sus4"},        // C F G
    {{true, false, true, false, false, false, false, true, false, false, true, false}, 18, "7Sus2"},        // C D G Bb
    {{true, false, false, false, false, true, false, true, false, false, true, false}, 19, "7Sus4"},        // C F G Bb
    
    // Add Chords (20-23)
    {{true, false, true, false, true, false, false, true, false, false, false, false}, 20, "Add9"},         // C E G D
    {{true, false, false, false, true, true, false, true, false, false, false, false}, 21, "Add11"},        // C E F G
    {{true, false, false, false, true, false, false, true, false, true, false, false}, 22, "Add6"},         // C E G A
    {{true, false, true, true, false, false, false, true, false, false, false, false}, 23, "MinAdd9"},      // C Eb G D
    
    // Altered Chords (24-29)
    {{true, true, false, false, true, false, false, true, false, false, true, false}, 24, "Dom7b9"},        // C E G Bb Db
    {{true, false, false, true, true, false, false, true, false, false, true, false}, 25, "Dom7#9"},        // C E G Bb D#
    {{true, false, false, false, true, false, true, true, false, false, true, false}, 26, "Dom7b5"},        // C E Gb Bb
    {{true, false, false, false, true, false, false, false, true, false, true, false}, 27, "Dom7#5"},       // C E G# Bb (Aug7)
    {{true, true, false, false, true, false, true, false, false, false, true, false}, 28, "7b9b5"},         // C E Gb Bb Db
    {{true, false, false, true, true, false, true, false, false, false, true, false}, 29, "7#9b5"},         // C E Gb Bb D#
    
    // Power Chord and Octave (30-31)
    {{true, false, false, false, false, false, false, true, false, false, false, false}, 30, "5"},          // C G (power chord)
    {{true, false, false, false, false, false, false, false, false, false, false, false}, 31, "Root"},      // C (single note)
}};

ChordAnalyzer::ChordAnalyzer()
    : confidenceThreshold_(0.5f)
    , temporalSmoothing_(0.3f)
{
}

Chord ChordAnalyzer::analyze(const std::array<bool, 12>& pitchClassSet) noexcept {
    Chord result;
    findBestMatch(pitchClassSet, result);
    return result;
}

void ChordAnalyzer::update(const std::array<bool, 12>& pitchClassSet) noexcept {
    previousChord_ = currentChord_;
    findBestMatch(pitchClassSet, currentChord_);
    
    // Apply temporal smoothing
    if (previousChord_.confidence > 0.0f) {
        currentChord_.confidence = 
            temporalSmoothing_ * currentChord_.confidence +
            (1.0f - temporalSmoothing_) * previousChord_.confidence;
    }
}

void ChordAnalyzer::setConfidenceThreshold(float threshold) noexcept {
    confidenceThreshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

void ChordAnalyzer::setTemporalSmoothing(float factor) noexcept {
    temporalSmoothing_ = std::clamp(factor, 0.0f, 1.0f);
}

float ChordAnalyzer::scoreAgainstTemplate(
    const std::array<bool, 12>& pitchClassSet,
    const ChordTemplate& template_,
    uint8_t root
) const noexcept {
    int matches = 0;        // Notes in template that are present
    int required = 0;       // Total notes in template
    int extra = 0;          // Notes present but not in template
    int pitchCount = 0;     // Total notes in input
    
    // Count pitches in input
    for (bool pitch : pitchClassSet) {
        if (pitch) ++pitchCount;
    }
    
    // If no pitches, no match
    if (pitchCount == 0) return 0.0f;
    
    // Check each pitch class
    for (int i = 0; i < 12; ++i) {
        int rotated = (i + root) % 12;
        bool inTemplate = template_.pattern[i];
        bool inInput = pitchClassSet[rotated];
        
        if (inTemplate) {
            ++required;
            if (inInput) {
                ++matches;
            }
        } else if (inInput) {
            ++extra;  // Penalize extra notes not in template
        }
    }
    
    // Scoring formula:
    // - Reward complete matches (all template notes present)
    // - Penalize missing notes from template
    // - Penalize extra notes not in template
    
    if (required == 0) return 0.0f;
    
    float completeness = static_cast<float>(matches) / required;
    float extraPenalty = 1.0f / (1.0f + 0.5f * extra);  // Gentle penalty for extra notes
    
    return completeness * extraPenalty;
}

void ChordAnalyzer::findBestMatch(
    const std::array<bool, 12>& pitchClassSet,
    Chord& outChord
) noexcept {
    float bestScore = 0.0f;
    uint8_t bestRoot = 0;
    uint8_t bestQuality = 0;
    
    // Try all templates at all roots
    for (uint8_t root = 0; root < 12; ++root) {
        for (const auto& template_ : kChordTemplates) {
            float score = scoreAgainstTemplate(pitchClassSet, template_, root);
            
            if (score > bestScore) {
                bestScore = score;
                bestRoot = root;
                bestQuality = template_.quality;
            }
        }
    }
    
    outChord.root = bestRoot;
    outChord.quality = bestQuality;
    outChord.confidence = bestScore;
    outChord.pitchClass = pitchClassSet;
}

// ============================================================================
// SIMD-optimized implementations
// ============================================================================

#ifdef __AVX2__

// AVX2 intrinsics version: Process 8 pitch classes at once
float ChordAnalyzer::scoreAgainstTemplateSIMD(
    const std::array<bool, 12>& pitchClassSet,
    const ChordTemplate& template_,
    uint8_t root
) const noexcept {
    // Pack bool arrays into bitmasks for SIMD
    alignas(32) uint32_t templateMask[8] = {0};
    alignas(32) uint32_t inputMask[8] = {0};
    
    // First 8 pitch classes
    for (int i = 0; i < 8; ++i) {
        int rotated = (i + root) % 12;
        templateMask[i] = template_.pattern[i] ? 0xFFFFFFFF : 0;
        inputMask[i] = pitchClassSet[rotated] ? 0xFFFFFFFF : 0;
    }
    
    // Load into AVX2 registers
    __m256i templateVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(templateMask));
    __m256i inputVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(inputMask));
    
    // Count matches: template AND input
    __m256i matches = _mm256_and_si256(templateVec, inputVec);
    
    // Count required: template bits
    __m256i required = templateVec;
    
    // Count extra: input AND NOT template
    __m256i notTemplate = _mm256_andnot_si256(templateVec, _mm256_set1_epi32(0xFFFFFFFF));
    __m256i extra = _mm256_and_si256(inputVec, notTemplate);
    
    // Horizontal sum using population count
    int matchCount = _mm_popcnt_u32(_mm256_movemask_epi8(matches)) / 4;
    int requiredCount = _mm_popcnt_u32(_mm256_movemask_epi8(required)) / 4;
    int extraCount = _mm_popcnt_u32(_mm256_movemask_epi8(extra)) / 4;
    
    // Process remaining 4 pitch classes (8-11) scalar
    for (int i = 8; i < 12; ++i) {
        int rotated = (i + root) % 12;
        bool inTemplate = template_.pattern[i];
        bool inInput = pitchClassSet[rotated];
        
        if (inTemplate) {
            ++requiredCount;
            if (inInput) ++matchCount;
        } else if (inInput) {
            ++extraCount;
        }
    }
    
    // Scoring (same as scalar version)
    if (requiredCount == 0) return 0.0f;
    
    float completeness = static_cast<float>(matchCount) / requiredCount;
    float extraPenalty = 1.0f / (1.0f + 0.5f * extraCount);
    
    return completeness * extraPenalty;
}

void ChordAnalyzer::findBestMatchSIMD(
    const std::array<bool, 12>& pitchClassSet,
    Chord& outChord
) noexcept {
    // Use SIMD for template scoring
    float bestScore = 0.0f;
    uint8_t bestRoot = 0;
    uint8_t bestQuality = 0;
    
    // Try all templates at all roots
    for (uint8_t root = 0; root < 12; ++root) {
        for (const auto& template_ : kChordTemplates) {
            float score = scoreAgainstTemplateSIMD(pitchClassSet, template_, root);
            
            if (score > bestScore) {
                bestScore = score;
                bestRoot = root;
                bestQuality = template_.quality;
            }
        }
    }
    
    outChord.root = bestRoot;
    outChord.quality = bestQuality;
    outChord.confidence = bestScore;
    outChord.pitchClass = pitchClassSet;
}

Chord ChordAnalyzer::analyzeSIMD(const std::array<bool, 12>& pitchClassSet) noexcept {
    Chord result;
    findBestMatchSIMD(pitchClassSet, result);
    return result;
}

#else // Scalar fallback

float ChordAnalyzer::scoreAgainstTemplateSIMD(
    const std::array<bool, 12>& pitchClassSet,
    const ChordTemplate& template_,
    uint8_t root
) const noexcept {
    // Fall back to scalar implementation
    return scoreAgainstTemplate(pitchClassSet, template_, root);
}

void ChordAnalyzer::findBestMatchSIMD(
    const std::array<bool, 12>& pitchClassSet,
    Chord& outChord
) noexcept {
    // Fall back to scalar implementation
    findBestMatch(pitchClassSet, outChord);
}

Chord ChordAnalyzer::analyzeSIMD(const std::array<bool, 12>& pitchClassSet) noexcept {
    Chord result;
    findBestMatch(pitchClassSet, result);
    return result;
}

#endif // __AVX2__

} // namespace penta::harmony
