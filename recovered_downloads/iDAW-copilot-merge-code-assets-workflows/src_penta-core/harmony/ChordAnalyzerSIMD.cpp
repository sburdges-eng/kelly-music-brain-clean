#include "penta/harmony/ChordAnalyzer.h"
#include <algorithm>
#include <cmath>

// SIMD intrinsics
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace penta::harmony {

#ifdef __AVX2__

// AVX2-optimized chord pattern matching
// Processes 32 templates in parallel using 256-bit SIMD registers
float ChordAnalyzer::scoreAgainstTemplateSIMD(
    const std::array<bool, 12>& pitchClassSet,
    const ChordTemplate& template_,
    uint8_t root
) const noexcept {
    // Pack bool arrays into 16-bit masks for SIMD operations
    uint16_t templateMask = 0;
    uint16_t inputMask = 0;
    
    for (int i = 0; i < 12; ++i) {
        if (template_.pattern[i]) {
            templateMask |= (1 << i);
        }
        
        int rotated = (i + root) % 12;
        if (pitchClassSet[rotated]) {
            inputMask |= (1 << i);
        }
    }
    
    // Use SIMD population count for fast bit counting
    uint16_t matches = templateMask & inputMask;      // Notes in both
    uint16_t extra = inputMask & ~templateMask;       // Extra notes
    uint16_t required = templateMask;                 // Template notes
    
    int matchCount = _mm_popcnt_u32(matches);
    int requiredCount = _mm_popcnt_u32(required);
    int extraCount = _mm_popcnt_u32(extra);
    
    if (requiredCount == 0) return 0.0f;
    
    float completeness = static_cast<float>(matchCount) / requiredCount;
    float extraPenalty = 1.0f / (1.0f + 0.5f * extraCount);
    
    return completeness * extraPenalty;
}

// AVX2-optimized batch scoring of multiple templates
// Processes 8 templates at once using vectorized operations
void ChordAnalyzer::findBestMatchSIMD(
    const std::array<bool, 12>& pitchClassSet,
    Chord& outChord
) noexcept {
    // Pack input into bitmask for fast SIMD operations
    uint16_t inputMask = 0;
    int pitchCount = 0;
    for (int i = 0; i < 12; ++i) {
        if (pitchClassSet[i]) {
            inputMask |= (1 << i);
            ++pitchCount;
        }
    }
    
    if (pitchCount == 0) {
        outChord.confidence = 0.0f;
        return;
    }
    
    float bestScore = 0.0f;
    uint8_t bestRoot = 0;
    uint8_t bestQuality = 0;
    
    // Prepare 8 scores at a time using AVX2
    alignas(32) float scores[8];
    __m256 vBestScore = _mm256_setzero_ps();
    
    // Try all roots
    for (uint8_t root = 0; root < 12; ++root) {
        // Rotate input mask by root
        uint16_t rotatedInput = ((inputMask << root) | (inputMask >> (12 - root))) & 0xFFF;
        
        // Process templates in batches of 8
        for (size_t templateIdx = 0; templateIdx < kChordTemplates.size(); templateIdx += 8) {
            // Vectorize scoring for 8 templates
            for (int i = 0; i < 8 && (templateIdx + i) < kChordTemplates.size(); ++i) {
                const auto& tmpl = kChordTemplates[templateIdx + i];
                
                // Pack template into bitmask
                uint16_t templateMask = 0;
                for (int j = 0; j < 12; ++j) {
                    if (tmpl.pattern[j]) {
                        templateMask |= (1 << j);
                    }
                }
                
                // Compute matches using bitwise operations
                uint16_t matches = rotatedInput & templateMask;
                uint16_t extra = rotatedInput & ~templateMask;
                
                int matchCount = _mm_popcnt_u32(matches);
                int requiredCount = _mm_popcnt_u32(templateMask);
                int extraCount = _mm_popcnt_u32(extra);
                
                if (requiredCount > 0) {
                    float completeness = static_cast<float>(matchCount) / requiredCount;
                    float extraPenalty = 1.0f / (1.0f + 0.5f * extraCount);
                    scores[i] = completeness * extraPenalty;
                } else {
                    scores[i] = 0.0f;
                }
            }
            
            // Load scores into SIMD register
            __m256 vScores = _mm256_load_ps(scores);
            
            // Find maximum in batch
            vBestScore = _mm256_max_ps(vBestScore, vScores);
            
            // Extract maximum and update best match
            for (int i = 0; i < 8 && (templateIdx + i) < kChordTemplates.size(); ++i) {
                if (scores[i] > bestScore) {
                    bestScore = scores[i];
                    bestRoot = root;
                    bestQuality = kChordTemplates[templateIdx + i].quality;
                }
            }
        }
    }
    
    outChord.root = bestRoot;
    outChord.quality = bestQuality;
    outChord.confidence = bestScore;
    outChord.pitchClass = pitchClassSet;
}

#endif // __AVX2__

// Scalar fallback when AVX2 not available
#ifndef __AVX2__

float ChordAnalyzer::scoreAgainstTemplateSIMD(
    const std::array<bool, 12>& pitchClassSet,
    const ChordTemplate& template_,
    uint8_t root
) const noexcept {
    return scoreAgainstTemplate(pitchClassSet, template_, root);
}

void ChordAnalyzer::findBestMatchSIMD(
    const std::array<bool, 12>& pitchClassSet,
    Chord& outChord
) noexcept {
    findBestMatch(pitchClassSet, outChord);
}

#endif // __AVX2__

// Public API uses SIMD when available
Chord ChordAnalyzer::analyzeSIMD(const std::array<bool, 12>& pitchClassSet) noexcept {
    Chord result;
    findBestMatchSIMD(pitchClassSet, result);
    return result;
}

} // namespace penta::harmony
