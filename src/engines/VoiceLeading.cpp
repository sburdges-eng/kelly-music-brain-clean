#include "VoiceLeading.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace kelly {

VoiceLeadingEngine::VoiceLeadingEngine() {}

std::vector<int> VoiceLeadingEngine::voice(
    const std::vector<int>& chordTones,
    const std::vector<int>& previousVoicing,
    int bassPitch
) {
    if (chordTones.empty()) return {};
    if (previousVoicing.empty()) {
        // First chord - use root position
        std::vector<int> result;
        int bass = bassPitch >= 0 ? bassPitch : chordTones[0] + 36;
        result.push_back(bass);
        for (size_t i = 1; i < chordTones.size(); ++i) {
            result.push_back(chordTones[i] + 48);
        }
        return result;
    }
    
    std::vector<int> newVoicing;
    std::vector<bool> used(chordTones.size(), false);
    
    // For each voice in previous voicing, find closest chord tone
    for (int prevPitch : previousVoicing) {
        int bestPitch = -1;
        int bestDistance = std::numeric_limits<int>::max();
        int bestIndex = -1;
        
        for (size_t i = 0; i < chordTones.size(); ++i) {
            if (used[i]) continue;
            
            // Try different octaves
            for (int octave = -2; octave <= 2; ++octave) {
                int candidate = chordTones[i] + (octave * 12) + 48;
                int distance = std::abs(candidate - prevPitch);
                
                if (distance < bestDistance && 
                    distance <= config_.maxVoiceMovement) {
                    // Check for parallel motion
                    bool wouldBeParallel = false;
                    if (!newVoicing.empty() && config_.avoidParallelFifths) {
                        int prevVoice = previousVoicing[newVoicing.size() - 1];
                        int newVoice = newVoicing.back();
                        wouldBeParallel = wouldCreateParallel(
                            prevVoice, newVoice, prevPitch, candidate, 7
                        );
                    }
                    
                    if (!wouldBeParallel) {
                        bestPitch = candidate;
                        bestDistance = distance;
                        bestIndex = i;
                    }
                }
            }
        }
        
        if (bestPitch >= 0) {
            newVoicing.push_back(bestPitch);
            if (bestIndex >= 0) used[bestIndex] = true;
        } else {
            // Couldn't find good voice leading, use closest available
            newVoicing.push_back(prevPitch);
        }
    }
    
    // Override bass if specified
    if (bassPitch >= 0 && !newVoicing.empty()) {
        newVoicing[0] = bassPitch;
    }
    
    return newVoicing;
}

VoiceLeadingResult VoiceLeadingEngine::analyze(
    const std::vector<int>& fromVoicing,
    const std::vector<int>& toVoicing
) {
    VoiceLeadingResult result;
    result.fromVoicing = fromVoicing;
    result.toVoicing = toVoicing;
    result.hasParallelFifths = false;
    result.hasParallelOctaves = false;
    
    size_t minSize = std::min(fromVoicing.size(), toVoicing.size());
    
    for (size_t i = 0; i < minSize; ++i) {
        VoiceMovement movement;
        movement.fromPitch = fromVoicing[i];
        movement.toPitch = toVoicing[i];
        movement.interval = toVoicing[i] - fromVoicing[i];
        
        if (movement.interval > 0) movement.direction = "up";
        else if (movement.interval < 0) movement.direction = "down";
        else movement.direction = "static";
        
        result.movements.push_back(movement);
    }
    
    // Check for parallel fifths and octaves
    for (size_t i = 0; i < minSize; ++i) {
        for (size_t j = i + 1; j < minSize; ++j) {
            int fromInterval = std::abs(fromVoicing[i] - fromVoicing[j]) % 12;
            int toInterval = std::abs(toVoicing[i] - toVoicing[j]) % 12;
            
            if (fromInterval == 7 && toInterval == 7) {
                result.hasParallelFifths = true;
            }
            if (fromInterval == 0 && toInterval == 0) {
                result.hasParallelOctaves = true;
            }
        }
    }
    
    result.smoothnessScore = calculateSmoothness(result.movements);
    return result;
}

std::vector<std::vector<int>> VoiceLeadingEngine::voiceProgression(
    const std::vector<std::vector<int>>& chordTones,
    int startingBass
) {
    std::vector<std::vector<int>> result;
    if (chordTones.empty()) return result;
    
    std::vector<int> previous;
    int currentBass = startingBass;
    
    for (const auto& chord : chordTones) {
        auto voiced = voice(chord, previous, currentBass);
        result.push_back(voiced);
        previous = voiced;
        if (!voiced.empty()) currentBass = voiced[0];
    }
    
    return result;
}

int VoiceLeadingEngine::findClosestPitch(int target, const std::vector<int>& candidates) {
    if (candidates.empty()) return target;
    
    int closest = candidates[0];
    int minDist = std::abs(target - closest);
    
    for (int c : candidates) {
        int dist = std::abs(target - c);
        if (dist < minDist) {
            minDist = dist;
            closest = c;
        }
    }
    
    return closest;
}

bool VoiceLeadingEngine::wouldCreateParallel(int v1From, int v1To, int v2From, int v2To, int interval) {
    int fromInterval = std::abs(v1From - v2From) % 12;
    int toInterval = std::abs(v1To - v2To) % 12;
    
    // Both intervals must be the target (5th = 7 semitones, octave = 0)
    if (fromInterval == interval && toInterval == interval) {
        // And both voices must move in same direction
        int dir1 = (v1To > v1From) ? 1 : (v1To < v1From) ? -1 : 0;
        int dir2 = (v2To > v2From) ? 1 : (v2To < v2From) ? -1 : 0;
        return dir1 == dir2 && dir1 != 0;
    }
    return false;
}

float VoiceLeadingEngine::calculateSmoothness(const std::vector<VoiceMovement>& movements) {
    if (movements.empty()) return 1.0f;
    
    float totalMovement = 0;
    int stepwiseCount = 0;
    
    for (const auto& m : movements) {
        totalMovement += std::abs(m.interval);
        if (std::abs(m.interval) <= 2) stepwiseCount++;
    }
    
    float avgMovement = totalMovement / movements.size();
    float stepwiseRatio = static_cast<float>(stepwiseCount) / movements.size();
    
    // Lower average movement and higher stepwise ratio = smoother
    float smoothness = (1.0f - avgMovement / 12.0f) * 0.5f + stepwiseRatio * 0.5f;
    return std::clamp(smoothness, 0.0f, 1.0f);
}

std::vector<int> VoiceLeadingEngine::invertVoicing(const std::vector<int>& voicing, int inversion) {
    if (voicing.empty()) return voicing;
    
    std::vector<int> result = voicing;
    for (int i = 0; i < inversion && !result.empty(); ++i) {
        result[0] += 12;
        std::rotate(result.begin(), result.begin() + 1, result.end());
    }
    return result;
}

} // namespace kelly
