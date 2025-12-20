#include "midi/GrooveEngine.h"
#include "common/MusicConstants.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <string>

namespace kelly {
using namespace MusicConstants;

// Constants from Python groove engine
constexpr float MAX_BEAT_DRIFT = 0.036f;  // ~35 ticks at 960 PPQ = ~0.036 beats
constexpr float HUMAN_LATENCY_BIAS = 0.005f;  // Slight behind-the-beat bias
constexpr int VELOCITY_MIN = 20;
constexpr int VELOCITY_MAX = 120;
constexpr float MAX_DROPOUT_PROB = 0.2f;  // 20% max dropout at complexity=1.0
constexpr float GHOST_NOTE_PROBABILITY = 0.15f;
constexpr float GHOST_NOTE_VELOCITY_MULT = 0.4f;

// Timing feel constants
constexpr float MAX_PUSH_PULL_BEATS = 0.05f;  // Max 5% of beat for push/pull
constexpr float SWING_MAX_OFFSET = 0.083f;    // Max swing offset (1/3 of 8th note = triplet)

GrooveEngine::GrooveEngine() : rng_(std::random_device{}()) {
    initializeTemplates();
}

std::vector<MidiNote> GrooveEngine::applyGroove(
    const std::vector<MidiNote>& notes,
    GrooveType grooveType,
    float humanizationLevel)
{
    if (humanizationLevel < 0.01f) return notes;
    
    std::vector<MidiNote> result;
    result.reserve(notes.size());
    
    std::uniform_real_distribution<float> timingDist(-MAX_BEAT_DRIFT, MAX_BEAT_DRIFT);
    std::uniform_real_distribution<float> velocityDist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dropoutDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> ghostDist(0.0f, 1.0f);
    
    for (const auto& note : notes) {
        // Dropout probability based on humanization level
        float dropoutProb = humanizationLevel * MAX_DROPOUT_PROB;
        if (dropoutDist(rng_) < dropoutProb) {
            continue;  // Drop this note
        }
        
        MidiNote processed = note;
        
        // Apply timing drift (with slight behind-the-beat bias)
        float timingDrift = timingDist(rng_) * humanizationLevel;
        timingDrift += HUMAN_LATENCY_BIAS * humanizationLevel;  // Slight lag
        processed.startBeat += timingDrift;
        
        // Apply groove type adjustments
        switch (grooveType) {
            case GrooveType::Swing:
                // Swing: delay off-beats by ~33%
                if (std::fmod(processed.startBeat * 2.0, 1.0) > 0.5) {
                    processed.startBeat += 0.033f * humanizationLevel;
                }
                break;
            case GrooveType::Syncopated:
                // Syncopation: push some notes ahead
                if (dropoutDist(rng_) < 0.3f) {
                    processed.startBeat -= 0.02f * humanizationLevel;
                }
                break;
            case GrooveType::Shuffle:
                // Shuffle: delay every other 8th note
                if (std::fmod(processed.startBeat * 2.0, 1.0) > 0.5) {
                    processed.startBeat += 0.05f * humanizationLevel;
                }
                break;
            case GrooveType::Halftime:
                // Halftime: stretch timing
                processed.startBeat *= 1.0f + (0.1f * humanizationLevel);
                break;
            default:  // Straight
                break;
        }
        
        // Apply velocity humanization
        int velocityChange = static_cast<int>(velocityDist(rng_) * humanizationLevel);
        processed.velocity = std::clamp(note.velocity + velocityChange, VELOCITY_MIN, VELOCITY_MAX);
        
        result.push_back(processed);
        
        // Add ghost notes occasionally
        if (ghostDist(rng_) < GHOST_NOTE_PROBABILITY * humanizationLevel) {
            MidiNote ghost = processed;
            ghost.startBeat += timingDist(rng_) * 0.5f;  // Near the original
            ghost.velocity = std::clamp(
                static_cast<int>(processed.velocity * GHOST_NOTE_VELOCITY_MULT),
                VELOCITY_MIN / 2,
                VELOCITY_MAX
            );
            ghost.duration *= 0.5f;  // Shorter ghost notes
            result.push_back(ghost);
        }
    }
    
    // Sort by startBeat to maintain order
    std::sort(result.begin(), result.end(),
        [](const MidiNote& a, const MidiNote& b) {
            return a.startBeat < b.startBeat;
        });
    
    return result;
}

std::vector<MidiNote> GrooveEngine::applyEmotionTiming(
    const std::vector<MidiNote>& notes,
    const EmotionNode& emotion)
{
    std::vector<MidiNote> result;
    result.reserve(notes.size());
    
    // Emotion-based timing adjustments
    // Sad emotions drag behind, angry emotions rush ahead
    float timingBias = 0.0f;
    
    if (emotion.valence < VALENCE_NEGATIVE) {
        // Negative emotions (sad, angry) - timing adjustments
        if (emotion.name.find("sad") != std::string::npos ||
            emotion.name.find("grief") != std::string::npos ||
            emotion.name.find("lonely") != std::string::npos) {
            timingBias = TIMING_BIAS_SAD * emotion.intensity;  // Drag behind
        } else if (emotion.name.find("angry") != std::string::npos ||
                   emotion.name.find("rage") != std::string::npos ||
                   emotion.name.find("fury") != std::string::npos) {
            timingBias = TIMING_BIAS_ANGRY * emotion.intensity;  // Rush ahead
        }
    }
    
    // Arousal affects timing tightness
    float timingVariance = (1.0f - emotion.arousal) * TIMING_VARIATION_BASE;  // Lower arousal = more variance
    std::uniform_real_distribution<float> timingDist(-timingVariance, timingVariance);
    
    for (const auto& note : notes) {
        MidiNote processed = note;
        processed.startBeat += timingBias + timingDist(rng_);
        result.push_back(processed);
    }
    
    return result;
}

void GrooveEngine::initializeTemplates() {
    // Funk Pocket - Deep pocket with emphasis on 2 and 4, slight push on 16ths
    templates_["funk"] = {
        "Funk Pocket",
        "Deep pocket with emphasis on 2 and 4, slight push on 16ths",
        0.15f,  // swing
        0.1f,   // slight push
        {0.0f, -0.008f, 0.005f, -0.005f,   // Beat 1: slight push on 16th notes
         0.012f, -0.006f, 0.008f, -0.004f, // Beat 2: laid back snare
         0.0f, -0.010f, 0.006f, -0.006f,   // Beat 3
         0.015f, -0.005f, 0.010f, -0.003f}, // Beat 4: laid back snare
        {110, 60, 85, 55,  // Beat 1: strong downbeat
         95, 55, 75, 50,   // Beat 2: snare accent
         85, 50, 70, 45,   // Beat 3
         100, 60, 80, 55}, // Beat 4: snare accent
        {90, 120}
    };
    
    // Jazz Swing - Classic triplet swing feel
    templates_["jazz"] = {
        "Jazz Swing",
        "Classic triplet swing feel with brush-like dynamics",
        0.67f,  // heavy swing
        0.0f,   // neutral
        {0.0f, 0.020f, 0.0f, 0.025f,     // Heavy swing on off-beats
         -0.005f, 0.022f, 0.0f, 0.028f,
         0.0f, 0.018f, -0.003f, 0.024f,
         -0.008f, 0.020f, 0.0f, 0.026f},
        {90, 50, 75, 55,
         85, 55, 70, 60,
         80, 50, 72, 58,
         88, 52, 78, 62},
        {100, 200}
    };
    
    // Rock Drive - Straight feel with strong backbeat, slight push
    templates_["rock"] = {
        "Rock Drive",
        "Straight feel with strong backbeat, slight push",
        0.0f,   // no swing
        0.15f,  // pushed
        {0.0f, -0.005f, 0.0f, -0.008f,     // Slight push throughout
         -0.003f, -0.006f, -0.002f, -0.007f,
         0.0f, -0.004f, 0.0f, -0.006f,
         -0.002f, -0.005f, -0.003f, -0.008f},
        {115, 70, 90, 65,  // Strong kick
         120, 80, 95, 70,  // Heavy snare
         100, 65, 85, 60,
         118, 75, 92, 68}, // Heavy snare
        {100, 140}
    };
    
    // Hip-Hop Pocket - Deep laid-back pocket
    templates_["hiphop"] = {
        "Hip-Hop Pocket",
        "Deep laid-back pocket with heavy ghost notes",
        0.25f,  // moderate swing
        -0.3f,  // laid back
        {0.0f, 0.015f, 0.008f, 0.018f,     // Very laid back
         0.020f, 0.012f, 0.015f, 0.020f,
         0.005f, 0.018f, 0.010f, 0.022f,
         0.025f, 0.015f, 0.018f, 0.025f},
        {120, 35, 45, 30,  // Strong kick, lots of ghosts
         60, 30, 40, 35,
         100, 38, 48, 32,
         55, 32, 42, 38},
        {75, 100}
    };
    
    // EDM Quantized - Machine-tight with subtle humanization
    templates_["edm"] = {
        "EDM Quantized",
        "Machine-tight with subtle humanization",
        0.0f,   // no swing
        0.0f,   // neutral
        {0.0f, 0.0f, 0.0f, 0.0f,       // Tight to grid
         0.001f, -0.001f, 0.001f, -0.001f, // Micro variations
         0.0f, 0.0f, 0.0f, 0.0f,
         -0.001f, 0.001f, -0.001f, 0.001f},
        {127, 95, 110, 90,
         125, 92, 108, 88,
         127, 93, 112, 91,
         124, 94, 106, 89},
        {120, 150}
    };
    
    // Latin Clave - Syncopated feel
    templates_["latin"] = {
        "Latin Clave",
        "Syncopated feel based on 3-2 clave",
        0.1f,   // slight swing
        0.0f,   // neutral
        {0.0f, 0.005f, -0.005f, 0.008f,    // Clave-based
         -0.008f, 0.006f, 0.0f, 0.010f,
         0.005f, -0.005f, 0.008f, 0.005f,
         -0.006f, 0.008f, -0.003f, 0.012f},
        {100, 70, 95, 65,
         85, 75, 80, 70,
         90, 68, 98, 72,
         88, 73, 85, 68},
        {90, 130}
    };
    
    // Blues Shuffle - 12/8 shuffle feel
    templates_["blues"] = {
        "Blues Shuffle",
        "12/8 shuffle feel with expressive dynamics",
        0.6f,   // heavy shuffle/swing
        0.0f,   // neutral
        {0.0f, 0.018f, 0.0f, 0.020f,     // Shuffle swing
         -0.005f, 0.022f, 0.0f, 0.025f,
         0.0f, 0.016f, -0.003f, 0.018f,
         -0.008f, 0.024f, 0.0f, 0.022f},
        {105, 55, 85, 60,
         95, 60, 80, 55,
         90, 52, 82, 58,
         98, 58, 88, 62},
        {70, 120}
    };
    
    // Lo-Fi Bedroom - Intentionally imperfect
    templates_["lofi"] = {
        "Lo-Fi Bedroom",
        "Intentionally imperfect, organic feel",
        0.35f,  // moderate swing
        -0.2f,  // slightly laid back
        {0.005f, 0.020f, -0.008f, 0.025f,  // Deliberately inconsistent
         -0.012f, 0.030f, 0.008f, 0.022f,
         0.015f, -0.005f, 0.028f, 0.010f,
         -0.010f, 0.035f, -0.005f, 0.028f},
        {95, 45, 70, 40,
         50, 42, 55, 38,
         80, 48, 65, 42,
         52, 40, 58, 35},
        {70, 95}
    };
}

std::vector<MidiNote> GrooveEngine::applyGrooveTemplate(
    const std::vector<MidiNote>& notes,
    const std::string& templateName,
    float humanizationLevel,
    float intensity)
{
    auto it = templates_.find(templateName);
    if (it == templates_.end()) {
        // Fallback to basic groove
        return applyGroove(notes, GrooveType::Straight, humanizationLevel);
    }
    
    const GrooveTemplateData& grooveTemplate = it->second;
    std::vector<MidiNote> result;
    result.reserve(notes.size());

    std::uniform_real_distribution<float> dropoutDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> ghostDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> microTimingDist(-0.002f, 0.002f);

    for (const auto& note : notes) {
        // Dropout probability
        float dropoutProb = humanizationLevel * MAX_DROPOUT_PROB;
        if (dropoutDist(rng_) < dropoutProb) {
            continue;
        }

        MidiNote processed = note;

        // Apply template timing deviation
        float timingDev = getTimingDeviation(grooveTemplate, processed.startBeat);
        processed.startBeat += timingDev * intensity;

        // Apply swing from template
        if (grooveTemplate.swingFactor > 0.01f) {
            float swingOffset = calculateSwingOffset(processed.startBeat, grooveTemplate.swingFactor);
            processed.startBeat += swingOffset * intensity;
        }

        // Apply push/pull from template
        if (std::abs(grooveTemplate.pushPull) > 0.01f) {
            float pushPullOffset = grooveTemplate.pushPull * MAX_PUSH_PULL_BEATS * intensity;
            processed.startBeat += pushPullOffset;
        }
        
        // Apply micro-timing humanization
        if (humanizationLevel > 0.01f) {
            float microTiming = microTimingDist(rng_) * humanizationLevel;
            processed.startBeat += microTiming;
        }
        
        // Apply template velocity curve
        int templateVel = getTemplateVelocity(grooveTemplate, processed.startBeat, note.velocity);
        if (intensity > 0.01f) {
            // Blend original velocity with template velocity
            processed.velocity = static_cast<int>(
                note.velocity * (1.0f - intensity) + templateVel * intensity
            );
        } else {
            processed.velocity = note.velocity;
        }
        processed.velocity = std::clamp(processed.velocity, VELOCITY_MIN, VELOCITY_MAX);
        
        result.push_back(processed);
        
        // Add ghost notes occasionally
        if (ghostDist(rng_) < GHOST_NOTE_PROBABILITY * humanizationLevel) {
            MidiNote ghost = processed;
            ghost.startBeat += microTimingDist(rng_) * 0.5f;
            ghost.velocity = std::clamp(
                static_cast<int>(processed.velocity * GHOST_NOTE_VELOCITY_MULT),
                VELOCITY_MIN / 2,
                VELOCITY_MAX
            );
            ghost.duration *= 0.5f;
            result.push_back(ghost);
        }
    }
    
    // Sort by startBeat
    std::sort(result.begin(), result.end(),
        [](const MidiNote& a, const MidiNote& b) {
            return a.startBeat < b.startBeat;
        });
    
    return result;
}

std::vector<MidiNote> GrooveEngine::applySwing(
    const std::vector<MidiNote>& notes,
    float swingAmount,
    float intensity)
{
    if (swingAmount < 0.01f || intensity < 0.01f) {
        return notes;
    }
    
    std::vector<MidiNote> result;
    result.reserve(notes.size());
    
    for (const auto& note : notes) {
        MidiNote processed = note;
        float swingOffset = calculateSwingOffset(processed.startBeat, swingAmount);
        processed.startBeat += swingOffset * intensity;
        result.push_back(processed);
    }
    
    return result;
}

std::vector<MidiNote> GrooveEngine::applyTimingFeel(
    const std::vector<MidiNote>& notes,
    float feel,
    float intensity)
{
    if (std::abs(feel) < 0.01f || intensity < 0.01f) {
        return notes;
    }
    
    std::vector<MidiNote> result;
    result.reserve(notes.size());
    
    // Clamp feel to -1.0 to +1.0
    feel = std::clamp(feel, -1.0f, 1.0f);
    float offset = feel * MAX_PUSH_PULL_BEATS * intensity;
    
    for (const auto& note : notes) {
        MidiNote processed = note;
        processed.startBeat += offset;
        result.push_back(processed);
    }
    
    return result;
}

std::vector<MidiNote> GrooveEngine::humanize(
    const std::vector<MidiNote>& notes,
    float humanizationLevel,
    float microTimingBias,
    float velocityVariation)
{
    if (humanizationLevel < 0.01f) {
        return notes;
    }
    
    std::vector<MidiNote> result;
    result.reserve(notes.size());
    
    std::normal_distribution<float> timingDist(0.0f, MAX_BEAT_DRIFT * humanizationLevel);
    std::normal_distribution<float> velocityDist(0.0f, velocityVariation * 127.0f);
    std::uniform_real_distribution<float> dropoutDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> ghostDist(0.0f, 1.0f);
    
    for (const auto& note : notes) {
        // Dropout probability
        float dropoutProb = humanizationLevel * MAX_DROPOUT_PROB;
        if (dropoutDist(rng_) < dropoutProb) {
            continue;
        }
        
        MidiNote processed = note;
        
        // Apply micro-timing with bias
        float timingOffset = timingDist(rng_);
        timingOffset += microTimingBias * MAX_BEAT_DRIFT * 0.5f;  // Bias adjustment
        timingOffset += HUMAN_LATENCY_BIAS * humanizationLevel;   // Slight lag
        processed.startBeat += timingOffset;
        
        // Apply velocity variation
        float velocityOffset = velocityDist(rng_);
        processed.velocity = std::clamp(
            static_cast<int>(note.velocity + velocityOffset),
            VELOCITY_MIN,
            VELOCITY_MAX
        );
        
        result.push_back(processed);
        
        // Add ghost notes
        if (ghostDist(rng_) < GHOST_NOTE_PROBABILITY * humanizationLevel) {
            MidiNote ghost = processed;
            ghost.startBeat += timingDist(rng_) * 0.5f;
            ghost.velocity = std::clamp(
                static_cast<int>(processed.velocity * GHOST_NOTE_VELOCITY_MULT),
                VELOCITY_MIN / 2,
                VELOCITY_MAX
            );
            ghost.duration *= 0.5f;
            result.push_back(ghost);
        }
    }
    
    // Sort by startBeat
    std::sort(result.begin(), result.end(),
        [](const MidiNote& a, const MidiNote& b) {
            return a.startBeat < b.startBeat;
        });
    
    return result;
}

float GrooveEngine::getTimingDeviation(const GrooveTemplateData& grooveTemplate, double beatPosition) const {
    if (grooveTemplate.timingDeviations.empty()) {
        return 0.0f;
    }

    int sixteenthPos = getSixteenthNotePosition(beatPosition);
    int index = sixteenthPos % static_cast<int>(grooveTemplate.timingDeviations.size());
    return grooveTemplate.timingDeviations[index];
}

int GrooveEngine::getTemplateVelocity(const GrooveTemplateData& grooveTemplate, double beatPosition, int originalVelocity) const {
    if (grooveTemplate.velocityCurve.empty()) {
        return originalVelocity;
    }

    int sixteenthPos = getSixteenthNotePosition(beatPosition);
    int index = sixteenthPos % static_cast<int>(grooveTemplate.velocityCurve.size());
    return grooveTemplate.velocityCurve[index];
}

float GrooveEngine::calculateSwingOffset(double beatPosition, float swingAmount) const {
    // Swing applies to off-beat 8th notes (the "and" of the beat)
    // Check if this is an off-beat 8th note
    double beatInMeasure = std::fmod(beatPosition, 1.0);
    
    // Check if we're on an off-beat 8th (between 0.5 and 1.0 within a beat)
    if (beatInMeasure >= 0.5 && beatInMeasure < 1.0) {
        // Calculate how far into the off-beat we are (0.0 to 0.5)
        double offBeatPosition = beatInMeasure - 0.5;
        
        // Swing amount: 0.0 = no swing, 0.67 = triplet swing (max)
        // At 0.67 swing, off-beat 8th notes are delayed by 1/3 of an 8th note
        float swingDelay = swingAmount * SWING_MAX_OFFSET;
        
        // Apply delay proportionally based on position within the off-beat
        return swingDelay;
    }
    
    return 0.0f;
}

int GrooveEngine::getSixteenthNotePosition(double beatPosition) const {
    // Get position within current beat (0.0 to 1.0)
    double beatInMeasure = std::fmod(beatPosition, 1.0);
    
    // Convert to 16th note position (0-15)
    int sixteenthPos = static_cast<int>(beatInMeasure * 16.0);
    return std::clamp(sixteenthPos, 0, 15);
}

std::vector<std::string> GrooveEngine::getTemplateNames() const {
    std::vector<std::string> names;
    names.reserve(templates_.size());
    for (const auto& pair : templates_) {
        names.push_back(pair.first);
    }
    return names;
}

const GrooveTemplateData* GrooveEngine::getTemplate(const std::string& name) const {
    auto it = templates_.find(name);
    if (it != templates_.end()) {
        return &it->second;
    }
    return nullptr;
}

} // namespace kelly
