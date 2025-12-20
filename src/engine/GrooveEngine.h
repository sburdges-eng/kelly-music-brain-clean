#pragma once
/**
 * GrooveEngine.h
 * 
 * Ported from Python: kellymidicompanion_groove_engine.py
 * Provides groove templates and humanization for MIDI generation.
 */

#include <string>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <algorithm>

namespace kelly {

// =============================================================================
// GROOVE TEMPLATE (from Python GrooveTemplate dataclass)
// =============================================================================

struct GrooveStep {
    float position;     // 0.0-1.0 within beat
    int velocity;       // MIDI velocity 0-127
    float duration;     // Note duration as fraction of beat
    bool isAccent;      // Is this an accented beat
};

struct GrooveTemplate {
    std::string name;
    std::pair<int, int> timeSignature = {4, 4};
    std::vector<GrooveStep> pattern;
    float swing = 0.0f;         // 0.0 = straight, 0.66 = triplet swing
    float shuffle = 0.0f;       // Shuffle amount
    float pushPull = 0.0f;      // -1.0 = laid back, +1.0 = pushed
    
    // Get number of steps per bar
    int stepsPerBar() const {
        return static_cast<int>(pattern.size()) * timeSignature.first;
    }
};

// =============================================================================
// HUMANIZATION SETTINGS (from Python GrooveSettings)
// =============================================================================

struct HumanizationSettings {
    // Timing variations
    float timingVariation = 0.02f;      // ±percentage of beat position
    float microTimingBias = 0.0f;       // Consistent early/late tendency
    bool applySwing = true;
    
    // Velocity variations
    float velocityVariation = 0.1f;     // ±percentage of velocity
    float accentStrength = 1.2f;        // Multiplier for accented notes
    int velocityFloor = 30;             // Minimum velocity
    int velocityCeiling = 127;          // Maximum velocity
    
    // Ghost notes
    float ghostNoteProbability = 0.1f;  // Chance of adding ghost notes
    float ghostNoteVelocity = 0.4f;     // Ghost note velocity (fraction of main)
    
    // Dynamics
    float dynamicWave = 0.0f;           // 0-1, adds crescendo/decrescendo waves
    int dynamicWavePeriod = 4;          // Bars per dynamic cycle
    
    // Groove pocket
    float pocketTightness = 0.5f;       // 0 = loose, 1 = tight
};

// =============================================================================
// MIDI NOTE EVENT (for humanization input/output)
// =============================================================================

struct MidiNoteEvent {
    int tick;           // Position in ticks
    int note;           // MIDI note number
    int velocity;       // MIDI velocity
    int duration;       // Duration in ticks
    int channel = 0;    // MIDI channel
    
    // Humanized versions (applied separately)
    int humanizedTick = 0;
    int humanizedVelocity = 0;
    int humanizedDuration = 0;
};

// =============================================================================
// GROOVE ENGINE CLASS
// =============================================================================

// Renamed to avoid conflict with src/midi/GrooveEngine
class GrooveTemplateEngine {
public:
    GrooveTemplateEngine(unsigned int seed = 0) {
        if (seed == 0) {
            std::random_device rd;
            seed = rd();
        }
        rng_.seed(seed);
        initializeTemplates();
    }
    
    /**
     * Get a groove template by name
     */
    const GrooveTemplate& getTemplate(const std::string& name) const {
        auto it = templates_.find(name);
        if (it != templates_.end()) {
            return it->second;
        }
        return templates_.at("straight");
    }
    
    /**
     * Get all available template names
     */
    std::vector<std::string> getTemplateNames() const {
        std::vector<std::string> names;
        for (const auto& pair : templates_) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    /**
     * Humanize a vector of MIDI events.
     * Core algorithm ported from Python humanize_drums()
     * 
     * @param events Input events to humanize
     * @param complexity Timing chaos level (0.0-1.0), maps to vulnerability
     * @param vulnerability Dynamic fragility (0.0-1.0)
     * @param ppq Pulses per quarter note
     * @param settings Fine-grained humanization settings
     * @return Humanized events
     */
    std::vector<MidiNoteEvent> humanize(
        const std::vector<MidiNoteEvent>& events,
        float complexity,
        float vulnerability,
        int ppq = 480,
        const HumanizationSettings& settings = HumanizationSettings()
    ) {
        std::vector<MidiNoteEvent> result = events;
        
        // Create distributions based on settings
        float timingStdDev = settings.timingVariation * ppq * complexity;
        float velocityStdDev = settings.velocityVariation * 127.0f * vulnerability;
        
        std::normal_distribution<float> timingDist(0.0f, timingStdDev);
        std::normal_distribution<float> velocityDist(0.0f, velocityStdDev);
        std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
        
        // Calculate bar positions for dynamic wave
        int ticksPerBar = ppq * 4; // Assuming 4/4
        
        for (size_t i = 0; i < result.size(); ++i) {
            auto& event = result[i];
            
            // --- TIMING HUMANIZATION ---
            float timingOffset = timingDist(rng_);
            
            // Apply micro-timing bias
            timingOffset += settings.microTimingBias * ppq * 0.05f;
            
            // Apply pocket tightness (reduce variation for tight pocket)
            timingOffset *= (1.0f - settings.pocketTightness * 0.5f);
            
            // Quantize slightly based on grid position
            int gridPosition = event.tick % ppq;
            if (gridPosition == 0 || gridPosition == ppq / 2) {
                // On main beats, reduce variation
                timingOffset *= 0.5f;
            }
            
            event.humanizedTick = event.tick + static_cast<int>(timingOffset);
            if (event.humanizedTick < 0) event.humanizedTick = 0;
            
            // --- VELOCITY HUMANIZATION ---
            float velocityOffset = velocityDist(rng_);
            
            // Apply dynamic wave
            if (settings.dynamicWave > 0.0f) {
                float barPosition = static_cast<float>(event.tick % (ticksPerBar * settings.dynamicWavePeriod)) 
                                  / static_cast<float>(ticksPerBar * settings.dynamicWavePeriod);
                float waveValue = std::sin(barPosition * 2.0f * 3.14159f);
                velocityOffset += waveValue * settings.dynamicWave * 20.0f;
            }
            
            // Check for accent (on main beats)
            bool isMainBeat = (event.tick % ppq) == 0;
            float accentMultiplier = isMainBeat ? settings.accentStrength : 1.0f;
            
            int newVelocity = static_cast<int>(
                event.velocity * accentMultiplier + velocityOffset
            );
            event.humanizedVelocity = std::clamp(
                newVelocity, 
                settings.velocityFloor, 
                settings.velocityCeiling
            );
            
            // --- DURATION HUMANIZATION ---
            // Slight variation in note length
            float durationVariation = 1.0f + (uniformDist(rng_) - 0.5f) * 0.1f * complexity;
            event.humanizedDuration = static_cast<int>(event.duration * durationVariation);
            if (event.humanizedDuration < 1) event.humanizedDuration = 1;
        }
        
        return result;
    }
    
    /**
     * Apply swing to tick position
     */
    int applySwing(int tick, float swingAmount, int ppq) const {
        if (swingAmount == 0.0f) return tick;
        
        int beatPosition = tick % ppq;
        int beatStart = tick - beatPosition;
        
        // Swing affects offbeats (second eighth note)
        int halfBeat = ppq / 2;
        if (beatPosition >= halfBeat && beatPosition < ppq) {
            // This is an offbeat - delay it
            float delay = (swingAmount - 0.5f) * halfBeat;
            return beatStart + halfBeat + static_cast<int>(delay);
        }
        
        return tick;
    }
    
    /**
     * Generate groove pattern for a given number of bars
     */
    std::vector<GrooveStep> generatePattern(
        const std::string& templateName,
        int bars,
        int ppq = 480
    ) const {
        const GrooveTemplate& templ = getTemplate(templateName);
        std::vector<GrooveStep> result;
        
        int stepsPerBeat = static_cast<int>(templ.pattern.size());
        int beatsPerBar = templ.timeSignature.first;
        int totalBeats = bars * beatsPerBar;
        
        for (int beat = 0; beat < totalBeats; ++beat) {
            for (int step = 0; step < stepsPerBeat; ++step) {
                GrooveStep gs = templ.pattern[step];
                gs.position = beat + templ.pattern[step].position;
                result.push_back(gs);
            }
        }
        
        return result;
    }
    
    /**
     * Create a custom template
     */
    void addTemplate(const std::string& name, const GrooveTemplate& templ) {
        templates_[name] = templ;
    }

private:
    std::map<std::string, GrooveTemplate> templates_;
    std::mt19937 rng_;
    
    void initializeTemplates() {
        // Straight 4/4 (from Python)
        templates_["straight"] = {
            "Straight",
            {4, 4},
            {
                {0.0f, 100, 0.25f, true},
                {0.25f, 80, 0.25f, false},
                {0.5f, 100, 0.25f, false},
                {0.75f, 80, 0.25f, false}
            },
            0.0f, 0.0f, 0.0f
        };
        
        // Swing (from Python)
        templates_["swing"] = {
            "Swing",
            {4, 4},
            {
                {0.0f, 100, 0.33f, true},
                {0.33f, 80, 0.33f, false},
                {0.5f, 100, 0.33f, false},
                {0.83f, 80, 0.17f, false}
            },
            0.66f, 0.0f, 0.0f
        };
        
        // Syncopated (from Python)
        templates_["syncopated"] = {
            "Syncopated",
            {4, 4},
            {
                {0.0f, 100, 0.125f, true},
                {0.125f, 60, 0.125f, false},
                {0.375f, 90, 0.25f, false},
                {0.625f, 85, 0.25f, false},
                {0.875f, 70, 0.125f, false}
            },
            0.0f, 0.0f, 0.0f
        };
        
        // Laid back (for grief/sadness)
        templates_["laidback"] = {
            "Laid Back",
            {4, 4},
            {
                {0.0f, 90, 0.3f, true},
                {0.25f, 70, 0.25f, false},
                {0.5f, 85, 0.25f, false},
                {0.75f, 65, 0.25f, false}
            },
            0.0f, 0.0f, -0.3f  // Pushed back
        };
        
        // Driving (for anger/intensity)
        templates_["driving"] = {
            "Driving",
            {4, 4},
            {
                {0.0f, 110, 0.2f, true},
                {0.25f, 100, 0.2f, true},
                {0.5f, 110, 0.2f, true},
                {0.75f, 100, 0.2f, true}
            },
            0.0f, 0.0f, 0.15f  // Pushed forward
        };
        
        // Halftime (for contemplative moods)
        templates_["halftime"] = {
            "Half Time",
            {4, 4},
            {
                {0.0f, 100, 0.5f, true},
                {0.5f, 90, 0.5f, false}
            },
            0.0f, 0.0f, 0.0f
        };
        
        // Shuffle
        templates_["shuffle"] = {
            "Shuffle",
            {4, 4},
            {
                {0.0f, 100, 0.22f, true},
                {0.22f, 75, 0.11f, false},
                {0.33f, 90, 0.22f, false},
                {0.55f, 75, 0.11f, false},
                {0.66f, 100, 0.22f, false},
                {0.88f, 75, 0.12f, false}
            },
            0.0f, 0.5f, 0.0f
        };
        
        // 6/8 compound
        templates_["compound68"] = {
            "6/8 Compound",
            {6, 8},
            {
                {0.0f, 100, 0.33f, true},
                {0.33f, 70, 0.33f, false},
                {0.66f, 70, 0.34f, false}
            },
            0.0f, 0.0f, 0.0f
        };
        
        // Breakbeat-style
        templates_["breakbeat"] = {
            "Breakbeat",
            {4, 4},
            {
                {0.0f, 110, 0.25f, true},
                {0.25f, 60, 0.125f, false},
                {0.375f, 90, 0.125f, false},
                {0.5f, 50, 0.125f, false},
                {0.625f, 100, 0.25f, true},
                {0.875f, 75, 0.125f, false}
            },
            0.0f, 0.0f, 0.0f
        };
        
        // Lo-fi (intentionally imperfect)
        templates_["lofi"] = {
            "Lo-Fi",
            {4, 4},
            {
                {0.0f, 85, 0.28f, true},
                {0.26f, 65, 0.24f, false},
                {0.52f, 80, 0.26f, false},
                {0.77f, 60, 0.23f, false}
            },
            0.55f, 0.1f, -0.1f
        };
    }
};

} // namespace kelly
