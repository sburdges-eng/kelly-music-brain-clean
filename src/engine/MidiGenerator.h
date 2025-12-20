#pragma once
/**
 * MidiGenerator.h
 *
 * Ported from Python: midi_generator.py
 * MIDI pipeline for generating therapeutic musical output.
 */

#include <string>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <algorithm>
#include <array>
#include "EmotionMapper.h"
#include "midi/GrooveEngine.h"  // Use explicit path to avoid confusion with engines/GrooveEngine.h
#include "common/MusicConstants.h"

namespace kelly {
using namespace MusicConstants;

// =============================================================================
// MIDI CONSTANTS
// =============================================================================

// MIDI_PPQ is defined in MusicConstants.h
constexpr int MIDI_CHANNEL_DRUMS = 9;   // Standard drum channel (10 in 1-indexed)

// =============================================================================
// CHORD STRUCTURE
// =============================================================================

struct Chord {
    std::vector<int> notes;             // MIDI note numbers
    std::string name;                   // e.g., "Am7", "Cmaj7"
    std::string quality;                // "major", "minor", "diminished", etc.
    int rootNote;                       // Root MIDI note
    int durationTicks;                  // Duration in ticks
    int startTick;                      // Start position in ticks
};

// =============================================================================
// MIDI NOTE (output format)
// =============================================================================

struct MidiNote {
    int tick;           // Start position in ticks
    int note;           // MIDI note number (0-127)
    int velocity;       // Velocity (0-127)
    int duration;       // Duration in ticks
    int channel;        // MIDI channel (0-15)
};

// =============================================================================
// SCALE DEFINITIONS
// =============================================================================

class ScaleDatabase {
public:
    // Interval patterns (semitones from root)
    static const std::array<int, 7>& getScale(const std::string& mode) {
        static const std::map<std::string, std::array<int, 7>> scales = {
            {"Ionian",      {0, 2, 4, 5, 7, 9, 11}},    // Major
            {"Dorian",      {0, 2, 3, 5, 7, 9, 10}},
            {"Phrygian",    {0, 1, 3, 5, 7, 8, 10}},
            {"Lydian",      {0, 2, 4, 6, 7, 9, 11}},
            {"Mixolydian",  {0, 2, 4, 5, 7, 9, 10}},
            {"Aeolian",     {0, 2, 3, 5, 7, 8, 10}},    // Natural Minor
            {"Locrian",     {0, 1, 3, 5, 6, 8, 10}},
            {"Major",       {0, 2, 4, 5, 7, 9, 11}},
            {"Minor",       {0, 2, 3, 5, 7, 8, 10}},
            {"Harmonic Minor", {0, 2, 3, 5, 7, 8, 11}},
            {"Melodic Minor", {0, 2, 3, 5, 7, 9, 11}},
        };

        auto it = scales.find(mode);
        if (it != scales.end()) {
            return it->second;
        }
        return scales.at("Aeolian");  // Default to natural minor
    }

    // Get note number for key (C=60, C#=61, etc.)
    static int keyToMidi(const std::string& key) {
        static const std::map<std::string, int> keyMap = {
            {"C", 60}, {"C#", 61}, {"Db", 61},
            {"D", 62}, {"D#", 63}, {"Eb", 63},
            {"E", 64}, {"F", 65}, {"F#", 66}, {"Gb", 66},
            {"G", 67}, {"G#", 68}, {"Ab", 68},
            {"A", 69}, {"A#", 70}, {"Bb", 70},
            {"B", 71},
            // Minor key names
            {"Cm", 60}, {"C#m", 61}, {"Dbm", 61},
            {"Dm", 62}, {"D#m", 63}, {"Ebm", 63},
            {"Em", 64}, {"Fm", 65}, {"F#m", 66}, {"Gbm", 66},
            {"Gm", 67}, {"G#m", 68}, {"Abm", 68},
            {"Am", 69}, {"A#m", 70}, {"Bbm", 70},
            {"Bm", 71}
        };

        auto it = keyMap.find(key);
        return it != keyMap.end() ? it->second : MIDI_C4;  // Default to C
    }
};

// =============================================================================
// CHORD PROGRESSION DATABASE
// =============================================================================

class ChordProgressionDatabase {
public:
    struct ProgressionEntry {
        std::string name;
        std::vector<int> degrees;       // Roman numeral degrees (1-7)
        std::vector<std::string> qualities;  // "major", "minor", "dim", etc.
        float emotionalValence;         // -1 to 1
        std::string description;
    };

    static std::vector<ProgressionEntry> getProgressionsForValence(float valence) {
        std::vector<ProgressionEntry> result;

        // All progressions
        static const std::vector<ProgressionEntry> allProgressions = {
            // Positive valence progressions
            {"I-V-vi-IV", {1, 5, 6, 4}, {"major", "major", "minor", "major"}, 0.7f, "Pop anthem"},
            {"I-IV-V-I", {1, 4, 5, 1}, {"major", "major", "major", "major"}, 0.8f, "Classic resolution"},
            {"I-IV-I-V", {1, 4, 1, 5}, {"major", "major", "major", "major"}, 0.6f, "Peaceful"},
            {"I-V-IV-V", {1, 5, 4, 5}, {"major", "major", "major", "major"}, 0.7f, "Uplifting"},

            // Neutral progressions
            {"I-vi-IV-V", {1, 6, 4, 5}, {"major", "minor", "major", "major"}, 0.3f, "50s progression"},
            {"I-V-vi-iii-IV", {1, 5, 6, 3, 4}, {"major", "major", "minor", "minor", "major"}, 0.2f, "Canon"},

            // Negative valence progressions (minor)
            {"i-bVI-bIII-bVII", {1, 6, 3, 7}, {"minor", "major", "major", "major"}, -0.6f, "Epic minor"},
            {"i-iv-bVI-V", {1, 4, 6, 5}, {"minor", "minor", "major", "major"}, -0.5f, "Melancholy"},
            {"i-bVII-bVI-bVII", {1, 7, 6, 7}, {"minor", "major", "major", "major"}, -0.7f, "Andalusian"},
            {"i-iv-v-i", {1, 4, 5, 1}, {"minor", "minor", "minor", "minor"}, -0.4f, "Natural minor"},

            // High tension/grief
            {"i-bVI-bIII-bVII", {1, 6, 3, 7}, {"minor", "major", "major", "major"}, -0.8f, "Grief descent"},
            {"i-bII-i-V", {1, 2, 1, 5}, {"minor", "major", "minor", "major"}, -0.9f, "Phrygian darkness"},
        };

        // Filter by valence proximity
        for (const auto& prog : allProgressions) {
            float valenceDiff = std::abs(prog.emotionalValence - valence);
            if (valenceDiff < 0.5f) {
                result.push_back(prog);
            }
        }

        // If nothing matched, return default based on sign
        if (result.empty()) {
            if (valence >= 0) {
                result.push_back(allProgressions[0]);  // I-V-vi-IV
            } else {
                result.push_back(allProgressions[6]);  // i-bVI-bIII-bVII
            }
        }

        return result;
    }
};

// =============================================================================
// MIDI GENERATOR CLASS
// =============================================================================

class MidiGenerator {
public:
    MidiGenerator(int tempo = 120, unsigned int seed = 0)
        : tempo_(tempo), grooveEngine_(seed) {
        if (seed == 0) {
            std::random_device rd;
            seed = rd();
        }
        rng_.seed(seed);
    }

    /**
     * Generate a chord progression.
     * Ported from Python generate_chord_progression()
     */
    std::vector<Chord> generateChordProgression(
        const std::string& key,
        const std::string& mode,
        int length = 4,
        float valence = 0.0f,
        bool allowDissonance = false
    ) {
        std::vector<Chord> chords;

        int root = ScaleDatabase::keyToMidi(key);
        const auto& scale = ScaleDatabase::getScale(mode);

        // Get progression template based on valence
        auto progressions = ChordProgressionDatabase::getProgressionsForValence(valence);

        // Select a progression
        std::uniform_int_distribution<size_t> progDist(0, progressions.size() - 1);
        const auto& progression = progressions[progDist(rng_)];

        int ticksPerBeat = MIDI_PPQ;
        int ticksPerChord = ticksPerBeat * 4 / std::max(1, static_cast<int>(progression.degrees.size()));

        int currentTick = 0;
        for (size_t i = 0; i < progression.degrees.size() && static_cast<int>(i) < length; ++i) {
            int degree = progression.degrees[i] - 1;  // 0-indexed
            const std::string& quality = progression.qualities[i];

            // Get root note for this degree
            int chordRoot = root + scale[degree % 7];

            // Build chord notes
            std::vector<int> notes;
            notes.push_back(chordRoot);  // Root

            if (quality == "major") {
                notes.push_back(chordRoot + 4);   // Major 3rd
                notes.push_back(chordRoot + 7);   // Perfect 5th
            } else if (quality == "minor") {
                notes.push_back(chordRoot + 3);   // Minor 3rd
                notes.push_back(chordRoot + 7);   // Perfect 5th
            } else if (quality == "dim" || quality == "diminished") {
                notes.push_back(chordRoot + 3);   // Minor 3rd
                notes.push_back(chordRoot + 6);   // Diminished 5th
            } else if (quality == "aug" || quality == "augmented") {
                notes.push_back(chordRoot + 4);   // Major 3rd
                notes.push_back(chordRoot + 8);   // Augmented 5th
            }

            // Add 7th if dissonance allowed
            if (allowDissonance) {
                if (quality == "major") {
                    notes.push_back(chordRoot + 11);  // Major 7th
                } else {
                    notes.push_back(chordRoot + 10);  // Minor 7th
                }
            }

            Chord chord;
            chord.notes = notes;
            chord.name = std::to_string(progression.degrees[i]) +
                        (quality == "minor" ? "m" : "");
            chord.quality = quality;
            chord.rootNote = chordRoot;
            chord.startTick = currentTick;
            chord.durationTicks = ticksPerChord;

            chords.push_back(chord);
            currentTick += ticksPerChord;
        }

        return chords;
    }

    /**
     * Generate MIDI notes from musical parameters.
     * Core generation function.
     */
    std::vector<MidiNote> generate(
        const MusicalParameters& params,
        int bars = 4,
        int channel = 0
    ) {
        std::vector<MidiNote> notes;

        // Generate chord progression
        std::string mode = params.modeSuggested;
        float valence = 0.0f;  // Will be passed from emotion
        auto chords = generateChordProgression(
            params.keySuggested,
            mode,
            bars,
            valence,
            params.dissonance > 0.5f
        );

        // Get groove template based on timing feel
        std::string grooveName = timingFeelToGroove(params.timingFeel);

        // Generate notes for each chord
        int ticksPerBar = MIDI_PPQ * 4;

        for (const auto& chord : chords) {
            // Add chord tones
            for (int note : chord.notes) {
                MidiNote mn;
                mn.tick = chord.startTick;
                mn.note = note;
                mn.duration = chord.durationTicks;
                mn.channel = channel;

                // Velocity based on params
                std::uniform_int_distribution<int> velDist(
                    params.velocityMin, params.velocityMax
                );
                mn.velocity = velDist(rng_);

                notes.push_back(mn);
            }

            // Add melodic elements based on density
            if (params.density > 0.3f) {
                addMelodicElements(notes, chord, params, channel);
            }

            // Add bass line
            addBassLine(notes, chord, params, channel);
        }

        // Apply humanization
        HumanizationSettings humanSettings;
        humanSettings.timingVariation = params.timingVariation;
        humanSettings.velocityVariation = params.velocityVariation;

        // Convert to MidiNoteEvent for humanization
        std::vector<MidiNoteEvent> events;
        for (const auto& mn : notes) {
            MidiNoteEvent evt;
            evt.tick = mn.tick;
            evt.note = mn.note;
            evt.velocity = mn.velocity;
            evt.duration = mn.duration;
            evt.channel = mn.channel;
            events.push_back(evt);
        }

        // Humanize
        float complexity = params.dissonance;
        float vulnerability = params.dynamicsRange;
        auto humanized = grooveEngine_.humanize(
            events, complexity, vulnerability, MIDI_PPQ, humanSettings
        );

        // Convert back
        notes.clear();
        for (const auto& evt : humanized) {
            MidiNote mn;
            mn.tick = evt.humanizedTick;
            mn.note = evt.note;
            mn.velocity = evt.humanizedVelocity;
            mn.duration = evt.humanizedDuration;
            mn.channel = evt.channel;
            notes.push_back(mn);
        }

        return notes;
    }

    /**
     * Generate from emotion parameters directly
     */
    std::vector<MidiNote> generateFromEmotion(
        float valence,
        float arousal,
        float intensity,
        const std::string& key = "C",
        const std::string& mode = "Aeolian",
        int bars = 4
    ) {
        EmotionalState state;
        state.valence = valence;
        state.arousal = arousal;
        state.intensity = intensity;

        EmotionMapper mapper;
        auto params = mapper.mapToParameters(state);
        params.keySuggested = key;
        params.modeSuggested = mode;

        return generate(params, bars);
    }

    // Accessors
    int tempo() const { return tempo_; }
    void setTempo(int tempo) { tempo_ = tempo; }

    GrooveEngine& grooveEngine() { return grooveEngine_; }
    const GrooveEngine& grooveEngine() const { return grooveEngine_; }

private:
    int tempo_;
    GrooveEngine grooveEngine_;
    std::mt19937 rng_;

    std::string timingFeelToGroove(TimingFeel feel) const {
        switch (feel) {
            case TimingFeel::Swung: return "swing";
            case TimingFeel::LaidBack: return "laidback";
            case TimingFeel::Pushed: return "driving";
            case TimingFeel::Rubato: return "lofi";
            default: return "straight";
        }
    }

    void addMelodicElements(
        std::vector<MidiNote>& notes,
        const Chord& chord,
        const MusicalParameters& params,
        int channel
    ) {
        // Add passing tones and melodic movement
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);

        if (prob(rng_) > params.spaceProbability) {
            int ticksPerBeat = MIDI_PPQ;
            int numBeats = chord.durationTicks / ticksPerBeat;

            for (int beat = 1; beat < numBeats; ++beat) {
                if (prob(rng_) > params.spaceProbability) {
                    // Choose a chord tone an octave up
                    std::uniform_int_distribution<size_t> noteDist(0, chord.notes.size() - 1);
                    int melodicNote = chord.notes[noteDist(rng_)] + 12;

                    MidiNote mn;
                    mn.tick = chord.startTick + beat * ticksPerBeat;
                    mn.note = melodicNote;
                    mn.duration = ticksPerBeat / 2;
                    mn.channel = channel;

                    std::uniform_int_distribution<int> velDist(
                        params.velocityMin,
                        static_cast<int>(params.velocityMax * 0.8f)
                    );
                    mn.velocity = velDist(rng_);

                    notes.push_back(mn);
                }
            }
        }
    }

    void addBassLine(
        std::vector<MidiNote>& notes,
        const Chord& chord,
        const MusicalParameters& params,
        int channel
    ) {
        // Add bass note (root, two octaves down)
        int bassNote = chord.rootNote - 24;
        if (bassNote < 28) bassNote += 12;  // Keep in reasonable range

        MidiNote mn;
        mn.tick = chord.startTick;
        mn.note = bassNote;
        mn.duration = chord.durationTicks;
        mn.channel = channel;

        // Bass slightly louder
        std::uniform_int_distribution<int> velDist(
            static_cast<int>(params.velocityMin * 1.1f),
            params.velocityMax
        );
        mn.velocity = velDist(rng_);

        notes.push_back(mn);
    }
};

} // namespace kelly
