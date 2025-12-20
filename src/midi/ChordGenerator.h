#pragma once

#include "common/Types.h"
#include "engines/VoiceLeading.h"
#include <vector>
#include <string>
#include <random>
#include <map>
#include <memory>
#include <mutex>

namespace kelly {

/**
 * ChordGenerator - CRITICAL
 * 
 * Generates chord progressions from:
 * - Emotion (VAD coordinates)
 * - Voice leading principles
 * - Progression families (jazz, pop, rock, blues, etc.)
 * 
 * Features:
 * - Emotion-to-progression mapping
 * - Smooth voice leading between chords
 * - Progression family selection by emotional character
 * - Rule-break support for dissonance/chromaticism
 * - Chord extensions based on intensity
 */
class ChordGenerator {
public:
    ChordGenerator();
    
    /**
     * Generate chord progression from emotional intent
     * Primary entry point - uses emotion, voice leading, and progression families
     */
    std::vector<Chord> generate(const IntentResult& intent, int bars = 4);
    
    /**
     * Generate progression with explicit parameters
     */
    std::vector<Chord> generateProgression(
        const std::string& mode,
        int rootNote,
        int bars,
        bool allowDissonance,
        float intensity
    );
    
    /**
     * Generate from progression family name
     * e.g., "I-V-vi-IV", "ii-V-I", "12-bar-blues"
     */
    std::vector<Chord> generateFromFamily(
        const std::string& familyName,
        const std::string& mode,
        int rootNote,
        int bars,
        float intensity = 0.5f
    );
    
    /**
     * Apply voice leading to existing chord progression
     */
    std::vector<Chord> applyVoiceLeading(
        const std::vector<Chord>& chords,
        VoiceLeadingStyle style = VoiceLeadingStyle::Smooth
    );

private:
    mutable std::mutex mutex_;  // Thread safety for generation
    mutable std::mt19937 rng_;
    std::unique_ptr<VoiceLeadingEngine> voiceLeadingEngine_;
    
    // =========================================================================
    // PROGRESSION FAMILIES
    // =========================================================================
    
    struct ProgressionFamily {
        std::string name;
        std::string category;  // "universal", "jazz", "rock", "minor", etc.
        std::vector<int> degrees;  // Scale degrees (0=I, 2=ii, 5=IV, etc.)
        std::vector<std::string> romanNumerals;
        float valenceRange[2];     // Min/max valence this works for
        float arousalRange[2];     // Min/max arousal this works for
        std::string feel;
        bool useExtensions;        // Whether to add 7ths, 9ths, etc.
    };
    
    std::vector<ProgressionFamily> families_;
    std::map<std::string, ProgressionFamily> familyMap_;
    
    // =========================================================================
    // EMOTION-BASED TEMPLATES (legacy support)
    // =========================================================================
    
    struct ProgressionTemplate {
        std::string name;
        std::vector<int> degrees;
        float valenceRange[2];
        float arousalRange[2];
    };
    
    std::vector<ProgressionTemplate> templates_;
    
    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    
    void initializeProgressionFamilies();
    void initializeTemplates();
    
    // =========================================================================
    // PROGRESSION SELECTION
    // =========================================================================
    
    /**
     * Select best progression family for emotion
     */
    const ProgressionFamily* selectFamilyForEmotion(
        float valence,
        float arousal,
        const std::string& mode
    ) const;
    
    /**
     * Score a progression family for emotion match
     */
    float scoreFamilyForEmotion(
        const ProgressionFamily& family,
        float valence,
        float arousal
    ) const;
    
    // =========================================================================
    // CHORD BUILDING
    // =========================================================================
    
    /**
     * Build a chord from scale degree
     */
    Chord buildChord(
        int degree,
        const std::string& mode,
        int rootNote,
        double startBeat,
        double duration,
        bool addExtension,
        int inversion = 0
    );
    
    /**
     * Get scale intervals for a mode
     */
    std::vector<int> getScaleIntervals(const std::string& mode) const;
    
    /**
     * Convert scale degree to MIDI note number
     */
    int degreeToMidiNote(
        int degree,
        const std::string& mode,
        int rootNote,
        int octave = 0
    ) const;
    
    /**
     * Build chord with extensions (7th, 9th, etc.)
     */
    void addExtensions(
        Chord& chord,
        int degree,
        const std::string& mode,
        float intensity
    );
    
    // =========================================================================
    // VOICE LEADING
    // =========================================================================
    
    /**
     * Apply voice leading to chord progression
     */
    void applyVoiceLeadingToProgression(
        std::vector<Chord>& chords,
        VoiceLeadingStyle style
    );
    
    /**
     * Convert chord to voicing (pitches in specific octaves)
     */
    std::vector<int> chordToVoicing(
        const Chord& chord,
        int bassNote,
        int numVoices = 4
    ) const;
    
    // =========================================================================
    // RULE BREAKS & MODIFICATIONS
    // =========================================================================
    
    /**
     * Apply dissonance based on rule-breaks
     */
    void applyDissonance(std::vector<Chord>& chords, float severity);
    
    /**
     * Add chromatic passing chords
     */
    void addChromaticism(std::vector<Chord>& chords, float severity);
    
    /**
     * Apply modal interchange (borrowed chords)
     */
    void applyModalInterchange(
        std::vector<Chord>& chords,
        const std::string& baseMode,
        float intensity
    );
    
    // =========================================================================
    // UTILITIES
    // =========================================================================
    
    /**
     * Determine root note from emotion
     */
    int selectRootNote(float valence, float intensity) const;
    
    /**
     * Generate chord name from pitches
     */
    std::string generateChordName(
        const std::vector<int>& pitches,
        int rootNote
    ) const;
    
    /**
     * Check if progression should loop
     */
    bool shouldLoopProgression(int bars, const std::vector<int>& degrees) const;
};

} // namespace kelly
