#pragma once
/**
 * Kelly.h
 *
 * Unified API for the Kelly MIDI Companion system.
 * Includes all ported Python modules in a single header.
 *
 * Usage:
 *   #include "Kelly.h"
 *
 *   kelly::KellyBrain brain;
 *   auto result = brain.processWound("feeling of loss", 0.8f);
 *   auto midi = brain.generateMidi(result, 4);
 */

#include "EmotionMapper.h"
#include "GrooveEngine.h"  // This is GrooveTemplateEngine (renamed)
#include "IntentProcessor.h"
#include "../midi/MidiGenerator.h"  // Use the correct MidiGenerator from midi/ directory
#include "VADSystem.h"
#include "EmotionThesaurus.h"

namespace kelly {

// =============================================================================
// KELLY BRAIN - High-level API (from Python MusicBrain class)
// =============================================================================

class KellyBrain {
public:
    KellyBrain(int tempo = 120, unsigned int seed = 0)
        : midiGenerator_(tempo, seed),
          vadSystem_(&intentProcessor_.thesaurus()) {}  // thesaurus() returns const&, which is fine

    /**
     * Process an emotional wound and get complete intent result.
     *
     * @param description Text description of the wound/feeling
     * @param intensity Emotional intensity (0.0-1.0)
     * @param source "internal" or "external"
     * @return Complete IntentResult with emotion, rule-breaks, and parameters
     */
    IntentResult processWound(
        const std::string& description,
        float intensity = 0.7f,
        const std::string& source = "internal"
    ) {
        Wound wound;
        wound.description = description;
        wound.intensity = intensity;
        wound.source = source;

        return intentProcessor_.processIntent(wound);
    }

    /**
     * Generate MIDI from an IntentResult.
     *
     * @param result Result from processWound()
     * @param bars Number of bars to generate
     * @param channel MIDI channel (0-15)
     * @return Vector of MidiNote events
     */
    std::vector<MidiNote> generateMidi(
        const IntentResult& result,
        int bars = 4,
        int channel = 0
    ) {
        return midiGenerator_.generate(result.musicalParams, bars, channel);
    }

    /**
     * Quick generation from text description.
     * Combines processWound() and generateMidi() in one call.
     *
     * @param description Text description of the wound/feeling
     * @param intensity Emotional intensity (0.0-1.0)
     * @param bars Number of bars to generate
     * @return Pair of IntentResult and MIDI notes
     */
    std::pair<IntentResult, std::vector<MidiNote>> quickGenerate(
        const std::string& description,
        float intensity = 0.7f,
        int bars = 4
    ) {
        auto result = processWound(description, intensity);
        auto midi = generateMidi(result, bars);
        return {result, midi};
    }

    /**
     * Generate from raw valence/arousal/intensity values.
     * Bypasses wound processing for direct emotional input.
     */
    std::vector<MidiNote> generateFromVAI(
        float valence,
        float arousal,
        float intensity,
        const std::string& key = "C",
        const std::string& mode = "Aeolian",
        int bars = 4
    ) {
        return midiGenerator_.generateFromEmotion(
            valence, arousal, intensity, key, mode, bars
        );
    }

    /**
     * Find the nearest emotion for given parameters.
     */
    const EmotionNode* findEmotion(float valence, float arousal, float intensity) const {
        return intentProcessor_.thesaurus().findNearest(valence, arousal, intensity);
    }

    /**
     * Find emotion by name.
     */
    const EmotionNode* findEmotionByName(const std::string& name) const {
        return intentProcessor_.thesaurus().findByName(name);
    }

    /**
     * Get all emotions in a category.
     */
    std::vector<const EmotionNode*> getEmotionsByCategory(EmotionCategory category) const {
        return intentProcessor_.thesaurus().getByCategory(category);
    }

    /**
     * Get list of available groove templates.
     */
    std::vector<std::string> getGrooveTemplates() const {
        return midiGenerator_.grooveEngine().getTemplateNames();
    }

    // Component accessors
    IntentProcessor& intentProcessor() { return intentProcessor_; }
    const IntentProcessor& intentProcessor() const { return intentProcessor_; }

    MidiGenerator& midiGenerator() { return midiGenerator_; }
    const MidiGenerator& midiGenerator() const { return midiGenerator_; }

    // Settings
    int tempo() const { return midiGenerator_.tempo(); }
    void setTempo(int tempo) { midiGenerator_.setTempo(tempo); }

    void clearHistory() {
        intentProcessor_.clearHistory();
        vadSystem_.clearHistory();
    }

    // =============================================================================
    // VAD SYSTEM INTEGRATION
    // =============================================================================

    /**
     * Process emotion ID with full VAD system (includes biometrics, context, trends)
     * @param emotionId Emotion ID from thesaurus
     * @param intensityModifier Intensity adjustment (0.0-2.0)
     * @param generateOSC Whether to generate OSC output
     * @return VAD system processing result
     */
    VADSystem::ProcessingResult processEmotionWithVAD(
        int emotionId,
        float intensityModifier = 1.0f,
        bool generateOSC = false
    ) {
        return vadSystem_.processEmotionId(emotionId, intensityModifier, generateOSC);
    }

    /**
     * Process biometric data with VAD system
     * @param biometricData Biometric readings
     * @param generateOSC Whether to generate OSC output
     * @return VAD system processing result
     */
    VADSystem::ProcessingResult processBiometricsWithVAD(
        const BiometricInput::BiometricData& biometricData,
        bool generateOSC = false
    ) {
        return vadSystem_.processBiometrics(biometricData, generateOSC);
    }

    /**
     * Process blended emotion + biometric input
     * @param emotionId Emotion ID
     * @param biometricData Biometric readings
     * @param emotionWeight Weight for emotion (0.0-1.0)
     * @param generateOSC Whether to generate OSC output
     * @return VAD system processing result
     */
    VADSystem::ProcessingResult processBlendedVAD(
        int emotionId,
        const BiometricInput::BiometricData& biometricData,
        float emotionWeight = 0.7f,
        bool generateOSC = false
    ) {
        return vadSystem_.processBlended(emotionId, biometricData, emotionWeight, generateOSC);
    }

    /**
     * Get current VAD trends
     */
    TrendMetrics getVADTrends() const {
        return vadSystem_.getCurrentTrends();
    }

    /**
     * Get resonance metrics
     */
    ResonanceMetrics getResonance() const {
        return vadSystem_.getResonance();
    }

    /**
     * Enable/disable context-aware adjustments (circadian, time-of-day)
     */
    void setContextAware(bool enabled) {
        vadSystem_.setContextAware(enabled);
    }

    /**
     * Set current time for context adjustments
     */
    void setCurrentTime(int hourOfDay, int dayOfWeek = -1) {
        vadSystem_.setCurrentTime(hourOfDay, dayOfWeek);
    }

    /**
     * Get smoothed VAD state
     */
    VADState getSmoothedVAD() const {
        return vadSystem_.getSmoothedVAD();
    }

    /**
     * Get VAD system accessor
     */
    VADSystem& vadSystem() { return vadSystem_; }
    const VADSystem& vadSystem() const { return vadSystem_; }

private:
    IntentProcessor intentProcessor_;
    MidiGenerator midiGenerator_;
    VADSystem vadSystem_;
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Convert MIDI note number to note name (e.g., 60 -> "C4")
 */
inline std::string midiNoteToName(int noteNumber) {
    static const char* noteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    int octave = (noteNumber / 12) - 1;
    int noteIndex = noteNumber % 12;
    return std::string(noteNames[noteIndex]) + std::to_string(octave);
}

/**
 * Convert note name to MIDI number (e.g., "C4" -> 60)
 */
inline int noteNameToMidi(const std::string& name) {
    static const std::map<std::string, int> noteOffsets = {
        {"C", 0}, {"C#", 1}, {"Db", 1}, {"D", 2}, {"D#", 3}, {"Eb", 3},
        {"E", 4}, {"F", 5}, {"F#", 6}, {"Gb", 6}, {"G", 7}, {"G#", 8},
        {"Ab", 8}, {"A", 9}, {"A#", 10}, {"Bb", 10}, {"B", 11}
    };

    // Parse note and octave
    std::string notePart;
    int octave = 4;  // Default

    for (size_t i = 0; i < name.size(); ++i) {
        if (std::isdigit(name[i]) || name[i] == '-') {
            notePart = name.substr(0, i);
            octave = std::stoi(name.substr(i));
            break;
        }
    }
    if (notePart.empty()) notePart = name;

    auto it = noteOffsets.find(notePart);
    if (it != noteOffsets.end()) {
        return (octave + 1) * 12 + it->second;
    }
    return 60;  // Default to C4
}

/**
 * Convert ticks to milliseconds at given tempo and PPQ.
 */
inline double ticksToMs(int ticks, int tempo, int ppq = MIDI_PPQ) {
    double msPerBeat = 60000.0 / tempo;
    double msPerTick = msPerBeat / ppq;
    return ticks * msPerTick;
}

/**
 * Convert milliseconds to ticks at given tempo and PPQ.
 */
inline int msToTicks(double ms, int tempo, int ppq = MIDI_PPQ) {
    double msPerBeat = 60000.0 / tempo;
    double msPerTick = msPerBeat / ppq;
    return static_cast<int>(ms / msPerTick);
}

/**
 * Get category name as string.
 */
inline std::string categoryToString(EmotionCategory category) {
    switch (category) {
        case EmotionCategory::Joy: return "Joy";
        case EmotionCategory::Sadness: return "Sadness";
        case EmotionCategory::Anger: return "Anger";
        case EmotionCategory::Fear: return "Fear";
        case EmotionCategory::Surprise: return "Surprise";
        case EmotionCategory::Disgust: return "Disgust";
        case EmotionCategory::Trust: return "Trust";
        case EmotionCategory::Anticipation: return "Anticipation";
        case EmotionCategory::Complex: return "Complex";
        default: return "Unknown";
    }
}

/**
 * Get timing feel as string.
 */
inline std::string timingFeelToString(TimingFeel feel) {
    switch (feel) {
        case TimingFeel::Straight: return "Straight";
        case TimingFeel::Swung: return "Swung";
        case TimingFeel::LaidBack: return "Laid Back";
        case TimingFeel::Pushed: return "Pushed";
        case TimingFeel::Rubato: return "Rubato";
        default: return "Unknown";
    }
}

/**
 * Get rule break type as string.
 */
inline std::string ruleBreakTypeToString(RuleBreakType type) {
    switch (type) {
        case RuleBreakType::Harmony: return "Harmony";
        case RuleBreakType::Rhythm: return "Rhythm";
        case RuleBreakType::Dynamics: return "Dynamics";
        case RuleBreakType::Structure: return "Structure";
        case RuleBreakType::Voice_Leading: return "Voice Leading";
        case RuleBreakType::Texture: return "Texture";
        case RuleBreakType::Range: return "Range";
        default: return "Unknown";
    }
}

} // namespace kelly
