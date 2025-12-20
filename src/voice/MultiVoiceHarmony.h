#pragma once

#include "voice/VoiceSynthesizer.h"
#include "voice/VocoderEngine.h"
#include "common/Types.h"
#include <vector>
#include <memory>

namespace kelly {

/**
 * MultiVoiceHarmony - Generate multiple voice parts for harmony
 *
 * Creates harmonized vocal parts (soprano, alto, tenor, bass)
 * from a melody and emotion context. Each voice part can have
 * different formant characteristics and pitch ranges.
 */
class MultiVoiceHarmony {
public:
    enum class VoicePart {
        Soprano,   // Highest voice (typically C5-C7)
        Alto,      // High voice (typically G4-C6)
        Tenor,     // Mid-high voice (typically C4-G5)
        Bass       // Low voice (typically C3-F4)
    };

    struct VoicePartConfig {
        VoicePart part;
        int pitchOffset;          // Semitones relative to melody (0 = unison, 7 = fifth, -12 = octave down)
        float volume;             // Volume level (0.0 to 1.0)
        VoiceType voiceType;      // Male/Female/Child/Neutral
        float formantShift;       // Formant shift multiplier
        float vibratoDepth;       // Vibrato depth for this voice
        float vibratoRate;        // Vibrato rate for this voice
    };

    MultiVoiceHarmony();
    ~MultiVoiceHarmony() = default;

    /**
     * Generate harmony voices from a melody
     * @param melodyNotes Original melody notes
     * @param emotion Emotion context for harmony generation
     * @param harmonyType Type of harmony (parallel, counterpoint, block chords, etc.)
     * @return Map of voice part to harmonized notes
     */
    std::map<VoicePart, std::vector<VoiceSynthesizer::VocalNote>> generateHarmony(
        const std::vector<VoiceSynthesizer::VocalNote>& melodyNotes,
        const EmotionNode& emotion,
        const std::string& harmonyType = "parallel"
    );

    /**
     * Generate 4-part harmony (SATB)
     * @param melodyNotes Original melody notes (typically assigned to soprano)
     * @param emotion Emotion context
     * @return Map with Soprano, Alto, Tenor, Bass parts
     */
    std::map<VoicePart, std::vector<VoiceSynthesizer::VocalNote>> generateSATB(
        const std::vector<VoiceSynthesizer::VocalNote>& melodyNotes,
        const EmotionNode& emotion
    );

    /**
     * Configure a voice part
     * @param part Voice part to configure
     * @param config Configuration parameters
     */
    void setVoicePartConfig(VoicePart part, const VoicePartConfig& config);

    /**
     * Get voice part configuration
     */
    VoicePartConfig getVoicePartConfig(VoicePart part) const;

    /**
     * Set harmony style (parallel thirds, parallel sixths, block chords, etc.)
     */
    void setHarmonyStyle(const std::string& style) { harmonyStyle_ = style; }

    /**
     * Synthesize all harmony voices to audio
     * @param harmonyParts Map of voice parts to notes
     * @param sampleRate Audio sample rate
     * @param emotion Optional emotion for vocal characteristics
     * @return Mixed audio buffer with all voices
     */
    std::vector<float> synthesizeHarmony(
        const std::map<VoicePart, std::vector<VoiceSynthesizer::VocalNote>>& harmonyParts,
        double sampleRate = 44100.0,
        const EmotionNode* emotion = nullptr
    );

private:
    // Voice part configurations
    std::map<VoicePart, VoicePartConfig> voiceConfigs_;
    std::string harmonyStyle_ = "parallel";

    // Voice synthesizers for each part
    std::map<VoicePart, std::unique_ptr<VoiceSynthesizer>> synthesizers_;

    /**
     * Generate parallel harmony (same intervals throughout)
     */
    std::vector<VoiceSynthesizer::VocalNote> generateParallelHarmony(
        const std::vector<VoiceSynthesizer::VocalNote>& melody,
        int intervalSemitones
    );

    /**
     * Generate block chord harmony (chord tones)
     */
    std::vector<VoiceSynthesizer::VocalNote> generateBlockChordHarmony(
        const std::vector<VoiceSynthesizer::VocalNote>& melody,
        const std::vector<int>& chordIntervals
    );

    /**
     * Generate counterpoint harmony (independent melodic lines)
     */
    std::vector<VoiceSynthesizer::VocalNote> generateCounterpointHarmony(
        const std::vector<VoiceSynthesizer::VocalNote>& melody,
        int voicePartIndex,
        const EmotionNode& emotion
    );

    /**
     * Convert MIDI pitch by semitones
     */
    static int transposePitch(int pitch, int semitones);

    /**
     * Get default voice part configuration
     */
    static VoicePartConfig getDefaultConfig(VoicePart part);

    /**
     * Initialize synthesizers for each voice part
     */
    void initializeSynthesizers();
};

} // namespace kelly
