#pragma once

#include "common/Types.h"
#include "voice/VocoderEngine.h"
#include "voice/LyricTypes.h"
// PitchPhonemeAligner.h removed to break circular dependency - include in .cpp if needed
#include "voice/ExpressionEngine.h"
#include <vector>
#include <string>
#include <optional>
#include <memory>

namespace kelly {

// Forward declarations
struct GeneratedMidi;
struct EmotionNode;
struct Wound;
class PitchPhonemeAligner;  // Forward declaration (used as unique_ptr)

/**
 * Voice Synthesizer - Generates vocal melodies and lyrics based on emotion.
 *
 * v2.0 feature: Synthesizes voice parts to accompany generated MIDI.
 * Now includes full vocoder integration for realistic voice synthesis.
 */
class VoiceSynthesizer {
public:
    VoiceSynthesizer();
    ~VoiceSynthesizer() = default;

    /**
     * Generate vocal melody based on emotion and MIDI context.
     * @param emotion The emotion to express
     * @param midiContext The generated MIDI to accompany
     * @param lyrics Optional lyric structure to align with melody
     * @return Vector of vocal notes (pitch, timing, lyrics)
     */
    struct VocalNote {
        int pitch;          // MIDI pitch
        double startBeat;   // Start position in beats
        double duration;    // Duration in beats
        std::string lyric;  // Optional lyric syllable
        float vibrato;      // Vibrato amount (0.0 to 1.0)
        VocalExpression expression;  // Expression parameters
    };

    std::vector<VocalNote> generateVocalMelody(
        const EmotionNode& emotion,
        const GeneratedMidi& midiContext,
        const LyricStructure* lyrics = nullptr
    );

    /**
     * Generate lyrics based on emotion and wound description.
     * @param emotion The emotion to express
     * @param wound The original wound description
     * @return Vector of lyric lines
     */
    std::vector<std::string> generateLyrics(
        const EmotionNode& emotion,
        const Wound& wound
    );

    /**
     * Synthesize audio from vocal notes.
     * @param notes The vocal notes to synthesize
     * @param sampleRate Target sample rate
     * @param emotion Optional emotion for vocal characteristics (if provided, uses emotion-based parameters)
     * @return Audio buffer with synthesized voice
     */
    std::vector<float> synthesizeAudio(
        const std::vector<VocalNote>& notes,
        double sampleRate = 44100.0,
        const EmotionNode* emotion = nullptr
    );

    /** Enable/disable voice synthesis */
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    /**
     * Prepare synthesizer for audio processing
     * @param sampleRate Audio sample rate
     */
    void prepare(double sampleRate);

    /**
     * Set BPM for timing calculations
     */
    void setBPM(float bpm);  // Implementation in .cpp to avoid requiring full PitchPhonemeAligner definition

    /**
     * Set voice type (Male, Female, Child, Neutral)
     */
    void setVoiceType(VoiceType voiceType);

    /**
     * Get current voice type
     */
    VoiceType getVoiceType() const { return voiceType_; }

    /**
     * Synthesize audio with real-time support (streaming)
     * Processes one block of audio samples
     * @param notes Vocal notes to synthesize
     * @param outputBuffer Output audio buffer
     * @param numSamples Number of samples to generate
     * @param currentSample Current playback position in samples
     * @param emotion Optional emotion for vocal characteristics
     */
    void synthesizeBlock(
        const std::vector<VocalNote>& notes,
        float* outputBuffer,
        int numSamples,
        int64_t currentSample,
        const EmotionNode* emotion = nullptr
    );

    /**
     * MIDI lyric event for export (MIDI text event 0xFF 05).
     * These can be included in MIDI file export.
     */
    struct MidiLyricEvent {
        int tick;           // MIDI tick position
        std::string text;   // Lyric text (syllable or word)
        double beat;        // Beat position (alternative to tick)

        MidiLyricEvent() : tick(0), beat(0.0) {}
    };

    /**
     * Generate MIDI lyric events from vocal notes.
     * These can be included when exporting MIDI files.
     * @param notes Vocal notes with lyrics
     * @param ticksPerBeat MIDI ticks per quarter note (default: 480)
     * @return Vector of MIDI lyric events
     */
    std::vector<MidiLyricEvent> generateMidiLyricEvents(
        const std::vector<VocalNote>& notes,
        int ticksPerBeat = 480
    ) const;

private:
    bool enabled_ = false;
    double sampleRate_ = 44100.0;
    float bpm_ = 120.0f;

    // Vocoder engine
    std::unique_ptr<VocoderEngine> vocoder_;

    // Envelope generator
    std::unique_ptr<ADSREnvelope> envelope_;

    // Portamento generator
    std::unique_ptr<PortamentoGenerator> portamento_;

    // Pitch-phoneme aligner
    std::unique_ptr<PitchPhonemeAligner> aligner_;

    // Expression engine
    std::unique_ptr<ExpressionEngine> expressionEngine_;

    // Voice type
    VoiceType voiceType_ = VoiceType::Neutral;
    VoiceTypeParams voiceParams_;

    // Current note being synthesized
    struct ActiveNote {
        const VocalNote* note = nullptr;
        int64_t startSample = 0;
        int64_t endSample = 0;
        bool isActive = false;
    };
    ActiveNote currentNote_;

    // Current vowel for formant selection
    VowelFormantDatabase::Vowel currentVowel_ = VowelFormantDatabase::Vowel::AH;

    /** Generate melody contour based on emotion */
    std::vector<int> generateMelodyContour(const EmotionNode& emotion, int numNotes);

    /** Map emotion to vocal characteristics */
    struct VocalCharacteristics {
        float brightness;    // 0.0 (dark) to 1.0 (bright)
        float breathiness;   // 0.0 (clear) to 1.0 (breathy)
        float vibratoRate;   // Vibrato speed in Hz
        float vibratoDepth;  // Vibrato depth (0.0 to 1.0)
    };

    VocalCharacteristics getVocalCharacteristics(const EmotionNode& emotion) const;

    /** Convert MIDI pitch to frequency in Hz */
    float midiToFrequency(int midiPitch) const;

    /** Convert beats to samples */
    int64_t beatsToSamples(double beats) const;

    /** Get vowel for a given pitch and optional emotion */
    VowelFormantDatabase::Vowel selectVowel(int midiPitch, const EmotionNode* emotion = nullptr) const;
};

} // namespace kelly
