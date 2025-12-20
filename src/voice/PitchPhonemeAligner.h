#pragma once

#include "voice/LyricTypes.h"
#include "voice/PhonemeConverter.h"
#include "common/Types.h"
// Include VoiceSynthesizer.h for VocalNote definition BEFORE namespace
// Note: VoiceSynthesizer.h does NOT include PitchPhonemeAligner.h, so no circular dependency
#include "voice/VoiceSynthesizer.h"
#include <vector>
#include <string>

namespace kelly {

// Forward declarations
struct GeneratedMidi;

/**
 * PitchPhonemeAligner - Aligns MIDI pitches to phoneme sequences.
 *
 * This class handles:
 * - Aligning MIDI pitches to phoneme sequences from lyrics
 * - Handling melisma (multiple notes per syllable)
 * - Supporting portamento between phonemes
 * - Calculating timing for each phoneme
 */
class PitchPhonemeAligner {
public:
    /**
     * Aligned phoneme with pitch and timing information.
     */
    struct AlignedPhoneme {
        Phoneme phoneme;
        int midiPitch;
        double startBeat;
        double duration;
        bool isStartOfSyllable;
        bool isEndOfSyllable;

        AlignedPhoneme()
            : midiPitch(60)
            , startBeat(0.0)
            , duration(0.0)
            , isStartOfSyllable(false)
            , isEndOfSyllable(false)
        {}
    };

    /**
     * Alignment result containing aligned phonemes and vocal notes.
     */
    struct AlignmentResult {
        std::vector<AlignedPhoneme> alignedPhonemes;
        std::vector<VoiceSynthesizer::VocalNote> vocalNotes;
    };

    PitchPhonemeAligner();
    ~PitchPhonemeAligner() = default;

    /**
     * Align lyrics to vocal melody notes.
     * @param lyrics Generated lyrics
     * @param vocalNotes Original vocal notes (may be modified)
     * @param midiContext MIDI context for timing
     * @return Alignment result with aligned phonemes and updated vocal notes
     */
    AlignmentResult alignLyricsToMelody(
        const LyricStructure& lyrics,
        const std::vector<VoiceSynthesizer::VocalNote>& vocalNotes,
        const GeneratedMidi* midiContext = nullptr
    );

    /**
     * Align phonemes to a single vocal note (handles melisma).
     * @param phonemes Phonemes for this note/syllable
     * @param note Vocal note to align to
     * @param beatPosition Start beat position
     * @return Vector of aligned phonemes
     */
    std::vector<AlignedPhoneme> alignPhonemesToNote(
        const std::vector<Phoneme>& phonemes,
        const VoiceSynthesizer::VocalNote& note,
        double beatPosition
    );

    /**
     * Calculate phoneme durations based on note duration and phoneme count.
     * @param noteDuration Total note duration in beats
     * @param phonemes Phonemes to distribute duration across
     * @return Vector of durations (in beats) for each phoneme
     */
    std::vector<double> calculatePhonemeDurations(
        double noteDuration,
        const std::vector<Phoneme>& phonemes
    );

    /**
     * Set BPM for timing calculations.
     */
    void setBPM(float bpm) { bpm_ = bpm; }

    /**
     * Set melisma handling mode.
     * @param allowMelisma If true, allow multiple notes per syllable
     */
    void setAllowMelisma(bool allowMelisma) { allowMelisma_ = allowMelisma; }

    /**
     * Set portamento time between phonemes.
     * @param portamentoTime Time in beats for portamento transitions
     */
    void setPortamentoTime(double portamentoTime) { portamentoTime_ = portamentoTime; }

private:
    float bpm_ = 120.0f;
    bool allowMelisma_ = true;
    double portamentoTime_ = 0.05;  // 50ms default

    PhonemeConverter phonemeConverter_;

    /**
     * Convert lyric line to phonemes.
     */
    std::vector<Phoneme> lineToPhonemes(const LyricLine& line);

    /**
     * Calculate phoneme duration based on its type and context.
     */
    double calculatePhonemeDuration(const Phoneme& phoneme, double baseDuration) const;
};

} // namespace kelly
