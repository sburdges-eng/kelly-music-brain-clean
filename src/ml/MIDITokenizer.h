#pragma once

#include "common/Types.h"
#include <vector>
#include <string>
#include <map>

namespace kelly {

/**
 * MIDITokenizer - Converts MIDI notes to/from compound tokens for transformer models.
 *
 * Implements the Compound Word Transformer tokenization scheme where each token
 * represents a complete note event (pitch, velocity, duration, position).
 */
class MIDITokenizer {
public:
    static constexpr int VOCAB_SIZE = 8192;
    static constexpr int DURATION_BINS = 8;
    static constexpr int VELOCITY_BINS = 32;
    static constexpr int POSITIONS_PER_BAR = 16;

    // Special tokens
    static constexpr int TOKEN_START = 0;
    static constexpr int TOKEN_END = 1;
    static constexpr int TOKEN_OFFSET = 2;  // Offset for note tokens

    MIDITokenizer() {
        initializeDurationBins();
    }

    /**
     * Encode MIDI notes to token sequence.
     * @param notes MIDI notes to encode
     * @return Vector of token indices
     */
    std::vector<int> encode(const std::vector<MidiNote>& notes) const;

    /**
     * Decode token sequence to MIDI notes.
     * @param tokens Token indices to decode
     * @return Vector of MIDI notes
     */
    std::vector<MidiNote> decode(const std::vector<int>& tokens) const;

    /**
     * Quantize duration to nearest bin.
     * @param duration Duration in ticks
     * @return Duration bin index
     */
    int quantizeDuration(int duration) const;

    /**
     * Get duration from bin index.
     * @param binIndex Duration bin index
     * @return Duration in ticks
     */
    int getDurationFromBin(int binIndex) const;

private:
    std::vector<int> durationBins_;

    void initializeDurationBins() {
        // Duration bins in ticks (assuming 480 ticks per quarter note)
        durationBins_ = {30, 60, 120, 240, 480, 960, 1920, 3840};
    }

    /**
     * Encode a single note to token index.
     */
    int encodeNote(const MidiNote& note, int positionInBar) const;

    /**
     * Decode a single token to note.
     */
    MidiNote decodeToken(int token, int& currentTime, int& currentBar) const;
};

} // namespace kelly
