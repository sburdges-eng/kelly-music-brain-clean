#include "ml/MIDITokenizer.h"
#include <algorithm>
#include <cmath>

namespace kelly {

std::vector<int> MIDITokenizer::encode(const std::vector<MidiNote>& notes) const {
    std::vector<int> tokens;
    tokens.push_back(TOKEN_START);

    int ticksPerBeat = 480;  // Standard MIDI resolution
    int ticksPerBar = ticksPerBeat * 4;  // 4/4 time

    for (const auto& note : notes) {
        // Calculate position within bar
        int noteStartTick = static_cast<int>(note.startBeat * ticksPerBeat);
        int positionInBar = (noteStartTick % ticksPerBar) / (ticksPerBar / POSITIONS_PER_BAR);

        int token = encodeNote(note, positionInBar);
        tokens.push_back(token + TOKEN_OFFSET);
    }

    tokens.push_back(TOKEN_END);
    return tokens;
}

std::vector<MidiNote> MIDITokenizer::decode(const std::vector<int>& tokens) const {
    std::vector<MidiNote> notes;

    int ticksPerBeat = 480;
    int ticksPerBar = ticksPerBeat * 4;
    int currentTime = 0;
    int currentBar = 0;

    for (int token : tokens) {
        if (token == TOKEN_START) {
            continue;
        }
        if (token == TOKEN_END) {
            break;
        }

        token -= TOKEN_OFFSET;
        if (token < 0) {
            continue;  // Invalid token
        }

        MidiNote note = decodeToken(token, currentTime, currentBar);
        notes.push_back(note);
    }

    return notes;
}

int MIDITokenizer::quantizeDuration(int duration) const {
    for (size_t i = 0; i < durationBins_.size(); ++i) {
        if (duration <= durationBins_[i]) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(durationBins_.size() - 1);
}

int MIDITokenizer::getDurationFromBin(int binIndex) const {
    if (binIndex < 0 || binIndex >= static_cast<int>(durationBins_.size())) {
        return durationBins_.back();
    }
    return durationBins_[binIndex];
}

int MIDITokenizer::encodeNote(const MidiNote& note, int positionInBar) const {
    // Encode: pitch + velocity_bin * 128 + duration_bin * 128 * 32 + position * 128 * 32 * 8
    int pitch = std::clamp(note.pitch, 0, 127);
    int velocityBin = std::clamp(note.velocity / 4, 0, VELOCITY_BINS - 1);
    int durationBin = quantizeDuration(static_cast<int>(note.duration * 480));  // Convert beats to ticks
    int position = std::clamp(positionInBar, 0, POSITIONS_PER_BAR - 1);

    return pitch +
           velocityBin * 128 +
           durationBin * 128 * VELOCITY_BINS +
           position * 128 * VELOCITY_BINS * DURATION_BINS;
}

MidiNote MIDITokenizer::decodeToken(int token, int& currentTime, int& currentBar) const {
    MidiNote note;

    int ticksPerBeat = 480;
    int ticksPerBar = ticksPerBeat * 4;

    // Decode token components
    int pitch = token % 128;
    int velocityBin = (token / 128) % VELOCITY_BINS;
    int durationBin = (token / (128 * VELOCITY_BINS)) % DURATION_BINS;
    int position = token / (128 * VELOCITY_BINS * DURATION_BINS);

    note.pitch = pitch;
    note.velocity = velocityBin * 4;  // Dequantize velocity
    note.duration = getDurationFromBin(durationBin) / static_cast<double>(ticksPerBeat);  // Convert ticks to beats

    // Calculate start time
    int positionTick = position * (ticksPerBar / POSITIONS_PER_BAR);
    int barStart = currentBar * ticksPerBar;
    note.startBeat = (barStart + positionTick) / static_cast<double>(ticksPerBeat);

    // Update current time and bar
    if (position == 0 && positionTick == 0) {
        currentBar++;
    }
    currentTime = static_cast<int>(note.startBeat * ticksPerBeat) + getDurationFromBin(durationBin);

    return note;
}

} // namespace kelly
