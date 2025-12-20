#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace kelly {

struct MidiNote {
    uint8_t note;
    uint8_t velocity;
    uint32_t time;  // in ticks
    uint32_t duration;  // in ticks
};

class MidiPipeline {
public:
    MidiPipeline();
    ~MidiPipeline() = default;

    void setTempo(int bpm);
    void addNote(const MidiNote& note);
    void clear();
    
    const std::vector<MidiNote>& getNotes() const { return notes_; }
    int getTempo() const { return tempo_; }

private:
    std::vector<MidiNote> notes_;
    int tempo_ = 120;
};

} // namespace kelly
