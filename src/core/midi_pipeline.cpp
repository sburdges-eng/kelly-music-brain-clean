#include "midi_pipeline.h"

namespace kelly {

MidiPipeline::MidiPipeline() = default;

void MidiPipeline::setTempo(int bpm) {
    tempo_ = bpm;
}

void MidiPipeline::addNote(const MidiNote& note) {
    notes_.push_back(note);
}

void MidiPipeline::clear() {
    notes_.clear();
}

} // namespace kelly
