#include "midi/MidiBuilder.h"
#include <algorithm>
#include <cmath>

namespace kelly {
using namespace MusicConstants;

MidiBuilder::MidiBuilder() {
    initializeChannelAssignments();
}

void MidiBuilder::initializeChannelAssignments() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Set default channel assignments from MusicConstants
    channelAssignments_["chords"] = MIDI_CHANNEL_CHORDS;
    channelAssignments_["melody"] = MIDI_CHANNEL_MELODY;
    channelAssignments_["bass"] = MIDI_CHANNEL_BASS;
    channelAssignments_["counterMelody"] = MIDI_CHANNEL_COUNTER_MELODY;
    channelAssignments_["pad"] = MIDI_CHANNEL_PAD;
    channelAssignments_["strings"] = MIDI_CHANNEL_STRINGS;
    channelAssignments_["fills"] = MIDI_CHANNEL_FILLS;
    channelAssignments_["rhythm"] = MIDI_CHANNEL_RHYTHM;
}

juce::MidiFile MidiBuilder::buildMidiFile(const GeneratedMidi& midi) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    juce::MidiFile file;
    file.setTicksPerQuarterNote(MIDI_PPQ);
    
    // Track 0: Tempo and time signature (meta track)
    juce::MidiMessageSequence metaTrack;
    
    // Time signature: 4/4
    metaTrack.addEvent(juce::MidiMessage::timeSignatureMetaEvent(4, 4));
    
    // Tempo: microseconds per quarter note
    // Formula: 60,000,000 microseconds / BPM = microseconds per beat
    int microsecondsPerBeat = MIDI_MICROSECONDS_PER_MINUTE / static_cast<int>(midi.bpm);
    metaTrack.addEvent(juce::MidiMessage::tempoMetaEvent(microsecondsPerBeat));
    
    file.addTrack(metaTrack);
    
    // Track 1: Chords
    if (!midi.chords.empty()) {
        juce::MidiMessageSequence chordTrack;
        int channel = getChannelForLayer("chords");
        for (const auto& chord : midi.chords) {
            addChordToSequence(chordTrack, chord, MIDI_PPQ, channel, MIDI_VELOCITY_MEDIUM);
        }
        chordTrack.updateMatchedPairs();
        file.addTrack(chordTrack);
    }
    
    // Track 2: Melody
    if (!midi.melody.empty()) {
        juce::MidiMessageSequence melodyTrack;
        int channel = getChannelForLayer("melody");
        addNotesToSequence(melodyTrack, midi.melody, MIDI_PPQ, channel);
        melodyTrack.updateMatchedPairs();
        file.addTrack(melodyTrack);
    }
    
    // Track 3: Bass
    if (!midi.bass.empty()) {
        juce::MidiMessageSequence bassTrack;
        int channel = getChannelForLayer("bass");
        addNotesToSequence(bassTrack, midi.bass, MIDI_PPQ, channel);
        bassTrack.updateMatchedPairs();
        file.addTrack(bassTrack);
    }
    
    // Track 4: Counter-melody
    if (!midi.counterMelody.empty()) {
        juce::MidiMessageSequence counterMelodyTrack;
        int channel = getChannelForLayer("counterMelody");
        addNotesToSequence(counterMelodyTrack, midi.counterMelody, MIDI_PPQ, channel);
        counterMelodyTrack.updateMatchedPairs();
        file.addTrack(counterMelodyTrack);
    }
    
    // Track 5: Pad
    if (!midi.pad.empty()) {
        juce::MidiMessageSequence padTrack;
        int channel = getChannelForLayer("pad");
        addNotesToSequence(padTrack, midi.pad, MIDI_PPQ, channel);
        padTrack.updateMatchedPairs();
        file.addTrack(padTrack);
    }
    
    // Track 6: Strings
    if (!midi.strings.empty()) {
        juce::MidiMessageSequence stringsTrack;
        int channel = getChannelForLayer("strings");
        addNotesToSequence(stringsTrack, midi.strings, MIDI_PPQ, channel);
        stringsTrack.updateMatchedPairs();
        file.addTrack(stringsTrack);
    }
    
    // Track 7: Fills
    if (!midi.fills.empty()) {
        juce::MidiMessageSequence fillsTrack;
        int channel = getChannelForLayer("fills");
        addNotesToSequence(fillsTrack, midi.fills, MIDI_PPQ, channel);
        fillsTrack.updateMatchedPairs();
        file.addTrack(fillsTrack);
    }
    
    return file;
}

juce::MidiBuffer MidiBuilder::buildMidiBuffer(const GeneratedMidi& midi,
                                               double sampleRate,
                                               float bpm) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    juce::MidiBuffer buffer;
    
    // Calculate samples per beat
    double samplesPerBeat = (sampleRate * 60.0) / bpm;
    
    // Add chords
    if (!midi.chords.empty()) {
        int channel = getChannelForLayer("chords");
        for (const auto& chord : midi.chords) {
            int startSample = beatsToSamples(chord.startBeat, sampleRate, bpm);
            int endSample = beatsToSamples(chord.startBeat + chord.duration, sampleRate, bpm);
            
            for (int pitch : chord.pitches) {
                // Clamp pitch to valid range
                pitch = std::clamp(pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
                
                buffer.addEvent(
                    juce::MidiMessage::noteOn(channel, pitch, clampVelocity(MIDI_VELOCITY_MEDIUM)),
                    startSample
                );
                buffer.addEvent(
                    juce::MidiMessage::noteOff(channel, pitch),
                    endSample
                );
            }
        }
    }
    
    // Add melody
    if (!midi.melody.empty()) {
        int channel = getChannelForLayer("melody");
        for (const auto& note : midi.melody) {
            int startSample = beatsToSamples(note.startBeat, sampleRate, bpm);
            int endSample = beatsToSamples(note.startBeat + note.duration, sampleRate, bpm);
            
            int pitch = std::clamp(note.pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
            
            buffer.addEvent(
                juce::MidiMessage::noteOn(channel, pitch, clampVelocity(note.velocity)),
                startSample
            );
            buffer.addEvent(
                juce::MidiMessage::noteOff(channel, pitch),
                endSample
            );
        }
    }
    
    // Add bass
    if (!midi.bass.empty()) {
        int channel = getChannelForLayer("bass");
        for (const auto& note : midi.bass) {
            int startSample = beatsToSamples(note.startBeat, sampleRate, bpm);
            int endSample = beatsToSamples(note.startBeat + note.duration, sampleRate, bpm);
            
            int pitch = std::clamp(note.pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
            
            buffer.addEvent(
                juce::MidiMessage::noteOn(channel, pitch, clampVelocity(note.velocity)),
                startSample
            );
            buffer.addEvent(
                juce::MidiMessage::noteOff(channel, pitch),
                endSample
            );
        }
    }
    
    // Add counter-melody
    if (!midi.counterMelody.empty()) {
        int channel = getChannelForLayer("counterMelody");
        for (const auto& note : midi.counterMelody) {
            int startSample = beatsToSamples(note.startBeat, sampleRate, bpm);
            int endSample = beatsToSamples(note.startBeat + note.duration, sampleRate, bpm);
            
            int pitch = std::clamp(note.pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
            
            buffer.addEvent(
                juce::MidiMessage::noteOn(channel, pitch, clampVelocity(note.velocity)),
                startSample
            );
            buffer.addEvent(
                juce::MidiMessage::noteOff(channel, pitch),
                endSample
            );
        }
    }
    
    // Add pad
    if (!midi.pad.empty()) {
        int channel = getChannelForLayer("pad");
        for (const auto& note : midi.pad) {
            int startSample = beatsToSamples(note.startBeat, sampleRate, bpm);
            int endSample = beatsToSamples(note.startBeat + note.duration, sampleRate, bpm);
            
            int pitch = std::clamp(note.pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
            
            buffer.addEvent(
                juce::MidiMessage::noteOn(channel, pitch, clampVelocity(note.velocity)),
                startSample
            );
            buffer.addEvent(
                juce::MidiMessage::noteOff(channel, pitch),
                endSample
            );
        }
    }
    
    // Add strings
    if (!midi.strings.empty()) {
        int channel = getChannelForLayer("strings");
        for (const auto& note : midi.strings) {
            int startSample = beatsToSamples(note.startBeat, sampleRate, bpm);
            int endSample = beatsToSamples(note.startBeat + note.duration, sampleRate, bpm);
            
            int pitch = std::clamp(note.pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
            
            buffer.addEvent(
                juce::MidiMessage::noteOn(channel, pitch, clampVelocity(note.velocity)),
                startSample
            );
            buffer.addEvent(
                juce::MidiMessage::noteOff(channel, pitch),
                endSample
            );
        }
    }
    
    // Add fills
    if (!midi.fills.empty()) {
        int channel = getChannelForLayer("fills");
        for (const auto& note : midi.fills) {
            int startSample = beatsToSamples(note.startBeat, sampleRate, bpm);
            int endSample = beatsToSamples(note.startBeat + note.duration, sampleRate, bpm);
            
            int pitch = std::clamp(note.pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
            
            buffer.addEvent(
                juce::MidiMessage::noteOn(channel, pitch, clampVelocity(note.velocity)),
                startSample
            );
            buffer.addEvent(
                juce::MidiMessage::noteOff(channel, pitch),
                endSample
            );
        }
    }
    
    return buffer;
}

void MidiBuilder::addChordToSequence(juce::MidiMessageSequence& seq,
                                      const Chord& chord,
                                      int ticksPerBeat,
                                      int channel,
                                      int velocity) {
    int startTick = beatsToTicks(chord.startBeat, ticksPerBeat);
    int endTick = beatsToTicks(chord.startBeat + chord.duration, ticksPerBeat);
    
    channel = clampChannel(channel);
    juce::uint8 vel = clampVelocity(velocity);
    
    for (int pitch : chord.pitches) {
        // Clamp pitch to valid range
        pitch = std::clamp(pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
        
        seq.addEvent(
            juce::MidiMessage::noteOn(channel, pitch, vel),
            startTick
        );
        seq.addEvent(
            juce::MidiMessage::noteOff(channel, pitch),
            endTick
        );
    }
}

void MidiBuilder::addNotesToSequence(juce::MidiMessageSequence& seq,
                                      const std::vector<MidiNote>& notes,
                                      int ticksPerBeat,
                                      int channel) {
    channel = clampChannel(channel);
    
    for (const auto& note : notes) {
        int startTick = beatsToTicks(note.startBeat, ticksPerBeat);
        int endTick = beatsToTicks(note.startBeat + note.duration, ticksPerBeat);
        
        int pitch = std::clamp(note.pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
        juce::uint8 vel = clampVelocity(note.velocity);
        
        seq.addEvent(
            juce::MidiMessage::noteOn(channel, pitch, vel),
            startTick
        );
        seq.addEvent(
            juce::MidiMessage::noteOff(channel, pitch),
            endTick
        );
    }
}

void MidiBuilder::setChannelForLayer(const std::string& layerName, int channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    channelAssignments_[layerName] = clampChannel(channel);
}

int MidiBuilder::getChannelForLayer(const std::string& layerName) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = channelAssignments_.find(layerName);
    if (it != channelAssignments_.end()) {
        return it->second;
    }
    // Default to channel 0 if layer not found
    return MIDI_CHANNEL_CHORDS;
}

void MidiBuilder::resetChannelAssignments() {
    std::lock_guard<std::mutex> lock(mutex_);
    initializeChannelAssignments();
}

// ============================================================================
// Helper Methods
// ============================================================================

int MidiBuilder::beatsToTicks(double beats, int ticksPerBeat) const {
    return static_cast<int>(std::round(beats * ticksPerBeat));
}

int MidiBuilder::beatsToSamples(double beats, double sampleRate, float bpm) const {
    double samplesPerBeat = (sampleRate * 60.0) / bpm;
    return static_cast<int>(std::round(beats * samplesPerBeat));
}

juce::uint8 MidiBuilder::clampVelocity(int velocity) const {
    return static_cast<juce::uint8>(std::clamp(velocity, MIDI_VELOCITY_MIN, MIDI_VELOCITY_MAX));
}

int MidiBuilder::clampChannel(int channel) const {
    return std::clamp(channel, MIDI_CHANNEL_MIN, MIDI_CHANNEL_MAX);
}

} // namespace kelly
