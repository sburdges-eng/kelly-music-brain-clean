#pragma once

#include "common/Types.h"
#include "common/MusicConstants.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <mutex>
#include <memory>
#include <map>
#include <string>

namespace kelly {

/**
 * Builds JUCE MIDI objects from Kelly's internal representations.
 * 
 * Thread-safe MIDI message builder that handles:
 * - Timing conversion (beats to ticks/samples)
 * - Velocity mapping
 * - Channel assignment per layer
 * - MIDI file and buffer generation
 */
class MidiBuilder {
public:
    MidiBuilder();
    ~MidiBuilder() = default;
    
    /**
     * Build a MidiFile for export/drag-drop.
     * Thread-safe.
     */
    juce::MidiFile buildMidiFile(const GeneratedMidi& midi);
    
    /**
     * Build a MidiBuffer for real-time output.
     * Thread-safe.
     * 
     * @param midi The generated MIDI data
     * @param sampleRate Audio sample rate
     * @param bpm Tempo in beats per minute
     * @return MidiBuffer ready for playback
     */
    juce::MidiBuffer buildMidiBuffer(const GeneratedMidi& midi, 
                                      double sampleRate, 
                                      float bpm);
    
    /**
     * Convert a single chord to MIDI messages.
     * 
     * @param seq Target MIDI sequence
     * @param chord Chord to convert
     * @param ticksPerBeat Ticks per quarter note
     * @param channel MIDI channel (0-15)
     * @param velocity Note velocity (0-127)
     */
    void addChordToSequence(juce::MidiMessageSequence& seq,
                            const Chord& chord,
                            int ticksPerBeat,
                            int channel,
                            int velocity);
    
    /**
     * Convert notes to MIDI messages.
     * 
     * @param seq Target MIDI sequence
     * @param notes Notes to convert
     * @param ticksPerBeat Ticks per quarter note
     * @param channel MIDI channel (0-15)
     */
    void addNotesToSequence(juce::MidiMessageSequence& seq,
                            const std::vector<MidiNote>& notes,
                            int ticksPerBeat,
                            int channel);
    
    /**
     * Set custom channel assignment for a layer.
     * 
     * @param layerName Layer name ("chords", "melody", "bass", etc.)
     * @param channel MIDI channel (0-15)
     */
    void setChannelForLayer(const std::string& layerName, int channel);
    
    /**
     * Get channel assignment for a layer.
     * 
     * @param layerName Layer name
     * @return MIDI channel (0-15)
     */
    int getChannelForLayer(const std::string& layerName) const;
    
    /**
     * Reset all channel assignments to defaults.
     */
    void resetChannelAssignments();
    
private:
    // Thread safety
    mutable std::mutex mutex_;
    
    // Channel assignments (defaults from MusicConstants)
    std::map<std::string, int> channelAssignments_;
    
    // Initialize default channel assignments
    void initializeChannelAssignments();
    
    // Helper: Convert beats to ticks
    int beatsToTicks(double beats, int ticksPerBeat) const;
    
    // Helper: Convert beats to samples
    int beatsToSamples(double beats, double sampleRate, float bpm) const;
    
    // Helper: Clamp velocity to valid range
    juce::uint8 clampVelocity(int velocity) const;
    
    // Helper: Clamp channel to valid range
    int clampChannel(int channel) const;
};

} // namespace kelly
