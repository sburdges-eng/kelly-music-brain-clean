/**
 * @file MidiSequence.cpp
 * @brief MIDI sequence implementation
 */

#include "daiw/midi/MidiSequence.h"
#include <map>

namespace daiw {
namespace midi {

std::vector<NoteEvent> MidiSequence::toNoteEvents() const {
    std::vector<NoteEvent> events;
    
    // Map to track active notes (key = channel*128 + note)
    std::map<uint16_t, MidiMessage> activeNotes;
    
    for (const auto& msg : messages_) {
        if (msg.isNoteOn()) {
            // Store note on message
            uint16_t key = (msg.getChannel() * 128) + msg.getNoteNumber();
            activeNotes[key] = msg;
        } else if (msg.isNoteOff()) {
            // Find matching note on
            uint16_t key = (msg.getChannel() * 128) + msg.getNoteNumber();
            auto it = activeNotes.find(key);
            if (it != activeNotes.end()) {
                // Create NoteEvent
                NoteEvent event;
                event.pitch = msg.getNoteNumber();
                event.velocity = it->second.getVelocity();
                event.startTick = it->second.getTimestamp();
                event.durationTicks = msg.getTimestamp() - it->second.getTimestamp();
                event.channel = msg.getChannel();
                events.push_back(event);
                
                // Remove from active notes
                activeNotes.erase(it);
            }
        }
    }
    
    // Handle any notes that never received a note off
    // (create events with default duration)
    for (const auto& [key, noteOn] : activeNotes) {
        NoteEvent event;
        event.pitch = noteOn.getNoteNumber();
        event.velocity = noteOn.getVelocity();
        event.startTick = noteOn.getTimestamp();
        event.durationTicks = ppq_;  // Default to 1 quarter note
        event.channel = noteOn.getChannel();
        events.push_back(event);
    }
    
    return events;
}

MidiSequence MidiSequence::fromNoteEvents(const std::vector<NoteEvent>& events, int ppq) {
    MidiSequence sequence(ppq);
    
    for (const auto& event : events) {
        // Create note on
        auto noteOn = MidiMessage::noteOn(event.channel, event.pitch, event.velocity);
        noteOn.setTimestamp(event.startTick);
        sequence.addMessage(noteOn);
        
        // Create note off
        auto noteOff = MidiMessage::noteOff(event.channel, event.pitch);
        noteOff.setTimestamp(event.endTick());
        sequence.addMessage(noteOff);
    }
    
    sequence.sort();
    return sequence;
}

} // namespace midi
} // namespace daiw
