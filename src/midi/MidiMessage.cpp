/**
 * @file MidiMessage.cpp
 * @brief MIDI message implementation
 */

#include "daiw/midi/MidiMessage.h"
#include <sstream>
#include <iomanip>

namespace daiw {
namespace midi {

std::string MidiMessage::toString() const {
    std::ostringstream oss;
    
    oss << "MIDI[" << std::hex << std::setfill('0');
    oss << std::setw(2) << static_cast<int>(data_[0]) << " ";
    oss << std::setw(2) << static_cast<int>(data_[1]) << " ";
    oss << std::setw(2) << static_cast<int>(data_[2]);
    oss << std::dec << "] @ " << timestamp_;
    
    // Add human-readable type
    oss << " (";
    switch (getType()) {
        case MessageType::NoteOn:
            if (data_[2] > 0) {
                oss << "Note On: " << static_cast<int>(getNoteNumber()) 
                    << " vel=" << static_cast<int>(getVelocity());
            } else {
                oss << "Note Off: " << static_cast<int>(getNoteNumber());
            }
            break;
        case MessageType::NoteOff:
            oss << "Note Off: " << static_cast<int>(getNoteNumber());
            break;
        case MessageType::ControlChange:
            oss << "CC: " << static_cast<int>(getControllerNumber()) 
                << " val=" << static_cast<int>(getControllerValue());
            break;
        case MessageType::PitchBend:
            oss << "Pitch Bend: " << getPitchBendValue();
            break;
        case MessageType::ProgramChange:
            oss << "Program Change: " << static_cast<int>(data_[1]);
            break;
        case MessageType::ChannelPressure:
            oss << "Channel Pressure: " << static_cast<int>(data_[1]);
            break;
        default:
            oss << "Type 0x" << std::hex << static_cast<int>(getType());
            break;
    }
    oss << " Ch:" << static_cast<int>(getChannel()) << ")";
    
    return oss.str();
}

} // namespace midi
} // namespace daiw
