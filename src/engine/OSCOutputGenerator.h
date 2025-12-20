#pragma once

#include "engine/VADCalculator.h"
#include "common/Types.h"
#include <string>
#include <vector>
#include <optional>

namespace kelly {

/**
 * OSC Output Generator
 * 
 * Generates OSC (Open Sound Control) messages for VAD states and music parameters.
 * OSC is used for real-time communication with external systems.
 * 
 * Message format:
 * /kelly/vad/valence float
 * /kelly/vad/arousal float
 * /kelly/vad/dominance float
 * /kelly/music/tempo int
 * /kelly/music/key string
 * /kelly/music/mode string
 * /kelly/music/parameters (bundle of all parameters)
 */
struct OSCMessage {
    std::string address;           // OSC address path (e.g., "/kelly/vad/valence")
    std::vector<float> floatArgs;  // Float arguments
    std::vector<int> intArgs;      // Integer arguments
    std::vector<std::string> stringArgs;  // String arguments
    
    OSCMessage(const std::string& addr) : address(addr) {}
};

/**
 * OSC Output Generator
 */
class OSCOutputGenerator {
public:
    OSCOutputGenerator(const std::string& baseAddress = "/kelly");
    
    /**
     * Generate OSC messages from VAD state
     * @param vad Current VAD state
     * @return Vector of OSC messages
     */
    std::vector<OSCMessage> generateFromVAD(const VADState& vad) const;
    
    /**
     * Generate OSC messages from musical parameters
     * @param params Musical parameters
     * @return Vector of OSC messages
     */
    std::vector<OSCMessage> generateFromMusicalParameters(const MusicalParameters& params) const;
    
    /**
     * Generate OSC messages from emotion ID
     * @param emotionId Emotion ID
     * @param vad Calculated VAD state
     * @return Vector of OSC messages
     */
    std::vector<OSCMessage> generateFromEmotion(int emotionId, const VADState& vad) const;
    
    /**
     * Generate OSC bundle (multiple messages grouped together)
     * @param messages Messages to bundle
     * @return Bundle message (address = baseAddress + "/bundle")
     */
    OSCMessage createBundle(const std::vector<OSCMessage>& messages) const;
    
    /**
     * Convert OSC message to string (for debugging/logging)
     */
    static std::string messageToString(const OSCMessage& msg);
    
    /**
     * Set base OSC address
     */
    void setBaseAddress(const std::string& base) { baseAddress_ = base; }
    
private:
    std::string baseAddress_;
    
    std::string buildAddress(const std::string& path) const;
};

} // namespace kelly
