#include "engine/OSCOutputGenerator.h"
#include <sstream>
#include <algorithm>

namespace kelly {

OSCOutputGenerator::OSCOutputGenerator(const std::string& baseAddress)
    : baseAddress_(baseAddress) {
}

std::vector<OSCMessage> OSCOutputGenerator::generateFromVAD(const VADState& vad) const {
    std::vector<OSCMessage> messages;
    
    // VAD values
    OSCMessage valenceMsg(buildAddress("/vad/valence"));
    valenceMsg.floatArgs.push_back(vad.valence);
    messages.push_back(valenceMsg);
    
    OSCMessage arousalMsg(buildAddress("/vad/arousal"));
    arousalMsg.floatArgs.push_back(vad.arousal);
    messages.push_back(arousalMsg);
    
    OSCMessage dominanceMsg(buildAddress("/vad/dominance"));
    dominanceMsg.floatArgs.push_back(vad.dominance);
    messages.push_back(dominanceMsg);
    
    // Timestamp
    OSCMessage timestampMsg(buildAddress("/vad/timestamp"));
    timestampMsg.floatArgs.push_back(static_cast<float>(vad.timestamp));
    messages.push_back(timestampMsg);
    
    return messages;
}

std::vector<OSCMessage> OSCOutputGenerator::generateFromMusicalParameters(
    const MusicalParameters& params
) const {
    std::vector<OSCMessage> messages;
    
    // Tempo
    OSCMessage tempoMsg(buildAddress("/music/tempo"));
    tempoMsg.intArgs.push_back(params.tempoSuggested);
    messages.push_back(tempoMsg);
    
    // Tempo range
    OSCMessage tempoMinMsg(buildAddress("/music/tempo_min"));
    tempoMinMsg.intArgs.push_back(params.tempoMin);
    messages.push_back(tempoMinMsg);
    
    OSCMessage tempoMaxMsg(buildAddress("/music/tempo_max"));
    tempoMaxMsg.intArgs.push_back(params.tempoMax);
    messages.push_back(tempoMaxMsg);
    
    // Key and mode
    OSCMessage keyMsg(buildAddress("/music/key"));
    keyMsg.stringArgs.push_back(params.keySuggested);
    messages.push_back(keyMsg);
    
    OSCMessage modeMsg(buildAddress("/music/mode"));
    modeMsg.stringArgs.push_back(params.modeSuggested);
    messages.push_back(modeMsg);
    
    // Expression parameters
    OSCMessage dissonanceMsg(buildAddress("/music/dissonance"));
    dissonanceMsg.floatArgs.push_back(params.dissonance);
    messages.push_back(dissonanceMsg);
    
    OSCMessage densityMsg(buildAddress("/music/density"));
    densityMsg.floatArgs.push_back(params.density);
    messages.push_back(densityMsg);
    
    OSCMessage spaceProbMsg(buildAddress("/music/space_probability"));
    spaceProbMsg.floatArgs.push_back(params.spaceProbability);
    messages.push_back(spaceProbMsg);
    
    // Dynamics
    OSCMessage dynamicsRangeMsg(buildAddress("/music/dynamics_range"));
    dynamicsRangeMsg.floatArgs.push_back(params.dynamicsRange);
    messages.push_back(dynamicsRangeMsg);
    
    OSCMessage velocityMinMsg(buildAddress("/music/velocity_min"));
    velocityMinMsg.intArgs.push_back(params.velocityMin);
    messages.push_back(velocityMinMsg);
    
    OSCMessage velocityMaxMsg(buildAddress("/music/velocity_max"));
    velocityMaxMsg.intArgs.push_back(params.velocityMax);
    messages.push_back(velocityMaxMsg);
    
    // Effects
    OSCMessage reverbMsg(buildAddress("/music/reverb_amount"));
    reverbMsg.floatArgs.push_back(params.reverbAmount);
    messages.push_back(reverbMsg);
    
    OSCMessage reverbDecayMsg(buildAddress("/music/reverb_decay"));
    reverbDecayMsg.floatArgs.push_back(params.reverbDecay);
    messages.push_back(reverbDecayMsg);
    
    OSCMessage brightnessMsg(buildAddress("/music/brightness"));
    brightnessMsg.floatArgs.push_back(params.brightness);
    messages.push_back(brightnessMsg);
    
    OSCMessage saturationMsg(buildAddress("/music/saturation"));
    saturationMsg.floatArgs.push_back(params.saturation);
    messages.push_back(saturationMsg);
    
    // Humanization
    OSCMessage timingVarMsg(buildAddress("/music/timing_variation"));
    timingVarMsg.floatArgs.push_back(params.timingVariation);
    messages.push_back(timingVarMsg);
    
    OSCMessage velocityVarMsg(buildAddress("/music/velocity_variation"));
    velocityVarMsg.floatArgs.push_back(params.velocityVariation);
    messages.push_back(velocityVarMsg);
    
    return messages;
}

std::vector<OSCMessage> OSCOutputGenerator::generateFromEmotion(
    int emotionId,
    const VADState& vad
) const {
    std::vector<OSCMessage> messages;
    
    // Emotion ID
    OSCMessage emotionIdMsg(buildAddress("/emotion/id"));
    emotionIdMsg.intArgs.push_back(emotionId);
    messages.push_back(emotionIdMsg);
    
    // VAD from emotion
    auto vadMessages = generateFromVAD(vad);
    messages.insert(messages.end(), vadMessages.begin(), vadMessages.end());
    
    return messages;
}

OSCMessage OSCOutputGenerator::createBundle(
    const std::vector<OSCMessage>& messages
) const {
    OSCMessage bundle(buildAddress("/bundle"));
    // In actual OSC implementation, this would create a proper bundle
    // For now, we'll just return a message with metadata
    bundle.intArgs.push_back(static_cast<int>(messages.size()));
    return bundle;
}

std::string OSCOutputGenerator::messageToString(const OSCMessage& msg) {
    std::ostringstream oss;
    oss << msg.address;
    
    if (!msg.floatArgs.empty()) {
        oss << " [float:";
        for (size_t i = 0; i < msg.floatArgs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << msg.floatArgs[i];
        }
        oss << "]";
    }
    
    if (!msg.intArgs.empty()) {
        oss << " [int:";
        for (size_t i = 0; i < msg.intArgs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << msg.intArgs[i];
        }
        oss << "]";
    }
    
    if (!msg.stringArgs.empty()) {
        oss << " [string:";
        for (size_t i = 0; i < msg.stringArgs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << msg.stringArgs[i] << "\"";
        }
        oss << "]";
    }
    
    return oss.str();
}

std::string OSCOutputGenerator::buildAddress(const std::string& path) const {
    if (path.empty()) {
        return baseAddress_;
    }
    
    if (path[0] == '/') {
        return baseAddress_ + path;
    } else {
        return baseAddress_ + "/" + path;
    }
}

} // namespace kelly
