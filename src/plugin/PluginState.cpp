#include "plugin/PluginState.h"
#include "plugin/PluginProcessor.h"
#include <juce_core/juce_core.h>

namespace kelly {

PluginState::PluginState() {
    // Ensure presets directory exists
    getPresetsDirectory();
}

//==============================================================================
// State Persistence
//==============================================================================

juce::ValueTree PluginState::saveState(juce::AudioProcessorValueTreeState& apvts,
                                       const juce::String& woundDescription,
                                       const std::optional<int>& selectedEmotionId,
                                       const CassetteState& cassetteState) const {
    // Copy APVTS state
    auto state = apvts.copyState();
    
    // Add wound description
    if (woundDescription.isNotEmpty()) {
        state.setProperty("woundDescription", woundDescription, nullptr);
    }
    
    // Add selected emotion ID
    if (selectedEmotionId.has_value()) {
        state.setProperty("selectedEmotionId", *selectedEmotionId, nullptr);
    }
    
    // Add cassette state
    saveEmotionSettings(state, cassetteState);
    
    return state;
}

bool PluginState::loadState(juce::AudioProcessorValueTreeState& apvts,
                             juce::String& woundDescription,
                             std::optional<int>& selectedEmotionId,
                             CassetteState& cassetteState,
                             const juce::ValueTree& state) const {
    if (!state.isValid()) {
        return false;
    }
    
    // Restore APVTS state
    apvts.replaceState(state);
    
    // Restore wound description
    if (state.hasProperty("woundDescription")) {
        woundDescription = state.getProperty("woundDescription").toString();
    } else {
        woundDescription = juce::String();
    }
    
    // Restore selected emotion ID
    if (state.hasProperty("selectedEmotionId")) {
        selectedEmotionId = static_cast<int>(state.getProperty("selectedEmotionId"));
    } else {
        selectedEmotionId = std::nullopt;
    }
    
    // Restore cassette state
    cassetteState = loadEmotionSettings(state);
    
    return true;
}

void PluginState::saveEmotionSettings(juce::ValueTree& state, const CassetteState& cassetteState) const {
    // Save SideA
    auto sideATree = juce::ValueTree("SideA");
    sideATree.setProperty("description", juce::String(cassetteState.sideA.description), nullptr);
    sideATree.setProperty("intensity", cassetteState.sideA.intensity, nullptr);
    if (cassetteState.sideA.emotionId.has_value()) {
        sideATree.setProperty("emotionId", *cassetteState.sideA.emotionId, nullptr);
    }
    state.appendChild(sideATree, nullptr);
    
    // Save SideB
    auto sideBTree = juce::ValueTree("SideB");
    sideBTree.setProperty("description", juce::String(cassetteState.sideB.description), nullptr);
    sideBTree.setProperty("intensity", cassetteState.sideB.intensity, nullptr);
    if (cassetteState.sideB.emotionId.has_value()) {
        sideBTree.setProperty("emotionId", *cassetteState.sideB.emotionId, nullptr);
    }
    state.appendChild(sideBTree, nullptr);
    
    // Save flipped state
    state.setProperty("isFlipped", cassetteState.isFlipped, nullptr);
}

CassetteState PluginState::loadEmotionSettings(const juce::ValueTree& state) const {
    CassetteState cassetteState;
    
    // Load SideA
    auto sideATree = state.getChildWithName("SideA");
    if (sideATree.isValid()) {
        cassetteState.sideA.description = sideATree.getProperty("description").toString().toStdString();
        cassetteState.sideA.intensity = static_cast<float>(sideATree.getProperty("intensity"));
        if (sideATree.hasProperty("emotionId")) {
            cassetteState.sideA.emotionId = static_cast<int>(sideATree.getProperty("emotionId"));
        }
    }

    // Load SideB
    auto sideBTree = state.getChildWithName("SideB");
    if (sideBTree.isValid()) {
        cassetteState.sideB.description = sideBTree.getProperty("description").toString().toStdString();
        cassetteState.sideB.intensity = static_cast<float>(sideBTree.getProperty("intensity"));
        if (sideBTree.hasProperty("emotionId")) {
            cassetteState.sideB.emotionId = static_cast<int>(sideBTree.getProperty("emotionId"));
        }
    }
    
    // Load flipped state
    if (state.hasProperty("isFlipped")) {
        cassetteState.isFlipped = static_cast<bool>(state.getProperty("isFlipped"));
    }
    
    return cassetteState;
}

//==============================================================================
// Preset Management
//==============================================================================

juce::File PluginState::getPresetsDirectory() const {
    // Use JUCE's user application data directory
    juce::File presetsDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Kelly")
        .getChildFile("Presets");
    
    // Create directory if it doesn't exist
    if (!presetsDir.exists()) {
        presetsDir.createDirectory();
    }
    
    return presetsDir;
}

juce::File PluginState::getPresetFile(const juce::String& presetName) const {
    juce::String sanitized = sanitizePresetName(presetName);
    return getPresetsDirectory().getChildFile(sanitized + ".json");
}

juce::String PluginState::sanitizePresetName(const juce::String& name) const {
    // Remove invalid filename characters
    juce::String sanitized = name;
    sanitized = sanitized.replaceCharacters("/\\:*?\"<>|", "_");
    sanitized = sanitized.trim();
    
    // Ensure not empty
    if (sanitized.isEmpty()) {
        sanitized = "Untitled";
    }
    
    return sanitized;
}

bool PluginState::savePreset(const juce::String& presetName,
                              const juce::String& description,
                              const juce::String& author,
                              juce::AudioProcessorValueTreeState& apvts,
                              const juce::String& woundDescription,
                              const std::optional<int>& selectedEmotionId,
                              const CassetteState& cassetteState) const {
    // Create preset from current state
    Preset preset = createPresetFromState(presetName, description, author, apvts,
                                          woundDescription, selectedEmotionId, cassetteState);
    
    // Convert to JSON
    juce::var json = preset.toJson();
    
    // Write to file
    juce::File presetFile = getPresetFile(presetName);
    juce::String jsonString = juce::JSON::toString(json, true);
    bool success = presetFile.replaceWithText(jsonString);

    return success;
}

std::optional<PluginState::Preset> PluginState::loadPreset(const juce::String& presetName) const {
    juce::File presetFile = getPresetFile(presetName);
    
    if (!presetFile.existsAsFile()) {
        return std::nullopt;
    }
    
    juce::var jsonData;
    juce::Result result = juce::JSON::parse(presetFile.loadFileAsString(), jsonData);
    
    if (result.failed()) {
        return std::nullopt;
    }
    
    return Preset::fromJson(jsonData);
}

bool PluginState::deletePreset(const juce::String& presetName) const {
    juce::File presetFile = getPresetFile(presetName);
    
    if (!presetFile.existsAsFile()) {
        return false;
    }
    
    return presetFile.deleteFile();
}

std::vector<juce::String> PluginState::getPresetNames() const {
    std::vector<juce::String> names;
    juce::File presetsDir = getPresetsDirectory();
    
    if (!presetsDir.exists()) {
        return names;
    }
    
    juce::Array<juce::File> presetFiles;
    presetsDir.findChildFiles(presetFiles, juce::File::findFiles, false, "*.json");
    
    for (const auto& file : presetFiles) {
        // Load preset to get the actual name (not filename)
        auto preset = loadPreset(file.getFileNameWithoutExtension());
        if (preset.has_value()) {
            names.push_back(preset->name);
        } else {
            // Fallback to filename if preset can't be loaded
            names.push_back(file.getFileNameWithoutExtension());
        }
    }
    
    return names;
}

std::optional<PluginState::Preset> PluginState::getPresetMetadata(const juce::String& presetName) const {
    // For now, just load the full preset (could be optimized to read only metadata)
    return loadPreset(presetName);
}

bool PluginState::presetExists(const juce::String& presetName) const {
    return getPresetFile(presetName).existsAsFile();
}

bool PluginState::applyPreset(const Preset& preset,
                               juce::AudioProcessorValueTreeState& apvts,
                               juce::String& woundDescription,
                               std::optional<int>& selectedEmotionId,
                               CassetteState& cassetteState) const {
    // Set all parameters
    setParameterValue(apvts, PluginProcessor::PARAM_VALENCE, preset.valence);
    setParameterValue(apvts, PluginProcessor::PARAM_AROUSAL, preset.arousal);
    setParameterValue(apvts, PluginProcessor::PARAM_INTENSITY, preset.intensity);
    setParameterValue(apvts, PluginProcessor::PARAM_COMPLEXITY, preset.complexity);
    setParameterValue(apvts, PluginProcessor::PARAM_HUMANIZE, preset.humanize);
    setParameterValue(apvts, PluginProcessor::PARAM_FEEL, preset.feel);
    setParameterValue(apvts, PluginProcessor::PARAM_DYNAMICS, preset.dynamics);
    // Bars is an int parameter
    if (auto* barsParam = dynamic_cast<juce::AudioParameterInt*>(apvts.getParameter(PluginProcessor::PARAM_BARS))) {
        auto range = barsParam->getRange();
        int minValue = range.getStart();
        int maxValue = range.getEnd();
        float normalized = static_cast<float>(preset.bars - minValue) / static_cast<float>(maxValue - minValue);
        barsParam->setValueNotifyingHost(normalized);
    }
    
    // Bypass is a bool parameter
    if (auto* bypassParam = dynamic_cast<juce::AudioParameterBool*>(apvts.getParameter(PluginProcessor::PARAM_BYPASS))) {
        bypassParam->setValueNotifyingHost(preset.bypass ? 1.0f : 0.0f);
    }
    
    // Set wound description and emotion ID
    woundDescription = preset.woundDescription;
    selectedEmotionId = preset.selectedEmotionId;
    
    // Set cassette state
    cassetteState = preset.cassetteState;
    
    return true;
}

PluginState::Preset PluginState::createPresetFromState(const juce::String& name,
                                                        const juce::String& description,
                                                        const juce::String& author,
                                                        juce::AudioProcessorValueTreeState& apvts,
                                                        const juce::String& woundDescription,
                                                        const std::optional<int>& selectedEmotionId,
                                                        const CassetteState& cassetteState) const {
    Preset preset;
    preset.name = name;
    preset.description = description;
    preset.author = author;
    preset.createdTime = juce::Time::getCurrentTime();
    preset.modifiedTime = juce::Time::getCurrentTime();
    
    // Extract parameter values
    preset.valence = getParameterValue(apvts, PluginProcessor::PARAM_VALENCE);
    preset.arousal = getParameterValue(apvts, PluginProcessor::PARAM_AROUSAL);
    preset.intensity = getParameterValue(apvts, PluginProcessor::PARAM_INTENSITY);
    preset.complexity = getParameterValue(apvts, PluginProcessor::PARAM_COMPLEXITY);
    preset.humanize = getParameterValue(apvts, PluginProcessor::PARAM_HUMANIZE);
    preset.feel = getParameterValue(apvts, PluginProcessor::PARAM_FEEL);
    preset.dynamics = getParameterValue(apvts, PluginProcessor::PARAM_DYNAMICS);
    // Bars is an int parameter - get actual value
    if (auto* barsParam = dynamic_cast<juce::AudioParameterInt*>(apvts.getParameter(PluginProcessor::PARAM_BARS))) {
        preset.bars = barsParam->get();
    } else {
        preset.bars = static_cast<int>(getParameterValue(apvts, PluginProcessor::PARAM_BARS));
    }
    
    // Bypass is a bool parameter
    if (auto* bypassParam = dynamic_cast<juce::AudioParameterBool*>(apvts.getParameter(PluginProcessor::PARAM_BYPASS))) {
        preset.bypass = bypassParam->get();
    } else {
        auto* rawParam = apvts.getRawParameterValue(PluginProcessor::PARAM_BYPASS);
        preset.bypass = rawParam && *rawParam > 0.5f;
    }
    
    // Copy wound description and emotion ID
    preset.woundDescription = woundDescription;
    preset.selectedEmotionId = selectedEmotionId;
    
    // Copy cassette state
    preset.cassetteState = cassetteState;
    
    return preset;
}

float PluginState::getParameterValue(juce::AudioProcessorValueTreeState& apvts, const char* paramId) const {
    // Get the actual parameter value (denormalized)
    auto* param = apvts.getParameter(paramId);
    if (!param) {
        return 0.0f;
    }
    
    // Get normalized value and convert to actual range
    float normalized = param->getValue();
    
    if (auto* floatParam = dynamic_cast<juce::AudioParameterFloat*>(param)) {
        auto range = floatParam->getNormalisableRange();
        return range.convertFrom0to1(normalized);
    } else if (auto* intParam = dynamic_cast<juce::AudioParameterInt*>(param)) {
        auto range = intParam->getNormalisableRange();
        return range.convertFrom0to1(normalized);
    } else if (auto* boolParam = dynamic_cast<juce::AudioParameterBool*>(param)) {
        return normalized > 0.5f ? 1.0f : 0.0f;
    }
    
    // Fallback: return normalized value
    return normalized;
}

void PluginState::setParameterValue(juce::AudioProcessorValueTreeState& apvts, const char* paramId, float value) const {
    auto* param = apvts.getParameter(paramId);
    if (param) {
        // Convert actual value to normalized 0-1 range
        // For AudioParameterFloat, we need to get the range and convert
        if (auto* floatParam = dynamic_cast<juce::AudioParameterFloat*>(param)) {
            auto range = floatParam->getNormalisableRange();
            float normalized = range.convertTo0to1(value);
            param->setValueNotifyingHost(normalized);
        } else if (auto* intParam = dynamic_cast<juce::AudioParameterInt*>(param)) {
            // For int parameters, convert to normalized based on range
            int minValue = intParam->getRange().getStart();
            int maxValue = intParam->getRange().getEnd();
            float normalized = static_cast<float>(static_cast<int>(value) - minValue) / static_cast<float>(maxValue - minValue);
            param->setValueNotifyingHost(normalized);
        } else if (auto* boolParam = dynamic_cast<juce::AudioParameterBool*>(param)) {
            // For bool parameters, 0.0 = false, 1.0 = true
            param->setValueNotifyingHost(value > 0.5f ? 1.0f : 0.0f);
        }
    }
}

//==============================================================================
// Preset Serialization
//==============================================================================

juce::ValueTree PluginState::Preset::toValueTree() const {
    juce::ValueTree tree("Preset");
    
    tree.setProperty("name", name, nullptr);
    tree.setProperty("description", juce::String(description), nullptr);
    tree.setProperty("author", author, nullptr);
    tree.setProperty("createdTime", createdTime.toMilliseconds(), nullptr);
    tree.setProperty("modifiedTime", modifiedTime.toMilliseconds(), nullptr);
    
    // Parameters
    tree.setProperty("valence", valence, nullptr);
    tree.setProperty("arousal", arousal, nullptr);
    tree.setProperty("intensity", intensity, nullptr);
    tree.setProperty("complexity", complexity, nullptr);
    tree.setProperty("humanize", humanize, nullptr);
    tree.setProperty("feel", feel, nullptr);
    tree.setProperty("dynamics", dynamics, nullptr);
    tree.setProperty("bars", bars, nullptr);
    tree.setProperty("bypass", bypass, nullptr);
    
    // Intent/Emotion
    tree.setProperty("woundDescription", woundDescription, nullptr);
    if (selectedEmotionId.has_value()) {
        tree.setProperty("selectedEmotionId", *selectedEmotionId, nullptr);
    }
    
    // Cassette state
    auto cassetteTree = juce::ValueTree("CassetteState");
    auto sideATree = juce::ValueTree("SideA");
    sideATree.setProperty("description", juce::String(cassetteState.sideA.description), nullptr);
    sideATree.setProperty("intensity", cassetteState.sideA.intensity, nullptr);
    if (cassetteState.sideA.emotionId.has_value()) {
        sideATree.setProperty("emotionId", *cassetteState.sideA.emotionId, nullptr);
    }
    cassetteTree.appendChild(sideATree, nullptr);
    
    auto sideBTree = juce::ValueTree("SideB");
    sideBTree.setProperty("description", juce::String(cassetteState.sideB.description), nullptr);
    sideBTree.setProperty("intensity", cassetteState.sideB.intensity, nullptr);
    if (cassetteState.sideB.emotionId.has_value()) {
        sideBTree.setProperty("emotionId", *cassetteState.sideB.emotionId, nullptr);
    }
    cassetteTree.appendChild(sideBTree, nullptr);
    cassetteTree.setProperty("isFlipped", cassetteState.isFlipped, nullptr);
    
    tree.appendChild(cassetteTree, nullptr);
    
    return tree;
}

std::optional<PluginState::Preset> PluginState::Preset::fromValueTree(const juce::ValueTree& tree) {
    if (!tree.isValid() || !tree.hasType("Preset")) {
        return std::nullopt;
    }
    
    Preset preset;
    
    preset.name = tree.getProperty("name").toString();
    preset.description = tree.getProperty("description").toString();
    preset.author = tree.getProperty("author").toString();
    
    if (tree.hasProperty("createdTime")) {
        preset.createdTime = juce::Time(tree.getProperty("createdTime").toString().getLargeIntValue());
    }
    if (tree.hasProperty("modifiedTime")) {
        preset.modifiedTime = juce::Time(tree.getProperty("modifiedTime").toString().getLargeIntValue());
    }
    
    // Parameters
    preset.valence = static_cast<float>(tree.getProperty("valence"));
    preset.arousal = static_cast<float>(tree.getProperty("arousal"));
    preset.intensity = static_cast<float>(tree.getProperty("intensity"));
    preset.complexity = static_cast<float>(tree.getProperty("complexity"));
    preset.humanize = static_cast<float>(tree.getProperty("humanize"));
    preset.feel = static_cast<float>(tree.getProperty("feel"));
    preset.dynamics = static_cast<float>(tree.getProperty("dynamics"));
    preset.bars = static_cast<int>(tree.getProperty("bars"));
    preset.bypass = static_cast<bool>(tree.getProperty("bypass"));
    
    // Intent/Emotion
    preset.woundDescription = tree.getProperty("woundDescription").toString();
    if (tree.hasProperty("selectedEmotionId")) {
        preset.selectedEmotionId = static_cast<int>(tree.getProperty("selectedEmotionId"));
    }
    
    // Cassette state
    auto cassetteTree = tree.getChildWithName("CassetteState");
    if (cassetteTree.isValid()) {
        auto sideATree = cassetteTree.getChildWithName("SideA");
        if (sideATree.isValid()) {
            preset.cassetteState.sideA.description = sideATree.getProperty("description").toString().toStdString();
            preset.cassetteState.sideA.intensity = static_cast<float>(sideATree.getProperty("intensity"));
            if (sideATree.hasProperty("emotionId")) {
                preset.cassetteState.sideA.emotionId = static_cast<int>(sideATree.getProperty("emotionId"));
            }
        }

        auto sideBTree = cassetteTree.getChildWithName("SideB");
        if (sideBTree.isValid()) {
            preset.cassetteState.sideB.description = sideBTree.getProperty("description").toString().toStdString();
            preset.cassetteState.sideB.intensity = static_cast<float>(sideBTree.getProperty("intensity"));
            if (sideBTree.hasProperty("emotionId")) {
                preset.cassetteState.sideB.emotionId = static_cast<int>(sideBTree.getProperty("emotionId"));
            }
        }
        
        preset.cassetteState.isFlipped = static_cast<bool>(cassetteTree.getProperty("isFlipped"));
    }
    
    return preset;
}

juce::var PluginState::Preset::toJson() const {
    // Note: JUCE's DynamicObject::Ptr is a reference-counted smart pointer,
    // so using 'new' is correct and the Ptr manages lifetime automatically
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    
    obj->setProperty("name", name);
    obj->setProperty("description", juce::String(description));
    obj->setProperty("author", author);
    obj->setProperty("createdTime", createdTime.toMilliseconds());
    obj->setProperty("modifiedTime", modifiedTime.toMilliseconds());
    
    // Parameters
    obj->setProperty("valence", valence);
    obj->setProperty("arousal", arousal);
    obj->setProperty("intensity", intensity);
    obj->setProperty("complexity", complexity);
    obj->setProperty("humanize", humanize);
    obj->setProperty("feel", feel);
    obj->setProperty("dynamics", dynamics);
    obj->setProperty("bars", bars);
    obj->setProperty("bypass", bypass);
    
    // Intent/Emotion
    obj->setProperty("woundDescription", woundDescription);
    if (selectedEmotionId.has_value()) {
        obj->setProperty("selectedEmotionId", *selectedEmotionId);
    }
    
    // Cassette state
    juce::DynamicObject::Ptr cassetteObj = new juce::DynamicObject();
    
    juce::DynamicObject::Ptr sideAObj = new juce::DynamicObject();
    sideAObj->setProperty("description", juce::String(cassetteState.sideA.description));
    sideAObj->setProperty("intensity", cassetteState.sideA.intensity);
    if (cassetteState.sideA.emotionId.has_value()) {
        sideAObj->setProperty("emotionId", *cassetteState.sideA.emotionId);
    }
    cassetteObj->setProperty("sideA", juce::var(sideAObj.get()));

    juce::DynamicObject::Ptr sideBObj = new juce::DynamicObject();
    sideBObj->setProperty("description", juce::String(cassetteState.sideB.description));
    sideBObj->setProperty("intensity", cassetteState.sideB.intensity);
    if (cassetteState.sideB.emotionId.has_value()) {
        sideBObj->setProperty("emotionId", *cassetteState.sideB.emotionId);
    }
    cassetteObj->setProperty("sideB", juce::var(sideBObj.get()));
    cassetteObj->setProperty("isFlipped", cassetteState.isFlipped);
    
    obj->setProperty("cassetteState", juce::var(cassetteObj.get()));
    
    return juce::var(obj.get());
}

std::optional<PluginState::Preset> PluginState::Preset::fromJson(const juce::var& json) {
    if (!json.isObject()) {
        return std::nullopt;
    }
    
    Preset preset;
    
    auto* obj = json.getDynamicObject();
    if (obj == nullptr) {
        return std::nullopt;
    }
    
    preset.name = obj->getProperty("name").toString();
    preset.description = obj->getProperty("description").toString();
    preset.author = obj->getProperty("author").toString();
    
    if (obj->hasProperty("createdTime")) {
        juce::int64 ms = static_cast<juce::int64>(obj->getProperty("createdTime"));
        preset.createdTime = juce::Time(ms);
    }
    if (obj->hasProperty("modifiedTime")) {
        juce::int64 ms = static_cast<juce::int64>(obj->getProperty("modifiedTime"));
        preset.modifiedTime = juce::Time(ms);
    }
    
    // Parameters
    preset.valence = static_cast<float>(obj->getProperty("valence"));
    preset.arousal = static_cast<float>(obj->getProperty("arousal"));
    preset.intensity = static_cast<float>(obj->getProperty("intensity"));
    preset.complexity = static_cast<float>(obj->getProperty("complexity"));
    preset.humanize = static_cast<float>(obj->getProperty("humanize"));
    preset.feel = static_cast<float>(obj->getProperty("feel"));
    preset.dynamics = static_cast<float>(obj->getProperty("dynamics"));
    preset.bars = static_cast<int>(obj->getProperty("bars"));
    preset.bypass = static_cast<bool>(obj->getProperty("bypass"));
    
    // Intent/Emotion
    preset.woundDescription = obj->getProperty("woundDescription").toString();
    if (obj->hasProperty("selectedEmotionId")) {
        preset.selectedEmotionId = static_cast<int>(obj->getProperty("selectedEmotionId"));
    }
    
    // Cassette state
    if (obj->hasProperty("cassetteState")) {
        auto cassetteVar = obj->getProperty("cassetteState");
        if (cassetteVar.isObject()) {
            auto* cassetteObj = cassetteVar.getDynamicObject();
            if (cassetteObj != nullptr) {
                if (cassetteObj->hasProperty("sideA")) {
                    auto sideAVar = cassetteObj->getProperty("sideA");
                    if (sideAVar.isObject()) {
                        auto* sideAObj = sideAVar.getDynamicObject();
                        if (sideAObj != nullptr) {
                            preset.cassetteState.sideA.description = sideAObj->getProperty("description").toString().toStdString();
                            preset.cassetteState.sideA.intensity = static_cast<float>(sideAObj->getProperty("intensity"));
                            if (sideAObj->hasProperty("emotionId")) {
                                preset.cassetteState.sideA.emotionId = static_cast<int>(sideAObj->getProperty("emotionId"));
                            }
                        }
                    }
                }

                if (cassetteObj->hasProperty("sideB")) {
                    auto sideBVar = cassetteObj->getProperty("sideB");
                    if (sideBVar.isObject()) {
                        auto* sideBObj = sideBVar.getDynamicObject();
                        if (sideBObj != nullptr) {
                            preset.cassetteState.sideB.description = sideBObj->getProperty("description").toString().toStdString();
                            preset.cassetteState.sideB.intensity = static_cast<float>(sideBObj->getProperty("intensity"));
                            if (sideBObj->hasProperty("emotionId")) {
                                preset.cassetteState.sideB.emotionId = static_cast<int>(sideBObj->getProperty("emotionId"));
                            }
                        }
                    }
                }
                
                preset.cassetteState.isFlipped = static_cast<bool>(cassetteObj->getProperty("isFlipped"));
            }
        }
    }
    
    return preset;
}

} // namespace kelly
