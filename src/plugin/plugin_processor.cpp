#include "plugin_processor.h"
#include "plugin_editor.h"
#include <chrono>
#include <map>

namespace kelly {

PluginProcessor::PluginProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("MIDI In", juce::AudioChannelSet::disabled(), true)
                         .withOutput("MIDI Out", juce::AudioChannelSet::disabled(), true)),
      parameters_(*this, nullptr, juce::Identifier("KellyEmotionProcessor"), createParameterLayout()),
      aiEngine_([]
      {
          ml::AIConfig cfg;
          auto onnxPath = juce::SystemStats::getEnvironmentVariable("KELLY_ONNX_MODEL", {});
          auto cloudUrl = juce::SystemStats::getEnvironmentVariable("KELLY_AI_ENDPOINT", {});
          if (onnxPath.isNotEmpty()) {
              cfg.onnxModelPath = onnxPath.toStdString();
          }
          if (cloudUrl.isNotEmpty()) {
              cfg.enableCloudFallback = true;
              cfg.cloudEndpoint = cloudUrl.toStdString();
          }
          return cfg;
      }()),
      cloudEngine_([]
      {
          ml::AIConfig cfg;
          auto onnxPath = juce::SystemStats::getEnvironmentVariable("KELLY_ONNX_MODEL", {});
          auto cloudUrl = juce::SystemStats::getEnvironmentVariable("KELLY_AI_ENDPOINT", {});
          if (onnxPath.isNotEmpty()) {
              cfg.onnxModelPath = onnxPath.toStdString();
          }
          if (cloudUrl.isNotEmpty()) {
              cfg.enableCloudFallback = true;
              cfg.cloudEndpoint = cloudUrl.toStdString();
          }
          return cfg;
      }()) {
    
    // Add parameter change listener
    parameters_.addParameterListener(ParameterID::emotionTag, this);
    parameters_.addParameterListener(ParameterID::valence, this);
    parameters_.addParameterListener(ParameterID::arousal, this);
    parameters_.addParameterListener(ParameterID::intensity, this);
    parameters_.addParameterListener(ParameterID::bars, this);
    parameters_.addParameterListener(ParameterID::enableCloud, this);
    parameters_.addParameterListener(ParameterID::generationRate, this);
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    // Emotion tag (choice parameter)
    juce::StringArray emotionTags;
    emotionTags.add("calm");
    emotionTags.add("excited");
    emotionTags.add("sad");
    emotionTags.add("angry");
    emotionTags.add("happy");
    emotionTags.add("anxious");
    emotionTags.add("peaceful");
    emotionTags.add("energetic");
    emotionTags.add("neutral");
    
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        ParameterID::emotionTag,
        "Emotion",
        emotionTags,
        0  // default: "calm"
    ));
    
    // Valence: -1.0 to 1.0
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        ParameterID::valence,
        "Valence",
        juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f),
        0.1f,
        juce::AudioParameterFloatAttributes().withLabel("Negative <-> Positive")
    ));
    
    // Arousal: 0.0 to 1.0
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        ParameterID::arousal,
        "Arousal",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.2f,
        juce::AudioParameterFloatAttributes().withLabel("Calm <-> Excited")
    ));
    
    // Intensity: 0.0 to 1.0
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        ParameterID::intensity,
        "Intensity",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.4f,
        juce::AudioParameterFloatAttributes().withLabel("Subtle <-> Extreme")
    ));
    
    // Bars: 1 to 8
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        ParameterID::bars,
        "Bars",
        1, 8, 1,
        juce::AudioParameterIntAttributes().withLabel("bars")
    ));
    
    // Enable cloud generation (boolean)
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        ParameterID::enableCloud,
        "Enable Cloud",
        false
    ));
    
    // Generation rate: 1 to 32 blocks (controls how often to generate)
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        ParameterID::generationRate,
        "Generation Rate",
        1, 32, 8,
        juce::AudioParameterIntAttributes().withLabel("blocks")
    ));
    
    return { params.begin(), params.end() };
}

void PluginProcessor::parameterChanged(const juce::String& parameterID, float newValue) {
    // Update cached emotion request when parameters change
    const juce::ScopedLock lock(emotionRequestLock_);
    
    if (parameterID == ParameterID::emotionTag) {
        auto* choiceParam = dynamic_cast<juce::AudioParameterChoice*>(
            parameters_.getParameter(ParameterID::emotionTag));
        if (choiceParam) {
            emotionRequest_.tag = choiceParam->getCurrentChoiceName().toStdString();
        }
    } else if (parameterID == ParameterID::valence) {
        emotionRequest_.valence = newValue;
    } else if (parameterID == ParameterID::arousal) {
        emotionRequest_.arousal = newValue;
    } else if (parameterID == ParameterID::intensity) {
        emotionRequest_.intensity = newValue;
    } else if (parameterID == ParameterID::bars) {
        emotionRequest_.bars = static_cast<int>(newValue);
    }
    
    // Update tempo if available
    emotionRequest_.tempo = static_cast<int>(bpm_);
    
    // Mark project as modified
    projectModified_ = true;
}

ml::EmotionRequest PluginProcessor::getEmotionRequest() const {
    const juce::ScopedLock lock(emotionRequestLock_);
    
    // Update from current parameter values
    auto* choiceParam = dynamic_cast<juce::AudioParameterChoice*>(
        parameters_.getParameter(ParameterID::emotionTag));
    if (choiceParam) {
        emotionRequest_.tag = choiceParam->getCurrentChoiceName().toStdString();
    }
    
    emotionRequest_.valence = parameters_.getRawParameterValue(ParameterID::valence)->load();
    emotionRequest_.arousal = parameters_.getRawParameterValue(ParameterID::arousal)->load();
    emotionRequest_.intensity = parameters_.getRawParameterValue(ParameterID::intensity)->load();
    emotionRequest_.bars = static_cast<int>(parameters_.getRawParameterValue(ParameterID::bars)->load());
    emotionRequest_.tempo = static_cast<int>(bpm_);
    
    return emotionRequest_;
}

bool PluginProcessor::isCloudEnabled() const {
    return parameters_.getRawParameterValue(ParameterID::enableCloud)->load() > 0.5f;
}

int PluginProcessor::getGenerationRate() const {
    return static_cast<int>(parameters_.getRawParameterValue(ParameterID::generationRate)->load());
}

void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;
    blockSize_ = samplesPerBlock;
    aiEngine_.setTransport(bpm_, sampleRate_, blockSize_);
    cloudEngine_.setTransport(bpm_, sampleRate_, blockSize_);
}

void PluginProcessor::releaseResources() {
    // Release resources
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;
    
    buffer.clear(); // MIDI effect: no audio path

    // Update transport info from host
    if (auto* playHead = getPlayHead()) {
        juce::AudioPlayHead::CurrentPositionInfo info;
        if (playHead->getCurrentPosition(info) && info.bpm > 0.0) {
            bpm_ = info.bpm;
            aiEngine_.setTransport(bpm_, sampleRate_, blockSize_);
            cloudEngine_.setTransport(bpm_, sampleRate_, blockSize_);
            
            // Update emotion request tempo
            {
                const juce::ScopedLock lock(emotionRequestLock_);
                emotionRequest_.tempo = static_cast<int>(bpm_);
            }
        }
    }

    // Get current generation rate
    int genRate = getGenerationRate();
    bool shouldGenerate = (generationCounter_.load() % genRate) == 0;
    
    if (shouldGenerate) {
        // Get current emotion request (thread-safe)
        auto request = getEmotionRequest();
        
        // Generate MIDI from AI (on-device first)
        try {
            auto generated = aiEngine_.generate(request);
            writeMidiToBuffer(generated, midiMessages, bpm_, sampleRate_, blockSize_);
            
            // Store generated events
            addGeneratedMidi(generated);
        } catch (const std::exception& e) {
            // Log error but continue processing
            juce::Logger::writeToLog("AI generation error: " + juce::String(e.what()));
        }
    }
    
    generationCounter_.fetch_add(1, std::memory_order_relaxed);

    // Merge any ready cloud results
    if (cloudFuture_.valid() &&
        cloudFuture_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        try {
            auto cloudEvents = cloudFuture_.get();
            {
                const juce::ScopedLock lock(cloudLock_);
                cloudMidi_.clear();
                writeMidiToBuffer(cloudEvents, cloudMidi_, bpm_, sampleRate_, blockSize_);
            }
            cloudInFlight_.store(false, std::memory_order_release);
            cloudCooldown_.store(0, std::memory_order_release);
        } catch (const std::exception& e) {
            juce::Logger::writeToLog("Cloud generation error: " + juce::String(e.what()));
            cloudInFlight_.store(false, std::memory_order_release);
        }
    }

    // Add cloud MIDI to output
    {
        const juce::ScopedLock lock(cloudLock_);
        if (cloudMidi_.getNumEvents() > 0) {
            midiMessages.addEvents(cloudMidi_, 0, blockSize_, 0);
            cloudMidi_.clear();
        }
    }

    // Schedule cloud inference if enabled and no local model
    bool cloudEnabled = isCloudEnabled();
    if (cloudEnabled && !cloudInFlight_.load(std::memory_order_acquire) && !aiEngine_.hasOnnx()) {
        int cooldown = cloudCooldown_.load(std::memory_order_acquire);
        if (cooldown > 8) { // throttle requests
            cloudInFlight_.store(true, std::memory_order_release);
            auto request = getEmotionRequest();
            cloudFuture_ = std::async(std::launch::async, [this, request]() {
                try {
                    return cloudEngine_.generate(request, true);
                } catch (const std::exception& e) {
                    juce::Logger::writeToLog("Async cloud generation error: " + juce::String(e.what()));
                    return std::vector<ml::MidiEvent>();
                }
            });
            cloudCooldown_.store(0, std::memory_order_release);
        } else {
            cloudCooldown_.store(cooldown + 1, std::memory_order_release);
        }
    }
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() {
    return new PluginEditor(*this);
}

void PluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
    // Save parameter state using ValueTree
    auto state = parameters_.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    // Restore parameter state from binary data
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    
    if (xmlState != nullptr) {
        if (xmlState->hasTagName(parameters_.state.getType())) {
            parameters_.replaceState(juce::ValueTree::fromXml(*xmlState));
            
            // Update emotion request from restored parameters
            parameterChanged(ParameterID::emotionTag, 0.0f);
            parameterChanged(ParameterID::valence, 0.0f);
            parameterChanged(ParameterID::arousal, 0.0f);
            parameterChanged(ParameterID::intensity, 0.0f);
            parameterChanged(ParameterID::bars, 0.0f);
        }
    }
}

bool PluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto in = layouts.getMainInputChannelSet();
    const auto out = layouts.getMainOutputChannelSet();
    return in == juce::AudioChannelSet::disabled() && out == juce::AudioChannelSet::disabled();
}

//==============================================================================
// Project Management
//==============================================================================

bool PluginProcessor::saveCurrentProject(const juce::File& file) {
    midikompanion::ProjectData data = getCurrentProjectData();
    
    bool success = projectManager_.saveProject(file, data);
    
    if (success) {
        currentProjectFile_ = file;
        projectModified_ = false;
        juce::Logger::writeToLog("Project saved: " + file.getFullPathName());
    } else {
        juce::Logger::writeToLog("Failed to save project: " + projectManager_.getLastError());
    }
    
    return success;
}

bool PluginProcessor::loadProject(const juce::File& file) {
    midikompanion::ProjectData data;
    
    bool success = projectManager_.loadProject(file, data);
    
    if (success) {
        currentProjectData_ = data;
        currentProjectFile_ = file;
        projectModified_ = false;
        
        // Restore tempo
        bpm_ = data.tempo;
        
        // Restore plugin state parameters
        for (const auto& [paramId, value] : data.pluginState.parameters) {
            if (auto* param = parameters_.getParameter(paramId)) {
                param->setValueNotifyingHost(value);
            }
        }
        
        // Restore generated tracks
        {
            const juce::ScopedLock lock(midiStorageLock_);
            generatedTracks_ = data.tracks;
            
            // Rebuild MIDI sequence from tracks
            generatedMidi_.clear();
            for (const auto& track : data.tracks) {
                for (const auto& note : track.notes) {
                    int startTicks = static_cast<int>(note.startBeat * 480.0);
                    int endTicks = static_cast<int>((note.startBeat + note.durationBeats) * 480.0);
                    
                    juce::MidiMessage noteOn = juce::MidiMessage::noteOn(
                        note.channel, note.pitch, static_cast<juce::uint8>(note.velocity));
                    noteOn.setTimeStamp(startTicks);
                    generatedMidi_.addEvent(noteOn);
                    
                    juce::MidiMessage noteOff = juce::MidiMessage::noteOff(note.channel, note.pitch);
                    noteOff.setTimeStamp(endTicks);
                    generatedMidi_.addEvent(noteOff);
                }
            }
            generatedMidi_.sort();
            generatedMidi_.updateMatchedPairs();
        }
        
        // Update emotion request
        {
            const juce::ScopedLock lock(emotionRequestLock_);
            emotionRequest_.valence = data.pluginState.emotionState.valence;
            emotionRequest_.arousal = data.pluginState.emotionState.arousal;
            emotionRequest_.intensity = data.pluginState.emotionState.intensity;
            emotionRequest_.tag = data.pluginState.emotionState.emotionTag;
            emotionRequest_.tempo = static_cast<int>(data.tempo);
        }
        
        juce::Logger::writeToLog("Project loaded: " + file.getFullPathName());
    } else {
        juce::Logger::writeToLog("Failed to load project: " + projectManager_.getLastError());
    }
    
    return success;
}

void PluginProcessor::createNewProject() {
    currentProjectData_ = midikompanion::ProjectData();
    currentProjectFile_ = juce::File();
    projectModified_ = false;
    
    {
        const juce::ScopedLock lock(midiStorageLock_);
        generatedMidi_.clear();
        generatedTracks_.clear();
    }
    
    bpm_ = 120.0;
    juce::Logger::writeToLog("New project created");
}

midikompanion::ProjectData PluginProcessor::getCurrentProjectData() const {
    midikompanion::ProjectData data = currentProjectData_;
    
    data.tempo = bpm_;
    
    // Update plugin state from current parameters
    data.pluginState.parameters.clear();
    for (auto* param : getParameters()) {
        if (auto* rangedParam = dynamic_cast<juce::RangedAudioParameter*>(param)) {
            data.pluginState.parameters[rangedParam->getParameterID().toStdString()] = 
                rangedParam->getValue();
        }
    }
    
    // Update emotion state
    {
        const juce::ScopedLock lock(emotionRequestLock_);
        data.pluginState.emotionState.emotionTag = emotionRequest_.tag;
        data.pluginState.emotionState.valence = emotionRequest_.valence;
        data.pluginState.emotionState.arousal = emotionRequest_.arousal;
        data.pluginState.emotionState.intensity = emotionRequest_.intensity;
    }
    
    data.pluginState.bars = static_cast<int>(parameters_.getRawParameterValue(ParameterID::bars)->load());
    data.pluginState.enableCloud = isCloudEnabled();
    data.pluginState.generationRate = getGenerationRate();
    
    {
        const juce::ScopedLock lock(midiStorageLock_);
        data.tracks = generatedTracks_;
    }
    
    return data;
}

//==============================================================================
// MIDI Export
//==============================================================================

midikompanion::MidiExportResult PluginProcessor::exportToMidi(const juce::File& file,
                                                              const midikompanion::MidiExportOptions& options) {
    const juce::ScopedLock lock(midiStorageLock_);
    
    if (generatedMidi_.getNumEvents() == 0 && generatedTracks_.empty()) {
        midikompanion::MidiExportResult result;
        result.success = false;
        result.errorMessage = "No MIDI data to export";
        return result;
    }
    
    midikompanion::MidiExportData exportData;
    exportData.tempo = bpm_;
    exportData.timeSignatureNumerator = 4;
    exportData.timeSignatureDenominator = 4;
    
    // Convert generated tracks to export format
    for (const auto& track : generatedTracks_) {
        midikompanion::MidiExportTrack exportTrack;
        exportTrack.name = track.name;
        exportTrack.midiChannel = track.midiChannel;
        
        for (const auto& note : track.notes) {
            midikompanion::MidiExportNote exportNote;
            exportNote.pitch = note.pitch;
            exportNote.velocity = note.velocity;
            exportNote.startBeat = note.startBeat;
            exportNote.durationBeats = note.durationBeats;
            exportNote.channel = note.channel;
            exportTrack.notes.push_back(exportNote);
        }
        
        exportData.tracks.push_back(exportTrack);
    }
    
    // If no tracks but have raw MIDI, convert it
    if (exportData.tracks.empty() && !generatedMidi_.isEmpty()) {
        midikompanion::MidiExportTrack defaultTrack;
        defaultTrack.name = "Generated";
        defaultTrack.midiChannel = 0;
        
        std::map<int, std::pair<double, int>> activeNotes;
        
        for (int i = 0; i < generatedMidi_.getNumEvents(); ++i) {
            auto* event = generatedMidi_.getEventPointer(i);
            auto& msg = event->message;
            double beat = msg.getTimeStamp() / 480.0;
            
            if (msg.isNoteOn() && msg.getVelocity() > 0) {
                activeNotes[msg.getNoteNumber()] = {beat, msg.getVelocity()};
            } else if (msg.isNoteOff() || (msg.isNoteOn() && msg.getVelocity() == 0)) {
                auto it = activeNotes.find(msg.getNoteNumber());
                if (it != activeNotes.end()) {
                    midikompanion::MidiExportNote exportNote;
                    exportNote.pitch = msg.getNoteNumber();
                    exportNote.velocity = it->second.second;
                    exportNote.startBeat = it->second.first;
                    exportNote.durationBeats = beat - it->second.first;
                    exportNote.channel = msg.getChannel() - 1;
                    defaultTrack.notes.push_back(exportNote);
                    activeNotes.erase(it);
                }
            }
        }
        
        if (!defaultTrack.notes.empty()) {
            exportData.tracks.push_back(defaultTrack);
        }
    }
    
    return midiExporter_.exportToFile(file, exportData, options);
}

void PluginProcessor::addGeneratedMidi(const std::vector<ml::MidiEvent>& events) {
    const juce::ScopedLock lock(midiStorageLock_);
    
    for (const auto& event : events) {
        int startTicks = static_cast<int>(event.startBeat * 480.0);
        int endTicks = static_cast<int>((event.startBeat + event.durationBeats) * 480.0);
        
        juce::MidiMessage noteOn = juce::MidiMessage::noteOn(
            event.channel, event.note, static_cast<juce::uint8>(event.velocity));
        noteOn.setTimeStamp(startTicks);
        generatedMidi_.addEvent(noteOn);
        
        juce::MidiMessage noteOff = juce::MidiMessage::noteOff(event.channel, event.note);
        noteOff.setTimeStamp(endTicks);
        generatedMidi_.addEvent(noteOff);
    }
    
    generatedMidi_.sort();
    generatedMidi_.updateMatchedPairs();
    
    // Create default track if empty
    if (generatedTracks_.empty()) {
        midikompanion::TrackData melodyTrack;
        melodyTrack.name = "Melody";
        melodyTrack.type = "melody";
        melodyTrack.midiChannel = 0;
        generatedTracks_.push_back(melodyTrack);
    }
    
    // Add notes to first track
    for (const auto& event : events) {
        midikompanion::MidiNoteData note;
        note.pitch = event.note;
        note.velocity = event.velocity;
        note.startBeat = event.startBeat;
        note.durationBeats = event.durationBeats;
        note.channel = event.channel;
        generatedTracks_[0].notes.push_back(note);
    }
    
    projectModified_ = true;
}

} // namespace kelly

// JUCE plugin entry point
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new kelly::PluginProcessor();
}
