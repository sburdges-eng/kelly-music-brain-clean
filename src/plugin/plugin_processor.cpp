#include "plugin_processor.h"
#include "plugin_editor.h"
#include <chrono>

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
        juce::AudioParameterFloatAttributes().withLabel("Negative ↔ Positive")
    ));
    
    // Arousal: 0.0 to 1.0
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        ParameterID::arousal,
        "Arousal",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.2f,
        juce::AudioParameterFloatAttributes().withLabel("Calm ↔ Excited")
    ));
    
    // Intensity: 0.0 to 1.0
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        ParameterID::intensity,
        "Intensity",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.4f,
        juce::AudioParameterFloatAttributes().withLabel("Subtle ↔ Extreme")
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

} // namespace kelly

// JUCE plugin entry point
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new kelly::PluginProcessor();
}
