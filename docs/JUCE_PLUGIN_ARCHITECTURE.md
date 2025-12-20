# JUCE Plugin Architecture

## Overview

The miDiKompanion project implements a JUCE-based audio plugin architecture that provides real-time emotion-driven MIDI generation and processing. The plugin follows JUCE's standard AudioProcessor/AudioProcessorEditor pattern with custom extensions for AI-powered music generation.

## Architecture Components

### 1. Plugin Processor (`PluginProcessor`)

The `PluginProcessor` class is the core audio processing unit that inherits from `juce::AudioProcessor`. It handles all real-time audio and MIDI processing.

#### Class Structure

```cpp
class PluginProcessor : public juce::AudioProcessor,
                         public juce::AudioProcessorValueTreeState::Listener {
    // Audio processing lifecycle
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    
    // Editor management
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;
    
    // Plugin metadata
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    
    // State persistence
    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;
    
    // Parameter management
    juce::AudioProcessorValueTreeState& getParameters();
    ml::EmotionRequest getEmotionRequest() const;
    bool isCloudEnabled() const;
    int getGenerationRate() const;
    
    // Parameter change callback
    void parameterChanged(const juce::String& parameterID, float newValue) override;
    
private:
    juce::AudioProcessorValueTreeState parameters_;
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
};
```

#### Key Responsibilities

1. **Audio Processing Lifecycle**
   - `prepareToPlay()`: Initializes DSP engines with sample rate and block size
   - `processBlock()`: Real-time audio/MIDI processing callback
   - `releaseResources()`: Cleanup when playback stops

2. **MIDI Processing**
   - Accepts MIDI input from host DAW
   - Generates MIDI output via AI inference engines
   - Merges local and cloud-generated MIDI events

3. **Transport Synchronization**
   - Reads BPM from host playhead
   - Updates AI engines with current tempo
   - Synchronizes MIDI generation with host timeline

4. **AI Engine Management**
   - On-device AI inference (ONNX models)
   - Cloud fallback for complex generation
   - Thread-safe async cloud requests

5. **Parameter Management**
   - `AudioProcessorValueTreeState` for all plugin parameters
   - DAW automation support for all parameters
   - Real-time parameter updates via listener callbacks
   - Thread-safe parameter access in audio thread

#### Implementation Details

**Bus Configuration**
```cpp
PluginProcessor::PluginProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("MIDI In", juce::AudioChannelSet::disabled(), true)
        .withOutput("MIDI Out", juce::AudioChannelSet::disabled(), true))
```

The plugin is configured as a MIDI effect with no audio I/O buses, making it suitable for MIDI processing and generation.

**AI Engine Initialization**
```cpp
ml::AIConfig cfg;
auto onnxPath = juce::SystemStats::getEnvironmentVariable("KELLY_ONNX_MODEL", {});
auto cloudUrl = juce::SystemStats::getEnvironmentVariable("KELLY_AI_ENDPOINT", {});
```

AI engines are configured via environment variables, allowing flexible deployment scenarios.

**Real-Time Processing**
```cpp
void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, 
                                    juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;
    buffer.clear(); // MIDI effect: no audio path
    
    // Update transport info
    if (auto* playHead = getPlayHead()) {
        juce::AudioPlayHead::CurrentPositionInfo info;
        if (playHead->getCurrentPosition(info) && info.bpm > 0.0) {
            bpm_ = info.bpm;
            aiEngine_.setTransport(bpm_, sampleRate_, blockSize_);
        }
    }
    
    // Generate MIDI from AI
    auto generated = aiEngine_.generate(emotionRequest_);
    writeMidiToBuffer(generated, midiMessages, bpm_, sampleRate_, blockSize_);
    
    // Merge cloud results if ready
    // ...
}
```

### 2. Plugin Editor (`PluginEditor`)

The `PluginEditor` class provides a complete user interface for the plugin, inheriting from `juce::AudioProcessorEditor` and `juce::Timer` for real-time updates.

#### Class Structure

```cpp
class PluginEditor : public juce::AudioProcessorEditor,
                     public juce::Timer {
public:
    explicit PluginEditor(PluginProcessor&);
    ~PluginEditor() override;
    
    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;  // Real-time UI updates
    
private:
    PluginProcessor& processor;
    
    // UI Components
    juce::Label titleLabel_;
    juce::ComboBox emotionComboBox_;
    
    juce::Slider valenceSlider_, arousalSlider_, intensitySlider_;
    juce::Slider barsSlider_, generationRateSlider_;
    juce::ToggleButton cloudToggle_;
    
    // Parameter attachments (maintain lifetime)
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> valenceAttachment_;
    // ... more attachments
    
    // Helper methods
    void setupSlider(...);
    void updateValueLabels();
};
```

#### Key Responsibilities

1. **UI Rendering**
   - `paint()`: Custom gradient background and styling
   - `resized()`: Professional layout management for all components

2. **Parameter Binding**
   - Binds all UI controls to processor parameters via attachments
   - Automatic parameter synchronization with DAW automation
   - Real-time value label updates

3. **Real-Time Updates**
   - Timer-based updates (100ms interval) for value displays
   - Status indicator showing cloud/local processing state
   - Dynamic UI feedback based on processor state

#### Current Implementation

The editor provides a complete, professional UI with:

**Parameter Controls:**
- Emotion selector (ComboBox with 9 emotion options)
- Valence slider (-1.0 to 1.0) with value display
- Arousal slider (0.0 to 1.0) with value display
- Intensity slider (0.0 to 1.0) with value display
- Bars slider (1 to 8) with value display
- Generation rate slider (1 to 32 blocks) with value display
- Cloud enable toggle button

**Visual Design:**
- Gradient background (dark blue theme)
- Professional spacing and layout
- Real-time value labels next to each slider
- Status indicator at bottom
- Border styling

**Example UI Setup:**
```cpp
void PluginEditor::setupSlider(juce::Slider& slider, juce::Label& label,
                               const juce::String& parameterID,
                               const juce::String& labelText,
                               juce::NormalisableRange<float> range) {
    label.setText(labelText, juce::dontSendNotification);
    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setRange(range.start, range.end, range.interval);
    
    // Create attachment for automatic parameter binding
    auto attachment = std::make_unique<
        juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.getParameters(), parameterID, slider);
    // Store attachment to maintain lifetime
}
```

## Plugin Parameters

The plugin exposes 7 automatable parameters via `AudioProcessorValueTreeState`:

| Parameter ID | Type | Range | Default | Description |
|--------------|------|-------|---------|-------------|
| `emotionTag` | Choice | 9 options | "calm" | Primary emotion tag for generation |
| `valence` | Float | -1.0 to 1.0 | 0.1 | Emotional valence (negative ↔ positive) |
| `arousal` | Float | 0.0 to 1.0 | 0.2 | Emotional arousal (calm ↔ excited) |
| `intensity` | Float | 0.0 to 1.0 | 0.4 | Emotional intensity (subtle ↔ extreme) |
| `bars` | Integer | 1 to 8 | 1 | Number of bars to generate |
| `enableCloud` | Boolean | true/false | false | Enable cloud-based generation |
| `generationRate` | Integer | 1 to 32 | 8 | Generation frequency (in audio blocks) |

All parameters support DAW automation and can be accessed programmatically:

```cpp
// Get parameter value (thread-safe)
float valence = processor.getParameters()
    .getRawParameterValue(ParameterID::valence)->load();

// Check cloud status
bool cloudEnabled = processor.isCloudEnabled();

// Get generation rate
int rate = processor.getGenerationRate();
```

## Plugin Entry Point

JUCE plugins require a factory function that creates the processor instance:

```cpp
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new kelly::PluginProcessor();
}
```

This function is called by JUCE's plugin wrapper to instantiate the plugin.

## VST3 Wrapper

The project includes VST3 wrapper files in `src/plugin/vst3/` that provide the VST3-specific interface layer. These files are currently disabled (`#if 0`) but provide a template for VST3 integration.

### VST3 Structure

```
VST3 Host
    ↓
JUCE VST3 Wrapper
    ↓
PluginProcessor (AudioProcessor)
    ↓
AI Inference Engines
```

## Real-Time Safety Considerations

### Thread Safety

1. **Audio Thread**: `processBlock()` runs on the audio thread
   - Must be lock-free and deterministic
   - No blocking operations
   - No dynamic memory allocation

2. **Cloud Requests**: Async operations use `std::future`
   ```cpp
   std::future<std::vector<ml::MidiEvent>> cloudFuture_;
   std::atomic<bool> cloudInFlight_{false};
   juce::CriticalSection cloudLock_;
   ```

3. **MIDI Buffer Merging**: Thread-safe MIDI buffer operations
   ```cpp
   {
       const juce::ScopedLock lock(cloudLock_);
       if (cloudMidi_.getNumEvents() > 0) {
           midiMessages.addEvents(cloudMidi_, 0, blockSize_, 0);
           cloudMidi_.clear();
       }
   }
   ```

### Performance Optimizations

1. **ScopedNoDenormals**: Prevents denormal floating-point numbers
   ```cpp
   juce::ScopedNoDenormals noDenormals;
   ```

2. **Cloud Request Throttling**: Prevents excessive cloud API calls
   ```cpp
   if (++cloudCooldown_ > 8) { // throttle requests
       cloudInFlight_.store(true, std::memory_order_release);
       // ...
   }
   ```

3. **Atomic Operations**: Lock-free state updates
   ```cpp
   std::atomic<bool> cloudInFlight_{false};
   ```

## State Management

### State Persistence

The plugin implements state save/restore for DAW project integration:

```cpp
void PluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
    // Save plugin state (emotion parameters, AI config, etc.)
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    // Restore plugin state from saved data
}
```

Currently, these methods are placeholders and should be implemented to save:
- Emotion request parameters
- AI engine configuration
- User preferences

## Integration with Core Systems

### AI Inference Engine

The plugin integrates with the ML inference system:

```cpp
ml::AIInferenceEngine aiEngine_;
ml::AIInferenceEngine cloudEngine_;
ml::EmotionRequest emotionRequest_{"calm", 0.1f, 0.2f, 0.4f, 120, 1};
```

### Emotion Processing

Emotion requests drive MIDI generation:
- Emotion name (e.g., "calm", "excited")
- Valence, arousal, intensity parameters
- Tempo and time signature

## Build Configuration

### CMake Integration

The plugin should be integrated into the CMake build system with:

1. **JUCE Module Configuration**
   - `juce_audio_processors` module
   - `juce_audio_utils` for utilities
   - `juce_gui_basics` for editor UI

2. **Plugin Format Targets**
   - VST3 target
   - AU target (macOS)
   - Standalone application target

3. **Dependencies**
   - AI inference engine library
   - MIDI processing utilities

## Implementation Examples

### Parameter Management with AudioProcessorValueTreeState

The plugin now implements a complete parameter system using `AudioProcessorValueTreeState`:

#### 1. Parameter Layout Definition

```cpp
juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    // Choice parameter for emotion selection
    juce::StringArray emotionTags;
    emotionTags.add("calm");
    emotionTags.add("excited");
    emotionTags.add("sad");
    // ... more emotions
    
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        ParameterID::emotionTag,
        "Emotion",
        emotionTags,
        0  // default: "calm"
    ));
    
    // Float parameter with range and label
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        ParameterID::valence,
        "Valence",
        juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f),
        0.1f,
        juce::AudioParameterFloatAttributes().withLabel("Negative ↔ Positive")
    ));
    
    return { params.begin(), params.end() };
}
```

#### 2. Parameter Listener Implementation

The processor listens to parameter changes and updates internal state:

```cpp
class PluginProcessor : public juce::AudioProcessor,
                         public juce::AudioProcessorValueTreeState::Listener {
    void parameterChanged(const juce::String& parameterID, float newValue) override {
        const juce::ScopedLock lock(emotionRequestLock_);
        
        if (parameterID == ParameterID::valence) {
            emotionRequest_.valence = newValue;
        } else if (parameterID == ParameterID::arousal) {
            emotionRequest_.arousal = newValue;
        }
        // ... handle other parameters
    }
};
```

#### 3. UI Component Binding

The editor binds UI components to parameters using attachments:

```cpp
// In PluginEditor constructor
valenceAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
    processor.getParameters(), ParameterID::valence, valenceSlider_);

cloudAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
    processor.getParameters(), ParameterID::enableCloud, cloudToggle_);
```

### State Persistence

#### Saving Plugin State

```cpp
void PluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
    // Save parameter state using ValueTree
    auto state = parameters_.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}
```

#### Loading Plugin State

```cpp
void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    // Restore parameter state from binary data
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    
    if (xmlState != nullptr) {
        if (xmlState->hasTagName(parameters_.state.getType())) {
            parameters_.replaceState(juce::ValueTree::fromXml(*xmlState));
            
            // Update internal state from restored parameters
            parameterChanged(ParameterID::emotionTag, 0.0f);
            parameterChanged(ParameterID::valence, 0.0f);
            // ... update all parameters
        }
    }
}
```

### Real-Time Parameter Access

Parameters can be accessed safely from the audio thread:

```cpp
void PluginProcessor::processBlock(...) {
    // Get parameter values (thread-safe, lock-free)
    float valence = parameters_.getRawParameterValue(ParameterID::valence)->load();
    bool cloudEnabled = parameters_.getRawParameterValue(ParameterID::enableCloud)->load() > 0.5f;
    
    // Use values in processing
    auto request = getEmotionRequest(); // Uses current parameter values
    auto generated = aiEngine_.generate(request);
}
```

### Error Handling

The implementation includes comprehensive error handling:

```cpp
void PluginProcessor::processBlock(...) {
    try {
        auto generated = aiEngine_.generate(request);
        writeMidiToBuffer(generated, midiMessages, bpm_, sampleRate_, blockSize_);
    } catch (const std::exception& e) {
        // Log error but continue processing (graceful degradation)
        juce::Logger::writeToLog("AI generation error: " + juce::String(e.what()));
    }
}
```

### UI Component Setup

The editor provides a complete UI with sliders, labels, and real-time updates:

```cpp
void PluginEditor::setupSlider(juce::Slider& slider, juce::Label& label,
                               const juce::String& parameterID,
                               const juce::String& labelText,
                               juce::NormalisableRange<float> range) {
    label.setText(labelText, juce::dontSendNotification);
    label.attachToComponent(&slider, true);
    
    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    slider.setRange(range.start, range.end, range.interval);
    
    // Create attachment for automatic parameter binding
    auto attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.getParameters(), parameterID, slider);
    
    // Store attachment to maintain lifetime
    // (stored in member variable based on parameterID)
}
```

### Timer-Based UI Updates

The editor uses a timer to update value labels and status:

```cpp
void PluginEditor::timerCallback() {
    // Update value labels from current parameter values
    auto* valenceParam = processor.getParameters().getRawParameterValue(ParameterID::valence);
    if (valenceParam) {
        valenceValueLabel_.setText(juce::String(valenceParam->load(), 2), juce::dontSendNotification);
    }
    
    // Update status based on processor state
    if (processor.isCloudEnabled()) {
        statusLabel_.setText("Cloud enabled", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::orange);
    } else {
        statusLabel_.setText("Local processing", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    }
}
```

## Future Enhancements

### Advanced UI Components

- Emotion wheel visualization (2D plot of valence/arousal)
- Real-time MIDI visualization (piano roll or note display)
- Preset management (save/load user presets)
- Parameter modulation visualization

### Plugin Formats

- **CLAP**: Modern cross-platform format with better modulation support
- **AAX**: Pro Tools compatibility
- **LV2**: Linux support

### Performance Optimizations

- Parameter smoothing for audio-rate changes
- SIMD-optimized MIDI buffer operations
- Lock-free parameter value caching

## Best Practices

### 1. Real-Time Safety

- ✅ Use lock-free data structures
- ✅ Avoid dynamic allocation in `processBlock()`
- ✅ Keep processing deterministic
- ❌ No blocking I/O in audio thread
- ❌ No mutex locks in audio thread

### 2. Parameter Management

- Use `AudioProcessorValueTreeState` for parameters
- Implement smooth parameter changes
- Support DAW automation

### 3. Error Handling

- Graceful degradation if AI engines fail
- Fallback to local processing
- User-visible error messages

### 4. Testing

- Unit tests for processor logic
- Integration tests with JUCE test framework
- Performance profiling for real-time guarantees

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Overall system architecture
- [PHASE3_DESIGN.md](./PHASE3_DESIGN.md) - Detailed design specifications
- [daw-programs.md](./daw-programs.md) - DAW integration patterns

## References

- [JUCE Documentation](https://juce.com/learn/documentation)
- [JUCE AudioProcessor Tutorial](https://docs.juce.com/master/tutorial_audio_processor.html)
- [VST3 SDK Documentation](https://developer.steinberg.help/display/VST/VST+3+SDK+Documentation)
