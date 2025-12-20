#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "../ml/ai_inference.h"
<<<<<<< Current (Your changes)
<<<<<<< Current (Your changes)
#include "../ml/NodeMLMapper.h"
=======
>>>>>>> Incoming (Background Agent changes)
=======
>>>>>>> Incoming (Background Agent changes)
#include "../project/ProjectManager.h"
#include "../midi/MidiExporter.h"
#include <atomic>
#include <future>

namespace kelly {

/**
 * @brief Parameter IDs for the plugin
 */
namespace ParameterID {
    constexpr const char* emotionTag = "emotionTag";
    constexpr const char* valence = "valence";
    constexpr const char* arousal = "arousal";
    constexpr const char* intensity = "intensity";
    constexpr const char* bars = "bars";
    constexpr const char* enableCloud = "enableCloud";
    constexpr const char* generationRate = "generationRate";
}

/**
 * @brief Main audio processor for Kelly Emotion Processor plugin
 * 
 * This processor generates MIDI based on emotional parameters using AI inference.
 * It supports both on-device (ONNX) and cloud-based generation.
 */
class PluginProcessor : public juce::AudioProcessor,
                         public juce::AudioProcessorValueTreeState::Listener {
public:
    PluginProcessor();
    ~PluginProcessor() override = default;

    //==============================================================================
    // AudioProcessor Interface
    //==============================================================================
    
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Kelly Emotion Processor"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return true; }
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    //==============================================================================
    // Parameter Access
    //==============================================================================
    
    /**
     * @brief Get the parameter tree state for UI binding
     */
    juce::AudioProcessorValueTreeState& getParameters() { return parameters_; }
    const juce::AudioProcessorValueTreeState& getParameters() const { return parameters_; }

    /**
     * @brief Get current emotion request based on parameters
     */
    ml::EmotionRequest getEmotionRequest() const;

    /**
     * @brief Check if cloud generation is enabled
     */
    bool isCloudEnabled() const;

    /**
     * @brief Get generation rate (how often to generate, in blocks)
     */
    int getGenerationRate() const;
    
    /**
     * @brief Enable/disable ML-enhanced generation
     */
    void setMLEnabled(bool enabled);
    
    /**
     * @brief Check if ML mode is enabled
     */
    bool isMLEnabled() const;
    
    /**
     * @brief Get the NodeMLMapper instance
     */
    midikompanion::ml::NodeMLMapper& getNodeMapper() { return nodeMapper_; }

    //==============================================================================
    // Project Management
    //==============================================================================
    
    /**
     * @brief Save current project to file
     * @param file Target file
     * @return True if successful
     */
    bool saveCurrentProject(const juce::File& file);
    
    /**
     * @brief Load project from file
     * @param file Source file
     * @return True if successful
     */
    bool loadProject(const juce::File& file);
    
    /**
     * @brief Create new empty project
     */
    void createNewProject();
    
    /**
     * @brief Get project manager reference
     */
    midikompanion::ProjectManager& getProjectManager() { return projectManager_; }
    
    /**
     * @brief Get current project data
     */
    midikompanion::ProjectData getCurrentProjectData() const;
    
    /**
     * @brief Check if project has unsaved changes
     */
    bool hasUnsavedChanges() const { return projectModified_; }
    
    /**
     * @brief Get current project file
     */
    juce::File getCurrentProjectFile() const { return currentProjectFile_; }
    
    //==============================================================================
    // MIDI Export
    //==============================================================================
    
    /**
     * @brief Export current session to MIDI file
     * @param file Target file
     * @param options Export options
     * @return Export result
     */
    midikompanion::MidiExportResult exportToMidi(const juce::File& file,
                                                 const midikompanion::MidiExportOptions& options = {});
    
    /**
     * @brief Get MIDI exporter reference
     */
    midikompanion::MidiExporter& getMidiExporter() { return midiExporter_; }
    
    /**
     * @brief Check if there is MIDI data to export
     */
    bool hasMidiData() const { return !generatedMidi_.isEmpty(); }
    
    /**
     * @brief Add generated MIDI events to internal storage
     */
    void addGeneratedMidi(const std::vector<ml::MidiEvent>& events);
    
    /**
     * @brief Clear generated MIDI data
     */
    void clearGeneratedMidi() { generatedMidi_.clear(); }
    
    /**
     * @brief Get current tempo
     */
    double getCurrentTempo() const { return bpm_; }

    //==============================================================================
    // Project Management
    //==============================================================================
    
    /**
     * @brief Save current project to file
     * @param file Target file
     * @return True if successful
     */
    bool saveCurrentProject(const juce::File& file);
    
    /**
     * @brief Load project from file
     * @param file Source file
     * @return True if successful
     */
    bool loadProject(const juce::File& file);
    
    /**
     * @brief Create new empty project
     */
    void createNewProject();
    
    /**
     * @brief Get project manager reference
     */
    midikompanion::ProjectManager& getProjectManager() { return projectManager_; }
    
    /**
     * @brief Get current project data
     */
    midikompanion::ProjectData getCurrentProjectData() const;
    
    /**
     * @brief Check if project has unsaved changes
     */
    bool hasUnsavedChanges() const { return projectModified_; }
    
    /**
     * @brief Get current project file
     */
    juce::File getCurrentProjectFile() const { return currentProjectFile_; }
    
    //==============================================================================
    // MIDI Export
    //==============================================================================
    
    /**
     * @brief Export current session to MIDI file
     * @param file Target file
     * @param options Export options
     * @return Export result
     */
    midikompanion::MidiExportResult exportToMidi(const juce::File& file,
                                                 const midikompanion::MidiExportOptions& options = {});
    
    /**
     * @brief Get MIDI exporter reference
     */
    midikompanion::MidiExporter& getMidiExporter() { return midiExporter_; }
    
    /**
     * @brief Check if there is MIDI data to export
     */
    bool hasMidiData() const { return !generatedMidi_.isEmpty(); }
    
    /**
     * @brief Add generated MIDI events to internal storage
     */
    void addGeneratedMidi(const std::vector<ml::MidiEvent>& events);
    
    /**
     * @brief Clear generated MIDI data
     */
    void clearGeneratedMidi() { generatedMidi_.clear(); }
    
    /**
     * @brief Get current tempo
     */
    double getCurrentTempo() const { return bpm_; }

    //==============================================================================
    // Project Management
    //==============================================================================
    
    /**
     * @brief Save current project to file
     * @param file Target file
     * @return True if successful
     */
    bool saveCurrentProject(const juce::File& file);
    
    /**
     * @brief Load project from file
     * @param file Source file
     * @return True if successful
     */
    bool loadProject(const juce::File& file);
    
    /**
     * @brief Create new empty project
     */
    void createNewProject();
    
    /**
     * @brief Get project manager reference
     */
    midikompanion::ProjectManager& getProjectManager() { return projectManager_; }
    
    /**
     * @brief Get current project data
     */
    midikompanion::ProjectData getCurrentProjectData() const;
    
    /**
     * @brief Check if project has unsaved changes
     */
    bool hasUnsavedChanges() const { return projectModified_; }
    
    /**
     * @brief Get current project file
     */
    juce::File getCurrentProjectFile() const { return currentProjectFile_; }
    
    //==============================================================================
    // MIDI Export
    //==============================================================================
    
    /**
     * @brief Export current session to MIDI file
     * @param file Target file
     * @param options Export options
     * @return Export result
     */
    midikompanion::MidiExportResult exportToMidi(const juce::File& file,
                                                 const midikompanion::MidiExportOptions& options = {});
    
    /**
     * @brief Get MIDI exporter reference
     */
    midikompanion::MidiExporter& getMidiExporter() { return midiExporter_; }
    
    /**
     * @brief Check if there is MIDI data to export
     */
    bool hasMidiData() const { return !generatedMidi_.isEmpty(); }
    
    /**
     * @brief Add generated MIDI events to internal storage
     */
    void addGeneratedMidi(const std::vector<ml::MidiEvent>& events);
    
    /**
     * @brief Clear generated MIDI data
     */
    void clearGeneratedMidi() { generatedMidi_.clear(); }
    
    /**
     * @brief Get current tempo
     */
    double getCurrentTempo() const { return bpm_; }

private:
    //==============================================================================
    // Parameter Tree Setup
    //==============================================================================
    
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    
    //==============================================================================
    // Parameter Change Callback (AudioProcessorValueTreeState::Listener)
    //==============================================================================
    
    void parameterChanged(const juce::String& parameterID, float newValue) override;

    //==============================================================================
    // Initialization
    //==============================================================================
    
    /**
     * @brief Initialize emotion thesaurus from JSON file
     */
    void initializeEmotionThesaurus();

    //==============================================================================
    // Member Variables
    //==============================================================================
    
    juce::AudioProcessorValueTreeState parameters_;
    
    double bpm_{120.0};
    double sampleRate_{44100.0};
    int blockSize_{512};

    ml::AIInferenceEngine aiEngine_;
    ml::AIInferenceEngine cloudEngine_;
    
    // ML-Node mapping system
    midikompanion::ml::NodeMLMapper nodeMapper_;
    std::atomic<bool> mlEnabled_{false};
    
    // Cached emotion request (updated from parameters)
    mutable ml::EmotionRequest emotionRequest_{"calm", 0.1f, 0.2f, 0.4f, 120, 1};
    mutable juce::CriticalSection emotionRequestLock_;

    std::future<std::vector<ml::MidiEvent>> cloudFuture_;
    std::atomic<bool> cloudInFlight_{false};
    std::atomic<int> cloudCooldown_{0};
    std::atomic<int> generationCounter_{0};
    juce::MidiBuffer cloudMidi_;
    juce::CriticalSection cloudLock_;

    // Project management
    midikompanion::ProjectManager projectManager_;
    midikompanion::MidiExporter midiExporter_;
    midikompanion::ProjectData currentProjectData_;
    juce::File currentProjectFile_;
    bool projectModified_{false};
    
    // Generated MIDI storage
    juce::MidiMessageSequence generatedMidi_;
    std::vector<midikompanion::TrackData> generatedTracks_;
    mutable juce::CriticalSection midiStorageLock_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

} // namespace kelly
