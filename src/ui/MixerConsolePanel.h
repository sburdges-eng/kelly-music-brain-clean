#pragma once
/*
 * MixerConsolePanel.h - DAW-Style Mixing Console Interface
 * =========================================================
 *
 * Familiar mixing console layout for audio engineers and DAW users.
 * Provides traditional channel strip controls:
 *
 * - Channel faders (volume)
 * - Pan controls
 * - Mute/Solo buttons
 * - Insert effects slots
 * - Send effects (reverb, delay, etc.)
 * - EQ controls (low, mid, high)
 * - Compression controls
 * - Metering (peak, RMS, VU)
 * - Automation lanes
 * - Routing matrix
 *
 * WORKFLOW:
 * - Multi-track view like Pro Tools, Logic, Ableton
 * - Each MIDI instrument gets a channel strip
 * - Visual feedback familiar to engineers
 * - Plugin-style effect inserts
 * - Bus routing and submixes
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "../music_theory/MusicTheoryBrain.h"
#include <memory>
#include <vector>

namespace midikompanion {

//==============================================================================
// Channel Strip Components
//==============================================================================

/**
 * Single channel strip (like one track in a DAW)
 */
class ChannelStrip : public juce::Component
{
public:
    explicit ChannelStrip(const std::string& channelName);
    ~ChannelStrip() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    //==========================================================================
    // Channel Controls
    //==========================================================================

    void setGain(float gainDb);         // -inf to +12 dB
    void setPan(float pan);              // -1.0 (left) to +1.0 (right)
    void setMute(bool muted);
    void setSolo(bool soloed);
    void setRecordArm(bool armed);

    float getGain() const { return gainDb_; }
    float getPan() const { return pan_; }
    bool isMuted() const { return muted_; }
    bool isSoloed() const { return soloed_; }

    //==========================================================================
    // EQ Controls (3-band)
    //==========================================================================

    struct EQBand {
        float frequency;    // Hz
        float gain;         // dB
        float q;            // Quality factor
    };

    void setLowEQ(float gain);      // Low shelf
    void setMidEQ(float gain);      // Parametric mid
    void setHighEQ(float gain);     // High shelf

    void setLowFreq(float hz);
    void setMidFreq(float hz);
    void setHighFreq(float hz);

    //==========================================================================
    // Dynamics (Compressor)
    //==========================================================================

    void setCompressorThreshold(float db);
    void setCompressorRatio(float ratio);
    void setCompressorAttack(float ms);
    void setCompressorRelease(float ms);
    void setCompressorMakeupGain(float db);

    //==========================================================================
    // Insert Effects
    //==========================================================================

    enum class InsertSlot {
        Slot1,
        Slot2,
        Slot3,
        Slot4
    };

    void addInsertEffect(InsertSlot slot, const std::string& effectName);
    void removeInsertEffect(InsertSlot slot);
    void bypassInsertEffect(InsertSlot slot, bool bypassed);

    //==========================================================================
    // Send Effects
    //==========================================================================

    void setSendLevel(int sendBus, float level);  // 0.0 - 1.0
    void setSendPan(int sendBus, float pan);

    //==========================================================================
    // Metering
    //==========================================================================

    void updateMeter(float peakLevel, float rmsLevel);
    float getPeakLevel() const { return peakLevel_; }
    float getRMSLevel() const { return rmsLevel_; }

    //==========================================================================
    // Callbacks
    //==========================================================================

    std::function<void(float)> onGainChanged;
    std::function<void(float)> onPanChanged;
    std::function<void(bool)> onMuteChanged;
    std::function<void(bool)> onSoloChanged;

private:
    std::string channelName_;

    // Controls
    std::unique_ptr<juce::Slider> gainFader_;
    std::unique_ptr<juce::Slider> panKnob_;
    std::unique_ptr<juce::TextButton> muteButton_;
    std::unique_ptr<juce::TextButton> soloButton_;
    std::unique_ptr<juce::TextButton> recordArmButton_;

    // EQ controls
    std::unique_ptr<juce::Slider> lowEQKnob_;
    std::unique_ptr<juce::Slider> midEQKnob_;
    std::unique_ptr<juce::Slider> highEQKnob_;
    std::unique_ptr<juce::Slider> lowFreqKnob_;
    std::unique_ptr<juce::Slider> midFreqKnob_;
    std::unique_ptr<juce::Slider> highFreqKnob_;

    // Compressor controls
    std::unique_ptr<juce::Slider> compThresholdKnob_;
    std::unique_ptr<juce::Slider> compRatioKnob_;
    std::unique_ptr<juce::Slider> compAttackKnob_;
    std::unique_ptr<juce::Slider> compReleaseKnob_;

    // Insert slots
    std::array<std::unique_ptr<juce::ComboBox>, 4> insertSlots_;

    // Send knobs
    std::vector<std::unique_ptr<juce::Slider>> sendKnobs_;

    // Metering
    std::unique_ptr<juce::Component> meterDisplay_;
    float peakLevel_;
    float rmsLevel_;

    // State
    float gainDb_;
    float pan_;
    bool muted_;
    bool soloed_;
    bool recordArmed_;

    EQBand lowEQ_;
    EQBand midEQ_;
    EQBand highEQ_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChannelStrip)
};

//==============================================================================
// Main Mixer Console Panel
//==============================================================================

class MixerConsolePanel : public juce::Component
{
public:
    MixerConsolePanel();
    ~MixerConsolePanel() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    //==========================================================================
    // Channel Management
    //==========================================================================

    /**
     * Add instrument channel
     */
    int addChannel(const std::string& name, const std::string& instrument);

    /**
     * Remove channel
     */
    void removeChannel(int channelIndex);

    /**
     * Get channel strip
     */
    ChannelStrip* getChannel(int channelIndex);

    /**
     * Get all channels
     */
    std::vector<ChannelStrip*> getAllChannels();

    //==========================================================================
    // Presets (Like DAW Templates)
    //==========================================================================

    struct MixerPreset {
        std::string name;
        std::string description;
        struct ChannelSetup {
            std::string name;
            std::string instrument;
            float gain;
            float pan;
            std::vector<std::string> insertEffects;
        };
        std::vector<ChannelSetup> channels;
    };

    void loadPreset(const MixerPreset& preset);
    void savePreset(const std::string& name);
    std::vector<MixerPreset> getAvailablePresets() const;

    //==========================================================================
    // Common DAW Templates
    //==========================================================================

    void loadRockBandTemplate();      // Drums, bass, guitars, vocals
    void loadOrchestralTemplate();    // Strings, brass, woodwinds, perc
    void loadElectronicTemplate();    // Synths, drums, FX
    void loadJazzComboTemplate();     // Piano, bass, drums, horns
    void loadSongwriterTemplate();    // Vocals, guitar, piano

    //==========================================================================
    // Master Section
    //==========================================================================

    void setMasterGain(float gainDb);
    void setMasterLimiter(bool enabled, float threshold);
    float getMasterGain() const { return masterGainDb_; }

    //==========================================================================
    // Effect Bus (Send/Return)
    //==========================================================================

    int addEffectBus(const std::string& name, const std::string& effectType);
    void setEffectBusLevel(int busIndex, float level);
    void setEffectBusParameters(int busIndex, const std::string& paramName,
                               float value);

    //==========================================================================
    // Routing Matrix
    //==========================================================================

    void routeChannelToOutput(int channelIndex, int outputBus);
    void createSubmix(const std::vector<int>& channelIndices,
                     const std::string& submixName);

    //==========================================================================
    // Automation
    //==========================================================================

    enum class AutomationMode {
        Off,
        Read,
        Touch,
        Latch,
        Write
    };

    void setChannelAutomationMode(int channelIndex, AutomationMode mode);
    void recordAutomation(int channelIndex, const std::string& parameter,
                         float value, double timestamp);

    struct AutomationPoint {
        double timestamp;
        float value;
        std::string parameter;  // "gain", "pan", "send1", etc.
    };

    std::vector<AutomationPoint> getChannelAutomation(int channelIndex) const;

    //==========================================================================
    // View Options
    //==========================================================================

    enum class ViewMode {
        MixerView,           // Traditional horizontal mixer
        TrackView,           // Vertical track list (like DAW)
        CompactView,         // Smaller channel strips
        FullView             // All controls visible
    };

    void setViewMode(ViewMode mode);
    void setShowEQ(bool show);
    void setShowCompressor(bool show);
    void setShowInserts(bool show);
    void setShowSends(bool show);
    void setShowMeters(bool show);

    //==========================================================================
    // MIDI Integration
    //==========================================================================

    /**
     * Route MIDI to specific channel
     */
    void routeMIDIToChannel(int channelIndex, const juce::MidiBuffer& midi);

    /**
     * Get mixed MIDI output
     */
    juce::MidiBuffer getMixedOutput() const;

    /**
     * Apply channel settings to MIDI (velocity, pan, effects)
     */
    void applyMixToMIDI(juce::MidiBuffer& buffer, int channelIndex);

    //==========================================================================
    // Snapshots (Like DAW Mixer Snapshots)
    //==========================================================================

    struct MixerSnapshot {
        std::string name;
        struct ChannelState {
            float gain;
            float pan;
            bool muted;
            bool soloed;
            std::map<std::string, float> parameters;
        };
        std::map<int, ChannelState> channelStates;
        double timestamp;
    };

    void saveSnapshot(const std::string& name);
    void loadSnapshot(const std::string& name);
    std::vector<MixerSnapshot> getSnapshots() const;

    //==========================================================================
    // Export
    //==========================================================================

    /**
     * Export mixer state as session file
     */
    bool exportSession(const juce::File& outputFile);

    /**
     * Import mixer state from session file
     */
    bool importSession(const juce::File& inputFile);

private:
    //==========================================================================
    // UI Components
    //==========================================================================

    // Channel strips container
    std::unique_ptr<juce::Viewport> channelViewport_;
    std::unique_ptr<juce::Component> channelContainer_;

    // Master section
    std::unique_ptr<ChannelStrip> masterChannel_;

    // Effect buses
    std::vector<std::unique_ptr<ChannelStrip>> effectBuses_;

    // Transport controls
    std::unique_ptr<juce::TextButton> playButton_;
    std::unique_ptr<juce::TextButton> stopButton_;
    std::unique_ptr<juce::TextButton> recordButton_;

    // View controls
    std::unique_ptr<juce::ComboBox> viewModeSelector_;
    std::unique_ptr<juce::TextButton> showEQButton_;
    std::unique_ptr<juce::TextButton> showCompButton_;
    std::unique_ptr<juce::TextButton> showInsertsButton_;
    std::unique_ptr<juce::TextButton> showSendsButton_;

    // Preset selector
    std::unique_ptr<juce::ComboBox> presetSelector_;
    std::unique_ptr<juce::TextButton> loadPresetButton_;
    std::unique_ptr<juce::TextButton> savePresetButton_;

    //==========================================================================
    // Data
    //==========================================================================

    std::vector<std::unique_ptr<ChannelStrip>> channels_;
    std::map<int, std::vector<AutomationPoint>> automation_;
    std::vector<MixerSnapshot> snapshots_;
    std::vector<MixerPreset> presets_;

    ViewMode viewMode_;
    float masterGainDb_;
    bool showEQ_;
    bool showCompressor_;
    bool showInserts_;
    bool showSends_;
    bool showMeters_;

    //==========================================================================
    // Music Theory Integration
    //==========================================================================

    std::unique_ptr<theory::MusicTheoryBrain> theoryBrain_;

    //==========================================================================
    // Template Initialization
    //==========================================================================

    void initializePresets();
    void createRockBandPreset();
    void createOrchestralPreset();
    void createElectronicPreset();
    void createJazzComboPreset();

    //==========================================================================
    // Layout Helpers
    //==========================================================================

    void layoutMixerView();
    void layoutTrackView();
    void layoutCompactView();

    //==========================================================================
    // Callbacks
    //==========================================================================

    void onChannelGainChanged(int channelIndex, float gain);
    void onChannelPanChanged(int channelIndex, float pan);
    void onChannelMuteChanged(int channelIndex, bool muted);
    void onChannelSoloChanged(int channelIndex, bool soloed);

    void onPresetSelected(int presetIndex);
    void onViewModeChanged();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MixerConsolePanel)
};

//==============================================================================
// Timeline View (DAW-Style Arrange Window)
//==============================================================================

class TimelinePanel : public juce::Component
{
public:
    TimelinePanel();
    ~TimelinePanel() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;

    //==========================================================================
    // Track Management
    //==========================================================================

    struct Track {
        std::string name;
        std::string instrument;
        juce::Colour color;
        std::vector<juce::MidiMessage> midiData;
        bool muted;
        bool soloed;
        int channelIndex;  // Links to mixer channel
    };

    int addTrack(const std::string& name, const std::string& instrument);
    void removeTrack(int trackIndex);
    Track* getTrack(int trackIndex);

    //==========================================================================
    // Region Management (Like DAW Clips/Regions)
    //==========================================================================

    struct Region {
        int trackIndex;
        double startTime;
        double duration;
        std::string name;
        juce::MidiBuffer midiData;
        juce::Colour color;
    };

    int addRegion(int trackIndex, double startTime, const juce::MidiBuffer& midi);
    void moveRegion(int regionIndex, double newStartTime);
    void resizeRegion(int regionIndex, double newDuration);
    void deleteRegion(int regionIndex);

    //==========================================================================
    // Playback
    //==========================================================================

    void setPlayheadPosition(double seconds);
    double getPlayheadPosition() const { return playheadPosition_; }

    void setLoop(bool enabled, double start, double end);

    //==========================================================================
    // Display
    //==========================================================================

    void setTimelineZoom(float zoom);
    void setVerticalZoom(float zoom);
    void snapToGrid(bool enable);
    void setGridResolution(double beats);

private:
    std::vector<Track> tracks_;
    std::vector<Region> regions_;

    double playheadPosition_;
    bool loopEnabled_;
    double loopStart_;
    double loopEnd_;
    float timelineZoom_;
    float verticalZoom_;
    bool snapToGrid_;
    double gridResolution_;

    void drawTimeline(juce::Graphics& g, juce::Rectangle<int> area);
    void drawTracks(juce::Graphics& g, juce::Rectangle<int> area);
    void drawRegions(juce::Graphics& g, juce::Rectangle<int> area);
    void drawPlayhead(juce::Graphics& g, juce::Rectangle<int> area);
    void drawGrid(juce::Graphics& g, juce::Rectangle<int> area);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TimelinePanel)
};

} // namespace midikompanion
