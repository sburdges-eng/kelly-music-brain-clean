#include "MixerConsolePanel.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_graphics/juce_graphics.h>
#include <algorithm>
#include <cmath>

namespace midikompanion {

namespace {
int slotToIndex(ChannelStrip::InsertSlot slot) {
    switch (slot) {
        case ChannelStrip::InsertSlot::Slot1: return 0;
        case ChannelStrip::InsertSlot::Slot2: return 1;
        case ChannelStrip::InsertSlot::Slot3: return 2;
        case ChannelStrip::InsertSlot::Slot4: return 3;
        default: return 0;
    }
}

float dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}
} // namespace

//==============================================================================
// ChannelStrip
//==============================================================================

ChannelStrip::ChannelStrip(const std::string& channelName)
    : channelName_(channelName),
      peakLevel_(0.0f),
      rmsLevel_(0.0f),
      gainDb_(0.0f),
      pan_(0.0f),
      muted_(false),
      soloed_(false),
      recordArmed_(false) {
    // Gain fader
    gainFader_ = std::make_unique<juce::Slider>();
    gainFader_->setSliderStyle(juce::Slider::LinearVertical);
    gainFader_->setRange(-60.0, 12.0, 0.1);
    gainFader_->setValue(0.0);
    gainFader_->onValueChange = [this] {
        gainDb_ = static_cast<float>(gainFader_->getValue());
        if (onGainChanged) onGainChanged(gainDb_);
    };
    addAndMakeVisible(*gainFader_);

    // Pan knob
    panKnob_ = std::make_unique<juce::Slider>();
    panKnob_->setSliderStyle(juce::Slider::Rotary);
    panKnob_->setRange(-1.0, 1.0, 0.01);
    panKnob_->setValue(0.0);
    panKnob_->onValueChange = [this] {
        pan_ = static_cast<float>(panKnob_->getValue());
        if (onPanChanged) onPanChanged(pan_);
    };
    addAndMakeVisible(*panKnob_);

    // Buttons
    muteButton_ = std::make_unique<juce::TextButton>("M");
    muteButton_->setClickingTogglesState(true);
    muteButton_->onClick = [this] {
        muted_ = muteButton_->getToggleState();
        if (onMuteChanged) onMuteChanged(muted_);
    };
    addAndMakeVisible(*muteButton_);

    soloButton_ = std::make_unique<juce::TextButton>("S");
    soloButton_->setClickingTogglesState(true);
    soloButton_->onClick = [this] {
        soloed_ = soloButton_->getToggleState();
        if (onSoloChanged) onSoloChanged(soloed_);
    };
    addAndMakeVisible(*soloButton_);

    recordArmButton_ = std::make_unique<juce::TextButton>("R");
    recordArmButton_->setClickingTogglesState(true);
    recordArmButton_->onClick = [this] { recordArmed_ = recordArmButton_->getToggleState(); };
    addAndMakeVisible(*recordArmButton_);

    // EQ controls
    auto makeKnob = [this](std::unique_ptr<juce::Slider>& knob, float min, float max, float init) {
        knob = std::make_unique<juce::Slider>();
        knob->setSliderStyle(juce::Slider::Rotary);
        knob->setRange(min, max, 0.1);
        knob->setValue(init);
        addAndMakeVisible(*knob);
    };
    makeKnob(lowEQKnob_, -12.0f, 12.0f, 0.0f);
    makeKnob(midEQKnob_, -12.0f, 12.0f, 0.0f);
    makeKnob(highEQKnob_, -12.0f, 12.0f, 0.0f);

    makeKnob(lowFreqKnob_, 40.0f, 200.0f, 100.0f);
    makeKnob(midFreqKnob_, 200.0f, 4000.0f, 1000.0f);
    makeKnob(highFreqKnob_, 4000.0f, 12000.0f, 8000.0f);

    // Compressor controls
    makeKnob(compThresholdKnob_, -60.0f, 0.0f, -18.0f);
    makeKnob(compRatioKnob_, 1.0f, 20.0f, 2.0f);
    makeKnob(compAttackKnob_, 0.1f, 100.0f, 10.0f);
    makeKnob(compReleaseKnob_, 10.0f, 500.0f, 120.0f);

    // Insert slots
    for (auto& slot : insertSlots_) {
        slot = std::make_unique<juce::ComboBox>();
        slot->addItem("None", 1);
        addAndMakeVisible(*slot);
    }

    // Send knobs (two by default)
    for (int i = 0; i < 2; ++i) {
        auto send = std::make_unique<juce::Slider>();
        send->setSliderStyle(juce::Slider::Rotary);
        send->setRange(0.0, 1.0, 0.01);
        send->setValue(0.0);
        sendKnobs_.push_back(std::move(send));
        addAndMakeVisible(*sendKnobs_.back());
    }

    meterDisplay_ = std::make_unique<juce::Component>();
    addAndMakeVisible(*meterDisplay_);
}

void ChannelStrip::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff222222));
    g.setColour(juce::Colours::white);
    g.drawText(channelName_, 0, 0, getWidth(), 20, juce::Justification::centred);

    // Meter
    const int meterHeight = 80;
    const int meterWidth = 12;
    int meterX = getWidth() - meterWidth - 6;
    int meterY = getHeight() - meterHeight - 6;
    g.setColour(juce::Colours::darkgrey);
    g.fillRect(meterX, meterY, meterWidth, meterHeight);

    float clampedPeak = std::clamp(peakLevel_, 0.0f, 1.0f);
    int peakFill = static_cast<int>(clampedPeak * meterHeight);
    g.setColour(juce::Colours::chartreuse);
    g.fillRect(meterX, meterY + (meterHeight - peakFill), meterWidth, peakFill);

    float clampedRms = std::clamp(rmsLevel_, 0.0f, 1.0f);
    int rmsFill = static_cast<int>(clampedRms * meterHeight);
    g.setColour(juce::Colours::orange);
    g.fillRect(meterX - 6, meterY + (meterHeight - rmsFill), meterWidth / 2, rmsFill);
}

void ChannelStrip::resized() {
    auto bounds = getLocalBounds().reduced(4);
    bounds.removeFromTop(20); // title

    gainFader_->setBounds(bounds.removeFromLeft(40));
    panKnob_->setBounds(bounds.removeFromTop(40).withWidth(60));

    auto btnRow = bounds.removeFromTop(24);
    muteButton_->setBounds(btnRow.removeFromLeft(24));
    soloButton_->setBounds(btnRow.removeFromLeft(24));
    recordArmButton_->setBounds(btnRow.removeFromLeft(24));

    auto eqRow1 = bounds.removeFromTop(40);
    lowEQKnob_->setBounds(eqRow1.removeFromLeft(40));
    midEQKnob_->setBounds(eqRow1.removeFromLeft(40));
    highEQKnob_->setBounds(eqRow1.removeFromLeft(40));

    auto eqRow2 = bounds.removeFromTop(40);
    lowFreqKnob_->setBounds(eqRow2.removeFromLeft(40));
    midFreqKnob_->setBounds(eqRow2.removeFromLeft(40));
    highFreqKnob_->setBounds(eqRow2.removeFromLeft(40));

    auto compRow = bounds.removeFromTop(40);
    compThresholdKnob_->setBounds(compRow.removeFromLeft(40));
    compRatioKnob_->setBounds(compRow.removeFromLeft(40));
    compAttackKnob_->setBounds(compRow.removeFromLeft(40));
    compReleaseKnob_->setBounds(compRow.removeFromLeft(40));

    auto insertRow = bounds.removeFromTop(24);
    for (auto& slot : insertSlots_) {
        slot->setBounds(insertRow.removeFromLeft(60));
    }

    auto sendRow = bounds.removeFromTop(40);
    for (auto& send : sendKnobs_) {
        send->setBounds(sendRow.removeFromLeft(40));
    }
}

void ChannelStrip::setGain(float gainDb) {
    gainDb_ = gainDb;
    if (gainFader_) gainFader_->setValue(gainDb, juce::dontSendNotification);
}

void ChannelStrip::setPan(float pan) {
    pan_ = juce::jlimit(-1.0f, 1.0f, pan);
    if (panKnob_) panKnob_->setValue(pan_, juce::dontSendNotification);
}

void ChannelStrip::setMute(bool muted) {
    muted_ = muted;
    if (muteButton_) muteButton_->setToggleState(muted, juce::dontSendNotification);
}

void ChannelStrip::setSolo(bool soloed) {
    soloed_ = soloed;
    if (soloButton_) soloButton_->setToggleState(soloed, juce::dontSendNotification);
}

void ChannelStrip::setRecordArm(bool armed) {
    recordArmed_ = armed;
    if (recordArmButton_) recordArmButton_->setToggleState(armed, juce::dontSendNotification);
}

void ChannelStrip::setLowEQ(float gain) { lowEQ_.gain = gain; if (lowEQKnob_) lowEQKnob_->setValue(gain, juce::dontSendNotification); }
void ChannelStrip::setMidEQ(float gain) { midEQ_.gain = gain; if (midEQKnob_) midEQKnob_->setValue(gain, juce::dontSendNotification); }
void ChannelStrip::setHighEQ(float gain) { highEQ_.gain = gain; if (highEQKnob_) highEQKnob_->setValue(gain, juce::dontSendNotification); }

void ChannelStrip::setLowFreq(float hz) { lowEQ_.frequency = hz; if (lowFreqKnob_) lowFreqKnob_->setValue(hz, juce::dontSendNotification); }
void ChannelStrip::setMidFreq(float hz) { midEQ_.frequency = hz; if (midFreqKnob_) midFreqKnob_->setValue(hz, juce::dontSendNotification); }
void ChannelStrip::setHighFreq(float hz) { highEQ_.frequency = hz; if (highFreqKnob_) highFreqKnob_->setValue(hz, juce::dontSendNotification); }

void ChannelStrip::setCompressorThreshold(float db) { if (compThresholdKnob_) compThresholdKnob_->setValue(db, juce::dontSendNotification); }
void ChannelStrip::setCompressorRatio(float ratio) { if (compRatioKnob_) compRatioKnob_->setValue(ratio, juce::dontSendNotification); }
void ChannelStrip::setCompressorAttack(float ms) { if (compAttackKnob_) compAttackKnob_->setValue(ms, juce::dontSendNotification); }
void ChannelStrip::setCompressorRelease(float ms) { if (compReleaseKnob_) compReleaseKnob_->setValue(ms, juce::dontSendNotification); }
void ChannelStrip::setCompressorMakeupGain(float db) { gainDb_ += db; setGain(gainDb_); }

std::vector<std::string> ChannelStrip::getInsertEffectNames() const {
    std::vector<std::string> names;
    for (const auto& slot : insertSlots_) {
        names.push_back(slot ? slot->getText().toStdString() : "None");
    }
    return names;
}

void ChannelStrip::setEQVisible(bool show) {
    if (lowEQKnob_) lowEQKnob_->setVisible(show);
    if (midEQKnob_) midEQKnob_->setVisible(show);
    if (highEQKnob_) highEQKnob_->setVisible(show);
    if (lowFreqKnob_) lowFreqKnob_->setVisible(show);
    if (midFreqKnob_) midFreqKnob_->setVisible(show);
    if (highFreqKnob_) highFreqKnob_->setVisible(show);
}

void ChannelStrip::setCompressorVisible(bool show) {
    if (compThresholdKnob_) compThresholdKnob_->setVisible(show);
    if (compRatioKnob_) compRatioKnob_->setVisible(show);
    if (compAttackKnob_) compAttackKnob_->setVisible(show);
    if (compReleaseKnob_) compReleaseKnob_->setVisible(show);
}

void ChannelStrip::setInsertVisible(bool show) {
    for (auto& slot : insertSlots_) {
        if (slot) slot->setVisible(show);
    }
}

void ChannelStrip::setSendsVisible(bool show) {
    for (auto& send : sendKnobs_) {
        if (send) send->setVisible(show);
    }
}

void ChannelStrip::setMetersVisible(bool show) {
    if (meterDisplay_) meterDisplay_->setVisible(show);
}

void ChannelStrip::addInsertEffect(InsertSlot slot, const std::string& effectName) {
    const int idx = slotToIndex(slot);
    if (idx >= 0 && idx < static_cast<int>(insertSlots_.size()) && insertSlots_[idx]) {
        insertSlots_[idx]->setText(effectName, juce::dontSendNotification);
    }
}

void ChannelStrip::removeInsertEffect(InsertSlot slot) {
    const int idx = slotToIndex(slot);
    if (idx >= 0 && idx < static_cast<int>(insertSlots_.size()) && insertSlots_[idx]) {
        insertSlots_[idx]->setText("None", juce::dontSendNotification);
    }
}

void ChannelStrip::bypassInsertEffect(InsertSlot slot, bool bypassed) {
    const int idx = slotToIndex(slot);
    if (idx >= 0 && idx < static_cast<int>(insertSlots_.size()) && insertSlots_[idx]) {
        insertSlots_[idx]->setEnabled(!bypassed);
    }
}

void ChannelStrip::setSendLevel(int sendBus, float level) {
    if (sendBus >= static_cast<int>(sendKnobs_.size())) {
        sendKnobs_.resize(static_cast<size_t>(sendBus) + 1);
        for (auto& send : sendKnobs_) {
            if (!send) {
                send = std::make_unique<juce::Slider>();
                send->setSliderStyle(juce::Slider::Rotary);
                send->setRange(0.0, 1.0, 0.01);
                addAndMakeVisible(*send);
            }
        }
    }
    if (sendKnobs_[sendBus]) {
        sendKnobs_[sendBus]->setValue(level, juce::dontSendNotification);
    }
}

void ChannelStrip::setSendPan(int /*sendBus*/, float pan) {
    pan_ = pan;
}

void ChannelStrip::updateMeter(float peakLevel, float rmsLevel) {
    peakLevel_ = peakLevel;
    rmsLevel_ = rmsLevel;
    repaint();
}

//==============================================================================
// MixerConsolePanel
//==============================================================================

MixerConsolePanel::MixerConsolePanel()
    : viewMode_(ViewMode::MixerView),
      masterGainDb_(0.0f),
      showEQ_(true),
      showCompressor_(true),
      showInserts_(true),
      showSends_(true),
      showMeters_(true) {
    channelViewport_ = std::make_unique<juce::Viewport>();
    channelContainer_ = std::make_unique<juce::Component>();
    channelViewport_->setViewedComponent(channelContainer_.get(), false);
    addAndMakeVisible(*channelViewport_);

    masterChannel_ = std::make_unique<ChannelStrip>("Master");
    addAndMakeVisible(*masterChannel_);

    playButton_ = std::make_unique<juce::TextButton>("Play");
    stopButton_ = std::make_unique<juce::TextButton>("Stop");
    recordButton_ = std::make_unique<juce::TextButton>("Record");
    addAndMakeVisible(*playButton_);
    addAndMakeVisible(*stopButton_);
    addAndMakeVisible(*recordButton_);

    viewModeSelector_ = std::make_unique<juce::ComboBox>();
    viewModeSelector_->addItem("Mixer", 1);
    viewModeSelector_->addItem("Tracks", 2);
    viewModeSelector_->addItem("Compact", 3);
    viewModeSelector_->addItem("Full", 4);
    viewModeSelector_->setSelectedId(1);
    viewModeSelector_->onChange = [this] { onViewModeChanged(); };
    addAndMakeVisible(*viewModeSelector_);

    showEQButton_ = std::make_unique<juce::TextButton>("EQ");
    showEQButton_->setClickingTogglesState(true);
    showEQButton_->setToggleState(true, juce::dontSendNotification);
    showEQButton_->onClick = [this] { setShowEQ(showEQButton_->getToggleState()); };
    addAndMakeVisible(*showEQButton_);

    showCompButton_ = std::make_unique<juce::TextButton>("Comp");
    showCompButton_->setClickingTogglesState(true);
    showCompButton_->setToggleState(true, juce::dontSendNotification);
    showCompButton_->onClick = [this] { setShowCompressor(showCompButton_->getToggleState()); };
    addAndMakeVisible(*showCompButton_);

    showInsertsButton_ = std::make_unique<juce::TextButton>("Inserts");
    showInsertsButton_->setClickingTogglesState(true);
    showInsertsButton_->setToggleState(true, juce::dontSendNotification);
    showInsertsButton_->onClick = [this] { setShowInserts(showInsertsButton_->getToggleState()); };
    addAndMakeVisible(*showInsertsButton_);

    showSendsButton_ = std::make_unique<juce::TextButton>("Sends");
    showSendsButton_->setClickingTogglesState(true);
    showSendsButton_->setToggleState(true, juce::dontSendNotification);
    showSendsButton_->onClick = [this] { setShowSends(showSendsButton_->getToggleState()); };
    addAndMakeVisible(*showSendsButton_);

    presetSelector_ = std::make_unique<juce::ComboBox>();
    presetSelector_->onChange = [this] { onPresetSelected(presetSelector_->getSelectedId() - 1); };
    addAndMakeVisible(*presetSelector_);

    loadPresetButton_ = std::make_unique<juce::TextButton>("Load");
    loadPresetButton_->onClick = [this] { onPresetSelected(presetSelector_->getSelectedId() - 1); };
    addAndMakeVisible(*loadPresetButton_);

    savePresetButton_ = std::make_unique<juce::TextButton>("Save");
    savePresetButton_->onClick = [this] { savePreset("User Preset"); };
    addAndMakeVisible(*savePresetButton_);

    initializePresets();
}

void MixerConsolePanel::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff181818));
}

void MixerConsolePanel::resized() {
    auto bounds = getLocalBounds().reduced(6);
    auto topRow = bounds.removeFromTop(40);
    playButton_->setBounds(topRow.removeFromLeft(60));
    stopButton_->setBounds(topRow.removeFromLeft(60));
    recordButton_->setBounds(topRow.removeFromLeft(60));
    viewModeSelector_->setBounds(topRow.removeFromLeft(120));
    presetSelector_->setBounds(topRow.removeFromLeft(160));
    loadPresetButton_->setBounds(topRow.removeFromLeft(70));
    savePresetButton_->setBounds(topRow.removeFromLeft(70));

    showEQButton_->setBounds(topRow.removeFromLeft(60));
    showCompButton_->setBounds(topRow.removeFromLeft(60));
    showInsertsButton_->setBounds(topRow.removeFromLeft(80));
    showSendsButton_->setBounds(topRow.removeFromLeft(70));

    auto masterBounds = bounds.removeFromRight(140);
    masterChannel_->setBounds(masterBounds);

    channelViewport_->setBounds(bounds);
    layoutMixerView();
}

int MixerConsolePanel::addChannel(const std::string& name, const std::string& /*instrument*/) {
    auto channel = std::make_unique<ChannelStrip>(name);
    auto* ptr = channel.get();
    channels_.push_back(std::move(channel));
    channelContainer_->addAndMakeVisible(ptr);
    resized();
    return static_cast<int>(channels_.size()) - 1;
}

void MixerConsolePanel::removeChannel(int channelIndex) {
    if (channelIndex < 0 || channelIndex >= static_cast<int>(channels_.size())) {
        return;
    }
    channelContainer_->removeChildComponent(channels_[channelIndex].get());
    channels_.erase(channels_.begin() + channelIndex);
    resized();
}

ChannelStrip* MixerConsolePanel::getChannel(int channelIndex) {
    if (channelIndex < 0 || channelIndex >= static_cast<int>(channels_.size())) {
        return nullptr;
    }
    return channels_[channelIndex].get();
}

std::vector<ChannelStrip*> MixerConsolePanel::getAllChannels() {
    std::vector<ChannelStrip*> result;
    for (auto& ch : channels_) {
        result.push_back(ch.get());
    }
    return result;
}

void MixerConsolePanel::loadPreset(const MixerPreset& preset) {
    channelContainer_->removeAllChildren();
    channels_.clear();
    for (const auto& ch : preset.channels) {
        int idx = addChannel(ch.name, ch.instrument);
        if (auto* channel = getChannel(idx)) {
            channel->setGain(ch.gain);
            channel->setPan(ch.pan);
            for (size_t i = 0; i < ch.insertEffects.size() && i < 4; ++i) {
                channel->addInsertEffect(static_cast<ChannelStrip::InsertSlot>(i), ch.insertEffects[i]);
            }
        }
    }
    resized();
}

void MixerConsolePanel::savePreset(const std::string& name) {
    MixerPreset preset;
    preset.name = name;
    preset.description = "Saved from current mixer state";
    for (const auto& ch : channels_) {
        MixerPreset::ChannelSetup setup;
        setup.name = ch->getName();
        setup.instrument = "";
        setup.gain = ch->getGain();
        setup.pan = ch->getPan();
        setup.insertEffects = ch->getInsertEffectNames();
        preset.channels.push_back(setup);
    }
    presets_.push_back(preset);
    presetSelector_->addItem(preset.name, presetSelector_->getNumItems() + 1);
}

std::vector<MixerConsolePanel::MixerPreset> MixerConsolePanel::getAvailablePresets() const {
    return presets_;
}

void MixerConsolePanel::loadRockBandTemplate() { if (!presets_.empty()) loadPreset(presets_.front()); }
void MixerConsolePanel::loadOrchestralTemplate() { if (presets_.size() > 1) loadPreset(presets_[1]); }
void MixerConsolePanel::loadElectronicTemplate() { if (presets_.size() > 2) loadPreset(presets_[2]); }
void MixerConsolePanel::loadJazzComboTemplate() { if (presets_.size() > 3) loadPreset(presets_[3]); }
void MixerConsolePanel::loadSongwriterTemplate() { if (presets_.size() > 4) loadPreset(presets_[4]); }

void MixerConsolePanel::setMasterGain(float gainDb) {
    masterGainDb_ = gainDb;
    if (masterChannel_) masterChannel_->setGain(gainDb);
}

void MixerConsolePanel::setMasterLimiter(bool enabled, float threshold) {
    juce::ignoreUnused(enabled, threshold);
}

int MixerConsolePanel::addEffectBus(const std::string& name, const std::string& /*effectType*/) {
    auto bus = std::make_unique<ChannelStrip>(name);
    auto* ptr = bus.get();
    effectBuses_.push_back(std::move(bus));
    addAndMakeVisible(ptr);
    resized();
    return static_cast<int>(effectBuses_.size()) - 1;
}

void MixerConsolePanel::setEffectBusLevel(int busIndex, float level) {
    if (busIndex >= 0 && busIndex < static_cast<int>(effectBuses_.size())) {
        effectBuses_[busIndex]->setGain(level);
    }
}

void MixerConsolePanel::setEffectBusParameters(int busIndex, const std::string& paramName, float value) {
    juce::ignoreUnused(busIndex, paramName, value);
}

void MixerConsolePanel::routeChannelToOutput(int channelIndex, int outputBus) {
    juce::Logger::writeToLog("Route channel " + juce::String(channelIndex) + " to bus " + juce::String(outputBus));
}

void MixerConsolePanel::createSubmix(const std::vector<int>& channelIndices,
                                     const std::string& submixName) {
    juce::Logger::writeToLog("Create submix " + juce::String(submixName));
    juce::ignoreUnused(channelIndices);
}

void MixerConsolePanel::setChannelAutomationMode(int channelIndex, AutomationMode mode) {
    juce::Logger::writeToLog("Automation mode for channel " + juce::String(channelIndex) + " set to " + juce::String(static_cast<int>(mode)));
}

void MixerConsolePanel::recordAutomation(int channelIndex, const std::string& parameter,
                                         float value, double timestamp) {
    AutomationPoint point{timestamp, value, parameter};
    automation_[channelIndex].push_back(point);
}

std::vector<MixerConsolePanel::AutomationPoint> MixerConsolePanel::getChannelAutomation(int channelIndex) const {
    auto it = automation_.find(channelIndex);
    if (it != automation_.end()) {
        return it->second;
    }
    return {};
}

void MixerConsolePanel::setViewMode(ViewMode mode) {
    viewMode_ = mode;
    onViewModeChanged();
}

void MixerConsolePanel::setShowEQ(bool show) {
    showEQ_ = show;
    for (auto& ch : channels_) {
        ch->setEQVisible(show);
    }
}

void MixerConsolePanel::setShowCompressor(bool show) {
    showCompressor_ = show;
    for (auto& ch : channels_) {
        ch->setCompressorVisible(show);
    }
}

void MixerConsolePanel::setShowInserts(bool show) {
    showInserts_ = show;
    for (auto& ch : channels_) {
        ch->setInsertVisible(show);
    }
}

void MixerConsolePanel::setShowSends(bool show) {
    showSends_ = show;
    for (auto& ch : channels_) {
        ch->setSendsVisible(show);
    }
}

void MixerConsolePanel::setShowMeters(bool show) {
    showMeters_ = show;
    for (auto& ch : channels_) {
        ch->setMetersVisible(show);
    }
}

void MixerConsolePanel::routeMIDIToChannel(int channelIndex, const juce::MidiBuffer& midi) {
    channelMidi_[channelIndex] = midi;
}

juce::MidiBuffer MixerConsolePanel::getMixedOutput() const {
    juce::MidiBuffer mixed;
    for (const auto& entry : channelMidi_) {
        const int channelIdx = entry.first;
        const auto* channel = const_cast<MixerConsolePanel*>(this)->getChannel(channelIdx);
        juce::MidiBuffer buffer = entry.second;
        if (channel) {
            const_cast<MixerConsolePanel*>(this)->applyMixToMIDI(buffer, channelIdx);
        }
        mixed.addEvents(buffer, 0, buffer.getLastEventTime() + 1, 0);
    }
    return mixed;
}

void MixerConsolePanel::applyMixToMIDI(juce::MidiBuffer& buffer, int channelIndex) {
    auto* channel = getChannel(channelIndex);
    if (!channel) return;

    float gainLinear = dbToLinear(channel->getGain());
    const int panValue = static_cast<int>(juce::jmap(channel->getPan(), -1.0f, 1.0f, 0.0f, 127.0f));

    juce::MidiBuffer updated;
    updated.addEvent(juce::MidiMessage::controllerEvent(1, 10, panValue), 0);

    for (const auto metadata : buffer) {
        auto msg = metadata.getMessage();
        int position = metadata.samplePosition;
        if (msg.isNoteOn()) {
            int velocity = static_cast<int>(msg.getVelocity() * gainLinear * 127.0f);
            velocity = juce::jlimit(1, 127, velocity);
            msg = juce::MidiMessage::noteOn(msg.getChannel(), msg.getNoteNumber(), (juce::uint8)velocity);
        }
        updated.addEvent(msg, position);
    }

    buffer.swapWith(updated);
}

void MixerConsolePanel::saveSnapshot(const std::string& name) {
    MixerSnapshot snapshot;
    snapshot.name = name;
    snapshot.timestamp = juce::Time::getMillisecondCounterHiRes();

    for (size_t i = 0; i < channels_.size(); ++i) {
        MixerSnapshot::ChannelState state;
        auto* ch = channels_[i].get();
        state.gain = ch->getGain();
        state.pan = ch->getPan();
        state.muted = ch->isMuted();
        state.soloed = ch->isSoloed();
        snapshot.channelStates[static_cast<int>(i)] = state;
    }

    snapshots_.push_back(snapshot);
}

void MixerConsolePanel::loadSnapshot(const std::string& name) {
    auto it = std::find_if(snapshots_.begin(), snapshots_.end(),
                           [&name](const MixerSnapshot& s) { return s.name == name; });
    if (it == snapshots_.end()) return;

    for (const auto& [idx, state] : it->channelStates) {
        if (auto* ch = getChannel(idx)) {
            ch->setGain(state.gain);
            ch->setPan(state.pan);
            ch->setMute(state.muted);
            ch->setSolo(state.soloed);
        }
    }
}

std::vector<MixerConsolePanel::MixerSnapshot> MixerConsolePanel::getSnapshots() const {
    return snapshots_;
}

bool MixerConsolePanel::exportSession(const juce::File& outputFile) {
    juce::FileOutputStream stream(outputFile);
    if (!stream.openedOk()) return false;

    stream << "Session\n";
    stream << "Channels: " << static_cast<int>(channels_.size()) << "\n";
    for (size_t i = 0; i < channels_.size(); ++i) {
        auto* ch = channels_[i].get();
        stream << "Channel " << i << " " << ch->getName() << " gain " << ch->getGain()
               << " pan " << ch->getPan() << " muted " << ch->isMuted() << "\n";
    }
    stream.flush();
    return true;
}

bool MixerConsolePanel::importSession(const juce::File& inputFile) {
    juce::ignoreUnused(inputFile);
    // Placeholder: session import not implemented
    return false;
}

void MixerConsolePanel::initializePresets() {
    presets_.clear();
    createRockBandPreset();
    createOrchestralPreset();
    createElectronicPreset();
    createJazzComboPreset();

    presetSelector_->clear();
    int id = 1;
    for (const auto& preset : presets_) {
        presetSelector_->addItem(preset.name, id++);
    }
}

void MixerConsolePanel::createRockBandPreset() {
    MixerPreset preset;
    preset.name = "Rock Band";
    preset.description = "Drums, bass, guitars, vocals";
    preset.channels = {
        {"Drums", "Kit", 0.0f, 0.0f, {}},
        {"Bass", "Bass Guitar", -3.0f, -0.1f, {}},
        {"Guitar L", "Electric Guitar", -6.0f, -0.4f, {}},
        {"Guitar R", "Electric Guitar", -6.0f, 0.4f, {}},
        {"Vocals", "Vox", -2.0f, 0.0f, {}},
    };
    presets_.push_back(preset);
}

void MixerConsolePanel::createOrchestralPreset() {
    MixerPreset preset;
    preset.name = "Orchestral";
    preset.description = "Strings, brass, winds, percussion";
    preset.channels = {
        {"Violins", "Strings", -6.0f, -0.2f, {}},
        {"Violas", "Strings", -6.0f, 0.0f, {}},
        {"Cellos", "Strings", -4.0f, 0.2f, {}},
        {"Brass", "Brass", -3.0f, 0.0f, {}},
        {"Winds", "Winds", -3.0f, 0.0f, {}},
        {"Percussion", "Perc", -2.0f, 0.0f, {}},
    };
    presets_.push_back(preset);
}

void MixerConsolePanel::createElectronicPreset() {
    MixerPreset preset;
    preset.name = "Electronic";
    preset.description = "Synths, drums, FX";
    preset.channels = {
        {"Kick", "Drums", -4.0f, 0.0f, {}},
        {"Snare", "Drums", -3.0f, 0.0f, {}},
        {"Bass", "Synth Bass", -6.0f, -0.2f, {}},
        {"Lead", "Synth Lead", -6.0f, 0.2f, {}},
        {"Pad", "Synth Pad", -8.0f, 0.0f, {}},
        {"FX", "FX", -10.0f, 0.0f, {}},
    };
    presets_.push_back(preset);
}

void MixerConsolePanel::createJazzComboPreset() {
    MixerPreset preset;
    preset.name = "Jazz Combo";
    preset.description = "Piano, bass, drums, horns";
    preset.channels = {
        {"Piano", "Keys", -6.0f, -0.1f, {}},
        {"Bass", "Bass", -4.0f, 0.0f, {}},
        {"Drums", "Kit", -5.0f, 0.0f, {}},
        {"Horn", "Horn", -6.0f, 0.1f, {}},
    };
    presets_.push_back(preset);
}

void MixerConsolePanel::layoutMixerView() {
    const int channelWidth = (viewMode_ == ViewMode::CompactView) ? 80 : 110;
    const int height = channelViewport_->getHeight();
    int x = 0;
    for (auto& ch : channels_) {
        ch->setBounds(x, 0, channelWidth, height);
        x += channelWidth + 8;
    }
    channelContainer_->setSize(std::max(x, channelViewport_->getWidth()), height);
}

void MixerConsolePanel::layoutTrackView() {
    layoutMixerView();
}

void MixerConsolePanel::layoutCompactView() {
    layoutMixerView();
}

void MixerConsolePanel::onChannelGainChanged(int /*channelIndex*/, float /*gain*/) {
    // Integration hook for audio engine
}

void MixerConsolePanel::onChannelPanChanged(int /*channelIndex*/, float /*pan*/) {}
void MixerConsolePanel::onChannelMuteChanged(int /*channelIndex*/, bool /*muted*/) {}
void MixerConsolePanel::onChannelSoloChanged(int /*channelIndex*/, bool /*soloed*/) {}

void MixerConsolePanel::onPresetSelected(int presetIndex) {
    if (presetIndex >= 0 && presetIndex < static_cast<int>(presets_.size())) {
        loadPreset(presets_[static_cast<size_t>(presetIndex)]);
    }
}

void MixerConsolePanel::onViewModeChanged() {
    switch (viewModeSelector_->getSelectedId()) {
        case 1: viewMode_ = ViewMode::MixerView; break;
        case 2: viewMode_ = ViewMode::TrackView; break;
        case 3: viewMode_ = ViewMode::CompactView; break;
        case 4: viewMode_ = ViewMode::FullView; break;
        default: viewMode_ = ViewMode::MixerView; break;
    }

    if (viewMode_ == ViewMode::MixerView) layoutMixerView();
    else if (viewMode_ == ViewMode::TrackView) layoutTrackView();
    else if (viewMode_ == ViewMode::CompactView) layoutCompactView();
    else layoutMixerView();
}

} // namespace midikompanion

