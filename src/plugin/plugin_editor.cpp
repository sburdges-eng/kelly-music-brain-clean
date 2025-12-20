#include "plugin_editor.h"

namespace kelly {

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p), processor(p) {
    
    // Title
    titleLabel_.setText("Kelly Emotion Processor", juce::dontSendNotification);
    titleLabel_.setFont(juce::Font(24.0f, juce::Font::bold));
    titleLabel_.setJustificationType(juce::Justification::centred);
    titleLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(titleLabel_);
    
    // Emotion selector
    emotionLabel_.setText("Emotion", juce::dontSendNotification);
    emotionLabel_.attachToComponent(&emotionComboBox_, false);
    addAndMakeVisible(emotionLabel_);
    
    emotionComboBox_.addItem("calm", 1);
    emotionComboBox_.addItem("excited", 2);
    emotionComboBox_.addItem("sad", 3);
    emotionComboBox_.addItem("angry", 4);
    emotionComboBox_.addItem("happy", 5);
    emotionComboBox_.addItem("anxious", 6);
    emotionComboBox_.addItem("peaceful", 7);
    emotionComboBox_.addItem("energetic", 8);
    emotionComboBox_.addItem("neutral", 9);
    emotionComboBox_.setSelectedId(1);
    emotionComboBox_.onChange = [this] {
        auto* param = processor.getParameters().getParameter(ParameterID::emotionTag);
        if (param) {
            auto* choiceParam = dynamic_cast<juce::AudioParameterChoice*>(param);
            if (choiceParam) {
                choiceParam->beginChangeGesture();
                choiceParam->setValueNotifyingHost(
                    static_cast<float>(emotionComboBox_.getSelectedId() - 1) / 9.0f);
                choiceParam->endChangeGesture();
            }
        }
    };
    
    // Create attachment for emotion combo box
    auto* emotionParam = processor.getParameters().getParameter(ParameterID::emotionTag);
    if (emotionParam) {
        auto* choiceParam = dynamic_cast<juce::AudioParameterChoice*>(emotionParam);
        if (choiceParam) {
            emotionComboBox_.setSelectedId(choiceParam->getIndex() + 1);
        }
    }
    addAndMakeVisible(emotionComboBox_);
    
    // Setup sliders
    setupSlider(valenceSlider_, valenceLabel_, ParameterID::valence, 
                "Valence", juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f));
    valenceValueLabel_.setText("0.00", juce::dontSendNotification);
    valenceValueLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(valenceValueLabel_);
    
    setupSlider(arousalSlider_, arousalLabel_, ParameterID::arousal,
                "Arousal", juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f));
    arousalValueLabel_.setText("0.00", juce::dontSendNotification);
    arousalValueLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(arousalValueLabel_);
    
    setupSlider(intensitySlider_, intensityLabel_, ParameterID::intensity,
                "Intensity", juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f));
    intensityValueLabel_.setText("0.00", juce::dontSendNotification);
    intensityValueLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(intensityValueLabel_);
    
    setupSlider(barsSlider_, barsLabel_, ParameterID::bars,
                "Bars", juce::NormalisableRange<float>(1.0f, 8.0f, 1.0f));
    barsSlider_.setTextValueSuffix(" bars");
    barsValueLabel_.setText("1", juce::dontSendNotification);
    barsValueLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(barsValueLabel_);
    
    setupSlider(generationRateSlider_, generationRateLabel_, ParameterID::generationRate,
                "Generation Rate", juce::NormalisableRange<float>(1.0f, 32.0f, 1.0f));
    generationRateSlider_.setTextValueSuffix(" blocks");
    generationRateValueLabel_.setText("8", juce::dontSendNotification);
    generationRateValueLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(generationRateValueLabel_);
    
    // Cloud toggle
    cloudToggle_.setButtonText("Enable Cloud Generation");
    cloudAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        processor.getParameters(), ParameterID::enableCloud, cloudToggle_);
    addAndMakeVisible(cloudToggle_);
    
    // Status label
    statusLabel_.setText("Ready", juce::dontSendNotification);
    statusLabel_.setJustificationType(juce::Justification::centred);
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    addAndMakeVisible(statusLabel_);
    
    // Update initial values
    updateValueLabels();
    
    // Start timer for periodic updates
    startTimer(100); // Update every 100ms
    
    setSize(500, 500);
}

PluginEditor::~PluginEditor() {
    stopTimer();
}

void PluginEditor::paint(juce::Graphics& g) {
    // Background gradient
    juce::ColourGradient gradient(
        juce::Colour(0xff1a1a2e), 0.0f, 0.0f,
        juce::Colour(0xff16213e), 0.0f, static_cast<float>(getHeight()),
        false
    );
    g.setGradientFill(gradient);
    g.fillAll();
    
    // Draw border
    g.setColour(juce::Colour(0xff4a90e2));
    g.drawRect(getLocalBounds(), 2);
}

void PluginEditor::resized() {
    auto bounds = getLocalBounds().reduced(20);
    
    // Title at top
    titleLabel_.setBounds(bounds.removeFromTop(40));
    bounds.removeFromTop(10);
    
    // Emotion selector
    auto emotionBounds = bounds.removeFromTop(60);
    emotionLabel_.setBounds(emotionBounds.removeFromLeft(100));
    emotionComboBox_.setBounds(emotionBounds.reduced(5));
    bounds.removeFromTop(10);
    
    // Valence slider
    auto valenceBounds = bounds.removeFromTop(60);
    valenceLabel_.setBounds(valenceBounds.removeFromLeft(100));
    auto sliderBounds = valenceBounds.removeFromLeft(valenceBounds.getWidth() - 60);
    valenceSlider_.setBounds(sliderBounds.reduced(5));
    valenceValueLabel_.setBounds(valenceBounds.reduced(5));
    bounds.removeFromTop(5);
    
    // Arousal slider
    auto arousalBounds = bounds.removeFromTop(60);
    arousalLabel_.setBounds(arousalBounds.removeFromLeft(100));
    sliderBounds = arousalBounds.removeFromLeft(arousalBounds.getWidth() - 60);
    arousalSlider_.setBounds(sliderBounds.reduced(5));
    arousalValueLabel_.setBounds(arousalBounds.reduced(5));
    bounds.removeFromTop(5);
    
    // Intensity slider
    auto intensityBounds = bounds.removeFromTop(60);
    intensityLabel_.setBounds(intensityBounds.removeFromLeft(100));
    sliderBounds = intensityBounds.removeFromLeft(intensityBounds.getWidth() - 60);
    intensitySlider_.setBounds(sliderBounds.reduced(5));
    intensityValueLabel_.setBounds(intensityBounds.reduced(5));
    bounds.removeFromTop(5);
    
    // Bars slider
    auto barsBounds = bounds.removeFromTop(60);
    barsLabel_.setBounds(barsBounds.removeFromLeft(100));
    sliderBounds = barsBounds.removeFromLeft(barsBounds.getWidth() - 60);
    barsSlider_.setBounds(sliderBounds.reduced(5));
    barsValueLabel_.setBounds(barsBounds.reduced(5));
    bounds.removeFromTop(5);
    
    // Generation rate slider
    auto genRateBounds = bounds.removeFromTop(60);
    generationRateLabel_.setBounds(genRateBounds.removeFromLeft(100));
    sliderBounds = genRateBounds.removeFromLeft(genRateBounds.getWidth() - 60);
    generationRateSlider_.setBounds(sliderBounds.reduced(5));
    generationRateValueLabel_.setBounds(genRateBounds.reduced(5));
    bounds.removeFromTop(10);
    
    // Cloud toggle
    cloudToggle_.setBounds(bounds.removeFromTop(30));
    bounds.removeFromTop(10);
    
    // Status label at bottom
    statusLabel_.setBounds(bounds.removeFromBottom(30));
}

void PluginEditor::timerCallback() {
    updateValueLabels();
    
    // Update status
    if (processor.isCloudEnabled()) {
        statusLabel_.setText("Cloud enabled", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::orange);
    } else {
        statusLabel_.setText("Local processing", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    }
}

void PluginEditor::setupSlider(juce::Slider& slider, juce::Label& label,
                               const juce::String& parameterID,
                               const juce::String& labelText,
                               juce::NormalisableRange<float> range) {
    label.setText(labelText, juce::dontSendNotification);
    label.attachToComponent(&slider, true);
    addAndMakeVisible(label);
    
    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    slider.setRange(range.start, range.end, range.interval);
    
    auto attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.getParameters(), parameterID, slider);
    
    if (parameterID == ParameterID::valence) {
        valenceAttachment_ = std::move(attachment);
    } else if (parameterID == ParameterID::arousal) {
        arousalAttachment_ = std::move(attachment);
    } else if (parameterID == ParameterID::intensity) {
        intensityAttachment_ = std::move(attachment);
    } else if (parameterID == ParameterID::bars) {
        barsAttachment_ = std::move(attachment);
    } else if (parameterID == ParameterID::generationRate) {
        generationRateAttachment_ = std::move(attachment);
    }
    
    addAndMakeVisible(slider);
}

void PluginEditor::updateValueLabels() {
    auto* valenceParam = processor.getParameters().getRawParameterValue(ParameterID::valence);
    if (valenceParam) {
        valenceValueLabel_.setText(juce::String(valenceParam->load(), 2), juce::dontSendNotification);
    }
    
    auto* arousalParam = processor.getParameters().getRawParameterValue(ParameterID::arousal);
    if (arousalParam) {
        arousalValueLabel_.setText(juce::String(arousalParam->load(), 2), juce::dontSendNotification);
    }
    
    auto* intensityParam = processor.getParameters().getRawParameterValue(ParameterID::intensity);
    if (intensityParam) {
        intensityValueLabel_.setText(juce::String(intensityParam->load(), 2), juce::dontSendNotification);
    }
    
    auto* barsParam = processor.getParameters().getRawParameterValue(ParameterID::bars);
    if (barsParam) {
        barsValueLabel_.setText(juce::String(static_cast<int>(barsParam->load())), juce::dontSendNotification);
    }
    
    auto* genRateParam = processor.getParameters().getRawParameterValue(ParameterID::generationRate);
    if (genRateParam) {
        generationRateValueLabel_.setText(juce::String(static_cast<int>(genRateParam->load())), juce::dontSendNotification);
    }
}

} // namespace kelly
