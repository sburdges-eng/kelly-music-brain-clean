#include "VocalControlPanel.h"
#include "../voice/LyricTypes.h"
#include "../voice/VoiceSynthesizer.h"
#include <cmath>

namespace kelly {

VocalControlPanel::VocalControlPanel()
    : currentVoiceType_(VoiceType::Neutral)
{
    setupSliders();

    // Setup voice type combo box
    voiceTypeComboBox_.addItem("Neutral", 1);
    voiceTypeComboBox_.addItem("Male", 2);
    voiceTypeComboBox_.addItem("Female", 3);
    voiceTypeComboBox_.addItem("Child", 4);
    voiceTypeComboBox_.setSelectedId(1);
    voiceTypeComboBox_.onChange = [this] { voiceTypeChanged(); };

    addAndMakeVisible(voiceTypeComboBox_);
    addAndMakeVisible(voiceTypeLabel_);

    // Add sliders and labels
    addAndMakeVisible(vibratoDepthSlider_);
    addAndMakeVisible(vibratoDepthLabel_);
    addAndMakeVisible(vibratoRateSlider_);
    addAndMakeVisible(vibratoRateLabel_);
    addAndMakeVisible(breathinessSlider_);
    addAndMakeVisible(breathinessLabel_);
    addAndMakeVisible(brightnessSlider_);
    addAndMakeVisible(brightnessLabel_);
    addAndMakeVisible(dynamicsSlider_);
    addAndMakeVisible(dynamicsLabel_);
    addAndMakeVisible(articulationSlider_);
    addAndMakeVisible(articulationLabel_);

    setOpaque(true);
}

void VocalControlPanel::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff2a2a2a)); // Dark panel background

    // Draw section title
    g.setColour(juce::Colours::white);
    g.setFont(18.0f);
    g.drawText("Vocal Controls", getLocalBounds().withHeight(30),
               juce::Justification::centredTop);
}

void VocalControlPanel::resized() {
    auto bounds = getLocalBounds();
    bounds.removeFromTop(35); // Title area

    const int sliderHeight = 60;
    const int labelHeight = 20;
    const int spacing = 5;
    int yPos = bounds.getY();

    // Voice type selector
    voiceTypeLabel_.setBounds(bounds.getX(), yPos, bounds.getWidth(), labelHeight);
    yPos += labelHeight + spacing;
    voiceTypeComboBox_.setBounds(bounds.getX() + 10, yPos, bounds.getWidth() - 20, 25);
    yPos += 30 + spacing;

    // Sliders
    auto setupSliderRow = [&](juce::Slider& slider, juce::Label& label, const juce::String& labelText) {
        label.setText(labelText, juce::dontSendNotification);
        label.setBounds(bounds.getX(), yPos, bounds.getWidth() / 2, labelHeight);
        slider.setBounds(bounds.getX() + bounds.getWidth() / 2, yPos,
                        bounds.getWidth() / 2, sliderHeight);
        yPos += sliderHeight + spacing;
    };

    setupSliderRow(vibratoDepthSlider_, vibratoDepthLabel_, "Vibrato Depth");
    setupSliderRow(vibratoRateSlider_, vibratoRateLabel_, "Vibrato Rate (Hz)");
    setupSliderRow(breathinessSlider_, breathinessLabel_, "Breathiness");
    setupSliderRow(brightnessSlider_, brightnessLabel_, "Brightness");
    setupSliderRow(dynamicsSlider_, dynamicsLabel_, "Dynamics");
    setupSliderRow(articulationSlider_, articulationLabel_, "Articulation");
}

void VocalControlPanel::setupSliders() {
    // Vibrato depth: 0.0 to 1.0
    vibratoDepthSlider_.setRange(0.0, 1.0, 0.01);
    vibratoDepthSlider_.setValue(currentExpression_.vibratoDepth);
    vibratoDepthSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    vibratoDepthSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    vibratoDepthSlider_.onValueChange = [this] { updateExpressionFromSliders(); };

    // Vibrato rate: 2.0 to 8.0 Hz
    vibratoRateSlider_.setRange(2.0, 8.0, 0.1);
    vibratoRateSlider_.setValue(currentExpression_.vibratoRate);
    vibratoRateSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    vibratoRateSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    vibratoRateSlider_.onValueChange = [this] { updateExpressionFromSliders(); };

    // Breathiness: 0.0 to 1.0
    breathinessSlider_.setRange(0.0, 1.0, 0.01);
    breathinessSlider_.setValue(currentExpression_.breathiness);
    breathinessSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    breathinessSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    breathinessSlider_.onValueChange = [this] { updateExpressionFromSliders(); };

    // Brightness: 0.0 to 1.0
    brightnessSlider_.setRange(0.0, 1.0, 0.01);
    brightnessSlider_.setValue(currentExpression_.brightness);
    brightnessSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    brightnessSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    brightnessSlider_.onValueChange = [this] { updateExpressionFromSliders(); };

    // Dynamics: 0.0 to 1.0
    dynamicsSlider_.setRange(0.0, 1.0, 0.01);
    dynamicsSlider_.setValue(currentExpression_.dynamics);
    dynamicsSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    dynamicsSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    dynamicsSlider_.onValueChange = [this] { updateExpressionFromSliders(); };

    // Articulation: 0.0 to 1.0 (0 = legato, 1 = staccato)
    articulationSlider_.setRange(0.0, 1.0, 0.01);
    articulationSlider_.setValue(currentExpression_.articulation);
    articulationSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    articulationSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    articulationSlider_.onValueChange = [this] { updateExpressionFromSliders(); };
}

VoiceType VocalControlPanel::getVoiceType() const {
    return currentVoiceType_;
}

VocalExpression VocalControlPanel::getExpression() const {
    return currentExpression_;
}

void VocalControlPanel::setVoiceType(VoiceType voiceType) {
    currentVoiceType_ = voiceType;

    int comboId = 1; // Neutral
    switch (voiceType) {
        case VoiceType::Male:
            comboId = 2;
            break;
        case VoiceType::Female:
            comboId = 3;
            break;
        case VoiceType::Child:
            comboId = 4;
            break;
        default:
            comboId = 1;
            break;
    }

    voiceTypeComboBox_.setSelectedId(comboId, juce::dontSendNotification);
}

void VocalControlPanel::setExpression(const VocalExpression& expression) {
    currentExpression_ = expression;

    vibratoDepthSlider_.setValue(expression.vibratoDepth, juce::dontSendNotification);
    vibratoRateSlider_.setValue(expression.vibratoRate, juce::dontSendNotification);
    breathinessSlider_.setValue(expression.breathiness, juce::dontSendNotification);
    brightnessSlider_.setValue(expression.brightness, juce::dontSendNotification);
    dynamicsSlider_.setValue(expression.dynamics, juce::dontSendNotification);
    articulationSlider_.setValue(expression.articulation, juce::dontSendNotification);
}

void VocalControlPanel::updateExpressionFromSliders() {
    currentExpression_.vibratoDepth = static_cast<float>(vibratoDepthSlider_.getValue());
    currentExpression_.vibratoRate = static_cast<float>(vibratoRateSlider_.getValue());
    currentExpression_.breathiness = static_cast<float>(breathinessSlider_.getValue());
    currentExpression_.brightness = static_cast<float>(brightnessSlider_.getValue());
    currentExpression_.dynamics = static_cast<float>(dynamicsSlider_.getValue());
    currentExpression_.articulation = static_cast<float>(articulationSlider_.getValue());

    if (onExpressionChanged) {
        onExpressionChanged(currentExpression_);
    }
}

void VocalControlPanel::voiceTypeChanged() {
    int selectedId = voiceTypeComboBox_.getSelectedId();

    VoiceType newType = VoiceType::Neutral;
    switch (selectedId) {
        case 2:
            newType = VoiceType::Male;
            break;
        case 3:
            newType = VoiceType::Female;
            break;
        case 4:
            newType = VoiceType::Child;
            break;
        default:
            newType = VoiceType::Neutral;
            break;
    }

    if (newType != currentVoiceType_) {
        currentVoiceType_ = newType;
        if (onVoiceTypeChanged) {
            onVoiceTypeChanged(currentVoiceType_);
        }
    }
}

} // namespace kelly
