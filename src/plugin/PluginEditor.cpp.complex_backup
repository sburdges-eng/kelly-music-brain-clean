#include "plugin/PluginEditor.h"
#include "midi/ChordGenerator.h"
#include "common/Types.h"
#include "ui/EmotionWheel.h"
#include "ui/CassetteView.h"

namespace kelly {

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(p), 
      processor_(p),
      valenceAttachment_(processor_.parameters, "valence", valenceSlider_),
      arousalAttachment_(processor_.parameters, "arousal", arousalSlider_),
      intensityAttachment_(processor_.parameters, "intensity", intensitySlider_),
      complexityAttachment_(processor_.parameters, "complexity", complexitySlider_),
      feelAttachment_(processor_.parameters, "feel", feelSlider_),
      dynamicsAttachment_(processor_.parameters, "dynamics", dynamicsSlider_),
      barsAttachment_(processor_.parameters, "bars", barsSlider_)
{
    setSize(500, 650);
    setResizable(true, true);
    setResizeLimits(MIN_WIDTH, MIN_HEIGHT, 800, 700);
    
    // =========================================================================
    // CATEGORY AND STYLE SELECTORS (Like Logic Session Player)
    // =========================================================================
    
    categoryLabel_.setText("Category", juce::dontSendNotification);
    categoryLabel_.setFont(juce::FontOptions(14.0f).withStyle("Bold"));
    categoryLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(categoryLabel_);
    
    const auto& categories = getCategories();
    for (size_t i = 0; i < categories.size(); ++i) {
        categorySelector_.addItem(categories[i].name, static_cast<int>(i + 1));
    }
    categorySelector_.setSelectedId(1);
    categorySelector_.onChange = [this] { onCategoryChanged(); };
    addAndMakeVisible(categorySelector_);
    
    styleLabel_.setText("Style", juce::dontSendNotification);
    styleLabel_.setFont(juce::FontOptions(14.0f).withStyle("Bold"));
    styleLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(styleLabel_);
    
    styleSelector_.onChange = [this] { onStyleChanged(); };
    addAndMakeVisible(styleSelector_);
    
    // Initialize style selector with first category's styles
    onCategoryChanged();
    
    // =========================================================================
    // PRESET SELECTOR
    // =========================================================================
    
    presetLabel_.setText("Emotion Preset", juce::dontSendNotification);
    presetLabel_.setFont(juce::FontOptions(14.0f).withStyle("Bold"));
    presetLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(presetLabel_);
    
    presetSelector_.addItem("Default", 1);
    const auto& presets = PluginProcessor::getEmotionPresets();
    for (size_t i = 0; i < presets.size(); ++i) {
        presetSelector_.addItem(presets[i].name, static_cast<int>(i + 2));
    }
    presetSelector_.setSelectedId(1);
    presetSelector_.onChange = [this] { onPresetChanged(); };
    addAndMakeVisible(presetSelector_);
    
    // =========================================================================
    // VALENCE SLIDER
    // =========================================================================
    
    valenceLabel_.setText("Valence", juce::dontSendNotification);
    valenceLabel_.setFont(juce::FontOptions(12.0f));
    valenceLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(valenceLabel_);
    
    valenceSlider_.setRange(-1.0, 1.0, 0.01);
    valenceSlider_.setValue(0.0);
    valenceSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    valenceSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    valenceSlider_.setColour(juce::Slider::thumbColourId, juce::Colours::blue);
    addAndMakeVisible(valenceSlider_);
    
    // =========================================================================
    // AROUSAL SLIDER
    // =========================================================================
    
    arousalLabel_.setText("Arousal", juce::dontSendNotification);
    arousalLabel_.setFont(juce::FontOptions(12.0f));
    arousalLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(arousalLabel_);
    
    arousalSlider_.setRange(0.0, 1.0, 0.01);
    arousalSlider_.setValue(0.5);
    arousalSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    arousalSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    arousalSlider_.setColour(juce::Slider::thumbColourId, juce::Colours::red);
    addAndMakeVisible(arousalSlider_);
    
    // =========================================================================
    // INTENSITY SLIDER
    // =========================================================================
    
    intensityLabel_.setText("Intensity", juce::dontSendNotification);
    intensityLabel_.setFont(juce::FontOptions(12.0f));
    intensityLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(intensityLabel_);
    
    intensitySlider_.setRange(0.0, 1.0, 0.01);
    intensitySlider_.setValue(0.7);
    intensitySlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    intensitySlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    intensitySlider_.setColour(juce::Slider::thumbColourId, juce::Colours::purple);
    addAndMakeVisible(intensitySlider_);
    
    // =========================================================================
    // FINE-TUNING CONTROLS
    // =========================================================================
    
    complexityLabel_.setText("Complexity", juce::dontSendNotification);
    complexityLabel_.setFont(juce::FontOptions(12.0f));
    complexityLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(complexityLabel_);
    
    complexitySlider_.setRange(0.0, 1.0, 0.01);
    complexitySlider_.setValue(0.5);
    complexitySlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    complexitySlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    complexitySlider_.setColour(juce::Slider::thumbColourId, juce::Colours::orange);
    addAndMakeVisible(complexitySlider_);
    
    feelLabel_.setText("Feel (Pull/Push)", juce::dontSendNotification);
    feelLabel_.setFont(juce::FontOptions(12.0f));
    feelLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(feelLabel_);
    
    feelSlider_.setRange(-1.0, 1.0, 0.01);
    feelSlider_.setValue(0.0);
    feelSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    feelSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    feelSlider_.setColour(juce::Slider::thumbColourId, juce::Colours::cyan);
    addAndMakeVisible(feelSlider_);
    
    dynamicsLabel_.setText("Dynamics", juce::dontSendNotification);
    dynamicsLabel_.setFont(juce::FontOptions(12.0f));
    dynamicsLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(dynamicsLabel_);
    
    dynamicsSlider_.setRange(0.0, 1.0, 0.01);
    dynamicsSlider_.setValue(0.75);
    dynamicsSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    dynamicsSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    dynamicsSlider_.setColour(juce::Slider::thumbColourId, juce::Colours::green);
    addAndMakeVisible(dynamicsSlider_);
    
    barsLabel_.setText("Bars", juce::dontSendNotification);
    barsLabel_.setFont(juce::FontOptions(12.0f));
    barsLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(barsLabel_);
    
    barsSlider_.setRange(4.0, 32.0, 1.0);
    barsSlider_.setValue(8.0);
    barsSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    barsSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    barsSlider_.setColour(juce::Slider::thumbColourId, juce::Colours::yellow);
    addAndMakeVisible(barsSlider_);
    
    // =========================================================================
    // CONTROL BUTTONS
    // =========================================================================
    
    generateButton_.setButtonText("GENERATE");
    generateButton_.onClick = [this] { onGenerate(); };
    addAndMakeVisible(generateButton_);
    
    playStopButton_.setButtonText("PLAY");
    playStopButton_.onClick = [this] { onPlayStop(); };
    playStopButton_.setEnabled(false);
    addAndMakeVisible(playStopButton_);
    
    exportButton_.setButtonText("EXPORT");
    exportButton_.onClick = [this] { onExport(); };
    exportButton_.setEnabled(false);
    addAndMakeVisible(exportButton_);
    
    // =========================================================================
    // STATUS DISPLAY
    // =========================================================================
    
    statusLabel_.setText("Select an emotion preset or adjust sliders, then click Generate", juce::dontSendNotification);
    statusLabel_.setFont(juce::FontOptions(11.0f));
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(statusLabel_);
    
    emotionDisplay_.setText("", juce::dontSendNotification);
    emotionDisplay_.setFont(juce::FontOptions(13.0f));
    emotionDisplay_.setColour(juce::Label::textColourId, juce::Colours::darkblue);
    addAndMakeVisible(emotionDisplay_);
    
    // =========================================================================
    // V2.0 FEATURES: Emotion Wheel and Cassette View
    // =========================================================================
    
    // Create emotion wheel
    emotionWheel_ = std::make_unique<EmotionWheel>();
    emotionWheel_->setThesaurus(&processor_.getIntentPipeline().thesaurus());
    emotionWheel_->onEmotionSelected([this](const EmotionNode& emotion) {
        // Update sliders when emotion is selected from wheel
        valenceSlider_.setValue(emotion.valence, juce::dontSendNotification);
        arousalSlider_.setValue(emotion.arousal, juce::dontSendNotification);
        intensitySlider_.setValue(emotion.intensity, juce::dontSendNotification);
        *processor_.parameters.getRawParameterValue("emotionId") = static_cast<float>(emotion.id);
        updateEmotionDisplay();
    });
    addAndMakeVisible(emotionWheel_.get());
    emotionWheel_->setVisible(false);  // Hidden by default, toggle with button
    
    // Toggle button for emotion wheel
    showWheelButton_.setButtonText("Show Emotion Wheel");
    showWheelButton_.setToggleState(false, juce::dontSendNotification);
    showWheelButton_.onClick = [this] {
        bool show = showWheelButton_.getToggleState();
        emotionWheel_->setVisible(show);
        showWheelButton_.setButtonText(show ? "Hide Emotion Wheel" : "Show Emotion Wheel");
        resized();
    };
    addAndMakeVisible(showWheelButton_);
    
    // Create cassette view (optional, can wrap content)
    cassetteView_ = std::make_unique<CassetteView>();
    cassetteView_->setVisible(false);  // Hidden by default, can be enabled
    addAndMakeVisible(cassetteView_.get());
    
    // Toggle button for cassette view
    showCassetteViewButton_.setButtonText("Show Cassette View");
    showCassetteViewButton_.setToggleState(false, juce::dontSendNotification);
    showCassetteViewButton_.onClick = [this] {
        bool show = showCassetteViewButton_.getToggleState();
        cassetteView_->setVisible(show);
        cassetteView_->setTapeAnimating(show);
        showCassetteViewButton_.setButtonText(show ? "Hide Cassette View" : "Show Cassette View");
        resized();
    };
    addAndMakeVisible(showCassetteViewButton_);
    
    // Start timer for UI updates
    startTimerHz(30);
}

PluginEditor::~PluginEditor() {
    stopTimer();
}

void PluginEditor::timerCallback() {
    updatePlayStopButton();
    updateEmotionDisplay();
}

void PluginEditor::updatePlayStopButton() {
    if (processor_.isPlaying()) {
        playStopButton_.setButtonText("STOP");
        playStopButton_.setColour(juce::TextButton::buttonColourId, juce::Colours::indianred);
    } else {
        playStopButton_.setButtonText("PLAY");
        playStopButton_.setColour(juce::TextButton::buttonColourId, 
            getLookAndFeel().findColour(juce::TextButton::buttonColourId));
    }
}

void PluginEditor::updateEmotionDisplay() {
    float valence = *processor_.parameters.getRawParameterValue("valence");
    float arousal = *processor_.parameters.getRawParameterValue("arousal");
    float intensity = *processor_.parameters.getRawParameterValue("intensity");
    
    juce::String display;
    display << "Valence: " << juce::String(valence, 2);
    display << " | Arousal: " << juce::String(arousal, 2);
    display << " | Intensity: " << juce::String(intensity, 2);
    
    emotionDisplay_.setText(display, juce::dontSendNotification);
}

std::vector<PluginEditor::Category> PluginEditor::getCategories() {
    return {
        {
            "Emotional Core",
            {"Joy", "Sadness", "Anger", "Fear", "Surprise", "Peace"},
            {{0.8f, 0.7f, 0.6f}, {-0.6f, 0.3f, 0.5f}, {-0.8f, 0.9f, 0.7f}, 
             {-0.7f, 0.8f, 0.7f}, {0.2f, 0.8f, 0.6f}, {0.5f, 0.1f, 0.3f}}
        },
        {
            "Therapeutic Journey",
            {"Grief to Acceptance", "Anxiety to Calm", "Anger to Release", 
             "Despair to Hope", "Loneliness to Connection", "Shame to Self-Love"},
            {{-0.9f, 0.3f, 0.9f}, {-0.6f, 0.7f, 0.7f}, {-0.8f, 0.9f, 0.7f},
             {-1.0f, 0.4f, 0.9f}, {-0.5f, 0.2f, 0.5f}, {-0.7f, 0.3f, 0.6f}}
        },
        {
            "Musical Expression",
            {"Melancholic", "Energetic", "Contemplative", "Passionate", 
             "Serene", "Intense"},
            {{-0.6f, 0.2f, 0.4f}, {0.6f, 0.9f, 0.8f}, {0.3f, 0.2f, 0.3f},
             {0.4f, 0.8f, 0.7f}, {0.6f, 0.1f, 0.3f}, {-0.5f, 0.9f, 0.8f}}
        },
        {
            "Intensity Levels",
            {"Subtle", "Moderate", "Intense", "Overwhelming", "Cathartic", "Transcendent"},
            {{0.2f, 0.2f, 0.2f}, {0.0f, 0.5f, 0.5f}, {-0.3f, 0.7f, 0.7f},
             {-0.5f, 0.9f, 0.9f}, {-0.2f, 0.8f, 0.8f}, {0.5f, 0.6f, 1.0f}}
        },
        {
            "Complex Emotions",
            {"Bittersweet", "Nostalgic", "Longing", "Gratitude", "Awe", "Catharsis"},
            {{-0.2f, 0.4f, 0.5f}, {-0.2f, 0.3f, 0.4f}, {-0.4f, 0.5f, 0.6f},
             {0.8f, 0.4f, 0.5f}, {0.7f, 0.4f, 0.6f}, {-0.1f, 0.7f, 0.8f}}
        }
    };
}

void PluginEditor::onCategoryChanged() {
    int categoryId = categorySelector_.getSelectedId();
    if (categoryId <= 0) return;
    
    const auto& categories = getCategories();
    if (categoryId > static_cast<int>(categories.size())) return;
    
    const auto& category = categories[categoryId - 1];
    
    // Update style selector with this category's styles
    styleSelector_.clear();
    for (size_t i = 0; i < category.styles.size(); ++i) {
        styleSelector_.addItem(category.styles[i], static_cast<int>(i + 1));
    }
    styleSelector_.setSelectedId(1);
    
    // Update emotion sliders based on first style
    onStyleChanged();
}

void PluginEditor::onStyleChanged() {
    int categoryId = categorySelector_.getSelectedId();
    int styleId = styleSelector_.getSelectedId();
    
    if (categoryId <= 0 || styleId <= 0) return;
    
    const auto& categories = getCategories();
    if (categoryId > static_cast<int>(categories.size())) return;
    
    const auto& category = categories[categoryId - 1];
    if (styleId <= 0 || styleId > static_cast<int>(category.styles.size())) return;
    if (styleId > static_cast<int>(category.styleParams.size())) return;
    
    // Get parameters for this style
    const auto& params = category.styleParams[styleId - 1];
    float valence = std::get<0>(params);
    float arousal = std::get<1>(params);
    float intensity = std::get<2>(params);
    
    // Update sliders
    valenceSlider_.setValue(valence, juce::dontSendNotification);
    arousalSlider_.setValue(arousal, juce::dontSendNotification);
    intensitySlider_.setValue(intensity, juce::dontSendNotification);
}

void PluginEditor::onPresetChanged() {
    int selectedId = presetSelector_.getSelectedId();
    if (selectedId > 0) {
        processor_.setCurrentProgram(selectedId - 1);
        
        // Update sliders to reflect preset values
        const auto& presets = PluginProcessor::getEmotionPresets();
        if (selectedId > 1 && selectedId <= static_cast<int>(presets.size() + 1)) {
            const auto& preset = presets[selectedId - 2];
            valenceSlider_.setValue(preset.valence, juce::dontSendNotification);
            arousalSlider_.setValue(preset.arousal, juce::dontSendNotification);
            intensitySlider_.setValue(preset.intensity, juce::dontSendNotification);
        }
    }
}

void PluginEditor::paint(juce::Graphics& g) {
    // Background
    g.fillAll(juce::Colour(0xFFF5F5F5));
    
    // Header
    g.setColour(juce::Colours::darkgrey);
    g.setFont(juce::FontOptions(24.0f).withStyle("Bold"));
    g.drawText("KELLY", getLocalBounds().removeFromTop(40), juce::Justification::centred);
    
    // Subtitle
    g.setFont(juce::FontOptions(11.0f));
    g.setColour(juce::Colours::grey);
    g.drawText("Therapeutic MIDI Companion", getLocalBounds().removeFromTop(55).translated(0, 35), 
               juce::Justification::centred);
    
    // Resize handle at bottom
    auto resizeArea = getLocalBounds().removeFromBottom(RESIZE_HANDLE_HEIGHT);
    g.setColour(juce::Colours::lightgrey);
    g.fillRect(resizeArea);
    g.setColour(juce::Colours::darkgrey);
    g.drawHorizontalLine(resizeArea.getY(), 0.0f, static_cast<float>(getWidth()));
    
    // Draw drag indicator (three horizontal lines)
    int centerX = getWidth() / 2;
    int centerY = resizeArea.getCentreY();
    g.setColour(juce::Colours::darkgrey.withAlpha(0.5f));
    for (int i = -1; i <= 1; ++i) {
        g.fillRect(centerX - 20, centerY + i * 2, 40, 1);
    }
}

void PluginEditor::resized() {
    auto bounds = getLocalBounds().reduced(20);
    
    // Header space
    bounds.removeFromTop(60);
    
    // Category and Style selectors (side by side)
    auto categoryArea = bounds.removeFromTop(50);
    auto leftHalf = categoryArea.removeFromLeft(categoryArea.getWidth() / 2).reduced(5, 0);
    auto rightHalf = categoryArea.reduced(5, 0);
    
    categoryLabel_.setBounds(leftHalf.removeFromTop(20));
    categorySelector_.setBounds(leftHalf.removeFromTop(25).reduced(0, 2));
    
    styleLabel_.setBounds(rightHalf.removeFromTop(20));
    styleSelector_.setBounds(rightHalf.removeFromTop(25).reduced(0, 2));
    
    bounds.removeFromTop(10);
    
    // Preset selector
    auto presetArea = bounds.removeFromTop(50);
    presetLabel_.setBounds(presetArea.removeFromTop(20));
    presetSelector_.setBounds(presetArea.removeFromTop(25).reduced(0, 2));
    
    bounds.removeFromTop(15);
    
    // Sliders
    auto sliderHeight = 50;
    auto sliderArea = bounds.removeFromTop(sliderHeight);
    valenceLabel_.setBounds(sliderArea.removeFromTop(18));
    valenceSlider_.setBounds(sliderArea.removeFromTop(25));
    
    bounds.removeFromTop(10);
    sliderArea = bounds.removeFromTop(sliderHeight);
    arousalLabel_.setBounds(sliderArea.removeFromTop(18));
    arousalSlider_.setBounds(sliderArea.removeFromTop(25));
    
    bounds.removeFromTop(10);
    sliderArea = bounds.removeFromTop(sliderHeight);
    intensityLabel_.setBounds(sliderArea.removeFromTop(18));
    intensitySlider_.setBounds(sliderArea.removeFromTop(25));
    
    // Fine-tuning controls
    bounds.removeFromTop(15);
    sliderArea = bounds.removeFromTop(sliderHeight);
    complexityLabel_.setBounds(sliderArea.removeFromTop(18));
    complexitySlider_.setBounds(sliderArea.removeFromTop(25));
    
    bounds.removeFromTop(10);
    sliderArea = bounds.removeFromTop(sliderHeight);
    feelLabel_.setBounds(sliderArea.removeFromTop(18));
    feelSlider_.setBounds(sliderArea.removeFromTop(25));
    
    bounds.removeFromTop(10);
    sliderArea = bounds.removeFromTop(sliderHeight);
    dynamicsLabel_.setBounds(sliderArea.removeFromTop(18));
    dynamicsSlider_.setBounds(sliderArea.removeFromTop(25));
    
    bounds.removeFromTop(10);
    sliderArea = bounds.removeFromTop(sliderHeight);
    barsLabel_.setBounds(sliderArea.removeFromTop(18));
    barsSlider_.setBounds(sliderArea.removeFromTop(25));
    
    // Buttons
    bounds.removeFromTop(20);
    auto buttonArea = bounds.removeFromTop(40);
    int buttonWidth = buttonArea.getWidth() / 3;
    generateButton_.setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(5));
    playStopButton_.setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(5));
    exportButton_.setBounds(buttonArea.reduced(5));
    
    // Emotion wheel toggle button (v2.0)
    bounds.removeFromTop(10);
    auto toggleArea = bounds.removeFromTop(30);
    showWheelButton_.setBounds(toggleArea.removeFromLeft(toggleArea.getWidth() / 2).reduced(5, 0));
    showCassetteViewButton_.setBounds(toggleArea.reduced(5, 0));
    
    // Emotion wheel (v2.0) - shown when toggle is on
    if (emotionWheel_ && emotionWheel_->isVisible()) {
        bounds.removeFromTop(10);
        auto wheelArea = bounds.removeFromTop(300);
        emotionWheel_->setBounds(wheelArea.reduced(10, 0));
    }
    
    // Status area (leave space for resize handle)
    bounds.removeFromBottom(RESIZE_HANDLE_HEIGHT);
    bounds.removeFromTop(15);
    emotionDisplay_.setBounds(bounds.removeFromTop(25));
    statusLabel_.setBounds(bounds.removeFromTop(20));
    
    // Cassette view (v2.0) - wraps entire content if enabled
    if (cassetteView_ && cassetteView_->isVisible()) {
        cassetteView_->setBounds(getLocalBounds());
        // Set main content as child of cassette view
        auto contentArea = getLocalBounds();
        contentArea.removeFromTop(60);  // Header
        contentArea.removeFromBottom(RESIZE_HANDLE_HEIGHT);
        // Note: In a full implementation, we'd move all controls into cassette view
    }
}

void PluginEditor::mouseDown(const juce::MouseEvent& e) {
    auto resizeArea = getLocalBounds().removeFromBottom(RESIZE_HANDLE_HEIGHT);
    if (resizeArea.contains(e.getPosition())) {
        isResizing_ = true;
        resizeStartPos_ = e.getScreenPosition();
        resizeStartSize_ = juce::Point<int>(getWidth(), getHeight());
    }
}

void PluginEditor::mouseDrag(const juce::MouseEvent& e) {
    if (isResizing_) {
        auto delta = e.getScreenPosition() - resizeStartPos_;
        int newHeight = resizeStartSize_.y + delta.y;
        newHeight = juce::jmax(MIN_HEIGHT, newHeight);
        setSize(getWidth(), newHeight);
    }
}

void PluginEditor::mouseUp(const juce::MouseEvent&) {
    isResizing_ = false;
}

void PluginEditor::onGenerate() {
    float valence = static_cast<float>(valenceSlider_.getValue());
    float arousal = static_cast<float>(arousalSlider_.getValue());
    float intensity = static_cast<float>(intensitySlider_.getValue());
    
    // Get fine-tuning parameters
    float complexity = static_cast<float>(complexitySlider_.getValue());
    float humanize = static_cast<float>(*processor_.parameters.getRawParameterValue("humanize"));
    float feel = static_cast<float>(feelSlider_.getValue());
    float dynamics = static_cast<float>(dynamicsSlider_.getValue());
    int bars = static_cast<int>(barsSlider_.getValue());
    
    // Update processor parameters to match sliders
    *processor_.parameters.getRawParameterValue("valence") = valence;
    *processor_.parameters.getRawParameterValue("arousal") = arousal;
    *processor_.parameters.getRawParameterValue("intensity") = intensity;
    *processor_.parameters.getRawParameterValue("complexity") = complexity;
    *processor_.parameters.getRawParameterValue("feel") = feel;
    *processor_.parameters.getRawParameterValue("dynamics") = dynamics;
    *processor_.parameters.getRawParameterValue("bars") = static_cast<float>(bars);
    
    // Find nearest emotion from thesaurus based on slider values
    auto emotion = processor_.getIntentPipeline().thesaurus().findNearest(valence, arousal, intensity);
    
    // Generate MIDI using the emotion with all fine-tuning parameters
    processor_.generateFromWound(emotion.name, intensity);
    processor_.triggerImmediateSend();
    
    playStopButton_.setEnabled(true);
    exportButton_.setEnabled(true);
    
    // Export to cache directory
    auto cacheDir = processor_.getCacheDirectory();
    auto timestamp = juce::Time::getCurrentTime().formatted("%Y%m%d_%H%M%S");
    auto midiFile = cacheDir.getChildFile("Kelly_MIDI_" + timestamp + ".mid");
    
    if (processor_.exportMidiToFile(midiFile)) {
        statusLabel_.setText("Generated! MIDI saved to cache. Drag into Logic or it will be deleted on next session.", 
                             juce::dontSendNotification);
    } else {
        statusLabel_.setText("Generated! Press PLAY in Logic to hear it.", 
                             juce::dontSendNotification);
    }
}

void PluginEditor::onPlayStop() {
    if (processor_.isPlaying()) {
        processor_.stopPlayback();
        statusLabel_.setText("Stopped.", juce::dontSendNotification);
    } else {
        processor_.startPlayback();
        statusLabel_.setText("Playing - press PLAY in Logic transport!", juce::dontSendNotification);
    }
    updatePlayStopButton();
}

void PluginEditor::onExport() {
    auto chooser = std::make_shared<juce::FileChooser>("Save MIDI File",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
        "*.mid");

    chooser->launchAsync(juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc) {
        auto file = fc.getResult();
        if (file != juce::File{}) {
            if (processor_.exportMidiToFile(file)) {
                statusLabel_.setText("Saved: " + file.getFileName(), juce::dontSendNotification);
            } else {
                statusLabel_.setText("Export failed!", juce::dontSendNotification);
            }
        }
    });
}

} // namespace kelly
