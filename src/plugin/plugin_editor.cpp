#include "plugin_editor.h"

namespace kelly {

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p), processor(p) {
    
    // Create menu bar
    menuBar_ = std::make_unique<juce::MenuBarComponent>(this);
    addAndMakeVisible(menuBar_.get());
    
    // Title
    titleLabel_.setText("Kelly Emotion Processor", juce::dontSendNotification);
    titleLabel_.setFont(juce::Font(24.0f, juce::Font::bold));
    titleLabel_.setJustificationType(juce::Justification::centred);
    titleLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(titleLabel_);
    
    // Project name label
    projectNameLabel_.setFont(juce::Font(12.0f));
    projectNameLabel_.setJustificationType(juce::Justification::centred);
    projectNameLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(projectNameLabel_);
    updateProjectLabel();
    
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
    emotionAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        processor.getParameters(), ParameterID::emotionTag, emotionComboBox_);
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
    
    // ML Mode toggle for A/B testing
    mlModeToggle_.setButtonText("ML Enhanced Mode");
    mlModeToggle_.setTooltip("Toggle between rule-based and ML-enhanced generation for A/B comparison");
    mlModeToggle_.onClick = [this] {
        bool mlEnabled = mlModeToggle_.getToggleState();
        mlStatusLabel_.setText(mlEnabled ? "ML Mode: Active" : "ML Mode: Off (Rule-Based)", 
                               juce::dontSendNotification);
        mlStatusLabel_.setColour(juce::Label::textColourId, 
                                mlEnabled ? juce::Colours::cyan : juce::Colours::grey);
    };
    addAndMakeVisible(mlModeToggle_);
    
    // ML Status label
    mlStatusLabel_.setText("ML Mode: Off (Rule-Based)", juce::dontSendNotification);
    mlStatusLabel_.setFont(juce::Font(11.0f));
    mlStatusLabel_.setJustificationType(juce::Justification::centredLeft);
    mlStatusLabel_.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(mlStatusLabel_);
    
    // Save button
    saveButton_.setButtonText("Save Project");
    saveButton_.onClick = [this] { saveProject(); };
    addAndMakeVisible(saveButton_);
    
    // Export button
    exportButton_.setButtonText("Export MIDI");
    exportButton_.onClick = [this] { exportMidi(); };
    addAndMakeVisible(exportButton_);
    
    // Status label
    statusLabel_.setText("Ready", juce::dontSendNotification);
    statusLabel_.setJustificationType(juce::Justification::centred);
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    addAndMakeVisible(statusLabel_);
    
    // Update initial values
    updateValueLabels();
    
    // Start timer for periodic updates
    startTimer(100); // Update every 100ms
    
    setSize(500, 620);
}

PluginEditor::~PluginEditor() {
    stopTimer();
    setLookAndFeel(nullptr);
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
    auto bounds = getLocalBounds();
    
    // Menu bar at top
    menuBar_->setBounds(bounds.removeFromTop(25));
    
    bounds = bounds.reduced(20);
    
    // Title at top
    titleLabel_.setBounds(bounds.removeFromTop(40));
    
    // Project name
    projectNameLabel_.setBounds(bounds.removeFromTop(20));
    bounds.removeFromTop(10);
    
    // Emotion selector
    auto emotionBounds = bounds.removeFromTop(60);
    emotionLabel_.setBounds(emotionBounds.removeFromLeft(100));
    emotionComboBox_.setBounds(emotionBounds.reduced(5));
    bounds.removeFromTop(10);
    
    // Valence slider
    auto valenceBounds = bounds.removeFromTop(50);
    valenceLabel_.setBounds(valenceBounds.removeFromLeft(100));
    auto sliderBounds = valenceBounds.removeFromLeft(valenceBounds.getWidth() - 60);
    valenceSlider_.setBounds(sliderBounds.reduced(5));
    valenceValueLabel_.setBounds(valenceBounds.reduced(5));
    bounds.removeFromTop(5);
    
    // Arousal slider
    auto arousalBounds = bounds.removeFromTop(50);
    arousalLabel_.setBounds(arousalBounds.removeFromLeft(100));
    sliderBounds = arousalBounds.removeFromLeft(arousalBounds.getWidth() - 60);
    arousalSlider_.setBounds(sliderBounds.reduced(5));
    arousalValueLabel_.setBounds(arousalBounds.reduced(5));
    bounds.removeFromTop(5);
    
    // Intensity slider
    auto intensityBounds = bounds.removeFromTop(50);
    intensityLabel_.setBounds(intensityBounds.removeFromLeft(100));
    sliderBounds = intensityBounds.removeFromLeft(intensityBounds.getWidth() - 60);
    intensitySlider_.setBounds(sliderBounds.reduced(5));
    intensityValueLabel_.setBounds(intensityBounds.reduced(5));
    bounds.removeFromTop(5);
    
    // Bars slider
    auto barsBounds = bounds.removeFromTop(50);
    barsLabel_.setBounds(barsBounds.removeFromLeft(100));
    sliderBounds = barsBounds.removeFromLeft(barsBounds.getWidth() - 60);
    barsSlider_.setBounds(sliderBounds.reduced(5));
    barsValueLabel_.setBounds(barsBounds.reduced(5));
    bounds.removeFromTop(5);
    
    // Generation rate slider
    auto genRateBounds = bounds.removeFromTop(50);
    generationRateLabel_.setBounds(genRateBounds.removeFromLeft(100));
    sliderBounds = genRateBounds.removeFromLeft(genRateBounds.getWidth() - 60);
    generationRateSlider_.setBounds(sliderBounds.reduced(5));
    generationRateValueLabel_.setBounds(genRateBounds.reduced(5));
    bounds.removeFromTop(10);
    
    // Cloud toggle
    cloudToggle_.setBounds(bounds.removeFromTop(30));
    bounds.removeFromTop(5);
    
    // ML Mode toggle for A/B testing
    auto mlToggleBounds = bounds.removeFromTop(25);
    mlModeToggle_.setBounds(mlToggleBounds.removeFromLeft(160));
    mlStatusLabel_.setBounds(mlToggleBounds);
    bounds.removeFromTop(10);
    
    // Action buttons
    auto buttonBounds = bounds.removeFromTop(35);
    saveButton_.setBounds(buttonBounds.removeFromLeft(buttonBounds.getWidth() / 2 - 5));
    buttonBounds.removeFromLeft(10);
    exportButton_.setBounds(buttonBounds);
    bounds.removeFromTop(10);
    
    // Status label at bottom
    statusLabel_.setBounds(bounds.removeFromBottom(30));
}

void PluginEditor::timerCallback() {
    updateValueLabels();
    updateProjectLabel();
    
    // Update status
    if (processor.isCloudEnabled()) {
        statusLabel_.setText("Cloud enabled", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::orange);
    } else {
        if (processor.hasMidiData()) {
            statusLabel_.setText("MIDI data ready for export", juce::dontSendNotification);
            statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightblue);
        } else {
            statusLabel_.setText("Local processing", juce::dontSendNotification);
            statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        }
    }
}

//==============================================================================
// MenuBarModel Implementation
//==============================================================================

juce::StringArray PluginEditor::getMenuBarNames() {
    return {"File", "Export"};
}

juce::PopupMenu PluginEditor::getMenuForIndex(int topLevelMenuIndex, const juce::String& menuName) {
    juce::PopupMenu menu;
    
    if (topLevelMenuIndex == 0) {  // File menu
        menu.addItem(cmdNewProject, "New Project");
        menu.addItem(cmdOpenProject, "Open Project...");
        menu.addSeparator();
        menu.addItem(cmdSaveProject, "Save Project");
        menu.addItem(cmdSaveProjectAs, "Save Project As...");
    }
    else if (topLevelMenuIndex == 1) {  // Export menu
        menu.addItem(cmdExportMidi, "Export MIDI...");
        menu.addItem(cmdExportMidiAs, "Export MIDI with Options...");
    }
    
    return menu;
}

void PluginEditor::menuItemSelected(int menuItemID, int topLevelMenuIndex) {
    switch (menuItemID) {
        case cmdNewProject:
            newProject();
            break;
        case cmdOpenProject:
            openProject();
            break;
        case cmdSaveProject:
            saveProject();
            break;
        case cmdSaveProjectAs:
            saveProjectAs();
            break;
        case cmdExportMidi:
            exportMidi();
            break;
        case cmdExportMidiAs:
            exportMidiWithOptions();
            break;
        default:
            break;
    }
}

//==============================================================================
// Project Management
//==============================================================================

void PluginEditor::newProject() {
    if (!checkUnsavedChanges()) return;
    
    processor.createNewProject();
    statusLabel_.setText("New project created", juce::dontSendNotification);
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    updateProjectLabel();
}

void PluginEditor::openProject() {
    if (!checkUnsavedChanges()) return;
    
    auto chooser = std::make_shared<juce::FileChooser>(
        "Open Project",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
        midikompanion::ProjectManager::getFileFilter()
    );
    
    chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc) {
            auto file = fc.getResult();
            if (file != juce::File()) {
                if (processor.loadProject(file)) {
                    statusLabel_.setText("Project loaded", juce::dontSendNotification);
                    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
                } else {
                    statusLabel_.setText("Failed to load project", juce::dontSendNotification);
                    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::red);
                }
                updateProjectLabel();
            }
        });
}

void PluginEditor::saveProject() {
    auto currentFile = processor.getCurrentProjectFile();
    
    if (currentFile == juce::File()) {
        saveProjectAs();
        return;
    }
    
    if (processor.saveCurrentProject(currentFile)) {
        statusLabel_.setText("Project saved", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    } else {
        statusLabel_.setText("Failed to save project", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::red);
    }
    updateProjectLabel();
}

void PluginEditor::saveProjectAs() {
    auto chooser = std::make_shared<juce::FileChooser>(
        "Save Project As",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
        midikompanion::ProjectManager::getFileFilter()
    );
    
    chooser->launchAsync(juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc) {
            auto file = fc.getResult();
            if (file != juce::File()) {
                // Ensure correct extension
                if (!file.hasFileExtension(midikompanion::ProjectManager::getFileExtension())) {
                    file = file.withFileExtension(midikompanion::ProjectManager::getFileExtension());
                }
                
                if (processor.saveCurrentProject(file)) {
                    statusLabel_.setText("Project saved", juce::dontSendNotification);
                    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
                } else {
                    statusLabel_.setText("Failed to save project", juce::dontSendNotification);
                    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::red);
                }
                updateProjectLabel();
            }
        });
}

void PluginEditor::exportMidi() {
    if (!processor.hasMidiData()) {
        statusLabel_.setText("No MIDI data to export", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::orange);
        return;
    }
    
    auto chooser = std::make_shared<juce::FileChooser>(
        "Export MIDI",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
        midikompanion::MidiExporter::getFileFilter()
    );
    
    chooser->launchAsync(juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc) {
            auto file = fc.getResult();
            if (file != juce::File()) {
                // Ensure correct extension
                if (!file.hasFileExtension(".mid") && !file.hasFileExtension(".midi")) {
                    file = file.withFileExtension(".mid");
                }
                
                midikompanion::MidiExportOptions options;
                options.tempo = processor.getCurrentTempo();
                
                auto result = processor.exportToMidi(file, options);
                
                if (result.success) {
                    statusLabel_.setText("MIDI exported: " + juce::String(result.totalNotes) + " notes",
                                         juce::dontSendNotification);
                    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
                } else {
                    statusLabel_.setText("Export failed: " + result.errorMessage,
                                         juce::dontSendNotification);
                    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::red);
                }
            }
        });
}

void PluginEditor::exportMidiWithOptions() {
    // For now, just call the regular export
    // In the future, this could show an options dialog
    exportMidi();
}

bool PluginEditor::checkUnsavedChanges() {
    if (!processor.hasUnsavedChanges()) {
        return true;
    }
    
    auto result = juce::AlertWindow::showYesNoCancelBox(
        juce::AlertWindow::QuestionIcon,
        "Unsaved Changes",
        "You have unsaved changes. Do you want to save before continuing?",
        "Save",
        "Don't Save",
        "Cancel"
    );
    
    if (result == 1) {  // Save
        saveProject();
        return true;
    }
    else if (result == 2) {  // Don't Save
        return true;
    }
    else {  // Cancel
        return false;
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

void PluginEditor::updateProjectLabel() {
    auto currentFile = processor.getCurrentProjectFile();
    juce::String projectName;
    
    if (currentFile == juce::File()) {
        projectName = "Untitled Project";
    } else {
        projectName = currentFile.getFileNameWithoutExtension();
    }
    
    if (processor.hasUnsavedChanges()) {
        projectName += " *";
    }
    
    projectNameLabel_.setText(projectName, juce::dontSendNotification);
}

} // namespace kelly
