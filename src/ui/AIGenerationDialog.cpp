#include "AIGenerationDialog.h"
#include "KellyLookAndFeel.h"

namespace kelly {

AIGenerationDialog::AIGenerationDialog() {
    setOpaque(true);
    setSize(450, 420);  // Increased height for API key field
    
    // Number of tracks
    numTracksLabel_.setText("Number of Tracks", juce::dontSendNotification);
    numTracksLabel_.setFont(juce::FontOptions(12.0f));
    numTracksLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textSecondary);
    numTracksLabel_.setTooltip("How many MIDI tracks to generate. Each track will be unique based on variability setting.");
    addAndMakeVisible(numTracksLabel_);
    
    numTracksSlider_.setRange(1.0, 16.0, 1.0);
    numTracksSlider_.setValue(4.0);
    numTracksSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    numTracksSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    numTracksSlider_.setTooltip("Set how many different MIDI tracks to generate (1-16)");
    addAndMakeVisible(numTracksSlider_);
    
    // Use Side A (Theory)
    useSideALabel_.setText("Use A-Side (Music Theory)", juce::dontSendNotification);
    useSideALabel_.setFont(juce::FontOptions(12.0f));
    useSideALabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textSecondary);
    useSideALabel_.setTooltip("Include music theory settings (key, scale, tempo, instruments) from A-side");
    addAndMakeVisible(useSideALabel_);
    
    useSideAToggle_.setButtonText("Enabled");
    useSideAToggle_.setToggleState(true, juce::dontSendNotification);
    useSideAToggle_.setTooltip("Toggle to include/exclude music theory settings in AI generation");
    addAndMakeVisible(useSideAToggle_);
    
    // Use Side B (Emotion)
    useSideBLabel_.setText("Use B-Side (Emotion)", juce::dontSendNotification);
    useSideBLabel_.setFont(juce::FontOptions(12.0f));
    useSideBLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textSecondary);
    useSideBLabel_.setTooltip("Include emotion settings (valence, arousal, intensity) from B-side");
    addAndMakeVisible(useSideBLabel_);
    
    useSideBToggle_.setButtonText("Enabled");
    useSideBToggle_.setToggleState(true, juce::dontSendNotification);
    useSideBToggle_.setTooltip("Toggle to include/exclude emotion settings in AI generation");
    addAndMakeVisible(useSideBToggle_);
    
    // Blend sides
    blendSidesLabel_.setText("Blend Both Sides", juce::dontSendNotification);
    blendSidesLabel_.setFont(juce::FontOptions(12.0f));
    blendSidesLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textSecondary);
    blendSidesLabel_.setTooltip("When enabled, combines both A-side and B-side settings for richer generation");
    addAndMakeVisible(blendSidesLabel_);
    
    blendSidesToggle_.setButtonText("Enabled");
    blendSidesToggle_.setToggleState(true, juce::dontSendNotification);
    blendSidesToggle_.setTooltip("Enable to blend music theory and emotion settings together");
    addAndMakeVisible(blendSidesToggle_);
    
    // Variability
    variabilityLabel_.setText("Variability (Different Each Time)", juce::dontSendNotification);
    variabilityLabel_.setFont(juce::FontOptions(12.0f));
    variabilityLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textSecondary);
    variabilityLabel_.setTooltip("How different each generation will be: 0 = consistent, 1 = very variable");
    addAndMakeVisible(variabilityLabel_);
    
    variabilitySlider_.setRange(0.0, 1.0, 0.01);
    variabilitySlider_.setValue(0.5);
    variabilitySlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    variabilitySlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    variabilitySlider_.setTooltip("Control randomness: lower = similar outputs, higher = more variation each time");
    addAndMakeVisible(variabilitySlider_);
    
    // Bars per track
    barsPerTrackLabel_.setText("Bars Per Track", juce::dontSendNotification);
    barsPerTrackLabel_.setFont(juce::FontOptions(12.0f));
    barsPerTrackLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textSecondary);
    barsPerTrackLabel_.setTooltip("Length of each generated track in bars (4-32 bars)");
    addAndMakeVisible(barsPerTrackLabel_);
    
    barsPerTrackSlider_.setRange(4.0, 32.0, 1.0);
    barsPerTrackSlider_.setValue(8.0);
    barsPerTrackSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    barsPerTrackSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    barsPerTrackSlider_.setTooltip("Set the length for each generated track");
    addAndMakeVisible(barsPerTrackSlider_);
    
    // API Key input
    apiKeyLabel_.setText("LLM API Key (Optional)", juce::dontSendNotification);
    apiKeyLabel_.setFont(juce::FontOptions(12.0f));
    addAndMakeVisible(apiKeyLabel_);
    
    apiKeyEditor_.setTextToShowWhenEmpty("Enter API key for AI learning...", juce::Colours::grey);
    apiKeyEditor_.setPasswordCharacter('*');
    apiKeyEditor_.setFont(juce::FontOptions(11.0f));
    apiKeyEditor_.setTooltip("Optional: Enter your LLM API key (OpenAI, Anthropic, etc.) to enable AI learning features");
    addAndMakeVisible(apiKeyEditor_);
    
    saveApiKeyLabel_.setText("Save API Key", juce::dontSendNotification);
    saveApiKeyLabel_.setFont(juce::FontOptions(11.0f));
    saveApiKeyLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textSecondary);
    saveApiKeyLabel_.setTooltip("Save API key locally for future use (stored securely)");
    addAndMakeVisible(saveApiKeyLabel_);
    
    saveApiKeyToggle_.setButtonText("Save");
    saveApiKeyToggle_.setToggleState(false, juce::dontSendNotification);
    saveApiKeyToggle_.setTooltip("Enable to save your API key for future sessions");
    addAndMakeVisible(saveApiKeyToggle_);
    
    // Buttons - Use documented colors
    generateButton_.setButtonText("GENERATE");
    generateButton_.setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF3B82F6));  // Focus Blue
    generateButton_.setTooltip("Generate AI MIDI tracks based on current settings");
    generateButton_.onClick = [this] {
        wasCancelled_ = false;
        modalResult_ = 1;
        cachedRequest_ = getRequest();  // Cache the request before closing
        // Find and close DialogWindow parent
        juce::Component* comp = this;
        while (comp != nullptr) {
            if (auto* dw = dynamic_cast<juce::DialogWindow*>(comp)) {
                dw->exitModalState(1);
                break;
            }
            comp = comp->getParentComponent();
        }
    };
    addAndMakeVisible(generateButton_);
    
    cancelButton_.setButtonText("CANCEL");
    cancelButton_.setColour(juce::TextButton::buttonColourId, KellyLookAndFeel::surfaceColor);
    cancelButton_.setTooltip("Cancel AI generation");
    cancelButton_.onClick = [this] {
        wasCancelled_ = true;
        modalResult_ = 0;
        // Find and close DialogWindow parent
        juce::Component* comp = this;
        while (comp != nullptr) {
            if (auto* dw = dynamic_cast<juce::DialogWindow*>(comp)) {
                dw->exitModalState(0);
                break;
            }
            comp = comp->getParentComponent();
        }
    };
    addAndMakeVisible(cancelButton_);
}

AIGenerationDialog::AIGenerationRequest AIGenerationDialog::showDialog(juce::Component* parent) {
    (void)parent;  // Suppress unused parameter warning
    juce::DialogWindow::LaunchOptions options;
    auto dialog = std::make_unique<AIGenerationDialog>();
    auto* dialogPtr = dialog.get();
    
    options.content.setOwned(dialog.release());
    options.content->setSize(450, 420);
    options.dialogTitle = "AI MIDI Generation";
    options.dialogBackgroundColour = juce::Colour(0xFF2D2D2D);
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = false;
    options.resizable = false;
    
    auto* dw = options.launchAsync();
    if (dw == nullptr) {
        return AIGenerationRequest{};
    }
    
    // Use modal state - the dialog will call exitModalState when buttons are clicked
    dw->enterModalState(true, nullptr, true);
    
    // After modal state exits, get the result from cached request
    // The dialog caches the request when Generate is clicked
    if (dialogPtr && !dialogPtr->wasCancelled_) {
        return dialogPtr->cachedRequest_;
    }
    
    return AIGenerationRequest{};
}

AIGenerationDialog::AIGenerationRequest AIGenerationDialog::getRequest() const {
    AIGenerationRequest request;
    request.numTracks = static_cast<int>(numTracksSlider_.getValue());
    request.useSideA = useSideAToggle_.getToggleState();
    request.useSideB = useSideBToggle_.getToggleState();
    request.variability = static_cast<float>(variabilitySlider_.getValue());
    request.barsPerTrack = static_cast<int>(barsPerTrackSlider_.getValue());
    request.blendSides = blendSidesToggle_.getToggleState();
    request.apiKey = apiKeyEditor_.getText().trim();
    return request;
}

void AIGenerationDialog::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();
    
    // Modern gradient background
    juce::ColourGradient gradient(
        KellyLookAndFeel::backgroundDark, bounds.getTopLeft(),
        KellyLookAndFeel::backgroundLight, bounds.getBottomLeft(),
        false
    );
    g.setGradientFill(gradient);
    g.fillAll();
    
    // Header with accent - Using exact documented colors
    auto headerBounds = bounds.removeFromTop(45.0f);
    juce::ColourGradient headerGradient(
        juce::Colour(0xFF3B82F6).withAlpha(0.2f), headerBounds.getTopLeft(),  // Focus Blue
        juce::Colour(0xFFA855F7).withAlpha(0.15f), headerBounds.getBottomLeft(),  // Creative Purple
        false
    );
    g.setGradientFill(headerGradient);
    g.fillRoundedRectangle(headerBounds, 0.0f);
    
    // Title with modern typography
    g.setColour(KellyLookAndFeel::textPrimary);
    g.setFont(juce::FontOptions(20.0f).withStyle("Bold"));
    g.drawText("AI MIDI Generation", headerBounds, juce::Justification::centred);
    
    // Accent line - Balance Green
    g.setColour(juce::Colour(0xFF22C55E).withAlpha(0.6f));  // Balance Green #22C55E
    auto accentLine = juce::Rectangle<float>(0.0f, headerBounds.getBottom() - 2.0f, headerBounds.getWidth(), 2.0f);
    g.fillRect(accentLine);
}

void AIGenerationDialog::resized() {
    auto bounds = getLocalBounds().reduced(20);
    bounds.removeFromTop(45);  // Title space
    
    // Number of tracks
    auto trackArea = bounds.removeFromTop(40);
    numTracksLabel_.setBounds(trackArea.removeFromLeft(150).removeFromTop(18));
    numTracksSlider_.setBounds(trackArea.removeFromTop(25));
    
    bounds.removeFromTop(10);
    
    // Side toggles
    auto sideArea = bounds.removeFromTop(30);
    useSideALabel_.setBounds(sideArea.removeFromLeft(180).removeFromTop(18));
    useSideAToggle_.setBounds(sideArea.removeFromLeft(80).removeFromTop(20));
    
    bounds.removeFromTop(5);
    sideArea = bounds.removeFromTop(30);
    useSideBLabel_.setBounds(sideArea.removeFromLeft(180).removeFromTop(18));
    useSideBToggle_.setBounds(sideArea.removeFromLeft(80).removeFromTop(20));
    
    bounds.removeFromTop(5);
    sideArea = bounds.removeFromTop(30);
    blendSidesLabel_.setBounds(sideArea.removeFromLeft(180).removeFromTop(18));
    blendSidesToggle_.setBounds(sideArea.removeFromLeft(80).removeFromTop(20));
    
    bounds.removeFromTop(15);
    
    // Variability
    auto varArea = bounds.removeFromTop(40);
    variabilityLabel_.setBounds(varArea.removeFromLeft(200).removeFromTop(18));
    variabilitySlider_.setBounds(varArea.removeFromTop(25));
    
    bounds.removeFromTop(10);
    
    // Bars per track
    auto barsArea = bounds.removeFromTop(40);
    barsPerTrackLabel_.setBounds(barsArea.removeFromLeft(150).removeFromTop(18));
    barsPerTrackSlider_.setBounds(barsArea.removeFromTop(25));
    
    bounds.removeFromTop(10);
    
    // API Key
    auto apiKeyArea = bounds.removeFromTop(50);
    apiKeyLabel_.setBounds(apiKeyArea.removeFromTop(18));
    apiKeyEditor_.setBounds(apiKeyArea.removeFromTop(25));
    
    auto saveKeyArea = apiKeyArea.removeFromTop(20);
    saveApiKeyLabel_.setBounds(saveKeyArea.removeFromLeft(100).removeFromTop(18));
    saveApiKeyToggle_.setBounds(saveKeyArea.removeFromLeft(60).removeFromTop(18));
    
    bounds.removeFromTop(10);
    
    // Buttons
    auto buttonArea = bounds.removeFromTop(35);
    int buttonWidth = buttonArea.getWidth() / 2;
    generateButton_.setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(5));
    cancelButton_.setBounds(buttonArea.reduced(5));
}

} // namespace kelly
