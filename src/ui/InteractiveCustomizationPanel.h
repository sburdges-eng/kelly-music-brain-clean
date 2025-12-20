#pragma once
/*
 * InteractiveCustomizationPanel.h - Unified Customization Interface
 * ================================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - UI Layer: EmotionWorkstation (main workstation component)
 * - UI Layer: MidiEditor (MIDI editing), NaturalLanguageEditor (natural language feedback)
 * - Type System: Types.h (GeneratedMidi via KellyTypes.h)
 * - UI Layer: Unified interface combining all customization components
 *
 * Purpose: Unified interface combining all customization components.
 *          Integrates preference visualization, suggestions, parameter morphing,
 *          MIDI editing, and natural language feedback.
 *
 * Features:
 * - User preference visualization
 * - Intelligent suggestions overlay
 * - Real-time parameter morphing
 * - MIDI/Sheet music editing
 * - Natural language feedback
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include "EmotionWorkstation.h"  // Main workstation
#include "MidiEditor.h"  // MIDI editing
#include "NaturalLanguageEditor.h"  // Natural language feedback
#include "../common/Types.h"  // GeneratedMidi (via KellyTypes.h)
#include <memory>

namespace kelly {

/**
 * InteractiveCustomizationPanel - Unified interface combining all customization components
 *
 * Integrates:
 * - User preference visualization
 * - Intelligent suggestions overlay
 * - Real-time parameter morphing
 * - MIDI/Sheet music editing
 * - Natural language feedback
 */
class InteractiveCustomizationPanel : public juce::Component {
public:
    explicit InteractiveCustomizationPanel(EmotionWorkstation& workstation);
    ~InteractiveCustomizationPanel() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Access to components
    MidiEditor& getMidiEditor() { return midiEditor_; }
    NaturalLanguageEditor& getNaturalLanguageEditor() { return naturalLanguageEditor_; }
    EmotionWorkstation& getWorkstation() { return workstation_; }

    // Preference visualization
    void showPreferences(bool show) { showPreferences_ = show; repaint(); }
    void showSuggestions(bool show) { showSuggestions_ = show; repaint(); }

    // Callbacks
    std::function<void(const GeneratedMidi&)> onMidiEdited;
    std::function<void(const juce::String& feedback)> onNaturalLanguageFeedback;

private:
    EmotionWorkstation& workstation_;

    // Editing components
    MidiEditor midiEditor_;
    NaturalLanguageEditor naturalLanguageEditor_;

    // UI state
    bool showPreferences_ = true;
    bool showSuggestions_ = true;

    // Layout
    enum class ViewMode {
        Parameters,      // Show parameter controls
        Editing,         // Show MIDI editor
        NaturalLanguage, // Show natural language input
        Combined         // Show all
    };
    ViewMode viewMode_ = ViewMode::Combined;

    // Helper methods
    void setupComponents();
    void drawPreferenceOverlay(juce::Graphics& g);
    void drawSuggestionOverlay(juce::Graphics& g);
};

} // namespace kelly
