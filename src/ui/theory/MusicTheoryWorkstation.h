#pragma once
/*
 * MusicTheoryWorkstation.h - Main Music Theory Learning Interface
 * ==============================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: MusicTheoryBrain (from src/music_theory/MusicTheoryBrain.h)
 * - UI Layer: Used by EmotionWorkstation (as new tab)
 * - UI Components: ConceptBrowser, LearningPanel, VirtualKeyboard
 * - Styling: midikompanion::MidiKompanionLookAndFeel
 *
 * Purpose: Main container for music theory learning interface with tabbed
 *          interface for Learning, Analysis, and Practice modes.
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include "../../music_theory/MusicTheoryBrain.h"
#include "../../music_theory/Types.h"
#include "ConceptBrowser.h"
#include "LearningPanel.h"
#include "VirtualKeyboard.h"
#include "../MidiKompanionLookAndFeel.h"
#include <memory>

namespace kelly {

/**
 * MusicTheoryWorkstation - Main music theory learning interface
 *
 * Features:
 * - Tabbed interface: Learning, Analysis, Practice
 * - Connects to MusicTheoryBrain for all theory operations
 * - Interactive concept browser
 * - Multi-style explanations
 * - MIDI analysis and visualization
 * - Practice session management
 */
class MusicTheoryWorkstation : public juce::Component {
public:
    explicit MusicTheoryWorkstation(midikompanion::theory::MusicTheoryBrain* brain);
    ~MusicTheoryWorkstation() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    /**
     * Analyze MIDI file and display results
     */
    void analyzeMIDI(const juce::MidiFile& midi);

    /**
     * Show concept in learning panel
     */
    void showConcept(const std::string& conceptName);

    /**
     * Start practice session
     */
    void startPracticeSession();

    /**
     * Display explanation with specified style
     */
    void displayExplanation(const std::string& text, midikompanion::theory::ExplanationType style);

    /**
     * Get access to sub-components
     */
    ConceptBrowser& getConceptBrowser() { return conceptBrowser_; }
    LearningPanel& getLearningPanel() { return learningPanel_; }
    VirtualKeyboard& getVirtualKeyboard() { return virtualKeyboard_; }

    /**
     * Set MusicTheoryBrain instance
     */
    void setMusicTheoryBrain(midikompanion::theory::MusicTheoryBrain* brain);

    // Callbacks
    std::function<void(const std::string& conceptName)> onConceptSelected;
    std::function<void(const juce::MidiFile& midi)> onMIDIAnalyzed;

private:
    // Non-owning pointer - lifetime managed by parent
    midikompanion::theory::MusicTheoryBrain* brain_ = nullptr;

    // Tabbed interface
    juce::TabbedComponent tabs_{juce::TabbedButtonBar::TabsAtTop};

    // Sub-components
    ConceptBrowser conceptBrowser_;
    LearningPanel learningPanel_;
    VirtualKeyboard virtualKeyboard_;

    // Analysis panel (for MIDI analysis results)
    juce::TextEditor analysisDisplay_;
    juce::Label analysisLabel_{"", "Analysis Results"};

    // Practice panel (for practice sessions)
    juce::TextEditor practiceDisplay_;
    juce::Label practiceLabel_{"", "Practice Session"};

    // Styling
    midikompanion::MidiKompanionLookAndFeel lookAndFeel_;

    void setupTabs();
    void setupAnalysisPanel();
    void setupPracticePanel();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MusicTheoryWorkstation)
};

} // namespace kelly
