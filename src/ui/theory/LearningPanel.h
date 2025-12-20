#pragma once
/*
 * LearningPanel.h - Music Theory Learning Panel
 * ============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: MusicTheoryBrain, KnowledgeGraph
 * - UI Layer: Used by MusicTheoryWorkstation
 * - UI Components: VirtualKeyboard (for interactive examples)
 *
 * Purpose: Displays lesson content with multiple explanation styles,
 *          interactive examples, and progress tracking.
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include "../../music_theory/MusicTheoryBrain.h"
#include "../../music_theory/Types.h"
#include <memory>

namespace kelly {

/**
 * LearningPanel - Interactive learning interface
 *
 * Features:
 * - Multiple explanation styles (Intuitive, Mathematical, Historical)
 * - Interactive examples with MIDI playback
 * - Progress tracking visualization
 * - Exercise integration
 */
class LearningPanel : public juce::Component {
public:
    explicit LearningPanel(midikompanion::theory::MusicTheoryBrain* brain);
    ~LearningPanel() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    /**
     * Display concept with explanations
     */
    void displayConcept(const std::string& conceptName);

    /**
     * Display explanation
     */
    void displayExplanation(const std::string& text, midikompanion::theory::ExplanationType style);

    /**
     * Set MusicTheoryBrain instance
     */
    void setMusicTheoryBrain(midikompanion::theory::MusicTheoryBrain* brain);

    /**
     * Set explanation style preference
     */
    void setExplanationStyle(midikompanion::theory::ExplanationType style);

private:
    // Non-owning pointer
    midikompanion::theory::MusicTheoryBrain* brain_ = nullptr;

    // UI Components
    juce::Label conceptTitle_;
    juce::TextEditor explanationDisplay_;
    juce::ComboBox styleSelector_;
    juce::Label styleLabel_{"", "Explanation Style"};
    juce::TextButton playExampleButton_{"Play Example"};
    juce::TextButton nextExerciseButton_{"Next Exercise"};

    // Current state
    std::string currentConcept_;
    midikompanion::theory::ExplanationType currentStyle_ =
        midikompanion::theory::ExplanationType::Intuitive;

    void setupComponents();
    void loadExplanation(const std::string& conceptName);
    void updateExplanationDisplay();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LearningPanel)
};

} // namespace kelly
