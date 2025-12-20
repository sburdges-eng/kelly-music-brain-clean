#pragma once
/*
 * WorkflowManager.h - Multi-Mode Workflow Interface Manager
 * ==========================================================
 *
 * Manages three professional workflow modes:
 *
 * 1. MUSICIAN MODE (Command Panel)
 *    - Natural language commands
 *    - Music theory terminology
 *    - For educated musicians who know theory
 *
 * 2. COMPOSER MODE (Score Entry)
 *    - Sheet music / notation interface
 *    - Staff-based entry
 *    - For composers who think in notation
 *
 * 3. ENGINEER MODE (Mixer Console)
 *    - DAW-style mixing console
 *    - Channel strips and faders
 *    - For audio engineers and DAW users
 *
 * Users can switch between modes freely, with all modes
 * sharing the same underlying MIDI data.
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include "MusicianCommandPanel.h"
#include "ScoreEntryPanel.h"
#include "MixerConsolePanel.h"
#include "../music_theory/MusicTheoryBrain.h"
#include <memory>

namespace midikompanion {

//==============================================================================
// Workflow Mode Enum
//==============================================================================

enum class WorkflowMode {
    Musician,    // Command-based (natural language)
    Composer,    // Notation-based (sheet music)
    Engineer,    // Mixer-based (DAW console)
    Hybrid       // Show multiple panels simultaneously
};

//==============================================================================
// Workflow Manager
//==============================================================================

class WorkflowManager : public juce::Component
{
public:
    WorkflowManager();
    ~WorkflowManager() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    //==========================================================================
    // Mode Switching
    //==========================================================================

    /**
     * Switch to different workflow mode
     */
    void setWorkflowMode(WorkflowMode mode);
    WorkflowMode getCurrentMode() const { return currentMode_; }

    /**
     * Get active panel for current mode
     */
    juce::Component* getActivePanel();

    //==========================================================================
    // Direct Panel Access
    //==========================================================================

    MusicianCommandPanel* getMusicianPanel() { return musicianPanel_.get(); }
    ScoreEntryPanel* getComposerPanel() { return composerPanel_.get(); }
    MixerConsolePanel* getEngineerPanel() { return engineerPanel_.get(); }

    //==========================================================================
    // Shared Context (Synchronized Across All Modes)
    //==========================================================================

    /**
     * Set project context (shared by all modes)
     */
    void setProjectKey(const std::string& key);
    void setProjectTimeSignature(int numerator, int denominator);
    void setProjectTempo(float bpm);
    void setProjectTotalBars(int bars);

    /**
     * Get project context
     */
    std::string getProjectKey() const { return projectKey_; }
    int getTimeSignatureNumerator() const { return timeSignatureNumerator_; }
    int getTimeSignatureDenominator() const { return timeSignatureDenominator_; }
    float getProjectTempo() const { return projectTempo_; }
    int getProjectTotalBars() const { return projectTotalBars_; }

    //==========================================================================
    // MIDI Data (Shared Across All Modes)
    //==========================================================================

    /**
     * Get combined MIDI from all modes
     */
    juce::MidiBuffer getCombinedMIDI() const;

    /**
     * Set MIDI data (updates all modes)
     */
    void setMIDIData(const juce::MidiBuffer& buffer);

    /**
     * Clear all MIDI data
     */
    void clearAllMIDI();

    //==========================================================================
    // Quick Start Wizards
    //==========================================================================

    /**
     * New user? Show quick start wizard
     */
    void showQuickStartWizard();

    /**
     * Let user choose preferred workflow
     */
    struct WorkflowSuggestion {
        WorkflowMode mode;
        std::string title;
        std::string description;
        std::vector<std::string> bestFor;
    };

    std::vector<WorkflowSuggestion> getWorkflowSuggestions() const;

    //==========================================================================
    // Templates & Presets
    //==========================================================================

    /**
     * Load complete project template (sets up all modes)
     */
    struct ProjectTemplate {
        std::string name;
        std::string description;
        std::string genre;

        // Context
        std::string key;
        int numerator;
        int denominator;
        float tempo;
        int bars;

        // Mode-specific setup
        std::vector<std::string> commandPanelCommands;
        ScoreEntryPanel::ScoreTemplate scoreTemplate;
        MixerConsolePanel::MixerPreset mixerPreset;
    };

    void loadProjectTemplate(const ProjectTemplate& template_);
    std::vector<ProjectTemplate> getAvailableTemplates() const;

    //==========================================================================
    // Common Templates
    //==========================================================================

    void loadSongwriterTemplate();     // Simple: melody + chords
    void loadRockBandTemplate();       // Drums, bass, 2 guitars, vocals
    void loadOrchestralTemplate();     // Full orchestra sections
    void loadElectronicTemplate();     // Synths, drums, FX
    void loadJazzComboTemplate();      // Piano, bass, drums, horns
    void loadSoloPerformerTemplate();  // One instrument + backing

    //==========================================================================
    // Hybrid Mode (Show Multiple Panels)
    //==========================================================================

    /**
     * In hybrid mode, show multiple panels simultaneously
     */
    void setHybridLayout(const std::vector<WorkflowMode>& visibleModes);

    /**
     * Split screen between two modes
     */
    void setSplitView(WorkflowMode leftMode, WorkflowMode rightMode);

    //==========================================================================
    // Export/Import
    //==========================================================================

    /**
     * Export entire project (all modes)
     */
    bool exportProject(const juce::File& outputFile);

    /**
     * Import project (restores all modes)
     */
    bool importProject(const juce::File& inputFile);

    /**
     * Export MIDI file (combined from all modes)
     */
    bool exportMIDI(const juce::File& outputFile);

    //==========================================================================
    // Settings
    //==========================================================================

    /**
     * User preferences
     */
    struct UserPreferences {
        WorkflowMode defaultMode;
        bool showModeSelector;
        bool syncMIDIAcrossModes;
        bool showQuickStartOnLaunch;
        std::string preferredFont;
        float uiScale;
    };

    void setUserPreferences(const UserPreferences& prefs);
    UserPreferences getUserPreferences() const { return userPreferences_; }

    /**
     * Save/load preferences
     */
    bool savePreferencesToFile(const juce::File& file);
    bool loadPreferencesFromFile(const juce::File& file);

private:
    //==========================================================================
    // UI Components
    //==========================================================================

    // Mode selector (tabs or buttons)
    std::unique_ptr<juce::TabbedComponent> modeTabs_;

    // Alternative: Mode buttons
    std::unique_ptr<juce::TextButton> musicianModeButton_;
    std::unique_ptr<juce::TextButton> composerModeButton_;
    std::unique_ptr<juce::TextButton> engineerModeButton_;
    std::unique_ptr<juce::TextButton> hybridModeButton_;

    // Mode panels
    std::unique_ptr<MusicianCommandPanel> musicianPanel_;
    std::unique_ptr<ScoreEntryPanel> composerPanel_;
    std::unique_ptr<MixerConsolePanel> engineerPanel_;

    // Current active panel container
    std::unique_ptr<juce::Component> activePanelContainer_;

    // Quick start wizard
    std::unique_ptr<juce::Component> quickStartWizard_;

    // Project info display
    std::unique_ptr<juce::Label> keyLabel_;
    std::unique_ptr<juce::Label> timeSignatureLabel_;
    std::unique_ptr<juce::Label> tempoLabel_;

    //==========================================================================
    // Shared State
    //==========================================================================

    WorkflowMode currentMode_;
    std::vector<WorkflowMode> hybridModes_;

    // Project context (synchronized across all modes)
    std::string projectKey_;
    int timeSignatureNumerator_;
    int timeSignatureDenominator_;
    float projectTempo_;
    int projectTotalBars_;

    // User preferences
    UserPreferences userPreferences_;

    //==========================================================================
    // Music Theory Integration
    //==========================================================================

    std::shared_ptr<theory::MusicTheoryBrain> theoryBrain_;

    //==========================================================================
    // Templates
    //==========================================================================

    std::vector<ProjectTemplate> templates_;
    void initializeTemplates();

    //==========================================================================
    // Mode Synchronization
    //==========================================================================

    /**
     * When MIDI changes in one mode, update others
     */
    void synchronizeMIDIAcrossModes();

    /**
     * When context changes, update all modes
     */
    void synchronizeContextAcrossModes();

    //==========================================================================
    // Layout Management
    //==========================================================================

    void layoutSingleMode();
    void layoutHybridMode();
    void layoutSplitView(WorkflowMode left, WorkflowMode right);

    //==========================================================================
    // Callbacks
    //==========================================================================

    void onModeButtonClicked(WorkflowMode mode);
    void onContextChanged();
    void onMIDIChanged();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WorkflowManager)
};

//==============================================================================
// Quick Start Wizard
//==============================================================================

class QuickStartWizard : public juce::Component
{
public:
    QuickStartWizard();
    ~QuickStartWizard() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    /**
     * Get user's workflow preference
     */
    WorkflowMode getSelectedMode() const { return selectedMode_; }

    /**
     * Show callback
     */
    std::function<void(WorkflowMode)> onModeSelected;
    std::function<void()> onCancelled;

private:
    // Mode selection cards
    std::unique_ptr<juce::TextButton> musicianCard_;
    std::unique_ptr<juce::TextButton> composerCard_;
    std::unique_ptr<juce::TextButton> engineerCard_;

    // Descriptions
    std::unique_ptr<juce::Label> musicianDescription_;
    std::unique_ptr<juce::Label> composerDescription_;
    std::unique_ptr<juce::Label> engineerDescription_;

    // Confirm button
    std::unique_ptr<juce::TextButton> confirmButton_;
    std::unique_ptr<juce::TextButton> cancelButton_;

    WorkflowMode selectedMode_;

    void onCardClicked(WorkflowMode mode);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(QuickStartWizard)
};

} // namespace midikompanion
