#pragma once
/*
 * MusicianCommandPanel.h - Expert Music Theory Command Interface
 * ================================================================
 *
 * Advanced UI for educated musicians to command MIDI generation using
 * music theory terminology and natural language:
 *
 * Examples:
 * - "Make the bass play a walking line in the bridge"
 * - "Add a tritone substitution on beat 3 of bar 8"
 * - "Change drums to half-time feel in the verse"
 * - "Guitar plays harmonics at the pre-chorus"
 * - "Add a pickslide 2 beats before the chorus"
 * - "Modulate to the relative minor in bar 16"
 * - "Bass follows the circle of fifths progression"
 * - "Add a Coltrane substitution over the ii-V-I"
 *
 * FEATURES:
 * - Natural language command parser
 * - Music theory vocabulary understanding
 * - Timeline-based editing (bar/beat precision)
 * - Instrument-specific commands
 * - Real-time MIDI preview
 * - Command history and undo/redo
 * - Autocomplete for music theory terms
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include "../music_theory/MusicTheoryBrain.h"
#include <memory>
#include <queue>

namespace midikompanion {

class MusicianCommandPanel : public juce::Component
                            , public juce::TextEditor::Listener
{
public:
    MusicianCommandPanel();
    ~MusicianCommandPanel() override = default;

    //==========================================================================
    // Component Overrides
    //==========================================================================

    void paint(juce::Graphics& g) override;
    void resized() override;

    //==========================================================================
    // Command Execution
    //==========================================================================

    /**
     * Execute a music theory command
     *
     * Examples:
     * - "bass plays root notes on downbeats"
     * - "add a drum fill in bar 8"
     * - "guitar plays arpeggios in the chorus"
     * - "modulate to D major at bar 16"
     * - "add a pickslide before the chorus"
     */
    struct CommandResult {
        bool success;
        std::string message;
        std::vector<juce::MidiMessage> generatedMidi;
        std::string explanation;  // What the system did and why
    };

    CommandResult executeCommand(const std::string& command);

    /**
     * Get current MIDI buffer (for preview/export)
     */
    juce::MidiBuffer getCurrentMidiBuffer() const;

    /**
     * Set the context for commands
     */
    void setCurrentProject(const std::string& projectPath);
    void setCurrentTimeSignature(int numerator, int denominator);
    void setCurrentKey(const std::string& key);
    void setCurrentTempo(float bpm);
    void setTotalBars(int bars);

    //==========================================================================
    // Command History
    //==========================================================================

    void undo();
    void redo();
    void clearHistory();
    std::vector<std::string> getCommandHistory() const;

private:
    //==========================================================================
    // UI Components
    //==========================================================================

    // Command input area
    std::unique_ptr<juce::TextEditor> commandInput_;
    std::unique_ptr<juce::TextButton> executeButton_;
    std::unique_ptr<juce::TextButton> clearButton_;

    // Quick action buttons (common commands)
    std::unique_ptr<juce::TextButton> addFillButton_;
    std::unique_ptr<juce::TextButton> modulateButton_;
    std::unique_ptr<juce::TextButton> substituteButton_;
    std::unique_ptr<juce::TextButton> reharmonizeButton_;

    // Output display
    std::unique_ptr<juce::TextEditor> outputDisplay_;
    std::unique_ptr<juce::Label> statusLabel_;

    // Command history
    std::unique_ptr<juce::ListBox> historyList_;
    std::unique_ptr<juce::TextButton> undoButton_;
    std::unique_ptr<juce::TextButton> redoButton_;

    // Autocomplete dropdown
    std::unique_ptr<juce::ComboBox> autocompleteDropdown_;

    // MIDI preview
    std::unique_ptr<juce::Component> midiPreviewComponent_;

    //==========================================================================
    // Music Theory Engine
    //==========================================================================

    std::unique_ptr<theory::MusicTheoryBrain> theoryBrain_;

    //==========================================================================
    // Command Parser
    //==========================================================================

    struct ParsedCommand {
        // What to do
        std::string action;  // "add", "change", "remove", "modulate", etc.

        // Target
        std::string instrument;  // "bass", "guitar", "drums", "piano"
        std::string element;     // "notes", "rhythm", "chord", "fill", "effect"

        // Specification
        std::string specification;  // "walking line", "half-time", "tritone sub"

        // Timing
        std::string section;     // "verse", "chorus", "bridge", "intro", "outro"
        int bar;                 // Bar number (1-indexed)
        float beat;              // Beat within bar (1.0, 1.5, etc.)
        std::string timing;      // "before", "after", "at", "during"

        // Music theory parameters
        std::string key;
        std::string scale;
        std::string chordQuality;
        std::string rhythmPattern;
        std::string articulation;
        float velocity;          // 0.0-1.0
        int octave;

        // Modifiers
        bool relative;           // "relative minor", "relative major"
        int semitones;          // Transposition amount
        std::vector<std::string> tags;  // Additional metadata
    };

    ParsedCommand parseCommand(const std::string& command) const;

    //==========================================================================
    // Command Execution Handlers
    //==========================================================================

    // High-level actions
    CommandResult handleAddCommand(const ParsedCommand& parsed);
    CommandResult handleChangeCommand(const ParsedCommand& parsed);
    CommandResult handleRemoveCommand(const ParsedCommand& parsed);
    CommandResult handleModulateCommand(const ParsedCommand& parsed);
    CommandResult handleSubstituteCommand(const ParsedCommand& parsed);
    CommandResult handleReharmonizeCommand(const ParsedCommand& parsed);

    // Instrument-specific
    CommandResult handleBassCommand(const ParsedCommand& parsed);
    CommandResult handleGuitarCommand(const ParsedCommand& parsed);
    CommandResult handleDrumCommand(const ParsedCommand& parsed);
    CommandResult handleKeyboardCommand(const ParsedCommand& parsed);

    // Effects and articulations
    CommandResult addPickslide(const ParsedCommand& parsed);
    CommandResult addHarmonics(const ParsedCommand& parsed);
    CommandResult addBend(const ParsedCommand& parsed);
    CommandResult addSlide(const ParsedCommand& parsed);
    CommandResult addFill(const ParsedCommand& parsed);

    //==========================================================================
    // MIDI Generation
    //==========================================================================

    struct MidiGenerationContext {
        std::string key;
        int numerator;
        int denominator;
        float bpm;
        int totalBars;
        std::map<std::string, std::vector<juce::MidiMessage>> trackData;
    };

    MidiGenerationContext context_;

    // Generate specific patterns
    std::vector<juce::MidiMessage> generateWalkingBass(
        const ParsedCommand& parsed
    );

    std::vector<juce::MidiMessage> generateArpeggio(
        const ParsedCommand& parsed
    );

    std::vector<juce::MidiMessage> generateChordVoicing(
        const ParsedCommand& parsed
    );

    std::vector<juce::MidiMessage> generateDrumPattern(
        const ParsedCommand& parsed
    );

    std::vector<juce::MidiMessage> generateMelodicLine(
        const ParsedCommand& parsed
    );

    // Effects as MIDI
    std::vector<juce::MidiMessage> createPickslideMidi(
        int bar,
        float beat,
        int channel
    );

    std::vector<juce::MidiMessage> createBendMidi(
        int note,
        int startBend,
        int endBend,
        int bar,
        float beat,
        int channel
    );

    //==========================================================================
    // Timing Utilities
    //==========================================================================

    // Convert musical time to MIDI ticks
    struct TimingInfo {
        int absoluteBar;     // Bar in song (1-indexed)
        float beatInBar;     // Beat within bar (1.0 = downbeat)
        int midiTick;        // Absolute MIDI tick
        double seconds;      // Time in seconds
    };

    TimingInfo parseTimingExpression(const std::string& expression) const;

    TimingInfo getTimingForSection(
        const std::string& section,
        const std::string& position = "start"
    ) const;

    int barBeatToTick(int bar, float beat) const;
    double barBeatToSeconds(int bar, float beat) const;

    //==========================================================================
    // Natural Language Processing
    //==========================================================================

    // Music theory vocabulary
    struct VocabularyEntry {
        std::string term;
        std::vector<std::string> synonyms;
        std::string category;  // "rhythm", "harmony", "articulation", etc.
        std::string description;
    };

    std::vector<VocabularyEntry> musicVocabulary_;
    void initializeMusicVocabulary();

    // Token matching
    std::string matchVocabularyTerm(const std::string& input) const;
    std::vector<std::string> tokenizeCommand(const std::string& command) const;

    // Intent detection
    std::string detectIntent(const std::vector<std::string>& tokens) const;
    std::string detectInstrument(const std::vector<std::string>& tokens) const;
    std::string detectTiming(const std::vector<std::string>& tokens) const;
    std::string detectMusicTheoryConcept(const std::vector<std::string>& tokens) const;

    //==========================================================================
    // Command Validation
    //==========================================================================

    struct ValidationResult {
        bool valid;
        std::string errorMessage;
        std::vector<std::string> suggestions;
    };

    ValidationResult validateCommand(const ParsedCommand& parsed) const;
    ValidationResult checkTimingValid(int bar, float beat) const;
    ValidationResult checkInstrumentExists(const std::string& instrument) const;
    ValidationResult checkMusicTheoryValid(const ParsedCommand& parsed) const;

    //==========================================================================
    // Autocomplete
    //==========================================================================

    std::vector<std::string> getAutocompleteSuggestions(
        const std::string& partial
    ) const;

    void updateAutocomplete(const std::string& currentText);

    //==========================================================================
    // Command History Management
    //==========================================================================

    struct CommandHistoryEntry {
        std::string command;
        ParsedCommand parsed;
        std::vector<juce::MidiMessage> previousState;
        std::vector<juce::MidiMessage> newState;
        juce::Time timestamp;
    };

    std::vector<CommandHistoryEntry> commandHistory_;
    int historyPosition_;

    void addToHistory(const CommandHistoryEntry& entry);

    //==========================================================================
    // Preset Commands
    //==========================================================================

    struct PresetCommand {
        std::string name;
        std::string description;
        std::string command;
        std::string category;
    };

    std::vector<PresetCommand> presetCommands_;
    void initializePresetCommands();

    void showPresetMenu();

    //==========================================================================
    // Callbacks
    //==========================================================================

    void textEditorTextChanged(juce::TextEditor& editor) override;
    void textEditorReturnKeyPressed(juce::TextEditor& editor) override;

    void onExecuteClicked();
    void onClearClicked();
    void onUndoClicked();
    void onRedoClicked();

    void onQuickActionClicked(const std::string& action);

    //==========================================================================
    // Help System
    //==========================================================================

    void showHelpDialog();
    std::string getCommandSyntaxHelp() const;
    std::vector<std::string> getExampleCommands() const;

    //==========================================================================
    // MIDI Preview
    //==========================================================================

    void updateMidiPreview();
    void playMidiPreview();

    //==========================================================================
    // Error Handling
    //==========================================================================

    void showError(const std::string& message);
    void showSuccess(const std::string& message);
    void showInfo(const std::string& message);

    //==========================================================================
    // Styling
    //==========================================================================

    juce::Colour commandPanelBackground_;
    juce::Colour inputBoxBackground_;
    juce::Colour buttonColor_;
    juce::Colour successColor_;
    juce::Colour errorColor_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MusicianCommandPanel)
};

//==============================================================================
// Command Examples Documentation Component
//==============================================================================

class CommandExamplesPanel : public juce::Component
{
public:
    CommandExamplesPanel();

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    struct CommandExample {
        std::string category;
        std::string command;
        std::string description;
    };

    std::vector<CommandExample> examples_;

    std::unique_ptr<juce::TextEditor> examplesDisplay_;

    void initializeExamples();
    std::string formatExamples() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CommandExamplesPanel)
};

} // namespace midikompanion
