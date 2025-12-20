#pragma once
/*
 * ScoreEntryPanel.h - Sheet Music / Score Entry Interface
 * ========================================================
 *
 * Interface for composers who think in musical notation.
 * Allows creation of music using traditional notation concepts:
 *
 * - Staff-based note entry
 * - Clef selection (treble, bass, alto, tenor)
 * - Time signature and key signature
 * - Note values (whole, half, quarter, eighth, etc.)
 * - Dynamics (pp, p, mp, mf, f, ff)
 * - Articulations (staccato, legato, accent, tenuto)
 * - Expression marks (crescendo, diminuendo, fermata)
 * - Chord symbols (C, Dm7, G7, etc.)
 * - Lyrics/text
 *
 * FLEXIBILITY:
 * - As detailed or simple as desired
 * - Quick sketch mode: just pitches and rhythms
 * - Full score mode: complete notation with all markings
 * - Playback preview
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include "../music_theory/MusicTheoryBrain.h"
#include <memory>
#include <vector>

namespace midikompanion {

//==============================================================================
// Note Entry Modes
//==============================================================================

enum class EntryMode {
    Simple,        // Just pitch + rhythm (quick sketching)
    Standard,      // Pitch, rhythm, dynamics
    Professional,  // Full notation with all markings
    Chord          // Chord symbol entry
};

//==============================================================================
// Musical Notation Types
//==============================================================================

enum class Clef {
    Treble,
    Bass,
    Alto,
    Tenor,
    Percussion
};

enum class NoteValue {
    Whole,         // 4 beats in 4/4
    Half,          // 2 beats
    Quarter,       // 1 beat
    Eighth,        // 0.5 beats
    Sixteenth,     // 0.25 beats
    ThirtySecond,  // 0.125 beats
    Dotted         // 1.5x value
};

enum class Dynamic {
    None,
    Pianissimo,    // pp
    Piano,         // p
    MezzoPiano,    // mp
    MezzoForte,    // mf
    Forte,         // f
    Fortissimo,    // ff
    Crescendo,     // <
    Diminuendo     // >
};

enum class Articulation {
    None,
    Staccato,      // Short, detached
    Legato,        // Smooth, connected
    Accent,        // Emphasized
    Tenuto,        // Full value
    Marcato,       // Very accented
    Fermata        // Hold longer
};

//==============================================================================
// Score Entry Data Structures
//==============================================================================

struct NotationNote {
    int pitch;                    // MIDI pitch (0-127)
    NoteValue duration;           // Note value
    bool dotted;                  // Dotted note (1.5x duration)
    bool triplet;                 // Part of triplet
    Dynamic dynamic;              // Volume marking
    Articulation articulation;    // How to play
    bool tie;                     // Tied to next note
    std::string lyric;            // Lyric syllable
    int measure;                  // Bar number (1-indexed)
    float beat;                   // Beat position in measure
};

struct ChordSymbol {
    std::string symbol;           // "C", "Dm7", "G7b9", etc.
    int measure;                  // Bar number
    float beat;                   // Beat position
};

struct TimeSignatureChange {
    int numerator;
    int denominator;
    int measure;                  // Where it takes effect
};

struct KeySignatureChange {
    std::string key;              // "C Major", "A Minor", etc.
    int measure;                  // Where it takes effect
};

struct TempoMarking {
    float bpm;
    std::string description;      // "Allegro", "Andante", etc.
    int measure;
};

//==============================================================================
// Main Score Entry Panel
//==============================================================================

class ScoreEntryPanel : public juce::Component
{
public:
    ScoreEntryPanel();
    ~ScoreEntryPanel() override = default;

    //==========================================================================
    // Component Overrides
    //==========================================================================

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;

    //==========================================================================
    // Entry Mode
    //==========================================================================

    void setEntryMode(EntryMode mode);
    EntryMode getEntryMode() const { return entryMode_; }

    //==========================================================================
    // Score Setup
    //==========================================================================

    void setTimeSignature(int numerator, int denominator);
    void setKeySignature(const std::string& key);
    void setTempo(float bpm, const std::string& description = "");
    void setClef(Clef clef);

    //==========================================================================
    // Note Entry
    //==========================================================================

    /**
     * Add note at current cursor position
     */
    void addNote(int pitch, NoteValue duration, bool dotted = false);

    /**
     * Add chord (multiple notes at same time)
     */
    void addChord(const std::vector<int>& pitches, NoteValue duration);

    /**
     * Add chord symbol above staff
     */
    void addChordSymbol(const std::string& symbol);

    /**
     * Add rest
     */
    void addRest(NoteValue duration);

    //==========================================================================
    // Note Editing
    //==========================================================================

    void setDynamic(Dynamic dynamic);
    void setArticulation(Articulation articulation);
    void addLyric(const std::string& text);
    void toggleDot();  // Make note dotted/undotted
    void makeTriplet(); // Convert to triplet
    void toggleTie();   // Tie to next note

    //==========================================================================
    // Navigation
    //==========================================================================

    void moveCursorForward();
    void moveCursorBackward();
    void moveCursorToMeasure(int measure);
    void moveCursorToNextMeasure();

    //==========================================================================
    // Quick Entry Templates
    //==========================================================================

    /**
     * Quick templates for common patterns
     */
    struct ScoreTemplate {
        std::string name;
        std::string description;
        std::vector<NotationNote> notes;
        std::vector<ChordSymbol> chords;
    };

    void loadTemplate(const ScoreTemplate& template_);
    std::vector<ScoreTemplate> getAvailableTemplates() const;

    /**
     * Quick entry: "C major scale, quarter notes"
     */
    void quickEntry(const std::string& description);

    //==========================================================================
    // MIDI Conversion
    //==========================================================================

    /**
     * Convert notation to MIDI
     */
    juce::MidiBuffer toMidiBuffer() const;

    /**
     * Convert MIDI to notation (import)
     */
    void fromMidiBuffer(const juce::MidiBuffer& buffer);

    //==========================================================================
    // Playback
    //==========================================================================

    void playFromStart();
    void playFromCursor();
    void stop();
    void toggleMetronome();

    //==========================================================================
    // Display Options
    //==========================================================================

    enum class ViewMode {
        SingleStaff,      // One staff (one instrument)
        GrandStaff,       // Two staves (piano)
        FullScore,        // Multiple instruments
        LeadSheet         // Melody + chords only
    };

    void setViewMode(ViewMode mode);
    void setZoom(float zoomFactor);  // 0.5 - 2.0
    void setShowChordSymbols(bool show);
    void setShowLyrics(bool show);
    void setShowDynamics(bool show);

    //==========================================================================
    // Export
    //==========================================================================

    /**
     * Export as MusicXML (standard notation format)
     */
    bool exportMusicXML(const juce::File& outputFile);

    /**
     * Export as PDF score
     */
    bool exportPDF(const juce::File& outputFile);

    /**
     * Export as MIDI file
     */
    bool exportMIDI(const juce::File& outputFile);

private:
    //==========================================================================
    // UI Components
    //==========================================================================

    // Entry mode selector
    std::unique_ptr<juce::ComboBox> entryModeSelector_;

    // Staff display area
    std::unique_ptr<juce::Viewport> scoreViewport_;
    std::unique_ptr<juce::Component> scoreDisplay_;

    // Note value buttons
    std::unique_ptr<juce::TextButton> wholeNoteButton_;
    std::unique_ptr<juce::TextButton> halfNoteButton_;
    std::unique_ptr<juce::TextButton> quarterNoteButton_;
    std::unique_ptr<juce::TextButton> eighthNoteButton_;
    std::unique_ptr<juce::TextButton> sixteenthNoteButton_;
    std::unique_ptr<juce::TextButton> dottedButton_;
    std::unique_ptr<juce::TextButton> tripletButton_;

    // Dynamics buttons
    std::unique_ptr<juce::TextButton> ppButton_;
    std::unique_ptr<juce::TextButton> pButton_;
    std::unique_ptr<juce::TextButton> mpButton_;
    std::unique_ptr<juce::TextButton> mfButton_;
    std::unique_ptr<juce::TextButton> fButton_;
    std::unique_ptr<juce::TextButton> ffButton_;

    // Articulation buttons
    std::unique_ptr<juce::TextButton> staccatoButton_;
    std::unique_ptr<juce::TextButton> legatoButton_;
    std::unique_ptr<juce::TextButton> accentButton_;
    std::unique_ptr<juce::TextButton> tenutoButton_;

    // Quick entry
    std::unique_ptr<juce::TextEditor> quickEntryInput_;
    std::unique_ptr<juce::TextButton> quickEntryButton_;

    // Playback controls
    std::unique_ptr<juce::TextButton> playButton_;
    std::unique_ptr<juce::TextButton> stopButton_;
    std::unique_ptr<juce::TextButton> metronomeButton_;

    // Time/Key signature
    std::unique_ptr<juce::Label> timeSignatureLabel_;
    std::unique_ptr<juce::Label> keySignatureLabel_;
    std::unique_ptr<juce::Label> tempoLabel_;

    // Virtual keyboard (for mouse entry)
    std::unique_ptr<juce::Component> virtualKeyboard_;

    //==========================================================================
    // Music Theory Integration
    //==========================================================================

    std::unique_ptr<theory::MusicTheoryBrain> theoryBrain_;

    //==========================================================================
    // Score Data
    //==========================================================================

    std::vector<NotationNote> notes_;
    std::vector<ChordSymbol> chordSymbols_;
    std::vector<TimeSignatureChange> timeSignatures_;
    std::vector<KeySignatureChange> keySignatures_;
    std::vector<TempoMarking> tempos_;

    EntryMode entryMode_;
    ViewMode viewMode_;
    Clef currentClef_;
    NoteValue currentNoteValue_;
    Dynamic currentDynamic_;
    Articulation currentArticulation_;

    int cursorMeasure_;
    float cursorBeat_;

    bool showChordSymbols_;
    bool showLyrics_;
    bool showDynamics_;
    float zoomFactor_;

    //==========================================================================
    // Staff Drawing
    //==========================================================================

    void drawStaff(juce::Graphics& g, juce::Rectangle<int> area);
    void drawClef(juce::Graphics& g, Clef clef, juce::Point<int> position);
    void drawTimeSignature(juce::Graphics& g, int numerator, int denominator,
                          juce::Point<int> position);
    void drawKeySignature(juce::Graphics& g, const std::string& key,
                         juce::Point<int> position);
    void drawNote(juce::Graphics& g, const NotationNote& note,
                 juce::Point<int> position);
    void drawChordSymbol(juce::Graphics& g, const ChordSymbol& chord,
                        juce::Point<int> position);
    void drawBarline(juce::Graphics& g, int x);
    void drawCursor(juce::Graphics& g);

    //==========================================================================
    // Pitch Mapping
    //==========================================================================

    /**
     * Convert mouse Y position to MIDI pitch based on staff
     */
    int staffPositionToPitch(int yPosition, Clef clef) const;

    /**
     * Convert MIDI pitch to staff Y position
     */
    int pitchToStaffPosition(int pitch, Clef clef) const;

    //==========================================================================
    // Quick Entry Parser
    //==========================================================================

    /**
     * Parse text like "C major scale quarter notes"
     * or "Dm7 G7 C whole notes"
     */
    void parseQuickEntry(const std::string& text);

    //==========================================================================
    // Template Library
    //==========================================================================

    void initializeTemplates();
    std::vector<ScoreTemplate> templates_;

    //==========================================================================
    // Callbacks
    //==========================================================================

    void onNoteValueSelected(NoteValue value);
    void onDynamicSelected(Dynamic dynamic);
    void onArticulationSelected(Articulation articulation);
    void onQuickEntryExecuted();
    void onPlayClicked();
    void onStopClicked();
    void onMetronomeToggled();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ScoreEntryPanel)
};

//==============================================================================
// Piano Roll View (Alternative to Staff Notation)
//==============================================================================

class PianoRollPanel : public juce::Component
{
public:
    PianoRollPanel();
    ~PianoRollPanel() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;

    /**
     * Convert to/from MIDI
     */
    juce::MidiBuffer toMidiBuffer() const;
    void fromMidiBuffer(const juce::MidiBuffer& buffer);

    /**
     * Display options
     */
    void setSnapToGrid(bool snap);
    void setGridResolution(NoteValue resolution);
    void setVerticalZoom(float zoom);
    void setHorizontalZoom(float zoom);

private:
    struct PianoRollNote {
        int pitch;
        float startTime;
        float duration;
        int velocity;
    };

    std::vector<PianoRollNote> notes_;
    bool snapToGrid_;
    NoteValue gridResolution_;
    float verticalZoom_;
    float horizontalZoom_;

    void drawPianoKeys(juce::Graphics& g, juce::Rectangle<int> area);
    void drawGrid(juce::Graphics& g, juce::Rectangle<int> area);
    void drawNotes(juce::Graphics& g, juce::Rectangle<int> area);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PianoRollPanel)
};

} // namespace midikompanion
