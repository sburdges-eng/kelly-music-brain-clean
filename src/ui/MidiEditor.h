#pragma once
/*
 * MidiEditor.h - Editable Piano Roll Component
 * ============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Type System: Types.h (GeneratedMidi, MidiNote via KellyTypes.h)
 * - UI Layer: Extends PianoRollPreview (read-only preview)
 * - UI Layer: Used by EmotionWorkstation (editable MIDI editing)
 * - Engine Layer: Displays and edits GeneratedMidi from MidiGenerator
 *
 * Purpose: Editable piano roll component that extends PianoRollPreview with
 *          full editing capabilities (add, move, resize, delete notes).
 *
 * Features:
 * - Click to add notes
 * - Drag to move/resize notes
 * - Multi-select (box select, lasso select, Ctrl+click)
 * - Keyboard shortcuts (Delete, Copy, Paste, Undo/Redo)
 * - Quantization and snapping
 * - Undo/redo system
 */

#include "PianoRollPreview.h"  // Base class for piano roll display
#include "EditCommand.h"  // Undo/redo command system
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <set>

namespace kelly {

/**
 * MidiEditor - Editable piano roll component
 *
 * Extends PianoRollPreview with full editing capabilities:
 * - Click to add notes
 * - Drag to move/resize notes
 * - Multi-select (box select, lasso select, Ctrl+click)
 * - Keyboard shortcuts (Delete, Copy, Paste, Undo/Redo)
 * - Quantization and snapping
 * - Undo/redo system
 */
class MidiEditor : public PianoRollPreview,
                   public juce::ComponentListener {
public:
    enum class Tool {
        Select,      // Selection tool
        Pencil,      // Add notes
        Eraser,      // Delete notes
        Zoom,        // Zoom tool
        Pan          // Pan tool
    };

    enum class SnapMode {
        None,        // No snapping
        Beat,        // Snap to beats
        HalfBeat,    // Snap to half beats
        QuarterBeat, // Snap to quarter beats
        EighthBeat,  // Snap to eighth beats
        SixteenthBeat // Snap to sixteenth beats
    };

    MidiEditor();
    ~MidiEditor() override;

    // Override setMidiData to enable editing
    void setMidiData(const GeneratedMidi& midi);
    void setEditable(bool editable) { editable_ = editable; repaint(); }
    bool isEditable() const { return editable_; }

    // Tool selection
    void setTool(Tool tool) { currentTool_ = tool; repaint(); }
    Tool getTool() const { return currentTool_; }

    // Snap settings
    void setSnapMode(SnapMode mode) { snapMode_ = mode; repaint(); }
    SnapMode getSnapMode() const { return snapMode_; }

    // Selection management
    void selectNote(size_t noteIndex, bool addToSelection = false);
    void deselectNote(size_t noteIndex);
    void selectAll();
    void deselectAll();
    void selectInRange(double startBeat, double endBeat, int minPitch, int maxPitch);
    const std::set<size_t>& getSelectedNotes() const { return selectedNotes_; }

    // Undo/redo
    void undo();
    void redo();
    bool canUndo() const { return commandManager_.canUndo(); }
    bool canRedo() const { return commandManager_.canRedo(); }

    // Quantization
    void quantizeSelected(double gridSize);
    void humanizeSelected(float timingAmount, float velocityAmount);

    // Copy/paste
    void copySelected();
    void paste(double pasteTime);
    bool hasClipboard() const { return !clipboard_.empty(); }

    // Delete
    void deleteSelected();

    // Property editing
    void setSelectedNoteProperty(size_t noteIndex, const juce::String& property, float value);

    // Callbacks
    std::function<void(const GeneratedMidi&)> onMidiChanged;
    std::function<void()> onSelectionChanged;

    // Override Component methods
    void paint(juce::Graphics& g) override;
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    void mouseUp(const juce::MouseEvent& e) override;
    void mouseMove(const juce::MouseEvent& e) override;
    void mouseDoubleClick(const juce::MouseEvent& e) override;
    bool keyPressed(const juce::KeyPress& key) override;

private:
    // Editing state
    bool editable_ = true;
    Tool currentTool_ = Tool::Select;
    SnapMode snapMode_ = SnapMode::QuarterBeat;

    // Selection
    std::set<size_t> selectedNotes_;
    juce::Point<int> selectionStart_;
    juce::Point<int> selectionEnd_;
    bool isSelecting_ = false;

    // Drag state
    struct DragState {
        bool isDragging = false;
        size_t noteIndex = 0;
        double startOffset = 0.0;  // Offset from note start when drag began
        int pitchOffset = 0;       // Offset from note pitch when drag began
        bool isResizing = false;    // Resizing note duration vs moving
        juce::Point<int> dragStart;
    };
    DragState dragState_;

    // Undo/redo
    CommandManager commandManager_;

    // Clipboard
    std::vector<MidiNote> clipboard_;

    // Helper methods
    void drawSelection(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawDragPreview(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawToolCursor(juce::Graphics& g, const juce::Point<int>& mousePos);

    // Hit testing
    struct HitTestResult {
        size_t noteIndex = SIZE_MAX;
        bool hitNote = false;
        bool hitStartHandle = false;  // Left edge for resizing
        bool hitEndHandle = false;    // Right edge for resizing
        double hitBeat = 0.0;
        int hitPitch = 60;
    };
    HitTestResult hitTest(const juce::Point<int>& pos) const;

    // Coordinate conversion
    double pixelToBeat(float x) const;
    float beatToPixel(double beat) const;
    int pixelToPitch(int y) const;
    int pitchToPixel(int pitch) const;

    // Snapping
    double snapBeat(double beat) const;
    int snapPitch(int pitch) const;

    // Note operations
    void addNoteAt(double beat, int pitch, double duration = 1.0, int velocity = 100);
    void deleteNote(size_t noteIndex);
    void moveNote(size_t noteIndex, double newBeat, int newPitch);
    void resizeNote(size_t noteIndex, double newDuration);

    // Selection operations
    void updateSelectionBox(const juce::Point<int>& start, const juce::Point<int>& end);
    std::set<size_t> getNotesInBox(const juce::Rectangle<int>& box) const;

    // Mutable reference to MIDI data (for editing)
    GeneratedMidi* editableMidi_ = nullptr;
    GeneratedMidi midiCopy_;  // Working copy for editing

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MidiEditor)
};

} // namespace kelly
