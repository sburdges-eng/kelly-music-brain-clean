#pragma once
/*
 * EditCommand.h - Undo/Redo Command System
 * ========================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Type System: Types.h (GeneratedMidi, MidiNote via KellyTypes.h)
 * - UI Layer: Used by MidiEditor (undo/redo functionality)
 * - UI Layer: CommandManager manages command stack
 *
 * Purpose: Command pattern implementation for undo/redo functionality in MIDI editor.
 *          All edit operations are wrapped in command objects for undo/redo support.
 *
 * Features:
 * - Command pattern for undo/redo
 * - Command merging (multiple operations can be merged)
 * - Command descriptions for UI display
 * - Command stack management
 */

#include "../common/Types.h"  // GeneratedMidi, MidiNote (via KellyTypes.h)
#include <juce_core/juce_core.h>
#include <memory>
#include <vector>

namespace kelly {

/**
 * EditCommand - Base class for undo/redo operations
 *
 * Uses Command pattern for undo/redo functionality in MIDI editor.
 * All edit operations are wrapped in command objects.
 */
class EditCommand {
public:
    virtual ~EditCommand() = default;

    /**
     * Execute the command (do or redo)
     */
    virtual void execute() = 0;

    /**
     * Undo the command
     */
    virtual void undo() = 0;

    /**
     * Get description of the command (for UI display)
     */
    virtual juce::String getDescription() const = 0;

    /**
     * Check if command can be merged with another command
     * (e.g., multiple note moves can be merged into one undo entry)
     */
    virtual bool canMergeWith(const EditCommand& other) const { return false; }

    /**
     * Merge this command with another (if canMergeWith returns true)
     */
    virtual void mergeWith(const EditCommand& other) {}
};

/**
 * CommandManager - Manages undo/redo stack
 */
class CommandManager {
public:
    CommandManager(int maxUndoSteps = 50);
    ~CommandManager() = default;

    /**
     * Execute a command and add to undo stack
     */
    void executeCommand(std::unique_ptr<EditCommand> command);

    /**
     * Undo last command
     */
    bool undo();

    /**
     * Redo last undone command
     */
    bool redo();

    /**
     * Check if undo is available
     */
    bool canUndo() const { return !undoStack_.empty(); }

    /**
     * Check if redo is available
     */
    bool canRedo() const { return !redoStack_.empty(); }

    /**
     * Get description of next undo operation
     */
    juce::String getUndoDescription() const;

    /**
     * Get description of next redo operation
     */
    juce::String getRedoDescription() const;

    /**
     * Clear undo/redo stacks
     */
    void clear();

    /**
     * Set maximum number of undo steps
     */
    void setMaxUndoSteps(int maxSteps);

private:
    std::vector<std::unique_ptr<EditCommand>> undoStack_;
    std::vector<std::unique_ptr<EditCommand>> redoStack_;
    int maxUndoSteps_;
};

// =============================================================================
// Concrete Command Classes
// =============================================================================

/**
 * AddNoteCommand - Add a new MIDI note
 */
class AddNoteCommand : public EditCommand {
public:
    AddNoteCommand(GeneratedMidi& midi, const MidiNote& note);
    void execute() override;
    void undo() override;
    juce::String getDescription() const override { return "Add Note"; }

private:
    GeneratedMidi& midi_;
    MidiNote note_;
    size_t noteIndex_;
};

/**
 * DeleteNoteCommand - Delete a MIDI note
 */
class DeleteNoteCommand : public EditCommand {
public:
    DeleteNoteCommand(GeneratedMidi& midi, size_t noteIndex);
    void execute() override;
    void undo() override;
    juce::String getDescription() const override { return "Delete Note"; }

private:
    GeneratedMidi& midi_;
    size_t noteIndex_;
    MidiNote deletedNote_;
};

/**
 * ModifyNoteCommand - Modify a MIDI note property
 */
class ModifyNoteCommand : public EditCommand {
public:
    enum Property {
        Pitch,
        Velocity,
        StartTime,
        Duration,
        Channel
    };

    ModifyNoteCommand(GeneratedMidi& midi, size_t noteIndex, Property property, float newValue);
    void execute() override;
    void undo() override;
    juce::String getDescription() const override;
    bool canMergeWith(const EditCommand& other) const override;
    void mergeWith(const EditCommand& other) override;

private:
    GeneratedMidi& midi_;
    size_t noteIndex_;
    Property property_;
    float newValue_;
    float oldValue_;
};

/**
 * MoveNoteCommand - Move a note in time or pitch
 */
class MoveNoteCommand : public EditCommand {
public:
    MoveNoteCommand(GeneratedMidi& midi, size_t noteIndex, double newStartBeat, int newPitch);
    void execute() override;
    void undo() override;
    juce::String getDescription() const override { return "Move Note"; }
    bool canMergeWith(const EditCommand& other) const override;
    void mergeWith(const EditCommand& other) override;

private:
    GeneratedMidi& midi_;
    size_t noteIndex_;
    double newStartBeat_;
    int newPitch_;
    double oldStartBeat_;
    int oldPitch_;
};

/**
 * MultiEditCommand - Execute multiple commands as one undo operation
 */
class MultiEditCommand : public EditCommand {
public:
    MultiEditCommand(juce::String description);
    void addCommand(std::unique_ptr<EditCommand> command);
    void execute() override;
    void undo() override;
    juce::String getDescription() const override { return description_; }

private:
    juce::String description_;
    std::vector<std::unique_ptr<EditCommand>> commands_;
};

} // namespace kelly
