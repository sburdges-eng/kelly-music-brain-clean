#include "EditCommand.h"
#include <algorithm>

namespace kelly {

// =============================================================================
// CommandManager
// =============================================================================

CommandManager::CommandManager(int maxUndoSteps)
    : maxUndoSteps_(maxUndoSteps) {
}

void CommandManager::executeCommand(std::unique_ptr<EditCommand> command) {
    if (!command) return;

    command->execute();
    undoStack_.push_back(std::move(command));

    // Limit undo stack size
    if (static_cast<int>(undoStack_.size()) > maxUndoSteps_) {
        undoStack_.erase(undoStack_.begin());
    }

    // Clear redo stack when new command is executed
    redoStack_.clear();
}

bool CommandManager::undo() {
    if (undoStack_.empty()) {
        return false;
    }

    auto command = std::move(undoStack_.back());
    undoStack_.pop_back();
    command->undo();
    redoStack_.push_back(std::move(command));
    return true;
}

bool CommandManager::redo() {
    if (redoStack_.empty()) {
        return false;
    }

    auto command = std::move(redoStack_.back());
    redoStack_.pop_back();
    command->execute();
    undoStack_.push_back(std::move(command));
    return true;
}

juce::String CommandManager::getUndoDescription() const {
    if (undoStack_.empty()) {
        return {};
    }
    return undoStack_.back()->getDescription();
}

juce::String CommandManager::getRedoDescription() const {
    if (redoStack_.empty()) {
        return {};
    }
    return redoStack_.back()->getDescription();
}

void CommandManager::clear() {
    undoStack_.clear();
    redoStack_.clear();
}

void CommandManager::setMaxUndoSteps(int maxSteps) {
    maxUndoSteps_ = maxSteps;
    // Trim undo stack if needed
    if (static_cast<int>(undoStack_.size()) > maxUndoSteps_) {
        undoStack_.erase(undoStack_.begin(),
                        undoStack_.begin() + (undoStack_.size() - maxUndoSteps_));
    }
}

// =============================================================================
// AddNoteCommand
// =============================================================================

AddNoteCommand::AddNoteCommand(GeneratedMidi& midi, const MidiNote& note)
    : midi_(midi), note_(note), noteIndex_(0) {
}

void AddNoteCommand::execute() {
    midi_.notes.push_back(note_);
    noteIndex_ = midi_.notes.size() - 1;
}

void AddNoteCommand::undo() {
    if (noteIndex_ < midi_.notes.size()) {
        midi_.notes.erase(midi_.notes.begin() + noteIndex_);
    }
}

// =============================================================================
// DeleteNoteCommand
// =============================================================================

DeleteNoteCommand::DeleteNoteCommand(GeneratedMidi& midi, size_t noteIndex)
    : midi_(midi), noteIndex_(noteIndex) {
    if (noteIndex_ < midi_.notes.size()) {
        deletedNote_ = midi_.notes[noteIndex_];
    }
}

void DeleteNoteCommand::execute() {
    if (noteIndex_ < midi_.notes.size()) {
        midi_.notes.erase(midi_.notes.begin() + noteIndex_);
    }
}

void DeleteNoteCommand::undo() {
    if (noteIndex_ <= midi_.notes.size()) {
        midi_.notes.insert(midi_.notes.begin() + noteIndex_, deletedNote_);
    }
}

// =============================================================================
// ModifyNoteCommand
// =============================================================================

ModifyNoteCommand::ModifyNoteCommand(GeneratedMidi& midi, size_t noteIndex, Property property, float newValue)
    : midi_(midi), noteIndex_(noteIndex), property_(property), newValue_(newValue), oldValue_(0.0f) {
    if (noteIndex_ < midi_.notes.size()) {
        const auto& note = midi_.notes[noteIndex_];
        switch (property_) {
            case Pitch:
                oldValue_ = static_cast<float>(note.pitch);
                break;
            case Velocity:
                oldValue_ = static_cast<float>(note.velocity);
                break;
            case StartTime:
                oldValue_ = static_cast<float>(note.startBeat);
                break;
            case Duration:
                oldValue_ = static_cast<float>(note.duration);
                break;
            case Channel:
                oldValue_ = static_cast<float>(note.channel);
                break;
        }
    }
}

void ModifyNoteCommand::execute() {
    if (noteIndex_ >= midi_.notes.size()) return;

    auto& note = midi_.notes[noteIndex_];
    switch (property_) {
        case Pitch:
            note.pitch = static_cast<int>(newValue_);
            break;
        case Velocity:
            note.velocity = static_cast<int>(juce::jlimit(0, 127, static_cast<int>(newValue_)));
            break;
        case StartTime:
            note.startBeat = newValue_;
            break;
        case Duration:
            note.duration = newValue_;
            break;
        case Channel:
            note.channel = static_cast<int>(juce::jlimit(0, 15, static_cast<int>(newValue_)));
            break;
    }
}

void ModifyNoteCommand::undo() {
    if (noteIndex_ >= midi_.notes.size()) return;

    auto& note = midi_.notes[noteIndex_];
    switch (property_) {
        case Pitch:
            note.pitch = static_cast<int>(oldValue_);
            break;
        case Velocity:
            note.velocity = static_cast<int>(oldValue_);
            break;
        case StartTime:
            note.startBeat = oldValue_;
            break;
        case Duration:
            note.duration = oldValue_;
            break;
        case Channel:
            note.channel = static_cast<int>(oldValue_);
            break;
    }
}

juce::String ModifyNoteCommand::getDescription() const {
    switch (property_) {
        case Pitch: return "Change Pitch";
        case Velocity: return "Change Velocity";
        case StartTime: return "Change Start Time";
        case Duration: return "Change Duration";
        case Channel: return "Change Channel";
        default: return "Modify Note";
    }
}

bool ModifyNoteCommand::canMergeWith(const EditCommand& other) const {
    const auto* otherCmd = dynamic_cast<const ModifyNoteCommand*>(&other);
    if (!otherCmd) return false;
    return otherCmd->noteIndex_ == noteIndex_ && otherCmd->property_ == property_;
}

void ModifyNoteCommand::mergeWith(const EditCommand& other) {
    const auto* otherCmd = dynamic_cast<const ModifyNoteCommand*>(&other);
    if (otherCmd) {
        newValue_ = otherCmd->newValue_;
    }
}

// =============================================================================
// MoveNoteCommand
// =============================================================================

MoveNoteCommand::MoveNoteCommand(GeneratedMidi& midi, size_t noteIndex, double newStartBeat, int newPitch)
    : midi_(midi), noteIndex_(noteIndex), newStartBeat_(newStartBeat), newPitch_(newPitch),
      oldStartBeat_(0.0), oldPitch_(60) {
    if (noteIndex_ < midi_.notes.size()) {
        const auto& note = midi_.notes[noteIndex_];
        oldStartBeat_ = note.startBeat;
        oldPitch_ = note.pitch;
    }
}

void MoveNoteCommand::execute() {
    if (noteIndex_ >= midi_.notes.size()) return;
    auto& note = midi_.notes[noteIndex_];
    note.startBeat = newStartBeat_;
    note.pitch = newPitch_;
}

void MoveNoteCommand::undo() {
    if (noteIndex_ >= midi_.notes.size()) return;
    auto& note = midi_.notes[noteIndex_];
    note.startBeat = oldStartBeat_;
    note.pitch = oldPitch_;
}

bool MoveNoteCommand::canMergeWith(const EditCommand& other) const {
    const auto* otherCmd = dynamic_cast<const MoveNoteCommand*>(&other);
    if (!otherCmd) return false;
    return otherCmd->noteIndex_ == noteIndex_;
}

void MoveNoteCommand::mergeWith(const EditCommand& other) {
    const auto* otherCmd = dynamic_cast<const MoveNoteCommand*>(&other);
    if (otherCmd) {
        newStartBeat_ = otherCmd->newStartBeat_;
        newPitch_ = otherCmd->newPitch_;
    }
}

// =============================================================================
// MultiEditCommand
// =============================================================================

MultiEditCommand::MultiEditCommand(juce::String description)
    : description_(description) {
}

void MultiEditCommand::addCommand(std::unique_ptr<EditCommand> command) {
    if (command) {
        commands_.push_back(std::move(command));
    }
}

void MultiEditCommand::execute() {
    for (auto& cmd : commands_) {
        cmd->execute();
    }
}

void MultiEditCommand::undo() {
    // Undo in reverse order
    for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
        (*it)->undo();
    }
}

} // namespace kelly
