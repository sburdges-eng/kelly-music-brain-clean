#include "MidiEditor.h"
#include "../ui/KellyLookAndFeel.h"
#include <algorithm>
#include <cmath>

namespace kelly {

MidiEditor::MidiEditor() : commandManager_(50) {
  setWantsKeyboardFocus(true);
  setMouseCursor(juce::MouseCursor::PointingHandCursor);
}

MidiEditor::~MidiEditor() { editableMidi_ = nullptr; }

void MidiEditor::setMidiData(const GeneratedMidi &midi) {
  PianoRollPreview::setMidiData(midi);

  // Create editable copy
  midiCopy_ = midi;
  editableMidi_ = &midiCopy_;

  // Clear selection when new data is set
  selectedNotes_.clear();

  repaint();
}

void MidiEditor::paint(juce::Graphics &g) {
  // Call parent paint for grid and notes
  PianoRollPreview::paint(g);

  auto bounds = getLocalBounds();

  // Draw selection
  if (!selectedNotes_.empty()) {
    drawSelection(g, bounds);
  }

  // Draw drag preview
  if (dragState_.isDragging) {
    drawDragPreview(g, bounds);
  }

  // Draw selection box
  if (isSelecting_) {
    g.setColour(juce::Colour(0x40FFFFFF));
    g.fillRect(juce::Rectangle<int>(selectionStart_, selectionEnd_));
    g.setColour(juce::Colour(0xFFFFFFFF));
    g.drawRect(juce::Rectangle<int>(selectionStart_, selectionEnd_), 1);
  }

  // Draw tool cursor
  if (editable_) {
    auto mousePos = getMouseXYRelative();
    drawToolCursor(g, mousePos);
  }
}

void MidiEditor::mouseDown(const juce::MouseEvent &e) {
  if (!editable_ || !editableMidi_) {
    PianoRollPreview::mouseDown(e);
    return;
  }

  auto hit = hitTest(e.getPosition());

  if (currentTool_ == Tool::Select) {
    if (e.mods.isLeftButtonDown()) {
      if (e.mods.isCommandDown() || e.mods.isCtrlDown()) {
        // Ctrl+click: toggle selection
        if (hit.hitNote) {
          if (selectedNotes_.count(hit.noteIndex)) {
            deselectNote(hit.noteIndex);
          } else {
            selectNote(hit.noteIndex, true);
          }
        } else {
          // Start box selection
          isSelecting_ = true;
          selectionStart_ = e.getPosition();
          selectionEnd_ = e.getPosition();
        }
      } else {
        // Regular click
        if (hit.hitNote) {
          // Clicked on note
          if (!selectedNotes_.count(hit.noteIndex)) {
            deselectAll();
            selectNote(hit.noteIndex);
          }

          // Start dragging
          dragState_.isDragging = true;
          dragState_.noteIndex = hit.noteIndex;
          dragState_.dragStart = e.getPosition();

          const auto &note = editableMidi_->notes[hit.noteIndex];
          dragState_.startOffset = hit.hitBeat - note.startBeat;
          dragState_.pitchOffset = hit.hitPitch - note.pitch;
          dragState_.isResizing = hit.hitEndHandle;
        } else {
          // Clicked on empty space - deselect all
          deselectAll();
          // Start box selection
          isSelecting_ = true;
          selectionStart_ = e.getPosition();
          selectionEnd_ = e.getPosition();
        }
      }
    }
  } else if (currentTool_ == Tool::Pencil) {
    if (e.mods.isLeftButtonDown()) {
      // Add note at clicked position
      double beat = snapBeat(hit.hitBeat);
      int pitch = snapPitch(hit.hitPitch);
      addNoteAt(beat, pitch);
    }
  } else if (currentTool_ == Tool::Eraser) {
    if (e.mods.isLeftButtonDown() && hit.hitNote) {
      deleteNote(hit.noteIndex);
    }
  }

  repaint();
}

void MidiEditor::mouseDrag(const juce::MouseEvent &e) {
  if (!editable_ || !editableMidi_) {
    PianoRollPreview::mouseDrag(e);
    return;
  }

  if (currentTool_ == Tool::Select) {
    if (isSelecting_) {
      selectionEnd_ = e.getPosition();
      repaint();
    } else if (dragState_.isDragging) {
      // Drag note
      auto hit = hitTest(e.getPosition());
      double newBeat = snapBeat(hit.hitBeat - dragState_.startOffset);
      int newPitch = snapPitch(hit.hitPitch - dragState_.pitchOffset);

      if (dragState_.isResizing) {
        const auto &note = editableMidi_->notes[dragState_.noteIndex];
        double newDuration = hit.hitBeat - note.startBeat;
        if (newDuration > 0.1) { // Minimum duration
          resizeNote(dragState_.noteIndex, newDuration);
        }
      } else {
        moveNote(dragState_.noteIndex, newBeat, newPitch);
      }
      repaint();
    }
  }
}

void MidiEditor::mouseUp(const juce::MouseEvent &e) {
  if (!editable_ || !editableMidi_) {
    PianoRollPreview::mouseUp(e);
    return;
  }

  if (isSelecting_) {
    // Finalize box selection
    auto box = juce::Rectangle<int>(selectionStart_, selectionEnd_)
                   .withSizeKeepingCentre(
                       std::abs(selectionEnd_.x - selectionStart_.x),
                       std::abs(selectionEnd_.y - selectionStart_.y));
    if (selectionEnd_.x < selectionStart_.x)
      box.setX(selectionEnd_.x);
    if (selectionEnd_.y < selectionStart_.y)
      box.setY(selectionEnd_.y);
    auto notesInBox = getNotesInBox(box);

    if (e.mods.isCommandDown() || e.mods.isCtrlDown()) {
      // Add to selection
      for (auto idx : notesInBox) {
        selectedNotes_.insert(idx);
      }
    } else {
      // Replace selection
      selectedNotes_ = notesInBox;
    }

    isSelecting_ = false;
    if (onSelectionChanged) {
      onSelectionChanged();
    }
  }

  if (dragState_.isDragging) {
    dragState_.isDragging = false;
    if (onMidiChanged && editableMidi_) {
      onMidiChanged(*editableMidi_);
    }
  }

  repaint();
}

void MidiEditor::mouseMove(const juce::MouseEvent &e) {
  if (!editable_)
    return;

  // Update cursor based on tool and hover state
  auto hit = hitTest(e.getPosition());
  if (hit.hitNote) {
    if (hit.hitStartHandle || hit.hitEndHandle) {
      setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
    } else {
      setMouseCursor(juce::MouseCursor::DraggingHandCursor);
    }
  } else {
    setMouseCursor(juce::MouseCursor::PointingHandCursor);
  }
}

void MidiEditor::mouseDoubleClick(const juce::MouseEvent &e) {
  if (!editable_ || !editableMidi_)
    return;

  auto hit = hitTest(e.getPosition());
  if (hit.hitNote) {
    // Double-click to edit note properties (could open property panel)
    // For now, just select the note
    deselectAll();
    selectNote(hit.noteIndex);
  } else {
    // Double-click empty space to add note
    double beat = snapBeat(hit.hitBeat);
    int pitch = snapPitch(hit.hitPitch);
    addNoteAt(beat, pitch);
  }
}

bool MidiEditor::keyPressed(const juce::KeyPress &key) {
  if (!editable_)
    return false;

  if (key == juce::KeyPress::deleteKey || key == juce::KeyPress::backspaceKey) {
    deleteSelected();
    return true;
  } else if (key.getKeyCode() == 'Z' && (key.getModifiers().isCommandDown() ||
                                         key.getModifiers().isCtrlDown())) {
    if (key.getModifiers().isShiftDown()) {
      redo();
    } else {
      undo();
    }
    return true;
  } else if (key.getKeyCode() == 'C' && (key.getModifiers().isCommandDown() ||
                                         key.getModifiers().isCtrlDown())) {
    copySelected();
    return true;
  } else if (key.getKeyCode() == 'V' && (key.getModifiers().isCommandDown() ||
                                         key.getModifiers().isCtrlDown())) {
    auto hit = hitTest(getMouseXYRelative());
    paste(hit.hitBeat);
    return true;
  } else if (key.getKeyCode() == 'A' && (key.getModifiers().isCommandDown() ||
                                         key.getModifiers().isCtrlDown())) {
    selectAll();
    return true;
  }

  return false;
}

// Selection methods
void MidiEditor::selectNote(size_t noteIndex, bool addToSelection) {
  if (!addToSelection) {
    selectedNotes_.clear();
  }
  if (noteIndex < editableMidi_->notes.size()) {
    selectedNotes_.insert(noteIndex);
    if (onSelectionChanged) {
      onSelectionChanged();
    }
    repaint();
  }
}

void MidiEditor::deselectNote(size_t noteIndex) {
  selectedNotes_.erase(noteIndex);
  if (onSelectionChanged) {
    onSelectionChanged();
  }
  repaint();
}

void MidiEditor::selectAll() {
  selectedNotes_.clear();
  for (size_t i = 0; i < editableMidi_->notes.size(); ++i) {
    selectedNotes_.insert(i);
  }
  if (onSelectionChanged) {
    onSelectionChanged();
  }
  repaint();
}

void MidiEditor::deselectAll() {
  selectedNotes_.clear();
  if (onSelectionChanged) {
    onSelectionChanged();
  }
  repaint();
}

void MidiEditor::selectInRange(double startBeat, double endBeat, int minPitch,
                               int maxPitch) {
  selectedNotes_.clear();
  for (size_t i = 0; i < editableMidi_->notes.size(); ++i) {
    const auto &note = editableMidi_->notes[i];
    if (note.startBeat >= startBeat && note.startBeat <= endBeat &&
        note.pitch >= minPitch && note.pitch <= maxPitch) {
      selectedNotes_.insert(i);
    }
  }
  if (onSelectionChanged) {
    onSelectionChanged();
  }
  repaint();
}

// Undo/redo
void MidiEditor::undo() {
  if (commandManager_.undo() && onMidiChanged && editableMidi_) {
    onMidiChanged(*editableMidi_);
    repaint();
  }
}

void MidiEditor::redo() {
  if (commandManager_.redo() && onMidiChanged && editableMidi_) {
    onMidiChanged(*editableMidi_);
    repaint();
  }
}

// Quantization
void MidiEditor::quantizeSelected(double gridSize) {
  if (selectedNotes_.empty() || !editableMidi_)
    return;

  auto multiCmd = std::make_unique<MultiEditCommand>("Quantize Notes");

  for (auto idx : selectedNotes_) {
    if (idx >= editableMidi_->notes.size())
      continue;
    auto &note = editableMidi_->notes[idx];
    double quantizedBeat = std::round(note.startBeat / gridSize) * gridSize;
    if (std::abs(quantizedBeat - note.startBeat) > 0.01) {
      multiCmd->addCommand(std::make_unique<ModifyNoteCommand>(
          *editableMidi_, idx, ModifyNoteCommand::StartTime,
          static_cast<float>(quantizedBeat)));
    }
  }

  if (multiCmd->getDescription() != "Quantize Notes" ||
      selectedNotes_.size() > 0) {
    commandManager_.executeCommand(std::move(multiCmd));
    if (onMidiChanged) {
      onMidiChanged(*editableMidi_);
    }
    repaint();
  }
}

void MidiEditor::humanizeSelected(float timingAmount, float velocityAmount) {
  // Implementation would randomize timing and velocity
  // For now, placeholder
}

// Copy/paste
void MidiEditor::copySelected() {
  clipboard_.clear();
  if (!editableMidi_)
    return;

  for (auto idx : selectedNotes_) {
    if (idx < editableMidi_->notes.size()) {
      clipboard_.push_back(editableMidi_->notes[idx]);
    }
  }
}

void MidiEditor::paste(double pasteTime) {
  if (clipboard_.empty() || !editableMidi_)
    return;

  auto multiCmd = std::make_unique<MultiEditCommand>("Paste Notes");

  // Find earliest note in clipboard to calculate offset
  double earliestBeat = std::numeric_limits<double>::max();
  for (const auto &note : clipboard_) {
    earliestBeat = std::min(earliestBeat, note.startBeat);
  }
  double timeOffset = pasteTime - earliestBeat;

  for (const auto &note : clipboard_) {
    MidiNote newNote = note;
    newNote.startBeat = note.startBeat + timeOffset;
    multiCmd->addCommand(
        std::make_unique<AddNoteCommand>(*editableMidi_, newNote));
  }

  commandManager_.executeCommand(std::move(multiCmd));
  if (onMidiChanged) {
    onMidiChanged(*editableMidi_);
  }
  repaint();
}

// Delete
void MidiEditor::deleteSelected() {
  if (selectedNotes_.empty() || !editableMidi_)
    return;

  auto multiCmd = std::make_unique<MultiEditCommand>("Delete Notes");

  // Delete in reverse order to maintain indices
  std::vector<size_t> sortedIndices(selectedNotes_.begin(),
                                    selectedNotes_.end());
  std::sort(sortedIndices.rbegin(), sortedIndices.rend());

  for (auto idx : sortedIndices) {
    if (idx < editableMidi_->notes.size()) {
      multiCmd->addCommand(
          std::make_unique<DeleteNoteCommand>(*editableMidi_, idx));
    }
  }

  commandManager_.executeCommand(std::move(multiCmd));
  selectedNotes_.clear();
  if (onMidiChanged) {
    onMidiChanged(*editableMidi_);
  }
  if (onSelectionChanged) {
    onSelectionChanged();
  }
  repaint();
}

// Property editing
void MidiEditor::setSelectedNoteProperty(size_t noteIndex,
                                         const juce::String &property,
                                         float value) {
  if (noteIndex >= editableMidi_->notes.size())
    return;

  ModifyNoteCommand::Property prop = ModifyNoteCommand::Pitch;
  if (property == "pitch")
    prop = ModifyNoteCommand::Pitch;
  else if (property == "velocity")
    prop = ModifyNoteCommand::Velocity;
  else if (property == "startTime")
    prop = ModifyNoteCommand::StartTime;
  else if (property == "duration")
    prop = ModifyNoteCommand::Duration;
  else if (property == "channel")
    prop = ModifyNoteCommand::Channel;
  else
    return;

  auto cmd = std::make_unique<ModifyNoteCommand>(*editableMidi_, noteIndex,
                                                 prop, value);
  commandManager_.executeCommand(std::move(cmd));
  if (onMidiChanged) {
    onMidiChanged(*editableMidi_);
  }
  repaint();
}

// Helper methods
void MidiEditor::drawSelection(juce::Graphics &g,
                               const juce::Rectangle<int> &bounds) {
  g.setColour(juce::Colour(0x4000AAFF));
  for (auto idx : selectedNotes_) {
    if (idx >= editableMidi_->notes.size())
      continue;
    const auto &note = editableMidi_->notes[idx];
    float x = static_cast<float>(beatToPixel(note.startBeat));
    double timeRange = timeEnd_ - timeStart_;
    float width = static_cast<float>(note.duration / timeRange *
                                     bounds.getWidth() * zoom_);
    float y = static_cast<float>(pitchToPixel(note.pitch));
    float height =
        bounds.getHeight() / static_cast<float>(pitchMax_ - pitchMin_);

    g.fillRect(x, y - height, width, height);
    g.setColour(juce::Colour(0xFF00AAFF));
    g.drawRect(x, y - height, width, height, 2.0f);
  }
}

void MidiEditor::drawDragPreview(juce::Graphics &g,
                                 const juce::Rectangle<int> &bounds) {
  // Draw preview of where note will be moved
  g.setColour(juce::Colour(0x60FFFFFF));
  // Implementation would show preview at drag position
}

void MidiEditor::drawToolCursor(juce::Graphics &g,
                                const juce::Point<int> &mousePos) {
  // Draw tool-specific cursor indicator
  g.setColour(juce::Colour(0x80FFFFFF));
  if (currentTool_ == Tool::Pencil) {
    g.fillEllipse(mousePos.x - 4, mousePos.y - 4, 8, 8);
  }
}

MidiEditor::HitTestResult
MidiEditor::hitTest(const juce::Point<int> &pos) const {
  HitTestResult result;
  if (!editableMidi_)
    return result;

  result.hitBeat = pixelToBeat(static_cast<float>(pos.x));
  result.hitPitch = pixelToPitch(pos.y);

  const float handleWidth = 5.0f;
  auto bounds = getLocalBounds();
  double timeRange = timeEnd_ - timeStart_;
  float noteHeight =
      bounds.getHeight() / static_cast<float>(pitchMax_ - pitchMin_);

  for (size_t i = 0; i < editableMidi_->notes.size(); ++i) {
    const auto &note = editableMidi_->notes[i];
    float x = static_cast<float>(beatToPixel(note.startBeat));
    float width = static_cast<float>(note.duration / timeRange *
                                     bounds.getWidth() * zoom_);
    float y = static_cast<float>(pitchToPixel(note.pitch));

    juce::Rectangle<float> noteRect(x, y - noteHeight, width, noteHeight);
    if (noteRect.contains(pos.toFloat())) {
      result.hitNote = true;
      result.noteIndex = i;

      // Check if hit resize handles
      if (std::abs(pos.x - x) < handleWidth) {
        result.hitStartHandle = true;
      } else if (std::abs(pos.x - (x + width)) < handleWidth) {
        result.hitEndHandle = true;
      }
      break;
    }
  }

  return result;
}

double MidiEditor::pixelToBeat(float x) const {
  auto bounds = getLocalBounds();
  double timeRange = timeEnd_ - timeStart_;
  double normalized = x / bounds.getWidth();
  return timeStart_ + normalized * timeRange;
}

float MidiEditor::beatToPixel(double beat) const {
  auto bounds = getLocalBounds();
  double timeRange = timeEnd_ - timeStart_;
  if (timeRange <= 0.0)
    return 0.0f;
  double normalized = (beat - timeStart_) / timeRange;
  return static_cast<float>(normalized * bounds.getWidth());
}

int MidiEditor::pixelToPitch(int y) const {
  auto bounds = getLocalBounds();
  int pitchRange = pitchMax_ - pitchMin_;
  if (pitchRange == 0)
    return pitchMin_;
  float normalized = 1.0f - static_cast<float>(y) / bounds.getHeight();
  return pitchMin_ + static_cast<int>(normalized * pitchRange);
}

int MidiEditor::pitchToPixel(int pitch) const {
  auto bounds = getLocalBounds();
  int pitchRange = pitchMax_ - pitchMin_;
  if (pitchRange == 0)
    return bounds.getHeight() / 2;
  float normalized =
      static_cast<float>(pitch - pitchMin_) / static_cast<float>(pitchRange);
  return bounds.getHeight() - static_cast<int>(normalized * bounds.getHeight());
}

double MidiEditor::snapBeat(double beat) const {
  if (snapMode_ == SnapMode::None)
    return beat;

  double gridSize = 1.0;
  switch (snapMode_) {
  case SnapMode::Beat:
    gridSize = 1.0;
    break;
  case SnapMode::HalfBeat:
    gridSize = 0.5;
    break;
  case SnapMode::QuarterBeat:
    gridSize = 0.25;
    break;
  case SnapMode::EighthBeat:
    gridSize = 0.125;
    break;
  case SnapMode::SixteenthBeat:
    gridSize = 0.0625;
    break;
  default:
    break;
  }

  return std::round(beat / gridSize) * gridSize;
}

int MidiEditor::snapPitch(int pitch) const {
  // No pitch snapping for now (could snap to scale)
  return pitch;
}

void MidiEditor::addNoteAt(double beat, int pitch, double duration,
                           int velocity) {
  if (!editableMidi_)
    return;

  MidiNote note;
  note.pitch = pitch;
  note.startBeat = beat;
  note.duration = duration;
  note.velocity = velocity;
  note.channel = 0;
  note.startTick = static_cast<int>(beat * 480);
  note.durationTicks = static_cast<int>(duration * 480);

  auto cmd = std::make_unique<AddNoteCommand>(*editableMidi_, note);
  commandManager_.executeCommand(std::move(cmd));

  // Select the new note
  if (!editableMidi_->notes.empty()) {
    selectNote(editableMidi_->notes.size() - 1);
  }

  if (onMidiChanged) {
    onMidiChanged(*editableMidi_);
  }
  repaint();
}

void MidiEditor::deleteNote(size_t noteIndex) {
  if (!editableMidi_ || noteIndex >= editableMidi_->notes.size())
    return;

  auto cmd = std::make_unique<DeleteNoteCommand>(*editableMidi_, noteIndex);
  commandManager_.executeCommand(std::move(cmd));

  selectedNotes_.erase(noteIndex);
  if (onMidiChanged) {
    onMidiChanged(*editableMidi_);
  }
  repaint();
}

void MidiEditor::moveNote(size_t noteIndex, double newBeat, int newPitch) {
  if (!editableMidi_ || noteIndex >= editableMidi_->notes.size())
    return;

  auto cmd = std::make_unique<MoveNoteCommand>(*editableMidi_, noteIndex,
                                               newBeat, newPitch);
  commandManager_.executeCommand(std::move(cmd));

  if (onMidiChanged) {
    onMidiChanged(*editableMidi_);
  }
  repaint();
}

void MidiEditor::resizeNote(size_t noteIndex, double newDuration) {
  if (!editableMidi_ || noteIndex >= editableMidi_->notes.size())
    return;

  auto cmd = std::make_unique<ModifyNoteCommand>(
      *editableMidi_, noteIndex, ModifyNoteCommand::Duration,
      static_cast<float>(newDuration));
  commandManager_.executeCommand(std::move(cmd));

  if (onMidiChanged) {
    onMidiChanged(*editableMidi_);
  }
  repaint();
}

std::set<size_t>
MidiEditor::getNotesInBox(const juce::Rectangle<int> &box) const {
  std::set<size_t> result;
  if (!editableMidi_)
    return result;

  double startBeat = pixelToBeat(static_cast<float>(box.getX()));
  double endBeat = pixelToBeat(static_cast<float>(box.getRight()));
  int minPitch = pixelToPitch(box.getBottom());
  int maxPitch = pixelToPitch(box.getY());

  if (minPitch > maxPitch)
    std::swap(minPitch, maxPitch);

  for (size_t i = 0; i < editableMidi_->notes.size(); ++i) {
    const auto &note = editableMidi_->notes[i];
    if (note.startBeat >= startBeat && note.startBeat <= endBeat &&
        note.pitch >= minPitch && note.pitch <= maxPitch) {
      result.insert(i);
    }
  }

  return result;
}

} // namespace kelly
