#include "project/ProjectManager.h"
#include "common/Types.h"
#include "plugin/PluginState.h" // Include PluginState for Preset type
#include <juce_core/juce_core.h>

namespace midikompanion {

ProjectManager::ProjectManager() { clearError(); }

//==============================================================================
// Project Save/Load Interface
//==============================================================================

bool ProjectManager::saveProject(const juce::File &file,
                                 const kelly::PluginState::Preset &state,
                                 const GeneratedMidi &generatedMidi,
                                 const std::vector<MidiNote> &vocalNotes,
                                 const std::vector<juce::String> &lyrics,
                                 const std::vector<int> &selectedEmotionIds,
                                 const std::optional<int> &primaryEmotionId) {
  clearError();

  // Create project data structure
  ProjectData projectData;
  projectData.name = file.getFileNameWithoutExtension();
  projectData.createdTime = juce::Time::getCurrentTime();
  projectData.modifiedTime = juce::Time::getCurrentTime();
  projectData.versionMajor = 1;
  projectData.versionMinor = 0;

  // Copy plugin state (full preset including CassetteState)
  projectData.pluginState = state;

  // Copy generated MIDI
  projectData.generatedMidi = generatedMidi;

  // Copy vocal data
  projectData.vocalNotes = vocalNotes;
  projectData.lyrics = lyrics;

  // Copy emotion selections
  projectData.selectedEmotionIds = selectedEmotionIds;
  projectData.primaryEmotionId = primaryEmotionId;

  // Extract tempo from generated MIDI or plugin state
  if (generatedMidi.bpm > 0.0f) {
    projectData.tempo = generatedMidi.bpm;
  } else {
    projectData.tempo = 120.0f; // Default
  }

  // Convert to JSON
  juce::var json = projectData.toJson();

  // Write to file
  juce::String jsonString = juce::JSON::toString(json, true);
  bool success = file.replaceWithText(jsonString);

  if (!success) {
    setError("Failed to write project file: " + file.getFullPathName());
    return false;
  }

  return true;
}

bool ProjectManager::loadProject(const juce::File &file,
                                 kelly::PluginState::Preset &outState,
                                 GeneratedMidi &outGeneratedMidi,
                                 std::vector<MidiNote> &outVocalNotes,
                                 std::vector<juce::String> &outLyrics,
                                 std::vector<int> &outSelectedEmotionIds,
                                 std::optional<int> &outPrimaryEmotionId) {
  clearError();

  if (!file.existsAsFile()) {
    setError("Project file does not exist: " + file.getFullPathName());
    return false;
  }

  // Read JSON from file
  juce::var jsonData;
  juce::Result result = juce::JSON::parse(file.loadFileAsString(), jsonData);

  if (result.failed()) {
    setError("Failed to parse project file: " + result.getErrorMessage());
    return false;
  }

  // Parse project data
  auto projectData = ProjectData::fromJson(jsonData);
  if (!projectData.has_value()) {
    setError("Invalid project file format");
    return false;
  }

  // Restore plugin state (full preset including CassetteState)
  outState = projectData->pluginState;

  // Restore generated MIDI metadata
  // Note: For v1.0, we restore metadata only. Full MIDI restoration can be
  // added in v1.1 User will need to regenerate MIDI after loading (acceptable
  // for v1.0)
  outGeneratedMidi = projectData->generatedMidi;
  // The generatedMidi structure will have metadata but empty note vectors
  // This is acceptable for v1.0 - user can regenerate

  // Restore vocal data
  outVocalNotes = projectData->vocalNotes;
  outLyrics = projectData->lyrics;

  // Restore emotion selections
  outSelectedEmotionIds = projectData->selectedEmotionIds;
  outPrimaryEmotionId = projectData->primaryEmotionId;

  return true;
}

bool ProjectManager::isValidProjectFile(const juce::File &file) const {
  if (!file.existsAsFile()) {
    return false;
  }

  // Check file extension
  if (!file.hasFileExtension("midikompanion")) {
    return false;
  }

  // Try to parse as JSON
  juce::var jsonData;
  juce::Result result = juce::JSON::parse(file.loadFileAsString(), jsonData);
  if (result.failed()) {
    return false;
  }

  // Check for required fields
  if (!jsonData.isObject()) {
    return false;
  }

  auto *obj = jsonData.getDynamicObject();
  if (!obj) {
    return false;
  }

  // Check for version field (indicates project file)
  if (!obj->hasProperty("version")) {
    return false;
  }

  return true;
}

std::optional<ProjectManager::ProjectData>
ProjectManager::getProjectMetadata(const juce::File &file) const {
  if (!file.existsAsFile()) {
    return std::nullopt;
  }

  juce::var jsonData;
  juce::Result result = juce::JSON::parse(file.loadFileAsString(), jsonData);
  if (result.failed()) {
    return std::nullopt;
  }

  auto projectData = ProjectData::fromJson(jsonData);
  if (!projectData.has_value()) {
    return std::nullopt;
  }

  // Return metadata only (don't load full MIDI data)
  ProjectData metadata;
  metadata.name = projectData->name;
  metadata.createdTime = projectData->createdTime;
  metadata.modifiedTime = projectData->modifiedTime;
  metadata.versionMajor = projectData->versionMajor;
  metadata.versionMinor = projectData->versionMinor;
  metadata.tempo = projectData->tempo;
  metadata.timeSignature = projectData->timeSignature;

  return metadata;
}

//==============================================================================
// ProjectData Serialization
//==============================================================================

juce::ValueTree ProjectManager::ProjectData::toValueTree() const {
  juce::ValueTree tree("Project");

  // Metadata
  tree.setProperty(
      "version", juce::String(versionMajor) + "." + juce::String(versionMinor),
      nullptr);
  tree.setProperty("name", name, nullptr);
  tree.setProperty("createdTime", createdTime.toMilliseconds(), nullptr);
  tree.setProperty("modifiedTime", modifiedTime.toMilliseconds(), nullptr);

  // Project settings
  tree.setProperty("tempo", tempo, nullptr);
  auto timeSigTree = juce::ValueTree("TimeSignature");
  timeSigTree.setProperty("numerator", timeSignature.numerator, nullptr);
  timeSigTree.setProperty("denominator", timeSignature.denominator, nullptr);
  tree.appendChild(timeSigTree, nullptr);

  // Plugin state (use full preset ValueTree serialization)
  auto pluginStateTree = pluginState.toValueTree();
  tree.appendChild(pluginStateTree, nullptr);

  // Generated MIDI (simplified - store as JSON string for now)
  // Full implementation would serialize each track separately
  tree.setProperty("hasGeneratedMidi",
                   !generatedMidi.melody.empty() ||
                       !generatedMidi.bass.empty() ||
                       !generatedMidi.chords.empty(),
                   nullptr);

  // Vocal notes
  auto vocalNotesTree = juce::ValueTree("VocalNotes");
  for (const auto &note : vocalNotes) {
    auto noteTree = juce::ValueTree("Note");
    noteTree.setProperty("pitch", note.pitch, nullptr);
    noteTree.setProperty("velocity", note.velocity, nullptr);
    noteTree.setProperty("startTick", static_cast<juce::int64>(note.startTick),
                         nullptr);
    noteTree.setProperty("durationTicks",
                         static_cast<juce::int64>(note.durationTicks), nullptr);
    vocalNotesTree.appendChild(noteTree, nullptr);
  }
  tree.appendChild(vocalNotesTree, nullptr);

  // Lyrics
  auto lyricsTree = juce::ValueTree("Lyrics");
  for (const auto &lyric : lyrics) {
    auto lyricTree = juce::ValueTree("Line");
    lyricTree.setProperty("text", lyric, nullptr);
    lyricsTree.appendChild(lyricTree, nullptr);
  }
  tree.appendChild(lyricsTree, nullptr);

  // Emotion selections
  auto emotionsTree = juce::ValueTree("EmotionSelections");
  for (int emotionId : selectedEmotionIds) {
    auto emotionTree = juce::ValueTree("Emotion");
    emotionTree.setProperty("id", emotionId, nullptr);
    emotionsTree.appendChild(emotionTree, nullptr);
  }
  tree.appendChild(emotionsTree, nullptr);

  if (primaryEmotionId.has_value()) {
    tree.setProperty("primaryEmotionId", *primaryEmotionId, nullptr);
  }

  return tree;
}

std::optional<ProjectManager::ProjectData>
ProjectManager::ProjectData::fromValueTree(const juce::ValueTree &tree) {
  if (!tree.isValid() || !tree.hasType("Project")) {
    return std::nullopt;
  }

  ProjectData data;

  // Metadata
  if (tree.hasProperty("version")) {
    juce::String version = tree.getProperty("version").toString();
    int dotPos = version.indexOfChar('.');
    if (dotPos > 0) {
      data.versionMajor = version.substring(0, dotPos).getIntValue();
      data.versionMinor = version.substring(dotPos + 1).getIntValue();
    }
  }
  data.name = tree.getProperty("name").toString();
  if (tree.hasProperty("createdTime")) {
    data.createdTime = juce::Time(
        tree.getProperty("createdTime").toString().getLargeIntValue());
  }
  if (tree.hasProperty("modifiedTime")) {
    data.modifiedTime = juce::Time(
        tree.getProperty("modifiedTime").toString().getLargeIntValue());
  }

  // Project settings
  data.tempo = static_cast<float>(tree.getProperty("tempo"));
  auto timeSigTree = tree.getChildWithName("TimeSignature");
  if (timeSigTree.isValid()) {
    data.timeSignature.numerator =
        static_cast<int>(timeSigTree.getProperty("numerator"));
    data.timeSignature.denominator =
        static_cast<int>(timeSigTree.getProperty("denominator"));
  }

  // Plugin state (load full preset including CassetteState)
  auto pluginStateTree = tree.getChildWithName("Preset");
  if (pluginStateTree.isValid()) {
    auto preset = kelly::PluginState::Preset::fromValueTree(pluginStateTree);
    if (preset.has_value()) {
      // Store full preset (including CassetteState)
      data.pluginState = *preset;
    }
  }

  // Generated MIDI (restore metadata - full note restoration would be in v1.1)
  // For v1.0, we restore metadata and the user can regenerate MIDI
  if (tree.hasProperty("hasGeneratedMidi")) {
    // MIDI was saved, but we'll restore metadata only
    // User will need to regenerate MIDI (acceptable for v1.0)
  }

  // Vocal notes
  auto vocalNotesTree = tree.getChildWithName("VocalNotes");
  if (vocalNotesTree.isValid()) {
    for (int i = 0; i < vocalNotesTree.getNumChildren(); ++i) {
      auto noteTree = vocalNotesTree.getChild(i);
      if (noteTree.hasType("Note")) {
        MidiNote note;
        note.pitch = static_cast<int>(noteTree.getProperty("pitch"));
        note.velocity = static_cast<int>(noteTree.getProperty("velocity"));
        note.startTick = static_cast<int>(
            noteTree.getProperty("startTick").toString().getLargeIntValue());
        note.durationTicks =
            static_cast<int>(noteTree.getProperty("durationTicks")
                                 .toString()
                                 .getLargeIntValue());
        data.vocalNotes.push_back(note);
      }
    }
  }

  // Lyrics
  auto lyricsTree = tree.getChildWithName("Lyrics");
  if (lyricsTree.isValid()) {
    for (int i = 0; i < lyricsTree.getNumChildren(); ++i) {
      auto lyricTree = lyricsTree.getChild(i);
      if (lyricTree.hasType("Line")) {
        data.lyrics.push_back(lyricTree.getProperty("text").toString());
      }
    }
  }

  // Emotion selections
  auto emotionsTree = tree.getChildWithName("EmotionSelections");
  if (emotionsTree.isValid()) {
    for (int i = 0; i < emotionsTree.getNumChildren(); ++i) {
      auto emotionTree = emotionsTree.getChild(i);
      if (emotionTree.hasType("Emotion")) {
        int emotionId = static_cast<int>(emotionTree.getProperty("id"));
        data.selectedEmotionIds.push_back(emotionId);
      }
    }
  }

  if (tree.hasProperty("primaryEmotionId")) {
    data.primaryEmotionId =
        static_cast<int>(tree.getProperty("primaryEmotionId"));
  }

  return data;
}

juce::var ProjectManager::ProjectData::toJson() const {
  juce::DynamicObject::Ptr obj = new juce::DynamicObject();

  // Metadata
  obj->setProperty("version", juce::String(versionMajor) + "." +
                                  juce::String(versionMinor));
  obj->setProperty("name", name);
  obj->setProperty("createdTime", createdTime.toMilliseconds());
  obj->setProperty("modifiedTime", modifiedTime.toMilliseconds());

  // Project settings
  obj->setProperty("tempo", tempo);
  juce::DynamicObject::Ptr timeSigObj = new juce::DynamicObject();
  timeSigObj->setProperty("numerator", timeSignature.numerator);
  timeSigObj->setProperty("denominator", timeSignature.denominator);
  obj->setProperty("timeSignature", juce::var(timeSigObj.get()));

  // Plugin state (serialize using existing PluginState::Preset serialization)
  obj->setProperty("pluginState", pluginState.toJson());

  // Generated MIDI (serialize full note data for all tracks)
  juce::DynamicObject::Ptr midiObj = new juce::DynamicObject();
  midiObj->setProperty("tempoBpm", generatedMidi.tempoBpm);
  midiObj->setProperty("bars", generatedMidi.bars);
  midiObj->setProperty("key", juce::String(generatedMidi.key));
  midiObj->setProperty("mode", juce::String(generatedMidi.mode));
  midiObj->setProperty("bpm", generatedMidi.bpm);
  midiObj->setProperty("lengthInBeats", generatedMidi.lengthInBeats);

  // Serialize all note tracks (create temporary ProjectManager to use instance
  // methods)
  ProjectManager tempManager;
  juce::Array<juce::var> melodyArray;
  for (const auto &note : generatedMidi.melody) {
    melodyArray.add(tempManager.serializeMidiNote(note));
  }
  midiObj->setProperty("melody", juce::var(melodyArray));

  juce::Array<juce::var> bassArray;
  for (const auto &note : generatedMidi.bass) {
    bassArray.add(tempManager.serializeMidiNote(note));
  }
  midiObj->setProperty("bass", juce::var(bassArray));

  juce::Array<juce::var> counterMelodyArray;
  for (const auto &note : generatedMidi.counterMelody) {
    counterMelodyArray.add(tempManager.serializeMidiNote(note));
  }
  midiObj->setProperty("counterMelody", juce::var(counterMelodyArray));

  juce::Array<juce::var> padArray;
  for (const auto &note : generatedMidi.pad) {
    padArray.add(tempManager.serializeMidiNote(note));
  }
  midiObj->setProperty("pad", juce::var(padArray));

  juce::Array<juce::var> stringsArray;
  for (const auto &note : generatedMidi.strings) {
    stringsArray.add(tempManager.serializeMidiNote(note));
  }
  midiObj->setProperty("strings", juce::var(stringsArray));

  juce::Array<juce::var> fillsArray;
  for (const auto &note : generatedMidi.fills) {
    fillsArray.add(tempManager.serializeMidiNote(note));
  }
  midiObj->setProperty("fills", juce::var(fillsArray));

  juce::Array<juce::var> rhythmArray;
  for (const auto &note : generatedMidi.rhythm) {
    rhythmArray.add(tempManager.serializeMidiNote(note));
  }
  midiObj->setProperty("rhythm", juce::var(rhythmArray));

  juce::Array<juce::var> drumGrooveArray;
  for (const auto &note : generatedMidi.drumGroove) {
    drumGrooveArray.add(tempManager.serializeMidiNote(note));
  }
  midiObj->setProperty("drumGroove", juce::var(drumGrooveArray));

  // Serialize chords
  juce::Array<juce::var> chordsArray;
  for (const auto &chord : generatedMidi.chords) {
    chordsArray.add(tempManager.serializeChord(chord));
  }
  midiObj->setProperty("chords", juce::var(chordsArray));

  obj->setProperty("generatedMidi", juce::var(midiObj.get()));

  // Vocal notes
  juce::Array<juce::var> vocalNotesArray;
  for (const auto &note : vocalNotes) {
    juce::DynamicObject::Ptr noteObj = new juce::DynamicObject();
    noteObj->setProperty("pitch", note.pitch);
    noteObj->setProperty("velocity", note.velocity);
    noteObj->setProperty("startTick", static_cast<juce::int64>(note.startTick));
    noteObj->setProperty("durationTicks",
                         static_cast<juce::int64>(note.durationTicks));
    vocalNotesArray.add(juce::var(noteObj.get()));
  }
  obj->setProperty("vocalNotes", juce::var(vocalNotesArray));

  // Lyrics
  juce::Array<juce::var> lyricsArray;
  for (const auto &lyric : lyrics) {
    lyricsArray.add(lyric);
  }
  obj->setProperty("lyrics", juce::var(lyricsArray));

  // Emotion selections
  juce::Array<juce::var> emotionsArray;
  for (int emotionId : selectedEmotionIds) {
    emotionsArray.add(emotionId);
  }
  obj->setProperty("selectedEmotionIds", juce::var(emotionsArray));

  if (primaryEmotionId.has_value()) {
    obj->setProperty("primaryEmotionId", *primaryEmotionId);
  }

  return juce::var(obj.get());
}

std::optional<ProjectManager::ProjectData>
ProjectManager::ProjectData::fromJson(const juce::var &json) {
  if (!json.isObject()) {
    return std::nullopt;
  }

  auto *obj = json.getDynamicObject();
  if (!obj) {
    return std::nullopt;
  }

  ProjectData data;

  // Metadata
  if (obj->hasProperty("version")) {
    juce::String version = obj->getProperty("version").toString();
    int dotPos = version.indexOfChar('.');
    if (dotPos > 0) {
      data.versionMajor = version.substring(0, dotPos).getIntValue();
      data.versionMinor = version.substring(dotPos + 1).getIntValue();
    }
  }
  data.name = obj->getProperty("name").toString();
  if (obj->hasProperty("createdTime")) {
    juce::int64 ms = static_cast<juce::int64>(obj->getProperty("createdTime"));
    data.createdTime = juce::Time(ms);
  }
  if (obj->hasProperty("modifiedTime")) {
    juce::int64 ms = static_cast<juce::int64>(obj->getProperty("modifiedTime"));
    data.modifiedTime = juce::Time(ms);
  }

  // Project settings
  data.tempo = static_cast<float>(obj->getProperty("tempo"));
  if (obj->hasProperty("timeSignature")) {
    auto timeSigVar = obj->getProperty("timeSignature");
    if (timeSigVar.isObject()) {
      auto *timeSigObj = timeSigVar.getDynamicObject();
      if (timeSigObj) {
        data.timeSignature.numerator =
            static_cast<int>(timeSigObj->getProperty("numerator"));
        data.timeSignature.denominator =
            static_cast<int>(timeSigObj->getProperty("denominator"));
      }
    }
  }

  // Plugin state (deserialize using existing PluginState::Preset
  // deserialization)
  if (obj->hasProperty("pluginState")) {
    auto preset =
        kelly::PluginState::Preset::fromJson(obj->getProperty("pluginState"));
    if (preset.has_value()) {
      // Store full preset (including CassetteState)
      data.pluginState = *preset;
    }
  }

  // Generated MIDI (deserialize full note data for all tracks)
  if (obj->hasProperty("generatedMidi")) {
    auto midiVar = obj->getProperty("generatedMidi");
    if (midiVar.isObject()) {
      auto *midiObj = midiVar.getDynamicObject();
      if (midiObj) {
        data.generatedMidi.tempoBpm =
            static_cast<int>(midiObj->getProperty("tempoBpm"));
        data.generatedMidi.bars =
            static_cast<int>(midiObj->getProperty("bars"));
        data.generatedMidi.key =
            midiObj->getProperty("key").toString().toStdString();
        data.generatedMidi.mode =
            midiObj->getProperty("mode").toString().toStdString();
        data.generatedMidi.bpm =
            static_cast<float>(midiObj->getProperty("bpm"));
        data.generatedMidi.lengthInBeats =
            static_cast<double>(midiObj->getProperty("lengthInBeats"));

        // Deserialize all note tracks
        ProjectManager tempManager;
        if (midiObj->hasProperty("melody")) {
          auto melodyVar = midiObj->getProperty("melody");
          if (melodyVar.isArray()) {
            auto *array = melodyVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                data.generatedMidi.melody.push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("bass")) {
          auto bassVar = midiObj->getProperty("bass");
          if (bassVar.isArray()) {
            auto *array = bassVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                data.generatedMidi.bass.push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("counterMelody")) {
          auto counterMelodyVar = midiObj->getProperty("counterMelody");
          if (counterMelodyVar.isArray()) {
            auto *array = counterMelodyVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                data.generatedMidi.counterMelody.push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("pad")) {
          auto padVar = midiObj->getProperty("pad");
          if (padVar.isArray()) {
            auto *array = padVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                data.generatedMidi.pad.push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("strings")) {
          auto stringsVar = midiObj->getProperty("strings");
          if (stringsVar.isArray()) {
            auto *array = stringsVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                data.generatedMidi.strings.push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("fills")) {
          auto fillsVar = midiObj->getProperty("fills");
          if (fillsVar.isArray()) {
            auto *array = fillsVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                data.generatedMidi.fills.push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("rhythm")) {
          auto rhythmVar = midiObj->getProperty("rhythm");
          if (rhythmVar.isArray()) {
            auto *array = rhythmVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                data.generatedMidi.rhythm.push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("drumGroove")) {
          auto drumGrooveVar = midiObj->getProperty("drumGroove");
          if (drumGrooveVar.isArray()) {
            auto *array = drumGrooveVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                data.generatedMidi.drumGroove.push_back(note);
              }
            }
          }
        }

        // Deserialize chords
        if (midiObj->hasProperty("chords")) {
          auto chordsVar = midiObj->getProperty("chords");
          if (chordsVar.isArray()) {
            auto *array = chordsVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              Chord chord;
              if (tempManager.deserializeChord(array->getReference(i), chord)) {
                data.generatedMidi.chords.push_back(chord);
              }
            }
          }
        }
      }
    }
  }

  // Vocal notes
  if (obj->hasProperty("vocalNotes")) {
    auto vocalNotesVar = obj->getProperty("vocalNotes");
    if (vocalNotesVar.isArray()) {
      auto *array = vocalNotesVar.getArray();
      for (int i = 0; i < array->size(); ++i) {
        auto noteVar = array->getReference(i);
        if (noteVar.isObject()) {
          auto *noteObj = noteVar.getDynamicObject();
          if (noteObj) {
            MidiNote note;
            note.pitch = static_cast<int>(noteObj->getProperty("pitch"));
            note.velocity = static_cast<int>(noteObj->getProperty("velocity"));
            note.startTick = static_cast<int>(noteObj->getProperty("startTick")
                                                  .toString()
                                                  .getLargeIntValue());
            note.durationTicks =
                static_cast<int>(noteObj->getProperty("durationTicks")
                                     .toString()
                                     .getLargeIntValue());
            data.vocalNotes.push_back(note);
          }
        }
      }
    }
  }

  // Lyrics
  if (obj->hasProperty("lyrics")) {
    auto lyricsVar = obj->getProperty("lyrics");
    if (lyricsVar.isArray()) {
      auto *array = lyricsVar.getArray();
      for (int i = 0; i < array->size(); ++i) {
        data.lyrics.push_back(array->getReference(i).toString());
      }
    }
  }

  // Emotion selections
  if (obj->hasProperty("selectedEmotionIds")) {
    auto emotionsVar = obj->getProperty("selectedEmotionIds");
    if (emotionsVar.isArray()) {
      auto *array = emotionsVar.getArray();
      for (int i = 0; i < array->size(); ++i) {
        data.selectedEmotionIds.push_back(
            static_cast<int>(array->getReference(i)));
      }
    }
  }

  if (obj->hasProperty("primaryEmotionId")) {
    data.primaryEmotionId =
        static_cast<int>(obj->getProperty("primaryEmotionId"));
  }

  return data;
}

//==============================================================================
// Helper Methods
//==============================================================================

juce::var
ProjectManager::serializeGeneratedMidi(const GeneratedMidi &midi) const {
  juce::DynamicObject::Ptr obj = new juce::DynamicObject();

  obj->setProperty("tempoBpm", midi.tempoBpm);
  obj->setProperty("bars", midi.bars);
  obj->setProperty("key", juce::String(midi.key));
  obj->setProperty("mode", juce::String(midi.mode));
  obj->setProperty("bpm", midi.bpm);
  obj->setProperty("lengthInBeats", midi.lengthInBeats);

  // Serialize tracks (simplified - store note counts for now)
  // Full implementation would serialize all notes
  obj->setProperty("melodyNoteCount", static_cast<int>(midi.melody.size()));
  obj->setProperty("bassNoteCount", static_cast<int>(midi.bass.size()));
  obj->setProperty("chordCount", static_cast<int>(midi.chords.size()));

  return juce::var(obj.get());
}

bool ProjectManager::deserializeGeneratedMidi(const juce::var &json,
                                              GeneratedMidi &outMidi) const {
  if (!json.isObject()) {
    return false;
  }

  auto *obj = json.getDynamicObject();
  if (!obj) {
    return false;
  }

  outMidi.tempoBpm = static_cast<int>(obj->getProperty("tempoBpm"));
  outMidi.bars = static_cast<int>(obj->getProperty("bars"));
  outMidi.key = obj->getProperty("key").toString().toStdString();
  outMidi.mode = obj->getProperty("mode").toString().toStdString();
  outMidi.bpm = static_cast<float>(obj->getProperty("bpm"));
  outMidi.lengthInBeats =
      static_cast<double>(obj->getProperty("lengthInBeats"));

  // Note: Full deserialization would restore all notes
  // For now, we just restore metadata
  // In production, you'd want to serialize/deserialize all notes

  return true;
}

juce::var ProjectManager::serializeMidiNote(const MidiNote &note) const {
  juce::DynamicObject::Ptr obj = new juce::DynamicObject();
  obj->setProperty("pitch", note.pitch);
  obj->setProperty("velocity", note.velocity);
  obj->setProperty("startTick", static_cast<juce::int64>(note.startTick));
  obj->setProperty("durationTicks",
                   static_cast<juce::int64>(note.durationTicks));
  return juce::var(obj.get());
}

bool ProjectManager::deserializeMidiNote(const juce::var &json,
                                         MidiNote &outNote) const {
  if (!json.isObject()) {
    return false;
  }

  auto *obj = json.getDynamicObject();
  if (!obj) {
    return false;
  }

  outNote.pitch = static_cast<int>(obj->getProperty("pitch"));
  outNote.velocity = static_cast<int>(obj->getProperty("velocity"));
  outNote.startTick = static_cast<int>(
      obj->getProperty("startTick").toString().getLargeIntValue());
  outNote.durationTicks = static_cast<int>(
      obj->getProperty("durationTicks").toString().getLargeIntValue());

  return true;
}

juce::var ProjectManager::serializeChord(const Chord &chord) const {
  juce::DynamicObject::Ptr obj = new juce::DynamicObject();
  obj->setProperty("symbol", juce::String(chord.symbol));
  obj->setProperty("root", juce::String(chord.root));
  obj->setProperty("quality", juce::String(chord.quality));
  obj->setProperty("startBeat", chord.startBeat);
  obj->setProperty("duration", chord.duration);

  juce::Array<juce::var> pitchesArray;
  for (int pitch : chord.pitches) {
    pitchesArray.add(pitch);
  }
  obj->setProperty("pitches", juce::var(pitchesArray));

  return juce::var(obj.get());
}

bool ProjectManager::deserializeChord(const juce::var &json,
                                      Chord &outChord) const {
  if (!json.isObject()) {
    return false;
  }

  auto *obj = json.getDynamicObject();
  if (!obj) {
    return false;
  }

  outChord.symbol = obj->getProperty("symbol").toString().toStdString();
  outChord.root = obj->getProperty("root").toString().toStdString();
  outChord.quality = obj->getProperty("quality").toString().toStdString();
  outChord.startBeat = static_cast<double>(obj->getProperty("startBeat"));
  outChord.duration = static_cast<double>(obj->getProperty("duration"));

  if (obj->hasProperty("pitches")) {
    auto pitchesVar = obj->getProperty("pitches");
    if (pitchesVar.isArray()) {
      auto *array = pitchesVar.getArray();
      for (int i = 0; i < array->size(); ++i) {
        outChord.pitches.push_back(static_cast<int>(array->getReference(i)));
      }
    }
  }

  return true;
}

bool ProjectManager::migrateProjectVersion(juce::ValueTree &tree,
                                           int fromVersion,
                                           int toVersion) const {
  // Version migration logic
  // For now, version 1.0 is the only version
  if (fromVersion == 1 && toVersion == 1) {
    return true; // No migration needed
  }

  // Future: Add migration logic for version 1.0 -> 1.1, etc.
  return false;
}

} // namespace midikompanion
