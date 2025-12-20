/**
 * @file ProjectManager.cpp
 * @brief Project management implementation for save/load functionality
 */

#include "ProjectManager.h"
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace midikompanion {

//==============================================================================
// Version helpers
//==============================================================================

juce::String ProjectVersion::toString() const {
    return juce::String(major) + "." + juce::String(minor) + "." + juce::String(patch);
}

ProjectVersion ProjectVersion::fromString(const juce::String& str) {
    ProjectVersion version;
    std::sscanf(str.toStdString().c_str(), "%d.%d.%d", &version.major, &version.minor, &version.patch);
    return version;
}

bool ProjectVersion::operator<(const ProjectVersion& other) const {
    if (major != other.major) return major < other.major;
    if (minor != other.minor) return minor < other.minor;
    return patch < other.patch;
}

bool ProjectVersion::operator==(const ProjectVersion& other) const {
    return major == other.major && minor == other.minor && patch == other.patch;
}

//==============================================================================
// Construction
//==============================================================================

ProjectManager::ProjectManager() = default;

ProjectVersion ProjectManager::getCurrentVersion() {
    return {1, 0, 0};
}

bool ProjectManager::isValidProjectFile(const juce::File& file) {
    return validateFile(file);
}

//==============================================================================
// Save/Load Operations
//==============================================================================

bool ProjectManager::saveProject(const juce::File& file, const ProjectData& data) {
    const juce::ScopedLock lock(lock_);
    lastError_.clear();
    
    // Update modification timestamp
    ProjectData mutableData = data;
    mutableData.metadata.modifiedDate = getCurrentTimestamp();
    
    // Set creation date if not set
    if (mutableData.metadata.createdDate.empty()) {
        mutableData.metadata.createdDate = mutableData.metadata.modifiedDate;
    }
    
    // Set current version
    mutableData.metadata.version = getCurrentVersion();
    
    // Convert to JSON
    juce::String json = projectToJson(mutableData);
    if (json.isEmpty()) {
        setError("Failed to serialize project data");
        return false;
    }
    
    // Create backup if file exists
    if (file.existsAsFile()) {
        createBackup(file);
    }
    
    // Write to file
    if (!file.replaceWithText(json)) {
        setError("Failed to write project file: " + file.getFullPathName());
        return false;
    }
    
    return true;
}

bool ProjectManager::loadProject(const juce::File& file, ProjectData& outData) {
    const juce::ScopedLock lock(lock_);
    lastError_.clear();
    
    // Check file exists
    if (!file.existsAsFile()) {
        setError("Project file not found: " + file.getFullPathName());
        return false;
    }
    
    // Validate file format
    if (!validateFile(file)) {
        return false;
    }
    
    // Read file contents
    juce::String json = file.loadFileAsString();
    if (json.isEmpty()) {
        setError("Failed to read project file or file is empty");
        return false;
    }
    
    // Parse JSON
    if (!jsonToProject(json, outData)) {
        return false;
    }
    
    // Check for version migration
    if (needsMigration(outData.metadata.version)) {
        if (!migrateProject(outData)) {
            setError("Failed to migrate project to current version");
            return false;
        }
    }
    
    return true;
}

bool ProjectManager::saveFromValueTree(const juce::File& file,
                                         const juce::AudioProcessorValueTreeState& state,
                                         const ProjectData& additionalData) {
    const juce::ScopedLock lock(lock_);
    ProjectData data = additionalData;
    
    // Extract parameters from ValueTree
    auto stateXml = state.copyState().createXml();
    if (stateXml) {
        // Store parameter values
        for (auto* param : state.processor.getParameters()) {
            auto* rangedParam = dynamic_cast<juce::RangedAudioParameter*>(param);
            if (rangedParam) {
                data.pluginState.parameters[rangedParam->getParameterID().toStdString()] = 
                    rangedParam->getValue();
            }
        }
    }
    
    return saveProject(file, data);
}

bool ProjectManager::loadToValueTree(const juce::File& file,
                                       juce::AudioProcessorValueTreeState& state,
                                       ProjectData& outData) {
    const juce::ScopedLock lock(lock_);
    if (!loadProject(file, outData)) {
        return false;
    }
    
    // Restore parameters to ValueTree
    for (const auto& [paramId, value] : outData.pluginState.parameters) {
        if (auto* param = state.getParameter(paramId)) {
            param->setValueNotifyingHost(value);
        }
    }
    
    return true;
}

//==============================================================================
// Version Migration
//==============================================================================

bool ProjectManager::needsMigration(const ProjectVersion& version) const {
    return version < getCurrentVersion();
}

bool ProjectManager::migrateProject(ProjectData& data) {
    // Migration from 0.9.x to 1.0.0
    if (data.metadata.version.major == 0) {
        // Add default emotion state if missing
        if (data.pluginState.emotionState.emotionTag.empty()) {
            data.pluginState.emotionState.emotionTag = "neutral";
            data.pluginState.emotionState.valence = 0.0f;
            data.pluginState.emotionState.arousal = 0.5f;
            data.pluginState.emotionState.intensity = 0.5f;
        }
        
        // Set default bars if not set
        if (data.pluginState.bars == 0) {
            data.pluginState.bars = 4;
        }
        
        data.metadata.version = {1, 0, 0};
    }
    
    // Future migrations would go here
    // if (data.metadata.version < ProjectVersion{1, 1, 0}) { ... }
    
    return true;
}

//==============================================================================
// File Operations
//==============================================================================

juce::File ProjectManager::createBackup(const juce::File& file) {
    if (!file.existsAsFile()) {
        return juce::File();
    }
    
    // Create backup filename with timestamp
    auto timestamp = juce::Time::getCurrentTime().formatted("%Y%m%d_%H%M%S");
    auto backupName = file.getFileNameWithoutExtension() + "_backup_" + 
                      timestamp + file.getFileExtension();
    auto backupFile = file.getSiblingFile(backupName);
    
    if (file.copyFileTo(backupFile)) {
        return backupFile;
    }
    
    return juce::File();
}

bool ProjectManager::validateFile(const juce::File& file) {
    // Check file extension
    if (file.getFileExtension().toLowerCase() != getFileExtension()) {
        // Try to parse anyway, might be a compatible format
    }
    
    // Check file size (max 100MB)
    if (file.getSize() > 100 * 1024 * 1024) {
        setError("Project file is too large (max 100MB)");
        return false;
    }
    
    // Try to parse as JSON
    juce::String content = file.loadFileAsString();
    auto parsed = juce::JSON::parse(content);
    
    if (parsed.isVoid()) {
        setError("Invalid project file format (not valid JSON)");
        return false;
    }
    
    // Check for required fields
    if (!parsed.hasProperty("metadata") || !parsed.hasProperty("pluginState")) {
        setError("Project file missing required fields");
        return false;
    }
    
    return true;
}

//==============================================================================
// Serialization Helpers
//==============================================================================

juce::String ProjectManager::projectToJson(const ProjectData& data) const {
    juce::DynamicObject::Ptr root = new juce::DynamicObject();
    
    // Version marker
    root->setProperty("version", juce::String(data.metadata.version.toString()));
    
    // Metadata
    root->setProperty("metadata", metadataToVar(data.metadata));
    
    // Tempo and time signature
    root->setProperty("tempo", data.tempo);
    
    juce::DynamicObject::Ptr timeSignature = new juce::DynamicObject();
    timeSignature->setProperty("numerator", data.timeSignature.numerator);
    timeSignature->setProperty("denominator", data.timeSignature.denominator);
    root->setProperty("timeSignature", juce::var(timeSignature.get()));
    
    // Plugin state
    root->setProperty("pluginState", pluginStateToVar(data.pluginState));
    
    // Tracks (generated MIDI)
    juce::Array<juce::var> tracksArray;
    for (const auto& track : data.tracks) {
        tracksArray.add(trackDataToVar(track));
    }
    root->setProperty("generatedMidi", tracksArray);
    
    // Vocal notes
    juce::Array<juce::var> vocalNotesArray;
    for (const auto& note : data.vocalNotes) {
        vocalNotesArray.add(vocalNoteToVar(note));
    }
    root->setProperty("vocalNotes", vocalNotesArray);
    
    // Lyrics
    juce::Array<juce::var> lyricsArray;
    for (const auto& line : data.lyrics) {
        lyricsArray.add(lyricLineToVar(line));
    }
    root->setProperty("lyrics", lyricsArray);
    
    // Project settings
    juce::DynamicObject::Ptr settings = new juce::DynamicObject();
    settings->setProperty("sampleRate", data.sampleRate);
    settings->setProperty("midiOutputDevice", juce::String(data.midiOutputDevice));
    root->setProperty("settings", juce::var(settings.get()));
    
    // Custom data
    if (!data.customData.empty()) {
        juce::DynamicObject::Ptr customObj = new juce::DynamicObject();
        for (const auto& [key, value] : data.customData) {
            customObj->setProperty(juce::Identifier(key), juce::String(value));
        }
        root->setProperty("customData", juce::var(customObj.get()));
    }
    
    return juce::JSON::toString(juce::var(root.get()), true);
}

bool ProjectManager::jsonToProject(const juce::String& json, ProjectData& outData) {
    auto parsed = juce::JSON::parse(json);
    
    if (parsed.isVoid()) {
        setError("Failed to parse project JSON");
        return false;
    }
    
    // Parse version
    if (parsed.hasProperty("version")) {
        auto versionStr = parsed["version"].toString().toStdString();
        sscanf(versionStr.c_str(), "%d.%d.%d", 
               &outData.metadata.version.major,
               &outData.metadata.version.minor,
               &outData.metadata.version.patch);
    }
    
    // Parse metadata
    if (parsed.hasProperty("metadata")) {
        if (!varToMetadata(parsed["metadata"], outData.metadata)) {
            setError("Failed to parse project metadata");
            return false;
        }
    }
    
    // Parse tempo
    if (parsed.hasProperty("tempo")) {
        outData.tempo = static_cast<double>(parsed["tempo"]);
    }
    
    // Parse time signature
    if (parsed.hasProperty("timeSignature")) {
        auto ts = parsed["timeSignature"];
        if (ts.hasProperty("numerator")) {
            outData.timeSignature.numerator = static_cast<int>(ts["numerator"]);
        }
        if (ts.hasProperty("denominator")) {
            outData.timeSignature.denominator = static_cast<int>(ts["denominator"]);
        }
    }
    
    // Parse plugin state
    if (parsed.hasProperty("pluginState")) {
        if (!varToPluginState(parsed["pluginState"], outData.pluginState)) {
            setError("Failed to parse plugin state");
            return false;
        }
    }
    
    // Parse tracks
    outData.tracks.clear();
    if (parsed.hasProperty("generatedMidi")) {
        auto tracksVar = parsed["generatedMidi"];
        if (tracksVar.isArray()) {
            for (int i = 0; i < tracksVar.size(); ++i) {
                TrackData track;
                if (varToTrackData(tracksVar[i], track)) {
                    outData.tracks.push_back(track);
                }
            }
        }
    }
    
    // Parse vocal notes
    outData.vocalNotes.clear();
    if (parsed.hasProperty("vocalNotes")) {
        auto notesVar = parsed["vocalNotes"];
        if (notesVar.isArray()) {
            for (int i = 0; i < notesVar.size(); ++i) {
                VocalNote note;
                if (varToVocalNote(notesVar[i], note)) {
                    outData.vocalNotes.push_back(note);
                }
            }
        }
    }
    
    // Parse lyrics
    outData.lyrics.clear();
    if (parsed.hasProperty("lyrics")) {
        auto lyricsVar = parsed["lyrics"];
        if (lyricsVar.isArray()) {
            for (int i = 0; i < lyricsVar.size(); ++i) {
                LyricLine line;
                if (varToLyricLine(lyricsVar[i], line)) {
                    outData.lyrics.push_back(line);
                }
            }
        }
    }
    
    // Parse settings
    if (parsed.hasProperty("settings")) {
        auto settings = parsed["settings"];
        if (settings.hasProperty("sampleRate")) {
            outData.sampleRate = static_cast<int>(settings["sampleRate"]);
        }
        if (settings.hasProperty("midiOutputDevice")) {
            outData.midiOutputDevice = settings["midiOutputDevice"].toString().toStdString();
        }
    }
    
    // Parse custom data
    outData.customData.clear();
    if (parsed.hasProperty("customData")) {
        auto custom = parsed["customData"];
        if (auto* obj = custom.getDynamicObject()) {
            for (const auto& prop : obj->getProperties()) {
                outData.customData[prop.name.toString().toStdString()] = 
                    prop.value.toString().toStdString();
            }
        }
    }
    
    return true;
}

juce::var ProjectManager::metadataToVar(const ProjectMetadata& metadata) const {
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("name", juce::String(metadata.name));
    obj->setProperty("author", juce::String(metadata.author));
    obj->setProperty("created", juce::String(metadata.createdDate));
    obj->setProperty("modified", juce::String(metadata.modifiedDate));
    obj->setProperty("version", juce::String(metadata.version.toString()));
    return juce::var(obj.get());
}

bool ProjectManager::varToMetadata(const juce::var& var, ProjectMetadata& outMetadata) const {
    if (!var.isObject()) return false;
    
    if (var.hasProperty("name")) {
        outMetadata.name = var["name"].toString().toStdString();
    }
    if (var.hasProperty("author")) {
        outMetadata.author = var["author"].toString().toStdString();
    }
    if (var.hasProperty("created")) {
        outMetadata.createdDate = var["created"].toString().toStdString();
    }
    if (var.hasProperty("modified")) {
        outMetadata.modifiedDate = var["modified"].toString().toStdString();
    }
    if (var.hasProperty("version")) {
        auto versionStr = var["version"].toString().toStdString();
        sscanf(versionStr.c_str(), "%d.%d.%d",
               &outMetadata.version.major,
               &outMetadata.version.minor,
               &outMetadata.version.patch);
    }
    
    return true;
}

juce::var ProjectManager::emotionStateToVar(const EmotionState& state) const {
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("nodeId", state.nodeId);
    obj->setProperty("valence", state.valence);
    obj->setProperty("arousal", state.arousal);
    obj->setProperty("dominance", state.dominance);
    obj->setProperty("intensity", state.intensity);
    obj->setProperty("emotionTag", juce::String(state.emotionTag));
    
    // Related node IDs
    juce::Array<juce::var> relatedArray;
    for (int id : state.relatedNodeIds) {
        relatedArray.add(id);
    }
    obj->setProperty("relatedNodeIds", relatedArray);
    
    // ML embedding (if present)
    if (state.mlEmbedding.has_value()) {
        juce::Array<juce::var> embeddingArray;
        for (float val : state.mlEmbedding.value()) {
            embeddingArray.add(val);
        }
        obj->setProperty("mlEmbedding", embeddingArray);
    }
    
    if (state.mlConfidence.has_value()) {
        obj->setProperty("mlConfidence", state.mlConfidence.value());
    }
    
    return juce::var(obj.get());
}

bool ProjectManager::varToEmotionState(const juce::var& var, EmotionState& outState) const {
    if (!var.isObject()) return false;
    
    if (var.hasProperty("nodeId")) {
        outState.nodeId = static_cast<int>(var["nodeId"]);
    }
    if (var.hasProperty("valence")) {
        outState.valence = static_cast<float>(var["valence"]);
    }
    if (var.hasProperty("arousal")) {
        outState.arousal = static_cast<float>(var["arousal"]);
    }
    if (var.hasProperty("dominance")) {
        outState.dominance = static_cast<float>(var["dominance"]);
    }
    if (var.hasProperty("intensity")) {
        outState.intensity = static_cast<float>(var["intensity"]);
    }
    if (var.hasProperty("emotionTag")) {
        outState.emotionTag = var["emotionTag"].toString().toStdString();
    }
    
    // Related node IDs
    outState.relatedNodeIds.clear();
    if (var.hasProperty("relatedNodeIds")) {
        auto relatedVar = var["relatedNodeIds"];
        if (relatedVar.isArray()) {
            for (int i = 0; i < relatedVar.size(); ++i) {
                outState.relatedNodeIds.push_back(static_cast<int>(relatedVar[i]));
            }
        }
    }
    
    // ML embedding
    if (var.hasProperty("mlEmbedding")) {
        auto embeddingVar = var["mlEmbedding"];
        if (embeddingVar.isArray()) {
            std::vector<float> embedding;
            for (int i = 0; i < embeddingVar.size(); ++i) {
                embedding.push_back(static_cast<float>(embeddingVar[i]));
            }
            outState.mlEmbedding = embedding;
        }
    }
    
    if (var.hasProperty("mlConfidence")) {
        outState.mlConfidence = static_cast<float>(var["mlConfidence"]);
    }
    
    return true;
}

juce::var ProjectManager::trackDataToVar(const TrackData& track) const {
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("name", juce::String(track.name));
    obj->setProperty("type", juce::String(track.type));
    obj->setProperty("midiChannel", track.midiChannel);
    obj->setProperty("volume", track.volume);
    obj->setProperty("pan", track.pan);
    obj->setProperty("muted", track.muted);
    obj->setProperty("soloed", track.soloed);
    
    // Notes
    juce::Array<juce::var> notesArray;
    for (const auto& note : track.notes) {
        juce::DynamicObject::Ptr noteObj = new juce::DynamicObject();
        noteObj->setProperty("pitch", note.pitch);
        noteObj->setProperty("velocity", note.velocity);
        noteObj->setProperty("startBeat", note.startBeat);
        noteObj->setProperty("durationBeats", note.durationBeats);
        noteObj->setProperty("channel", note.channel);
        notesArray.add(juce::var(noteObj.get()));
    }
    obj->setProperty("notes", notesArray);
    
    return juce::var(obj.get());
}

bool ProjectManager::varToTrackData(const juce::var& var, TrackData& outTrack) const {
    if (!var.isObject()) return false;
    
    if (var.hasProperty("name")) {
        outTrack.name = var["name"].toString().toStdString();
    }
    if (var.hasProperty("type")) {
        outTrack.type = var["type"].toString().toStdString();
    }
    if (var.hasProperty("midiChannel")) {
        outTrack.midiChannel = static_cast<int>(var["midiChannel"]);
    }
    if (var.hasProperty("volume")) {
        outTrack.volume = static_cast<float>(var["volume"]);
    }
    if (var.hasProperty("pan")) {
        outTrack.pan = static_cast<float>(var["pan"]);
    }
    if (var.hasProperty("muted")) {
        outTrack.muted = static_cast<bool>(var["muted"]);
    }
    if (var.hasProperty("soloed")) {
        outTrack.soloed = static_cast<bool>(var["soloed"]);
    }
    
    // Notes
    outTrack.notes.clear();
    if (var.hasProperty("notes")) {
        auto notesVar = var["notes"];
        if (notesVar.isArray()) {
            for (int i = 0; i < notesVar.size(); ++i) {
                auto noteVar = notesVar[i];
                MidiNoteData note;
                if (noteVar.hasProperty("pitch")) {
                    note.pitch = static_cast<int>(noteVar["pitch"]);
                }
                if (noteVar.hasProperty("velocity")) {
                    note.velocity = static_cast<int>(noteVar["velocity"]);
                }
                if (noteVar.hasProperty("startBeat")) {
                    note.startBeat = static_cast<double>(noteVar["startBeat"]);
                }
                if (noteVar.hasProperty("durationBeats")) {
                    note.durationBeats = static_cast<double>(noteVar["durationBeats"]);
                }
                if (noteVar.hasProperty("channel")) {
                    note.channel = static_cast<int>(noteVar["channel"]);
                }
                outTrack.notes.push_back(note);
            }
        }
    }
    
    return true;
}

juce::var ProjectManager::vocalNoteToVar(const VocalNote& note) const {
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("syllable", juce::String(note.syllable));
    obj->setProperty("startBeat", note.startBeat);
    obj->setProperty("durationBeats", note.durationBeats);
    obj->setProperty("pitch", note.pitch);
    obj->setProperty("phonemeBlend", note.phonemeBlend);
    return juce::var(obj.get());
}

bool ProjectManager::varToVocalNote(const juce::var& var, VocalNote& outNote) const {
    if (!var.isObject()) return false;
    
    if (var.hasProperty("syllable")) {
        outNote.syllable = var["syllable"].toString().toStdString();
    }
    if (var.hasProperty("startBeat")) {
        outNote.startBeat = static_cast<double>(var["startBeat"]);
    }
    if (var.hasProperty("durationBeats")) {
        outNote.durationBeats = static_cast<double>(var["durationBeats"]);
    }
    if (var.hasProperty("pitch")) {
        outNote.pitch = static_cast<int>(var["pitch"]);
    }
    if (var.hasProperty("phonemeBlend")) {
        outNote.phonemeBlend = static_cast<float>(var["phonemeBlend"]);
    }
    
    return true;
}

juce::var ProjectManager::lyricLineToVar(const LyricLine& line) const {
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("text", juce::String(line.text));
    obj->setProperty("startBeat", line.startBeat);
    obj->setProperty("endBeat", line.endBeat);
    
    // Vocal notes
    juce::Array<juce::var> notesArray;
    for (const auto& note : line.vocalNotes) {
        notesArray.add(vocalNoteToVar(note));
    }
    obj->setProperty("vocalNotes", notesArray);
    
    return juce::var(obj.get());
}

bool ProjectManager::varToLyricLine(const juce::var& var, LyricLine& outLine) const {
    if (!var.isObject()) return false;
    
    if (var.hasProperty("text")) {
        outLine.text = var["text"].toString().toStdString();
    }
    if (var.hasProperty("startBeat")) {
        outLine.startBeat = static_cast<double>(var["startBeat"]);
    }
    if (var.hasProperty("endBeat")) {
        outLine.endBeat = static_cast<double>(var["endBeat"]);
    }
    
    // Vocal notes
    outLine.vocalNotes.clear();
    if (var.hasProperty("vocalNotes")) {
        auto notesVar = var["vocalNotes"];
        if (notesVar.isArray()) {
            for (int i = 0; i < notesVar.size(); ++i) {
                VocalNote note;
                if (varToVocalNote(notesVar[i], note)) {
                    outLine.vocalNotes.push_back(note);
                }
            }
        }
    }
    
    return true;
}

juce::var ProjectManager::pluginStateToVar(const PluginState& state) const {
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    
    // Emotion state
    obj->setProperty("emotionState", emotionStateToVar(state.emotionState));
    
    // Parameters map
    juce::DynamicObject::Ptr paramsObj = new juce::DynamicObject();
    for (const auto& [key, value] : state.parameters) {
        paramsObj->setProperty(juce::Identifier(key), value);
    }
    obj->setProperty("parameters", juce::var(paramsObj.get()));
    
    // Recorder state
    obj->setProperty("isRecording", state.isRecording);
    obj->setProperty("isPlaying", state.isPlaying);
    obj->setProperty("playheadPosition", state.playheadPosition);
    
    // Wound description
    obj->setProperty("woundDescription", juce::String(state.woundDescription));
    
    // Generation settings
    obj->setProperty("bars", state.bars);
    obj->setProperty("enableCloud", state.enableCloud);
    obj->setProperty("generationRate", state.generationRate);
    
    return juce::var(obj.get());
}

bool ProjectManager::varToPluginState(const juce::var& var, PluginState& outState) const {
    if (!var.isObject()) return false;
    
    // Emotion state
    if (var.hasProperty("emotionState")) {
        varToEmotionState(var["emotionState"], outState.emotionState);
    }
    
    // Parameters map
    outState.parameters.clear();
    if (var.hasProperty("parameters")) {
        auto params = var["parameters"];
        if (auto* obj = params.getDynamicObject()) {
            for (const auto& prop : obj->getProperties()) {
                outState.parameters[prop.name.toString().toStdString()] = 
                    static_cast<float>(prop.value);
            }
        }
    }
    
    // Recorder state
    if (var.hasProperty("isRecording")) {
        outState.isRecording = static_cast<bool>(var["isRecording"]);
    }
    if (var.hasProperty("isPlaying")) {
        outState.isPlaying = static_cast<bool>(var["isPlaying"]);
    }
    if (var.hasProperty("playheadPosition")) {
        outState.playheadPosition = static_cast<double>(var["playheadPosition"]);
    }
    
    // Wound description
    if (var.hasProperty("woundDescription")) {
        outState.woundDescription = var["woundDescription"].toString().toStdString();
    }
    
    // Generation settings
    if (var.hasProperty("bars")) {
        outState.bars = static_cast<int>(var["bars"]);
    }
    if (var.hasProperty("enableCloud")) {
        outState.enableCloud = static_cast<bool>(var["enableCloud"]);
    }
    if (var.hasProperty("generationRate")) {
        outState.generationRate = static_cast<int>(var["generationRate"]);
    }
    
    return true;
}

//==============================================================================
// Helper Methods
//==============================================================================

std::string ProjectManager::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

void ProjectManager::setError(const juce::String& error) {
    lastError_ = error;
    DBG("ProjectManager Error: " + error);
}

} // namespace midikompanion
