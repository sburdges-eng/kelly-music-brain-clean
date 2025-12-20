/**
 * @file StemExporter.cpp
 * @brief Stem exporter implementation
 */

#include "daiw/export/StemExporter.h"
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace daiw {
namespace export_ns {

namespace fs = std::filesystem;

std::vector<StemExportResult> StemExporter::exportAllStems(
    const project::ProjectFile& project,
    const std::string& outputDirectory,
    const ExportOptions& options) {
    
    const auto& tracks = project.getTracks();
    std::vector<size_t> allIndices(tracks.size());
    for (size_t i = 0; i < tracks.size(); ++i) {
        allIndices[i] = i;
    }
    
    return exportSelectedStems(project, allIndices, outputDirectory, options);
}

std::vector<StemExportResult> StemExporter::exportSelectedStems(
    const project::ProjectFile& project,
    const std::vector<size_t>& trackIndices,
    const std::string& outputDirectory,
    const ExportOptions& options) {
    
    std::vector<StemExportResult> results;
    const auto& tracks = project.getTracks();
    
    // Create output directory if it doesn't exist
    fs::create_directories(outputDirectory);
    
    for (size_t i = 0; i < trackIndices.size(); ++i) {
        size_t trackIdx = trackIndices[i];
        
        if (trackIdx >= tracks.size()) {
            StemExportResult result;
            result.success = false;
            result.errorMessage = "Track index out of range";
            results.push_back(result);
            continue;
        }
        
        const auto& track = tracks[trackIdx];
        
        // Progress callback
        if (progressCallback_) {
            progressCallback_(i, trackIndices.size(), track.name);
        }
        
        // Generate filename
        std::string filepath = generateStemFilename(
            track.name, trackIdx, outputDirectory, options.format, options.filenameSuffix);
        
        // Export track
        auto result = exportTrack(track, filepath, options);
        results.push_back(result);
    }
    
    return results;
}

StemExportResult StemExporter::exportTrack(
    const project::Track& track,
    const std::string& filepath,
    const ExportOptions& options) {
    
    StemExportResult result;
    result.trackName = track.name;
    result.filepath = filepath;
    
    try {
        audio::AudioFile audioFile;
        
        if (track.type == project::TrackType::Audio) {
            // Load audio file
            if (!track.audioFilePath.empty()) {
                if (!audioFile.read(track.audioFilePath)) {
                    result.success = false;
                    result.errorMessage = "Failed to read audio file: " + track.audioFilePath;
                    return result;
                }
            } else {
                result.success = false;
                result.errorMessage = "Audio track has no file path";
                return result;
            }
        } else if (track.type == project::TrackType::MIDI) {
            // Render MIDI to audio (stub)
            audioFile = renderMidiTrack(track, 10.0, options.sampleRate);
            
            if (audioFile.getData().empty()) {
                result.success = false;
                result.errorMessage = "MIDI rendering not implemented (stub only)";
                return result;
            }
        } else {
            result.success = false;
            result.errorMessage = "Unsupported track type";
            return result;
        }
        
        // Apply track volume/pan (basic mixing)
        auto& data = audioFile.getData();
        for (auto& sample : data) {
            sample *= track.volume;
        }
        
        // Normalize if requested
        if (options.normalizeStems) {
            normalizeAudio(audioFile);
        }
        
        // Write file
        if (!audioFile.write(filepath, options.format, options.sampleFormat)) {
            result.success = false;
            result.errorMessage = "Failed to write audio file";
            return result;
        }
        
        result.success = true;
        result.numSamples = audioFile.getInfo().numSamples;
        result.durationSeconds = audioFile.getInfo().durationSeconds;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
    }
    
    return result;
}

audio::AudioFile StemExporter::renderMidiTrack(
    const project::Track& track,
    double durationSeconds,
    SampleRate sampleRate) {
    
    // Basic MIDI rendering implementation using simple synthesizer
    // For production: integrate a full synth/sampler engine (JUCE, SFizz, etc.)
    
    if (track.midiSequence.empty()) {
        // Return empty audio file if no MIDI events
        audio::AudioFile emptyFile;
        emptyFile.setData({}, 1, sampleRate);
        return emptyFile;
    }
    
    // Calculate duration from MIDI sequence if not provided
    if (durationSeconds <= 0.0) {
        // Estimate duration from last MIDI event
        // Assuming 120 BPM and PPQ of 480
        const int ppq = track.midiSequence.getPPQ();
        const double bpm = 120.0;  // Default tempo
        TickCount lastTick = track.midiSequence.getDuration();
        durationSeconds = (lastTick / static_cast<double>(ppq)) * (60.0 / bpm);
        durationSeconds = std::max(durationSeconds, 1.0);  // Minimum 1 second
    }
    
    const size_t numSamples = static_cast<size_t>(durationSeconds * sampleRate);
    std::vector<audio::Sample> audioData(numSamples, 0.0f);
    
    // Simple synthesizer: generate sine waves for each active note
    struct ActiveNote {
        MidiNote note;
        float startTime;
        float velocity;
        bool isActive;
    };
    
    std::vector<ActiveNote> activeNotes;
    const auto& messages = track.midiSequence.getMessages();
    
    // Convert MIDI ticks to sample positions
    const int ppq = track.midiSequence.getPPQ();
    const double bpm = 120.0;  // Default tempo - in production, get from project
    const double ticksPerSecond = (ppq * bpm) / 60.0;
    
    for (const auto& msg : messages) {
        double timeInSeconds = msg.getTimestamp() / ticksPerSecond;
        size_t samplePos = static_cast<size_t>(timeInSeconds * sampleRate);
        
        if (samplePos >= numSamples) continue;
        
        if (msg.isNoteOn() && msg.getVelocity() > 0) {
            // Note on
            ActiveNote note;
            note.note = msg.getNoteNumber();
            note.startTime = static_cast<float>(timeInSeconds);
            note.velocity = static_cast<float>(msg.getVelocity()) / 127.0f;
            note.isActive = true;
            activeNotes.push_back(note);
        } else if (msg.isNoteOff() || (msg.isNoteOn() && msg.getVelocity() == 0)) {
            // Note off - find and remove matching note
            for (auto it = activeNotes.begin(); it != activeNotes.end(); ++it) {
                if (it->note == msg.getNoteNumber() && it->isActive) {
                    it->isActive = false;
                    break;
                }
            }
        }
    }
    
    // Generate audio for active notes
    for (size_t sample = 0; sample < numSamples; ++sample) {
        float time = static_cast<float>(sample) / static_cast<float>(sampleRate);
        float sampleValue = 0.0f;
        
        for (auto& note : activeNotes) {
            if (!note.isActive) continue;
            
            float noteDuration = time - note.startTime;
            if (noteDuration < 0.0f) continue;
            
            // Calculate frequency from MIDI note number
            float frequency = 440.0f * std::pow(2.0f, (static_cast<float>(note.note) - 69.0f) / 12.0f);
            
            // Generate sine wave with envelope (simple ADSR-like)
            float amplitude = note.velocity;
            if (noteDuration < 0.01f) {
                // Attack: 10ms
                amplitude *= noteDuration / 0.01f;
            } else if (noteDuration > durationSeconds - 0.1f) {
                // Release: 100ms fade out
                float releaseTime = durationSeconds - noteDuration;
                amplitude *= std::max(0.0f, releaseTime / 0.1f);
            }
            
            // Generate sine wave
            constexpr float PI = 3.14159265358979323846f;
            float phase = 2.0f * PI * frequency * time;
            sampleValue += amplitude * std::sin(phase) * 0.2f;  // Scale down to prevent clipping
        }
        
        audioData[sample] = sampleValue;
    }
    
    // Create audio file
    audio::AudioFile audioFile;
    audioFile.setData(audioData, 1, sampleRate);  // Mono output
    return audioFile;
}

void StemExporter::normalizeAudio(audio::AudioFile& audio, float targetLevel) {
    auto& data = audio.getData();
    
    if (data.empty()) {
        return;
    }
    
    // Find peak
    float peak = 0.0f;
    for (const auto& sample : data) {
        peak = std::max(peak, std::abs(sample));
    }
    
    if (peak > 0.0f) {
        float gain = targetLevel / peak;
        for (auto& sample : data) {
            sample *= gain;
        }
    }
}

std::string StemExporter::generateStemFilename(
    const std::string& trackName,
    size_t trackIndex,
    const std::string& outputDirectory,
    audio::AudioFormat format,
    const std::string& suffix) {
    
    std::string sanitized = sanitizeFilename(trackName);
    if (sanitized.empty()) {
        sanitized = "Track_" + std::to_string(trackIndex);
    }
    
    std::string filename = sanitized;
    if (!suffix.empty()) {
        filename += "_" + suffix;
    }
    filename += getFileExtension(format);
    
    fs::path fullPath = fs::path(outputDirectory) / filename;
    return fullPath.string();
}

std::string StemExporter::sanitizeFilename(const std::string& name) {
    std::string result;
    for (char c : name) {
        if (std::isalnum(c) || c == '_' || c == '-' || c == ' ') {
            result += c;
        } else {
            result += '_';
        }
    }
    return result;
}

std::string StemExporter::getFileExtension(audio::AudioFormat format) {
    switch (format) {
        case audio::AudioFormat::WAV:  return ".wav";
        case audio::AudioFormat::AIFF: return ".aiff";
        case audio::AudioFormat::FLAC: return ".flac";
        default: return ".wav";
    }
}

} // namespace export_ns
} // namespace daiw
