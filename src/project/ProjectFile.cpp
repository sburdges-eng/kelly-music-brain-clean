/**
 * @file ProjectFile.cpp
 * @brief Project file implementation with JSON serialization
 *
 * Uses simple JSON formatting for now.
 * For production: consider using a JSON library (nlohmann/json, rapidjson, etc.)
 */

#include "daiw/project/ProjectFile.h"
#include <fstream>
#include <sstream>
#include <iomanip>

namespace daiw {
namespace project {

ProjectFile::ProjectFile() {
    tempo_.bpm = 120.0f;
    timeSignature_.numerator = 4;
    timeSignature_.denominator = 4;
}

bool ProjectFile::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return fromJSON(buffer.str());
}

bool ProjectFile::save(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    file << toJSON();
    return true;
}

std::string ProjectFile::toJSON() const {
    // Simple JSON formatting without external library
    // For production: use nlohmann::json or similar
    
    std::ostringstream json;
    json << std::fixed << std::setprecision(2);
    
    json << "{\n";
    
    // Metadata
    json << "  \"metadata\": {\n";
    json << "    \"name\": \"" << metadata_.name << "\",\n";
    json << "    \"author\": \"" << metadata_.author << "\",\n";
    json << "    \"created\": \"" << metadata_.createdDate << "\",\n";
    json << "    \"modified\": \"" << metadata_.modifiedDate << "\",\n";
    json << "    \"version\": \"" << metadata_.versionMajor << "." << metadata_.versionMinor << "\"\n";
    json << "  },\n";
    
    // Project settings
    json << "  \"settings\": {\n";
    json << "    \"tempo\": " << tempo_.bpm << ",\n";
    json << "    \"timeSignature\": \"" << static_cast<int>(timeSignature_.numerator) 
         << "/" << static_cast<int>(timeSignature_.denominator) << "\",\n";
    json << "    \"sampleRate\": " << sampleRate_ << "\n";
    json << "  },\n";
    
    // Mixer
    json << "  \"mixer\": {\n";
    json << "    \"masterVolume\": " << mixer_.masterVolume << ",\n";
    json << "    \"masterMuted\": " << (mixer_.masterMuted ? "true" : "false") << "\n";
    json << "  },\n";
    
    // Tracks
    json << "  \"tracks\": [\n";
    for (size_t i = 0; i < tracks_.size(); ++i) {
        const auto& track = tracks_[i];
        json << "    {\n";
        json << "      \"name\": \"" << track.name << "\",\n";
        json << "      \"type\": \"" << (track.type == TrackType::MIDI ? "midi" : 
                                        track.type == TrackType::Audio ? "audio" : "aux") << "\",\n";
        json << "      \"index\": " << track.index << ",\n";
        json << "      \"muted\": " << (track.muted ? "true" : "false") << ",\n";
        json << "      \"soloed\": " << (track.soloed ? "true" : "false") << ",\n";
        json << "      \"volume\": " << track.volume << ",\n";
        json << "      \"pan\": " << track.pan;
        
        if (track.type == TrackType::MIDI) {
            json << ",\n      \"midiEvents\": " << track.midiSequence.size();
        } else if (track.type == TrackType::Audio) {
            json << ",\n      \"audioFile\": \"" << track.audioFilePath << "\"";
        }
        
        json << "\n    }";
        if (i < tracks_.size() - 1) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    
    json << "}\n";
    
    return json.str();
}

bool ProjectFile::fromJSON(const std::string& json) {
    // Basic JSON parsing implementation
    // For production: use nlohmann::json or similar for robustness
    
    if (json.empty()) {
        return false;
    }
    
    // Simple string extraction helper
    auto extractString = [](const std::string& str, const std::string& key) -> std::string {
        size_t pos = str.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = str.find(":", pos);
        if (pos == std::string::npos) return "";
        pos = str.find("\"", pos);
        if (pos == std::string::npos) return "";
        size_t start = pos + 1;
        size_t end = str.find("\"", start);
        if (end == std::string::npos) return "";
        return str.substr(start, end - start);
    };
    
    auto extractNumber = [](const std::string& str, const std::string& key) -> float {
        size_t pos = str.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0f;
        pos = str.find(":", pos);
        if (pos == std::string::npos) return 0.0f;
        pos = str.find_first_not_of(" \t", pos + 1);
        if (pos == std::string::npos) return 0.0f;
        size_t end = pos;
        while (end < str.length() && (std::isdigit(str[end]) || str[end] == '.' || str[end] == '-' || str[end] == 'e' || str[end] == 'E' || str[end] == '+' || str[end] == '-')) {
            end++;
        }
        try {
            return std::stof(str.substr(pos, end - pos));
        } catch (...) {
            return 0.0f;
        }
    };
    
    auto extractBool = [](const std::string& str, const std::string& key) -> bool {
        size_t pos = str.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        pos = str.find(":", pos);
        if (pos == std::string::npos) return false;
        pos = str.find_first_not_of(" \t", pos + 1);
        if (pos == std::string::npos) return false;
        std::string value = str.substr(pos, 4);
        return value == "true";
    };
    
    // Parse metadata
    metadata_.name = extractString(json, "name");
    metadata_.author = extractString(json, "author");
    metadata_.createdDate = extractString(json, "created");
    metadata_.modifiedDate = extractString(json, "modified");
    
    // Parse settings
    tempo_.bpm = extractNumber(json, "tempo");
    
    // Parse time signature (format: "4/4")
    size_t tsPos = json.find("\"timeSignature\"");
    if (tsPos != std::string::npos) {
        tsPos = json.find("\"", tsPos + 15);
        if (tsPos != std::string::npos) {
            size_t tsStart = tsPos + 1;
            size_t tsEnd = json.find("\"", tsStart);
            if (tsEnd != std::string::npos) {
                std::string tsStr = json.substr(tsStart, tsEnd - tsStart);
                size_t slashPos = tsStr.find('/');
                if (slashPos != std::string::npos) {
                    try {
                        timeSignature_.numerator = static_cast<uint8_t>(std::stoi(tsStr.substr(0, slashPos)));
                        timeSignature_.denominator = static_cast<uint8_t>(std::stoi(tsStr.substr(slashPos + 1)));
                    } catch (...) {
                        // Keep defaults
                    }
                }
            }
        }
    }
    
    sampleRate_ = static_cast<SampleRate>(extractNumber(json, "sampleRate"));
    
    // Parse mixer
    mixer_.masterVolume = extractNumber(json, "masterVolume");
    mixer_.masterMuted = extractBool(json, "masterMuted");
    
    // Parse tracks (simplified - just count them for now)
    // Full implementation would parse each track's properties
    size_t tracksStart = json.find("\"tracks\"");
    if (tracksStart != std::string::npos) {
        tracksStart = json.find("[", tracksStart);
        if (tracksStart != std::string::npos) {
            // Count track objects
            size_t pos = tracksStart + 1;
            int braceDepth = 0;
            int trackCount = 0;
            bool inString = false;
            
            while (pos < json.length()) {
                char c = json[pos];
                if (c == '"' && (pos == 0 || json[pos-1] != '\\')) {
                    inString = !inString;
                } else if (!inString) {
                    if (c == '{') {
                        if (braceDepth == 0) trackCount++;
                        braceDepth++;
                    } else if (c == '}') {
                        braceDepth--;
                        if (braceDepth < 0) break;
                    } else if (c == ']' && braceDepth == 0) {
                        break;
                    }
                }
                pos++;
            }
            
            // Create placeholder tracks if needed
            while (static_cast<int>(tracks_.size()) < trackCount) {
                Track track;
                track.name = "Track " + std::to_string(tracks_.size() + 1);
                track.index = static_cast<int>(tracks_.size());
                tracks_.push_back(track);
            }
        }
    }
    
    return true;
}

} // namespace project
} // namespace daiw
