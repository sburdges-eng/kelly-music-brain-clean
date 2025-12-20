#include "penta/harmony/HarmonyEngine.h"

namespace penta::harmony {

HarmonyEngine::HarmonyEngine(const Config& config)
    : config_(config)
{
    chordAnalyzer_ = std::make_unique<ChordAnalyzer>();
    scaleDetector_ = std::make_unique<ScaleDetector>();
    voiceLeading_ = std::make_unique<VoiceLeading>();
    
    activeNotes_.fill(0);
    pitchClassSet_.fill(false);
}

HarmonyEngine::~HarmonyEngine() = default;

void HarmonyEngine::processNotes(const Note* notes, size_t count) noexcept {
    // Update active notes and pitch class set
    for (size_t i = 0; i < count; ++i) {
        const auto& note = notes[i];
        
        if (note.velocity > 0) {
            activeNotes_[note.pitch] = note.velocity;
            pitchClassSet_[note.pitch % 12] = true;
        } else {
            activeNotes_[note.pitch] = 0;
            // Check if this was the last note of this pitch class
            bool hasNote = false;
            for (int j = note.pitch % 12; j < 128; j += 12) {
                if (activeNotes_[j] > 0) {
                    hasNote = true;
                    break;
                }
            }
            if (!hasNote) {
                pitchClassSet_[note.pitch % 12] = false;
            }
        }
    }
    
    updateChordAnalysis();
    
    if (config_.enableScaleDetection) {
        updateScaleDetection();
    }
}

void HarmonyEngine::updateChordAnalysis() noexcept {
    chordAnalyzer_->update(pitchClassSet_);
    Chord newChord = chordAnalyzer_->getCurrentChord();
    
    // Only add to history if chord changed significantly
    if (newChord.root != currentChord_.root || 
        newChord.quality != currentChord_.quality ||
        newChord.confidence > 0.7f) {
        
        // Add to history (non-RT allocation, but limited size)
        // Use deque for efficient front removal
        chordHistory_.push_back(newChord);
        if (chordHistory_.size() > 1000) {  // Limit history to prevent unbounded growth
            // Remove oldest entry efficiently with deque
            chordHistory_.erase(chordHistory_.begin());
        }
    }
    
    currentChord_ = newChord;
}

void HarmonyEngine::updateScaleDetection() noexcept {
    // Build weighted histogram from active notes
    std::array<float, 12> histogram{};
    for (size_t i = 0; i < 128; ++i) {
        if (activeNotes_[i] > 0) {
            histogram[i % 12] += activeNotes_[i] / 127.0f;
        }
    }
    
    scaleDetector_->update(histogram);
    Scale newScale = scaleDetector_->getCurrentScale();
    
    // Only add to history if scale changed significantly
    if (newScale.tonic != currentScale_.tonic || 
        newScale.mode != currentScale_.mode ||
        newScale.confidence > 0.7f) {
        
        // Add to history (non-RT allocation, but limited size)
        // Use deque for efficient front removal
        scaleHistory_.push_back(newScale);
        if (scaleHistory_.size() > 1000) {  // Limit history to prevent unbounded growth
            // Remove oldest entry efficiently with deque
            scaleHistory_.erase(scaleHistory_.begin());
        }
    }
    
    currentScale_ = newScale;
}

std::vector<Note> HarmonyEngine::suggestVoiceLeading(
    const Chord& targetChord,
    const std::vector<Note>& currentVoices
) const noexcept {
    if (!config_.enableVoiceLeading) {
        return {};
    }
    
    return voiceLeading_->findOptimalVoicing(targetChord, currentVoices);
}

void HarmonyEngine::updateConfig(const Config& config) {
    config_ = config;
    
    if (chordAnalyzer_) {
        chordAnalyzer_->setConfidenceThreshold(config.confidenceThreshold);
    }
    
    if (scaleDetector_) {
        scaleDetector_->setConfidenceThreshold(config.confidenceThreshold);
    }
}

std::vector<Chord> HarmonyEngine::getChordHistory(size_t maxCount) const {
    // Return most recent chords up to maxCount
    if (chordHistory_.empty()) {
        return {currentChord_};
    }
    
    size_t count = std::min(maxCount, chordHistory_.size());
    size_t startIdx = chordHistory_.size() - count;
    
    return std::vector<Chord>(
        chordHistory_.begin() + startIdx,
        chordHistory_.end()
    );
}

std::vector<Scale> HarmonyEngine::getScaleHistory(size_t maxCount) const {
    // Return most recent scales up to maxCount
    if (scaleHistory_.empty()) {
        return {currentScale_};
    }
    
    size_t count = std::min(maxCount, scaleHistory_.size());
    size_t startIdx = scaleHistory_.size() - count;
    
    return std::vector<Scale>(
        scaleHistory_.begin() + startIdx,
        scaleHistory_.end()
    );
}

} // namespace penta::harmony
