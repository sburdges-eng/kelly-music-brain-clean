#include "penta/harmony/HarmonyEngine.h"
#include <algorithm>
#include <vector>

namespace penta::harmony {

// History tracking - Note: Ideally these should be member variables in the header
// Using static storage as workaround. For proper implementation, add to HarmonyEngine class:
//   std::vector<Chord> chordHistory_;
//   std::vector<Scale> scaleHistory_;
//   static constexpr size_t kMaxHistorySize = 100;

namespace {
    // Per-instance history storage (using thread_local for thread safety)
    thread_local std::vector<Chord> g_chordHistory;
    thread_local std::vector<Scale> g_scaleHistory;
    thread_local Chord g_lastChord{};
    thread_local Scale g_lastScale{};
    constexpr size_t kMaxHistorySize = 100;
}

HarmonyEngine::HarmonyEngine(const Config& config)
    : config_(config)
{
    chordAnalyzer_ = std::make_unique<ChordAnalyzer>();
    scaleDetector_ = std::make_unique<ScaleDetector>();
    voiceLeading_ = std::make_unique<VoiceLeading>();
    
    activeNotes_.fill(0);
    pitchClassSet_.fill(false);
    
    // Initialize history tracking
    g_chordHistory.clear();
    g_scaleHistory.clear();
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
    
    // Track chord changes in history
    // Only add to history if chord actually changed (compare root and quality)
    bool chordChanged = (newChord.root != g_lastChord.root || 
                        newChord.quality != g_lastChord.quality);
    
    if (chordChanged && g_chordHistory.size() < kMaxHistorySize) {
        g_chordHistory.push_back(newChord);
        // Keep history size manageable
        if (g_chordHistory.size() > kMaxHistorySize) {
            g_chordHistory.erase(g_chordHistory.begin());
        }
    }
    
    g_lastChord = newChord;
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
    
    // Track scale changes in history
    // Only add to history if scale actually changed (compare root and type)
    bool scaleChanged = (newScale.root != g_lastScale.root || 
                        newScale.type != g_lastScale.type);
    
    if (scaleChanged && g_scaleHistory.size() < kMaxHistorySize) {
        g_scaleHistory.push_back(newScale);
        // Keep history size manageable
        if (g_scaleHistory.size() > kMaxHistorySize) {
            g_scaleHistory.erase(g_scaleHistory.begin());
        }
    }
    
    g_lastScale = newScale;
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
    // Return history, limited to maxCount
    std::vector<Chord> result;
    
    // Include current chord if not already in history
    bool currentInHistory = false;
    if (!g_chordHistory.empty()) {
        const Chord& last = g_chordHistory.back();
        currentInHistory = (currentChord_.root == last.root && 
                           currentChord_.quality == last.quality);
    }
    
    // Start with current chord if not in history
    if (!currentInHistory) {
        result.push_back(currentChord_);
    }
    
    // Add history in reverse order (most recent first)
    size_t startIdx = g_chordHistory.size() > maxCount ? 
                      g_chordHistory.size() - maxCount : 0;
    
    for (size_t i = g_chordHistory.size(); i > startIdx; --i) {
        result.push_back(g_chordHistory[i - 1]);
        if (result.size() >= maxCount) break;
    }
    
    return result;
}

std::vector<Scale> HarmonyEngine::getScaleHistory(size_t maxCount) const {
    // Return history, limited to maxCount
    std::vector<Scale> result;
    
    // Include current scale if not already in history
    bool currentInHistory = false;
    if (!g_scaleHistory.empty()) {
        const Scale& last = g_scaleHistory.back();
        currentInHistory = (currentScale_.root == last.root && 
                           currentScale_.type == last.type);
    }
    
    // Start with current scale if not in history
    if (!currentInHistory) {
        result.push_back(currentScale_);
    }
    
    // Add history in reverse order (most recent first)
    size_t startIdx = g_scaleHistory.size() > maxCount ? 
                      g_scaleHistory.size() - maxCount : 0;
    
    for (size_t i = g_scaleHistory.size(); i > startIdx; --i) {
        result.push_back(g_scaleHistory[i - 1]);
        if (result.size() >= maxCount) break;
    }
    
    return result;
}

} // namespace penta::harmony
