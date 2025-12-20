#pragma once

/**
 * VADSystem.h
 * 
 * Integrated VAD (Valence-Arousal-Dominance) calculation system.
 * 
 * This is the main integration point that combines:
 * - VAD calculations from emotion IDs
 * - Biometric → VAD mapping
 * - Emotion-to-music parameter mapping
 * - Context-aware adjustments
 * - Resonance/coherence calculations
 * - MIDI/OSC output generation
 * - Predictive trend analysis
 * - Integration with existing Kelly engines
 */

#include "engine/VADCalculator.h"
#include "engine/ResonanceCalculator.h"
#include "engine/PredictiveTrendAnalyzer.h"
#include "engine/OSCOutputGenerator.h"
#include "engine/EmotionMapper.h"
#include "engine/EmotionThesaurus.h"
#include "biometric/BiometricInput.h"
#include "common/Types.h"
#include <vector>
#include <optional>
#include <chrono>
#include <string>

namespace kelly {

/**
 * VAD System - Main integration class
 */
class VADSystem {
public:
    VADSystem(const EmotionThesaurus* thesaurus = nullptr);
    
    /**
     * Process emotion ID and generate complete VAD + music output
     * @param emotionId Emotion ID from thesaurus
     * @param intensityModifier Intensity adjustment (0.0-2.0)
     * @param generateMIDI Whether to generate MIDI output
     * @param generateOSC Whether to generate OSC output
     * @return Complete processing result
     */
    struct ProcessingResult {
        VADState vad;                           // Calculated VAD state
        MusicalParameters musicParams;          // Music parameters from VAD
        std::vector<OSCMessage> oscMessages;    // OSC output (if requested)
        ResonanceMetrics resonance;             // Resonance metrics
        TrendPrediction trend;                  // Trend prediction
        bool success;                           // Whether processing succeeded
    };
    
    ProcessingResult processEmotionId(
        int emotionId,
        float intensityModifier = 1.0f,
        bool generateOSC = false
    );
    
    /**
     * Process biometric data and generate VAD + music output
     * @param biometricData Biometric readings
     * @param generateOSC Whether to generate OSC output
     * @return Complete processing result
     */
    ProcessingResult processBiometrics(
        const BiometricInput::BiometricData& biometricData,
        bool generateOSC = false
    );
    
    /**
     * Process blended emotion + biometric input
     * @param emotionId Emotion ID
     * @param biometricData Biometric readings
     * @param emotionWeight Weight for emotion (0.0-1.0), biometric weight = 1.0 - emotionWeight
     * @param generateOSC Whether to generate OSC output
     * @return Complete processing result
     */
    ProcessingResult processBlended(
        int emotionId,
        const BiometricInput::BiometricData& biometricData,
        float emotionWeight = 0.7f,
        bool generateOSC = false
    );
    
    /**
     * Get current trend analysis
     */
    TrendMetrics getCurrentTrends() const;
    
    /**
     * Get resonance between emotion and biometrics
     */
    ResonanceMetrics getResonance() const;
    
    /**
     * Enable/disable context-aware adjustments
     */
    void setContextAware(bool enabled) { contextAware_ = enabled; }
    bool isContextAware() const { return contextAware_; }
    
    /**
     * Set current time for context adjustments
     */
    void setCurrentTime(int hourOfDay, int dayOfWeek = -1);
    
    /**
     * Get smoothed VAD state (from trend analyzer)
     */
    VADState getSmoothedVAD() const;
    
    /**
     * Clear history
     */
    void clearHistory();
    
    /**
     * Get emotion VAD history (for analysis)
     */
    const std::vector<VADState>& getEmotionVADHistory() const { return emotionVADHistory_; }
    
    /**
     * Get biometric VAD history (for analysis)
     */
    const std::vector<VADState>& getBiometricVADHistory() const { return biometricVADHistory_; }
    
    /**
     * Get biometric data history
     */
    const std::vector<BiometricInput::BiometricData>& getBiometricHistory() const { return biometricHistory_; }
    
    /**
     * Validate emotion ID
     */
    bool isValidEmotionId(int emotionId) const;
    
    /**
     * Validate biometric data
     */
    bool isValidBiometricData(const BiometricInput::BiometricData& data) const;
    
    /**
     * Get current emotion VAD (most recent)
     */
    std::optional<VADState> getCurrentEmotionVAD() const;
    
    /**
     * Get current biometric VAD (most recent)
     */
    std::optional<VADState> getCurrentBiometricVAD() const;
    
    /**
     * Calculate VAD from emotion name (alternative to ID)
     */
    ProcessingResult processEmotionName(
        const std::string& emotionName,
        float intensityModifier = 1.0f,
        bool generateOSC = false
    );
    
    /**
     * Get statistics about VAD history
     */
    struct VADStatistics {
        VADState mean;
        VADState stdDev;
        VADState min;
        VADState max;
        size_t sampleCount;
    };
    
    VADStatistics getEmotionVADStatistics() const;
    VADStatistics getBiometricVADStatistics() const;
    
    /**
     * Detect anomalies in VAD history
     */
    std::vector<size_t> detectAnomalies(float threshold = 0.3f) const;
    
    /**
     * Get accessors for internal components
     */
    VADCalculator& vadCalculator() { return vadCalculator_; }
    const VADCalculator& vadCalculator() const { return vadCalculator_; }
    
    ResonanceCalculator& resonanceCalculator() { return resonanceCalculator_; }
    const ResonanceCalculator& resonanceCalculator() const { return resonanceCalculator_; }
    
    PredictiveTrendAnalyzer& trendAnalyzer() { return trendAnalyzer_; }
    const PredictiveTrendAnalyzer& trendAnalyzer() const { return trendAnalyzer_; }
    
private:
    VADCalculator vadCalculator_;
    ResonanceCalculator resonanceCalculator_;
    PredictiveTrendAnalyzer trendAnalyzer_;
    OSCOutputGenerator oscGenerator_;
    EmotionMapper emotionMapper_;
    const EmotionThesaurus* thesaurus_;  // Non-owning pointer - lifetime guaranteed by IntentPipeline
    
    bool contextAware_ = true;
    int currentHour_ = 12;  // Default to noon
    int currentDayOfWeek_ = 0;  // Default to Sunday
    
    // History for resonance calculation
    std::vector<VADState> emotionVADHistory_;
    std::vector<VADState> biometricVADHistory_;
    std::vector<BiometricInput::BiometricData> biometricHistory_;
    
    static constexpr size_t MAX_HISTORY = 50;
    
    void updateHistory(const VADState& emotionVAD, const VADState& biometricVAD,
                      const BiometricInput::BiometricData& biometricData);
};

} // namespace kelly
