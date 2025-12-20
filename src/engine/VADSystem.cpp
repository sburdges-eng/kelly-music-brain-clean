#include "engine/VADSystem.h"
#include <algorithm>
#include <ctime>
#include <numeric>
#include <cmath>

namespace kelly {

VADSystem::VADSystem(const EmotionThesaurus* thesaurus)
    : vadCalculator_(thesaurus),
      trendAnalyzer_(20),
      oscGenerator_("/kelly"),
      thesaurus_(thesaurus) {
    
    // Set current time from system clock
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm* localTime = std::localtime(&time_t);
    currentHour_ = localTime->tm_hour;
    currentDayOfWeek_ = localTime->tm_wday;
}

VADSystem::ProcessingResult VADSystem::processEmotionId(
    int emotionId,
    float intensityModifier,
    bool generateOSC
) {
    ProcessingResult result;
    result.success = false;
    
    if (!thesaurus_) {
        return result;
    }
    
    // Validate emotion ID
    if (!isValidEmotionId(emotionId)) {
        return result;
    }
    
    // Validate intensity modifier
    intensityModifier = std::clamp(intensityModifier, 0.0f, 2.0f);
    
    // Calculate VAD from emotion
    VADState emotionVAD = vadCalculator_.calculateFromEmotionId(emotionId, intensityModifier);
    
    // Apply context adjustments
    if (contextAware_) {
        emotionVAD = vadCalculator_.applyContextAdjustments(
            emotionVAD, currentHour_, currentDayOfWeek_
        );
    }
    
    // Add to history
    emotionVADHistory_.push_back(emotionVAD);
    if (emotionVADHistory_.size() > MAX_HISTORY) {
        emotionVADHistory_.erase(emotionVADHistory_.begin());
    }
    
    // Add to trend analyzer
    trendAnalyzer_.addVADState(emotionVAD);
    
    // Convert to EmotionalState and get music parameters
    EmotionalState emotionalState = emotionVAD.toEmotionalState();
    result.musicParams = emotionMapper_.mapToParameters(emotionalState);
    
    // Generate OSC if requested
    if (generateOSC) {
        result.oscMessages = oscGenerator_.generateFromEmotion(emotionId, emotionVAD);
        auto musicOSC = oscGenerator_.generateFromMusicalParameters(result.musicParams);
        result.oscMessages.insert(
            result.oscMessages.end(),
            musicOSC.begin(),
            musicOSC.end()
        );
    }
    
    // Calculate resonance (if we have biometric history)
    if (!biometricVADHistory_.empty() && !emotionVADHistory_.empty()) {
        result.resonance = resonanceCalculator_.calculateResonance(
            emotionVADHistory_.back(),
            biometricVADHistory_.back()
        );
    }
    
    // Get trend prediction
    result.trend = trendAnalyzer_.predictNext(1);
    
    result.vad = emotionVAD;
    result.success = true;
    
    return result;
}

VADSystem::ProcessingResult VADSystem::processBiometrics(
    const BiometricInput::BiometricData& biometricData,
    bool generateOSC
) {
    ProcessingResult result;
    result.success = false;
    
    // Validate biometric data
    if (!isValidBiometricData(biometricData)) {
        return result;
    }
    
    // Calculate VAD from biometrics
    VADState biometricVAD = vadCalculator_.calculateFromBiometrics(biometricData);
    
    // Apply context adjustments
    if (contextAware_) {
        biometricVAD = vadCalculator_.applyContextAdjustments(
            biometricVAD, currentHour_, currentDayOfWeek_
        );
    }
    
    // Add to history
    biometricVADHistory_.push_back(biometricVAD);
    if (biometricVADHistory_.size() > MAX_HISTORY) {
        biometricVADHistory_.erase(biometricVADHistory_.begin());
    }
    
    biometricHistory_.push_back(biometricData);
    if (biometricHistory_.size() > MAX_HISTORY) {
        biometricHistory_.erase(biometricHistory_.begin());
    }
    
    // Add to trend analyzer
    trendAnalyzer_.addVADState(biometricVAD);
    
    // Convert to EmotionalState and get music parameters
    EmotionalState emotionalState = biometricVAD.toEmotionalState();
    result.musicParams = emotionMapper_.mapToParameters(emotionalState);
    
    // Generate OSC if requested
    if (generateOSC) {
        result.oscMessages = oscGenerator_.generateFromVAD(biometricVAD);
        auto musicOSC = oscGenerator_.generateFromMusicalParameters(result.musicParams);
        result.oscMessages.insert(
            result.oscMessages.end(),
            musicOSC.begin(),
            musicOSC.end()
        );
    }
    
    // Calculate biometric coherence
    if (biometricHistory_.size() > 1) {
        result.resonance.biometricCoherence = resonanceCalculator_.calculateBiometricCoherence(
            biometricData,
            biometricHistory_
        );
    }
    
    // Get trend prediction
    result.trend = trendAnalyzer_.predictNext(1);
    
    result.vad = biometricVAD;
    result.success = true;
    
    return result;
}

VADSystem::ProcessingResult VADSystem::processBlended(
    int emotionId,
    const BiometricInput::BiometricData& biometricData,
    float emotionWeight,
    bool generateOSC
) {
    ProcessingResult result;
    result.success = false;
    
    if (!thesaurus_) {
        return result;
    }
    
    // Validate inputs
    if (!isValidEmotionId(emotionId)) {
        return result;
    }
    
    if (!isValidBiometricData(biometricData)) {
        return result;
    }
    
    // Clamp emotion weight
    emotionWeight = std::clamp(emotionWeight, 0.0f, 1.0f);
    
    // Calculate VAD from both sources
    VADState emotionVAD = vadCalculator_.calculateFromEmotionId(emotionId);
    VADState biometricVAD = vadCalculator_.calculateFromBiometrics(biometricData);
    
    // Apply context adjustments
    if (contextAware_) {
        emotionVAD = vadCalculator_.applyContextAdjustments(
            emotionVAD, currentHour_, currentDayOfWeek_
        );
        biometricVAD = vadCalculator_.applyContextAdjustments(
            biometricVAD, currentHour_, currentDayOfWeek_
        );
    }
    
    // Blend the states
    std::vector<VADState> states = {emotionVAD, biometricVAD};
    std::vector<float> weights = {emotionWeight, 1.0f - emotionWeight};
    VADState blendedVAD = vadCalculator_.blendStates(states, weights);
    
    // Update history
    updateHistory(emotionVAD, biometricVAD, biometricData);
    
    // Add to trend analyzer
    trendAnalyzer_.addVADState(blendedVAD);
    
    // Calculate resonance
    result.resonance = resonanceCalculator_.calculateResonance(emotionVAD, biometricVAD);
    
    // Calculate temporal stability
    if (emotionVADHistory_.size() > 1) {
        result.resonance.temporalStability = resonanceCalculator_.calculateTemporalStability(
            emotionVADHistory_
        );
    }
    
    // Convert to EmotionalState and get music parameters
    EmotionalState emotionalState = blendedVAD.toEmotionalState();
    result.musicParams = emotionMapper_.mapToParameters(emotionalState);
    
    // Generate OSC if requested
    if (generateOSC) {
        result.oscMessages = oscGenerator_.generateFromVAD(blendedVAD);
        auto musicOSC = oscGenerator_.generateFromMusicalParameters(result.musicParams);
        result.oscMessages.insert(
            result.oscMessages.end(),
            musicOSC.begin(),
            musicOSC.end()
        );
    }
    
    // Get trend prediction
    result.trend = trendAnalyzer_.predictNext(1);
    
    result.vad = blendedVAD;
    result.success = true;
    
    return result;
}

TrendMetrics VADSystem::getCurrentTrends() const {
    return trendAnalyzer_.calculateTrends();
}

ResonanceMetrics VADSystem::getResonance() const {
    ResonanceMetrics metrics;
    
    if (!emotionVADHistory_.empty() && !biometricVADHistory_.empty()) {
        metrics = resonanceCalculator_.calculateResonance(
            emotionVADHistory_.back(),
            biometricVADHistory_.back()
        );
    }
    
    return metrics;
}

void VADSystem::setCurrentTime(int hourOfDay, int dayOfWeek) {
    currentHour_ = std::clamp(hourOfDay, 0, 23);
    if (dayOfWeek >= 0) {
        currentDayOfWeek_ = std::clamp(dayOfWeek, 0, 6);
    }
}

VADState VADSystem::getSmoothedVAD() const {
    return trendAnalyzer_.getSmoothedState(5);
}

void VADSystem::clearHistory() {
    emotionVADHistory_.clear();
    biometricVADHistory_.clear();
    biometricHistory_.clear();
    trendAnalyzer_.clear();
}

void VADSystem::updateHistory(
    const VADState& emotionVAD,
    const VADState& biometricVAD,
    const BiometricInput::BiometricData& biometricData
) {
    emotionVADHistory_.push_back(emotionVAD);
    if (emotionVADHistory_.size() > MAX_HISTORY) {
        emotionVADHistory_.erase(emotionVADHistory_.begin());
    }
    
    biometricVADHistory_.push_back(biometricVAD);
    if (biometricVADHistory_.size() > MAX_HISTORY) {
        biometricVADHistory_.erase(biometricVADHistory_.begin());
    }
    
    biometricHistory_.push_back(biometricData);
    if (biometricHistory_.size() > MAX_HISTORY) {
        biometricHistory_.erase(biometricHistory_.begin());
    }
}

bool VADSystem::isValidEmotionId(int emotionId) const {
    if (!thesaurus_) {
        return false;
    }
    
    auto emotion = thesaurus_->findById(emotionId);
    return emotion.has_value();
}

bool VADSystem::isValidBiometricData(const BiometricInput::BiometricData& data) const {
    // Validate ranges
    if (data.heartRate) {
        float hr = *data.heartRate;
        if (hr < 30.0f || hr > 220.0f) {
            return false;  // Unrealistic heart rate
        }
    }
    
    if (data.heartRateVariability) {
        float hrv = *data.heartRateVariability;
        if (hrv < 0.0f || hrv > 200.0f) {
            return false;  // Unrealistic HRV
        }
    }
    
    if (data.skinConductance) {
        float eda = *data.skinConductance;
        if (eda < 0.0f || eda > 100.0f) {
            return false;  // Unrealistic EDA
        }
    }
    
    if (data.temperature) {
        float temp = *data.temperature;
        if (temp < 30.0f || temp > 45.0f) {
            return false;  // Unrealistic temperature
        }
    }
    
    if (data.movement) {
        float move = *data.movement;
        if (move < 0.0f || move > 1.0f) {
            return false;  // Movement should be normalized
        }
    }
    
    // At least one biometric signal should be present
    return data.heartRate.has_value() || 
           data.heartRateVariability.has_value() ||
           data.skinConductance.has_value() ||
           data.temperature.has_value() ||
           data.movement.has_value();
}

std::optional<VADState> VADSystem::getCurrentEmotionVAD() const {
    if (emotionVADHistory_.empty()) {
        return std::nullopt;
    }
    return emotionVADHistory_.back();
}

std::optional<VADState> VADSystem::getCurrentBiometricVAD() const {
    if (biometricVADHistory_.empty()) {
        return std::nullopt;
    }
    return biometricVADHistory_.back();
}

VADSystem::ProcessingResult VADSystem::processEmotionName(
    const std::string& emotionName,
    float intensityModifier,
    bool generateOSC
) {
    ProcessingResult result;
    result.success = false;
    
    if (!thesaurus_) {
        return result;
    }
    
    auto emotionOpt = thesaurus_->findByName(emotionName);
    if (!emotionOpt) {
        return result;
    }
    
    return processEmotionId(emotionOpt->id, intensityModifier, generateOSC);
}

VADSystem::VADStatistics VADSystem::getEmotionVADStatistics() const {
    VADStatistics stats;
    stats.sampleCount = emotionVADHistory_.size();
    
    if (emotionVADHistory_.empty()) {
        return stats;
    }
    
    // Calculate mean
    VADState sum;
    for (const auto& state : emotionVADHistory_) {
        sum.valence += state.valence;
        sum.arousal += state.arousal;
        sum.dominance += state.dominance;
    }
    
    float n = static_cast<float>(emotionVADHistory_.size());
    stats.mean.valence = sum.valence / n;
    stats.mean.arousal = sum.arousal / n;
    stats.mean.dominance = sum.dominance / n;
    
    // Calculate standard deviation
    VADState variance;
    for (const auto& state : emotionVADHistory_) {
        float dv = state.valence - stats.mean.valence;
        float da = state.arousal - stats.mean.arousal;
        float dd = state.dominance - stats.mean.dominance;
        variance.valence += dv * dv;
        variance.arousal += da * da;
        variance.dominance += dd * dd;
    }
    
    stats.stdDev.valence = std::sqrt(variance.valence / n);
    stats.stdDev.arousal = std::sqrt(variance.arousal / n);
    stats.stdDev.dominance = std::sqrt(variance.dominance / n);
    
    // Find min/max
    stats.min = emotionVADHistory_[0];
    stats.max = emotionVADHistory_[0];
    
    for (const auto& state : emotionVADHistory_) {
        stats.min.valence = std::min(stats.min.valence, state.valence);
        stats.min.arousal = std::min(stats.min.arousal, state.arousal);
        stats.min.dominance = std::min(stats.min.dominance, state.dominance);
        
        stats.max.valence = std::max(stats.max.valence, state.valence);
        stats.max.arousal = std::max(stats.max.arousal, state.arousal);
        stats.max.dominance = std::max(stats.max.dominance, state.dominance);
    }
    
    return stats;
}

VADSystem::VADStatistics VADSystem::getBiometricVADStatistics() const {
    VADStatistics stats;
    stats.sampleCount = biometricVADHistory_.size();
    
    if (biometricVADHistory_.empty()) {
        return stats;
    }
    
    // Calculate mean
    VADState sum;
    for (const auto& state : biometricVADHistory_) {
        sum.valence += state.valence;
        sum.arousal += state.arousal;
        sum.dominance += state.dominance;
    }
    
    float n = static_cast<float>(biometricVADHistory_.size());
    stats.mean.valence = sum.valence / n;
    stats.mean.arousal = sum.arousal / n;
    stats.mean.dominance = sum.dominance / n;
    
    // Calculate standard deviation
    VADState variance;
    for (const auto& state : biometricVADHistory_) {
        float dv = state.valence - stats.mean.valence;
        float da = state.arousal - stats.mean.arousal;
        float dd = state.dominance - stats.mean.dominance;
        variance.valence += dv * dv;
        variance.arousal += da * da;
        variance.dominance += dd * dd;
    }
    
    stats.stdDev.valence = std::sqrt(variance.valence / n);
    stats.stdDev.arousal = std::sqrt(variance.arousal / n);
    stats.stdDev.dominance = std::sqrt(variance.dominance / n);
    
    // Find min/max
    stats.min = biometricVADHistory_[0];
    stats.max = biometricVADHistory_[0];
    
    for (const auto& state : biometricVADHistory_) {
        stats.min.valence = std::min(stats.min.valence, state.valence);
        stats.min.arousal = std::min(stats.min.arousal, state.arousal);
        stats.min.dominance = std::min(stats.min.dominance, state.dominance);
        
        stats.max.valence = std::max(stats.max.valence, state.valence);
        stats.max.arousal = std::max(stats.max.arousal, state.arousal);
        stats.max.dominance = std::max(stats.max.dominance, state.dominance);
    }
    
    return stats;
}

std::vector<size_t> VADSystem::detectAnomalies(float threshold) const {
    std::vector<size_t> anomalies;
    
    if (emotionVADHistory_.size() < 3) {
        return anomalies;
    }
    
    // Use resonance calculator's anomaly detection
    auto anomalyScores = resonanceCalculator_.detectAnomalies(emotionVADHistory_);
    
    for (size_t i = 0; i < anomalyScores.size(); ++i) {
        if (anomalyScores[i] > threshold) {
            anomalies.push_back(i);
        }
    }
    
    return anomalies;
}

} // namespace kelly
