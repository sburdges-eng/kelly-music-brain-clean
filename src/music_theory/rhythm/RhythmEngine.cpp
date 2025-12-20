/*
 * RhythmEngine.cpp - Time, Meter, Groove, and Rhythm Analysis Implementation
 * ===========================================================================
 */

#include "RhythmEngine.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <random>

namespace midikompanion::theory {

//==============================================================================
// Constructor
//==============================================================================

RhythmEngine::RhythmEngine(std::shared_ptr<CoreTheoryEngine> coreTheory)
    : coreTheory_(coreTheory)
{
    initializePatternDatabase();
    initializeEmotionRhythmMappings();
    initializeClavePatterns();
}

//==============================================================================
// Time Signature and Meter
//==============================================================================

TimeSignature RhythmEngine::analyzeTimeSignature(
    const std::vector<float>& onsetTimes,
    float duration) const
{
    TimeSignature ts;

    if (onsetTimes.empty()) {
        ts.numerator = 4;
        ts.denominator = 4;
        ts.feel = "Duple";
        ts.bodyResponse = "Walking rhythm";
        return ts;
    }

    // Calculate average inter-onset interval
    auto ioi = onsetsToIOI(onsetTimes);
    float avgIOI = calculateMean(ioi);

    // Estimate beat duration
    float beatDuration = avgIOI;

    // Count beats in duration
    int estimatedBeats = static_cast<int>(duration / beatDuration);

    // Common time signatures
    if (estimatedBeats % 4 == 0) {
        ts.numerator = 4;
        ts.denominator = 4;
    } else if (estimatedBeats % 3 == 0) {
        ts.numerator = 3;
        ts.denominator = 4;
    } else if (estimatedBeats % 6 == 0) {
        ts.numerator = 6;
        ts.denominator = 8;
    } else if (estimatedBeats % 5 == 0) {
        ts.numerator = 5;
        ts.denominator = 4;
    } else if (estimatedBeats % 7 == 0) {
        ts.numerator = 7;
        ts.denominator = 8;
    } else {
        ts.numerator = 4;
        ts.denominator = 4;
    }

    ts.feel = detectMeterFeel(ts.numerator, ts.denominator);
    ts.bodyResponse = getBodyResponse(ts);

    return ts;
}

std::string RhythmEngine::detectMeterFeel(int numerator, int denominator) const {
    // Compound meters (6/8, 9/8, 12/8)
    if ((numerator == 6 || numerator == 9 || numerator == 12) && denominator == 8) {
        return "Compound";
    }

    // Triple meters (3/4, 3/8)
    if (numerator == 3) {
        return "Triple";
    }

    // Duple meters (2/4, 4/4)
    if (numerator == 2 || numerator == 4) {
        return "Duple";
    }

    // Asymmetric meters (5/4, 7/8)
    if (numerator == 5 || numerator == 7) {
        return "Asymmetric";
    }

    return "Irregular";
}

std::string RhythmEngine::getBodyResponse(const TimeSignature& ts) const {
    if (ts.numerator == 4 && ts.denominator == 4) {
        return "Walking rhythm (left-right-left-right)";
    } else if (ts.numerator == 3 && ts.denominator == 4) {
        return "Waltz (strong-weak-weak)";
    } else if (ts.numerator == 6 && ts.denominator == 8) {
        return "Lilting (ONE-two-three-FOUR-five-six)";
    } else if (ts.numerator == 2 && ts.denominator == 4) {
        return "March (LEFT-right)";
    } else if (ts.numerator == 5 && ts.denominator == 4) {
        return "Asymmetric (3+2 or 2+3 grouping)";
    } else if (ts.numerator == 7 && ts.denominator == 8) {
        return "Complex asymmetric (3+2+2 or 2+2+3)";
    }

    return "Irregular pulse";
}

std::string RhythmEngine::explainTimeSignature(
    const TimeSignature& ts,
    ExplanationDepth depth) const
{
    std::ostringstream explanation;

    switch (depth) {
        case ExplanationDepth::Simple:
            explanation << ts.numerator << "/" << ts.denominator << " time signature. "
                       << "Feel: " << ts.feel;
            break;

        case ExplanationDepth::Intermediate:
            explanation << ts.numerator << "/" << ts.denominator << " has "
                       << ts.numerator << " beats per bar.\n"
                       << "Feel: " << ts.feel << "\n"
                       << "Body response: " << ts.bodyResponse;
            break;

        case ExplanationDepth::Advanced:
            explanation << "Time signature: " << ts.numerator << "/" << ts.denominator << "\n"
                       << "Metric feel: " << ts.feel << "\n"
                       << "Each beat subdivides into " << ts.denominator << " note values.\n"
                       << "Physical embodiment: " << ts.bodyResponse << "\n"
                       << "This creates a " << (ts.numerator % 2 == 0 ? "symmetrical" : "asymmetrical")
                       << " rhythmic structure.";
            break;

        case ExplanationDepth::Expert:
            explanation << "Metric analysis: " << ts.numerator << "/" << ts.denominator << "\n"
                       << "Beat hierarchy: ";
            if (ts.numerator == 4) {
                explanation << "Strong-weak-medium-weak";
            } else if (ts.numerator == 3) {
                explanation << "Strong-weak-weak";
            } else if (ts.numerator == 6) {
                explanation << "Strong-weak-weak-medium-weak-weak (compound duple)";
            }
            explanation << "\nPsychoacoustic grouping tendencies favor "
                       << (ts.numerator % 2 == 0 ? "binary" : "ternary") << " subdivisions.";
            break;
    }

    return explanation.str();
}

std::vector<RhythmEngine::TimeSignatureSuggestion>
RhythmEngine::suggestTimeSignatureForEmotion(const std::string& emotion) const
{
    std::vector<TimeSignatureSuggestion> suggestions;

    auto it = emotionRhythmMappings_.find(emotion);
    if (it != emotionRhythmMappings_.end()) {
        for (const auto& meter : it->second.preferredMeters) {
            TimeSignatureSuggestion suggestion;
            suggestion.timeSignature = meter;
            suggestion.emotionalEffect = "Evokes " + emotion;
            suggestion.explanation = "This meter creates the rhythmic feel associated with " + emotion;
            suggestions.push_back(suggestion);
        }
    } else {
        // Default suggestions
        TimeSignatureSuggestion defaultSugg;
        defaultSugg.timeSignature = {4, 4, "Duple", "Walking rhythm"};
        defaultSugg.emotionalEffect = "Neutral, stable";
        defaultSugg.explanation = "Common 4/4 time provides stable foundation";
        suggestions.push_back(defaultSugg);
    }

    return suggestions;
}

//==============================================================================
// Groove Analysis
//==============================================================================

GrooveAnalysis RhythmEngine::analyzeGroove(
    const std::vector<float>& onsetTimes,
    const std::vector<int>& velocities,
    const TimeSignature& timeSignature) const
{
    GrooveAnalysis analysis;

    if (onsetTimes.empty()) return analysis;

    // Store actual onsets
    analysis.actualOnsets = onsetTimes;

    // Quantize to get reference grid
    analysis.quantizedOnsets = quantizeToGrid(onsetTimes, 16);

    // Calculate micro-timing shifts
    analysis.microTimingShifts = analyzeMicroTiming(onsetTimes, analysis.quantizedOnsets);

    // Describe pocket
    analysis.pocketDescription = describePocket(analysis.microTimingShifts);

    // Calculate pocket width
    analysis.pocketWidth = calculatePocketWidth(analysis.microTimingShifts);

    // Detect groove quality
    float swingRatio = calculateSwingRatio(onsetTimes, 8);
    float syncopation = detectSyncopation(onsetTimes, timeSignature);

    if (swingRatio > 1.4f) {
        analysis.grooveQuality = "Swung";
    } else if (syncopation > 0.6f) {
        analysis.grooveQuality = "Syncopated";
    } else if (analysis.pocketWidth < 10.0f) {
        analysis.grooveQuality = "Tight";
    } else {
        analysis.grooveQuality = "Straight";
    }

    return analysis;
}

std::vector<float> RhythmEngine::quantizeWithGroove(
    const std::vector<float>& onsetTimes,
    float quantizeStrength,
    bool preserveSwing) const
{
    std::vector<float> quantized;

    // Calculate swing ratio before quantizing
    float swingRatio = 1.0f;
    if (preserveSwing) {
        swingRatio = calculateSwingRatio(onsetTimes, 8);
    }

    // Quantize to grid
    auto gridSnapped = quantizeToGrid(onsetTimes, 16);

    // Blend between original and quantized
    for (size_t i = 0; i < onsetTimes.size(); ++i) {
        float original = onsetTimes[i];
        float snapped = gridSnapped[i];
        float blended = original + quantizeStrength * (snapped - original);
        quantized.push_back(blended);
    }

    // Reapply swing if requested
    if (preserveSwing && swingRatio > 1.1f) {
        RhythmicPattern pattern;
        pattern.onsetTimes = quantized;
        auto swung = applySwing(pattern, swingRatio);
        quantized = swung.onsetTimes;
    }

    return quantized;
}

float RhythmEngine::calculateSwingRatio(
    const std::vector<float>& onsetTimes,
    int subdivision) const
{
    if (onsetTimes.size() < 3) return 1.0f;

    // Find pairs of notes on swing subdivisions
    auto swingPairs = findSwingPairs(onsetTimes, subdivision);

    if (swingPairs.empty()) return 1.0f;

    // Calculate ratio of first to second note in each pair
    std::vector<float> ratios;
    for (size_t i = 0; i < swingPairs.size() - 1; i += 2) {
        float first = swingPairs[i + 1] - swingPairs[i];
        if (i + 2 < swingPairs.size()) {
            float second = swingPairs[i + 2] - swingPairs[i + 1];
            if (second > 0.001f) {
                ratios.push_back(first / second);
            }
        }
    }

    if (ratios.empty()) return 1.0f;

    // Average swing ratio
    float avgRatio = calculateMean(ratios);

    return std::clamp(avgRatio, 1.0f, 2.5f);
}

std::string RhythmEngine::detectGrooveType(const GrooveAnalysis& analysis) const {
    float avgShift = calculateAverageShift(analysis.microTimingShifts);
    float stdDev = calculateStdDev(analysis.microTimingShifts);

    if (analysis.grooveQuality == "Swung") {
        return "Swing 8ths";
    } else if (analysis.grooveQuality == "Syncopated") {
        return "Syncopated backbeat";
    } else if (stdDev < 5.0f) {
        return "Straight 16ths";
    } else if (avgShift < -10.0f) {
        return "Laid back (behind beat)";
    } else if (avgShift > 10.0f) {
        return "Pushing (ahead of beat)";
    }

    return "Straight feel";
}

std::vector<float> RhythmEngine::analyzeMicroTiming(
    const std::vector<float>& onsetTimes,
    const std::vector<float>& quantizedTimes) const
{
    std::vector<float> shifts;

    size_t minSize = std::min(onsetTimes.size(), quantizedTimes.size());

    for (size_t i = 0; i < minSize; ++i) {
        // Difference in milliseconds (assuming times are in seconds)
        float shift = (onsetTimes[i] - quantizedTimes[i]) * 1000.0f;
        shifts.push_back(shift);
    }

    return shifts;
}

std::string RhythmEngine::describePocket(
    const std::vector<float>& microTimingShifts) const
{
    if (microTimingShifts.empty()) {
        return "No timing data";
    }

    float avgShift = calculateAverageShift(microTimingShifts);
    float stdDev = calculateStdDev(microTimingShifts);

    return classifyPocketStyle(avgShift, stdDev);
}

float RhythmEngine::calculatePocketWidth(
    const std::vector<float>& microTimingShifts) const
{
    if (microTimingShifts.empty()) return 0.0f;

    // Pocket width = standard deviation of timing shifts
    return calculateStdDev(microTimingShifts);
}

//==============================================================================
// Polyrhythm and Polymeter
//==============================================================================

RhythmEngine::PolyrhythmAnalysis RhythmEngine::detectPolyrhythm(
    const std::vector<std::vector<float>>& layers) const
{
    PolyrhythmAnalysis analysis;

    if (layers.size() < 2) return analysis;

    // Calculate densities of each layer
    std::vector<int> noteCounts;
    for (const auto& layer : layers) {
        noteCounts.push_back(static_cast<int>(layer.size()));
    }

    // Find ratio between first two layers
    if (noteCounts.size() >= 2) {
        int gcd = calculateGCD(noteCounts[0], noteCounts[1]);
        int ratio1 = noteCounts[0] / gcd;
        int ratio2 = noteCounts[1] / gcd;

        analysis.ratios = {ratio1, ratio2};
        analysis.description = std::to_string(ratio1) + " against " + std::to_string(ratio2);

        // Calculate tension based on ratio complexity
        float complexity = static_cast<float>(ratio1 * ratio2) /
                          static_cast<float>(ratio1 + ratio2);
        analysis.tension = std::clamp(complexity / 10.0f, 0.0f, 1.0f);

        // Describe perceptual effect
        if (ratio1 == 3 && ratio2 == 2) {
            analysis.perceptualEffect = "Creates rolling, forward momentum (hemiola)";
            analysis.examples.push_back("America - Leonard Bernstein");
        } else if (ratio1 == 4 && ratio2 == 3) {
            analysis.perceptualEffect = "Complex polymetric texture";
        } else {
            analysis.perceptualEffect = "Rhythmic independence between voices";
        }

        // Find meeting points
        if (layers.size() >= 2) {
            float duration = std::max(
                layers[0].empty() ? 0.0f : layers[0].back(),
                layers[1].empty() ? 0.0f : layers[1].back()
            );
            analysis.meetingPoints = findMeetingPoints(analysis.ratios, duration);
        }
    }

    return analysis;
}

RhythmEngine::PolymeterAnalysis RhythmEngine::detectPolymeter(
    const std::vector<std::vector<float>>& layers,
    float duration) const
{
    PolymeterAnalysis analysis;

    if (layers.size() < 2) return analysis;

    // Analyze time signature of each layer
    for (const auto& layer : layers) {
        TimeSignature ts = analyzeTimeSignature(layer, duration);
        analysis.meters.push_back(ts);
    }

    if (analysis.meters.size() >= 2) {
        // Calculate when meters align
        int lcm = calculateLCM(analysis.meters[0].numerator,
                              analysis.meters[1].numerator);
        analysis.barsUntilAlignment = lcm / analysis.meters[0].numerator;

        analysis.perceptualEffect = "Creates rhythmic tension and interest";
        analysis.explanation = "The meters align every " +
                              std::to_string(analysis.barsUntilAlignment) + " bars";
    }

    return analysis;
}

float RhythmEngine::calculateRhythmicTension(
    const std::vector<std::vector<float>>& layers) const
{
    if (layers.size() < 2) return 0.0f;

    // Check for polyrhythm
    auto polyAnalysis = detectPolyrhythm(layers);

    return polyAnalysis.tension;
}

std::vector<float> RhythmEngine::findMeetingPoints(
    const std::vector<int>& ratios,
    float duration) const
{
    std::vector<float> meetingPoints;

    if (ratios.size() < 2) return meetingPoints;

    int lcm = calculateLCM(ratios[0], ratios[1]);
    float meetingInterval = duration / static_cast<float>(lcm);

    for (int i = 0; i <= lcm; ++i) {
        meetingPoints.push_back(i * meetingInterval);
    }

    return meetingPoints;
}

//==============================================================================
// Rhythmic Pattern Generation
//==============================================================================

RhythmicPattern RhythmEngine::generatePatternForEmotion(
    const std::string& emotion,
    const TimeSignature& timeSignature,
    float density,
    int numBars) const
{
    RhythmicPattern pattern;
    pattern.grooveName = emotion + " groove";

    auto it = emotionRhythmMappings_.find(emotion);

    float targetDensity = density;
    float syncopationLevel = 0.5f;

    if (it != emotionRhythmMappings_.end()) {
        targetDensity = it->second.targetDensity * density;
        syncopationLevel = it->second.syncopationLevel;

        // Use suggested pattern if available
        if (!it->second.suggestedPatterns.empty()) {
            std::string patternName = it->second.suggestedPatterns[0];
            return getCommonPattern(patternName, timeSignature);
        }
    }

    // Generate pattern based on density
    float beatDuration = 1.0f; // 1 beat
    float totalDuration = numBars * timeSignature.numerator * beatDuration;

    int numNotes = static_cast<int>(totalDuration * targetDensity * timeSignature.numerator);

    // Generate evenly spaced notes
    for (int i = 0; i < numNotes; ++i) {
        float time = (i * totalDuration) / numNotes;
        pattern.onsetTimes.push_back(time);
        pattern.durations.push_back(beatDuration / 4.0f); // Quarter note durations
    }

    // Add syncopation if requested
    if (syncopationLevel > 0.3f) {
        pattern = addSyncopation(pattern, syncopationLevel);
    }

    pattern.perceptualGroove = "Generated for emotion: " + emotion;

    return pattern;
}

RhythmicPattern RhythmEngine::generatePatternFromDescription(
    const std::string& description,
    const TimeSignature& timeSignature,
    int numBars) const
{
    // Check if it's in our database
    for (const auto& tmpl : patternDatabase_) {
        if (tmpl.name == description || tmpl.feel == description) {
            RhythmicPattern pattern;
            pattern.grooveName = tmpl.name;
            pattern.perceptualGroove = tmpl.feel;

            // Scale pattern to numBars
            for (int bar = 0; bar < numBars; ++bar) {
                for (float onset : tmpl.onsetPattern) {
                    pattern.onsetTimes.push_back(onset + bar * timeSignature.numerator);
                }
            }

            // Generate durations
            for (size_t i = 0; i < pattern.onsetTimes.size(); ++i) {
                float duration = 0.25f; // Default quarter note
                if (i < pattern.onsetTimes.size() - 1) {
                    duration = std::min(0.25f,
                                       pattern.onsetTimes[i+1] - pattern.onsetTimes[i]);
                }
                pattern.durations.push_back(duration);
            }

            return pattern;
        }
    }

    // Default pattern
    return generatePatternForEmotion("neutral", timeSignature, 0.5f, numBars);
}

RhythmicPattern RhythmEngine::addSyncopation(
    const RhythmicPattern& pattern,
    float syncopationAmount) const
{
    RhythmicPattern syncopated = pattern;

    if (syncopationAmount < 0.1f) return syncopated;

    // Shift some notes off the beat
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < syncopated.onsetTimes.size(); ++i) {
        if (dis(gen) < syncopationAmount) {
            // Shift note slightly (8th or 16th note offset)
            float offset = 0.125f; // 8th note
            syncopated.onsetTimes[i] += offset;
        }
    }

    return syncopated;
}

RhythmicPattern RhythmEngine::applySwing(
    const RhythmicPattern& pattern,
    float swingRatio) const
{
    RhythmicPattern swung = pattern;

    if (swingRatio <= 1.0f) return swung;

    // Apply swing to alternating notes
    for (size_t i = 1; i < swung.onsetTimes.size(); i += 2) {
        float prevTime = swung.onsetTimes[i - 1];
        float interval = swung.onsetTimes[i] - prevTime;

        // Adjust timing based on swing ratio
        float firstPart = interval * (swingRatio / (swingRatio + 1.0f));
        swung.onsetTimes[i] = prevTime + firstPart;
    }

    swung.swingRatio = swingRatio;
    swung.grooveName = "Swung " + pattern.grooveName;

    return swung;
}

RhythmicPattern RhythmEngine::humanizePattern(
    const RhythmicPattern& pattern,
    float humanization,
    const std::string& pocketStyle) const
{
    RhythmicPattern humanized = pattern;

    if (humanization < 0.01f) return humanized;

    std::random_device rd;
    std::mt19937 gen(rd());

    // Add random timing variations
    std::uniform_real_distribution<> timingDis(-humanization * 20.0f, humanization * 20.0f);

    // Pocket bias
    float pocketBias = 0.0f;
    if (pocketStyle == "Behind") {
        pocketBias = -humanization * 10.0f;
    } else if (pocketStyle == "Ahead") {
        pocketBias = humanization * 10.0f;
    }

    for (size_t i = 0; i < humanized.onsetTimes.size(); ++i) {
        float variation = timingDis(gen) + pocketBias;
        humanized.onsetTimes[i] += variation / 1000.0f; // Convert ms to seconds
    }

    // Add velocity variations (if we had velocity data)
    // This would randomize velocities slightly

    return humanized;
}

//==============================================================================
// Rhythmic Complexity Analysis
//==============================================================================

float RhythmEngine::calculateComplexity(const RhythmicPattern& pattern) const {
    if (pattern.onsetTimes.empty()) return 0.0f;

    float complexity = 0.0f;

    // Factor 1: Number of unique inter-onset intervals
    auto ioi = onsetsToIOI(pattern.onsetTimes);
    std::set<float> uniqueIOI(ioi.begin(), ioi.end());
    float ioiComplexity = std::min(1.0f, uniqueIOI.size() / 10.0f);
    complexity += ioiComplexity * 0.4f;

    // Factor 2: Syncopation level
    TimeSignature ts{4, 4, "Duple", "Walking"};
    float syncopation = detectSyncopation(pattern.onsetTimes, ts);
    complexity += syncopation * 0.3f;

    // Factor 3: Micro-timing variation
    auto quantized = quantizeToGrid(pattern.onsetTimes, 16);
    auto microTiming = analyzeMicroTiming(pattern.onsetTimes, quantized);
    float microVariation = calculateStdDev(microTiming) / 50.0f; // Normalize
    complexity += std::min(1.0f, microVariation) * 0.3f;

    return std::clamp(complexity, 0.0f, 1.0f);
}

float RhythmEngine::detectSyncopation(
    const std::vector<float>& onsetTimes,
    const TimeSignature& timeSignature) const
{
    if (onsetTimes.empty()) return 0.0f;

    int syncopatedNotes = 0;

    for (float time : onsetTimes) {
        if (!isOnStrongBeat(time, timeSignature) && !isOnWeakBeat(time, timeSignature)) {
            syncopatedNotes++;
        }
    }

    return static_cast<float>(syncopatedNotes) / onsetTimes.size();
}

float RhythmEngine::calculateDensity(
    const std::vector<float>& onsetTimes,
    float duration) const
{
    if (duration <= 0.0f) return 0.0f;

    return onsetTimes.size() / duration; // Notes per beat
}

std::vector<RhythmEngine::RhythmicMotif> RhythmEngine::detectMotifs(
    const std::vector<float>& onsetTimes,
    float minOccurrences) const
{
    std::vector<RhythmicMotif> motifs;

    if (onsetTimes.size() < 4) return motifs;

    // Convert to IOI pattern
    auto ioi = onsetsToIOI(onsetTimes);

    // Look for repeating subsequences
    for (size_t patternLength = 2; patternLength <= ioi.size() / 2; ++patternLength) {
        for (size_t start = 0; start <= ioi.size() - patternLength; ++start) {
            std::vector<float> pattern(ioi.begin() + start,
                                      ioi.begin() + start + patternLength);

            // Count occurrences
            int occurrences = 0;
            std::vector<float> positions;

            for (size_t pos = 0; pos <= ioi.size() - patternLength; ++pos) {
                std::vector<float> candidate(ioi.begin() + pos,
                                            ioi.begin() + pos + patternLength);

                float similarity = calculatePatternSimilarity(pattern, candidate);
                if (similarity > 0.9f) {
                    occurrences++;
                    positions.push_back(onsetTimes[pos]);
                }
            }

            if (occurrences >= minOccurrences) {
                RhythmicMotif motif;
                motif.pattern = pattern;
                motif.occurrences = occurrences;
                motif.positions = positions;
                motif.name = "Motif-" + std::to_string(motifs.size() + 1);
                motifs.push_back(motif);

                // Skip ahead to avoid detecting sub-patterns
                start += patternLength;
            }
        }
    }

    return motifs;
}

std::optional<std::string> RhythmEngine::detectClavePattern(
    const std::vector<float>& onsetTimes) const
{
    if (onsetTimes.size() < 5) return std::nullopt;

    // Normalize to 2-bar cycle
    auto normalized = onsetTimes;
    if (!normalized.empty()) {
        float first = normalized[0];
        for (auto& time : normalized) {
            time -= first;
        }
    }

    // Compare with known clave patterns
    for (const auto& [name, pattern] : clavePatterns_) {
        if (normalized.size() != pattern.size()) continue;

        float similarity = calculatePatternSimilarity(normalized, pattern);
        if (similarity > 0.85f) {
            return name;
        }
    }

    return std::nullopt;
}

//==============================================================================
// Rhythmic Perception
//==============================================================================

std::vector<float> RhythmEngine::analyzeBeatStrength(
    const std::vector<float>& onsetTimes,
    const std::vector<int>& velocities,
    const TimeSignature& timeSignature) const
{
    std::vector<float> beatStrengths(timeSignature.numerator, 0.0f);

    if (onsetTimes.size() != velocities.size()) {
        return beatStrengths;
    }

    // Accumulate velocities for each beat position
    for (size_t i = 0; i < onsetTimes.size(); ++i) {
        int beatPosition = static_cast<int>(onsetTimes[i]) % timeSignature.numerator;
        beatStrengths[beatPosition] += velocities[i] / 127.0f;
    }

    // Normalize
    float maxStrength = *std::max_element(beatStrengths.begin(), beatStrengths.end());
    if (maxStrength > 0.0f) {
        for (auto& strength : beatStrengths) {
            strength /= maxStrength;
        }
    }

    return beatStrengths;
}

float RhythmEngine::calculateStability(const std::vector<float>& onsetTimes) const {
    if (onsetTimes.size() < 3) return 1.0f;

    auto ioi = onsetsToIOI(onsetTimes);
    float variance = calculateVariance(ioi);

    // Low variance = high stability
    float stability = 1.0f / (1.0f + variance * 10.0f);

    return std::clamp(stability, 0.0f, 1.0f);
}

std::vector<RhythmEngine::TempoChange> RhythmEngine::detectTempoChanges(
    const std::vector<float>& onsetTimes) const
{
    std::vector<TempoChange> changes;

    if (onsetTimes.size() < 10) return changes;

    // Calculate local tempos in windows
    const size_t windowSize = 5;

    std::vector<float> localTempos;
    for (size_t i = 0; i < onsetTimes.size() - windowSize; ++i) {
        std::vector<float> window(onsetTimes.begin() + i,
                                  onsetTimes.begin() + i + windowSize);
        float tempo = calculateTempo(window);
        localTempos.push_back(tempo);
    }

    // Detect significant tempo changes
    for (size_t i = 1; i < localTempos.size(); ++i) {
        float tempoChange = localTempos[i] - localTempos[i - 1];
        float percentChange = std::abs(tempoChange) / localTempos[i - 1];

        if (percentChange > 0.1f) { // 10% change
            TempoChange change;
            change.startTime = onsetTimes[i];
            change.endTime = onsetTimes[std::min(i + windowSize, onsetTimes.size() - 1)];
            change.startTempo = localTempos[i - 1];
            change.endTempo = localTempos[i];

            if (tempoChange > 0) {
                change.type = "Accelerando";
                change.musicalEffect = "Building energy and excitement";
            } else {
                change.type = "Ritardando";
                change.musicalEffect = "Slowing down, creating resolution";
            }

            changes.push_back(change);
        }
    }

    return changes;
}

std::string RhythmEngine::explainRhythmicFeel(
    const RhythmicPattern& pattern,
    ExplanationDepth depth) const
{
    std::ostringstream explanation;

    float density = calculateDensity(pattern.onsetTimes,
                                    pattern.onsetTimes.empty() ? 1.0f : pattern.onsetTimes.back());
    float complexity = calculateComplexity(pattern);

    switch (depth) {
        case ExplanationDepth::Simple:
            explanation << "A " << pattern.grooveName << " pattern";
            break;

        case ExplanationDepth::Intermediate:
            explanation << "Groove: " << pattern.grooveName << "\n"
                       << "Density: " << density << " notes per beat\n"
                       << "Feel: " << pattern.perceptualGroove;
            break;

        case ExplanationDepth::Advanced:
            explanation << "Rhythmic analysis:\n"
                       << "Groove type: " << pattern.grooveName << "\n"
                       << "Note density: " << density << " per beat\n"
                       << "Complexity: " << (complexity * 100) << "%\n"
                       << "Swing ratio: " << pattern.swingRatio << "\n"
                       << "Perceptual effect: " << pattern.perceptualGroove;
            break;

        case ExplanationDepth::Expert:
            explanation << "Complete rhythmic analysis:\n"
                       << "IOI pattern: ";
            auto ioi = onsetsToIOI(pattern.onsetTimes);
            for (size_t i = 0; i < std::min(ioi.size(), size_t(5)); ++i) {
                explanation << ioi[i] << " ";
            }
            explanation << "\nSwing: " << pattern.swingRatio << ":1 ratio\n"
                       << "Complexity score: " << complexity << "\n"
                       << "Micro-timing characteristics create: " << pattern.perceptualGroove;
            break;
    }

    return explanation.str();
}

//==============================================================================
// Common Patterns Database
//==============================================================================

RhythmicPattern RhythmEngine::getCommonPattern(
    const std::string& name,
    const TimeSignature& timeSignature) const
{
    for (const auto& tmpl : patternDatabase_) {
        if (tmpl.name == name) {
            RhythmicPattern pattern;
            pattern.grooveName = tmpl.name;
            pattern.onsetTimes = tmpl.onsetPattern;
            pattern.perceptualGroove = tmpl.feel;

            // Generate durations
            for (size_t i = 0; i < pattern.onsetTimes.size(); ++i) {
                float duration = 0.25f;
                if (i < pattern.onsetTimes.size() - 1) {
                    duration = pattern.onsetTimes[i + 1] - pattern.onsetTimes[i];
                }
                pattern.durations.push_back(duration);
            }

            return pattern;
        }
    }

    // Return default pattern
    RhythmicPattern defaultPattern;
    defaultPattern.grooveName = "Quarter notes";
    defaultPattern.onsetTimes = {0.0f, 1.0f, 2.0f, 3.0f};
    defaultPattern.durations = {1.0f, 1.0f, 1.0f, 1.0f};
    defaultPattern.perceptualGroove = "Steady beat";

    return defaultPattern;
}

std::vector<std::string> RhythmEngine::listCommonPatterns() const {
    std::vector<std::string> names;

    for (const auto& tmpl : patternDatabase_) {
        names.push_back(tmpl.name);
    }

    return names;
}

std::vector<RhythmicPattern> RhythmEngine::findSimilarPatterns(
    const RhythmicPattern& pattern,
    float similarityThreshold) const
{
    std::vector<RhythmicPattern> similar;

    auto patternIOI = onsetsToIOI(pattern.onsetTimes);

    for (const auto& tmpl : patternDatabase_) {
        auto tmplIOI = onsetsToIOI(tmpl.onsetPattern);

        float similarity = calculatePatternSimilarity(patternIOI, tmplIOI);

        if (similarity >= similarityThreshold) {
            RhythmicPattern match;
            match.grooveName = tmpl.name;
            match.onsetTimes = tmpl.onsetPattern;
            match.perceptualGroove = tmpl.feel;
            similar.push_back(match);
        }
    }

    return similar;
}

//==============================================================================
// Utilities
//==============================================================================

std::vector<float> RhythmEngine::onsetsToIOI(const std::vector<float>& onsetTimes) const {
    std::vector<float> ioi;

    for (size_t i = 1; i < onsetTimes.size(); ++i) {
        ioi.push_back(onsetTimes[i] - onsetTimes[i - 1]);
    }

    return ioi;
}

std::vector<float> RhythmEngine::ioiToOnsets(const std::vector<float>& intervals) const {
    std::vector<float> onsets;
    float currentTime = 0.0f;

    onsets.push_back(currentTime);

    for (float interval : intervals) {
        currentTime += interval;
        onsets.push_back(currentTime);
    }

    return onsets;
}

std::vector<float> RhythmEngine::quantizeToGrid(
    const std::vector<float>& onsetTimes,
    int subdivision) const
{
    std::vector<float> quantized;

    for (float time : onsetTimes) {
        float snapped = snapToGrid(time, subdivision);
        quantized.push_back(snapped);
    }

    return quantized;
}

float RhythmEngine::calculateTempo(const std::vector<float>& onsetTimes) const {
    if (onsetTimes.size() < 2) return 120.0f;

    auto ioi = onsetsToIOI(onsetTimes);
    float avgIOI = calculateMean(ioi);

    if (avgIOI <= 0.0f) return 120.0f;

    // Convert to BPM (assuming IOI in beats)
    float bpm = 60.0f / avgIOI;

    return std::clamp(bpm, 40.0f, 240.0f);
}

std::vector<std::pair<float, TimeSignature>> RhythmEngine::detectTimeSignatureChanges(
    const std::vector<float>& onsetTimes,
    float duration) const
{
    std::vector<std::pair<float, TimeSignature>> changes;

    // Simplified: detect meter changes in windows
    // This would require more sophisticated analysis

    return changes;
}

//==============================================================================
// Internal Helpers
//==============================================================================

void RhythmEngine::initializePatternDatabase() {
    patternDatabase_ = {
        {"four-on-the-floor",
         {0.0f, 1.0f, 2.0f, 3.0f},
         "Electronic",
         "Driving, steady pulse",
         {"Donna Summer - I Feel Love"}},

        {"backbeat",
         {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f},
         "Rock",
         "Strong 2 and 4 accents",
         {"Most rock songs"}},

        {"swing 8ths",
         {0.0f, 0.67f, 1.33f, 2.0f, 2.67f, 3.33f},
         "Jazz",
         "Triplet-based swing feel",
         {"Take Five - Dave Brubeck"}},

        {"bossa nova",
         {0.0f, 0.5f, 1.5f, 2.5f, 3.0f},
         "Latin",
         "Syncopated, lilting",
         {"Girl from Ipanema"}},

        {"clave 3-2",
         {0.0f, 1.5f, 3.0f, 4.0f, 5.5f},
         "Afro-Cuban",
         "3-2 son clave pattern",
         {"Oye Como Va - Tito Puente"}}
    };
}

void RhythmEngine::initializeEmotionRhythmMappings() {
    emotionRhythmMappings_["excitement"] = {
        "excitement",
        4.0f,  // High density
        0.7f,  // High syncopation
        {"trap hi-hats", "double-time feel"},
        {{4, 4, "Duple", "Walking rhythm"}}
    };

    emotionRhythmMappings_["calm"] = {
        "calm",
        1.0f,  // Low density
        0.1f,  // Low syncopation
        {"quarter notes", "half notes"},
        {{4, 4, "Duple", "Walking rhythm"}, {3, 4, "Triple", "Waltz"}}
    };

    emotionRhythmMappings_["tension"] = {
        "tension",
        3.0f,  // Medium-high density
        0.8f,  // Very high syncopation
        {"syncopated backbeat"},
        {{5, 4, "Asymmetric", "3+2 grouping"}, {7, 8, "Asymmetric", "Complex"}}
    };
}

void RhythmEngine::initializeClavePatterns() {
    // Normalized to 2-bar cycle (8 beats in 4/4)
    clavePatterns_["son clave 3-2"] = {0.0f, 1.5f, 3.0f, 4.0f, 5.5f};
    clavePatterns_["son clave 2-3"] = {0.0f, 1.5f, 4.0f, 5.5f, 7.0f};
    clavePatterns_["rumba clave 3-2"] = {0.0f, 1.5f, 3.0f, 4.0f, 6.0f};
    clavePatterns_["bossa nova"] = {0.0f, 1.0f, 1.5f, 3.0f, 4.5f, 6.0f, 7.0f};
}

float RhythmEngine::snapToGrid(float time, int subdivision) const {
    float gridSize = 1.0f / (subdivision / 4.0f); // Assuming quarter note = 1.0
    return std::round(time / gridSize) * gridSize;
}

std::vector<float> RhythmEngine::createGrid(float duration, int subdivision) const {
    std::vector<float> grid;
    float gridSize = 1.0f / (subdivision / 4.0f);

    for (float t = 0.0f; t <= duration; t += gridSize) {
        grid.push_back(t);
    }

    return grid;
}

float RhythmEngine::calculateMean(const std::vector<float>& values) const {
    if (values.empty()) return 0.0f;

    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / values.size();
}

float RhythmEngine::calculateStdDev(const std::vector<float>& values) const {
    if (values.size() < 2) return 0.0f;

    float mean = calculateMean(values);
    float variance = 0.0f;

    for (float val : values) {
        float diff = val - mean;
        variance += diff * diff;
    }

    variance /= values.size();

    return std::sqrt(variance);
}

float RhythmEngine::calculateVariance(const std::vector<float>& values) const {
    if (values.empty()) return 0.0f;

    float stdDev = calculateStdDev(values);
    return stdDev * stdDev;
}

float RhythmEngine::calculatePatternSimilarity(
    const std::vector<float>& pattern1,
    const std::vector<float>& pattern2) const
{
    if (pattern1.size() != pattern2.size()) {
        return 0.0f;
    }

    if (pattern1.empty()) return 1.0f;

    // Calculate normalized correlation
    float sumDiff = 0.0f;
    float maxDiff = 0.0f;

    for (size_t i = 0; i < pattern1.size(); ++i) {
        float diff = std::abs(pattern1[i] - pattern2[i]);
        sumDiff += diff;
        maxDiff = std::max(maxDiff, std::max(pattern1[i], pattern2[i]));
    }

    float avgDiff = sumDiff / pattern1.size();

    if (maxDiff <= 0.0f) return 1.0f;

    float similarity = 1.0f - (avgDiff / maxDiff);

    return std::clamp(similarity, 0.0f, 1.0f);
}

std::vector<float> RhythmEngine::findSwingPairs(
    const std::vector<float>& onsetTimes,
    int subdivision) const
{
    std::vector<float> pairs;

    float beatDuration = 4.0f / subdivision; // e.g., 8th note = 0.5 beats

    // Find consecutive notes that could be swing pairs
    for (size_t i = 0; i < onsetTimes.size() - 1; ++i) {
        float interval = onsetTimes[i + 1] - onsetTimes[i];

        if (interval < beatDuration * 1.5f) { // Within swing range
            pairs.push_back(onsetTimes[i]);
            pairs.push_back(onsetTimes[i + 1]);
        }
    }

    return pairs;
}

std::vector<float> RhythmEngine::detectBeats(
    const std::vector<float>& onsetTimes,
    const TimeSignature& timeSignature) const
{
    std::vector<float> beats;

    if (onsetTimes.empty()) return beats;

    float tempo = calculateTempo(onsetTimes);
    float beatDuration = 60.0f / tempo;

    float duration = onsetTimes.back();

    for (float t = 0.0f; t <= duration; t += beatDuration) {
        beats.push_back(t);
    }

    return beats;
}

int RhythmEngine::calculateGCD(int a, int b) const {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int RhythmEngine::calculateLCM(int a, int b) const {
    return (a * b) / calculateGCD(a, b);
}

bool RhythmEngine::isOnStrongBeat(float time, const TimeSignature& ts) const {
    float beatTime = std::fmod(time, static_cast<float>(ts.numerator));
    float roundedBeat = std::round(beatTime);

    // Strong beats: 1 in any meter, 3 in 4/4
    return (roundedBeat == 0.0f || (ts.numerator == 4 && roundedBeat == 2.0f));
}

bool RhythmEngine::isOnWeakBeat(float time, const TimeSignature& ts) const {
    float beatTime = std::fmod(time, static_cast<float>(ts.numerator));
    float roundedBeat = std::round(beatTime);

    // Weak beats depend on meter
    return !isOnStrongBeat(time, ts) &&
           std::abs(beatTime - roundedBeat) < 0.1f;
}

float RhythmEngine::calculateAverageShift(const std::vector<float>& shifts) const {
    return calculateMean(shifts);
}

std::string RhythmEngine::classifyPocketStyle(float avgShift, float stdDev) const {
    if (stdDev < 5.0f) {
        if (std::abs(avgShift) < 3.0f) {
            return "On the beat (tight pocket)";
        } else if (avgShift < -3.0f) {
            return "Behind the beat (laid back)";
        } else {
            return "Ahead of the beat (pushing)";
        }
    } else if (stdDev < 15.0f) {
        return "Medium pocket (some timing variation)";
    } else {
        return "Loose pocket (high timing variation)";
    }
}

} // namespace midikompanion::theory
