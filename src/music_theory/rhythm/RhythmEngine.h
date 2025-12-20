#pragma once
/*
 * RhythmEngine.h - Time, Meter, Groove, and Rhythm Analysis
 * ==========================================================
 *
 * Analyzes and generates rhythmic structures:
 * - Meter and time signature analysis
 * - Groove detection and quantization
 * - Polyrhythm and polymeter identification
 * - Micro-timing and "feel" analysis
 * - Rhythmic pattern generation
 * - Swing ratio calculation
 *
 * CONNECTIONS:
 * - Uses: CoreTheoryEngine (for mathematical patterns)
 * - Used by: IntentPipeline (rhythm generation from emotion)
 * - Used by: Existing groove system (enhance with theory)
 * - UI: Shows why certain grooves create specific feels
 */

#include "../Types.h"
#include "../core/CoreTheoryEngine.h"
#include <memory>
#include <chrono>

namespace midikompanion::theory {

class RhythmEngine {
public:
    explicit RhythmEngine(std::shared_ptr<CoreTheoryEngine> coreTheory);
    ~RhythmEngine() = default;

    //==========================================================================
    // Time Signature and Meter
    //==========================================================================

    /**
     * Analyze time signature from MIDI events
     *
     * @param onsetTimes MIDI note onset times (in beats)
     * @param duration Total duration (in beats)
     * @return Detected time signature with feel and body response
     *
     * Example:
     *   auto ts = analyzeTimeSignature(onsets, 16.0);
     *   // Returns: {numerator: 4, denominator: 4, feel: "Duple",
     *   //           bodyResponse: "Walking rhythm"}
     */
    TimeSignature analyzeTimeSignature(const std::vector<float>& onsetTimes,
                                       float duration) const;

    /**
     * Detect meter feel (Duple, Triple, Compound)
     */
    std::string detectMeterFeel(int numerator, int denominator) const;

    /**
     * Get body response for meter
     *
     * Examples:
     * - 4/4: "Walking rhythm" (left, right, left, right)
     * - 3/4: "Waltz" (strong-weak-weak)
     * - 6/8: "Lilting" (compound duple)
     * - 5/4: "Asymmetric" (3+2 or 2+3)
     */
    std::string getBodyResponse(const TimeSignature& ts) const;

    /**
     * Explain why a time signature feels the way it does
     */
    std::string explainTimeSignature(const TimeSignature& ts,
                                     ExplanationDepth depth) const;

    /**
     * Suggest time signature for emotional effect
     *
     * @param emotion Target emotion
     * @return Recommended time signatures with explanations
     */
    struct TimeSignatureSuggestion {
        TimeSignature timeSignature;
        std::string emotionalEffect;
        std::string explanation;
        std::vector<std::string> examples; // Songs using this meter
    };

    std::vector<TimeSignatureSuggestion> suggestTimeSignatureForEmotion(
        const std::string& emotion
    ) const;

    //==========================================================================
    // Groove Analysis
    //==========================================================================

    /**
     * Analyze groove from MIDI performance
     *
     * @param onsetTimes Actual note onset times (seconds or beats)
     * @param velocities MIDI velocities (0-127)
     * @param timeSignature Time signature context
     * @return Complete groove analysis with feel and pocket
     *
     * Example:
     *   auto groove = analyzeGroove(onsets, velocities, ts);
     *   // Returns: Swing ratio, pocket description, micro-timing shifts
     */
    GrooveAnalysis analyzeGroove(
        const std::vector<float>& onsetTimes,
        const std::vector<int>& velocities,
        const TimeSignature& timeSignature
    ) const;

    /**
     * Quantize performance while preserving groove
     *
     * @param onsetTimes Original onset times
     * @param quantizeStrength 0.0 = no quantize, 1.0 = full quantize
     * @param preserveSwing Keep swing ratio intact
     * @return Quantized onsets with micro-timing preserved
     */
    std::vector<float> quantizeWithGroove(
        const std::vector<float>& onsetTimes,
        float quantizeStrength = 0.8f,
        bool preserveSwing = true
    ) const;

    /**
     * Calculate swing ratio
     *
     * @param onsetTimes Onset times
     * @param subdivision Beat subdivision (8th notes, 16th notes)
     * @return Swing ratio (1.0 = straight, 2.0 = triplet swing, 1.5 = medium swing)
     *
     * Example:
     *   auto ratio = calculateSwingRatio(onsets, 8); // 8th note swing
     *   // Returns: 1.67 (medium swing)
     */
    float calculateSwingRatio(const std::vector<float>& onsetTimes,
                             int subdivision = 8) const;

    /**
     * Detect groove type
     *
     * Examples: "Straight 16ths", "Swing 8ths", "Half-time feel",
     *           "Dilla swing", "Boom bap", "Trap hi-hats"
     */
    std::string detectGrooveType(const GrooveAnalysis& analysis) const;

    /**
     * Analyze micro-timing (the "pocket")
     *
     * @param onsetTimes Actual performance times
     * @param quantizedTimes Perfectly quantized grid
     * @return Micro-timing shifts in milliseconds
     *
     * Positive = ahead of beat, Negative = behind beat
     */
    std::vector<float> analyzeMicroTiming(
        const std::vector<float>& onsetTimes,
        const std::vector<float>& quantizedTimes
    ) const;

    /**
     * Describe pocket quality
     *
     * Returns: "On the beat", "Behind the beat (laid back)",
     *          "Ahead of the beat (pushing)", "Unstable (rushing/dragging)"
     */
    std::string describePocket(const std::vector<float>& microTimingShifts) const;

    /**
     * Calculate pocket width (acceptable timing variation)
     *
     * Tight pocket: ±5ms
     * Medium pocket: ±15ms
     * Loose pocket: ±30ms
     */
    float calculatePocketWidth(const std::vector<float>& microTimingShifts) const;

    //==========================================================================
    // Polyrhythm and Polymeter
    //==========================================================================

    /**
     * Detect polyrhythm (multiple rhythmic layers)
     *
     * @param layers Multiple rhythm tracks
     * @return Polyrhythm analysis with ratios
     *
     * Example:
     *   auto poly = detectPolyrhythm({{layer1}, {layer2}});
     *   // Returns: "3:2 polyrhythm" (3 against 2)
     */
    struct PolyrhythmAnalysis {
        std::vector<int> ratios;           // [3, 2] for 3:2
        std::string description;           // "3 against 2"
        std::vector<float> meetingPoints;  // Where rhythms align
        float tension;                     // Rhythmic tension (0-1)
        std::string perceptualEffect;      // "Creates forward momentum"
        std::vector<std::string> examples; // Famous uses
    };

    PolyrhythmAnalysis detectPolyrhythm(
        const std::vector<std::vector<float>>& layers
    ) const;

    /**
     * Detect polymeter (multiple simultaneous meters)
     *
     * Example: 3/4 melody over 4/4 accompaniment
     */
    struct PolymeterAnalysis {
        std::vector<TimeSignature> meters;
        int barsUntilAlignment;            // How many bars until they sync
        std::string perceptualEffect;
        std::string explanation;
    };

    PolymeterAnalysis detectPolymeter(
        const std::vector<std::vector<float>>& layers,
        float duration
    ) const;

    /**
     * Calculate rhythmic tension between layers
     * @return 0.0 (in phase) to 1.0 (maximum tension)
     */
    float calculateRhythmicTension(
        const std::vector<std::vector<float>>& layers
    ) const;

    /**
     * Find meeting points (where polyrhythms align)
     */
    std::vector<float> findMeetingPoints(
        const std::vector<int>& ratios,
        float duration
    ) const;

    //==========================================================================
    // Rhythmic Pattern Generation
    //==========================================================================

    /**
     * Generate rhythmic pattern for emotion
     *
     * @param emotion Target emotion
     * @param timeSignature Time signature context
     * @param density Note density (0.0 = sparse, 1.0 = dense)
     * @param numBars Number of bars to generate
     * @return Rhythmic pattern with onsets and durations
     *
     * Example:
     *   auto pattern = generatePatternForEmotion("excitement", ts, 0.7, 4);
     *   // Returns: High-density pattern with syncopation
     */
    RhythmicPattern generatePatternForEmotion(
        const std::string& emotion,
        const TimeSignature& timeSignature,
        float density = 0.5f,
        int numBars = 4
    ) const;

    /**
     * Generate pattern from description
     *
     * Examples: "driving 8th notes", "syncopated backbeat", "trap hi-hats"
     */
    RhythmicPattern generatePatternFromDescription(
        const std::string& description,
        const TimeSignature& timeSignature,
        int numBars = 4
    ) const;

    /**
     * Add syncopation to pattern
     *
     * @param pattern Original pattern
     * @param syncopationAmount 0.0 = none, 1.0 = maximum
     * @return Pattern with syncopated accents
     */
    RhythmicPattern addSyncopation(
        const RhythmicPattern& pattern,
        float syncopationAmount
    ) const;

    /**
     * Apply swing to pattern
     *
     * @param pattern Straight pattern
     * @param swingRatio Swing amount (1.0 = straight, 2.0 = triplet)
     * @return Swung pattern
     */
    RhythmicPattern applySwing(
        const RhythmicPattern& pattern,
        float swingRatio
    ) const;

    /**
     * Humanize pattern (add micro-timing variations)
     *
     * @param pattern Perfect pattern
     * @param humanization Amount of variation (0.0-1.0)
     * @param pocketStyle "On beat", "Behind", "Ahead"
     * @return Humanized pattern with natural timing
     */
    RhythmicPattern humanizePattern(
        const RhythmicPattern& pattern,
        float humanization = 0.3f,
        const std::string& pocketStyle = "On beat"
    ) const;

    //==========================================================================
    // Rhythmic Complexity Analysis
    //==========================================================================

    /**
     * Calculate rhythmic complexity
     *
     * Factors:
     * - Number of unique durations
     * - Syncopation level
     * - Polyrhythmic elements
     * - Micro-timing variation
     *
     * @return Complexity score (0.0 = simple, 1.0 = very complex)
     */
    float calculateComplexity(const RhythmicPattern& pattern) const;

    /**
     * Detect syncopation
     *
     * @return Syncopation score (0.0 = none, 1.0 = highly syncopated)
     */
    float detectSyncopation(const std::vector<float>& onsetTimes,
                           const TimeSignature& timeSignature) const;

    /**
     * Analyze rhythmic density
     *
     * @return Notes per beat
     */
    float calculateDensity(const std::vector<float>& onsetTimes,
                          float duration) const;

    /**
     * Detect rhythmic motifs (repeating patterns)
     */
    struct RhythmicMotif {
        std::vector<float> pattern;        // IOI pattern (inter-onset intervals)
        int occurrences;                   // How many times it appears
        std::vector<float> positions;      // Where it appears
        std::string name;                  // "Charleston rhythm", "Clave"
    };

    std::vector<RhythmicMotif> detectMotifs(
        const std::vector<float>& onsetTimes,
        float minOccurrences = 2
    ) const;

    /**
     * Identify clave patterns
     *
     * Detects: Son clave, Rumba clave, Bossa nova clave
     */
    std::optional<std::string> detectClavePattern(
        const std::vector<float>& onsetTimes
    ) const;

    //==========================================================================
    // Rhythmic Perception
    //==========================================================================

    /**
     * Analyze beat strength (which beats feel accented)
     *
     * @param onsetTimes Note onsets
     * @param velocities MIDI velocities
     * @param timeSignature Time signature
     * @return Beat strength for each beat position
     *
     * Example:
     *   auto strengths = analyzeBeatStrength(onsets, velocities, ts);
     *   // Returns: [1.0, 0.3, 0.7, 0.4] for 4/4 (downbeat strongest)
     */
    std::vector<float> analyzeBeatStrength(
        const std::vector<float>& onsetTimes,
        const std::vector<int>& velocities,
        const TimeSignature& timeSignature
    ) const;

    /**
     * Calculate rhythmic stability
     *
     * Stable = consistent timing, clear pulse
     * Unstable = rushing, dragging, inconsistent
     *
     * @return Stability score (0.0 = unstable, 1.0 = rock solid)
     */
    float calculateStability(const std::vector<float>& onsetTimes) const;

    /**
     * Detect tempo changes (rubato, accelerando, ritardando)
     */
    struct TempoChange {
        float startTime;
        float endTime;
        float startTempo;
        float endTempo;
        std::string type;                  // "Accelerando", "Ritardando", "Rubato"
        std::string musicalEffect;
    };

    std::vector<TempoChange> detectTempoChanges(
        const std::vector<float>& onsetTimes
    ) const;

    /**
     * Explain why a rhythm feels the way it does
     */
    std::string explainRhythmicFeel(
        const RhythmicPattern& pattern,
        ExplanationDepth depth
    ) const;

    //==========================================================================
    // Common Rhythmic Patterns Database
    //==========================================================================

    /**
     * Get common rhythmic pattern by name
     *
     * Examples: "four-on-the-floor", "backbeat", "double-time feel",
     *           "Charleston", "Bossa nova", "Afro-Cuban 6/8"
     */
    RhythmicPattern getCommonPattern(
        const std::string& name,
        const TimeSignature& timeSignature
    ) const;

    /**
     * List all available patterns
     */
    std::vector<std::string> listCommonPatterns() const;

    /**
     * Find similar rhythmic patterns
     */
    std::vector<RhythmicPattern> findSimilarPatterns(
        const RhythmicPattern& pattern,
        float similarityThreshold = 0.7f
    ) const;

    //==========================================================================
    // Utilities
    //==========================================================================

    /**
     * Convert onset times to inter-onset intervals (IOI)
     */
    std::vector<float> onsetsToIOI(const std::vector<float>& onsetTimes) const;

    /**
     * Convert inter-onset intervals to onset times
     */
    std::vector<float> ioiToOnsets(const std::vector<float>& intervals) const;

    /**
     * Quantize to grid
     *
     * @param onsetTimes Original times
     * @param subdivision Grid subdivision (16 = 16th notes)
     * @return Quantized times
     */
    std::vector<float> quantizeToGrid(
        const std::vector<float>& onsetTimes,
        int subdivision = 16
    ) const;

    /**
     * Calculate tempo from onset times
     * @return BPM
     */
    float calculateTempo(const std::vector<float>& onsetTimes) const;

    /**
     * Detect time signature changes
     */
    std::vector<std::pair<float, TimeSignature>> detectTimeSignatureChanges(
        const std::vector<float>& onsetTimes,
        float duration
    ) const;

private:
    //==========================================================================
    // Internal Data
    //==========================================================================

    std::shared_ptr<CoreTheoryEngine> coreTheory_;

    // Common rhythmic patterns database
    struct PatternTemplate {
        std::string name;
        std::vector<float> onsetPattern;   // Normalized to 1 bar
        std::string genre;
        std::string feel;
        std::vector<std::string> examples;
    };
    std::vector<PatternTemplate> patternDatabase_;

    // Emotion → rhythm mappings
    struct EmotionRhythmMap {
        std::string emotion;
        float targetDensity;               // Notes per beat
        float syncopationLevel;            // 0-1
        std::vector<std::string> suggestedPatterns;
        std::vector<TimeSignature> preferredMeters;
    };
    std::map<std::string, EmotionRhythmMap> emotionRhythmMappings_;

    // Clave patterns (normalized)
    std::map<std::string, std::vector<float>> clavePatterns_;

    //==========================================================================
    // Internal Helpers
    //==========================================================================

    void initializePatternDatabase();
    void initializeEmotionRhythmMappings();
    void initializeClavePatterns();

    // Grid and quantization
    float snapToGrid(float time, int subdivision) const;
    std::vector<float> createGrid(float duration, int subdivision) const;

    // Statistical analysis
    float calculateMean(const std::vector<float>& values) const;
    float calculateStdDev(const std::vector<float>& values) const;
    float calculateVariance(const std::vector<float>& values) const;

    // Pattern matching
    float calculatePatternSimilarity(
        const std::vector<float>& pattern1,
        const std::vector<float>& pattern2
    ) const;

    // Swing detection
    std::vector<float> findSwingPairs(
        const std::vector<float>& onsetTimes,
        int subdivision
    ) const;

    // Beat detection
    std::vector<float> detectBeats(
        const std::vector<float>& onsetTimes,
        const TimeSignature& timeSignature
    ) const;

    // Polyrhythm calculation
    int calculateGCD(int a, int b) const;
    int calculateLCM(int a, int b) const;

    // Syncopation scoring
    bool isOnStrongBeat(float time, const TimeSignature& ts) const;
    bool isOnWeakBeat(float time, const TimeSignature& ts) const;

    // Micro-timing
    float calculateAverageShift(const std::vector<float>& shifts) const;
    std::string classifyPocketStyle(float avgShift, float stdDev) const;
};

} // namespace midikompanion::theory
