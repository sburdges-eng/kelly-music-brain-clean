#pragma once
/*
 * HarmonyEngine.h - Chord Relationships and Progressions
 * =======================================================
 *
 * Analyzes and generates harmonic structures:
 * - Chord construction and quality analysis
 * - Functional harmony (Tonic-Dominant-Subdominant)
 * - Voice leading and part-writing
 * - Chord substitutions and reharmonization
 * - Emotional mapping (emotion → chord progression)
 *
 * CONNECTIONS:
 * - Uses: CoreTheoryEngine (scales, intervals)
 * - Uses: EmotionThesaurus (emotional context)
 * - Used by: IntentPipeline (MIDI generation)
 * - UI: Shows why specific progressions create emotional effects
 */

#include "../Types.h"
#include "../core/CoreTheoryEngine.h"
#include <memory>

namespace midikompanion::theory {

class HarmonyEngine {
public:
    explicit HarmonyEngine(std::shared_ptr<CoreTheoryEngine> coreTheory);
    ~HarmonyEngine() = default;

    //==========================================================================
    // Chord Construction
    //==========================================================================

    /**
     * Build chord from root note and quality
     *
     * @param rootNote MIDI root note (0-127)
     * @param quality Chord quality (Major, Minor, Dominant7, etc.)
     * @param voicing Voicing style (Close, Open, Drop2, etc.)
     * @return Complete chord with all notes and metadata
     *
     * Example:
     *   auto cmaj7 = buildChord(60, ChordQuality::Major7, VoicingType::Close);
     *   // Returns: {root: 60, notes: [60, 64, 67, 71], symbol: "Cmaj7"}
     */
    enum class VoicingType {
        Close,      // All notes within an octave
        Open,       // Spread across multiple octaves
        Drop2,      // Jazz: 2nd voice dropped an octave
        Drop3,      // Jazz: 3rd voice dropped an octave
        Root,       // Root position (1-3-5-7)
        FirstInv,   // First inversion (3-5-7-1)
        SecondInv,  // Second inversion (5-7-1-3)
        ThirdInv    // Third inversion (7-1-3-5)
    };

    Chord buildChord(int rootNote,
                    ChordQuality quality,
                    VoicingType voicing = VoicingType::Close) const;

    /**
     * Analyze chord from MIDI notes
     *
     * @param notes MIDI note numbers
     * @return Identified chord with quality, root, and inversions
     *
     * Example:
     *   auto chord = analyzeChord({60, 64, 67}); // C Major
     *   // Returns: {root: 60, quality: Major, symbol: "C", romanNumeral: "I"}
     */
    Chord analyzeChord(const std::vector<int>& notes) const;

    /**
     * Identify chord extensions (9th, 11th, 13th)
     */
    std::vector<std::string> identifyExtensions(const std::vector<int>& notes) const;

    /**
     * Detect chord inversion
     * @return 0=root position, 1=first inversion, 2=second, 3=third
     */
    int detectInversion(const Chord& chord) const;

    /**
     * Get all possible voicings for a chord
     */
    std::vector<Chord> getAllVoicings(int rootNote, ChordQuality quality) const;

    //==========================================================================
    // Functional Harmony Analysis
    //==========================================================================

    /**
     * Analyze chord progression using functional harmony
     *
     * @param progression Sequence of chords
     * @param key Current key
     * @return Complete analysis with Roman numerals, function, tension
     *
     * Example:
     *   // C-Am-F-G progression
     *   auto analysis = analyzeProgression(chords, "C");
     *   // Returns: {romanNumerals: ["I", "vi", "IV", "V"],
     *   //           functionalAnalysis: "T-T-S-D",
     *   //           explanation: "Classic pop progression with subdominant..."}
     */
    struct ProgressionAnalysis {
        std::vector<std::string> romanNumerals;
        std::vector<std::string> functions;      // "Tonic", "Dominant", "Subdominant"
        std::string functionalAnalysis;          // "T-D-T-S-D-T"
        std::array<float, 100> tensionCurve;    // Tension over time (0.0-1.0)
        std::vector<std::string> cadences;       // "Perfect Authentic", "Half"
        std::string emotionalArc;                // "Building tension → resolution"
        std::string explanation;
        std::vector<std::string> famousExamples;
        float cohesion;                          // How well chords fit together (0-1)
    };

    ProgressionAnalysis analyzeProgression(const std::vector<Chord>& chords,
                                          const std::string& key) const;

    /**
     * Detect cadence type (Perfect Authentic, Plagal, Half, Deceptive)
     */
    struct Cadence {
        std::string type;           // "Perfect Authentic Cadence"
        std::string romanNumerals;  // "V7 → I"
        float finality;             // 1.0 = complete resolution, 0.0 = open
        std::string explanation;
    };

    Cadence detectCadence(const Chord& penultimate, const Chord& final) const;

    /**
     * Calculate harmonic function (Tonic, Dominant, Subdominant)
     */
    std::string getHarmonicFunction(const Chord& chord, const std::string& key) const;

    /**
     * Get Roman numeral analysis
     */
    std::string getRomanNumeral(const Chord& chord, const std::string& key) const;

    /**
     * Calculate tension level of chord in context
     * @return 0.0 (stable/tonic) to 1.0 (maximum tension)
     */
    float calculateTension(const Chord& chord, const std::string& key) const;

    //==========================================================================
    // Voice Leading
    //==========================================================================

    /**
     * Generate smooth voice leading between two chords
     *
     * @param fromChord Starting chord
     * @param toChord Destination chord
     * @param style Voice leading style (Bach SATB, Jazz, etc.)
     * @param numVoices Number of voices (default: 4 for SATB)
     * @return Voice leading suggestion with all voice movements
     *
     * Example:
     *   auto voicing = suggestVoiceLeading(cMajor, gMajor,
     *                                      VoiceLeadingStyle::Bach, 4);
     *   // Returns: Smooth SATB voice leading with common tone retention
     */
    VoiceLeadingSuggestion suggestVoiceLeading(
        const Chord& fromChord,
        const Chord& toChord,
        VoiceLeadingStyle style = VoiceLeadingStyle::Bach,
        int numVoices = 4
    ) const;

    /**
     * Check for voice leading errors
     */
    struct VoiceLeadingError {
        std::string errorType;      // "Parallel Fifths", "Hidden Octaves"
        std::vector<int> voices;    // Which voices have the error
        std::string explanation;
        std::string whenAcceptable; // "Acceptable in metal riffs"
        bool isStrict;              // True = always wrong, False = context-dependent
    };

    std::vector<VoiceLeadingError> checkVoiceLeading(
        const std::vector<std::vector<int>>& voices,
        VoiceLeadingStyle style
    ) const;

    /**
     * Find common tones between chords (for smooth transitions)
     */
    std::vector<int> findCommonTones(const Chord& chord1, const Chord& chord2) const;

    /**
     * Calculate voice leading smoothness
     * @return 0.0 (very jumpy) to 1.0 (perfectly smooth)
     */
    float calculateSmoothness(const std::vector<std::vector<int>>& voices) const;

    //==========================================================================
    // Chord Substitution and Reharmonization
    //==========================================================================

    /**
     * Suggest chord substitutions
     *
     * @param chord Original chord
     * @param key Current key
     * @param substitutionType Type of substitution to apply
     * @return List of possible substitute chords with explanations
     *
     * Example:
     *   auto subs = suggestSubstitutions(dominantChord, "C",
     *                                    SubstitutionType::TritoneSub);
     *   // Returns: Db7 (tritone substitution for G7)
     */
    enum class SubstitutionType {
        Diatonic,       // Stay in key (vi for I, etc.)
        TritoneSub,     // Dominant substitution
        Secondary,      // Secondary dominants (V/V, V/ii)
        Modal,          // Borrowed chords from parallel modes
        Chromatic,      // Chromatic mediants
        All             // Show all possibilities
    };

    struct Substitution {
        Chord originalChord;
        Chord substituteChord;
        SubstitutionType type;
        std::string explanation;
        std::string soundDescription;  // "Darker", "Jazzier", "More tense"
        float tensionChange;            // How tension changes (-1.0 to +1.0)
        std::vector<std::string> examples; // Songs using this substitution
    };

    std::vector<Substitution> suggestSubstitutions(
        const Chord& chord,
        const std::string& key,
        SubstitutionType type = SubstitutionType::All
    ) const;

    /**
     * Generate secondary dominant
     * @param targetChord Chord to tonicize
     * @return Secondary dominant (e.g., V/V, V/ii)
     */
    Chord generateSecondaryDominant(const Chord& targetChord) const;

    /**
     * Get borrowed chords from parallel mode
     * @param key Current key (e.g., "C major")
     * @return Chords borrowed from parallel minor/major
     */
    std::vector<Chord> getBorrowedChords(const std::string& key) const;

    /**
     * Reharmonize melody
     * @param melody Sequence of MIDI notes
     * @param style Harmonic style (Jazz, Classical, Pop, etc.)
     * @return Suggested chord progression under melody
     */
    ChordProgression reharmonizeMelody(
        const std::vector<int>& melody,
        const std::string& key,
        const std::string& style = "Pop"
    ) const;

    //==========================================================================
    // Emotion-Driven Harmony Generation
    //==========================================================================

    /**
     * Generate chord progression for specific emotion
     *
     * @param emotion Emotion name ("joy", "grief", "hope", etc.)
     * @param key Starting key
     * @param numChords Number of chords to generate (default: 4)
     * @param allowRuleBreaking Allow unconventional progressions (default: true)
     * @return Progression designed to evoke the emotion
     *
     * Example:
     *   auto prog = generateProgressionForEmotion("grief", "D minor", 4);
     *   // Returns: Progression with descending bass, minor chords,
     *   //          suspended resolutions, explanation of why it works
     */
    ChordProgression generateProgressionForEmotion(
        const std::string& emotion,
        const std::string& key,
        int numChords = 4,
        bool allowRuleBreaking = true
    ) const;

    /**
     * Map emotion to harmonic characteristics
     */
    struct EmotionalHarmony {
        std::vector<ChordQuality> preferredQualities;
        std::vector<std::string> preferredProgressions; // Roman numerals
        float tensionLevel;                             // Target tension
        std::string bassMovement;                       // "Descending", "Static"
        std::vector<std::string> avoidances;
        std::string explanation;
    };

    EmotionalHarmony getEmotionalHarmony(const std::string& emotion) const;

    /**
     * Analyze emotional content of progression
     * @return Detected emotions with confidence scores
     */
    std::map<std::string, float> analyzeEmotionalContent(
        const ChordProgression& progression
    ) const;

    //==========================================================================
    // Common Progressions Database
    //==========================================================================

    /**
     * Get common progression by name
     *
     * Examples: "12-bar blues", "I-V-vi-IV", "ii-V-I", "Pachelbel Canon"
     */
    ChordProgression getCommonProgression(
        const std::string& name,
        const std::string& key
    ) const;

    /**
     * List all available common progressions
     */
    std::vector<std::string> listCommonProgressions() const;

    /**
     * Find similar progressions in database
     */
    std::vector<ChordProgression> findSimilarProgressions(
        const ChordProgression& progression,
        float similarityThreshold = 0.7f
    ) const;

    //==========================================================================
    // Utilities
    //==========================================================================

    /**
     * Transpose chord progression to new key
     */
    ChordProgression transposeProgression(
        const ChordProgression& progression,
        int semitones
    ) const;

    /**
     * Transpose chord progression to new key (by name)
     */
    ChordProgression transposeProgression(
        const ChordProgression& progression,
        const std::string& fromKey,
        const std::string& toKey
    ) const;

    /**
     * Convert chord symbols to MIDI notes
     * @param chordSymbol "Cmaj7", "Dm7b5", "G7#9"
     * @return MIDI notes for the chord
     */
    std::vector<int> chordSymbolToMidi(const std::string& chordSymbol,
                                       int octave = 4) const;

    /**
     * Convert MIDI notes to chord symbol
     */
    std::string midiToChordSymbol(const std::vector<int>& notes) const;

    /**
     * Explain why a progression works (or doesn't)
     */
    std::string explainProgression(const ChordProgression& progression,
                                   ExplanationDepth depth) const;

private:
    //==========================================================================
    // Internal Data
    //==========================================================================

    std::shared_ptr<CoreTheoryEngine> coreTheory_;

    // Common chord progressions database
    struct ProgressionTemplate {
        std::string name;
        std::vector<std::string> romanNumerals;
        std::string genre;
        std::vector<std::string> examples;
        std::string emotionalEffect;
    };
    std::vector<ProgressionTemplate> progressionDatabase_;

    // Emotion → harmony mappings
    struct EmotionHarmonyMap {
        std::string emotion;
        std::vector<int> preferredIntervals;
        std::vector<ChordQuality> preferredQualities;
        float targetTension;
        std::vector<std::string> progressionPatterns;
    };
    std::map<std::string, EmotionHarmonyMap> emotionMappings_;

    // Voice leading rules database
    struct VoiceLeadingRule {
        std::string ruleName;
        VoiceLeadingStyle applicableStyle;
        bool isStrict;  // True = always apply, False = preference
        std::string explanation;
        std::string whenToBreak;
    };
    std::vector<VoiceLeadingRule> voiceLeadingRules_;

    //==========================================================================
    // Internal Helpers
    //==========================================================================

    void initializeProgressionDatabase();
    void initializeEmotionMappings();
    void initializeVoiceLeadingRules();

    // Chord analysis helpers
    int identifyRoot(const std::vector<int>& notes) const;
    ChordQuality identifyQuality(const std::vector<int>& intervals) const;
    std::vector<int> getIntervalsFromRoot(const std::vector<int>& notes, int root) const;

    // Voice leading calculation
    float calculateVoiceDistance(const std::vector<std::vector<int>>& voices) const;
    bool hasParallelMotion(int voice1, int voice2,
                          const std::vector<int>& fromChord,
                          const std::vector<int>& toChord,
                          int interval) const;

    // Functional harmony helpers
    int getScaleDegree(int midiNote, const std::string& key) const;
    std::string getFunctionFromDegree(int degree) const;
    float calculateTensionFromFunction(const std::string& function) const;

    // Emotion mapping
    std::vector<std::string> selectChordQualities(const std::string& emotion) const;
    std::vector<std::string> selectProgressionPattern(const std::string& emotion) const;
    float getEmotionalTensionTarget(const std::string& emotion) const;

    // Substitution logic
    Chord tritoneSubstitute(const Chord& chord) const;
    std::vector<Chord> getDiatonicSubstitutes(const Chord& chord,
                                              const std::string& key) const;
    std::vector<Chord> getModalInterchange(const Chord& chord,
                                           const std::string& key) const;
};

} // namespace midikompanion::theory
