#pragma once
/*
 * MusicTheoryBrain.h - Main Music Theory Engine Integration
 * ==========================================================
 *
 * Unified API that integrates all music theory engines:
 * - CoreTheoryEngine: Intervals, scales, tuning systems
 * - HarmonyEngine: Chords, progressions, voice leading
 * - RhythmEngine: Groove, meter, polyrhythm
 * - KnowledgeGraph: Concept relationships, learning paths
 *
 * This is the primary interface for:
 * - IntentPipeline (MIDI generation from emotion)
 * - EmotionWorkstation UI (interactive learning)
 * - Python Adaptive Teacher (ML-driven personalization)
 *
 * DESIGN PHILOSOPHY:
 * - Simple high-level API for common tasks
 * - Direct access to specialized engines when needed
 * - Automatic cross-engine integration (e.g., chord analysis uses intervals)
 * - Educational explanations at every level
 */

#include "Types.h"
#include "core/CoreTheoryEngine.h"
#include "harmony/HarmonyEngine.h"
#include "rhythm/RhythmEngine.h"
#include "knowledge/KnowledgeGraph.h"
#include <memory>

namespace midikompanion::theory {

class MusicTheoryBrain {
public:
    MusicTheoryBrain();
    ~MusicTheoryBrain() = default;

    //==========================================================================
    // High-Level Analysis (Automatic Integration)
    //==========================================================================

    /**
     * Analyze complete MIDI performance
     *
     * Automatically detects and explains:
     * - Key and scale
     * - Chord progression with functional analysis
     * - Rhythm and groove
     * - Time signature
     * - Detected music theory concepts
     *
     * @param midiNotes MIDI note numbers
     * @param onsetTimes Note onset times (seconds)
     * @param velocities MIDI velocities
     * @return Complete analysis with explanations
     */
    struct CompleteAnalysis {
        // Harmonic analysis
        std::string detectedKey;
        Scale detectedScale;
        std::vector<Chord> chords;
        HarmonyEngine::ProgressionAnalysis progressionAnalysis;

        // Rhythmic analysis
        TimeSignature timeSignature;
        GrooveAnalysis grooveAnalysis;
        std::vector<RhythmEngine::PolyrhythmAnalysis> polyrhythms;

        // Detected concepts
        std::vector<KnowledgeGraph::IdentifiedConcept> concepts;

        // Educational explanations
        std::string overallExplanation;
        std::vector<std::string> theoryConcepts;
        std::vector<std::string> learningOpportunities;
    };

    CompleteAnalysis analyzeMIDI(
        const std::vector<int>& midiNotes,
        const std::vector<float>& onsetTimes,
        const std::vector<int>& velocities
    ) const;

    /**
     * Analyze and explain in one call
     *
     * Simplified analysis with explanation at specified depth
     *
     * @param midiNotes MIDI notes
     * @param onsetTimes Onset times
     * @param depth Explanation depth
     * @return Human-readable analysis
     */
    std::string analyzeAndExplain(
        const std::vector<int>& midiNotes,
        const std::vector<float>& onsetTimes,
        ExplanationDepth depth = ExplanationDepth::Intermediate
    ) const;

    //==========================================================================
    // Generation (Emotion → Music)
    //==========================================================================

    /**
     * Generate complete musical structure from emotion
     *
     * Creates:
     * - Chord progression
     * - Rhythmic pattern
     * - Time signature
     * - Key and scale
     * - Complete explanation of choices
     *
     * @param emotion Target emotion
     * @param numBars Number of bars to generate
     * @param allowRuleBreaking Allow unconventional choices
     * @return Complete musical structure with MIDI data
     */
    struct GeneratedMusic {
        ChordProgression progression;
        RhythmicPattern rhythm;
        TimeSignature timeSignature;
        std::string key;
        Scale scale;

        // MIDI output
        std::vector<int> midiNotes;
        std::vector<float> onsetTimes;
        std::vector<int> velocities;

        // Explanations
        std::string whyThisProgression;
        std::string whyThisRhythm;
        std::string emotionalExplanation;
        std::vector<std::string> famousExamples;
    };

    GeneratedMusic generateFromEmotion(
        const std::string& emotion,
        int numBars = 4,
        bool allowRuleBreaking = true
    ) const;

    /**
     * Generate MIDI from music theory description
     *
     * Examples:
     * - "ii-V-I in C major with swing feel"
     * - "Four-on-the-floor in F minor"
     * - "Bossa nova clave with jazzy chords"
     *
     * @param description Natural language description
     * @param numBars Number of bars
     * @return Generated MIDI with explanation
     */
    GeneratedMusic generateFromDescription(
        const std::string& description,
        int numBars = 4
    ) const;

    /**
     * Reharmonize melody
     *
     * @param melody MIDI melody
     * @param style Harmonic style
     * @return New chord progression with explanation
     */
    struct ReharmonizationResult {
        ChordProgression newProgression;
        ChordProgression originalProgression;
        std::vector<HarmonyEngine::Substitution> substitutionsMade;
        std::string explanation;
    };

    ReharmonizationResult reharmonizeMelody(
        const std::vector<int>& melody,
        const std::string& style = "Jazz"
    ) const;

    //==========================================================================
    // Interactive Learning
    //==========================================================================

    /**
     * Ask the music theory brain a question
     *
     * Examples:
     * - "What is a tritone substitution?"
     * - "Why does this chord progression work?"
     * - "How do I modulate from C to G?"
     *
     * @param question Natural language question
     * @param userProfile User's learning profile
     * @return Answer tailored to user's level
     */
    std::string askQuestion(
        const std::string& question,
        const UserProfile& userProfile
    ) const;

    /**
     * Get learning path to master a concept
     *
     * @param targetConcept Goal (e.g., "Jazz Reharmonization")
     * @param userProfile Current user state
     * @return Step-by-step curriculum
     */
    KnowledgeGraph::Curriculum getCustomLearningPath(
        const std::string& targetConcept,
        const UserProfile& userProfile
    ) const;

    /**
     * Practice session generator
     *
     * Generates exercises based on:
     * - User's current mastery
     * - Struggling concepts
     * - Recommended review (spaced repetition)
     *
     * @param userProfile User state
     * @param duration Session duration (minutes)
     * @return Practice session with exercises
     */
    struct PracticeSession {
        std::vector<Exercise> exercises;
        std::vector<std::string> focusAreas;
        std::string sessionGoal;
        int estimatedDuration;
    };

    PracticeSession generatePracticeSession(
        UserProfile& userProfile,
        int duration = 30
    ) const;

    /**
     * Explain why something sounds the way it does
     *
     * "Why does this sound sad/happy/tense/resolved?"
     *
     * @param midiNotes MIDI notes to analyze
     * @param onsetTimes Onset times
     * @param perceivedEmotion What it sounds like
     * @return Multi-level explanation
     */
    std::string explainEmotionalEffect(
        const std::vector<int>& midiNotes,
        const std::vector<float>& onsetTimes,
        const std::string& perceivedEmotion
    ) const;

    //==========================================================================
    // Validation and Suggestions
    //==========================================================================

    /**
     * Check for music theory errors
     *
     * @param midiNotes MIDI to check
     * @param onsetTimes Onset times
     * @param style Style context (Bach, Jazz, Pop, etc.)
     * @return Errors with explanations of when they're acceptable
     */
    struct TheoryValidation {
        std::vector<HarmonyEngine::VoiceLeadingError> voiceLeadingIssues;
        std::vector<std::string> harmonicIssues;
        std::vector<std::string> rhythmicIssues;

        // Context-aware feedback
        std::vector<std::string> suggestions;
        std::vector<std::string> whenAcceptable;

        bool hasIssues;
    };

    TheoryValidation validateTheory(
        const std::vector<int>& midiNotes,
        const std::vector<float>& onsetTimes,
        const std::string& style = "General"
    ) const;

    /**
     * Suggest improvements
     *
     * @param midiNotes Current MIDI
     * @param onsetTimes Onset times
     * @param targetEmotion Desired emotional effect
     * @return Suggestions with reasoning
     */
    struct ImprovementSuggestion {
        std::string suggestion;
        std::string reasoning;
        GeneratedMusic improvedVersion;
        float expectedImprovement;  // 0-1
    };

    std::vector<ImprovementSuggestion> suggestImprovements(
        const std::vector<int>& midiNotes,
        const std::vector<float>& onsetTimes,
        const std::string& targetEmotion
    ) const;

    /**
     * Suggest chord substitutions for existing progression
     *
     * @param progression Current progression
     * @param style Substitution style (Jazz, Classical, etc.)
     * @return Multiple substitution options with explanations
     */
    std::vector<HarmonyEngine::Substitution> suggestSubstitutions(
        const ChordProgression& progression,
        const std::string& style
    ) const;

    //==========================================================================
    // Direct Engine Access
    //==========================================================================

    /**
     * Get direct access to specialized engines
     *
     * For advanced usage when you need specific engine features
     */
    const CoreTheoryEngine& getCoreTheory() const { return *coreTheory_; }
    const HarmonyEngine& getHarmony() const { return *harmony_; }
    const RhythmEngine& getRhythm() const { return *rhythm_; }
    const KnowledgeGraph& getKnowledge() const { return *knowledge_; }

    /**
     * Get mutable access (for updating user profiles)
     */
    CoreTheoryEngine& getCoreTheory() { return *coreTheory_; }
    HarmonyEngine& getHarmony() { return *harmony_; }
    RhythmEngine& getRhythm() { return *rhythm_; }
    KnowledgeGraph& getKnowledge() { return *knowledge_; }

    //==========================================================================
    // Common Quick Actions
    //==========================================================================

    /**
     * Quick interval identification
     */
    std::string identifyInterval(int note1, int note2) const;

    /**
     * Quick chord identification
     */
    std::string identifyChord(const std::vector<int>& notes) const;

    /**
     * Quick scale identification
     */
    std::string identifyScale(const std::vector<int>& notes) const;

    /**
     * Quick rhythm identification
     */
    std::string identifyRhythm(const std::vector<float>& onsetTimes) const;

    /**
     * Find what key these notes are in
     */
    std::string findKey(const std::vector<int>& notes) const;

    /**
     * Suggest next chord in progression
     */
    std::vector<Chord> suggestNextChord(
        const std::vector<Chord>& progressionSoFar,
        const std::string& key
    ) const;

    //==========================================================================
    // Settings and Configuration
    //==========================================================================

    /**
     * Set preferred explanation style for user
     */
    void setPreferredExplanationStyle(ExplanationType type);

    /**
     * Set default explanation depth
     */
    void setDefaultExplanationDepth(ExplanationDepth depth);

    /**
     * Enable/disable rule-breaking suggestions
     */
    void setAllowRuleBreaking(bool allow);

    /**
     * Load knowledge base from JSON files
     */
    bool loadKnowledgeBase(const std::string& dataDirectory);

    /**
     * Get system statistics
     */
    struct SystemStats {
        int totalConcepts;
        int totalProgressions;
        int totalRhythmicPatterns;
        KnowledgeGraph::GraphStatistics knowledgeStats;
    };

    SystemStats getStatistics() const;

private:
    //==========================================================================
    // Engine Instances
    //==========================================================================

    std::shared_ptr<CoreTheoryEngine> coreTheory_;
    std::shared_ptr<HarmonyEngine> harmony_;
    std::shared_ptr<RhythmEngine> rhythm_;
    std::shared_ptr<KnowledgeGraph> knowledge_;

    //==========================================================================
    // Configuration
    //==========================================================================

    ExplanationType preferredExplanationType_;
    ExplanationDepth defaultExplanationDepth_;
    bool allowRuleBreaking_;

    //==========================================================================
    // Internal Helpers
    //==========================================================================

    // Analysis helpers
    std::string detectKeyFromNotes(const std::vector<int>& notes) const;
    Scale detectScaleFromNotes(const std::vector<int>& notes) const;
    std::vector<Chord> detectChordsFromMIDI(
        const std::vector<int>& notes,
        const std::vector<float>& onsetTimes
    ) const;

    // Generation helpers
    std::vector<int> progressionToMIDI(
        const ChordProgression& progression,
        int octave = 4
    ) const;

    std::vector<float> rhythmToOnsetTimes(
        const RhythmicPattern& rhythm
    ) const;

    std::vector<int> generateVelocities(
        size_t numNotes,
        const GrooveAnalysis& groove
    ) const;

    // Natural language processing (simplified)
    struct ParsedDescription {
        std::string progression;
        std::string key;
        std::string rhythmPattern;
        std::string style;
    };

    ParsedDescription parseDescription(const std::string& description) const;

    // Question answering
    std::string answerIntervalQuestion(const std::string& question,
                                      const UserProfile& profile) const;
    std::string answerChordQuestion(const std::string& question,
                                   const UserProfile& profile) const;
    std::string answerProgressionQuestion(const std::string& question,
                                         const UserProfile& profile) const;
    std::string answerRhythmQuestion(const std::string& question,
                                    const UserProfile& profile) const;

    // Explanation generation
    std::string explainWithStyle(
        const std::string& conceptName,
        ExplanationType type,
        ExplanationDepth depth
    ) const;

    // MIDI grouping (for chord detection)
    struct MIDIGroup {
        std::vector<int> notes;
        float startTime;
        float duration;
    };

    std::vector<MIDIGroup> groupSimultaneousNotes(
        const std::vector<int>& notes,
        const std::vector<float>& onsetTimes,
        float tolerance = 0.05f
    ) const;

    // Emotion mapping
    std::vector<std::string> getEmotionalCharacteristics(
        const std::string& emotion
    ) const;

    // Error checking
    bool hasParallelFifths(const std::vector<Chord>& chords) const;
    bool hasUnresolvedTension(const ChordProgression& progression) const;

    // String utilities
    std::string toLowerCase(const std::string& str) const;
    std::vector<std::string> tokenize(const std::string& str) const;
    bool containsKeyword(const std::string& text, const std::string& keyword) const;
};

} // namespace midikompanion::theory
