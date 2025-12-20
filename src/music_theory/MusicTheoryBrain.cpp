/*
 * MusicTheoryBrain.cpp - Main Music Theory Engine Integration Implementation
 * ===========================================================================
 */

#include "MusicTheoryBrain.h"
#include <algorithm>
#include <sstream>
#include <map>
#include <cmath>

namespace midikompanion::theory {

//==============================================================================
// Constructor
//==============================================================================

MusicTheoryBrain::MusicTheoryBrain()
    : preferredExplanationType_(ExplanationType::Intuitive)
    , defaultExplanationDepth_(ExplanationDepth::Intermediate)
    , allowRuleBreaking_(true)
{
    // Initialize all engines
    coreTheory_ = std::make_shared<CoreTheoryEngine>();
    harmony_ = std::make_shared<HarmonyEngine>(coreTheory_);
    rhythm_ = std::make_shared<RhythmEngine>(coreTheory_);
    knowledge_ = std::make_shared<KnowledgeGraph>();
}

//==============================================================================
// High-Level Analysis
//==============================================================================

MusicTheoryBrain::CompleteAnalysis MusicTheoryBrain::analyzeMIDI(
    const std::vector<int>& midiNotes,
    const std::vector<float>& onsetTimes,
    const std::vector<int>& velocities) const
{
    CompleteAnalysis analysis;

    if (midiNotes.empty()) return analysis;

    // 1. Detect key and scale
    analysis.detectedKey = detectKeyFromNotes(midiNotes);
    analysis.detectedScale = detectScaleFromNotes(midiNotes);

    // 2. Detect chords
    analysis.chords = detectChordsFromMIDI(midiNotes, onsetTimes);

    // 3. Analyze chord progression
    if (!analysis.chords.empty()) {
        analysis.progressionAnalysis =
            harmony_->analyzeProgression(analysis.chords, analysis.detectedKey);
    }

    // 4. Analyze rhythm
    if (!onsetTimes.empty()) {
        float duration = onsetTimes.back();
        analysis.timeSignature = rhythm_->analyzeTimeSignature(onsetTimes, duration);

        if (!velocities.empty()) {
            analysis.grooveAnalysis =
                rhythm_->analyzeGroove(onsetTimes, velocities, analysis.timeSignature);
        }
    }

    // 5. Detect polyrhythms (if multiple layers)
    // Simplified: would need layer separation

    // 6. Identify concepts from MIDI
    analysis.concepts = knowledge_->identifyConceptsFromMIDI(midiNotes, onsetTimes);

    // 7. Generate overall explanation
    std::ostringstream explanation;
    explanation << "Musical Analysis:\n\n";
    explanation << "Key: " << analysis.detectedKey << "\n";
    explanation << "Scale: " << analysis.detectedScale.name << "\n";

    if (!analysis.chords.empty()) {
        explanation << "Chord Progression: ";
        for (const auto& rn : analysis.progressionAnalysis.romanNumerals) {
            explanation << rn << " ";
        }
        explanation << "\n";
        explanation << "Functional Analysis: "
                   << analysis.progressionAnalysis.functionalAnalysis << "\n";
    }

    explanation << "Time Signature: "
               << analysis.timeSignature.numerator << "/"
               << analysis.timeSignature.denominator << "\n";
    explanation << "Feel: " << analysis.timeSignature.feel << "\n";

    if (!analysis.grooveAnalysis.grooveQuality.empty()) {
        explanation << "Groove: " << analysis.grooveAnalysis.grooveQuality << "\n";
        explanation << "Pocket: " << analysis.grooveAnalysis.pocketDescription << "\n";
    }

    analysis.overallExplanation = explanation.str();

    // 8. Identify learning opportunities
    for (const auto& concept : analysis.concepts) {
        analysis.learningOpportunities.push_back(
            "Learn about: " + concept.concept + " (" + concept.explanation + ")"
        );
    }

    return analysis;
}

std::string MusicTheoryBrain::analyzeAndExplain(
    const std::vector<int>& midiNotes,
    const std::vector<float>& onsetTimes,
    ExplanationDepth depth) const
{
    auto analysis = analyzeMIDI(midiNotes, onsetTimes, {});

    if (depth == ExplanationDepth::Simple) {
        return "Key: " + analysis.detectedKey +
               ", Time: " + std::to_string(analysis.timeSignature.numerator) +
               "/" + std::to_string(analysis.timeSignature.denominator);
    }

    return analysis.overallExplanation;
}

//==============================================================================
// Generation (Emotion → Music)
//==============================================================================

MusicTheoryBrain::GeneratedMusic MusicTheoryBrain::generateFromEmotion(
    const std::string& emotion,
    int numBars,
    bool allowRuleBreaking) const
{
    GeneratedMusic result;

    // 1. Select key and scale based on emotion
    if (emotion == "joy" || emotion == "happy") {
        result.key = "C Major";
        result.scale = coreTheory_->generateMajorScale(60); // C4
    } else if (emotion == "grief" || emotion == "sad") {
        result.key = "D Minor";
        result.scale = coreTheory_->generateNaturalMinorScale(62); // D4
    } else if (emotion == "hope") {
        result.key = "F Major";
        result.scale = coreTheory_->generateMajorScale(65); // F4
    } else if (emotion == "tension" || emotion == "anxious") {
        result.key = "B Minor";
        result.scale = coreTheory_->generateNaturalMinorScale(71); // B4
    } else {
        // Default
        result.key = "C Major";
        result.scale = coreTheory_->generateMajorScale(60);
    }

    // 2. Generate chord progression for emotion
    result.progression = harmony_->generateProgressionForEmotion(
        emotion, result.key, numBars, allowRuleBreaking
    );

    // 3. Suggest time signature for emotion
    auto tsSuggestions = rhythm_->suggestTimeSignatureForEmotion(emotion);
    if (!tsSuggestions.empty()) {
        result.timeSignature = tsSuggestions[0].timeSignature;
    } else {
        result.timeSignature = {4, 4, "Duple", "Walking rhythm"};
    }

    // 4. Generate rhythmic pattern for emotion
    result.rhythm = rhythm_->generatePatternForEmotion(
        emotion, result.timeSignature, 0.5f, numBars
    );

    // 5. Convert to MIDI
    result.midiNotes = progressionToMIDI(result.progression, 4);
    result.onsetTimes = rhythmToOnsetTimes(result.rhythm);

    // Generate velocities based on emotion
    int baseVelocity = 80;
    if (emotion == "excitement" || emotion == "angry") {
        baseVelocity = 100;
    } else if (emotion == "calm" || emotion == "peaceful") {
        baseVelocity = 60;
    }

    for (size_t i = 0; i < result.midiNotes.size(); ++i) {
        result.velocities.push_back(baseVelocity);
    }

    // 6. Generate explanations
    result.whyThisProgression = "This progression was chosen to evoke " + emotion +
                               " through " + result.progression.harmonyExplanation;

    result.whyThisRhythm = "The rhythmic pattern supports " + emotion +
                          " with " + result.rhythm.perceptualGroove;

    result.emotionalExplanation =
        "Musical structure designed to create emotional response of " + emotion;

    result.famousExamples = result.progression.famousExamples;

    return result;
}

MusicTheoryBrain::GeneratedMusic MusicTheoryBrain::generateFromDescription(
    const std::string& description,
    int numBars) const
{
    GeneratedMusic result;

    // Parse description
    auto parsed = parseDescription(description);

    // Set key
    if (!parsed.key.empty()) {
        result.key = parsed.key;
    } else {
        result.key = "C Major";
    }

    // Generate scale
    int rootNote = 60; // C4
    result.scale = coreTheory_->generateMajorScale(rootNote);

    // Get progression
    if (!parsed.progression.empty()) {
        result.progression = harmony_->getCommonProgression(
            parsed.progression, result.key
        );
    } else {
        // Default progression
        result.progression.key = result.key;
        result.progression.romanNumerals = {"I", "IV", "V", "I"};
    }

    // Get rhythmic pattern
    TimeSignature ts{4, 4, "Duple", "Walking rhythm"};
    result.timeSignature = ts;

    if (!parsed.rhythmPattern.empty()) {
        result.rhythm = rhythm_->getCommonPattern(parsed.rhythmPattern, ts);
    } else {
        result.rhythm = rhythm_->generatePatternFromDescription("steady", ts, numBars);
    }

    // Convert to MIDI
    result.midiNotes = progressionToMIDI(result.progression, 4);
    result.onsetTimes = rhythmToOnsetTimes(result.rhythm);

    for (size_t i = 0; i < result.midiNotes.size(); ++i) {
        result.velocities.push_back(80);
    }

    result.whyThisProgression = "Generated from description: " + description;

    return result;
}

MusicTheoryBrain::ReharmonizationResult MusicTheoryBrain::reharmonizeMelody(
    const std::vector<int>& melody,
    const std::string& style) const
{
    ReharmonizationResult result;

    if (melody.empty()) return result;

    // Detect original key
    std::string key = detectKeyFromNotes(melody);

    // Generate original simple harmonization
    result.originalProgression = harmony_->reharmonizeMelody(melody, key, "Simple");

    // Generate new harmonization
    result.newProgression = harmony_->reharmonizeMelody(melody, key, style);

    // Compare and identify substitutions
    if (result.originalProgression.chords.size() ==
        result.newProgression.chords.size()) {

        for (size_t i = 0; i < result.originalProgression.chords.size(); ++i) {
            auto& origChord = result.originalProgression.chords[i];
            auto& newChord = result.newProgression.chords[i];

            if (origChord.symbol != newChord.symbol) {
                // Substitution occurred
                auto subs = harmony_->suggestSubstitutions(
                    origChord, key, HarmonyEngine::SubstitutionType::All
                );

                for (const auto& sub : subs) {
                    if (sub.substituteChord.symbol == newChord.symbol) {
                        result.substitutionsMade.push_back(sub);
                        break;
                    }
                }
            }
        }
    }

    // Generate explanation
    std::ostringstream explanation;
    explanation << "Reharmonization in " << style << " style:\n";
    explanation << "Made " << result.substitutionsMade.size() << " substitutions.\n";

    for (const auto& sub : result.substitutionsMade) {
        explanation << "- " << sub.originalChord.symbol << " → "
                   << sub.substituteChord.symbol << ": "
                   << sub.explanation << "\n";
    }

    result.explanation = explanation.str();

    return result;
}

//==============================================================================
// Interactive Learning
//==============================================================================

std::string MusicTheoryBrain::askQuestion(
    const std::string& question,
    const UserProfile& userProfile) const
{
    std::string lowerQuestion = toLowerCase(question);

    // Route to appropriate handler
    if (containsKeyword(lowerQuestion, "interval")) {
        return answerIntervalQuestion(question, userProfile);
    } else if (containsKeyword(lowerQuestion, "chord")) {
        return answerChordQuestion(question, userProfile);
    } else if (containsKeyword(lowerQuestion, "progression")) {
        return answerProgressionQuestion(question, userProfile);
    } else if (containsKeyword(lowerQuestion, "rhythm") ||
               containsKeyword(lowerQuestion, "groove")) {
        return answerRhythmQuestion(question, userProfile);
    }

    // General search in knowledge base
    auto concepts = knowledge_->searchConcepts(question);
    if (!concepts.empty()) {
        auto concept = concepts[0];
        return knowledge_->explainConcept(
            concept.concept,
            userProfile.preferredExplanationStyle,
            defaultExplanationDepth_
        );
    }

    return "I'm not sure about that. Could you rephrase your question?";
}

KnowledgeGraph::Curriculum MusicTheoryBrain::getCustomLearningPath(
    const std::string& targetConcept,
    const UserProfile& userProfile) const
{
    return knowledge_->generateCurriculum(userProfile, {targetConcept}, 20);
}

MusicTheoryBrain::PracticeSession MusicTheoryBrain::generatePracticeSession(
    UserProfile& userProfile,
    int duration) const
{
    PracticeSession session;

    // Get struggling concepts
    auto struggling = knowledge_->identifyStrugglingConcepts(userProfile);

    // Get review concepts (spaced repetition)
    auto review = knowledge_->suggestReview(userProfile);

    // Combine and prioritize
    std::vector<std::string> focusAreas;
    focusAreas.insert(focusAreas.end(), struggling.begin(), struggling.end());
    focusAreas.insert(focusAreas.end(), review.begin(), review.end());

    // Remove duplicates
    std::sort(focusAreas.begin(), focusAreas.end());
    focusAreas.erase(std::unique(focusAreas.begin(), focusAreas.end()),
                    focusAreas.end());

    // Generate exercises
    int timeRemaining = duration;
    DifficultyLevel level = userProfile.currentLevel;

    for (const auto& concept : focusAreas) {
        if (timeRemaining <= 0) break;

        Exercise exercise = knowledge_->generateExercise(concept, level);
        session.exercises.push_back(exercise);

        timeRemaining -= 5; // Assume 5 minutes per exercise
    }

    session.focusAreas = focusAreas;
    session.estimatedDuration = duration - timeRemaining;
    session.sessionGoal = "Practice " + std::to_string(session.exercises.size()) +
                         " concepts";

    return session;
}

std::string MusicTheoryBrain::explainEmotionalEffect(
    const std::vector<int>& midiNotes,
    const std::vector<float>& onsetTimes,
    const std::string& perceivedEmotion) const
{
    auto analysis = analyzeMIDI(midiNotes, onsetTimes, {});

    std::ostringstream explanation;

    explanation << "Why this sounds " << perceivedEmotion << ":\n\n";

    // Harmonic explanation
    if (!analysis.chords.empty()) {
        explanation << "HARMONY:\n";
        explanation << "The chord progression creates " << perceivedEmotion
                   << " through:\n";
        explanation << "- " << analysis.progressionAnalysis.functionalAnalysis
                   << " functional movement\n";
        explanation << "- Average tension: "
                   << (analysis.progressionAnalysis.tensionCurve[50] * 100)
                   << "%\n";
        explanation << "- " << analysis.progressionAnalysis.emotionalArc << "\n\n";
    }

    // Rhythmic explanation
    explanation << "RHYTHM:\n";
    explanation << "The rhythmic feel supports " << perceivedEmotion << " with:\n";
    explanation << "- " << analysis.grooveAnalysis.grooveQuality << " groove\n";
    explanation << "- " << analysis.grooveAnalysis.pocketDescription << "\n\n";

    // Scale/mode explanation
    explanation << "TONALITY:\n";
    explanation << "The " << analysis.detectedScale.name << " scale provides:\n";
    explanation << "- " << analysis.detectedScale.characteristicSound << "\n";

    return explanation.str();
}

//==============================================================================
// Validation and Suggestions
//==============================================================================

MusicTheoryBrain::TheoryValidation MusicTheoryBrain::validateTheory(
    const std::vector<int>& midiNotes,
    const std::vector<float>& onsetTimes,
    const std::string& style) const
{
    TheoryValidation validation;
    validation.hasIssues = false;

    if (midiNotes.empty()) return validation;

    // Detect chords
    auto chords = detectChordsFromMIDI(midiNotes, onsetTimes);

    if (chords.size() < 2) return validation;

    // Check voice leading
    for (size_t i = 0; i < chords.size() - 1; ++i) {
        VoiceLeadingStyle voiceStyle = VoiceLeadingStyle::Pop;
        if (style == "Bach" || style == "Classical") {
            voiceStyle = VoiceLeadingStyle::Bach;
        } else if (style == "Jazz") {
            voiceStyle = VoiceLeadingStyle::Jazz;
        }

        auto suggestion = harmony_->suggestVoiceLeading(
            chords[i], chords[i + 1], voiceStyle
        );

        // Check for errors
        auto errors = harmony_->checkVoiceLeading(suggestion.voices, voiceStyle);

        validation.voiceLeadingIssues.insert(
            validation.voiceLeadingIssues.end(),
            errors.begin(),
            errors.end()
        );

        if (!errors.empty()) {
            validation.hasIssues = true;
        }
    }

    // Generate suggestions
    if (validation.hasIssues) {
        validation.suggestions.push_back(
            "Consider adjusting voice leading to avoid parallel motion"
        );
    }

    // Context awareness
    for (const auto& error : validation.voiceLeadingIssues) {
        validation.whenAcceptable.push_back(error.whenAcceptable);
    }

    return validation;
}

std::vector<MusicTheoryBrain::ImprovementSuggestion>
MusicTheoryBrain::suggestImprovements(
    const std::vector<int>& midiNotes,
    const std::vector<float>& onsetTimes,
    const std::string& targetEmotion) const
{
    std::vector<ImprovementSuggestion> suggestions;

    // Analyze current music
    auto current = analyzeMIDI(midiNotes, onsetTimes, {});

    // Analyze target emotion characteristics
    auto emotionalChars = getEmotionalCharacteristics(targetEmotion);

    // Suggest harmonic improvements
    if (!current.chords.empty()) {
        // Check if progression matches emotion
        auto emotionAnalysis = harmony_->analyzeEmotionalContent(
            current.progressionAnalysis
        );

        bool matchesTarget = false;
        for (const auto& [emotion, confidence] : emotionAnalysis) {
            if (emotion == targetEmotion && confidence > 0.6f) {
                matchesTarget = true;
            }
        }

        if (!matchesTarget) {
            ImprovementSuggestion suggestion;
            suggestion.suggestion = "Adjust chord progression to better match " +
                                   targetEmotion;
            suggestion.reasoning = "Current progression doesn't strongly evoke " +
                                  targetEmotion;
            suggestion.improvedVersion = generateFromEmotion(targetEmotion, 4, true);
            suggestion.expectedImprovement = 0.7f;
            suggestions.push_back(suggestion);
        }
    }

    // Suggest rhythmic improvements
    if (!onsetTimes.empty()) {
        ImprovementSuggestion suggestion;
        suggestion.suggestion = "Adjust rhythmic feel";
        suggestion.reasoning = "Rhythm could better support emotional intent";
        suggestion.expectedImprovement = 0.5f;
        suggestions.push_back(suggestion);
    }

    return suggestions;
}

std::vector<HarmonyEngine::Substitution> MusicTheoryBrain::suggestSubstitutions(
    const ChordProgression& progression,
    const std::string& style) const
{
    std::vector<HarmonyEngine::Substitution> allSubs;

    for (const auto& chord : progression.chords) {
        HarmonyEngine::SubstitutionType type = HarmonyEngine::SubstitutionType::All;

        if (style == "Jazz") {
            type = HarmonyEngine::SubstitutionType::TritoneSub;
        } else if (style == "Classical") {
            type = HarmonyEngine::SubstitutionType::Diatonic;
        }

        auto subs = harmony_->suggestSubstitutions(chord, progression.key, type);
        allSubs.insert(allSubs.end(), subs.begin(), subs.end());
    }

    return allSubs;
}

//==============================================================================
// Common Quick Actions
//==============================================================================

std::string MusicTheoryBrain::identifyInterval(int note1, int note2) const {
    auto interval = coreTheory_->calculateInterval(note1, note2);
    return interval.intervalName;
}

std::string MusicTheoryBrain::identifyChord(const std::vector<int>& notes) const {
    auto chord = harmony_->analyzeChord(notes);
    return chord.symbol;
}

std::string MusicTheoryBrain::identifyScale(const std::vector<int>& notes) const {
    auto scale = detectScaleFromNotes(notes);
    return scale.name;
}

std::string MusicTheoryBrain::identifyRhythm(
    const std::vector<float>& onsetTimes) const
{
    TimeSignature ts{4, 4, "Duple", "Walking rhythm"};
    GrooveAnalysis analysis = rhythm_->analyzeGroove(onsetTimes, {}, ts);
    return rhythm_->detectGrooveType(analysis);
}

std::string MusicTheoryBrain::findKey(const std::vector<int>& notes) const {
    return detectKeyFromNotes(notes);
}

std::vector<Chord> MusicTheoryBrain::suggestNextChord(
    const std::vector<Chord>& progressionSoFar,
    const std::string& key) const
{
    std::vector<Chord> suggestions;

    if (progressionSoFar.empty()) {
        // Suggest tonic
        suggestions.push_back(harmony_->buildChord(60, ChordQuality::Major));
        return suggestions;
    }

    // Analyze last chord's function
    auto lastChord = progressionSoFar.back();
    std::string function = harmony_->getHarmonicFunction(lastChord, key);

    // Suggest based on function
    if (function == "Tonic") {
        // After tonic, go to subdominant or dominant
        suggestions.push_back(harmony_->buildChord(65, ChordQuality::Major)); // IV
        suggestions.push_back(harmony_->buildChord(67, ChordQuality::Dominant7)); // V7
    } else if (function == "Dominant") {
        // After dominant, resolve to tonic
        suggestions.push_back(harmony_->buildChord(60, ChordQuality::Major)); // I
    } else if (function == "Subdominant") {
        // After subdominant, go to dominant or tonic
        suggestions.push_back(harmony_->buildChord(67, ChordQuality::Dominant7)); // V7
        suggestions.push_back(harmony_->buildChord(60, ChordQuality::Major)); // I
    }

    return suggestions;
}

//==============================================================================
// Settings and Configuration
//==============================================================================

void MusicTheoryBrain::setPreferredExplanationStyle(ExplanationType type) {
    preferredExplanationType_ = type;
}

void MusicTheoryBrain::setDefaultExplanationDepth(ExplanationDepth depth) {
    defaultExplanationDepth_ = depth;
}

void MusicTheoryBrain::setAllowRuleBreaking(bool allow) {
    allowRuleBreaking_ = allow;
}

bool MusicTheoryBrain::loadKnowledgeBase(const std::string& dataDirectory) {
    return knowledge_->loadFromJSON(dataDirectory);
}

MusicTheoryBrain::SystemStats MusicTheoryBrain::getStatistics() const {
    SystemStats stats;

    stats.knowledgeStats = knowledge_->getStatistics();
    stats.totalConcepts = stats.knowledgeStats.totalConcepts;

    stats.totalProgressions = harmony_->listCommonProgressions().size();
    stats.totalRhythmicPatterns = rhythm_->listCommonPatterns().size();

    return stats;
}

//==============================================================================
// Internal Helpers - Analysis
//==============================================================================

std::string MusicTheoryBrain::detectKeyFromNotes(
    const std::vector<int>& notes) const
{
    if (notes.empty()) return "C Major";

    // Simplified key detection: analyze pitch class distribution
    std::map<int, int> pitchClassCounts;

    for (int note : notes) {
        int pc = note % 12;
        pitchClassCounts[pc]++;
    }

    // Find most common pitch class (likely tonic)
    int maxCount = 0;
    int tonicPC = 0;

    for (const auto& [pc, count] : pitchClassCounts) {
        if (count > maxCount) {
            maxCount = count;
            tonicPC = pc;
        }
    }

    // Convert pitch class to note name
    const std::string noteNames[] = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };

    // Determine if major or minor (simplified heuristic)
    bool hasMajorThird = pitchClassCounts.find((tonicPC + 4) % 12) != pitchClassCounts.end();
    bool hasMinorThird = pitchClassCounts.find((tonicPC + 3) % 12) != pitchClassCounts.end();

    std::string quality = hasMajorThird ? " Major" : " Minor";

    return noteNames[tonicPC] + quality;
}

Scale MusicTheoryBrain::detectScaleFromNotes(const std::vector<int>& notes) const {
    if (notes.empty()) {
        return coreTheory_->generateMajorScale(60);
    }

    // Determine root note
    int rootNote = notes[0];

    // Try to match against known scales
    auto majorScale = coreTheory_->generateMajorScale(rootNote);
    auto minorScale = coreTheory_->generateNaturalMinorScale(rootNote);

    // Check which scale fits better
    int majorMatches = 0;
    int minorMatches = 0;

    for (int note : notes) {
        if (coreTheory_->isNoteInScale(note, majorScale)) {
            majorMatches++;
        }
        if (coreTheory_->isNoteInScale(note, minorScale)) {
            minorMatches++;
        }
    }

    return (majorMatches >= minorMatches) ? majorScale : minorScale;
}

std::vector<Chord> MusicTheoryBrain::detectChordsFromMIDI(
    const std::vector<int>& notes,
    const std::vector<float>& onsetTimes) const
{
    std::vector<Chord> chords;

    // Group simultaneous notes
    auto groups = groupSimultaneousNotes(notes, onsetTimes);

    for (const auto& group : groups) {
        if (group.notes.size() >= 3) {
            auto chord = harmony_->analyzeChord(group.notes);
            chords.push_back(chord);
        }
    }

    return chords;
}

//==============================================================================
// Internal Helpers - Generation
//==============================================================================

std::vector<int> MusicTheoryBrain::progressionToMIDI(
    const ChordProgression& progression,
    int octave) const
{
    std::vector<int> midiNotes;

    for (const auto& chord : progression.chords) {
        // Add chord notes
        for (int note : chord.notes) {
            midiNotes.push_back(note);
        }
    }

    return midiNotes;
}

std::vector<float> MusicTheoryBrain::rhythmToOnsetTimes(
    const RhythmicPattern& rhythm) const
{
    return rhythm.onsetTimes;
}

std::vector<int> MusicTheoryBrain::generateVelocities(
    size_t numNotes,
    const GrooveAnalysis& groove) const
{
    std::vector<int> velocities;

    int baseVelocity = 80;

    for (size_t i = 0; i < numNotes; ++i) {
        // Add variation based on position
        int velocity = baseVelocity + (i % 4 == 0 ? 10 : 0); // Accent downbeats
        velocities.push_back(std::clamp(velocity, 1, 127));
    }

    return velocities;
}

//==============================================================================
// Internal Helpers - Parsing
//==============================================================================

MusicTheoryBrain::ParsedDescription MusicTheoryBrain::parseDescription(
    const std::string& description) const
{
    ParsedDescription parsed;

    std::string lower = toLowerCase(description);

    // Parse progression
    if (lower.find("ii-v-i") != std::string::npos) {
        parsed.progression = "ii-V-I";
    } else if (lower.find("i-v-vi-iv") != std::string::npos) {
        parsed.progression = "I-V-vi-IV";
    } else if (lower.find("12-bar") != std::string::npos) {
        parsed.progression = "12-bar blues";
    }

    // Parse key
    if (lower.find("c major") != std::string::npos) {
        parsed.key = "C Major";
    } else if (lower.find("f minor") != std::string::npos) {
        parsed.key = "F Minor";
    } else if (lower.find("g major") != std::string::npos) {
        parsed.key = "G Major";
    }

    // Parse rhythm pattern
    if (lower.find("four-on-the-floor") != std::string::npos) {
        parsed.rhythmPattern = "four-on-the-floor";
    } else if (lower.find("swing") != std::string::npos) {
        parsed.rhythmPattern = "swing 8ths";
    } else if (lower.find("bossa") != std::string::npos) {
        parsed.rhythmPattern = "bossa nova";
    }

    // Parse style
    if (lower.find("jazz") != std::string::npos) {
        parsed.style = "Jazz";
    } else if (lower.find("classical") != std::string::npos) {
        parsed.style = "Classical";
    } else if (lower.find("pop") != std::string::npos) {
        parsed.style = "Pop";
    }

    return parsed;
}

//==============================================================================
// Internal Helpers - Question Answering
//==============================================================================

std::string MusicTheoryBrain::answerIntervalQuestion(
    const std::string& question,
    const UserProfile& profile) const
{
    // Check if asking "what is"
    if (toLowerCase(question).find("what is") != std::string::npos) {
        // Extract interval name from question
        if (question.find("fifth") != std::string::npos) {
            return coreTheory_->explainInterval(7, defaultExplanationDepth_);
        } else if (question.find("third") != std::string::npos) {
            return coreTheory_->explainInterval(4, defaultExplanationDepth_);
        }
    }

    return "I can explain intervals. Try asking 'What is a perfect fifth?'";
}

std::string MusicTheoryBrain::answerChordQuestion(
    const std::string& question,
    const UserProfile& profile) const
{
    return "Chord questions coming soon. Ask about specific chord types.";
}

std::string MusicTheoryBrain::answerProgressionQuestion(
    const std::string& question,
    const UserProfile& profile) const
{
    return "Progression questions coming soon. Ask about specific progressions.";
}

std::string MusicTheoryBrain::answerRhythmQuestion(
    const std::string& question,
    const UserProfile& profile) const
{
    return "Rhythm questions coming soon. Ask about time signatures or grooves.";
}

//==============================================================================
// Internal Helpers - Utilities
//==============================================================================

std::vector<MusicTheoryBrain::MIDIGroup>
MusicTheoryBrain::groupSimultaneousNotes(
    const std::vector<int>& notes,
    const std::vector<float>& onsetTimes,
    float tolerance) const
{
    std::vector<MIDIGroup> groups;

    if (notes.size() != onsetTimes.size()) return groups;

    // Group notes that occur within tolerance window
    size_t i = 0;
    while (i < notes.size()) {
        MIDIGroup group;
        group.startTime = onsetTimes[i];
        group.notes.push_back(notes[i]);

        // Find all notes within tolerance
        for (size_t j = i + 1; j < notes.size(); ++j) {
            if (std::abs(onsetTimes[j] - group.startTime) <= tolerance) {
                group.notes.push_back(notes[j]);
            } else {
                break;
            }
        }

        // Calculate duration (until next group)
        if (i + group.notes.size() < notes.size()) {
            group.duration = onsetTimes[i + group.notes.size()] - group.startTime;
        } else {
            group.duration = 1.0f; // Default duration
        }

        groups.push_back(group);
        i += group.notes.size();
    }

    return groups;
}

std::vector<std::string> MusicTheoryBrain::getEmotionalCharacteristics(
    const std::string& emotion) const
{
    // Return characteristics for emotion
    std::vector<std::string> characteristics;

    if (emotion == "joy") {
        characteristics = {"Major tonality", "Moderate-fast tempo", "High energy"};
    } else if (emotion == "grief") {
        characteristics = {"Minor tonality", "Slow tempo", "Low energy"};
    } else if (emotion == "hope") {
        characteristics = {"Major tonality", "Ascending melodies", "Building dynamics"};
    }

    return characteristics;
}

std::string MusicTheoryBrain::toLowerCase(const std::string& str) const {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::vector<std::string> MusicTheoryBrain::tokenize(const std::string& str) const {
    std::vector<std::string> tokens;
    std::istringstream stream(str);
    std::string token;

    while (stream >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

bool MusicTheoryBrain::containsKeyword(
    const std::string& text,
    const std::string& keyword) const
{
    return text.find(keyword) != std::string::npos;
}

} // namespace midikompanion::theory
