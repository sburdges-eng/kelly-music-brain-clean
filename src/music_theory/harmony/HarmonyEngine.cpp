/*
 * HarmonyEngine.cpp - Chord Relationships and Progressions Implementation
 * ========================================================================
 */

#include "HarmonyEngine.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace midikompanion::theory {

//==============================================================================
// Constructor
//==============================================================================

HarmonyEngine::HarmonyEngine(std::shared_ptr<CoreTheoryEngine> coreTheory)
    : coreTheory_(coreTheory)
{
    initializeProgressionDatabase();
    initializeEmotionMappings();
    initializeVoiceLeadingRules();
}

//==============================================================================
// Chord Construction
//==============================================================================

Chord HarmonyEngine::buildChord(int rootNote, ChordQuality quality, VoicingType voicing) const {
    Chord chord;
    chord.rootNote = rootNote;
    chord.quality = quality;
    chord.notes.clear();

    // Build chord tones based on quality
    std::vector<int> intervals;

    switch (quality) {
        case ChordQuality::Major:
            intervals = {0, 4, 7}; // R, M3, P5
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1);
            break;

        case ChordQuality::Minor:
            intervals = {0, 3, 7}; // R, m3, P5
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "m";
            break;

        case ChordQuality::Diminished:
            intervals = {0, 3, 6}; // R, m3, d5
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "dim";
            break;

        case ChordQuality::Augmented:
            intervals = {0, 4, 8}; // R, M3, A5
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "aug";
            break;

        case ChordQuality::Dominant7:
            intervals = {0, 4, 7, 10}; // R, M3, P5, m7
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "7";
            break;

        case ChordQuality::Major7:
            intervals = {0, 4, 7, 11}; // R, M3, P5, M7
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "maj7";
            break;

        case ChordQuality::Minor7:
            intervals = {0, 3, 7, 10}; // R, m3, P5, m7
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "m7";
            break;

        case ChordQuality::HalfDiminished7:
            intervals = {0, 3, 6, 10}; // R, m3, d5, m7
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "m7b5";
            break;

        case ChordQuality::FullyDiminished7:
            intervals = {0, 3, 6, 9}; // R, m3, d5, d7
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "dim7";
            break;

        case ChordQuality::Sus2:
            intervals = {0, 2, 7}; // R, M2, P5
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "sus2";
            break;

        case ChordQuality::Sus4:
            intervals = {0, 5, 7}; // R, P4, P5
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "sus4";
            break;

        case ChordQuality::Add9:
            intervals = {0, 4, 7, 14}; // R, M3, P5, M9
            chord.symbol = coreTheory_->midiToNoteName(rootNote).substr(0,
                          coreTheory_->midiToNoteName(rootNote).length() - 1) + "add9";
            break;

        default:
            intervals = {0, 4, 7}; // Default to major
            break;
    }

    // Apply voicing
    switch (voicing) {
        case VoicingType::Close:
            // All notes within an octave
            for (int interval : intervals) {
                chord.notes.push_back(rootNote + interval);
            }
            break;

        case VoicingType::Open:
            // Spread across 2 octaves
            for (size_t i = 0; i < intervals.size(); ++i) {
                int note = rootNote + intervals[i];
                if (i > 0 && i < intervals.size() - 1) {
                    note += 12; // Raise middle voices
                }
                chord.notes.push_back(note);
            }
            break;

        case VoicingType::Drop2:
            // Jazz voicing: drop 2nd voice down an octave
            for (size_t i = 0; i < intervals.size(); ++i) {
                int note = rootNote + intervals[i];
                if (i == 1) { // Drop the 2nd voice
                    note -= 12;
                }
                chord.notes.push_back(note);
            }
            std::sort(chord.notes.begin(), chord.notes.end());
            break;

        case VoicingType::Drop3:
            // Jazz voicing: drop 3rd voice down an octave
            for (size_t i = 0; i < intervals.size(); ++i) {
                int note = rootNote + intervals[i];
                if (i == 2 && intervals.size() >= 3) {
                    note -= 12;
                }
                chord.notes.push_back(note);
            }
            std::sort(chord.notes.begin(), chord.notes.end());
            break;

        default:
            // Root position (same as Close)
            for (int interval : intervals) {
                chord.notes.push_back(rootNote + interval);
            }
            break;
    }

    // Calculate tension (dominant chords = high tension)
    if (quality == ChordQuality::Dominant7 || quality == ChordQuality::Diminished) {
        chord.tension = 0.8f;
    } else if (quality == ChordQuality::Major || quality == ChordQuality::Minor) {
        chord.tension = 0.2f;
    } else {
        chord.tension = 0.5f;
    }

    return chord;
}

Chord HarmonyEngine::analyzeChord(const std::vector<int>& notes) const {
    if (notes.empty()) {
        return Chord{};
    }

    Chord chord;

    // Normalize to same octave for analysis
    std::vector<int> normalizedNotes = notes;
    std::sort(normalizedNotes.begin(), normalizedNotes.end());

    // Reduce to pitch classes (0-11)
    std::vector<int> pitchClasses;
    for (int note : normalizedNotes) {
        int pc = note % 12;
        if (std::find(pitchClasses.begin(), pitchClasses.end(), pc) == pitchClasses.end()) {
            pitchClasses.push_back(pc);
        }
    }

    // Identify root (assume lowest note for now)
    chord.rootNote = normalizedNotes[0];
    int rootPC = chord.rootNote % 12;

    // Calculate intervals from root
    std::vector<int> intervals;
    for (int pc : pitchClasses) {
        int interval = (pc - rootPC + 12) % 12;
        intervals.push_back(interval);
    }
    std::sort(intervals.begin(), intervals.end());

    // Identify quality based on intervals
    chord.quality = identifyQuality(intervals);
    chord.notes = notes;

    return chord;
}

std::vector<std::string> HarmonyEngine::identifyExtensions(const std::vector<int>& notes) const {
    std::vector<std::string> extensions;

    if (notes.empty()) return extensions;

    int root = notes[0] % 12;

    for (int note : notes) {
        int interval = (note - root + 12) % 12;

        switch (interval) {
            case 2:
            case 14:
                extensions.push_back("9");
                break;
            case 5:
            case 17:
                extensions.push_back("11");
                break;
            case 9:
            case 21:
                extensions.push_back("13");
                break;
        }
    }

    return extensions;
}

int HarmonyEngine::detectInversion(const Chord& chord) const {
    if (chord.notes.empty()) return 0;

    int bassNote = chord.notes[0] % 12;
    int rootNote = chord.rootNote % 12;

    if (bassNote == rootNote) return 0; // Root position

    // Calculate interval from root to bass
    int interval = (bassNote - rootNote + 12) % 12;

    if (interval == 3 || interval == 4) return 1; // First inversion (3rd in bass)
    if (interval == 7) return 2; // Second inversion (5th in bass)
    if (interval == 10 || interval == 11) return 3; // Third inversion (7th in bass)

    return 0;
}

std::vector<Chord> HarmonyEngine::getAllVoicings(int rootNote, ChordQuality quality) const {
    std::vector<Chord> voicings;

    // Generate all voicing types
    std::vector<VoicingType> types = {
        VoicingType::Close,
        VoicingType::Open,
        VoicingType::Drop2,
        VoicingType::Drop3
    };

    for (auto type : types) {
        voicings.push_back(buildChord(rootNote, quality, type));
    }

    return voicings;
}

//==============================================================================
// Functional Harmony Analysis
//==============================================================================

HarmonyEngine::ProgressionAnalysis HarmonyEngine::analyzeProgression(
    const std::vector<Chord>& chords,
    const std::string& key) const
{
    ProgressionAnalysis analysis;

    if (chords.empty()) return analysis;

    // Analyze each chord
    for (size_t i = 0; i < chords.size(); ++i) {
        const auto& chord = chords[i];

        // Get Roman numeral
        std::string rn = getRomanNumeral(chord, key);
        analysis.romanNumerals.push_back(rn);

        // Get harmonic function
        std::string func = getHarmonicFunction(chord, key);
        analysis.functions.push_back(func);

        // Calculate tension
        float tension = calculateTension(chord, key);

        // Fill tension curve (simplified)
        int startIdx = (i * 100) / chords.size();
        int endIdx = ((i + 1) * 100) / chords.size();
        for (int j = startIdx; j < endIdx && j < 100; ++j) {
            analysis.tensionCurve[j] = tension;
        }
    }

    // Build functional analysis string (T-D-S)
    std::ostringstream funcStream;
    for (size_t i = 0; i < analysis.functions.size(); ++i) {
        if (analysis.functions[i] == "Tonic") funcStream << "T";
        else if (analysis.functions[i] == "Dominant") funcStream << "D";
        else if (analysis.functions[i] == "Subdominant") funcStream << "S";

        if (i < analysis.functions.size() - 1) funcStream << "-";
    }
    analysis.functionalAnalysis = funcStream.str();

    // Detect cadences
    if (chords.size() >= 2) {
        auto cadence = detectCadence(chords[chords.size() - 2], chords[chords.size() - 1]);
        analysis.cadences.push_back(cadence.type);
    }

    // Determine emotional arc
    float startTension = analysis.tensionCurve[0];
    float endTension = analysis.tensionCurve[99];

    if (endTension < startTension) {
        analysis.emotionalArc = "Tension → Resolution";
    } else if (endTension > startTension) {
        analysis.emotionalArc = "Building tension (unresolved)";
    } else {
        analysis.emotionalArc = "Stable throughout";
    }

    // Calculate cohesion (how well chords fit together)
    float avgTensionChange = 0.0f;
    for (size_t i = 1; i < chords.size(); ++i) {
        float t1 = calculateTension(chords[i-1], key);
        float t2 = calculateTension(chords[i], key);
        avgTensionChange += std::abs(t2 - t1);
    }
    if (chords.size() > 1) {
        avgTensionChange /= (chords.size() - 1);
    }
    analysis.cohesion = 1.0f - std::min(avgTensionChange, 1.0f);

    // Generate explanation
    std::ostringstream explain;
    explain << "Progression: ";
    for (size_t i = 0; i < analysis.romanNumerals.size(); ++i) {
        explain << analysis.romanNumerals[i];
        if (i < analysis.romanNumerals.size() - 1) explain << " - ";
    }
    explain << "\nFunctional analysis: " << analysis.functionalAnalysis;
    explain << "\nEmotional arc: " << analysis.emotionalArc;
    analysis.explanation = explain.str();

    return analysis;
}

HarmonyEngine::Cadence HarmonyEngine::detectCadence(
    const Chord& penultimate,
    const Chord& final) const
{
    Cadence cadence;

    // Simplified cadence detection based on tension
    float tension1 = penultimate.tension;
    float tension2 = final.tension;

    if (tension1 > 0.7f && tension2 < 0.3f) {
        cadence.type = "Perfect Authentic Cadence";
        cadence.romanNumerals = "V → I";
        cadence.finality = 1.0f;
        cadence.explanation = "Strong dominant to tonic motion creates complete resolution";
    } else if (tension1 < 0.4f && tension2 < 0.3f) {
        cadence.type = "Plagal Cadence";
        cadence.romanNumerals = "IV → I";
        cadence.finality = 0.8f;
        cadence.explanation = "Subdominant to tonic ('Amen' cadence)";
    } else if (tension1 > 0.6f && tension2 > 0.5f) {
        cadence.type = "Half Cadence";
        cadence.romanNumerals = "→ V";
        cadence.finality = 0.3f;
        cadence.explanation = "Ends on dominant, creating expectation";
    } else if (tension1 > 0.7f && tension2 < 0.5f && tension2 > 0.3f) {
        cadence.type = "Deceptive Cadence";
        cadence.romanNumerals = "V → vi";
        cadence.finality = 0.4f;
        cadence.explanation = "Expected resolution subverted";
    } else {
        cadence.type = "Imperfect Cadence";
        cadence.finality = 0.5f;
        cadence.explanation = "Partial resolution";
    }

    return cadence;
}

std::string HarmonyEngine::getHarmonicFunction(const Chord& chord, const std::string& key) const {
    int degree = getScaleDegree(chord.rootNote, key);
    return getFunctionFromDegree(degree);
}

std::string HarmonyEngine::getRomanNumeral(const Chord& chord, const std::string& key) const {
    int degree = getScaleDegree(chord.rootNote, key);

    // Roman numeral mapping
    const std::string numerals[] = {"I", "II", "III", "IV", "V", "VI", "VII"};

    if (degree < 1 || degree > 7) return "?";

    std::string rn = numerals[degree - 1];

    // Lowercase for minor chords
    if (chord.quality == ChordQuality::Minor ||
        chord.quality == ChordQuality::Diminished ||
        chord.quality == ChordQuality::HalfDiminished7) {
        std::transform(rn.begin(), rn.end(), rn.begin(), ::tolower);
    }

    // Add quality markers
    if (chord.quality == ChordQuality::Diminished ||
        chord.quality == ChordQuality::FullyDiminished7) {
        rn += "°";
    } else if (chord.quality == ChordQuality::HalfDiminished7) {
        rn += "ø";
    } else if (chord.quality == ChordQuality::Dominant7) {
        rn += "7";
    } else if (chord.quality == ChordQuality::Major7) {
        rn += "maj7";
    } else if (chord.quality == ChordQuality::Minor7) {
        rn += "7";
    }

    return rn;
}

float HarmonyEngine::calculateTension(const Chord& chord, const std::string& key) const {
    int degree = getScaleDegree(chord.rootNote, key);
    std::string function = getFunctionFromDegree(degree);

    float baseTension = calculateTensionFromFunction(function);

    // Modify based on chord quality
    if (chord.quality == ChordQuality::Dominant7 ||
        chord.quality == ChordQuality::Diminished) {
        baseTension += 0.2f;
    } else if (chord.quality == ChordQuality::Major ||
               chord.quality == ChordQuality::Minor) {
        baseTension -= 0.1f;
    }

    return std::clamp(baseTension, 0.0f, 1.0f);
}

//==============================================================================
// Voice Leading
//==============================================================================

VoiceLeadingSuggestion HarmonyEngine::suggestVoiceLeading(
    const Chord& fromChord,
    const Chord& toChord,
    VoiceLeadingStyle style,
    int numVoices) const
{
    VoiceLeadingSuggestion suggestion;

    // Find common tones
    auto commonTones = findCommonTones(fromChord, toChord);

    // Generate voice leading for SATB (4 voices)
    suggestion.voices.resize(numVoices);

    for (int i = 0; i < numVoices; ++i) {
        suggestion.voices[i].push_back(fromChord.notes[i % fromChord.notes.size()]);

        // Prefer common tone if available
        if (!commonTones.empty() && i < static_cast<int>(commonTones.size())) {
            suggestion.voices[i].push_back(commonTones[i]);
        } else {
            // Move to nearest chord tone
            int targetNote = toChord.notes[i % toChord.notes.size()];
            suggestion.voices[i].push_back(targetNote);
        }
    }

    // Check for errors
    auto errors = checkVoiceLeading(suggestion.voices, style);
    for (const auto& error : errors) {
        suggestion.exceptions.push_back(error.errorType + ": " + error.explanation);
    }

    // Calculate smoothness
    suggestion.smoothnessScore = calculateSmoothness(suggestion.voices);

    // Document followed rules
    suggestion.rules.push_back("Common tones retained where possible");
    suggestion.rules.push_back("Voice leading prefers stepwise motion");

    return suggestion;
}

std::vector<HarmonyEngine::VoiceLeadingError> HarmonyEngine::checkVoiceLeading(
    const std::vector<std::vector<int>>& voices,
    VoiceLeadingStyle style) const
{
    std::vector<VoiceLeadingError> errors;

    if (voices.size() < 2 || voices[0].size() < 2) {
        return errors;
    }

    // Check for parallel fifths and octaves
    for (size_t i = 0; i < voices.size() - 1; ++i) {
        for (size_t j = i + 1; j < voices.size(); ++j) {
            // Check each chord change
            for (size_t chord = 0; chord < voices[i].size() - 1; ++chord) {
                int interval1 = std::abs(voices[i][chord] - voices[j][chord]) % 12;
                int interval2 = std::abs(voices[i][chord+1] - voices[j][chord+1]) % 12;

                // Parallel fifths
                if (interval1 == 7 && interval2 == 7 &&
                    voices[i][chord] != voices[i][chord+1]) {
                    VoiceLeadingError error;
                    error.errorType = "Parallel Fifths";
                    error.voices = {static_cast<int>(i), static_cast<int>(j)};
                    error.explanation = "Two voices moving in parallel perfect fifths";
                    error.whenAcceptable = "Acceptable in rock/metal power chords";
                    error.isStrict = (style == VoiceLeadingStyle::Bach);
                    errors.push_back(error);
                }

                // Parallel octaves
                if (interval1 == 0 && interval2 == 0 &&
                    voices[i][chord] != voices[i][chord+1]) {
                    VoiceLeadingError error;
                    error.errorType = "Parallel Octaves";
                    error.voices = {static_cast<int>(i), static_cast<int>(j)};
                    error.explanation = "Two voices moving in parallel octaves";
                    error.whenAcceptable = "Common in unison doubling";
                    error.isStrict = (style == VoiceLeadingStyle::Bach);
                    errors.push_back(error);
                }
            }
        }
    }

    return errors;
}

std::vector<int> HarmonyEngine::findCommonTones(const Chord& chord1, const Chord& chord2) const {
    std::vector<int> common;

    for (int note1 : chord1.notes) {
        int pc1 = note1 % 12;
        for (int note2 : chord2.notes) {
            int pc2 = note2 % 12;
            if (pc1 == pc2) {
                common.push_back(note2);
                break;
            }
        }
    }

    return common;
}

float HarmonyEngine::calculateSmoothness(const std::vector<std::vector<int>>& voices) const {
    float totalDistance = calculateVoiceDistance(voices);
    int numTransitions = 0;

    for (const auto& voice : voices) {
        numTransitions += voice.size() - 1;
    }

    if (numTransitions == 0) return 1.0f;

    float avgDistance = totalDistance / numTransitions;

    // Smooth = small intervals (0-2 semitones = 1.0, >7 semitones = 0.0)
    return std::clamp(1.0f - (avgDistance / 7.0f), 0.0f, 1.0f);
}

//==============================================================================
// Chord Substitution
//==============================================================================

std::vector<HarmonyEngine::Substitution> HarmonyEngine::suggestSubstitutions(
    const Chord& chord,
    const std::string& key,
    SubstitutionType type) const
{
    std::vector<Substitution> substitutions;

    // Tritone substitution (for dominant chords)
    if (type == SubstitutionType::TritoneSub || type == SubstitutionType::All) {
        if (chord.quality == ChordQuality::Dominant7) {
            Substitution sub;
            sub.originalChord = chord;
            sub.substituteChord = tritoneSubstitute(chord);
            sub.type = SubstitutionType::TritoneSub;
            sub.explanation = "Tritone substitution: shares tritone with original dominant";
            sub.soundDescription = "Jazzier, more chromatic";
            sub.tensionChange = 0.1f;
            sub.examples.push_back("Stella By Starlight");
            substitutions.push_back(sub);
        }
    }

    // Diatonic substitutions
    if (type == SubstitutionType::Diatonic || type == SubstitutionType::All) {
        auto diatonicSubs = getDiatonicSubstitutes(chord, key);
        for (const auto& subChord : diatonicSubs) {
            Substitution sub;
            sub.originalChord = chord;
            sub.substituteChord = subChord;
            sub.type = SubstitutionType::Diatonic;
            sub.explanation = "Diatonic substitute with similar function";
            sub.soundDescription = "Stays in key, subtle variation";
            sub.tensionChange = 0.0f;
            substitutions.push_back(sub);
        }
    }

    // Secondary dominants
    if (type == SubstitutionType::Secondary || type == SubstitutionType::All) {
        Substitution sub;
        sub.originalChord = chord;
        sub.substituteChord = generateSecondaryDominant(chord);
        sub.type = SubstitutionType::Secondary;
        sub.explanation = "Secondary dominant: temporarily tonicizes the target chord";
        sub.soundDescription = "Adds forward motion and color";
        sub.tensionChange = 0.3f;
        sub.examples.push_back("Beatles - Something");
        substitutions.push_back(sub);
    }

    // Modal interchange
    if (type == SubstitutionType::Modal || type == SubstitutionType::All) {
        auto modalSubs = getModalInterchange(chord, key);
        for (const auto& subChord : modalSubs) {
            Substitution sub;
            sub.originalChord = chord;
            sub.substituteChord = subChord;
            sub.type = SubstitutionType::Modal;
            sub.explanation = "Borrowed from parallel mode";
            sub.soundDescription = "Darker/brighter color shift";
            sub.tensionChange = -0.2f;
            sub.examples.push_back("Beatles - Michelle (bVI chord)");
            substitutions.push_back(sub);
        }
    }

    return substitutions;
}

Chord HarmonyEngine::generateSecondaryDominant(const Chord& targetChord) const {
    // Secondary dominant is a dominant 7th built a fifth above target
    int secondaryRoot = targetChord.rootNote + 7; // Up a fifth
    return buildChord(secondaryRoot, ChordQuality::Dominant7);
}

std::vector<Chord> HarmonyEngine::getBorrowedChords(const std::string& key) const {
    std::vector<Chord> borrowed;

    // Simplified: Return a few common borrowed chords
    // bVI (flat-VI from parallel minor)
    // bVII (flat-VII from parallel minor)
    // iv (minor iv from parallel minor)

    // This would require more sophisticated key parsing
    // For now, return empty vector
    return borrowed;
}

ChordProgression HarmonyEngine::reharmonizeMelody(
    const std::vector<int>& melody,
    const std::string& key,
    const std::string& style) const
{
    ChordProgression progression;
    progression.key = key;

    // Simplified reharmonization: place chord every 2 notes
    for (size_t i = 0; i < melody.size(); i += 2) {
        int melodyNote = melody[i];

        // Build chord with melody note on top
        // For simplicity, use major/minor triads
        Chord chord = buildChord(melodyNote, ChordQuality::Major);
        progression.chords.push_back(chord);
    }

    return progression;
}

//==============================================================================
// Emotion-Driven Generation
//==============================================================================

ChordProgression HarmonyEngine::generateProgressionForEmotion(
    const std::string& emotion,
    const std::string& key,
    int numChords,
    bool allowRuleBreaking) const
{
    ChordProgression progression;
    progression.key = key;

    // Get emotion harmony characteristics
    auto emotionChars = getEmotionalHarmony(emotion);

    // Select progression pattern based on emotion
    auto patterns = selectProgressionPattern(emotion);

    // Build chords from pattern
    // Simplified: Generate based on common patterns
    std::vector<std::string> romanNumerals;

    if (emotion == "joy" || emotion == "happiness") {
        romanNumerals = {"I", "V", "vi", "IV"}; // Pop progression
    } else if (emotion == "grief" || emotion == "sadness") {
        romanNumerals = {"i", "VI", "III", "VII"}; // Minor progression
    } else if (emotion == "hope") {
        romanNumerals = {"vi", "IV", "I", "V"}; // Ascending energy
    } else {
        // Default
        romanNumerals = {"I", "IV", "V", "I"};
    }

    // Limit to requested number of chords
    if (static_cast<int>(romanNumerals.size()) > numChords) {
        romanNumerals.resize(numChords);
    }

    progression.romanNumerals = romanNumerals;

    // Generate explanation
    std::ostringstream explain;
    explain << "Progression for emotion '" << emotion << "':\n";
    explain << "Pattern: ";
    for (const auto& rn : romanNumerals) {
        explain << rn << " ";
    }
    explain << "\n" << emotionChars.explanation;
    progression.harmonyExplanation = explain.str();

    return progression;
}

HarmonyEngine::EmotionalHarmony HarmonyEngine::getEmotionalHarmony(
    const std::string& emotion) const
{
    EmotionalHarmony eh;

    // Check emotion mappings
    auto it = emotionMappings_.find(emotion);
    if (it != emotionMappings_.end()) {
        eh.preferredQualities = it->second.preferredQualities;
        eh.tensionLevel = it->second.targetTension;
        eh.preferredProgressions = it->second.progressionPatterns;
    } else {
        // Default values
        eh.preferredQualities = {ChordQuality::Major, ChordQuality::Minor};
        eh.tensionLevel = 0.5f;
        eh.preferredProgressions = {"I-IV-V-I"};
    }

    eh.explanation = "Chords selected to evoke " + emotion;

    return eh;
}

std::map<std::string, float> HarmonyEngine::analyzeEmotionalContent(
    const ChordProgression& progression) const
{
    std::map<std::string, float> emotions;

    // Analyze tension curve
    float avgTension = 0.0f;
    for (float t : progression.tensionCurve) {
        avgTension += t;
    }
    avgTension /= 100.0f;

    // Map tension to emotions
    if (avgTension < 0.3f) {
        emotions["peaceful"] = 0.8f;
        emotions["content"] = 0.7f;
    } else if (avgTension > 0.7f) {
        emotions["anxious"] = 0.8f;
        emotions["tense"] = 0.9f;
    } else {
        emotions["neutral"] = 0.5f;
    }

    return emotions;
}

//==============================================================================
// Common Progressions Database
//==============================================================================

ChordProgression HarmonyEngine::getCommonProgression(
    const std::string& name,
    const std::string& key) const
{
    ChordProgression progression;
    progression.key = key;

    // Search database
    for (const auto& tmpl : progressionDatabase_) {
        if (tmpl.name == name) {
            progression.romanNumerals = tmpl.romanNumerals;
            progression.harmonyExplanation = tmpl.emotionalEffect;
            progression.famousExamples = tmpl.examples;
            break;
        }
    }

    return progression;
}

std::vector<std::string> HarmonyEngine::listCommonProgressions() const {
    std::vector<std::string> names;

    for (const auto& tmpl : progressionDatabase_) {
        names.push_back(tmpl.name);
    }

    return names;
}

std::vector<ChordProgression> HarmonyEngine::findSimilarProgressions(
    const ChordProgression& progression,
    float similarityThreshold) const
{
    std::vector<ChordProgression> similar;

    // Compare with database (simplified comparison)
    for (const auto& tmpl : progressionDatabase_) {
        // Compare roman numerals
        int matches = 0;
        for (size_t i = 0; i < std::min(progression.romanNumerals.size(),
                                       tmpl.romanNumerals.size()); ++i) {
            if (progression.romanNumerals[i] == tmpl.romanNumerals[i]) {
                matches++;
            }
        }

        float similarity = static_cast<float>(matches) /
                          std::max(progression.romanNumerals.size(),
                                  tmpl.romanNumerals.size());

        if (similarity >= similarityThreshold) {
            ChordProgression prog;
            prog.romanNumerals = tmpl.romanNumerals;
            prog.harmonyExplanation = tmpl.name;
            similar.push_back(prog);
        }
    }

    return similar;
}

//==============================================================================
// Utilities
//==============================================================================

ChordProgression HarmonyEngine::transposeProgression(
    const ChordProgression& progression,
    int semitones) const
{
    ChordProgression transposed = progression;

    // Transpose all chords
    for (auto& chord : transposed.chords) {
        chord.rootNote += semitones;
        for (auto& note : chord.notes) {
            note += semitones;
        }
    }

    return transposed;
}

ChordProgression HarmonyEngine::transposeProgression(
    const ChordProgression& progression,
    const std::string& fromKey,
    const std::string& toKey) const
{
    // Calculate semitone difference
    // Simplified: would need key name parsing
    int semitones = 0; // Placeholder

    return transposeProgression(progression, semitones);
}

std::vector<int> HarmonyEngine::chordSymbolToMidi(
    const std::string& chordSymbol,
    int octave) const
{
    // Simplified parser
    // Parse root note
    int rootMidi = (octave * 12) + 60; // C4 = 60

    // Parse quality (would need full parser)
    ChordQuality quality = ChordQuality::Major;

    if (chordSymbol.find("m7") != std::string::npos) {
        quality = ChordQuality::Minor7;
    } else if (chordSymbol.find("m") != std::string::npos) {
        quality = ChordQuality::Minor;
    } else if (chordSymbol.find("7") != std::string::npos) {
        quality = ChordQuality::Dominant7;
    }

    auto chord = buildChord(rootMidi, quality);
    return chord.notes;
}

std::string HarmonyEngine::midiToChordSymbol(const std::vector<int>& notes) const {
    auto chord = analyzeChord(notes);
    return chord.symbol;
}

std::string HarmonyEngine::explainProgression(
    const ChordProgression& progression,
    ExplanationDepth depth) const
{
    std::ostringstream explanation;

    switch (depth) {
        case ExplanationDepth::Simple:
            explanation << "A chord progression that moves from "
                       << progression.romanNumerals.front()
                       << " to " << progression.romanNumerals.back();
            break;

        case ExplanationDepth::Intermediate:
            explanation << "Functional analysis: " << progression.functionalAnalysis
                       << "\nCreates " << progression.harmonyExplanation;
            break;

        case ExplanationDepth::Advanced:
            explanation << "Complete harmonic analysis:\n"
                       << "Roman numerals: ";
            for (const auto& rn : progression.romanNumerals) {
                explanation << rn << " ";
            }
            explanation << "\nFunctional: " << progression.functionalAnalysis
                       << "\nTension arc creates: " << progression.harmonyExplanation;
            break;

        case ExplanationDepth::Expert:
            explanation << "Voice leading analysis with tension curve data...\n"
                       << "Harmonic rhythm and cadential points...";
            break;
    }

    return explanation.str();
}

//==============================================================================
// Internal Helpers
//==============================================================================

void HarmonyEngine::initializeProgressionDatabase() {
    // Common progressions
    progressionDatabase_ = {
        {"I-V-vi-IV", {"I", "V", "vi", "IV"}, "Pop",
         {"Let It Be", "Don't Stop Believin'"}, "Anthemic, uplifting"},
        {"ii-V-I", {"ii", "V", "I"}, "Jazz",
         {"Autumn Leaves"}, "Classic jazz resolution"},
        {"I-IV-V", {"I", "IV", "V"}, "Rock",
         {"La Bamba", "Twist and Shout"}, "Simple, energetic"},
        {"12-bar blues", {"I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"},
         "Blues", {"Sweet Home Chicago"}, "Traditional blues form"},
        {"i-VI-III-VII", {"i", "VI", "III", "VII"}, "Minor Pop",
         {"Losing My Religion"}, "Melancholic descent"}
    };
}

void HarmonyEngine::initializeEmotionMappings() {
    // Joy / Happiness
    emotionMappings_["joy"] = {
        "joy",
        {4, 7, 11}, // Major thirds, fifths, major sevenths
        {ChordQuality::Major, ChordQuality::Major7, ChordQuality::Add9},
        0.3f, // Low tension
        {"I-V-vi-IV", "I-IV-V"}
    };

    // Grief / Sadness
    emotionMappings_["grief"] = {
        "grief",
        {3, 7, 10}, // Minor thirds, fifths, minor sevenths
        {ChordQuality::Minor, ChordQuality::Minor7, ChordQuality::Diminished},
        0.6f, // Moderate-high tension
        {"i-VI-III-VII", "i-iv-V"}
    };

    // Hope
    emotionMappings_["hope"] = {
        "hope",
        {4, 7}, // Major intervals
        {ChordQuality::Major, ChordQuality::Sus4},
        0.4f, // Moderate tension (building)
        {"vi-IV-I-V", "I-V-I"}
    };
}

void HarmonyEngine::initializeVoiceLeadingRules() {
    voiceLeadingRules_ = {
        {"No parallel fifths", VoiceLeadingStyle::Bach, true,
         "Parallel perfect fifths weaken independence of voices",
         "Acceptable in rock power chords"},
        {"No parallel octaves", VoiceLeadingStyle::Bach, true,
         "Parallel octaves reduce to single voice",
         "Common in unison doubling"},
        {"Resolve tendency tones", VoiceLeadingStyle::Bach, false,
         "Leading tone up, 7th down",
         "Can be suspended for effect"},
        {"Smooth voice leading preferred", VoiceLeadingStyle::Jazz, false,
         "Prefer stepwise motion",
         "Leaps okay for dramatic effect"}
    };
}

int HarmonyEngine::identifyRoot(const std::vector<int>& notes) const {
    if (notes.empty()) return 0;
    return notes[0]; // Simplified: assume lowest note is root
}

ChordQuality HarmonyEngine::identifyQuality(const std::vector<int>& intervals) const {
    if (intervals.empty()) return ChordQuality::Major;

    // Check for common patterns
    if (intervals.size() >= 3) {
        if (intervals[1] == 4 && intervals[2] == 7) {
            return ChordQuality::Major; // M3 + P5
        } else if (intervals[1] == 3 && intervals[2] == 7) {
            return ChordQuality::Minor; // m3 + P5
        } else if (intervals[1] == 3 && intervals[2] == 6) {
            return ChordQuality::Diminished; // m3 + d5
        } else if (intervals[1] == 4 && intervals[2] == 8) {
            return ChordQuality::Augmented; // M3 + A5
        }
    }

    // Check for seventh chords
    if (intervals.size() >= 4) {
        if (intervals[1] == 4 && intervals[2] == 7 && intervals[3] == 10) {
            return ChordQuality::Dominant7;
        } else if (intervals[1] == 4 && intervals[2] == 7 && intervals[3] == 11) {
            return ChordQuality::Major7;
        } else if (intervals[1] == 3 && intervals[2] == 7 && intervals[3] == 10) {
            return ChordQuality::Minor7;
        } else if (intervals[1] == 3 && intervals[2] == 6 && intervals[3] == 10) {
            return ChordQuality::HalfDiminished7;
        }
    }

    return ChordQuality::Major; // Default
}

std::vector<int> HarmonyEngine::getIntervalsFromRoot(
    const std::vector<int>& notes,
    int root) const
{
    std::vector<int> intervals;
    int rootPC = root % 12;

    for (int note : notes) {
        int notePC = note % 12;
        int interval = (notePC - rootPC + 12) % 12;
        intervals.push_back(interval);
    }

    std::sort(intervals.begin(), intervals.end());
    return intervals;
}

float HarmonyEngine::calculateVoiceDistance(
    const std::vector<std::vector<int>>& voices) const
{
    float totalDistance = 0.0f;

    for (const auto& voice : voices) {
        for (size_t i = 1; i < voice.size(); ++i) {
            totalDistance += std::abs(voice[i] - voice[i-1]);
        }
    }

    return totalDistance;
}

bool HarmonyEngine::hasParallelMotion(
    int voice1, int voice2,
    const std::vector<int>& fromChord,
    const std::vector<int>& toChord,
    int interval) const
{
    if (voice1 >= static_cast<int>(fromChord.size()) ||
        voice2 >= static_cast<int>(fromChord.size()) ||
        voice1 >= static_cast<int>(toChord.size()) ||
        voice2 >= static_cast<int>(toChord.size())) {
        return false;
    }

    int interval1 = std::abs(fromChord[voice1] - fromChord[voice2]) % 12;
    int interval2 = std::abs(toChord[voice1] - toChord[voice2]) % 12;

    return (interval1 == interval && interval2 == interval);
}

int HarmonyEngine::getScaleDegree(int midiNote, const std::string& key) const {
    // Simplified: assume C major
    // Would need proper key parsing
    int pc = midiNote % 12;

    // C major scale: C=1, D=2, E=3, F=4, G=5, A=6, B=7
    const int degrees[] = {1, -1, 2, -1, 3, 4, -1, 5, -1, 6, -1, 7};

    return degrees[pc];
}

std::string HarmonyEngine::getFunctionFromDegree(int degree) const {
    switch (degree) {
        case 1:
        case 3:
        case 6:
            return "Tonic";
        case 5:
        case 7:
            return "Dominant";
        case 2:
        case 4:
            return "Subdominant";
        default:
            return "Unknown";
    }
}

float HarmonyEngine::calculateTensionFromFunction(const std::string& function) const {
    if (function == "Tonic") return 0.2f;
    if (function == "Subdominant") return 0.5f;
    if (function == "Dominant") return 0.9f;
    return 0.5f;
}

std::vector<std::string> HarmonyEngine::selectChordQualities(
    const std::string& emotion) const
{
    auto it = emotionMappings_.find(emotion);
    if (it != emotionMappings_.end()) {
        std::vector<std::string> qualities;
        for (auto q : it->second.preferredQualities) {
            qualities.push_back("quality"); // Simplified
        }
        return qualities;
    }
    return {"Major"};
}

std::vector<std::string> HarmonyEngine::selectProgressionPattern(
    const std::string& emotion) const
{
    auto it = emotionMappings_.find(emotion);
    if (it != emotionMappings_.end()) {
        return it->second.progressionPatterns;
    }
    return {"I-IV-V-I"};
}

float HarmonyEngine::getEmotionalTensionTarget(const std::string& emotion) const {
    auto it = emotionMappings_.find(emotion);
    if (it != emotionMappings_.end()) {
        return it->second.targetTension;
    }
    return 0.5f;
}

Chord HarmonyEngine::tritoneSubstitute(const Chord& chord) const {
    // Tritone sub is 6 semitones (tritone) away
    int newRoot = chord.rootNote + 6;
    return buildChord(newRoot, ChordQuality::Dominant7);
}

std::vector<Chord> HarmonyEngine::getDiatonicSubstitutes(
    const Chord& chord,
    const std::string& key) const
{
    std::vector<Chord> substitutes;

    // Diatonic substitutes share common tones
    // I can sub for vi, vi can sub for I
    // IV can sub for ii, etc.

    // Simplified: return a few options
    int degree = getScaleDegree(chord.rootNote, key);

    if (degree == 1) {
        // I can be substituted by vi
        Chord sub = buildChord(chord.rootNote + 9, ChordQuality::Minor);
        substitutes.push_back(sub);
    }

    return substitutes;
}

std::vector<Chord> HarmonyEngine::getModalInterchange(
    const Chord& chord,
    const std::string& key) const
{
    std::vector<Chord> borrowed;

    // Borrow chords from parallel mode
    // Major borrows from minor: iv, bVI, bVII
    // Minor borrows from major: IV, VI, VII

    // Simplified implementation
    return borrowed;
}

} // namespace midikompanion::theory
