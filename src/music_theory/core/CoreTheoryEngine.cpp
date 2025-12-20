#include "CoreTheoryEngine.h"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace midikompanion::theory {

//==============================================================================
// Static Data
//==============================================================================

const std::array<std::string, 13> CoreTheoryEngine::intervalNames_ = {
    "Perfect Unison",
    "Minor Second",
    "Major Second",
    "Minor Third",
    "Major Third",
    "Perfect Fourth",
    "Tritone",
    "Perfect Fifth",
    "Minor Sixth",
    "Major Sixth",
    "Minor Seventh",
    "Major Seventh",
    "Perfect Octave"
};

const std::array<IntervalQuality, 13> CoreTheoryEngine::intervalQualities_ = {
    IntervalQuality::Perfect,  // Unison
    IntervalQuality::Minor,    // m2
    IntervalQuality::Major,    // M2
    IntervalQuality::Minor,    // m3
    IntervalQuality::Major,    // M3
    IntervalQuality::Perfect,  // P4
    IntervalQuality::Augmented,// Tritone (Aug 4th or Dim 5th)
    IntervalQuality::Perfect,  // P5
    IntervalQuality::Minor,    // m6
    IntervalQuality::Major,    // M6
    IntervalQuality::Minor,    // m7
    IntervalQuality::Major,    // M7
    IntervalQuality::Perfect   // Octave
};

//==============================================================================
// Constructor
//==============================================================================

CoreTheoryEngine::CoreTheoryEngine() {
    initializePatternDatabase();
}

//==============================================================================
// Interval Mathematics
//==============================================================================

Interval CoreTheoryEngine::calculateInterval(int semitones, TuningSystem tuning) const {
    // Reduce to simple interval (within one octave)
    int simpleInterval = semitones % 12;
    int octaveDisplacement = semitones / 12;

    Interval interval;
    interval.semitones = semitones;
    interval.simpleInterval = simpleInterval + 1; // 1-based (1-8)
    interval.octaveDisplacement = octaveDisplacement;
    interval.cents = 0.0f; // Base 12-TET, adjustable for microtonal

    // Get quality and name
    if (simpleInterval >= 0 && simpleInterval < 13) {
        interval.quality = intervalQualities_[simpleInterval];
        interval.intervalName = intervalNames_[simpleInterval];
    } else {
        interval.quality = IntervalQuality::Custom;
        interval.intervalName = "Unknown Interval";
    }

    // Calculate frequency ratio
    interval.frequencyRatio = getFrequencyRatio(semitones, tuning);

    // Calculate consonance
    interval.consonanceScore = calculateConsonance(semitones, tuning);

    // Generate explanations
    interval.acousticExplanation = generateAcousticExplanation(simpleInterval);
    interval.perceptualEffect = generatePerceptualEffect(simpleInterval);
    interval.historicalUsage = generateHistoricalUsage(simpleInterval);

    // Position in harmonic series (approximation)
    interval.harmonicSeriesPosition = calculateHarmonicPosition(simpleInterval);

    return interval;
}

Interval CoreTheoryEngine::calculateInterval(int midiNote1, int midiNote2,
                                             TuningSystem tuning) const {
    int semitones = std::abs(midiNote2 - midiNote1);
    return calculateInterval(semitones, tuning);
}

float CoreTheoryEngine::getFrequencyRatio(int semitones, TuningSystem tuning) const {
    switch (tuning) {
        case TuningSystem::TwelveTET:
            return std::pow(TWELVE_TET_RATIO, semitones);

        case TuningSystem::JustIntonation:
            // Just intonation uses pure frequency ratios
            switch (semitones % 12) {
                case 0:  return 1.0f;        // Unison (1:1)
                case 1:  return 16.0f/15.0f; // Minor second
                case 2:  return 9.0f/8.0f;   // Major second
                case 3:  return 6.0f/5.0f;   // Minor third
                case 4:  return 5.0f/4.0f;   // Major third
                case 5:  return 4.0f/3.0f;   // Perfect fourth
                case 6:  return 45.0f/32.0f; // Tritone
                case 7:  return 3.0f/2.0f;   // Perfect fifth (!)
                case 8:  return 8.0f/5.0f;   // Minor sixth
                case 9:  return 5.0f/3.0f;   // Major sixth
                case 10: return 9.0f/5.0f;   // Minor seventh
                case 11: return 15.0f/8.0f;  // Major seventh
                default: return 2.0f;        // Octave
            }

        case TuningSystem::Pythagorean:
            // Pythagorean: stacked perfect fifths (3:2)
            // Simplified approximation
            return std::pow(3.0f/2.0f, (semitones * 7) % 12) / std::pow(2.0f, (semitones * 7) / 12);

        default:
            return std::pow(TWELVE_TET_RATIO, semitones);
    }
}

float CoreTheoryEngine::calculateConsonance(int semitones, TuningSystem tuning) const {
    // Consonance based on:
    // 1. Frequency ratio simplicity
    // 2. Harmonic overlap
    // 3. Psychoacoustic roughness

    int simple = semitones % 12;

    // Perfect consonances: unison, P5, P4, octave
    if (simple == 0 || simple == 5 || simple == 7 || simple == 12) {
        return 1.0f;
    }

    // Imperfect consonances: M3, m3, M6, m6
    if (simple == 3 || simple == 4 || simple == 8 || simple == 9) {
        return 0.7f;
    }

    // Dissonances: m2, M2, tritone, m7, M7
    if (simple == 1 || simple == 2 || simple == 6 || simple == 10 || simple == 11) {
        return 0.2f;
    }

    return 0.5f; // Default
}

std::string CoreTheoryEngine::getIntervalName(int semitones) const {
    int simple = semitones % 12;
    if (simple >= 0 && simple < 13) {
        return intervalNames_[simple];
    }
    return "Unknown Interval";
}

IntervalQuality CoreTheoryEngine::getIntervalQuality(int semitones) const {
    int simple = semitones % 12;
    if (simple >= 0 && simple < 13) {
        return intervalQualities_[simple];
    }
    return IntervalQuality::Custom;
}

std::string CoreTheoryEngine::explainInterval(int semitones, ExplanationDepth depth) const {
    int simple = semitones % 12;
    std::ostringstream explanation;

    switch (depth) {
        case ExplanationDepth::Simple:
            explanation << "A " << getIntervalName(simple) << " is the distance of "
                       << semitones << " semitones (half steps).";
            break;

        case ExplanationDepth::Intermediate:
            explanation << "A " << getIntervalName(simple) << " spans "
                       << semitones << " semitones.\n\n"
                       << "Consonance: " << (calculateConsonance(simple, TuningSystem::TwelveTET) * 100) << "%\n"
                       << "Perceptual effect: " << generatePerceptualEffect(simple);
            break;

        case ExplanationDepth::Advanced:
            explanation << "A " << getIntervalName(simple) << " (Perfect Fifth)\n\n"
                       << "Acoustic properties:\n"
                       << "- Frequency ratio: " << getFrequencyRatio(simple, TuningSystem::JustIntonation)
                       << " (Just) vs " << getFrequencyRatio(simple, TuningSystem::TwelveTET) << " (12-TET)\n"
                       << "- Waveforms align every 3 cycles (top note) and 2 cycles (bottom note)\n"
                       << "- This simple ratio creates consonance\n\n"
                       << generateAcousticExplanation(simple);
            break;

        case ExplanationDepth::Expert:
            explanation << "Perfect Fifth - Deep Analysis\n\n"
                       << "Mathematics:\n"
                       << "- 12-TET: 2^(7/12) = " << getFrequencyRatio(7, TuningSystem::TwelveTET) << "\n"
                       << "- Just: 3:2 = 1.500 (pure)\n"
                       << "- Cents difference: " << (1200.0 * std::log2(1.5 / getFrequencyRatio(7, TuningSystem::TwelveTET))) << " cents\n\n"
                       << "Harmonic series position: 3rd harmonic\n"
                       << "Psychoacoustic roughness: Minimal (high consonance)\n\n"
                       << generateHistoricalUsage(simple);
            break;
    }

    return explanation.str();
}

//==============================================================================
// Scale Generation
//==============================================================================

Scale CoreTheoryEngine::generateScale(int rootNote, const std::vector<int>& pattern,
                                      TuningSystem tuning) const {
    Scale scale;
    scale.rootNote = rootNote;
    scale.type = ScaleType::Custom;

    // Generate scale degrees
    int currentNote = 0;
    scale.degrees.push_back(0); // Root

    for (int interval : pattern) {
        currentNote += interval;
        scale.degrees.push_back(currentNote % 12);
    }

    // Generate name (if recognizable pattern)
    scale.name = midiToNoteName(rootNote) + " " + identifyScalePattern(pattern);

    return scale;
}

Scale CoreTheoryEngine::generateMajorScale(int rootNote) const {
    return generateScale(rootNote, {2, 2, 1, 2, 2, 2, 1});
}

Scale CoreTheoryEngine::generateNaturalMinorScale(int rootNote) const {
    return generateScale(rootNote, {2, 1, 2, 2, 1, 2, 2});
}

Scale CoreTheoryEngine::generateHarmonicMinorScale(int rootNote) const {
    return generateScale(rootNote, {2, 1, 2, 2, 1, 3, 1});
}

Scale CoreTheoryEngine::generateMelodicMinorScale(int rootNote) const {
    // Ascending melodic minor
    return generateScale(rootNote, {2, 1, 2, 2, 2, 2, 1});
}

Scale CoreTheoryEngine::generateMode(int rootNote, int modeNumber) const {
    // Modes of the major scale
    std::vector<std::vector<int>> modes = {
        {2, 2, 1, 2, 2, 2, 1}, // Ionian (Major)
        {2, 1, 2, 2, 2, 1, 2}, // Dorian
        {1, 2, 2, 2, 1, 2, 2}, // Phrygian
        {2, 2, 2, 1, 2, 2, 1}, // Lydian
        {2, 2, 1, 2, 2, 1, 2}, // Mixolydian
        {2, 1, 2, 2, 1, 2, 2}, // Aeolian (Natural Minor)
        {1, 2, 2, 1, 2, 2, 2}  // Locrian
    };

    if (modeNumber >= 1 && modeNumber <= 7) {
        return generateScale(rootNote, modes[modeNumber - 1]);
    }

    return generateMajorScale(rootNote);
}

Scale CoreTheoryEngine::generatePentatonicScale(int rootNote, bool major) const {
    if (major) {
        return generateScale(rootNote, {2, 2, 3, 2, 3}); // Major pentatonic
    } else {
        return generateScale(rootNote, {3, 2, 2, 3, 2}); // Minor pentatonic
    }
}

Scale CoreTheoryEngine::generateBluesScale(int rootNote) const {
    return generateScale(rootNote, {3, 2, 1, 1, 3, 2}); // Blues scale with b5
}

std::vector<int> CoreTheoryEngine::getScaleNotes(const Scale& scale, int octaveRange) const {
    std::vector<int> notes;

    for (int octave = 0; octave < octaveRange; ++octave) {
        for (int degree : scale.degrees) {
            int midiNote = scale.rootNote + degree + (octave * 12);
            if (midiNote >= 0 && midiNote <= 127) {
                notes.push_back(midiNote);
            }
        }
    }

    return notes;
}

bool CoreTheoryEngine::isNoteInScale(int midiNote, const Scale& scale) const {
    int pitchClass = midiNote % 12;
    int rootPitchClass = scale.rootNote % 12;

    for (int degree : scale.degrees) {
        if ((rootPitchClass + degree) % 12 == pitchClass) {
            return true;
        }
    }

    return false;
}

//==============================================================================
// Circle of Fifths
//==============================================================================

CoreTheoryEngine::CircleOfFifths CoreTheoryEngine::getCircleOfFifths() const {
    if (!circleInitialized_) {
        initializeCircleOfFifths();
    }
    return circleOfFifths_;
}

void CoreTheoryEngine::initializeCircleOfFifths() const {
    circleOfFifths_.majorKeys = {
        "C", "G", "D", "A", "E", "B", "F#/Gb", "Db", "Ab", "Eb", "Bb", "F"
    };

    circleOfFifths_.minorKeys = {
        "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m/Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm"
    };

    circleOfFifths_.sharpsOrFlats = {
        0,   // C (no sharps/flats)
        1,   // G (1 sharp)
        2,   // D (2 sharps)
        3,   // A (3 sharps)
        4,   // E (4 sharps)
        5,   // B (5 sharps)
        6,   // F#/Gb (6 sharps/6 flats - enharmonic)
        -5,  // Db (5 flats)
        -4,  // Ab (4 flats)
        -3,  // Eb (3 flats)
        -2,  // Bb (2 flats)
        -1   // F (1 flat)
    };

    circleInitialized_ = true;
}

int CoreTheoryEngine::modulationDistance(const std::string& fromKey,
                                        const std::string& toKey) const {
    auto circle = getCircleOfFifths();

    int fromPos = -1, toPos = -1;

    for (int i = 0; i < 12; ++i) {
        if (circle.majorKeys[i] == fromKey || circle.minorKeys[i] == fromKey) {
            fromPos = i;
        }
        if (circle.majorKeys[i] == toKey || circle.minorKeys[i] == toKey) {
            toPos = i;
        }
    }

    if (fromPos == -1 || toPos == -1) {
        return -1; // Invalid key
    }

    int distance = std::abs(toPos - fromPos);
    return std::min(distance, 12 - distance); // Shortest path around circle
}

std::vector<std::string> CoreTheoryEngine::suggestModulations(const std::string& currentKey,
                                                             int maxDistance) const {
    auto circle = getCircleOfFifths();
    std::vector<std::string> suggestions;

    for (int i = 0; i < 12; ++i) {
        if (modulationDistance(currentKey, circle.majorKeys[i]) <= maxDistance) {
            suggestions.push_back(circle.majorKeys[i]);
        }
        if (modulationDistance(currentKey, circle.minorKeys[i]) <= maxDistance) {
            suggestions.push_back(circle.minorKeys[i]);
        }
    }

    // Remove current key from suggestions
    suggestions.erase(
        std::remove(suggestions.begin(), suggestions.end(), currentKey),
        suggestions.end()
    );

    return suggestions;
}

//==============================================================================
// Frequency Calculations
//==============================================================================

double CoreTheoryEngine::getFrequency(int midiNote, TuningSystem tuning, double concertA) const {
    switch (tuning) {
        case TuningSystem::TwelveTET:
            return calculateFrequency12TET(midiNote, concertA);
        case TuningSystem::JustIntonation:
            return calculateFrequencyJust(midiNote, concertA);
        case TuningSystem::Pythagorean:
            return calculateFrequencyPythagorean(midiNote, concertA);
        default:
            return calculateFrequency12TET(midiNote, concertA);
    }
}

double CoreTheoryEngine::calculateFrequency12TET(int midiNote, double concertA) const {
    // A4 = MIDI note 69
    return concertA * std::pow(TWELVE_TET_RATIO, midiNote - 69);
}

double CoreTheoryEngine::calculateFrequencyJust(int midiNote, double concertA) const {
    // Simplified Just Intonation (C major key)
    // This is a complex topic - full implementation would require key context
    return calculateFrequency12TET(midiNote, concertA); // Fallback for now
}

double CoreTheoryEngine::calculateFrequencyPythagorean(int midiNote, double concertA) const {
    // Pythagorean tuning based on stacked perfect fifths
    return calculateFrequency12TET(midiNote, concertA); // Fallback for now
}

//==============================================================================
// Utilities
//==============================================================================

std::string CoreTheoryEngine::midiToNoteName(int midiNote, bool preferSharps) const {
    const std::array<std::string, 12> noteNamesSharps = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };
    const std::array<std::string, 12> noteNamesFlats = {
        "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"
    };

    int pitchClass = midiNote % 12;
    int octave = (midiNote / 12) - 1; // MIDI octave system

    const auto& noteNames = preferSharps ? noteNamesSharps : noteNamesFlats;
    return noteNames[pitchClass] + std::to_string(octave);
}

int CoreTheoryEngine::noteNameToMidi(const std::string& noteName) const {
    // Parse note name (e.g., "C4", "F#5", "Bb3")
    auto parsed = parseNoteName(noteName);
    if (!parsed) {
        return -1; // Invalid
    }

    return (parsed->octave + 1) * 12 + parsed->pitchClass;
}

//==============================================================================
// Private Helpers
//==============================================================================

std::string CoreTheoryEngine::generateAcousticExplanation(int semitones) const {
    // Simplified explanations for common intervals
    switch (semitones) {
        case 0: return "Unison - same frequency, perfect alignment.";
        case 7: return "Perfect Fifth - 3:2 ratio. Waveforms align every 3:2 cycles.";
        case 5: return "Perfect Fourth - 4:3 ratio. Strong consonance.";
        case 4: return "Major Third - 5:4 ratio. Imperfect but pleasant consonance.";
        case 3: return "Minor Third - 6:5 ratio. Darker color than major third.";
        default: return "Frequency ratio: " + std::to_string(getFrequencyRatio(semitones, TuningSystem::JustIntonation));
    }
}

std::string CoreTheoryEngine::generatePerceptualEffect(int semitones) const {
    switch (semitones) {
        case 0: return "Identity, unity";
        case 1: return "Harsh, dissonant, tense";
        case 2: return "Whole step, gentle dissonance";
        case 3: return "Minor third, melancholic";
        case 4: return "Major third, bright, happy";
        case 5: return "Perfect fourth, stable, open";
        case 6: return "Tritone, unstable, needs resolution";
        case 7: return "Perfect fifth, powerful, hollow";
        case 8: return "Minor sixth, bittersweet";
        case 9: return "Major sixth, warm, expansive";
        case 10: return "Minor seventh, bluesy, unresolved";
        case 11: return "Major seventh, jazzy, sophisticated";
        case 12: return "Octave, completion, identity";
        default: return "Complex interval";
    }
}

std::string CoreTheoryEngine::generateHistoricalUsage(int semitones) const {
    switch (semitones) {
        case 7: return "Medieval organum (parallel fifths). Power chords in rock. Dominant-tonic in classical harmony.";
        case 4: return "Renaissance sweet sound. Beatles melodies. Pop music foundation.";
        case 6: return "Diabolus in musica (devil in music) in medieval times. Jazz tritone substitution. Wagner's instability.";
        default: return "Used across all eras of Western music.";
    }
}

float CoreTheoryEngine::calculateHarmonicPosition(int semitones) const {
    // Approximate position in harmonic series
    switch (semitones % 12) {
        case 0:  return 1.0f;  // Fundamental
        case 12: return 2.0f;  // 2nd harmonic (octave)
        case 7:  return 3.0f;  // 3rd harmonic (P5 + octave)
        case 4:  return 5.0f;  // 5th harmonic (M3 + 2 octaves)
        default: return 0.0f;  // Not directly in series
    }
}

std::string CoreTheoryEngine::identifyScalePattern(const std::vector<int>& pattern) const {
    if (pattern == std::vector<int>{2, 2, 1, 2, 2, 2, 1}) return "Major (Ionian)";
    if (pattern == std::vector<int>{2, 1, 2, 2, 1, 2, 2}) return "Natural Minor (Aeolian)";
    if (pattern == std::vector<int>{2, 1, 2, 2, 2, 1, 2}) return "Dorian";
    if (pattern == std::vector<int>{1, 2, 2, 2, 1, 2, 2}) return "Phrygian";
    if (pattern == std::vector<int>{2, 2, 2, 1, 2, 2, 1}) return "Lydian";
    if (pattern == std::vector<int>{2, 2, 1, 2, 2, 1, 2}) return "Mixolydian";
    if (pattern == std::vector<int>{1, 2, 2, 1, 2, 2, 2}) return "Locrian";
    if (pattern == std::vector<int>{2, 2, 3, 2, 3}) return "Major Pentatonic";
    if (pattern == std::vector<int>{3, 2, 2, 3, 2}) return "Minor Pentatonic";
    if (pattern == std::vector<int>{3, 2, 1, 1, 3, 2}) return "Blues";
    return "Custom";
}

std::optional<CoreTheoryEngine::ParsedNote> CoreTheoryEngine::parseNoteName(const std::string& noteName) const {
    if (noteName.length() < 2) {
        return std::nullopt;
    }

    ParsedNote parsed;

    // Parse pitch class
    char noteChar = noteName[0];
    int pitchClass = -1;

    switch (noteChar) {
        case 'C': pitchClass = 0; break;
        case 'D': pitchClass = 2; break;
        case 'E': pitchClass = 4; break;
        case 'F': pitchClass = 5; break;
        case 'G': pitchClass = 7; break;
        case 'A': pitchClass = 9; break;
        case 'B': pitchClass = 11; break;
        default: return std::nullopt;
    }

    // Parse accidental
    size_t octavePos = 1;
    if (noteName.length() > 2) {
        if (noteName[1] == '#') {
            pitchClass = (pitchClass + 1) % 12;
            parsed.sharp = true;
            octavePos = 2;
        } else if (noteName[1] == 'b') {
            pitchClass = (pitchClass + 11) % 12; // -1 mod 12
            parsed.sharp = false;
            octavePos = 2;
        }
    }

    // Parse octave
    try {
        parsed.octave = std::stoi(noteName.substr(octavePos));
    } catch (...) {
        return std::nullopt;
    }

    parsed.pitchClass = pitchClass;
    return parsed;
}

void CoreTheoryEngine::initializePatternDatabase() {
    // Common melodic patterns
    patternDatabase_ = {
        {"{2,2,-1,4}", "Beethoven's 5th motif", {"Symphony No. 5"}},
        {"{5,4,5}", "Star Wars leap", {"Main Theme"}},
        {"{-3,-3,-3,-12}", "Fate knocking", {"Beethoven Symphony 5"}}
    };
}

} // namespace midikompanion::theory
