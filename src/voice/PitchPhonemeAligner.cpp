#include "voice/PitchPhonemeAligner.h"
#include "voice/VoiceSynthesizer.h"  // Include here to break circular dependency
#include "common/Types.h"
#include <algorithm>
#include <cmath>

namespace kelly {

PitchPhonemeAligner::PitchPhonemeAligner()
    : bpm_(120.0f)
    , allowMelisma_(true)
    , portamentoTime_(0.05)
{
}

PitchPhonemeAligner::AlignmentResult PitchPhonemeAligner::alignLyricsToMelody(
    const LyricStructure& lyrics,
    const std::vector<VoiceSynthesizer::VocalNote>& vocalNotes,
    const GeneratedMidi* midiContext)
{
    AlignmentResult result;

    if (lyrics.sections.empty() || vocalNotes.empty()) {
        return result;
    }

    // Extract all phonemes from lyrics
    std::vector<Phoneme> allPhonemes;
    std::vector<std::vector<Phoneme>> linePhonemes;

    for (const auto& section : lyrics.sections) {
        for (const auto& line : section.lines) {
            std::vector<Phoneme> phonemes = lineToPhonemes(line);
            linePhonemes.push_back(phonemes);
            allPhonemes.insert(allPhonemes.end(), phonemes.begin(), phonemes.end());
        }
    }

    if (allPhonemes.empty()) {
        return result;
    }

    // Align phonemes to vocal notes
    // Simple alignment: distribute phonemes across available notes
    size_t phonemeIndex = 0;
    double currentBeat = 0.0;

    result.vocalNotes = vocalNotes;

    for (size_t i = 0; i < vocalNotes.size() && phonemeIndex < allPhonemes.size(); ++i) {
        const auto& note = vocalNotes[i];

        // Determine how many phonemes this note should contain
        // For now, use a simple distribution based on note duration
        int phonemesPerNote = 1;
        if (allowMelisma_ && note.duration > 0.5) {
            // Longer notes can have multiple phonemes
            phonemesPerNote = static_cast<int>(std::ceil(note.duration / 0.25)); // ~1 phoneme per 0.25 beats
        }

        // Collect phonemes for this note
        std::vector<Phoneme> notePhonemes;
        bool isStartOfSyllable = (phonemeIndex == 0);

        for (int p = 0; p < phonemesPerNote && phonemeIndex < allPhonemes.size(); ++p) {
            notePhonemes.push_back(allPhonemes[phonemeIndex]);
            phonemeIndex++;
        }

        bool isEndOfSyllable = (phonemeIndex >= allPhonemes.size() ||
                                 phonemeIndex % 3 == 0); // Rough syllable boundary

        // Align phonemes to this note
        std::vector<AlignedPhoneme> aligned = alignPhonemesToNote(notePhonemes, note, note.startBeat);

        // Mark syllable boundaries
        if (!aligned.empty()) {
            aligned.front().isStartOfSyllable = isStartOfSyllable;
            aligned.back().isEndOfSyllable = isEndOfSyllable;
        }

        result.alignedPhonemes.insert(result.alignedPhonemes.end(), aligned.begin(), aligned.end());

        // Update vocal note with lyric
        if (!notePhonemes.empty()) {
            // Get text from first phoneme's syllable (simplified)
            result.vocalNotes[i].lyric = ""; // Will be set from lyrics structure
        }
    }

    return result;
}

std::vector<PitchPhonemeAligner::AlignedPhoneme> PitchPhonemeAligner::alignPhonemesToNote(
    const std::vector<Phoneme>& phonemes,
    const VoiceSynthesizer::VocalNote& note,
    double beatPosition)
{
    std::vector<AlignedPhoneme> aligned;

    if (phonemes.empty()) {
        return aligned;
    }

    // Calculate durations for each phoneme
    std::vector<double> durations = calculatePhonemeDurations(note.duration, phonemes);

    double currentBeat = beatPosition;

    for (size_t i = 0; i < phonemes.size(); ++i) {
        AlignedPhoneme alignedPhoneme;
        alignedPhoneme.phoneme = phonemes[i];
        alignedPhoneme.midiPitch = note.pitch;
        alignedPhoneme.startBeat = currentBeat;
        alignedPhoneme.duration = durations[i];
        alignedPhoneme.isStartOfSyllable = (i == 0);
        alignedPhoneme.isEndOfSyllable = (i == phonemes.size() - 1);

        aligned.push_back(alignedPhoneme);
        currentBeat += durations[i];

        // Add portamento time between phonemes (except for last one)
        if (i < phonemes.size() - 1 && portamentoTime_ > 0.0) {
            currentBeat += portamentoTime_;
        }
    }

    return aligned;
}

std::vector<double> PitchPhonemeAligner::calculatePhonemeDurations(
    double noteDuration,
    const std::vector<Phoneme>& phonemes)
{
    std::vector<double> durations;

    if (phonemes.empty()) {
        return durations;
    }

    if (phonemes.size() == 1) {
        durations.push_back(noteDuration);
        return durations;
    }

    // Calculate base durations (weighted by phoneme type)
    double totalWeight = 0.0;
    std::vector<double> weights;

    for (const auto& phoneme : phonemes) {
        double weight = calculatePhonemeDuration(phoneme, 1.0);
        weights.push_back(weight);
        totalWeight += weight;
    }

    // Distribute duration proportionally
    double remainingDuration = noteDuration;

    for (size_t i = 0; i < phonemes.size(); ++i) {
        double duration;
        if (i == phonemes.size() - 1) {
            // Last phoneme gets remaining time
            duration = remainingDuration;
        } else {
            duration = noteDuration * (weights[i] / totalWeight);
            remainingDuration -= duration;
        }
        durations.push_back(duration);
    }

    return durations;
}

std::vector<Phoneme> PitchPhonemeAligner::lineToPhonemes(const LyricLine& line) {
    std::vector<Phoneme> phonemes;

    // Convert line text to phonemes
    std::vector<Phoneme> textPhonemes = phonemeConverter_.textToPhonemes(line.text);

    // If syllables are already populated, use those phonemes
    if (!line.syllables.empty()) {
        for (const auto& syllable : line.syllables) {
            phonemes.insert(phonemes.end(), syllable.phonemes.begin(), syllable.phonemes.end());
        }
    } else {
        // Use phonemes from text conversion
        phonemes = textPhonemes;
    }

    return phonemes;
}

double PitchPhonemeAligner::calculatePhonemeDuration(const Phoneme& phoneme, double baseDuration) const {
    // Vowels typically last longer than consonants
    // Diphthongs are longer than simple vowels

    double multiplier = 1.0;

    if (phoneme.type == "vowel") {
        multiplier = 1.5;  // Vowels are longer
    } else if (phoneme.type == "diphthong") {
        multiplier = 2.0;  // Diphthongs are longest
    } else if (phoneme.type == "consonant") {
        multiplier = 0.8;  // Consonants are shorter
    }

    // Use phoneme's default duration as a guide (normalized)
    double normalizedDuration = phoneme.duration_ms / 150.0; // Normalize to ~150ms

    return baseDuration * multiplier * normalizedDuration;
}

} // namespace kelly
