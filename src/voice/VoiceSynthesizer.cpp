#include "voice/VoiceSynthesizer.h"
#include "voice/PitchPhonemeAligner.h"  // Needed for aligner_ member access
#include "common/Types.h"
#include "common/MusicConstants.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace kelly {
using namespace MusicConstants;

VoiceSynthesizer::VoiceSynthesizer()
    : enabled_(false)
    , sampleRate_(44100.0)
    , bpm_(120.0f)
    , vocoder_(std::make_unique<VocoderEngine>())
    , envelope_(std::make_unique<ADSREnvelope>())
    , portamento_(std::make_unique<PortamentoGenerator>())
    , aligner_(std::make_unique<PitchPhonemeAligner>())
    , expressionEngine_(std::make_unique<ExpressionEngine>())
{
    vocoder_->prepare(sampleRate_);
    envelope_->prepare(sampleRate_);
    portamento_->prepare(sampleRate_);
    aligner_->setBPM(bpm_);
}

void VoiceSynthesizer::prepare(double sampleRate) {
    sampleRate_ = sampleRate;
    vocoder_->prepare(sampleRate);
    envelope_->prepare(sampleRate);
    portamento_->prepare(sampleRate);
    aligner_->setBPM(bpm_);
}

std::vector<VoiceSynthesizer::VocalNote> VoiceSynthesizer::generateVocalMelody(
    const EmotionNode& emotion,
    const GeneratedMidi& midiContext,
    const LyricStructure* lyrics)
{
    if (!enabled_) return {};

    std::vector<VocalNote> vocalNotes;

    // Generate melody contour based on emotion
    int numNotes = static_cast<int>(midiContext.lengthInBeats / BEATS_PER_EIGHTH_NOTE);  // ~2 notes per beat
    auto contour = generateMelodyContour(emotion, numNotes);

    // Map contour to actual pitches in a singable range (adjusted for voice type)
    int basePitch = static_cast<int>(voiceParams_.pitchRangeMin +
                                    (voiceParams_.pitchRangeMax - voiceParams_.pitchRangeMin) * 0.5f);
    if (emotion.valence < VALENCE_NEGATIVE) {
        basePitch = static_cast<int>(voiceParams_.pitchRangeMin +
                                    (voiceParams_.pitchRangeMax - voiceParams_.pitchRangeMin) * 0.3f);
    } else if (emotion.valence > VALENCE_POSITIVE) {
        basePitch = static_cast<int>(voiceParams_.pitchRangeMin +
                                    (voiceParams_.pitchRangeMax - voiceParams_.pitchRangeMin) * 0.7f);
    }

    double beatPosition = 0.0;
    double noteDuration = BEATS_PER_EIGHTH_NOTE;  // Half-beat notes

    for (size_t i = 0; i < contour.size() && beatPosition < midiContext.lengthInBeats; ++i) {
        VocalNote note;
        note.pitch = basePitch + contour[i];
        note.startBeat = beatPosition;
        note.duration = noteDuration;
        note.vibrato = emotion.intensity * 0.3f;  // More vibrato for intense emotions

        // Clamp to singable range for voice type
        note.pitch = std::clamp(note.pitch,
                                static_cast<int>(voiceParams_.pitchRangeMin),
                                static_cast<int>(voiceParams_.pitchRangeMax));

        // Apply expression based on emotion
        VocalExpression baseExpression;
        note.expression = expressionEngine_->applyEmotionExpression(baseExpression, emotion);
        note.expression.vibratoDepth = note.vibrato;

        vocalNotes.push_back(note);
        beatPosition += noteDuration;
    }

    // If lyrics are provided, align them to the melody
    if (lyrics) {
        aligner_->setBPM(bpm_);
        auto alignmentResult = aligner_->alignLyricsToMelody(*lyrics, vocalNotes, &midiContext);

        // Update vocal notes with lyrics from alignment
        for (size_t i = 0; i < vocalNotes.size() && i < alignmentResult.vocalNotes.size(); ++i) {
            if (!alignmentResult.vocalNotes[i].lyric.empty()) {
                vocalNotes[i].lyric = alignmentResult.vocalNotes[i].lyric;
            }
        }
    }

    return vocalNotes;
}

std::vector<int> VoiceSynthesizer::generateMelodyContour(const EmotionNode& emotion, int numNotes) {
    std::vector<int> contour;
    contour.reserve(numNotes);

    // Simple contour generation based on emotion
    // Negative emotions: descending patterns
    // Positive emotions: ascending patterns
    // High arousal: more variation

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> intervalDist(-3, 3);

    int currentOffset = 0;
    for (int i = 0; i < numNotes; ++i) {
        // Bias based on valence
        int bias = emotion.valence > 0 ? 1 : -1;
        int interval = intervalDist(rng) + bias;

        // High arousal = more variation
        if (emotion.arousal > 0.7f) {
            interval += (intervalDist(rng) % 2);
        }

        currentOffset += interval;
        currentOffset = std::clamp(currentOffset, -12, 12);  // Keep within octave
        contour.push_back(currentOffset);
    }

    return contour;
}

std::vector<std::string> VoiceSynthesizer::generateLyrics(
    const EmotionNode& emotion,
    const Wound& wound)
{
    if (!enabled_) return {};

    std::vector<std::string> lyrics;

    // Simple lyric generation based on emotion keywords
    // In a full implementation, this would use NLP or template-based generation

    if (emotion.name.find("sad") != std::string::npos ||
        emotion.name.find("lonely") != std::string::npos) {
        lyrics.push_back("Alone in the silence");
        lyrics.push_back("Echoes of what was");
        lyrics.push_back("Searching for meaning");
    } else if (emotion.name.find("joy") != std::string::npos ||
               emotion.name.find("happy") != std::string::npos) {
        lyrics.push_back("Light fills the spaces");
        lyrics.push_back("Moments of wonder");
        lyrics.push_back("Dancing in freedom");
    } else if (emotion.name.find("anger") != std::string::npos ||
               emotion.name.find("rage") != std::string::npos) {
        lyrics.push_back("Fire in the darkness");
        lyrics.push_back("Breaking the silence");
        lyrics.push_back("Voice of the wounded");
    } else {
        // Generic emotional expression
        lyrics.push_back("Feel what I feel");
        lyrics.push_back("Words cannot capture");
        lyrics.push_back("Music tells the truth");
    }

    return lyrics;
}

std::vector<float> VoiceSynthesizer::synthesizeAudio(
    const std::vector<VocalNote>& notes,
    double sampleRate,
    const EmotionNode* emotion)
{
    if (!enabled_ || notes.empty()) return {};

    // Update sample rate if changed
    if (sampleRate != sampleRate_) {
        prepare(sampleRate);
    }

    // Calculate total duration in beats
    double totalDuration = 0.0;
    for (const auto& note : notes) {
        totalDuration = std::max(totalDuration, note.startBeat + note.duration);
    }

    // Convert beats to samples
    int64_t totalSamples = beatsToSamples(totalDuration);
    std::vector<float> output(totalSamples, 0.0f);

    // Synthesize each note
    for (const auto& note : notes) {
        int64_t startSample = beatsToSamples(note.startBeat);
        int64_t durationSamples = beatsToSamples(note.duration);
        int64_t endSample = startSample + durationSamples;

        // Clamp to buffer bounds
        if (startSample < 0 || startSample >= totalSamples) continue;
        if (endSample > totalSamples) endSample = totalSamples;

        // Get vocal characteristics (use expression from note if available, otherwise emotion-based)
        VocalCharacteristics chars;
        // Check if expression has been set (vibratoRate > 0 indicates expression was applied)
        if (note.expression.vibratoRate > 0.0f) {
            // Use expression from note
            chars.brightness = note.expression.brightness;
            chars.breathiness = note.expression.breathiness;
            chars.vibratoRate = note.expression.vibratoRate;
            chars.vibratoDepth = note.expression.vibratoDepth;
        } else if (emotion) {
            chars = getVocalCharacteristics(*emotion);
            // Override with note's specific vibrato setting
            chars.vibratoDepth = note.vibrato > 0.0f ? note.vibrato : chars.vibratoDepth;
        } else {
            // Default characteristics
            chars.brightness = 0.5f;
            chars.breathiness = 0.2f;
            chars.vibratoRate = 5.0f;
            chars.vibratoDepth = note.vibrato;
        }

        // Get formant data for this pitch (use emotion if available)
        VowelFormantDatabase::Vowel targetVowel = selectVowel(note.pitch, emotion);
        auto formantData = VowelFormantDatabase::getFormants(targetVowel);

        // Apply voice type formant shifts
        for (size_t i = 0; i < 4; ++i) {
            float shift = 1.0f;
            if (i == 0) shift = voiceParams_.formantShiftF1;
            else if (i == 1) shift = voiceParams_.formantShiftF2;
            else if (i == 2) shift = voiceParams_.formantShiftF3;
            else shift = voiceParams_.formantShift;

            formantData.frequencies[i] *= shift;
        }

        // Also set formant shift in vocoder
        vocoder_->setFormantShift(voiceParams_.formantShift);

        // Smooth formant transition if vowel changed
        if (targetVowel != currentVowel_) {
            // Calculate transition time based on note duration (5-10% of note duration)
            float transitionTime = static_cast<float>(note.duration) * 0.05f;
            transitionTime = std::clamp(transitionTime, 0.01f, 0.1f);  // 10ms to 100ms

            vocoder_->setTargetFormants(formantData.frequencies, formantData.bandwidths, transitionTime);
            currentVowel_ = targetVowel;
        }

        // Convert MIDI pitch to frequency
        float frequency = midiToFrequency(note.pitch);

        // Set portamento target
        portamento_->setTargetPitch(frequency);

        // Configure envelope (short attack, medium decay, sustain, release)
        envelope_->attackTime = 0.01f;   // 10ms attack
        envelope_->decayTime = 0.05f;    // 50ms decay
        envelope_->sustainLevel = 0.7f;  // 70% sustain
        envelope_->releaseTime = 0.1f;   // 100ms release
        envelope_->prepare(sampleRate_);
        envelope_->trigger();

        // Generate expression curve for this note
        float notePosition = 0.0f;
        float noteDuration = static_cast<float>(durationSamples);

        // Synthesize this note
        for (int64_t i = startSample; i < endSample; ++i) {
            int64_t noteOffset = i - startSample;
            notePosition = static_cast<float>(noteOffset) / noteDuration;

            // Check if note should be released
            int64_t releaseStart = endSample - beatsToSamples(envelope_->releaseTime);
            if (i >= releaseStart && envelope_->getValue() > 0.0f) {
                envelope_->release();
            }

            // Get current pitch with portamento
            float currentPitch = portamento_->getCurrentPitch();

            // Get envelope value
            float envelopeValue = envelope_->getValue();
            if (envelopeValue <= 0.0f && i > startSample) {
                break; // Note finished
            }

            // Get expression at current position
            VocalExpression currentExpression = note.expression;
            if (note.expression.vibratoRate > 0.0f) {
                currentExpression = expressionEngine_->getExpressionAtPosition(note.expression, notePosition);
            }

            // Apply dynamics curve (crescendo/diminuendo)
            float dynamicsMultiplier = 1.0f;
            if (currentExpression.crescendo > 0.0f || currentExpression.diminuendo > 0.0f) {
                std::vector<float> dynamicsCurve = expressionEngine_->generateDynamicsCurve(
                    note.duration, currentExpression.crescendo, currentExpression.diminuendo, 1);
                if (!dynamicsCurve.empty()) {
                    int curveIndex = std::min(static_cast<int>(notePosition * dynamicsCurve.size()),
                                            static_cast<int>(dynamicsCurve.size() - 1));
                    dynamicsMultiplier = dynamicsCurve[curveIndex];
                }
            }

            // Update characteristics with expression
            chars.brightness = currentExpression.brightness;
            chars.breathiness = currentExpression.breathiness;
            chars.vibratoRate = currentExpression.vibratoRate;
            chars.vibratoDepth = currentExpression.vibratoDepth;

            // Synthesize sample
            float sample = vocoder_->processSample(
                currentPitch,
                formantData.frequencies,
                formantData.bandwidths,
                chars.vibratoDepth,
                chars.vibratoRate,
                chars.breathiness,
                chars.brightness
            );

            // Apply envelope and dynamics
            sample *= envelopeValue * dynamicsMultiplier * currentExpression.dynamics;

            // Mix into output (simple mix for now)
            output[i] += sample * 0.5f; // Scale down to prevent clipping
        }

        // Reset portamento for next note
        portamento_->reset();
    }

    // Normalize if needed to prevent clipping
    float maxValue = 0.0f;
    for (float sample : output) {
        maxValue = std::max(maxValue, std::abs(sample));
    }
    if (maxValue > 1.0f) {
        float scale = 0.95f / maxValue;
        for (float& sample : output) {
            sample *= scale;
        }
    }

    return output;
}

void VoiceSynthesizer::synthesizeBlock(
    const std::vector<VocalNote>& notes,
    float* outputBuffer,
    int numSamples,
    int64_t currentSample,
    const EmotionNode* emotion)
{
    if (!enabled_ || notes.empty() || !outputBuffer) {
        std::fill(outputBuffer, outputBuffer + numSamples, 0.0f);
        return;
    }

    // Clear output buffer
    std::fill(outputBuffer, outputBuffer + numSamples, 0.0f);

    // Find active notes in this block
    for (const auto& note : notes) {
        int64_t startSample = beatsToSamples(note.startBeat);
        int64_t endSample = startSample + beatsToSamples(note.duration);

        // Check if note overlaps with current block
        if (endSample < currentSample || startSample > currentSample + numSamples) {
            continue; // Note doesn't overlap
        }

        // Get vocal characteristics (use expression from note if available, otherwise emotion-based)
        VocalCharacteristics chars;
        if (note.expression.vibratoRate > 0.0f) {
            chars.brightness = note.expression.brightness;
            chars.breathiness = note.expression.breathiness;
            chars.vibratoRate = note.expression.vibratoRate;
            chars.vibratoDepth = note.expression.vibratoDepth;
        } else if (emotion) {
            chars = getVocalCharacteristics(*emotion);
            chars.vibratoDepth = note.vibrato > 0.0f ? note.vibrato : chars.vibratoDepth;
        } else {
            chars.brightness = 0.5f;
            chars.breathiness = 0.2f;
            chars.vibratoRate = 5.0f;
            chars.vibratoDepth = note.vibrato;
        }

        // Get formant data (use emotion if available)
        VowelFormantDatabase::Vowel targetVowel = selectVowel(note.pitch, emotion);
        auto formantData = VowelFormantDatabase::getFormants(targetVowel);

        // Apply voice type formant shifts
        for (size_t i = 0; i < 4; ++i) {
            float shift = 1.0f;
            if (i == 0) shift = voiceParams_.formantShiftF1;
            else if (i == 1) shift = voiceParams_.formantShiftF2;
            else if (i == 2) shift = voiceParams_.formantShiftF3;
            else shift = voiceParams_.formantShift;

            formantData.frequencies[i] *= shift;
        }

        vocoder_->setFormantShift(voiceParams_.formantShift);

        // Smooth formant transition if vowel changed (for real-time synthesis)
        if (targetVowel != currentVowel_) {
            float transitionTime = 0.05f;  // 50ms transition for real-time
            vocoder_->setTargetFormants(formantData.frequencies, formantData.bandwidths, transitionTime);
            currentVowel_ = targetVowel;
        }

        // Convert MIDI pitch to frequency
        float frequency = midiToFrequency(note.pitch);
        portamento_->setTargetPitch(frequency);

        // Process each sample in the block
        for (int i = 0; i < numSamples; ++i) {
            int64_t globalSample = currentSample + i;

            // Check if we're within the note's range
            if (globalSample < startSample || globalSample >= endSample) {
                continue;
            }

            // Get current pitch with portamento
            float currentPitch = portamento_->getCurrentPitch();

            // Simple envelope (could be improved with proper ADSR per note)
            int64_t noteOffset = globalSample - startSample;
            float notePosition = static_cast<float>(noteOffset) / static_cast<float>(endSample - startSample);
            float envelopeValue = 1.0f;

            // Simple fade in/out
            if (noteOffset < beatsToSamples(0.01)) {
                envelopeValue = static_cast<float>(noteOffset) / static_cast<float>(beatsToSamples(0.01));
            } else if (globalSample > endSample - beatsToSamples(0.1)) {
                int64_t releaseOffset = endSample - globalSample;
                envelopeValue = static_cast<float>(releaseOffset) / static_cast<float>(beatsToSamples(0.1));
            }

            // Get expression at current position
            VocalExpression currentExpression = note.expression;
            if (note.expression.vibratoRate > 0.0f) {
                currentExpression = expressionEngine_->getExpressionAtPosition(note.expression, notePosition);
                chars.brightness = currentExpression.brightness;
                chars.breathiness = currentExpression.breathiness;
                chars.vibratoRate = currentExpression.vibratoRate;
                chars.vibratoDepth = currentExpression.vibratoDepth;
            }

            // Synthesize sample
            float sample = vocoder_->processSample(
                currentPitch,
                formantData.frequencies,
                formantData.bandwidths,
                chars.vibratoDepth,
                chars.vibratoRate,
                chars.breathiness,
                chars.brightness
            );

            // Apply envelope, dynamics, and mix
            float dynamicsMultiplier = currentExpression.dynamics;
            outputBuffer[i] += sample * envelopeValue * dynamicsMultiplier * 0.5f;
        }
    }
}

float VoiceSynthesizer::midiToFrequency(int midiPitch) const {
    // MIDI note 69 (A4) = 440 Hz
    return 440.0f * std::pow(2.0f, (midiPitch - 69) / 12.0f);
}

int64_t VoiceSynthesizer::beatsToSamples(double beats) const {
    // Convert beats to samples: beats * (60 / BPM) * sampleRate
    return static_cast<int64_t>(beats * (60.0 / static_cast<double>(bpm_)) * sampleRate_);
}

VowelFormantDatabase::Vowel VoiceSynthesizer::selectVowel(int midiPitch, const EmotionNode* emotion) const {
    // Select vowel based on pitch and optionally emotion
    // Low valence (negative emotions) -> more open vowels (AH, OH)
    // High valence (positive emotions) -> more close vowels (EE, IY)
    // High arousal -> brighter vowels

    if (emotion) {
        // Emotion-based selection
        if (emotion->valence < -0.3f) {
            // Negative emotions -> open vowels
            if (midiPitch < 60) return VowelFormantDatabase::Vowel::AH;
            else if (midiPitch < 72) return VowelFormantDatabase::Vowel::OH;
            else return VowelFormantDatabase::Vowel::UH;
        } else if (emotion->valence > 0.3f) {
            // Positive emotions -> close vowels
            if (midiPitch < 60) return VowelFormantDatabase::Vowel::EH;
            else if (midiPitch < 72) return VowelFormantDatabase::Vowel::IH;
            else return VowelFormantDatabase::Vowel::EE;
        }
    }

    // Default pitch-based selection
    if (midiPitch < 60) {
        return VowelFormantDatabase::Vowel::AH;  // Open vowel for low pitches
    } else if (midiPitch < 72) {
        return VowelFormantDatabase::Vowel::EH;  // Mid-open
    } else if (midiPitch < 84) {
        return VowelFormantDatabase::Vowel::IH;  // Mid-close
    } else {
        return VowelFormantDatabase::Vowel::EE;  // Close vowel for high pitches
    }
}

VoiceSynthesizer::VocalCharacteristics VoiceSynthesizer::getVocalCharacteristics(const EmotionNode& emotion) const {
    VocalCharacteristics chars;

    // Map emotion valence to vocal brightness
    // Positive valence = brighter, negative = darker
    // Range: -1.0 to 1.0 -> 0.0 to 1.0 (brightness)
    chars.brightness = (emotion.valence + 1.0f) * 0.5f;
    chars.brightness = std::clamp(chars.brightness, 0.2f, 0.9f); // Clamp to reasonable range

    // Map emotion arousal to vibrato rate
    // High arousal = faster vibrato
    // Range: 0.0 to 1.0 (arousal) -> 3.0 to 7.0 Hz (vibrato rate)
    chars.vibratoRate = 3.0f + (emotion.arousal * 4.0f);
    chars.vibratoRate = std::clamp(chars.vibratoRate, 3.0f, 7.0f);

    // Map emotion intensity to vibrato depth
    // High intensity = deeper vibrato
    chars.vibratoDepth = emotion.intensity * 0.5f;
    chars.vibratoDepth = std::clamp(chars.vibratoDepth, 0.1f, 0.8f);

    // Map emotion dominance to breathiness (indirectly affects dynamics)
    // High dominance = less breathy (more confident/powerful)
    // Low dominance = more breathy (softer, more vulnerable)
    chars.breathiness = (1.0f - emotion.dominance) * 0.5f;
    chars.breathiness = std::clamp(chars.breathiness, 0.1f, 0.6f);

    // Additional mappings based on emotion category
    switch (emotion.categoryEnum) {
        case EmotionCategory::Joy:
            chars.brightness = std::max(chars.brightness, 0.7f);
            chars.vibratoRate = std::max(chars.vibratoRate, 5.0f);
            break;

        case EmotionCategory::Sadness:
            chars.brightness = std::min(chars.brightness, 0.4f);
            chars.breathiness = std::max(chars.breathiness, 0.3f);
            chars.vibratoRate = std::min(chars.vibratoRate, 4.5f);
            break;

        case EmotionCategory::Anger:
            chars.brightness = 0.6f; // Medium brightness for intensity
            chars.vibratoRate = std::max(chars.vibratoRate, 5.5f);
            chars.breathiness = std::min(chars.breathiness, 0.3f); // Less breathy
            break;

        case EmotionCategory::Fear:
            chars.brightness = std::min(chars.brightness, 0.5f);
            chars.breathiness = std::max(chars.breathiness, 0.4f);
            break;

        default:
            break;
    }

    return chars;
}

void VoiceSynthesizer::setVoiceType(VoiceType voiceType) {
    voiceType_ = voiceType;
    voiceParams_.type = voiceType;

    // Set voice type parameters based on type
    switch (voiceType) {
        case VoiceType::Male:
            voiceParams_.formantShift = 0.85f;
            voiceParams_.pitchRangeMin = 48.0f;  // C3
            voiceParams_.pitchRangeMax = 78.0f;  // F#5
            voiceParams_.formantShiftF1 = 0.85f;
            voiceParams_.formantShiftF2 = 0.85f;
            voiceParams_.formantShiftF3 = 0.90f;
            break;

        case VoiceType::Female:
            voiceParams_.formantShift = 1.15f;
            voiceParams_.pitchRangeMin = 60.0f;  // C4
            voiceParams_.pitchRangeMax = 90.0f;  // F#6
            voiceParams_.formantShiftF1 = 1.15f;
            voiceParams_.formantShiftF2 = 1.15f;
            voiceParams_.formantShiftF3 = 1.20f;
            break;

        case VoiceType::Child:
            voiceParams_.formantShift = 1.30f;
            voiceParams_.pitchRangeMin = 72.0f;  // C5
            voiceParams_.pitchRangeMax = 96.0f;  // C7
            voiceParams_.formantShiftF1 = 1.30f;
            voiceParams_.formantShiftF2 = 1.30f;
            voiceParams_.formantShiftF3 = 1.35f;
            break;

        case VoiceType::Neutral:
        default:
            voiceParams_.formantShift = 1.0f;
            voiceParams_.pitchRangeMin = 48.0f;  // C3
            voiceParams_.pitchRangeMax = 84.0f;  // C6
            voiceParams_.formantShiftF1 = 1.0f;
            voiceParams_.formantShiftF2 = 1.0f;
            voiceParams_.formantShiftF3 = 1.0f;
            break;
    }

    // Apply formant shift to vocoder
    vocoder_->setFormantShift(voiceParams_.formantShift);
}

std::vector<VoiceSynthesizer::MidiLyricEvent> VoiceSynthesizer::generateMidiLyricEvents(
    const std::vector<VocalNote>& notes,
    int ticksPerBeat) const
{
    std::vector<MidiLyricEvent> events;

    for (const auto& note : notes) {
        if (!note.lyric.empty()) {
            MidiLyricEvent event;
            event.text = note.lyric;
            event.beat = note.startBeat;
            // Convert beats to MIDI ticks
            event.tick = static_cast<int>(note.startBeat * ticksPerBeat);
            events.push_back(event);
        }
    }

    return events;
}

void VoiceSynthesizer::setBPM(float bpm) {
    bpm_ = bpm;
    if (aligner_) {
        aligner_->setBPM(bpm);
    }
}

} // namespace kelly
