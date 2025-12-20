#include "voice/MultiVoiceHarmony.h"
#include <algorithm>
#include <cmath>

namespace kelly {

MultiVoiceHarmony::MultiVoiceHarmony() {
    // Initialize default configurations for each voice part
    voiceConfigs_[VoicePart::Soprano] = getDefaultConfig(VoicePart::Soprano);
    voiceConfigs_[VoicePart::Alto] = getDefaultConfig(VoicePart::Alto);
    voiceConfigs_[VoicePart::Tenor] = getDefaultConfig(VoicePart::Tenor);
    voiceConfigs_[VoicePart::Bass] = getDefaultConfig(VoicePart::Bass);

    initializeSynthesizers();
}

void MultiVoiceHarmony::initializeSynthesizers() {
    for (auto& [part, config] : voiceConfigs_) {
        auto synth = std::make_unique<VoiceSynthesizer>();
        synth->setVoiceType(config.voiceType);
        synth->prepare(44100.0);  // Default sample rate, can be updated
        synthesizers_[part] = std::move(synth);
    }
}

MultiVoiceHarmony::VoicePartConfig MultiVoiceHarmony::getDefaultConfig(VoicePart part) {
    VoicePartConfig config;
    config.part = part;

    switch (part) {
        case VoicePart::Soprano:
            config.pitchOffset = 0;      // Melody line
            config.volume = 1.0f;
            config.voiceType = VoiceType::Female;
            config.formantShift = 1.15f;  // Higher formants
            config.vibratoDepth = 0.3f;
            config.vibratoRate = 5.5f;
            break;

        case VoicePart::Alto:
            config.pitchOffset = -5;     // Third below
            config.volume = 0.9f;
            config.voiceType = VoiceType::Female;
            config.formantShift = 1.05f;
            config.vibratoDepth = 0.25f;
            config.vibratoRate = 5.0f;
            break;

        case VoicePart::Tenor:
            config.pitchOffset = -12;    // Octave below
            config.volume = 0.85f;
            config.voiceType = VoiceType::Male;
            config.formantShift = 0.85f;  // Lower formants
            config.vibratoDepth = 0.2f;
            config.vibratoRate = 4.5f;
            break;

        case VoicePart::Bass:
            config.pitchOffset = -19;    // Fifth below tenor
            config.volume = 0.8f;
            config.voiceType = VoiceType::Male;
            config.formantShift = 0.75f;  // Lowest formants
            config.vibratoDepth = 0.15f;
            config.vibratoRate = 4.0f;
            break;
    }

    return config;
}

std::map<MultiVoiceHarmony::VoicePart, std::vector<VoiceSynthesizer::VocalNote>>
MultiVoiceHarmony::generateHarmony(
    const std::vector<VoiceSynthesizer::VocalNote>& melodyNotes,
    const EmotionNode& emotion,
    const std::string& harmonyType)
{
    std::map<VoicePart, std::vector<VoiceSynthesizer::VocalNote>> harmonyParts;

    if (melodyNotes.empty()) {
        return harmonyParts;
    }

    // Generate SATB harmony by default
    if (harmonyType == "satb" || harmonyType == "4part") {
        return generateSATB(melodyNotes, emotion);
    }

    // Generate based on harmony style
    if (harmonyType == "parallel") {
        // Parallel thirds and sixths
        harmonyParts[VoicePart::Soprano] = melodyNotes;
        harmonyParts[VoicePart::Alto] = generateParallelHarmony(melodyNotes, -3);
        harmonyParts[VoicePart::Tenor] = generateParallelHarmony(melodyNotes, -7);
        harmonyParts[VoicePart::Bass] = generateParallelHarmony(melodyNotes, -12);
    } else if (harmonyType == "block") {
        // Block chord harmony
        harmonyParts[VoicePart::Soprano] = melodyNotes;
        harmonyParts[VoicePart::Alto] = generateBlockChordHarmony(melodyNotes, {-3, -7, -12});
    } else {
        // Default: parallel harmony
        harmonyParts[VoicePart::Soprano] = melodyNotes;
        harmonyParts[VoicePart::Alto] = generateParallelHarmony(melodyNotes, -5);
    }

    return harmonyParts;
}

std::map<MultiVoiceHarmony::VoicePart, std::vector<VoiceSynthesizer::VocalNote>>
MultiVoiceHarmony::generateSATB(
    const std::vector<VoiceSynthesizer::VocalNote>& melodyNotes,
    const EmotionNode& emotion)
{
    std::map<VoicePart, std::vector<VoiceSynthesizer::VocalNote>> harmonyParts;

    // Soprano: melody line
    harmonyParts[VoicePart::Soprano] = melodyNotes;

    // Alto: third below soprano
    harmonyParts[VoicePart::Alto] = generateParallelHarmony(melodyNotes, -4);

    // Tenor: octave below soprano, adjusted for chord tones
    auto tenorNotes = generateParallelHarmony(melodyNotes, -12);
    // Adjust tenor to be within range (C4-G5 typically)
    for (auto& note : tenorNotes) {
        if (note.pitch < 60) {  // Below C4
            note.pitch += 12;   // Octave up
        }
    }
    harmonyParts[VoicePart::Tenor] = tenorNotes;

    // Bass: fifth below tenor (or root of chord)
    harmonyParts[VoicePart::Bass] = generateParallelHarmony(tenorNotes, -7);

    return harmonyParts;
}

std::vector<VoiceSynthesizer::VocalNote> MultiVoiceHarmony::generateParallelHarmony(
    const std::vector<VoiceSynthesizer::VocalNote>& melody,
    int intervalSemitones)
{
    std::vector<VoiceSynthesizer::VocalNote> harmony;

    for (const auto& note : melody) {
        VoiceSynthesizer::VocalNote harmonyNote = note;
        harmonyNote.pitch = transposePitch(note.pitch, intervalSemitones);
        harmony.push_back(harmonyNote);
    }

    return harmony;
}

std::vector<VoiceSynthesizer::VocalNote> MultiVoiceHarmony::generateBlockChordHarmony(
    const std::vector<VoiceSynthesizer::VocalNote>& melody,
    const std::vector<int>& chordIntervals)
{
    std::vector<VoiceSynthesizer::VocalNote> harmony;

    for (const auto& note : melody) {
        // Use first interval for this voice part
        if (!chordIntervals.empty()) {
            VoiceSynthesizer::VocalNote harmonyNote = note;
            harmonyNote.pitch = transposePitch(note.pitch, chordIntervals[0]);
            harmony.push_back(harmonyNote);
        }
    }

    return harmony;
}

std::vector<VoiceSynthesizer::VocalNote> MultiVoiceHarmony::generateCounterpointHarmony(
    const std::vector<VoiceSynthesizer::VocalNote>& melody,
    int voicePartIndex,
    const EmotionNode& emotion)
{
    // Simplified counterpoint: creates a complementary melodic line
    // In a full implementation, this would follow counterpoint rules
    std::vector<VoiceSynthesizer::VocalNote> counterpoint;

    int offset = -7 * (voicePartIndex + 1);  // Fifth intervals

    for (const auto& note : melody) {
        VoiceSynthesizer::VocalNote cpNote = note;
        cpNote.pitch = transposePitch(note.pitch, offset);
        // Slightly modify timing for independence
        cpNote.startBeat += 0.05;
        counterpoint.push_back(cpNote);
    }

    return counterpoint;
}

int MultiVoiceHarmony::transposePitch(int pitch, int semitones) {
    int newPitch = pitch + semitones;
    // Clamp to MIDI range
    return std::max(0, std::min(127, newPitch));
}

void MultiVoiceHarmony::setVoicePartConfig(VoicePart part, const VoicePartConfig& config) {
    voiceConfigs_[part] = config;

    // Update synthesizer for this part
    if (synthesizers_.find(part) != synthesizers_.end()) {
        synthesizers_[part]->setVoiceType(config.voiceType);
    }
}

MultiVoiceHarmony::VoicePartConfig MultiVoiceHarmony::getVoicePartConfig(VoicePart part) const {
    auto it = voiceConfigs_.find(part);
    if (it != voiceConfigs_.end()) {
        return it->second;
    }
    return getDefaultConfig(part);
}

std::vector<float> MultiVoiceHarmony::synthesizeHarmony(
    const std::map<VoicePart, std::vector<VoiceSynthesizer::VocalNote>>& harmonyParts,
    double sampleRate,
    const EmotionNode* emotion)
{
    // Synthesize each voice part
    std::map<VoicePart, std::vector<float>> voiceAudio;

    for (const auto& [part, notes] : harmonyParts) {
        if (synthesizers_.find(part) != synthesizers_.end()) {
            auto& synth = synthesizers_[part];
            synth->prepare(sampleRate);

            auto audio = synth->synthesizeAudio(notes, sampleRate, emotion);

            // Apply volume scaling
            const auto& config = voiceConfigs_[part];
            for (float& sample : audio) {
                sample *= config.volume;
            }

            voiceAudio[part] = audio;
        }
    }

    // Mix all voices together
    size_t maxLength = 0;
    for (const auto& [part, audio] : voiceAudio) {
        maxLength = std::max(maxLength, audio.size());
    }

    std::vector<float> mixedAudio(maxLength, 0.0f);

    for (const auto& [part, audio] : voiceAudio) {
        for (size_t i = 0; i < audio.size() && i < mixedAudio.size(); ++i) {
            mixedAudio[i] += audio[i];
        }
    }

    // Normalize to prevent clipping
    float maxAmplitude = 0.0f;
    for (float sample : mixedAudio) {
        maxAmplitude = std::max(maxAmplitude, std::abs(sample));
    }

    if (maxAmplitude > 1.0f) {
        float scale = 0.95f / maxAmplitude;
        for (float& sample : mixedAudio) {
            sample *= scale;
        }
    }

    return mixedAudio;
}

} // namespace kelly
