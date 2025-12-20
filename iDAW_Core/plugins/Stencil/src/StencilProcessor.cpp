/**
 * StencilProcessor.cpp - Sidechain/Ducking Implementation
 */

#include "../include/StencilProcessor.h"

namespace iDAW {

StencilProcessor::StencilProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withInput("Sidechain", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

StencilProcessor::~StencilProcessor() = default;

void StencilProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    m_sampleRate = sampleRate;
    m_envelopeState = 0.0f;
    m_lfoPhase = 0.0f;
    m_midiTriggered = false;
    updateCoefficients();
    m_prepared = true;
}

void StencilProcessor::releaseResources()
{
    m_prepared = false;
}

void StencilProcessor::updateCoefficients()
{
    // Calculate envelope coefficients (one-pole filter)
    float attackMs = m_attackMs.load();
    float releaseMs = m_releaseMs.load();

    m_attackCoeff = std::exp(-1.0f / (static_cast<float>(m_sampleRate) * attackMs / 1000.0f));
    m_releaseCoeff = std::exp(-1.0f / (static_cast<float>(m_sampleRate) * releaseMs / 1000.0f));
}

float StencilProcessor::calculateEnvelope(float inputLevel)
{
    // One-pole envelope follower
    float coeff = (inputLevel > m_envelopeState) ? m_attackCoeff : m_releaseCoeff;
    m_envelopeState = coeff * m_envelopeState + (1.0f - coeff) * inputLevel;
    return m_envelopeState;
}

float StencilProcessor::processInternalLFO()
{
    float rate = m_internalRateHz.load();
    float phaseIncrement = rate / static_cast<float>(m_sampleRate);

    m_lfoPhase += phaseIncrement;
    if (m_lfoPhase >= 1.0f)
        m_lfoPhase -= 1.0f;

    // Generate a ducking curve (sharp attack, slow release feel)
    // Using a modified sine wave
    float lfoValue = 0.5f * (1.0f - std::cos(2.0f * M_PI * m_lfoPhase));
    return lfoValue;
}

void StencilProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                     juce::MidiBuffer& midiMessages)
{
    if (!m_prepared)
        return;

    const int numSamples = buffer.getNumSamples();
    const int numChannels = std::min(buffer.getNumChannels(), 2);

    // Get parameters
    const float depth = m_depth.load();
    const float threshold = dbToLinear(m_thresholdDb.load());
    const float mix = m_mix.load();

    // Handle MIDI triggers
    for (const auto& metadata : midiMessages)
    {
        auto message = metadata.getMessage();
        if (message.isNoteOn())
            m_midiTriggered = true;
        else if (message.isNoteOff())
            m_midiTriggered = false;
    }

    float maxInputLevel = 0.0f;
    float maxOutputLevel = 0.0f;

    for (int sample = 0; sample < numSamples; ++sample)
    {
        // Get sidechain input level based on source
        float sidechainLevel = 0.0f;

        switch (m_source)
        {
            case SidechainSource::EXTERNAL:
            {
                // Use external sidechain bus (channels 2-3 if available)
                // For now, fall back to self-sidechain
                float sum = 0.0f;
                for (int ch = 0; ch < numChannels; ++ch)
                    sum += std::abs(buffer.getSample(ch, sample));
                sidechainLevel = sum / static_cast<float>(numChannels);
                break;
            }

            case SidechainSource::INTERNAL:
                sidechainLevel = processInternalLFO();
                break;

            case SidechainSource::MIDI_TRIGGER:
                sidechainLevel = m_midiTriggered ? 1.0f : 0.0f;
                break;
        }

        maxInputLevel = std::max(maxInputLevel, sidechainLevel);

        // Calculate envelope
        float envelope = calculateEnvelope(sidechainLevel);

        // Calculate ducking amount (0 = no ducking, 1 = full ducking)
        float duckAmount = 0.0f;
        if (m_source == SidechainSource::INTERNAL || m_source == SidechainSource::MIDI_TRIGGER)
        {
            duckAmount = envelope;
        }
        else
        {
            // Threshold-based ducking for external source
            if (envelope > threshold)
            {
                duckAmount = (envelope - threshold) / (1.0f - threshold + 0.001f);
                duckAmount = std::min(duckAmount, 1.0f);
            }
        }

        // Apply depth
        float gainReduction = 1.0f - (duckAmount * depth);

        // Apply to all channels
        for (int ch = 0; ch < numChannels; ++ch)
        {
            float dry = buffer.getSample(ch, sample);
            float wet = dry * gainReduction;
            float output = dry * (1.0f - mix) + wet * mix;
            buffer.setSample(ch, sample, output);

            maxOutputLevel = std::max(maxOutputLevel, std::abs(output));
        }
    }

    // Update metering
    m_currentDucking.store(1.0f - (m_depth.load() * m_envelopeState));
    m_inputLevel.store(linearToDb(maxInputLevel));
    m_outputLevel.store(linearToDb(maxOutputLevel));

    // Update visual state
    {
        std::lock_guard<std::mutex> lock(m_visualMutex);
        m_visualState.cutoutDepth = m_envelopeState * m_depth.load();
        m_visualState.inputLevel = maxInputLevel;
        m_visualState.outputLevel = maxOutputLevel;
        m_visualState.cutoutProgress += 0.01f;
        if (m_visualState.cutoutProgress > 1.0f)
            m_visualState.cutoutProgress -= 1.0f;
    }
}

// Parameter setters
void StencilProcessor::setAttack(float attackMs)
{
    m_attackMs.store(std::clamp(attackMs, StencilConfig::MIN_ATTACK_MS, StencilConfig::MAX_ATTACK_MS));
    updateCoefficients();
}

void StencilProcessor::setRelease(float releaseMs)
{
    m_releaseMs.store(std::clamp(releaseMs, StencilConfig::MIN_RELEASE_MS, StencilConfig::MAX_RELEASE_MS));
    updateCoefficients();
}

void StencilProcessor::setDepth(float depth)
{
    m_depth.store(std::clamp(depth, StencilConfig::MIN_DEPTH, StencilConfig::MAX_DEPTH));
}

void StencilProcessor::setThreshold(float thresholdDb)
{
    m_thresholdDb.store(std::clamp(thresholdDb, StencilConfig::MIN_THRESHOLD_DB, StencilConfig::MAX_THRESHOLD_DB));
}

void StencilProcessor::setMix(float mix)
{
    m_mix.store(std::clamp(mix, 0.0f, 1.0f));
}

void StencilProcessor::setSource(SidechainSource source)
{
    m_source = source;
}

void StencilProcessor::setInternalRate(float rateHz)
{
    m_internalRateHz.store(std::clamp(rateHz, 0.1f, 20.0f));
}

// Ghost Hands
void StencilProcessor::applyAISuggestion(const juce::String& suggestion)
{
    if (suggestion.containsIgnoreCase("pump") || suggestion.containsIgnoreCase("edm"))
    {
        setAttack(1.0f);
        setRelease(200.0f);
        setDepth(0.9f);
        setSource(SidechainSource::INTERNAL);
        setInternalRate(4.0f);
    }
    else if (suggestion.containsIgnoreCase("subtle") || suggestion.containsIgnoreCase("gentle"))
    {
        setAttack(20.0f);
        setRelease(300.0f);
        setDepth(0.4f);
    }
    else if (suggestion.containsIgnoreCase("aggressive"))
    {
        setAttack(0.5f);
        setRelease(100.0f);
        setDepth(1.0f);
    }
}

// Visual state
CutoutVisualState StencilProcessor::getVisualState() const
{
    std::lock_guard<std::mutex> lock(m_visualMutex);
    return m_visualState;
}

// Editor
juce::AudioProcessorEditor* StencilProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

// State save/load
void StencilProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    struct State {
        float attackMs, releaseMs, depth, thresholdDb, mix, internalRateHz;
        int source;
    };

    State state{
        m_attackMs.load(),
        m_releaseMs.load(),
        m_depth.load(),
        m_thresholdDb.load(),
        m_mix.load(),
        m_internalRateHz.load(),
        static_cast<int>(m_source)
    };

    destData.append(&state, sizeof(State));
}

void StencilProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    struct State {
        float attackMs, releaseMs, depth, thresholdDb, mix, internalRateHz;
        int source;
    };

    if (sizeInBytes >= static_cast<int>(sizeof(State)))
    {
        const State* state = static_cast<const State*>(data);
        setAttack(state->attackMs);
        setRelease(state->releaseMs);
        setDepth(state->depth);
        setThreshold(state->thresholdDb);
        setMix(state->mix);
        setInternalRate(state->internalRateHz);
        m_source = static_cast<SidechainSource>(state->source);
    }
}

} // namespace iDAW
