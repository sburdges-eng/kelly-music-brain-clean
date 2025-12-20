/**
 * BrushProcessor.cpp - Modulated Filter Implementation
 */

#include "../include/BrushProcessor.h"
#include <random>

namespace iDAW {

BrushProcessor::BrushProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    m_visualState.trailPos.fill(0.5f);
}

BrushProcessor::~BrushProcessor() = default;

void BrushProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    m_sampleRate = sampleRate;

    // Reset filter state
    for (auto& state : m_svfState)
    {
        state.low = 0.0f;
        state.band = 0.0f;
        state.high = 0.0f;
        state.notch = 0.0f;
    }

    m_lfoPhase = 0.0f;
    m_envelopeState = 0.0f;
    m_lastRandomValue = 0.0f;
    m_randomHoldCounter = 0.0f;

    // Calculate envelope coefficients
    float attackMs = m_envAttackMs.load();
    float releaseMs = m_envReleaseMs.load();
    m_envAttackCoeff = std::exp(-1.0f / (static_cast<float>(m_sampleRate) * attackMs / 1000.0f));
    m_envReleaseCoeff = std::exp(-1.0f / (static_cast<float>(m_sampleRate) * releaseMs / 1000.0f));

    m_prepared = true;
}

void BrushProcessor::releaseResources()
{
    m_prepared = false;
}

void BrushProcessor::updateLFO()
{
    float rate = m_lfoRateHz.load();
    m_lfoPhase += rate / static_cast<float>(m_sampleRate);

    if (m_lfoPhase >= 1.0f)
    {
        m_lfoPhase -= 1.0f;

        // Update random value for sample-and-hold
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        m_lastRandomValue = dist(rng);
    }
}

float BrushProcessor::getLFOValue()
{
    float phase = m_lfoPhase;

    switch (m_lfoWaveform)
    {
        case LFOWaveform::SINE:
            return std::sin(2.0f * M_PI * phase);

        case LFOWaveform::TRIANGLE:
            return 4.0f * std::abs(phase - 0.5f) - 1.0f;

        case LFOWaveform::SAW_UP:
            return 2.0f * phase - 1.0f;

        case LFOWaveform::SAW_DOWN:
            return 1.0f - 2.0f * phase;

        case LFOWaveform::SQUARE:
            return (phase < 0.5f) ? 1.0f : -1.0f;

        case LFOWaveform::RANDOM_HOLD:
            return m_lastRandomValue;

        default:
            return 0.0f;
    }
}

float BrushProcessor::getEnvelopeValue(float inputLevel)
{
    float coeff = (inputLevel > m_envelopeState) ? m_envAttackCoeff : m_envReleaseCoeff;
    m_envelopeState = coeff * m_envelopeState + (1.0f - coeff) * inputLevel;
    return m_envelopeState;
}

void BrushProcessor::processSVF(float input, float cutoff, float resonance, int channel)
{
    // State Variable Filter (Chamberlin implementation)
    // Clamp cutoff to prevent instability
    float f = 2.0f * std::sin(M_PI * std::min(cutoff / static_cast<float>(m_sampleRate), 0.25f));

    // Q from resonance (0 = Q of 0.5, 1 = high Q approaching self-oscillation)
    float q = 1.0f - resonance * 0.95f;

    auto& state = m_svfState[channel];

    // Two iterations for better stability at high frequencies
    for (int i = 0; i < 2; ++i)
    {
        state.low = state.low + f * state.band;
        state.high = (i == 0 ? input : 0.0f) - state.low - q * state.band;
        state.band = f * state.high + state.band;
        state.notch = state.high + state.low;
    }
}

void BrushProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                   juce::MidiBuffer& /*midiMessages*/)
{
    if (!m_prepared)
        return;

    const int numSamples = buffer.getNumSamples();
    const int numChannels = std::min(buffer.getNumChannels(), 2);

    // Get parameters
    const float baseCutoff = m_cutoffHz.load();
    const float resonance = m_resonance.load();
    const float lfoDepth = m_lfoDepth.load();
    const float envDepth = m_envDepth.load();
    const float mix = m_mix.load();

    for (int sample = 0; sample < numSamples; ++sample)
    {
        updateLFO();

        // Calculate input level for envelope follower
        float inputSum = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch)
            inputSum += std::abs(buffer.getSample(ch, sample));
        float inputLevel = inputSum / static_cast<float>(numChannels);

        // Get modulation values
        float lfoValue = getLFOValue();
        float envValue = getEnvelopeValue(inputLevel);

        // Calculate modulated cutoff (exponential scaling)
        float cutoffMod = 1.0f;

        // LFO modulation (bipolar, +/- 2 octaves at full depth)
        cutoffMod *= std::pow(2.0f, lfoValue * lfoDepth * 2.0f);

        // Envelope modulation (unipolar, +2 octaves at full depth)
        cutoffMod *= std::pow(2.0f, envValue * envDepth * 2.0f);

        float modulatedCutoff = baseCutoff * cutoffMod;
        modulatedCutoff = std::clamp(modulatedCutoff, BrushConfig::MIN_CUTOFF_HZ, BrushConfig::MAX_CUTOFF_HZ);

        m_currentCutoff.store(modulatedCutoff);

        for (int ch = 0; ch < numChannels; ++ch)
        {
            float dry = buffer.getSample(ch, sample);

            // Process SVF
            processSVF(dry, modulatedCutoff, resonance, ch);

            // Select output based on filter type
            float wet = 0.0f;
            switch (m_filterType)
            {
                case FilterType::LOWPASS:
                    wet = m_svfState[ch].low;
                    break;
                case FilterType::HIGHPASS:
                    wet = m_svfState[ch].high;
                    break;
                case FilterType::BANDPASS:
                    wet = m_svfState[ch].band;
                    break;
                case FilterType::NOTCH:
                    wet = m_svfState[ch].notch;
                    break;
            }

            // Apply resonance boost to output
            wet *= 1.0f + resonance * 0.5f;

            // Soft clip to prevent resonance blowup
            wet = std::tanh(wet);

            // Mix
            float output = dry * (1.0f - mix) + wet * mix;
            buffer.setSample(ch, sample, output);
        }
    }

    // Update visual state
    {
        std::lock_guard<std::mutex> lock(m_visualMutex);
        float normalizedCutoff = std::log2(m_currentCutoff.load() / 20.0f) / std::log2(20000.0f / 20.0f);
        m_visualState.strokePosition = normalizedCutoff;
        m_visualState.strokeIntensity = resonance;
        m_visualState.strokeAngle = m_lfoPhase * 2.0f * M_PI;
        m_visualState.wetness = mix;

        // Update trail positions (smooth following)
        m_visualState.trailPos[0] += (normalizedCutoff - m_visualState.trailPos[0]) * 0.1f;
        m_visualState.trailPos[1] += (m_visualState.trailPos[0] - m_visualState.trailPos[1]) * 0.1f;
    }
}

// Parameter setters
void BrushProcessor::setCutoff(float cutoffHz)
{
    m_cutoffHz.store(std::clamp(cutoffHz, BrushConfig::MIN_CUTOFF_HZ, BrushConfig::MAX_CUTOFF_HZ));
}

void BrushProcessor::setResonance(float resonance)
{
    m_resonance.store(std::clamp(resonance, BrushConfig::MIN_RESONANCE, BrushConfig::MAX_RESONANCE));
}

void BrushProcessor::setFilterType(FilterType type)
{
    m_filterType = type;
}

void BrushProcessor::setLFORate(float rateHz)
{
    m_lfoRateHz.store(std::clamp(rateHz, 0.01f, BrushConfig::MAX_LFO_RATE_HZ));
}

void BrushProcessor::setLFODepth(float depth)
{
    m_lfoDepth.store(std::clamp(depth, 0.0f, BrushConfig::MAX_LFO_DEPTH));
}

void BrushProcessor::setLFOWaveform(LFOWaveform waveform)
{
    m_lfoWaveform = waveform;
}

void BrushProcessor::setEnvDepth(float depth)
{
    m_envDepth.store(std::clamp(depth, 0.0f, BrushConfig::MAX_ENV_DEPTH));
}

void BrushProcessor::setEnvAttack(float attackMs)
{
    m_envAttackMs.store(std::clamp(attackMs, 0.1f, 1000.0f));
    m_envAttackCoeff = std::exp(-1.0f / (static_cast<float>(m_sampleRate) * attackMs / 1000.0f));
}

void BrushProcessor::setEnvRelease(float releaseMs)
{
    m_envReleaseMs.store(std::clamp(releaseMs, 10.0f, 5000.0f));
    m_envReleaseCoeff = std::exp(-1.0f / (static_cast<float>(m_sampleRate) * releaseMs / 1000.0f));
}

void BrushProcessor::setMix(float mix)
{
    m_mix.store(std::clamp(mix, 0.0f, 1.0f));
}

// Ghost Hands
void BrushProcessor::applyAISuggestion(const juce::String& suggestion)
{
    if (suggestion.containsIgnoreCase("wah") || suggestion.containsIgnoreCase("funky"))
    {
        setFilterType(FilterType::BANDPASS);
        setCutoff(800.0f);
        setResonance(0.7f);
        setLFORate(2.0f);
        setLFODepth(0.8f);
        setLFOWaveform(LFOWaveform::TRIANGLE);
    }
    else if (suggestion.containsIgnoreCase("sweep") || suggestion.containsIgnoreCase("filter sweep"))
    {
        setFilterType(FilterType::LOWPASS);
        setCutoff(500.0f);
        setResonance(0.5f);
        setLFORate(0.1f);
        setLFODepth(1.0f);
        setLFOWaveform(LFOWaveform::TRIANGLE);
    }
    else if (suggestion.containsIgnoreCase("resonant") || suggestion.containsIgnoreCase("acidic"))
    {
        setFilterType(FilterType::LOWPASS);
        setCutoff(1000.0f);
        setResonance(0.9f);
        setEnvDepth(0.8f);
    }
    else if (suggestion.containsIgnoreCase("random") || suggestion.containsIgnoreCase("chaotic"))
    {
        setFilterType(FilterType::BANDPASS);
        setCutoff(2000.0f);
        setResonance(0.6f);
        setLFORate(4.0f);
        setLFODepth(0.7f);
        setLFOWaveform(LFOWaveform::RANDOM_HOLD);
    }
    else if (suggestion.containsIgnoreCase("subtle") || suggestion.containsIgnoreCase("gentle"))
    {
        setFilterType(FilterType::LOWPASS);
        setCutoff(5000.0f);
        setResonance(0.2f);
        setLFORate(0.5f);
        setLFODepth(0.3f);
    }
}

// Visual state
BrushstrokeVisualState BrushProcessor::getVisualState() const
{
    std::lock_guard<std::mutex> lock(m_visualMutex);
    return m_visualState;
}

// Editor
juce::AudioProcessorEditor* BrushProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

// State save/load
void BrushProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    struct State {
        float cutoffHz, resonance, lfoRateHz, lfoDepth;
        float envDepth, envAttackMs, envReleaseMs, mix;
        int filterType, lfoWaveform;
    };

    State state{
        m_cutoffHz.load(),
        m_resonance.load(),
        m_lfoRateHz.load(),
        m_lfoDepth.load(),
        m_envDepth.load(),
        m_envAttackMs.load(),
        m_envReleaseMs.load(),
        m_mix.load(),
        static_cast<int>(m_filterType),
        static_cast<int>(m_lfoWaveform)
    };

    destData.append(&state, sizeof(State));
}

void BrushProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    struct State {
        float cutoffHz, resonance, lfoRateHz, lfoDepth;
        float envDepth, envAttackMs, envReleaseMs, mix;
        int filterType, lfoWaveform;
    };

    if (sizeInBytes >= static_cast<int>(sizeof(State)))
    {
        const State* state = static_cast<const State*>(data);
        setCutoff(state->cutoffHz);
        setResonance(state->resonance);
        setLFORate(state->lfoRateHz);
        setLFODepth(state->lfoDepth);
        setEnvDepth(state->envDepth);
        setEnvAttack(state->envAttackMs);
        setEnvRelease(state->envReleaseMs);
        setMix(state->mix);
        m_filterType = static_cast<FilterType>(state->filterType);
        m_lfoWaveform = static_cast<LFOWaveform>(state->lfoWaveform);
    }
}

} // namespace iDAW
