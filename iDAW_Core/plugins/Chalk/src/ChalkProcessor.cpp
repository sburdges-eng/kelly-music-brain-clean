/**
 * ChalkProcessor.cpp - Lo-fi/Bitcrusher Implementation
 */

#include "../include/ChalkProcessor.h"

namespace iDAW {

ChalkProcessor::ChalkProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      m_rng(std::random_device{}())
{
    m_holdSample.fill(0.0f);
    m_holdCounter.fill(0.0f);
    m_lpfState.fill(0.0f);
    m_visualState.frequencyBins.fill(0.0f);
}

ChalkProcessor::~ChalkProcessor() = default;

void ChalkProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    m_sampleRate = sampleRate;
    m_holdSample.fill(0.0f);
    m_holdCounter.fill(0.0f);
    m_lpfState.fill(0.0f);
    m_wowPhase = 0.0f;
    m_flutterPhase = 0.0f;
    m_prepared = true;
}

void ChalkProcessor::releaseResources()
{
    m_prepared = false;
}

float ChalkProcessor::quantize(float sample, int bits)
{
    if (bits >= 16)
        return sample;

    // Calculate number of quantization levels
    float levels = std::pow(2.0f, static_cast<float>(bits));
    float halfLevels = levels / 2.0f;

    // Quantize
    float quantized = std::round(sample * halfLevels) / halfLevels;
    return std::clamp(quantized, -1.0f, 1.0f);
}

float ChalkProcessor::generateNoise()
{
    // Pink-ish noise (filtered white noise)
    static float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;

    float white = m_noiseDist(m_rng);

    // Simple pink noise approximation
    b0 = 0.99765f * b0 + white * 0.0990460f;
    b1 = 0.96300f * b1 + white * 0.2965164f;
    b2 = 0.57000f * b2 + white * 1.0526913f;

    float pink = (b0 + b1 + b2 + white * 0.1848f) * 0.11f;
    return pink;
}

float ChalkProcessor::generateCrackle()
{
    // Vinyl crackle: occasional pops and continuous dust
    float crackle = 0.0f;

    // Random pops (sparse)
    if (m_crackleDist(m_rng) > 0.9998f)
    {
        crackle = m_noiseDist(m_rng) * 0.5f;
    }

    // Continuous dust (light noise)
    if (m_crackleDist(m_rng) > 0.995f)
    {
        crackle += m_noiseDist(m_rng) * 0.1f;
    }

    return crackle;
}

float ChalkProcessor::processLowpass(float sample, int channel)
{
    // One-pole low-pass filter
    // Cutoff frequency based on warmth (1 = 2kHz, 0 = 20kHz)
    float warmth = m_warmth.load();
    float cutoffHz = 20000.0f * std::pow(0.1f, warmth);  // 20kHz to 2kHz

    // Calculate coefficient
    float rc = 1.0f / (2.0f * M_PI * cutoffHz);
    float dt = 1.0f / static_cast<float>(m_sampleRate);
    float alpha = dt / (rc + dt);

    m_lpfState[channel] = m_lpfState[channel] + alpha * (sample - m_lpfState[channel]);
    return m_lpfState[channel];
}

void ChalkProcessor::updateLFO()
{
    // Wow (slow pitch modulation)
    float wowRate = 0.5f;  // 0.5 Hz
    m_wowPhase += wowRate / static_cast<float>(m_sampleRate);
    if (m_wowPhase >= 1.0f)
        m_wowPhase -= 1.0f;

    // Flutter (faster pitch modulation)
    float flutterRate = m_flutterRate.load();
    m_flutterPhase += flutterRate / static_cast<float>(m_sampleRate);
    if (m_flutterPhase >= 1.0f)
        m_flutterPhase -= 1.0f;
}

void ChalkProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                   juce::MidiBuffer& /*midiMessages*/)
{
    if (!m_prepared)
        return;

    const int numSamples = buffer.getNumSamples();
    const int numChannels = std::min(buffer.getNumChannels(), 2);

    // Get parameters
    const int bitDepth = m_bitDepth.load();
    const float srReduction = m_srReduction.load();
    const float noiseLevel = m_noiseLevel.load();
    const float crackleLevel = m_crackleLevel.load();
    const float mix = m_mix.load();

    float maxLevel = 0.0f;

    for (int sample = 0; sample < numSamples; ++sample)
    {
        updateLFO();

        // Generate noise and crackle once per sample (mono)
        float noise = generateNoise() * noiseLevel;
        float crackle = generateCrackle() * crackleLevel;

        for (int ch = 0; ch < numChannels; ++ch)
        {
            float dry = buffer.getSample(ch, sample);
            float wet = dry;

            // Sample rate reduction (sample and hold)
            m_holdCounter[ch] += 1.0f;
            if (m_holdCounter[ch] >= srReduction)
            {
                m_holdSample[ch] = wet;
                m_holdCounter[ch] = 0.0f;
            }
            wet = m_holdSample[ch];

            // Bit depth reduction
            wet = quantize(wet, bitDepth);

            // Add noise
            wet += noise;

            // Add crackle
            wet += crackle;

            // Apply warmth (low-pass filter)
            wet = processLowpass(wet, ch);

            // Clamp
            wet = std::clamp(wet, -1.0f, 1.0f);

            // Mix
            float output = dry * (1.0f - mix) + wet * mix;
            buffer.setSample(ch, sample, output);

            maxLevel = std::max(maxLevel, std::abs(output));
        }
    }

    // Update visual state
    {
        std::lock_guard<std::mutex> lock(m_visualMutex);
        m_visualState.chalkDensity = static_cast<float>(16 - bitDepth) / 15.0f;
        m_visualState.smearAmount = (srReduction - 1.0f) / 99.0f;
        m_visualState.noiseFloor = noiseLevel + crackleLevel;
        m_visualState.particleCount = noiseLevel * 100.0f;
    }
}

// Parameter setters
void ChalkProcessor::setBitDepth(int bits)
{
    m_bitDepth.store(std::clamp(bits, ChalkConfig::MIN_BIT_DEPTH, ChalkConfig::MAX_BIT_DEPTH));
}

void ChalkProcessor::setSampleRateReduction(float factor)
{
    m_srReduction.store(std::clamp(factor, ChalkConfig::MIN_SAMPLE_RATE_REDUCTION, ChalkConfig::MAX_SAMPLE_RATE_REDUCTION));
}

void ChalkProcessor::setNoiseLevel(float level)
{
    m_noiseLevel.store(std::clamp(level, 0.0f, ChalkConfig::MAX_NOISE_LEVEL));
}

void ChalkProcessor::setCrackleLevel(float level)
{
    m_crackleLevel.store(std::clamp(level, 0.0f, ChalkConfig::MAX_CRACKLE));
}

void ChalkProcessor::setWarmth(float warmth)
{
    m_warmth.store(std::clamp(warmth, 0.0f, 1.0f));
}

void ChalkProcessor::setWowDepth(float depthCents)
{
    m_wowDepth.store(std::clamp(depthCents, 0.0f, ChalkConfig::MAX_WOW_DEPTH));
}

void ChalkProcessor::setFlutterRate(float rateHz)
{
    m_flutterRate.store(std::clamp(rateHz, 0.1f, ChalkConfig::MAX_FLUTTER_RATE));
}

void ChalkProcessor::setMix(float mix)
{
    m_mix.store(std::clamp(mix, 0.0f, 1.0f));
}

// Presets
void ChalkProcessor::applyPreset(LofiPreset preset)
{
    switch (preset)
    {
        case LofiPreset::CLEAN:
            setBitDepth(16);
            setSampleRateReduction(1.0f);
            setNoiseLevel(0.0f);
            setCrackleLevel(0.0f);
            setWarmth(0.0f);
            break;

        case LofiPreset::CASSETTE:
            setBitDepth(14);
            setSampleRateReduction(1.0f);
            setNoiseLevel(0.02f);
            setCrackleLevel(0.0f);
            setWarmth(0.3f);
            setWowDepth(5.0f);
            break;

        case LofiPreset::VINYL:
            setBitDepth(16);
            setSampleRateReduction(1.0f);
            setNoiseLevel(0.01f);
            setCrackleLevel(0.3f);
            setWarmth(0.4f);
            break;

        case LofiPreset::TELEPHONE:
            setBitDepth(8);
            setSampleRateReduction(5.5f);
            setNoiseLevel(0.05f);
            setCrackleLevel(0.0f);
            setWarmth(0.7f);
            break;

        case LofiPreset::RADIO:
            setBitDepth(10);
            setSampleRateReduction(3.0f);
            setNoiseLevel(0.04f);
            setCrackleLevel(0.1f);
            setWarmth(0.5f);
            break;

        case LofiPreset::CHIPTUNE:
            setBitDepth(4);
            setSampleRateReduction(8.0f);
            setNoiseLevel(0.0f);
            setCrackleLevel(0.0f);
            setWarmth(0.0f);
            break;

        case LofiPreset::CUSTOM:
        default:
            break;
    }
}

// Ghost Hands
void ChalkProcessor::applyAISuggestion(const juce::String& suggestion)
{
    if (suggestion.containsIgnoreCase("cassette") || suggestion.containsIgnoreCase("tape"))
    {
        applyPreset(LofiPreset::CASSETTE);
    }
    else if (suggestion.containsIgnoreCase("vinyl") || suggestion.containsIgnoreCase("record"))
    {
        applyPreset(LofiPreset::VINYL);
    }
    else if (suggestion.containsIgnoreCase("telephone") || suggestion.containsIgnoreCase("phone"))
    {
        applyPreset(LofiPreset::TELEPHONE);
    }
    else if (suggestion.containsIgnoreCase("radio") || suggestion.containsIgnoreCase("am"))
    {
        applyPreset(LofiPreset::RADIO);
    }
    else if (suggestion.containsIgnoreCase("8-bit") || suggestion.containsIgnoreCase("chiptune") ||
             suggestion.containsIgnoreCase("retro"))
    {
        applyPreset(LofiPreset::CHIPTUNE);
    }
    else if (suggestion.containsIgnoreCase("clean") || suggestion.containsIgnoreCase("pristine"))
    {
        applyPreset(LofiPreset::CLEAN);
    }
    else if (suggestion.containsIgnoreCase("lofi") || suggestion.containsIgnoreCase("lo-fi"))
    {
        // Generic lo-fi preset
        setBitDepth(12);
        setSampleRateReduction(2.0f);
        setNoiseLevel(0.03f);
        setCrackleLevel(0.1f);
        setWarmth(0.5f);
    }
}

// Visual state
DustyVisualState ChalkProcessor::getVisualState() const
{
    std::lock_guard<std::mutex> lock(m_visualMutex);
    return m_visualState;
}

// Editor
juce::AudioProcessorEditor* ChalkProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

// State save/load
void ChalkProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    struct State {
        int bitDepth;
        float srReduction, noiseLevel, crackleLevel, warmth;
        float wowDepth, flutterRate, mix;
    };

    State state{
        m_bitDepth.load(),
        m_srReduction.load(),
        m_noiseLevel.load(),
        m_crackleLevel.load(),
        m_warmth.load(),
        m_wowDepth.load(),
        m_flutterRate.load(),
        m_mix.load()
    };

    destData.append(&state, sizeof(State));
}

void ChalkProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    struct State {
        int bitDepth;
        float srReduction, noiseLevel, crackleLevel, warmth;
        float wowDepth, flutterRate, mix;
    };

    if (sizeInBytes >= static_cast<int>(sizeof(State)))
    {
        const State* state = static_cast<const State*>(data);
        setBitDepth(state->bitDepth);
        setSampleRateReduction(state->srReduction);
        setNoiseLevel(state->noiseLevel);
        setCrackleLevel(state->crackleLevel);
        setWarmth(state->warmth);
        setWowDepth(state->wowDepth);
        setFlutterRate(state->flutterRate);
        setMix(state->mix);
    }
}

} // namespace iDAW
