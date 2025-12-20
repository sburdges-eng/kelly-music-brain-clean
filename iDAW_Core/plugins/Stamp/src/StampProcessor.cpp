/**
 * StampProcessor.cpp - Stutter/Repeater Implementation
 */

#include "../include/StampProcessor.h"
#include <random>

namespace iDAW {

StampProcessor::StampProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      m_rng(std::random_device{}())
{
    m_writeIndex.fill(0);
    m_playbackPosition.fill(0.0f);
}

StampProcessor::~StampProcessor() = default;

void StampProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    m_sampleRate = sampleRate;

    // Allocate capture buffers
    int bufferSize = static_cast<int>(sampleRate * 2.0);  // 2 seconds
    bufferSize = std::min(bufferSize, StampConfig::MAX_BUFFER_SIZE);

    for (auto& buffer : m_captureBuffer)
    {
        buffer.resize(bufferSize, 0.0f);
    }

    m_writeIndex.fill(0);
    m_playbackPosition.fill(0.0f);
    m_capturedLength = 0;
    m_capturing = true;
    m_repeatPhase = 0.0f;
    m_repeatCount = 0;
    m_currentGain = 1.0f;
    m_playingForward = true;

    m_prepared = true;
}

void StampProcessor::releaseResources()
{
    m_prepared = false;
}

double StampProcessor::getTailLengthSeconds() const
{
    return m_sliceLengthMs.load() / 1000.0 * 4.0;  // Allow for several repeats
}

void StampProcessor::captureToBuffer(const float* input, int channel, int numSamples)
{
    auto& buffer = m_captureBuffer[channel];
    int bufferSize = static_cast<int>(buffer.size());

    for (int i = 0; i < numSamples; ++i)
    {
        buffer[m_writeIndex[channel]] = input[i];
        m_writeIndex[channel] = (m_writeIndex[channel] + 1) % bufferSize;
    }
}

float StampProcessor::readFromBuffer(int channel, float position)
{
    auto& buffer = m_captureBuffer[channel];
    int bufferSize = static_cast<int>(buffer.size());

    // Wrap position
    while (position < 0)
        position += bufferSize;
    while (position >= bufferSize)
        position -= bufferSize;

    // Linear interpolation
    int index0 = static_cast<int>(position);
    int index1 = (index0 + 1) % bufferSize;
    float frac = position - static_cast<float>(index0);

    return buffer[index0] * (1.0f - frac) + buffer[index1] * frac;
}

float StampProcessor::getSyncedRepeatRate(double bpm)
{
    if (m_sync == StampSync::FREE)
        return m_repeatRateHz.load();

    // Calculate Hz from note division
    float beatsPerSecond = static_cast<float>(bpm) / 60.0f;

    switch (m_sync)
    {
        case StampSync::WHOLE:
            return beatsPerSecond / 4.0f;
        case StampSync::HALF:
            return beatsPerSecond / 2.0f;
        case StampSync::QUARTER:
            return beatsPerSecond;
        case StampSync::EIGHTH:
            return beatsPerSecond * 2.0f;
        case StampSync::SIXTEENTH:
            return beatsPerSecond * 4.0f;
        case StampSync::THIRTY_SECOND:
            return beatsPerSecond * 8.0f;
        case StampSync::TRIPLET_QUARTER:
            return beatsPerSecond * 1.5f;
        case StampSync::TRIPLET_EIGHTH:
            return beatsPerSecond * 3.0f;
        default:
            return m_repeatRateHz.load();
    }
}

void StampProcessor::updateFromPlayHead(juce::AudioPlayHead* playHead)
{
    if (playHead != nullptr)
    {
        if (auto posInfo = playHead->getPosition())
        {
            if (posInfo->getBpm().hasValue())
            {
                m_hostBPM = *posInfo->getBpm();
            }
        }
    }
}

void StampProcessor::trigger()
{
    m_isActive.store(true);
    m_capturing = false;

    // Calculate slice length in samples
    float sliceLengthMs = m_sliceLengthMs.load();
    m_capturedLength = static_cast<int>(m_sampleRate * sliceLengthMs / 1000.0f);
    m_capturedLength = std::min(m_capturedLength, static_cast<int>(m_captureBuffer[0].size()));

    // Reset playback
    m_playbackPosition.fill(0.0f);
    m_repeatPhase = 0.0f;
    m_repeatCount = 0;
    m_currentGain = 1.0f;
    m_playingForward = true;

    // For random mode, pick a random start position
    if (m_repeatMode == RepeatMode::RANDOM)
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        m_randomSliceStart = dist(m_rng);
    }
}

void StampProcessor::release()
{
    m_isActive.store(false);
    m_capturing = true;
}

void StampProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                   juce::MidiBuffer& midiMessages)
{
    if (!m_prepared)
        return;

    updateFromPlayHead(getPlayHead());

    const int numSamples = buffer.getNumSamples();
    const int numChannels = std::min(buffer.getNumChannels(), 2);

    // Handle MIDI triggers
    for (const auto& metadata : midiMessages)
    {
        auto message = metadata.getMessage();
        if (message.isNoteOn())
            trigger();
        else if (message.isNoteOff())
            release();
    }

    // Get parameters
    const float repeatRate = getSyncedRepeatRate(m_hostBPM);
    const float decay = m_decay.load();
    const float pitchShift = m_pitchShift.load();
    const float mix = m_mix.load();
    const bool isActive = m_isActive.load();

    // Calculate playback speed from pitch shift
    float playbackSpeed = std::pow(2.0f, pitchShift / 12.0f);

    // Calculate repeat interval in samples
    float repeatIntervalSamples = static_cast<float>(m_sampleRate) / repeatRate;

    for (int sample = 0; sample < numSamples; ++sample)
    {
        float wetL = 0.0f, wetR = 0.0f;

        if (isActive && m_capturedLength > 0)
        {
            // Read from captured buffer
            int bufferSize = static_cast<int>(m_captureBuffer[0].size());

            // Calculate read position based on mode
            float readPosL = 0.0f, readPosR = 0.0f;

            switch (m_repeatMode)
            {
                case RepeatMode::NORMAL:
                    readPosL = static_cast<float>(m_writeIndex[0]) - m_capturedLength + m_playbackPosition[0];
                    readPosR = static_cast<float>(m_writeIndex[1]) - m_capturedLength + m_playbackPosition[1];
                    break;

                case RepeatMode::REVERSE:
                    readPosL = static_cast<float>(m_writeIndex[0]) - m_playbackPosition[0];
                    readPosR = static_cast<float>(m_writeIndex[1]) - m_playbackPosition[1];
                    break;

                case RepeatMode::PING_PONG:
                    if (m_playingForward)
                    {
                        readPosL = static_cast<float>(m_writeIndex[0]) - m_capturedLength + m_playbackPosition[0];
                        readPosR = static_cast<float>(m_writeIndex[1]) - m_capturedLength + m_playbackPosition[1];
                    }
                    else
                    {
                        readPosL = static_cast<float>(m_writeIndex[0]) - m_playbackPosition[0];
                        readPosR = static_cast<float>(m_writeIndex[1]) - m_playbackPosition[1];
                    }
                    break;

                case RepeatMode::RANDOM:
                {
                    float startOffset = m_randomSliceStart * (bufferSize - m_capturedLength);
                    readPosL = startOffset + m_playbackPosition[0];
                    readPosR = startOffset + m_playbackPosition[1];
                    break;
                }
            }

            wetL = readFromBuffer(0, readPosL) * m_currentGain;
            wetR = (numChannels > 1) ? readFromBuffer(1, readPosR) * m_currentGain : wetL;

            // Advance playback position
            m_playbackPosition[0] += playbackSpeed;
            m_playbackPosition[1] += playbackSpeed;

            // Check for repeat
            if (m_playbackPosition[0] >= m_capturedLength)
            {
                m_playbackPosition.fill(0.0f);
                m_currentGain *= decay;
                m_repeatCount++;

                // Update mode-specific state
                if (m_repeatMode == RepeatMode::PING_PONG)
                {
                    m_playingForward = !m_playingForward;
                }
                else if (m_repeatMode == RepeatMode::RANDOM)
                {
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    m_randomSliceStart = dist(m_rng);
                }

                // Stop if gain is too low
                if (m_currentGain < 0.001f)
                {
                    release();
                }
            }
        }
        else
        {
            // Capture mode - record input
            for (int ch = 0; ch < numChannels; ++ch)
            {
                const float* channelData = buffer.getReadPointer(ch);
                captureToBuffer(channelData + sample, ch, 1);
            }
        }

        // Mix and output
        for (int ch = 0; ch < numChannels; ++ch)
        {
            float dry = buffer.getSample(ch, sample);
            float wet = (ch == 0) ? wetL : wetR;
            float output = dry * (1.0f - mix) + wet * mix;

            // Also keep capturing while playing if we're doing the dry signal
            if (isActive && m_capturing)
            {
                captureToBuffer(&dry, ch, 1);
            }

            buffer.setSample(ch, sample, output);
        }
    }

    // Update visual state
    {
        std::lock_guard<std::mutex> lock(m_visualMutex);
        m_visualState.isStamping = isActive;
        m_visualState.stampProgress = m_capturedLength > 0 ?
            m_playbackPosition[0] / m_capturedLength : 0.0f;
        m_visualState.inkIntensity = m_currentGain;
        m_visualState.stampCount = m_repeatCount;
        m_visualState.pressureLevel = mix;
    }
}

// Parameter setters
void StampProcessor::setRepeatRate(float rateHz)
{
    m_repeatRateHz.store(std::clamp(rateHz, StampConfig::MIN_REPEAT_RATE, StampConfig::MAX_REPEAT_RATE));
}

void StampProcessor::setSliceLength(float lengthMs)
{
    float maxLengthMs = static_cast<float>(StampConfig::MAX_BUFFER_SIZE) / static_cast<float>(m_sampleRate) * 1000.0f;
    m_sliceLengthMs.store(std::clamp(lengthMs, 10.0f, maxLengthMs));
}

void StampProcessor::setDecay(float decay)
{
    m_decay.store(std::clamp(decay, 0.0f, StampConfig::MAX_DECAY));
}

void StampProcessor::setPitchShift(float semitones)
{
    m_pitchShift.store(std::clamp(semitones, -StampConfig::MAX_PITCH_SHIFT, StampConfig::MAX_PITCH_SHIFT));
}

void StampProcessor::setRepeatMode(RepeatMode mode)
{
    m_repeatMode = mode;
}

void StampProcessor::setSync(StampSync sync)
{
    m_sync = sync;
}

void StampProcessor::setMix(float mix)
{
    m_mix.store(std::clamp(mix, 0.0f, 1.0f));
}

void StampProcessor::setActive(bool active)
{
    if (active)
        trigger();
    else
        release();
}

// Ghost Hands
void StampProcessor::applyAISuggestion(const juce::String& suggestion)
{
    if (suggestion.containsIgnoreCase("stutter") || suggestion.containsIgnoreCase("glitch"))
    {
        setRepeatRate(16.0f);
        setSliceLength(62.5f);
        setDecay(0.9f);
        setRepeatMode(RepeatMode::NORMAL);
    }
    else if (suggestion.containsIgnoreCase("tape stop"))
    {
        setRepeatRate(4.0f);
        setSliceLength(250.0f);
        setPitchShift(-12.0f);
        setDecay(0.5f);
    }
    else if (suggestion.containsIgnoreCase("beatrepeat") || suggestion.containsIgnoreCase("beat repeat"))
    {
        setSync(StampSync::EIGHTH);
        setDecay(0.95f);
        setRepeatMode(RepeatMode::NORMAL);
    }
    else if (suggestion.containsIgnoreCase("reverse") || suggestion.containsIgnoreCase("backwards"))
    {
        setRepeatMode(RepeatMode::REVERSE);
        setDecay(0.85f);
    }
    else if (suggestion.containsIgnoreCase("random") || suggestion.containsIgnoreCase("glitchy"))
    {
        setRepeatMode(RepeatMode::RANDOM);
        setRepeatRate(8.0f);
        setSliceLength(100.0f);
    }
    else if (suggestion.containsIgnoreCase("subtle"))
    {
        setRepeatRate(2.0f);
        setDecay(0.6f);
        setMix(0.5f);
    }
}

// Visual state
RubberStampVisualState StampProcessor::getVisualState() const
{
    std::lock_guard<std::mutex> lock(m_visualMutex);
    return m_visualState;
}

// Editor
juce::AudioProcessorEditor* StampProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

// State save/load
void StampProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    struct State {
        float repeatRateHz, sliceLengthMs, decay, pitchShift, mix;
        int repeatMode, sync;
    };

    State state{
        m_repeatRateHz.load(),
        m_sliceLengthMs.load(),
        m_decay.load(),
        m_pitchShift.load(),
        m_mix.load(),
        static_cast<int>(m_repeatMode),
        static_cast<int>(m_sync)
    };

    destData.append(&state, sizeof(State));
}

void StampProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    struct State {
        float repeatRateHz, sliceLengthMs, decay, pitchShift, mix;
        int repeatMode, sync;
    };

    if (sizeInBytes >= static_cast<int>(sizeof(State)))
    {
        const State* state = static_cast<const State*>(data);
        setRepeatRate(state->repeatRateHz);
        setSliceLength(state->sliceLengthMs);
        setDecay(state->decay);
        setPitchShift(state->pitchShift);
        setMix(state->mix);
        m_repeatMode = static_cast<RepeatMode>(state->repeatMode);
        m_sync = static_cast<StampSync>(state->sync);
    }
}

} // namespace iDAW
