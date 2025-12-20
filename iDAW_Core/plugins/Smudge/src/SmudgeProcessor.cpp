/**
 * SmudgeProcessor.cpp - Implementation of Plugin 004: "The Smudge"
 *
 * Profile: 'Convolution Reverb' with Scrapbook UI
 *
 * Implements uniformly-partitioned FFT convolution for efficient
 * real-time convolution with arbitrarily long impulse responses.
 */

#include "SmudgeProcessor.h"
#include <cmath>
#include <algorithm>

namespace iDAW {

// Hop size is half the FFT size for 50% overlap
static constexpr int HOP_SIZE = SmudgeConfig::FFT_SIZE / 2;
// Number of complex bins (DC to Nyquist)
static constexpr int NUM_BINS = SmudgeConfig::FFT_SIZE / 2 + 1;
// Window correction factor for 50% overlap Hann window
static constexpr float WINDOW_CORRECTION = 2.0f;

SmudgeProcessor::SmudgeProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    m_fft = std::make_unique<juce::dsp::FFT>(SmudgeConfig::FFT_ORDER);
    m_window = std::make_unique<juce::dsp::WindowingFunction<float>>(
        SmudgeConfig::FFT_SIZE,
        juce::dsp::WindowingFunction<float>::hann
    );

    // Initialize IR library mapping
    m_irLibrary["Cave"] = "IR_Cave.wav";
    m_irLibrary["Room"] = "IR_Studio.wav";
    m_irLibrary["Hall"] = "IR_Hall.wav";
    m_irLibrary["Plate"] = "IR_Plate.wav";
    m_irLibrary["Spring"] = "IR_Spring.wav";
}

SmudgeProcessor::~SmudgeProcessor() = default;

void SmudgeProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    m_sampleRate = sampleRate;

    // Initialize FFT work buffer (needs 2x FFT_SIZE for JUCE's real FFT)
    m_fftWorkBuffer.resize(SmudgeConfig::FFT_SIZE * 2, 0.0f);
    m_inputSpectrum.resize(NUM_BINS);
    m_accumSpectrum.resize(NUM_BINS);

    // Initialize per-channel buffers
    for (int ch = 0; ch < 2; ++ch) {
        m_inputFIFO[ch].resize(SmudgeConfig::FFT_SIZE, 0.0f);
        m_inputFIFOIndex[ch] = 0;
        m_overlapBuffer[ch].resize(SmudgeConfig::FFT_SIZE, 0.0f);
    }

    // Pre-delay buffer
    int maxPreDelaySamples = static_cast<int>(SmudgeConfig::MAX_PREDELAY_MS * sampleRate / 1000.0);
    m_preDelayBuffer.resize(maxPreDelaySamples, 0.0f);
    m_preDelayWriteIndex = 0;

    // High-cut filter
    updateHighCutFilter();

    // Re-prepare IR if one is loaded
    if (!m_currentIR.samples.empty()) {
        prepareIR();
    }

    m_prepared = true;
}

void SmudgeProcessor::releaseResources() {
    for (auto& fifo : m_inputFIFO) fifo.clear();
    for (auto& overlap : m_overlapBuffer) overlap.clear();
    m_fftWorkBuffer.clear();
    m_prepared = false;
}

void SmudgeProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) {
    if (!m_prepared) {
        return;
    }

    juce::ScopedNoDenormals noDenormals;

    const int numSamples = buffer.getNumSamples();
    const int numChannels = std::min(buffer.getNumChannels(), 2);
    float mix = m_mix.load();

    // If no IR loaded, pass through
    if (m_irPartitions.empty() || m_numPartitions == 0) {
        return;
    }

    // Process each channel
    for (int channel = 0; channel < numChannels; ++channel) {
        const float* input = buffer.getReadPointer(channel);
        float* output = buffer.getWritePointer(channel);

        // Create temporary wet buffer
        std::vector<float> wetSignal(numSamples, 0.0f);

        // Process convolution for this channel
        processConvolutionChannel(channel, input, wetSignal.data(), numSamples);

        // Apply pre-delay to wet signal (only for channel 0, copied to others)
        if (channel == 0) {
            int preDelaySamples = static_cast<int>(m_preDelayMs.load() * m_sampleRate / 1000.0);
            preDelaySamples = std::min(preDelaySamples, static_cast<int>(m_preDelayBuffer.size()) - 1);

            for (int i = 0; i < numSamples; ++i) {
                int readIndex = (m_preDelayWriteIndex - preDelaySamples + m_preDelayBuffer.size()) % m_preDelayBuffer.size();
                float delayed = m_preDelayBuffer[readIndex];
                m_preDelayBuffer[m_preDelayWriteIndex] = wetSignal[i];
                m_preDelayWriteIndex = (m_preDelayWriteIndex + 1) % m_preDelayBuffer.size();
                wetSignal[i] = delayed;
            }
        }

        // Mix dry/wet
        for (int i = 0; i < numSamples; ++i) {
            output[i] = input[i] * (1.0f - mix) + wetSignal[i] * mix;
        }
    }

    // Copy left to right if mono input
    if (numChannels == 1 && buffer.getNumChannels() > 1) {
        buffer.copyFrom(1, 0, buffer, 0, 0, numSamples);
    }
}

void SmudgeProcessor::processConvolutionChannel(int channel, const float* input, float* output, int numSamples) {
    auto& inputFIFO = m_inputFIFO[channel];
    auto& overlapBuffer = m_overlapBuffer[channel];
    int& fifoIndex = m_inputFIFOIndex[channel];

    for (int i = 0; i < numSamples; ++i) {
        // Store input in FIFO
        inputFIFO[fifoIndex] = input[i];

        // Output from overlap buffer
        output[i] = overlapBuffer[fifoIndex];

        // Clear the overlap position we just read
        overlapBuffer[fifoIndex] = 0.0f;

        // Advance FIFO index
        fifoIndex++;

        // When we've accumulated HOP_SIZE samples, process an FFT frame
        if (fifoIndex >= HOP_SIZE) {
            fifoIndex = 0;
            processFFTFrame(channel);
        }
    }
}

void SmudgeProcessor::processFFTFrame(int channel) {
    auto& inputFIFO = m_inputFIFO[channel];
    auto& overlapBuffer = m_overlapBuffer[channel];

    // Copy input to FFT work buffer with zero-padding
    // Use the full FIFO (last FFT_SIZE samples, but we only have HOP_SIZE new ones)
    std::fill(m_fftWorkBuffer.begin(), m_fftWorkBuffer.end(), 0.0f);

    // For overlap-save, we need the last FFT_SIZE samples
    // Since we're using 50% overlap, we need previous HOP_SIZE + current HOP_SIZE
    // But our FIFO only stores HOP_SIZE, so we use zero-padding for first half
    // This implements a simplified version - full overlap-add approach
    for (int i = 0; i < HOP_SIZE; ++i) {
        m_fftWorkBuffer[i + HOP_SIZE] = inputFIFO[i];
    }

    // Apply window
    m_window->multiplyWithWindowingTable(m_fftWorkBuffer.data(), SmudgeConfig::FFT_SIZE);

    // Perform forward FFT
    m_fft->performRealOnlyForwardTransform(m_fftWorkBuffer.data());

    // Extract complex spectrum from interleaved real/imag format
    for (int bin = 0; bin < NUM_BINS; ++bin) {
        m_inputSpectrum[bin] = std::complex<float>(
            m_fftWorkBuffer[bin * 2],
            m_fftWorkBuffer[bin * 2 + 1]
        );
    }

    // Shift the FDL (frequency-domain delay line)
    m_fdlIndex = (m_fdlIndex + m_numPartitions - 1) % m_numPartitions;

    // Store current input spectrum in FDL
    m_fdlBuffer[m_fdlIndex] = m_inputSpectrum;

    // Accumulate convolution result: sum of input[k] * IR[k] for all partitions
    std::fill(m_accumSpectrum.begin(), m_accumSpectrum.end(), std::complex<float>(0, 0));

    for (int p = 0; p < m_numPartitions; ++p) {
        int fdlIdx = (m_fdlIndex + p) % m_numPartitions;
        const auto& inputPart = m_fdlBuffer[fdlIdx];
        const auto& irPart = m_irPartitions[p];

        // Complex multiply and accumulate
        for (int bin = 0; bin < NUM_BINS; ++bin) {
            m_accumSpectrum[bin] += inputPart[bin] * irPart[bin];
        }
    }

    // Convert back to interleaved format for IFFT
    for (int bin = 0; bin < NUM_BINS; ++bin) {
        m_fftWorkBuffer[bin * 2] = m_accumSpectrum[bin].real();
        m_fftWorkBuffer[bin * 2 + 1] = m_accumSpectrum[bin].imag();
    }

    // Perform inverse FFT
    m_fft->performRealOnlyInverseTransform(m_fftWorkBuffer.data());

    // Apply window and add to overlap buffer (overlap-add)
    m_window->multiplyWithWindowingTable(m_fftWorkBuffer.data(), SmudgeConfig::FFT_SIZE);

    float scale = WINDOW_CORRECTION / static_cast<float>(SmudgeConfig::FFT_SIZE);

    for (int i = 0; i < SmudgeConfig::FFT_SIZE; ++i) {
        int idx = (i) % SmudgeConfig::FFT_SIZE;
        overlapBuffer[idx] += m_fftWorkBuffer[i] * scale;
    }
}

void SmudgeProcessor::prepareIR() {
    if (m_currentIR.samples.empty()) {
        m_numPartitions = 0;
        m_irPartitions.clear();
        m_fdlBuffer.clear();
        return;
    }

    // Calculate number of partitions needed
    int irLength = static_cast<int>(m_currentIR.samples.size());
    m_numPartitions = (irLength + HOP_SIZE - 1) / HOP_SIZE;

    // Resize partition storage
    m_irPartitions.resize(m_numPartitions);

    // FFT each partition of the IR
    std::vector<float> partitionBuffer(SmudgeConfig::FFT_SIZE * 2, 0.0f);

    for (int p = 0; p < m_numPartitions; ++p) {
        m_irPartitions[p].resize(NUM_BINS);

        // Clear partition buffer
        std::fill(partitionBuffer.begin(), partitionBuffer.end(), 0.0f);

        // Copy IR segment to buffer (zero-padded to FFT_SIZE)
        int startIdx = p * HOP_SIZE;
        int copyLen = std::min(HOP_SIZE, irLength - startIdx);

        if (copyLen > 0) {
            for (int i = 0; i < copyLen; ++i) {
                partitionBuffer[i] = m_currentIR.samples[startIdx + i];
            }
        }

        // FFT the partition
        m_fft->performRealOnlyForwardTransform(partitionBuffer.data());

        // Store as complex
        for (int bin = 0; bin < NUM_BINS; ++bin) {
            m_irPartitions[p][bin] = std::complex<float>(
                partitionBuffer[bin * 2],
                partitionBuffer[bin * 2 + 1]
            );
        }
    }

    // Initialize FDL buffer
    m_fdlBuffer.resize(m_numPartitions);
    for (auto& fdl : m_fdlBuffer) {
        fdl.resize(NUM_BINS, std::complex<float>(0, 0));
    }
    m_fdlIndex = 0;
}

bool SmudgeProcessor::loadIR(const juce::File& file) {
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (!reader) return false;

    // Read IR samples
    int numSamples = static_cast<int>(reader->lengthInSamples);
    numSamples = std::min(numSamples, SmudgeConfig::MAX_IR_LENGTH);

    m_currentIR.samples.resize(numSamples);
    juce::AudioBuffer<float> tempBuffer(1, numSamples);
    reader->read(&tempBuffer, 0, numSamples, 0, true, false);

    std::memcpy(m_currentIR.samples.data(), tempBuffer.getReadPointer(0), numSamples * sizeof(float));
    m_currentIR.sampleRate = static_cast<int>(reader->sampleRate);
    m_currentIR.name = file.getFileNameWithoutExtension();
    m_currentIRName = m_currentIR.name;

    prepareIR();
    return true;
}

bool SmudgeProcessor::loadIRFromLibrary(const juce::String& name) {
    auto it = m_irLibrary.find(name);
    if (it == m_irLibrary.end()) return false;

    // Generate a simple synthetic IR for the library presets
    // In a real implementation, load from bundled resources
    m_currentIR.samples.resize(static_cast<int>(m_sampleRate * 2));  // 2 second IR

    // Different decay characteristics based on space type
    float decayRate = 3.0f;  // Default
    float diffusion = 0.5f;

    if (name == "Cave") {
        decayRate = 1.5f;   // Longer decay
        diffusion = 0.8f;   // More diffuse
    } else if (name == "Room") {
        decayRate = 5.0f;   // Shorter decay
        diffusion = 0.3f;   // Less diffuse
    } else if (name == "Hall") {
        decayRate = 2.0f;
        diffusion = 0.6f;
    } else if (name == "Plate") {
        decayRate = 3.5f;
        diffusion = 0.4f;
    } else if (name == "Spring") {
        decayRate = 4.0f;
        diffusion = 0.2f;   // Metallic, less diffuse
    }

    // Generate exponentially decaying noise
    for (size_t i = 0; i < m_currentIR.samples.size(); ++i) {
        float t = static_cast<float>(i) / static_cast<float>(m_sampleRate);
        float decay = std::exp(-t * decayRate);

        // Mix of random noise and filtered noise for diffusion
        float noise = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);

        // Simple lowpass for diffusion effect
        static float lastNoise = 0.0f;
        noise = noise * (1.0f - diffusion) + lastNoise * diffusion;
        lastNoise = noise;

        m_currentIR.samples[i] = noise * decay * 0.5f;
    }

    // Add initial spike for direct sound
    if (!m_currentIR.samples.empty()) {
        m_currentIR.samples[0] = 0.8f;
    }

    m_currentIR.name = name;
    m_currentIRName = name;

    {
        std::lock_guard<std::mutex> lock(m_visualMutex);
        m_visualState.currentSpace = name;
        m_visualState.tearProgress = 1.0f;  // Trigger tear animation
    }

    prepareIR();
    return true;
}

std::vector<juce::String> SmudgeProcessor::getAvailableIRs() const {
    std::vector<juce::String> names;
    for (const auto& pair : m_irLibrary) {
        names.push_back(pair.first);
    }
    return names;
}

void SmudgeProcessor::applyAISuggestion(const juce::String& spaceName) {
    loadIRFromLibrary(spaceName);
}

void SmudgeProcessor::setMix(float mix) { m_mix.store(std::clamp(mix, 0.0f, 1.0f)); }
void SmudgeProcessor::setDecay(float decay) { m_decay.store(std::clamp(decay, 0.5f, SmudgeConfig::MAX_DECAY)); }
void SmudgeProcessor::setPreDelay(float ms) { m_preDelayMs.store(std::clamp(ms, 0.0f, SmudgeConfig::MAX_PREDELAY_MS)); }
void SmudgeProcessor::setHighCut(float hz) { m_highCutHz.store(std::clamp(hz, 1000.0f, 20000.0f)); updateHighCutFilter(); }

void SmudgeProcessor::updateHighCutFilter() {
    if (m_sampleRate > 0) {
        m_highCutCoeffs = juce::dsp::IIR::Coefficients<float>::makeLowPass(m_sampleRate, m_highCutHz.load());
        m_highCutFilter.coefficients = m_highCutCoeffs;
    }
}

ScrapbookVisualState SmudgeProcessor::getVisualState() const {
    std::lock_guard<std::mutex> lock(m_visualMutex);
    return m_visualState;
}

void SmudgeProcessor::setPhotoCorner(float x, float y) {
    std::lock_guard<std::mutex> lock(m_visualMutex);
    m_visualState.photoCornerX = x;
    m_visualState.photoCornerY = y;

    // Map corner position to decay
    float decay = 0.5f + (x * 0.5f + y * 0.5f) * 2.5f;
    m_decay.store(decay);
}

double SmudgeProcessor::getTailLengthSeconds() const {
    return m_decay.load() * 2.0;
}

void SmudgeProcessor::getStateInformation(juce::MemoryBlock& destData) {
    float mix = m_mix.load();
    float decay = m_decay.load();
    float preDelay = m_preDelayMs.load();
    float highCut = m_highCutHz.load();

    destData.append(&mix, sizeof(float));
    destData.append(&decay, sizeof(float));
    destData.append(&preDelay, sizeof(float));
    destData.append(&highCut, sizeof(float));
}

void SmudgeProcessor::setStateInformation(const void* data, int sizeInBytes) {
    if (sizeInBytes >= 4 * sizeof(float)) {
        const float* floatData = static_cast<const float*>(data);
        m_mix.store(floatData[0]);
        m_decay.store(floatData[1]);
        m_preDelayMs.store(floatData[2]);
        m_highCutHz.store(floatData[3]);
    }
}

juce::AudioProcessorEditor* SmudgeProcessor::createEditor() {
    return new juce::GenericAudioProcessorEditor(*this);
}

} // namespace iDAW

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new iDAW::SmudgeProcessor();
}
