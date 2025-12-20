/**
 * @file AudioFile.cpp
 * @brief Audio file implementation
 *
 * NOTE: This is a stub implementation that works without libsndfile.
 * For production use, integrate libsndfile for robust file I/O.
 *
 * Current implementation:
 * - Generates test data (sine waves)
 * - Placeholder read/write methods
 * - Basic WAV header writing (simple PCM only)
 */

#include "daiw/audio/AudioFile.h"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace daiw {
namespace audio {

// Simple WAV header structure
struct WAVHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t fileSize = 0;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmtSize = 16;
    uint16_t audioFormat = 3;  // 3 = IEEE float
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint32_t byteRate = 0;
    uint16_t blockAlign = 0;
    uint16_t bitsPerSample = 32;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize = 0;
};

struct AudioFile::Impl {
    // Placeholder for libsndfile handle
    // In full implementation: SNDFILE* sndfile = nullptr;
};

AudioFile::AudioFile() : impl_(std::make_unique<Impl>()) {}

AudioFile::~AudioFile() = default;

AudioFile::AudioFile(AudioFile&&) noexcept = default;
AudioFile& AudioFile::operator=(AudioFile&&) noexcept = default;

bool AudioFile::read(const std::string& filepath) {
    // Basic WAV file reading implementation
    // For production: integrate libsndfile for robust multi-format support
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read RIFF header
    char riff[4];
    file.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) {
        return false;  // Not a RIFF file
    }
    
    uint32_t fileSize;
    file.read(reinterpret_cast<char*>(&fileSize), sizeof(fileSize));
    
    // Read WAVE identifier
    char wave[4];
    file.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0) {
        return false;  // Not a WAV file
    }

    // Find and read fmt chunk
    // NOTE: WAV chunks can appear in any order, so we must find fmt before processing data
    WAVHeader header;
    bool foundFmt = false;
    bool foundData = false;
    std::streampos dataChunkPos = 0;
    uint32_t dataChunkSize = 0;
    
    // First pass: find fmt chunk and record data chunk position
    while (!file.eof() && (!foundFmt || !foundData)) {
        char chunkId[4];
        file.read(chunkId, 4);
        if (file.eof()) break;
        
        uint32_t chunkSize;
        file.read(reinterpret_cast<char*>(&chunkSize), sizeof(chunkSize));
        
        if (std::strncmp(chunkId, "fmt ", 4) == 0) {
            // Read fmt chunk - must be processed first for format info
            file.read(reinterpret_cast<char*>(&header.fmtSize), sizeof(header.fmtSize));
            file.read(reinterpret_cast<char*>(&header.audioFormat), sizeof(header.audioFormat));
            file.read(reinterpret_cast<char*>(&header.numChannels), sizeof(header.numChannels));
            file.read(reinterpret_cast<char*>(&header.sampleRate), sizeof(header.sampleRate));
            file.read(reinterpret_cast<char*>(&header.byteRate), sizeof(header.byteRate));
            file.read(reinterpret_cast<char*>(&header.blockAlign), sizeof(header.blockAlign));
            file.read(reinterpret_cast<char*>(&header.bitsPerSample), sizeof(header.bitsPerSample));
            
            // Skip any extra fmt data
            if (chunkSize > 16) {
                file.seekg(chunkSize - 16, std::ios::cur);
            }
            foundFmt = true;
        } else if (std::strncmp(chunkId, "data", 4) == 0) {
            // Record data chunk position - we'll read it after fmt is found
            dataChunkSize = chunkSize;
            dataChunkPos = file.tellg();
            foundData = true;
            // Skip past data chunk for now (in case fmt comes after)
            file.seekg(chunkSize, std::ios::cur);
        } else {
            // Skip unknown chunks
            file.seekg(chunkSize, std::ios::cur);
        }
    }
    
    // Verify we found both required chunks
    if (!foundFmt || !foundData) {
        return false;
    }
    
    // Validate format values before using them
    if (header.numChannels == 0 || header.sampleRate == 0 || header.bitsPerSample == 0) {
        return false;  // Invalid format chunk data
    }
    
    // Now read the data chunk with valid format information
    file.seekg(dataChunkPos);
    header.dataSize = dataChunkSize;
    
    // Set file info using validated format data
    info_.format = AudioFormat::WAV;
    info_.sampleRate = header.sampleRate;
    info_.numChannels = header.numChannels;
    info_.sampleFormat = (header.bitsPerSample == 32) ? SampleFormat::Float32 : SampleFormat::Int16;
    
    size_t bytesPerSample = header.bitsPerSample / 8;
    size_t totalSamples = header.dataSize / (header.numChannels * bytesPerSample);
    info_.numSamples = totalSamples;
    info_.durationSeconds = static_cast<double>(totalSamples) / info_.sampleRate;
    
    // Read audio data with validated format
    if (header.bitsPerSample == 32 && header.audioFormat == 3) {
        // IEEE float
        data_.resize(totalSamples * header.numChannels);
        file.read(reinterpret_cast<char*>(data_.data()), header.dataSize);
    } else if (header.bitsPerSample == 16) {
        // 16-bit integer - convert to float
        std::vector<int16_t> intData(totalSamples * header.numChannels);
        file.read(reinterpret_cast<char*>(intData.data()), header.dataSize);
        data_.resize(intData.size());
        for (size_t i = 0; i < intData.size(); ++i) {
            data_[i] = static_cast<float>(intData[i]) / 32768.0f;
        }
    } else {
        // Unsupported format
        return false;
    }
    
    return true;
}

bool AudioFile::write(const std::string& filepath,
                     AudioFormat format,
                     SampleFormat sampleFormat) {
    // Basic WAV file writing implementation
    // For production: integrate libsndfile for multi-format support (AIFF, FLAC, etc.)
    
    if (format != AudioFormat::WAV) {
        // Only WAV format supported in basic implementation
        // For other formats, integrate libsndfile
        return false;
    }

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Determine output format
    uint16_t bitsPerSample = 32;
    uint16_t audioFormat = 3;  // IEEE float
    
    if (sampleFormat == SampleFormat::Int16) {
        bitsPerSample = 16;
        audioFormat = 1;  // PCM
    }

    // Calculate sizes
    uint32_t dataSize = static_cast<uint32_t>(data_.size() * sizeof(Sample));
    if (sampleFormat == SampleFormat::Int16) {
        dataSize = static_cast<uint32_t>(data_.size() * sizeof(int16_t));
    }
    
    uint16_t blockAlign = info_.numChannels * (bitsPerSample / 8);
    uint32_t byteRate = info_.sampleRate * blockAlign;
    uint32_t fileSize = 36 + dataSize;

    // Write RIFF header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&fileSize), sizeof(fileSize));
    file.write("WAVE", 4);
    
    // Write fmt chunk
    file.write("fmt ", 4);
    uint32_t fmtSize = 16;
    file.write(reinterpret_cast<const char*>(&fmtSize), sizeof(fmtSize));
    file.write(reinterpret_cast<const char*>(&audioFormat), sizeof(audioFormat));
    file.write(reinterpret_cast<const char*>(&info_.numChannels), sizeof(info_.numChannels));
    file.write(reinterpret_cast<const char*>(&info_.sampleRate), sizeof(info_.sampleRate));
    file.write(reinterpret_cast<const char*>(&byteRate), sizeof(byteRate));
    file.write(reinterpret_cast<const char*>(&blockAlign), sizeof(blockAlign));
    file.write(reinterpret_cast<const char*>(&bitsPerSample), sizeof(bitsPerSample));
    
    // Write data chunk
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));
    
    // Write audio data
    if (sampleFormat == SampleFormat::Float32) {
        file.write(reinterpret_cast<const char*>(data_.data()), dataSize);
    } else if (sampleFormat == SampleFormat::Int16) {
        // Convert float to 16-bit integer
        std::vector<int16_t> intData(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            float clamped = std::max(-1.0f, std::min(1.0f, data_[i]));
            intData[i] = static_cast<int16_t>(clamped * 32767.0f);
        }
        file.write(reinterpret_cast<const char*>(intData.data()), dataSize);
    }

    return file.good();
}

void AudioFile::setData(const std::vector<Sample>& data,
                       uint32_t numChannels,
                       SampleRate sampleRate) {
    data_ = data;
    info_.numChannels = numChannels;
    info_.sampleRate = sampleRate;
    info_.numSamples = data.size() / numChannels;
    info_.durationSeconds = static_cast<double>(info_.numSamples) / sampleRate;
    info_.format = AudioFormat::WAV;
    info_.sampleFormat = SampleFormat::Float32;
}

std::vector<Sample> AudioFile::getChannelData(uint32_t channel) const {
    if (channel >= info_.numChannels) {
        return {};
    }

    std::vector<Sample> channelData;
    channelData.reserve(info_.numSamples);

    for (size_t i = channel; i < data_.size(); i += info_.numChannels) {
        channelData.push_back(data_[i]);
    }

    return channelData;
}

void AudioFile::setChannelData(const std::vector<std::vector<Sample>>& channels,
                              SampleRate sampleRate) {
    if (channels.empty()) {
        return;
    }

    info_.numChannels = static_cast<uint32_t>(channels.size());
    info_.numSamples = channels[0].size();
    info_.sampleRate = sampleRate;
    info_.durationSeconds = static_cast<double>(info_.numSamples) / sampleRate;
    info_.format = AudioFormat::WAV;
    info_.sampleFormat = SampleFormat::Float32;

    // Interleave channels
    data_.resize(info_.numSamples * info_.numChannels);
    for (size_t sample = 0; sample < info_.numSamples; ++sample) {
        for (size_t ch = 0; ch < info_.numChannels; ++ch) {
            data_[sample * info_.numChannels + ch] = channels[ch][sample];
        }
    }
}

bool AudioFile::convertSampleRate(SampleRate targetRate) {
    // Basic sample rate conversion using linear interpolation
    // For production: use libsamplerate or libresample for higher quality
    
    if (targetRate == info_.sampleRate) {
        return true;  // No conversion needed
    }
    
    if (data_.empty() || info_.numChannels == 0 || info_.numSamples == 0) {
        return false;  // No data to convert
    }
    
    double ratio = static_cast<double>(targetRate) / static_cast<double>(info_.sampleRate);
    size_t newNumSamples = static_cast<size_t>(std::ceil(info_.numSamples * ratio));
    
    // Ensure we have at least one output sample
    if (newNumSamples == 0) {
        newNumSamples = 1;
    }
    
    std::vector<Sample> newData(newNumSamples * info_.numChannels, 0.0f);
    
    // Pre-calculate the maximum valid source index to avoid repeated bounds checks
    const size_t maxSrcSample = info_.numSamples - 1;
    const size_t dataSize = data_.size();
    
    // Linear interpolation for each channel
    for (uint32_t ch = 0; ch < info_.numChannels; ++ch) {
        for (size_t i = 0; i < newNumSamples; ++i) {
            // Calculate source position: for output sample i, find corresponding source position
            // srcPos = i * (srcRate / targetRate) = i / ratio
            double srcPos = static_cast<double>(i) / ratio;
            
            // Clamp source position to valid range to prevent out-of-bounds access
            if (srcPos < 0.0) {
                srcPos = 0.0;
            } else if (srcPos > static_cast<double>(maxSrcSample)) {
                srcPos = static_cast<double>(maxSrcSample);
            }
            
            size_t srcIndex = static_cast<size_t>(srcPos);
            double fraction = srcPos - srcIndex;
            
            // Ensure srcIndex doesn't exceed max valid index
            if (srcIndex > maxSrcSample) {
                srcIndex = maxSrcSample;
                fraction = 0.0;
            }
            
            size_t srcIdx1 = srcIndex * info_.numChannels + ch;
            size_t srcIdx2 = (srcIndex + 1) * info_.numChannels + ch;
            size_t dstIdx = i * info_.numChannels + ch;
            
            // Safe bounds check with guaranteed valid access
            if (srcIdx1 < dataSize) {
                if (srcIdx2 < dataSize && fraction > 0.0) {
                    // Linear interpolation between two samples
                    newData[dstIdx] = static_cast<Sample>(
                        data_[srcIdx1] * (1.0 - fraction) + 
                        data_[srcIdx2] * fraction);
                } else {
                    // Use single sample (at boundary or no interpolation needed)
                    newData[dstIdx] = data_[srcIdx1];
                }
            }
            // If srcIdx1 is out of bounds, newData[dstIdx] stays at initialized 0.0f
        }
    }
    
    // Update file info
    data_ = std::move(newData);
    info_.numSamples = newNumSamples;
    info_.sampleRate = targetRate;
    info_.durationSeconds = static_cast<double>(newNumSamples) / targetRate;
    
    return true;
}

AudioFile AudioFile::generateSineWave(float frequency,
                                     double durationSeconds,
                                     SampleRate sampleRate,
                                     float amplitude) {
    AudioFile file;
    
    const size_t numSamples = static_cast<size_t>(durationSeconds * sampleRate);
    std::vector<Sample> data(numSamples);
    
    const float angularFreq = 2.0f * static_cast<float>(M_PI) * frequency;
    
    for (size_t i = 0; i < numSamples; ++i) {
        float time = static_cast<float>(i) / static_cast<float>(sampleRate);
        data[i] = amplitude * std::sin(angularFreq * time);
    }
    
    file.setData(data, 1, sampleRate);
    return file;
}

AudioFormat AudioFile::detectFormat(const std::string& filepath) {
    if (filepath.length() < 4) {
        return AudioFormat::Unknown;
    }

    std::string ext = filepath.substr(filepath.length() - 4);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".wav") return AudioFormat::WAV;
    if (ext == "aiff" || ext == ".aif") return AudioFormat::AIFF;
    if (ext == "flac") return AudioFormat::FLAC;

    return AudioFormat::Unknown;
}

} // namespace audio
} // namespace daiw
