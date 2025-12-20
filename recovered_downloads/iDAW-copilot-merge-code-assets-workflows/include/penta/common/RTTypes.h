#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace penta {

// Real-time safe types and constants
constexpr size_t kMaxPolyphony = 128;
constexpr size_t kMaxMidiChannels = 16;
constexpr double kDefaultSampleRate = 48000.0;
constexpr size_t kDefaultBufferSize = 512;

// MIDI note representation
struct Note {
    uint8_t pitch;      // 0-127
    uint8_t velocity;   // 0-127
    uint8_t channel;    // 0-15
    uint64_t timestamp; // samples since start
    
    constexpr Note() : pitch(0), velocity(0), channel(0), timestamp(0) {}
    constexpr Note(uint8_t p, uint8_t v, uint8_t c = 0, uint64_t t = 0)
        : pitch(p), velocity(v), channel(c), timestamp(t) {}
};

// Chord representation
struct Chord {
    std::array<bool, 12> pitchClass; // Pitch class set
    uint8_t root;                    // Root note (0-11)
    uint8_t quality;                 // Major, minor, dim, aug, etc.
    float confidence;                // 0.0-1.0
    
    Chord() : pitchClass{}, root(0), quality(0), confidence(0.0f) {}
};

// Scale representation
struct Scale {
    std::array<bool, 12> degrees;    // Scale degrees
    uint8_t tonic;                   // Tonic note (0-11)
    uint8_t mode;                    // Ionian, Dorian, etc.
    float confidence;                // 0.0-1.0
    
    Scale() : degrees{}, tonic(0), mode(0), confidence(0.0f) {}
};

// Timing information
struct TimingInfo {
    std::atomic<double> tempo;       // BPM
    std::atomic<uint64_t> barStart;  // Sample position of current bar
    std::atomic<uint32_t> numerator; // Time signature
    std::atomic<uint32_t> denominator;
    std::atomic<uint64_t> samplePosition;
    
    TimingInfo() 
        : tempo(120.0)
        , barStart(0)
        , numerator(4)
        , denominator(4)
        , samplePosition(0) {}
};

// Audio buffer (non-RT allocation, RT usage)
template<typename T>
struct AudioBuffer {
    std::vector<T> data;
    size_t channels;
    size_t frames;
    
    AudioBuffer(size_t ch = 2, size_t fr = kDefaultBufferSize)
        : data(ch * fr, T{})
        , channels(ch)
        , frames(fr) {}
    
    T* getChannelData(size_t channel) {
        return data.data() + (channel * frames);
    }
    
    const T* getChannelData(size_t channel) const {
        return data.data() + (channel * frames);
    }
    
    void resize(size_t ch, size_t fr) {
        channels = ch;
        frames = fr;
        data.resize(ch * fr);
    }
};

using AudioBufferF = AudioBuffer<float>;
using AudioBufferD = AudioBuffer<double>;

} // namespace penta
