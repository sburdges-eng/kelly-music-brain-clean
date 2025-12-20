/**
 * @file ai_inference.h
 * @brief On-device + hybrid AI inference for MIDI generation.
 */

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <optional>
#include <string>
#include <vector>

#ifndef KELLY_HAS_ONNX
#define KELLY_HAS_ONNX 0
#endif

#if KELLY_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace kelly::ml {

struct EmotionRequest {
    std::string tag{"neutral"};
    float valence{0.0f};
    float arousal{0.0f};
    float intensity{0.5f};
    int tempo{120};
    int bars{1};
};

struct MidiEvent {
    int note{60};
    int velocity{96};
    float startBeat{0.0f};
    float durationBeats{0.25f};
    int channel{1};
};

struct AIConfig {
    std::string onnxModelPath;
    bool enableCloudFallback{false};
    std::string cloudEndpoint;
};

class AIInferenceEngine {
public:
    explicit AIInferenceEngine(AIConfig config = {});

    void setTransport(double bpm, double sampleRate, int blockSize);
    bool loadOnnxModel(const std::string& path);
    bool hasOnnx() const noexcept;

    std::vector<MidiEvent> generate(const EmotionRequest& request, bool allowNetwork = false);

private:
    bool tryOnnx(const EmotionRequest& request, std::vector<MidiEvent>& out);
    bool tryCloud(const EmotionRequest& request, std::vector<MidiEvent>& out);
    std::vector<MidiEvent> generateHeuristic(const EmotionRequest& request);

    double bpm_{120.0};
    double sampleRate_{44100.0};
    int blockSize_{512};
    AIConfig config_;

#if KELLY_HAS_ONNX
    Ort::Env onnxEnv_;
    std::unique_ptr<Ort::Session> session_;
#endif
};

void writeMidiToBuffer(const std::vector<MidiEvent>& events,
                       juce::MidiBuffer& buffer,
                       double bpm,
                       double sampleRate,
                       int blockSize);

} // namespace kelly::ml
