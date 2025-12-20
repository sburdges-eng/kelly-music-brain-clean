/**
 * @file ai_inference.cpp
 * @brief Hybrid (on-device + optional cloud) AI inference for MIDI.
 */

#include "ai_inference.h"

#include <random>

#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_events/juce_events.h>

namespace kelly::ml {

AIInferenceEngine::AIInferenceEngine(AIConfig config) : config_(std::move(config))
#if KELLY_HAS_ONNX
                                                      , onnxEnv_(ORT_LOGGING_LEVEL_WARNING, "kelly-ai")
#endif
{
#if KELLY_HAS_ONNX
    if (!config_.onnxModelPath.empty()) {
        loadOnnxModel(config_.onnxModelPath);
    }
#endif
}

void AIInferenceEngine::setTransport(double bpm, double sampleRate, int blockSize) {
    bpm_ = bpm > 0.0 ? bpm : bpm_;
    sampleRate_ = sampleRate > 0.0 ? sampleRate : sampleRate_;
    blockSize_ = blockSize > 0 ? blockSize : blockSize_;
}

bool AIInferenceEngine::loadOnnxModel(const std::string& path) {
#if KELLY_HAS_ONNX
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(onnxEnv_, path.c_str(), opts);
        return true;
    } catch (const Ort::Exception&) {
        session_.reset();
        return false;
    }
#else
    juce::ignoreUnused(path);
    return false;
#endif
}

std::vector<MidiEvent> AIInferenceEngine::generate(const EmotionRequest& request, bool allowNetwork) {
    std::vector<MidiEvent> out;

    if (tryOnnx(request, out)) {
        return out;
    }

    if (config_.enableCloudFallback && allowNetwork && tryCloud(request, out)) {
        return out;
    }

    return generateHeuristic(request);
}

bool AIInferenceEngine::hasOnnx() const noexcept {
#if KELLY_HAS_ONNX
    return session_ != nullptr;
#else
    return false;
#endif
}

bool AIInferenceEngine::tryOnnx(const EmotionRequest& request, std::vector<MidiEvent>& out) {
#if KELLY_HAS_ONNX
    if (!session_) {
        return false;
    }

    // Stub: in lieu of a real model contract, feed a small feature vector
    // and emit a simple descending figure based on first output element.
    try {
        constexpr size_t kInputSize = 4;
        float features[kInputSize] = {request.valence, request.arousal, request.intensity, static_cast<float>(request.tempo)};

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> inputShape{1, static_cast<int64_t>(kInputSize)};

        auto inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, features, kInputSize, inputShape.data(), inputShape.size());

        const char* inputNames[] = {"input"};
        const char* outputNames[] = {"output"};

        auto outputValues = session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
        if (outputValues.empty() || !outputValues.front().IsTensor()) {
            return false;
        }

        auto& tensor = outputValues.front();
        float* output = tensor.GetTensorMutableData<float>();
        auto count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        if (count == 0) {
            return false;
        }

        const float base = output[0];
        MidiEvent event;
        event.note = 60 + static_cast<int>(base * 4.0f);
        event.velocity = juce::jlimit(40, 120, static_cast<int>(90 + base * 20.0f));
        event.startBeat = 0.0f;
        event.durationBeats = 0.5f;
        out.push_back(event);
        return true;
    } catch (const Ort::Exception&) {
        return false;
    }
#else
    juce::ignoreUnused(request, out);
    return false;
#endif
}

bool AIInferenceEngine::tryCloud(const EmotionRequest& request, std::vector<MidiEvent>& out) {
    if (!config_.enableCloudFallback || config_.cloudEndpoint.empty()) {
        return false;
    }

    juce::DynamicObject json;
    json.setProperty("emotion", request.tag);
    json.setProperty("valence", request.valence);
    json.setProperty("arousal", request.arousal);
    json.setProperty("intensity", request.intensity);
    json.setProperty("tempo", request.tempo);
    json.setProperty("bars", request.bars);

    juce::URL url(config_.cloudEndpoint);
    auto body = juce::JSON::toString(&json);
    auto response = url.readEntireStreamAsString(false, nullptr, nullptr, body, "application/json");
    if (response.isEmpty()) {
        return false;
    }

    juce::var parsed = juce::JSON::fromString(response);
    if (!parsed.isArray()) {
        return false;
    }

    auto* arr = parsed.getArray();
    for (const auto& v : *arr) {
        if (!v.isObject()) continue;
        auto* obj = v.getDynamicObject();
        MidiEvent ev;
        ev.note = static_cast<int>(obj->getProperty("note", 60));
        ev.velocity = static_cast<int>(obj->getProperty("velocity", 96));
        ev.startBeat = static_cast<float>(obj->getProperty("startBeat", 0.0f));
        ev.durationBeats = static_cast<float>(obj->getProperty("durationBeats", 0.25f));
        ev.channel = static_cast<int>(obj->getProperty("channel", 1));
        out.push_back(ev);
    }

    return !out.empty();
}

std::vector<MidiEvent> AIInferenceEngine::generateHeuristic(const EmotionRequest& request) {
    std::vector<MidiEvent> events;

    const int baseNote = request.valence >= 0.0f ? 60 : 57; // C3 vs A2
    const float energy = juce::jlimit(0.2f, 1.0f, std::abs(request.arousal) + 0.3f);
    const float density = juce::jlimit(1, 4, request.bars);

    std::mt19937 rng{static_cast<uint32_t>(std::hash<std::string>{}(request.tag))};
    std::uniform_int_distribution<int> noteSpread(-5, 5);

    for (int step = 0; step < density * 4; ++step) {
        MidiEvent ev;
        ev.startBeat = step * 0.5f;
        ev.durationBeats = 0.35f;
        ev.note = baseNote + noteSpread(rng);
        ev.velocity = static_cast<int>(juce::jlimit(50.0f, 120.0f, 70.0f + energy * 40.0f));
        events.push_back(ev);
    }

    return events;
}

void writeMidiToBuffer(const std::vector<MidiEvent>& events,
                       juce::MidiBuffer& buffer,
                       double bpm,
                       double sampleRate,
                       int blockSize) {
    if (events.empty()) {
        return;
    }

    const double samplesPerBeat = sampleRate * 60.0 / std::max(1.0, bpm);

    for (const auto& ev : events) {
        const int startSample = juce::jlimit(0, blockSize - 1,
                                             static_cast<int>(ev.startBeat * samplesPerBeat));
        const int endSample = juce::jlimit(startSample, blockSize - 1,
                                           startSample + static_cast<int>(ev.durationBeats * samplesPerBeat));

        buffer.addEvent(juce::MidiMessage::noteOn(ev.channel, ev.note, (juce::uint8)ev.velocity), startSample);
        buffer.addEvent(juce::MidiMessage::noteOff(ev.channel, ev.note), endSample);
    }
}

} // namespace kelly::ml
