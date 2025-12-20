/**
 * @file MLInterface.cpp
 * @brief Implementation of RT-safe ML inference interface
 */

#include "penta/ml/MLInterface.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifdef PENTA_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace penta::ml {

/**
 * @brief Private implementation (PIMPL pattern)
 */
class MLInterface::Impl {
public:
    explicit Impl(const MLConfig& config) : config_(config) {
        stats_.total_requests = 0;
        stats_.completed_requests = 0;
        stats_.failed_requests = 0;
        stats_.queue_overflows = 0;
        stats_.avg_latency_ms = 0.0f;
        stats_.max_latency_ms = 0.0f;
    }

    ~Impl() {
        stop();
    }

    bool start() {
        if (running_.exchange(true)) {
            return false;  // Already running
        }

        inference_thread_ = std::thread([this] { inferenceLoop(); });
        return true;
    }

    void stop() {
        if (running_.exchange(false)) {
            // Wake up inference thread
            {
                std::lock_guard<std::mutex> lock(cv_mutex_);
            }
            cv_.notify_one();

            if (inference_thread_.joinable()) {
                inference_thread_.join();
            }
        }
    }

    bool isRunning() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    bool submitRequest(const InferenceRequest& request) noexcept {
        if (!request_queue_.try_push(request)) {
            stats_.queue_overflows++;
            return false;
        }
        stats_.total_requests++;

        // Wake up inference thread (non-blocking notification)
        cv_.notify_one();
        return true;
    }

    bool pollResult(InferenceResult& result) noexcept {
        return result_queue_.try_pop(result);
    }

    uint64_t getNextRequestId() noexcept {
        return next_request_id_.fetch_add(1, std::memory_order_relaxed);
    }

    bool loadModel(ModelType type, const std::string& path) {
        std::lock_guard<std::mutex> lock(model_mutex_);

#ifdef PENTA_HAS_ONNX
        try {
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef PENTA_HAS_COREML
            // Add CoreML execution provider on Apple platforms
            if (config_.use_coreml) {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(options, 0));
            }
#endif

            auto session = std::make_unique<Ort::Session>(env_, path.c_str(), options);
            models_[type] = std::move(session);
            return true;
        } catch (const Ort::Exception& e) {
            // Log error
            return false;
        }
#else
        // Stub implementation without ONNX
        loaded_models_.insert(type);
        return true;
#endif
    }

    void unloadModel(ModelType type) {
        std::lock_guard<std::mutex> lock(model_mutex_);
#ifdef PENTA_HAS_ONNX
        models_.erase(type);
#else
        loaded_models_.erase(type);
#endif
    }

    bool isModelLoaded(ModelType type) const {
        std::lock_guard<std::mutex> lock(model_mutex_);
#ifdef PENTA_HAS_ONNX
        return models_.find(type) != models_.end();
#else
        return loaded_models_.find(type) != loaded_models_.end();
#endif
    }

    Stats getStats() const {
        return stats_;
    }

private:
    void inferenceLoop() {
        while (running_.load(std::memory_order_acquire)) {
            InferenceRequest request;

            // Wait for request or shutdown
            {
                std::unique_lock<std::mutex> lock(cv_mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                    return !request_queue_.empty() || !running_.load();
                });
            }

            // Process all pending requests
            while (request_queue_.try_pop(request)) {
                auto start = std::chrono::high_resolution_clock::now();

                InferenceResult result = runInference(request);

                auto end = std::chrono::high_resolution_clock::now();
                result.latency_ms = std::chrono::duration<float, std::milli>(end - start).count();

                // Update statistics
                updateStats(result);

                // Push result to output queue
                if (!result_queue_.try_push(result)) {
                    // Result queue full, drop oldest
                    InferenceResult dummy;
                    result_queue_.try_pop(dummy);
                    result_queue_.try_push(result);
                }
            }
        }
    }

    InferenceResult runInference(const InferenceRequest& request) {
        InferenceResult result{};
        result.model_type = request.model_type;
        result.request_id = request.request_id;

#ifdef PENTA_HAS_ONNX
        std::lock_guard<std::mutex> lock(model_mutex_);
        auto it = models_.find(request.model_type);
        if (it == models_.end()) {
            result.success = false;
            return result;
        }

        try {
            auto& session = it->second;
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // Create input tensor
            std::array<int64_t, 2> input_shape = {1, static_cast<int64_t>(request.input_size)};
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(request.input_data.data()),
                request.input_size,
                input_shape.data(),
                input_shape.size());

            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name = session->GetInputNameAllocated(0, allocator);
            auto output_name = session->GetOutputNameAllocated(0, allocator);

            const char* input_names[] = {input_name.get()};
            const char* output_names[] = {output_name.get()};

            // Run inference
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, 1);

            // Copy output
            auto* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            result.output_size = static_cast<size_t>(output_shape[1]);
            std::copy(output_data, output_data + result.output_size, result.output_data.begin());

            // Extract confidence from model output
            // For classification models: max probability is confidence
            // For regression models: use a reasonable default
            if (result.output_size > 0) {
                float max_val = *std::max_element(
                    result.output_data.begin(),
                    result.output_data.begin() + result.output_size);
                // Clamp confidence to [0, 1] range
                result.confidence = std::min(1.0f, std::max(0.0f, max_val));
            } else {
                result.confidence = 0.0f;
            }
            result.success = true;
        } catch (const Ort::Exception& e) {
            result.success = false;
        }
#else
        // Stub implementation: echo input as output
        std::copy(request.input_data.begin(),
                  request.input_data.begin() + request.input_size,
                  result.output_data.begin());
        result.output_size = request.input_size;
        result.confidence = 0.5f;
        result.success = true;
#endif

        return result;
    }

    void updateStats(const InferenceResult& result) {
        if (result.success) {
            stats_.completed_requests++;
            // Exponential moving average for latency
            float alpha = 0.1f;
            stats_.avg_latency_ms = alpha * result.latency_ms +
                                    (1.0f - alpha) * stats_.avg_latency_ms;
            if (result.latency_ms > stats_.max_latency_ms) {
                stats_.max_latency_ms = result.latency_ms;
            }
        } else {
            stats_.failed_requests++;
        }
    }

    MLConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> next_request_id_{1};

    // Lock-free queues for RT communication
    LockFreeQueue<InferenceRequest, 16> request_queue_;
    LockFreeQueue<InferenceResult, 16> result_queue_;

    // Inference thread
    std::thread inference_thread_;
    std::mutex cv_mutex_;
    std::condition_variable cv_;

    // Model storage
    mutable std::mutex model_mutex_;
#ifdef PENTA_HAS_ONNX
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "penta"};
    std::unordered_map<ModelType, std::unique_ptr<Ort::Session>> models_;
#else
    std::unordered_set<ModelType> loaded_models_;
#endif

    // Statistics
    mutable Stats stats_{};
};

// ============================================================================
// MLInterface public implementation
// ============================================================================

MLInterface::MLInterface(const MLConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MLInterface::~MLInterface() = default;

bool MLInterface::start() {
    return impl_->start();
}

void MLInterface::stop() {
    impl_->stop();
}

bool MLInterface::isRunning() const noexcept {
    return impl_->isRunning();
}

bool MLInterface::submitRequest(const InferenceRequest& request) noexcept {
    return impl_->submitRequest(request);
}

bool MLInterface::pollResult(InferenceResult& result) noexcept {
    return impl_->pollResult(result);
}

uint64_t MLInterface::getNextRequestId() noexcept {
    return impl_->getNextRequestId();
}

bool MLInterface::loadModel(ModelType type, const std::string& path) {
    return impl_->loadModel(type, path);
}

void MLInterface::unloadModel(ModelType type) {
    impl_->unloadModel(type);
}

bool MLInterface::isModelLoaded(ModelType type) const {
    return impl_->isModelLoaded(type);
}

MLInterface::Stats MLInterface::getStats() const {
    return impl_->getStats();
}

}  // namespace penta::ml
