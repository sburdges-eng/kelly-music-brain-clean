/*
 * RTNeural Model Loading Test
 * ===========================
 * C++ test to verify RTNeural JSON models can be loaded correctly.
 * Compile with: g++ -std=c++17 -I/path/to/RTNeural test_model_loading.cpp -o test_model_loading
 */

#ifdef ENABLE_RTNEURAL

#include <RTNeural/RTNeural.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

bool test_model_loading(const std::string& json_path) {
    std::cout << "Testing model: " << json_path << std::endl;

    try {
        // Open JSON file
        std::ifstream jsonStream(json_path, std::ifstream::binary);

        if (!jsonStream.is_open()) {
            std::cerr << "Error: Could not open file: " << json_path << std::endl;
            return false;
        }

        // Parse JSON
        auto model = RTNeural::json_parser::parseJson<float>(jsonStream);

        if (!model) {
            std::cerr << "Error: Failed to parse JSON model" << std::endl;
            return false;
        }

        std::cout << "✓ Model loaded successfully" << std::endl;

        // Reset model state
        model->reset();

        // Get model input/output sizes (if available)
        // Note: RTNeural API may vary - adjust based on your version
        std::cout << "✓ Model state reset" << std::endl;

        // Test inference with dummy input
        // Note: Adjust based on your model's input size
        std::vector<float> input(128, 0.5f);  // Default to 128 for EmotionRecognizer
        std::vector<float> output;

        auto start = std::chrono::high_resolution_clock::now();
        model->forward(input.data());
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double latency_ms = duration.count() / 1000.0;

        std::cout << "✓ Inference completed" << std::endl;
        std::cout << "  Latency: " << latency_ms << " ms" << std::endl;

        if (latency_ms < 10.0) {
            std::cout << "✓ Latency meets target (<10ms)" << std::endl;
        } else {
            std::cout << "⚠ Latency exceeds target (<10ms)" << std::endl;
        }

        jsonStream.close();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model.json> [model2.json ...]" << std::endl;
        return 1;
    }

    std::cout << "RTNeural Model Loading Test" << std::endl;
    std::cout << "===========================" << std::endl;

    int passed = 0;
    int failed = 0;

    for (int i = 1; i < argc; ++i) {
        std::cout << std::endl;
        if (test_model_loading(argv[i])) {
            passed++;
        } else {
            failed++;
        }
    }

    std::cout << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;

    return failed > 0 ? 1 : 0;
}

#else

#include <iostream>

int main() {
    std::cout << "RTNeural not enabled. Compile with -DENABLE_RTNEURAL" << std::endl;
    return 1;
}

#endif
