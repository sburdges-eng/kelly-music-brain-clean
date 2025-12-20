# C++ Programming Guide

This guide covers C++ programming fundamentals and best practices for the penta-core team.

## Overview

C++ is a powerful, high-performance programming language widely used in systems programming, game development, audio processing, and embedded systems. It provides low-level memory control while supporting object-oriented, generic, and functional programming paradigms.

## Core Concepts

### Modern C++ Standards

- **C++11**: Lambda expressions, smart pointers, move semantics, auto keyword
- **C++14**: Generic lambdas, variable templates, relaxed constexpr
- **C++17**: Structured bindings, if/switch with initializers, std::optional
- **C++20**: Concepts, ranges, coroutines, modules
- **C++23**: Latest standard with improved features

### Memory Management

Modern C++ emphasizes safe memory management through smart pointers:

```cpp
#include <memory>
#include <vector>

// Use smart pointers instead of raw pointers
auto uniquePtr = std::make_unique<int>(42);
auto sharedPtr = std::make_shared<std::vector<float>>();

// RAII (Resource Acquisition Is Initialization)
class AudioBuffer {
private:
    std::unique_ptr<float[]> data;
    size_t size;
    
public:
    AudioBuffer(size_t bufferSize) 
        : data(std::make_unique<float[]>(bufferSize))
        , size(bufferSize) {}
    
    // No need for explicit destructor - smart pointer handles cleanup
};
```

### Templates and Generic Programming

```cpp
template<typename T>
class RingBuffer {
    std::vector<T> buffer;
    size_t writePos = 0;
    size_t readPos = 0;
    
public:
    explicit RingBuffer(size_t capacity) : buffer(capacity) {}
    
    void write(T value) {
        buffer[writePos % buffer.size()] = std::move(value);
        ++writePos;
    }
    
    T read() {
        T value = std::move(buffer[readPos % buffer.size()]);
        ++readPos;
        return value;
    }
};
```

## Audio Programming with C++

### Basic Audio Processing

```cpp
#include <cmath>
#include <numbers>
#include <vector>

class SineWaveGenerator {
    static constexpr double TwoPi = 2.0 * std::numbers::pi;
    
    double phase = 0.0;
    double frequency;
    double sampleRate;
    
public:
    SineWaveGenerator(double freq, double sr) 
        : frequency(freq), sampleRate(sr) {}
    
    void generate(std::vector<float>& buffer) {
        const double phaseIncrement = TwoPi * frequency / sampleRate;
        
        for (auto& sample : buffer) {
            sample = static_cast<float>(std::sin(phase));
            phase += phaseIncrement;
            
            if (phase >= TwoPi) {
                phase -= TwoPi;
            }
        }
    }
};
```

### Thread-Safe Audio Programming

```cpp
#include <atomic>
#include <thread>

class AudioParameter {
    std::atomic<float> value;
    
public:
    AudioParameter(float initial) : value(initial) {}
    
    void set(float newValue) {
        value.store(newValue, std::memory_order_relaxed);
    }
    
    float get() const {
        return value.load(std::memory_order_relaxed);
    }
};
```

## Build Systems

### CMake

CMake is the standard build system for C++ projects:

```cmake
cmake_minimum_required(VERSION 3.16)
project(PentaCore VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(penta-audio STATIC
    src/audio_processor.cpp
    src/ring_buffer.cpp
)

target_include_directories(penta-audio PUBLIC include)
```

### Building with CMake

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build .

# Run tests
ctest
```

## Best Practices

1. **Use Modern C++ Features**: Prefer smart pointers, RAII, and standard library containers
2. **Follow the Rule of Zero/Five**: Let the compiler generate special member functions when possible
3. **Prefer const and constexpr**: Make immutability explicit
4. **Use static analysis tools**: Clang-Tidy, cppcheck, and address sanitizers
5. **Write Unit Tests**: Use frameworks like Google Test or Catch2

## Resources

- [C++ Reference](https://en.cppreference.com/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [Modern C++ Features](https://github.com/AnthonyCalandra/modern-cpp-features)
- [Learn C++](https://www.learncpp.com/)
