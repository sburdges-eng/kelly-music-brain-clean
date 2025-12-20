#pragma once

#include "penta/common/RTTypes.h"
#include "penta/diagnostics/PerformanceMonitor.h"
#include "penta/diagnostics/AudioAnalyzer.h"
#include <memory>
#include <string>

namespace penta::diagnostics {

/**
 * Main diagnostics engine
 * Provides performance monitoring and audio analysis
 */
class DiagnosticsEngine {
public:
    struct Config {
        bool enablePerformanceMonitoring;
        bool enableAudioAnalysis;
        size_t updateIntervalMs;
        
        Config() 
            : enablePerformanceMonitoring(true)
            , enableAudioAnalysis(true)
            , updateIntervalMs(100)
        {}
    };
    
    struct SystemStats {
        // Performance
        float cpuUsagePercent;
        float averageLatencyMs;
        float peakLatencyMs;
        size_t xrunCount;
        
        // Audio
        float rmsLevel;
        float peakLevel;
        float dynamicRange;
        bool clipping;
        
        // Memory
        size_t memoryUsedBytes;
        size_t memoryAvailableBytes;
    };
    
    explicit DiagnosticsEngine(const Config& config = Config{});
    ~DiagnosticsEngine();
    
    // Non-copyable, movable
    DiagnosticsEngine(const DiagnosticsEngine&) = delete;
    DiagnosticsEngine& operator=(const DiagnosticsEngine&) = delete;
    DiagnosticsEngine(DiagnosticsEngine&&) noexcept = default;
    DiagnosticsEngine& operator=(DiagnosticsEngine&&) noexcept = default;
    
    // RT-safe: Begin performance measurement
    void beginMeasurement() noexcept;
    
    // RT-safe: End performance measurement
    void endMeasurement() noexcept;
    
    // RT-safe: Analyze audio buffer
    void analyzeAudio(const float* buffer, size_t frames, size_t channels) noexcept;
    
    // Non-RT: Get current statistics
    SystemStats getStats() const;
    
    // Non-RT: Get performance report
    std::string getPerformanceReport() const;
    
    // Non-RT: Get audio analysis report
    std::string getAudioReport() const;
    
    // Non-RT: Reset all statistics
    void reset();
    
    // Configuration
    void updateConfig(const Config& config);
    
private:
    Config config_;
    
    std::unique_ptr<PerformanceMonitor> perfMonitor_;
    std::unique_ptr<AudioAnalyzer> audioAnalyzer_;
};

} // namespace penta::diagnostics
