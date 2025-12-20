#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

namespace penta::diagnostics {

/**
 * Real-time performance monitoring with minimal overhead
 * Tracks CPU usage, latency, and xruns
 */
class PerformanceMonitor {
public:
    static constexpr size_t kHistorySize = 1000;
    
    PerformanceMonitor();
    ~PerformanceMonitor() = default;
    
    // RT-safe: Start timing measurement
    void beginMeasurement() noexcept;
    
    // RT-safe: End timing measurement
    void endMeasurement() noexcept;
    
    // RT-safe: Record xrun (buffer underrun/overrun)
    void recordXrun() noexcept;
    
    // Non-RT: Get average latency in microseconds
    float getAverageLatencyUs() const;
    
    // Non-RT: Get peak latency in microseconds
    float getPeakLatencyUs() const;
    
    // Non-RT: Get CPU usage percentage estimate
    float getCpuUsagePercent(size_t bufferSize, double sampleRate) const;
    
    // Non-RT: Get xrun count
    size_t getXrunCount() const { return xrunCount_.load(); }
    
    // Non-RT: Reset statistics
    void reset();
    
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    
    TimePoint measurementStart_;
    std::vector<uint64_t> latencyHistory_;
    std::atomic<size_t> historyIndex_;
    std::atomic<uint64_t> peakLatencyUs_;
    std::atomic<size_t> xrunCount_;
};

} // namespace penta::diagnostics
