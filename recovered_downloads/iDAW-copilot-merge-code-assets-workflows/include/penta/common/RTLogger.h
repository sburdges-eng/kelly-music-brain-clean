#pragma once

#include <array>
#include <atomic>
#include <string>
#include <thread>

namespace penta {

enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error
};

/**
 * Real-time safe logging system
 * Uses lock-free queue to defer string formatting to non-RT thread
 */
class RTLogger {
public:
    static constexpr size_t kMaxMessageSize = 256;
    static constexpr size_t kQueueSize = 1024;
    
    RTLogger();
    ~RTLogger();
    
    // Non-copyable, non-movable
    RTLogger(const RTLogger&) = delete;
    RTLogger& operator=(const RTLogger&) = delete;
    
    // RT-safe logging (lock-free)
    void logRT(LogLevel level, const char* message) noexcept;
    
    // Non-RT logging (for Python bridge)
    void log(LogLevel level, const std::string& message);
    
    // Control
    void start();
    void stop();
    void setMinLevel(LogLevel level) { minLevel_.store(level); }
    
private:
    struct LogMessage {
        LogLevel level;
        std::array<char, kMaxMessageSize> text;
        std::atomic<bool> ready;
        
        LogMessage() : level(LogLevel::Info), text{}, ready(false) {}
    };
    
    void processingThread();
    
    std::array<LogMessage, kQueueSize> messageQueue_;
    std::atomic<size_t> writeIndex_;
    std::atomic<size_t> readIndex_;
    std::atomic<LogLevel> minLevel_;
    std::atomic<bool> running_;
    std::thread processingThread_;
};

// Global logger instance
RTLogger& getLogger();

// Convenience macros
#define PENTA_LOG_RT_DEBUG(msg) penta::getLogger().logRT(penta::LogLevel::Debug, msg)
#define PENTA_LOG_RT_INFO(msg) penta::getLogger().logRT(penta::LogLevel::Info, msg)
#define PENTA_LOG_RT_WARNING(msg) penta::getLogger().logRT(penta::LogLevel::Warning, msg)
#define PENTA_LOG_RT_ERROR(msg) penta::getLogger().logRT(penta::LogLevel::Error, msg)

} // namespace penta
