#include "penta/diagnostics/DiagnosticsEngine.h"
#include <sstream>
#include <iomanip>
#include <cmath>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
#elif __APPLE__
    #include <mach/mach.h>
    #include <sys/sysctl.h>
#else
    #include <sys/sysinfo.h>
    #include <unistd.h>
#endif

namespace penta::diagnostics {

namespace {

// Get current process memory usage in bytes
size_t getProcessMemoryUsage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.PrivateUsage;
    }
    return 0;
#elif __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
#else
    // Linux: Read from /proc/self/statm
    long pageSize = sysconf(_SC_PAGESIZE);
    FILE* file = fopen("/proc/self/statm", "r");
    if (!file) return 0;
    
    unsigned long vmSize, rss;
    if (fscanf(file, "%lu %lu", &vmSize, &rss) == 2) {
        fclose(file);
        return rss * pageSize;
    }
    fclose(file);
    return 0;
#endif
}

// Get available system memory in bytes
size_t getAvailableMemory() {
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        return statex.ullAvailPhys;
    }
    return 0;
#elif __APPLE__
    vm_statistics64_data_t vmStats;
    mach_msg_type_number_t infoCount = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vmStats, &infoCount) == KERN_SUCCESS) {
        return vmStats.free_count * vm_page_size;
    }
    return 0;
#else
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return si.freeram * si.mem_unit;
    }
    return 0;
#endif
}

} // anonymous namespace

DiagnosticsEngine::DiagnosticsEngine(const Config& config)
    : config_(config)
    , perfMonitor_(std::make_unique<PerformanceMonitor>())
    , audioAnalyzer_(std::make_unique<AudioAnalyzer>())
{
}

DiagnosticsEngine::~DiagnosticsEngine() = default;

void DiagnosticsEngine::beginMeasurement() noexcept {
    if (config_.enablePerformanceMonitoring && perfMonitor_) {
        perfMonitor_->beginMeasurement();
    }
}

void DiagnosticsEngine::endMeasurement() noexcept {
    if (config_.enablePerformanceMonitoring && perfMonitor_) {
        perfMonitor_->endMeasurement();
    }
}

void DiagnosticsEngine::analyzeAudio(const float* buffer, size_t frames, size_t channels) noexcept {
    if (config_.enableAudioAnalysis && audioAnalyzer_) {
        audioAnalyzer_->analyze(buffer, frames, channels);
    }
}

DiagnosticsEngine::SystemStats DiagnosticsEngine::getStats() const {
    SystemStats stats{};
    
    if (perfMonitor_) {
        stats.cpuUsagePercent = perfMonitor_->getCpuUsagePercent(512, 48000.0);
        stats.averageLatencyMs = perfMonitor_->getAverageLatencyUs() / 1000.0f;
        stats.peakLatencyMs = perfMonitor_->getPeakLatencyUs() / 1000.0f;
        stats.xrunCount = perfMonitor_->getXrunCount();
    }
    
    if (audioAnalyzer_) {
        stats.rmsLevel = audioAnalyzer_->getRmsLevel();
        stats.peakLevel = audioAnalyzer_->getPeakLevel();
        stats.dynamicRange = audioAnalyzer_->getDynamicRange();
        stats.clipping = audioAnalyzer_->isClipping();
    }
    
    stats.memoryUsedBytes = getProcessMemoryUsage();
    stats.memoryAvailableBytes = getAvailableMemory();
    
    return stats;
}

std::string DiagnosticsEngine::getPerformanceReport() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    oss << "=== Performance Report ===\n";
    
    if (perfMonitor_) {
        float avgLatency = perfMonitor_->getAverageLatencyUs();
        float peakLatency = perfMonitor_->getPeakLatencyUs();
        float cpuUsage = perfMonitor_->getCpuUsagePercent(512, 48000.0);
        size_t xruns = perfMonitor_->getXrunCount();
        
        oss << "CPU Usage:       " << cpuUsage << "%\n";
        oss << "Avg Latency:     " << (avgLatency / 1000.0f) << " ms\n";
        oss << "Peak Latency:    " << (peakLatency / 1000.0f) << " ms\n";
        oss << "XRun Count:      " << xruns << "\n";
    }
    
    size_t memUsed = getProcessMemoryUsage();
    size_t memAvail = getAvailableMemory();
    
    oss << "Memory Used:     " << (memUsed / 1024.0 / 1024.0) << " MB\n";
    oss << "Memory Available: " << (memAvail / 1024.0 / 1024.0) << " MB\n";
    
    return oss.str();
}

std::string DiagnosticsEngine::getAudioReport() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    
    oss << "=== Audio Analysis Report ===\n";
    
    if (audioAnalyzer_) {
        float rms = audioAnalyzer_->getRmsLevel();
        float peak = audioAnalyzer_->getPeakLevel();
        float dynRange = audioAnalyzer_->getDynamicRange();
        bool clipping = audioAnalyzer_->isClipping();
        
        // Convert to dB
        float rmsDb = rms > 1e-10f ? 20.0f * std::log10(rms) : -100.0f;
        float peakDb = peak > 1e-10f ? 20.0f * std::log10(peak) : -100.0f;
        
        oss << "RMS Level:       " << rmsDb << " dBFS (" << rms << ")\n";
        oss << "Peak Level:      " << peakDb << " dBFS (" << peak << ")\n";
        oss << "Dynamic Range:   " << dynRange << " dB\n";
        oss << "Clipping:        " << (clipping ? "YES" : "NO") << "\n";
    }
    
    return oss.str();
}

void DiagnosticsEngine::reset() {
    if (perfMonitor_) {
        perfMonitor_->reset();
    }
    if (audioAnalyzer_) {
        audioAnalyzer_->reset();
    }
}

void DiagnosticsEngine::updateConfig(const Config& config) {
    config_ = config;
}

} // namespace penta::diagnostics
