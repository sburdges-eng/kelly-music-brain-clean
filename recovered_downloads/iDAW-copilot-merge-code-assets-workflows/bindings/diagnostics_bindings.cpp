#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "penta/diagnostics/DiagnosticsEngine.h"

namespace py = pybind11;
using namespace penta::diagnostics;

void bind_diagnostics(py::module_& m) {
    // SystemStats structure
    py::class_<DiagnosticsEngine::SystemStats>(m, "SystemStats")
        .def(py::init<>())
        .def_readonly("cpu_usage_percent", &DiagnosticsEngine::SystemStats::cpuUsagePercent)
        .def_readonly("average_latency_ms", &DiagnosticsEngine::SystemStats::averageLatencyMs)
        .def_readonly("peak_latency_ms", &DiagnosticsEngine::SystemStats::peakLatencyMs)
        .def_readonly("xrun_count", &DiagnosticsEngine::SystemStats::xrunCount)
        .def_readonly("rms_level", &DiagnosticsEngine::SystemStats::rmsLevel)
        .def_readonly("peak_level", &DiagnosticsEngine::SystemStats::peakLevel)
        .def_readonly("dynamic_range", &DiagnosticsEngine::SystemStats::dynamicRange)
        .def_readonly("clipping", &DiagnosticsEngine::SystemStats::clipping)
        .def_readonly("memory_used_bytes", &DiagnosticsEngine::SystemStats::memoryUsedBytes)
        .def_readonly("memory_available_bytes", &DiagnosticsEngine::SystemStats::memoryAvailableBytes)
        .def("__repr__", [](const DiagnosticsEngine::SystemStats& s) {
            return "SystemStats(CPU=" + std::to_string(s.cpuUsagePercent) + 
                   "%, latency=" + std::to_string(s.averageLatencyMs) + "ms)";
        });
    
    // DiagnosticsEngine configuration
    py::class_<DiagnosticsEngine::Config>(m, "DiagnosticsConfig")
        .def(py::init<>())
        .def_readwrite("enable_performance_monitoring", 
            &DiagnosticsEngine::Config::enablePerformanceMonitoring)
        .def_readwrite("enable_audio_analysis", 
            &DiagnosticsEngine::Config::enableAudioAnalysis)
        .def_readwrite("update_interval_ms", 
            &DiagnosticsEngine::Config::updateIntervalMs);
    
    // DiagnosticsEngine
    py::class_<DiagnosticsEngine>(m, "DiagnosticsEngine")
        .def(py::init<const DiagnosticsEngine::Config&>(),
            py::arg("config") = DiagnosticsEngine::Config())
        .def("begin_measurement", &DiagnosticsEngine::beginMeasurement,
            "Start performance measurement (RT-safe)")
        .def("end_measurement", &DiagnosticsEngine::endMeasurement,
            "End performance measurement (RT-safe)")
        .def("analyze_audio", 
            [](DiagnosticsEngine& self, py::array_t<float> buffer, int channels) {
                auto info = buffer.request();
                if (info.ndim != 1) {
                    throw std::runtime_error("Buffer must be 1-dimensional");
                }
                size_t frames = info.shape[0] / static_cast<size_t>(channels);
                self.analyzeAudio(static_cast<float*>(info.ptr), frames, 
                                static_cast<size_t>(channels));
            },
            py::arg("buffer"), 
            py::arg("channels") = 2,
            "Analyze audio buffer (RT-safe)")
        .def("get_stats", &DiagnosticsEngine::getStats,
            "Get current system statistics")
        .def("get_performance_report", &DiagnosticsEngine::getPerformanceReport,
            "Get detailed performance report")
        .def("get_audio_report", &DiagnosticsEngine::getAudioReport,
            "Get detailed audio analysis report")
        .def("reset", &DiagnosticsEngine::reset,
            "Reset all statistics")
        .def("update_config", &DiagnosticsEngine::updateConfig,
            py::arg("config"),
            "Update engine configuration");
}
