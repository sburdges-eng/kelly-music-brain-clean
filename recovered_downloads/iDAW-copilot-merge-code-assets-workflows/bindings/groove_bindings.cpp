#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "penta/groove/GrooveEngine.h"
#include "penta/groove/OnsetDetector.h"
#include "penta/groove/TempoEstimator.h"
#include "penta/groove/RhythmQuantizer.h"

namespace py = pybind11;
using namespace penta::groove;

void bind_groove(py::module_& m) {
    // GridResolution enum
    py::enum_<RhythmQuantizer::GridResolution>(m, "GridResolution")
        .value("WHOLE", RhythmQuantizer::GridResolution::Whole)
        .value("HALF", RhythmQuantizer::GridResolution::Half)
        .value("QUARTER", RhythmQuantizer::GridResolution::Quarter)
        .value("EIGHTH", RhythmQuantizer::GridResolution::Eighth)
        .value("SIXTEENTH", RhythmQuantizer::GridResolution::Sixteenth)
        .value("THIRTY_SECOND", RhythmQuantizer::GridResolution::ThirtySecond)
        .export_values();
    
    // GrooveAnalysis structure
    py::class_<GrooveEngine::GrooveAnalysis>(m, "GrooveAnalysis")
        .def(py::init<>())
        .def_readonly("current_tempo", &GrooveEngine::GrooveAnalysis::currentTempo)
        .def_readonly("tempo_confidence", &GrooveEngine::GrooveAnalysis::tempoConfidence)
        .def_readonly("onset_positions", &GrooveEngine::GrooveAnalysis::onsetPositions)
        .def_readonly("onset_strengths", &GrooveEngine::GrooveAnalysis::onsetStrengths)
        .def_readonly("time_signature_num", &GrooveEngine::GrooveAnalysis::timeSignatureNum)
        .def_readonly("time_signature_den", &GrooveEngine::GrooveAnalysis::timeSignatureDen)
        .def_readonly("swing", &GrooveEngine::GrooveAnalysis::swing)
        .def("__repr__", [](const GrooveEngine::GrooveAnalysis& g) {
            return "GrooveAnalysis(tempo=" + std::to_string(g.currentTempo) + 
                   " BPM, confidence=" + std::to_string(g.tempoConfidence) + ")";
        });
    
    // GrooveEngine configuration
    py::class_<GrooveEngine::Config>(m, "GrooveConfig")
        .def(py::init<>())
        .def_readwrite("sample_rate", &GrooveEngine::Config::sampleRate)
        .def_readwrite("hop_size", &GrooveEngine::Config::hopSize)
        .def_readwrite("min_tempo", &GrooveEngine::Config::minTempo)
        .def_readwrite("max_tempo", &GrooveEngine::Config::maxTempo)
        .def_readwrite("enable_quantization", &GrooveEngine::Config::enableQuantization)
        .def_readwrite("quantization_strength", &GrooveEngine::Config::quantizationStrength);
    
    // GrooveEngine
    py::class_<GrooveEngine>(m, "GrooveEngine")
        .def(py::init<const GrooveEngine::Config&>(),
            py::arg("config") = GrooveEngine::Config{})
        .def("process_audio", [](GrooveEngine& self, py::array_t<float> buffer) {
            py::buffer_info info = buffer.request();
            if (info.ndim != 1) {
                throw std::runtime_error("Audio buffer must be 1-dimensional");
            }
            self.processAudio(static_cast<float*>(info.ptr), info.shape[0]);
        }, py::arg("buffer"),
        "Process audio buffer for groove analysis")
        .def("get_analysis", &GrooveEngine::getAnalysis,
            py::return_value_policy::copy,
            "Get current groove analysis results")
        .def("quantize_to_grid", &GrooveEngine::quantizeToGrid,
            py::arg("timestamp"),
            "Quantize timestamp to rhythmic grid")
        .def("apply_swing", &GrooveEngine::applySwing,
            py::arg("position"),
            "Apply swing to position")
        .def("update_config", &GrooveEngine::updateConfig,
            py::arg("config"),
            "Update engine configuration")
        .def("reset", &GrooveEngine::reset,
            "Reset analysis state");
    
    // RhythmQuantizer configuration
    py::class_<RhythmQuantizer::Config>(m, "QuantizerConfig")
        .def(py::init<>())
        .def_readwrite("resolution", &RhythmQuantizer::Config::resolution)
        .def_readwrite("strength", &RhythmQuantizer::Config::strength)
        .def_readwrite("enable_swing", &RhythmQuantizer::Config::enableSwing)
        .def_readwrite("swing_amount", &RhythmQuantizer::Config::swingAmount)
        .def_readwrite("time_signature_num", &RhythmQuantizer::Config::timeSignatureNum)
        .def_readwrite("time_signature_den", &RhythmQuantizer::Config::timeSignatureDen);
}
