#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "penta/harmony/HarmonyEngine.h"
#include "penta/harmony/ChordAnalyzer.h"
#include "penta/harmony/ScaleDetector.h"
#include "penta/harmony/VoiceLeading.h"
#include "penta/common/RTTypes.h"

namespace py = pybind11;
using namespace penta;
using namespace penta::harmony;

void bind_harmony(py::module_& m) {
    // Note structure
    py::class_<Note>(m, "Note")
        .def(py::init<>())
        .def(py::init<uint8_t, uint8_t, uint8_t, uint64_t>(),
            py::arg("pitch"), py::arg("velocity"), 
            py::arg("channel") = 0, py::arg("timestamp") = 0)
        .def_readwrite("pitch", &Note::pitch)
        .def_readwrite("velocity", &Note::velocity)
        .def_readwrite("channel", &Note::channel)
        .def_readwrite("timestamp", &Note::timestamp)
        .def("__repr__", [](const Note& n) {
            return "Note(pitch=" + std::to_string(n.pitch) + 
                   ", velocity=" + std::to_string(n.velocity) + ")";
        });
    
    // Chord structure
    py::class_<Chord>(m, "Chord")
        .def(py::init<>())
        .def_readonly("root", &Chord::root)
        .def_readonly("quality", &Chord::quality)
        .def_readonly("confidence", &Chord::confidence)
        .def_property_readonly("pitch_classes", [](const Chord& c) {
            std::vector<int> pcs;
            for (int i = 0; i < 12; ++i) {
                if (c.pitchClass[i]) pcs.push_back(i);
            }
            return pcs;
        })
        .def("__repr__", [](const Chord& c) {
            return "Chord(root=" + std::to_string(c.root) + 
                   ", quality=" + std::to_string(c.quality) +
                   ", confidence=" + std::to_string(c.confidence) + ")";
        });
    
    // Scale structure
    py::class_<Scale>(m, "Scale")
        .def(py::init<>())
        .def_readonly("tonic", &Scale::tonic)
        .def_readonly("mode", &Scale::mode)
        .def_readonly("confidence", &Scale::confidence)
        .def_property_readonly("degrees", [](const Scale& s) {
            std::vector<int> degs;
            for (int i = 0; i < 12; ++i) {
                if (s.degrees[i]) degs.push_back(i);
            }
            return degs;
        })
        .def("__repr__", [](const Scale& s) {
            return "Scale(tonic=" + std::to_string(s.tonic) + 
                   ", mode=" + std::to_string(s.mode) +
                   ", confidence=" + std::to_string(s.confidence) + ")";
        });
    
    // HarmonyEngine configuration
    py::class_<HarmonyEngine::Config>(m, "HarmonyConfig")
        .def(py::init<>())
        .def_readwrite("sample_rate", &HarmonyEngine::Config::sampleRate)
        .def_readwrite("analysis_window_size", &HarmonyEngine::Config::analysisWindowSize)
        .def_readwrite("enable_voice_leading", &HarmonyEngine::Config::enableVoiceLeading)
        .def_readwrite("enable_scale_detection", &HarmonyEngine::Config::enableScaleDetection)
        .def_readwrite("confidence_threshold", &HarmonyEngine::Config::confidenceThreshold);
    
    // HarmonyEngine
    py::class_<HarmonyEngine>(m, "HarmonyEngine")
        .def(py::init<const HarmonyEngine::Config&>(),
            py::arg("config") = HarmonyEngine::Config{})
        .def("process_notes", [](HarmonyEngine& self, const std::vector<Note>& notes) {
            self.processNotes(notes.data(), notes.size());
        }, py::arg("notes"),
        "Process MIDI notes for harmony analysis")
        .def("get_current_chord", &HarmonyEngine::getCurrentChord,
            py::return_value_policy::copy,
            "Get currently detected chord")
        .def("get_current_scale", &HarmonyEngine::getCurrentScale,
            py::return_value_policy::copy,
            "Get currently detected scale")
        .def("suggest_voice_leading", &HarmonyEngine::suggestVoiceLeading,
            py::arg("target_chord"), py::arg("current_voices"),
            "Get voice leading suggestions for target chord")
        .def("update_config", &HarmonyEngine::updateConfig,
            py::arg("config"),
            "Update engine configuration")
        .def("get_chord_history", &HarmonyEngine::getChordHistory,
            py::arg("max_count") = 100,
            "Get chord analysis history")
        .def("get_scale_history", &HarmonyEngine::getScaleHistory,
            py::arg("max_count") = 100,
            "Get scale detection history");
    
    // VoiceLeading configuration
    py::class_<VoiceLeading::Config>(m, "VoiceLeadingConfig")
        .def(py::init<>())
        .def_readwrite("max_voice_distance", &VoiceLeading::Config::maxVoiceDistance)
        .def_readwrite("parallel_penalty", &VoiceLeading::Config::parallelPenalty)
        .def_readwrite("contrary_bonus", &VoiceLeading::Config::contraryBonus)
        .def_readwrite("allow_voice_crossing", &VoiceLeading::Config::allowVoiceCrossing);
}
