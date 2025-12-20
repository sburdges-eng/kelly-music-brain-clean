/**
 * kelly_bridge.cpp
 *
 * Python-C++ bridge using pybind11
 * Exposes Kelly MIDI Companion C++ functionality to Python
 */

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include Types.h first to establish the correct Chord and MidiNote structures
#include "common/Types.h"
#include "engine/EmotionThesaurus.h"
#include "engine/IntentPipeline.h"

// Note: KellyBrain and EmotionMapper are not included due to type conflicts
// The bridge provides IntentPipeline and EmotionThesaurus APIs instead

namespace py = pybind11;
using namespace kelly;

// =============================================================================
// Python bindings for Kelly MIDI Companion
// =============================================================================

PYBIND11_MODULE(kelly_bridge, m) {
  m.doc() = "Kelly MIDI Companion - Python-C++ Bridge for Advanced Features";

  // =========================================================================
  // Enums
  // =========================================================================

  py::enum_<EmotionCategory>(m, "EmotionCategory")
      .value("Joy", EmotionCategory::Joy)
      .value("Sadness", EmotionCategory::Sadness)
      .value("Anger", EmotionCategory::Anger)
      .value("Fear", EmotionCategory::Fear)
      .value("Surprise", EmotionCategory::Surprise)
      .value("Disgust", EmotionCategory::Disgust)
      .value("Trust", EmotionCategory::Trust)
      .value("Anticipation", EmotionCategory::Anticipation);

  py::enum_<RuleBreakType>(m, "RuleBreakType")
      .value("None", RuleBreakType::None)
      .value("ModalMixture", RuleBreakType::ModalMixture)
      .value("ParallelMotion", RuleBreakType::ParallelMotion)
      .value("UnresolvedTension", RuleBreakType::UnresolvedTension)
      .value("CrossRhythm", RuleBreakType::CrossRhythm)
      .value("RegisterShift", RuleBreakType::RegisterShift)
      .value("DynamicContrast", RuleBreakType::DynamicContrast)
      .value("HarmonicAmbiguity", RuleBreakType::HarmonicAmbiguity);

  // =========================================================================
  // Wound
  // =========================================================================

  py::class_<Wound>(m, "Wound")
      .def(py::init<>())
      .def_readwrite("description", &Wound::description)
      .def_readwrite("desire", &Wound::desire)
      .def_readwrite("urgency", &Wound::urgency)
      .def_readwrite("expression", &Wound::expression)
      .def_readwrite("intensity", &Wound::intensity)
      .def_readwrite("source", &Wound::source)
      .def_readwrite("primaryEmotion", &Wound::primaryEmotion)
      .def_readwrite("secondaryEmotion", &Wound::secondaryEmotion)
      .def("__repr__", [](const Wound &w) {
        return "Wound(description='" + w.description +
               "', intensity=" + std::to_string(w.intensity) + ", source='" +
               w.source + "')";
      });

  // =========================================================================
  // EmotionNode
  // =========================================================================

  py::class_<EmotionNode>(m, "EmotionNode")
      .def_readonly("id", &EmotionNode::id)
      .def_readonly("name", &EmotionNode::name)
      .def_readonly("category", &EmotionNode::category)
      .def_readonly("valence", &EmotionNode::valence)
      .def_readonly("arousal", &EmotionNode::arousal)
      .def_readonly("intensity", &EmotionNode::intensity)
      .def_readonly("dominance", &EmotionNode::dominance)
      .def_readonly("relatedEmotions", &EmotionNode::relatedEmotions)
      .def("__repr__", [](const EmotionNode &e) {
        return "EmotionNode(id=" + std::to_string(e.id) + ", name='" + e.name +
               "', valence=" + std::to_string(e.valence) +
               ", arousal=" + std::to_string(e.arousal) +
               ", intensity=" + std::to_string(e.intensity) + ")";
      });

  // =========================================================================
  // RuleBreak
  // =========================================================================

  py::class_<RuleBreak>(m, "RuleBreak")
      .def_readonly("type", &RuleBreak::type)
      .def_property_readonly("severity",
                             [](const RuleBreak &rb) { return rb.severity(); })
      .def_readonly("description", &RuleBreak::description)
      .def_property_readonly("reason",
                             [](const RuleBreak &rb) { return rb.reason(); })
      .def_readonly("justification", &RuleBreak::justification)
      .def_readonly("intensity", &RuleBreak::intensity)
      .def("__repr__", [](const RuleBreak &rb) {
        return "RuleBreak(type=" + std::to_string(static_cast<int>(rb.type)) +
               ", severity=" + std::to_string(rb.severity()) +
               ", description='" + rb.description + "')";
      });

  // =========================================================================
  // MusicalParameters - Removed (not in current Types.h)
  // IntentResult contains musical parameters directly as fields
  // =========================================================================

  // =========================================================================
  // IntentResult
  // =========================================================================

  py::class_<IntentResult>(m, "IntentResult")
      .def_property_readonly(
          "wound",
          [](const IntentResult &ir) -> const Wound & { return ir.wound(); })
      .def_readonly("emotion", &IntentResult::emotion)
      .def_readonly("ruleBreaks", &IntentResult::ruleBreaks)
      .def_readonly("mode", &IntentResult::mode)
      .def_readonly("tempo", &IntentResult::tempo)
      .def_readonly("tempoBpm", &IntentResult::tempoBpm)
      .def_readonly("dynamicRange", &IntentResult::dynamicRange)
      .def_readonly("allowDissonance", &IntentResult::allowDissonance)
      .def_readonly("syncopationLevel", &IntentResult::syncopationLevel)
      .def_readonly("humanization", &IntentResult::humanization)
      .def_readonly("sourceWound", &IntentResult::sourceWound)
      .def("__repr__", [](const IntentResult &ir) {
        return "IntentResult(emotion='" + ir.emotion.name +
               "', ruleBreaks=" + std::to_string(ir.ruleBreaks.size()) + ")";
      });

  // =========================================================================
  // MidiNote
  // =========================================================================

  py::class_<MidiNote>(m, "MidiNote")
      .def(py::init<>())
      .def_readwrite("pitch", &MidiNote::pitch)
      .def_readwrite("velocity", &MidiNote::velocity)
      .def_readwrite("startBeat", &MidiNote::startBeat)
      .def_readwrite("duration", &MidiNote::duration)
      .def("__repr__", [](const MidiNote &n) {
        return "MidiNote(pitch=" + std::to_string(n.pitch) +
               ", velocity=" + std::to_string(n.velocity) +
               ", startBeat=" + std::to_string(n.startBeat) +
               ", duration=" + std::to_string(n.duration) + ")";
      });

  // =========================================================================
  // Chord
  // =========================================================================

  py::class_<Chord>(m, "Chord")
      .def_readonly("pitches", &Chord::pitches)
      .def_readonly("name", &Chord::name)
      .def_readonly("startBeat", &Chord::startBeat)
      .def_readonly("duration", &Chord::duration)
      .def("__repr__", [](const Chord &c) {
        return "Chord(name='" + c.name +
               "', pitches=" + std::to_string(c.pitches.size()) +
               ", startBeat=" + std::to_string(c.startBeat) + ")";
      });

  // =========================================================================
  // GeneratedMidi
  // =========================================================================

  py::class_<GeneratedMidi>(m, "GeneratedMidi")
      .def_readonly("chords", &GeneratedMidi::chords)
      .def_readonly("melody", &GeneratedMidi::melody)
      .def_readonly("bass", &GeneratedMidi::bass)
      .def_readonly("counterMelody", &GeneratedMidi::counterMelody)
      .def_readonly("pad", &GeneratedMidi::pad)
      .def_readonly("strings", &GeneratedMidi::strings)
      .def_readonly("fills", &GeneratedMidi::fills)
      .def_readonly("lengthInBeats", &GeneratedMidi::lengthInBeats)
      .def_readonly("bpm", &GeneratedMidi::bpm)
      .def("__repr__", [](const GeneratedMidi &gm) {
        return "GeneratedMidi(chords=" + std::to_string(gm.chords.size()) +
               ", melody=" + std::to_string(gm.melody.size()) +
               ", bass=" + std::to_string(gm.bass.size()) +
               ", counterMelody=" + std::to_string(gm.counterMelody.size()) +
               ", pad=" + std::to_string(gm.pad.size()) +
               ", strings=" + std::to_string(gm.strings.size()) +
               ", fills=" + std::to_string(gm.fills.size()) +
               ", lengthInBeats=" + std::to_string(gm.lengthInBeats) + ")";
      });

  // =========================================================================
  // KellyBrain - Main API
  // =========================================================================
  // Note: KellyBrain temporarily disabled due to MidiGenerator.h conflicts
  // Use IntentPipeline and EmotionThesaurus directly instead
  // TODO: Fix MidiGenerator.h structure conflicts and re-enable KellyBrain

  // =========================================================================
  // SideA / SideB
  // =========================================================================

  py::class_<SideA>(m, "SideA")
      .def(py::init<>())
      .def_readwrite("description", &SideA::description)
      .def_readwrite("intensity", &SideA::intensity)
      .def_readwrite("emotionId", &SideA::emotionId);

  py::class_<SideB>(m, "SideB")
      .def(py::init<>())
      .def_readwrite("description", &SideB::description)
      .def_readwrite("intensity", &SideB::intensity)
      .def_readwrite("emotionId", &SideB::emotionId);

  // =========================================================================
  // IntentPipeline - Advanced API
  // =========================================================================

  py::class_<IntentPipeline>(m, "IntentPipeline")
      .def(py::init<>())
      .def("process", &IntentPipeline::process, py::arg("wound"),
           "Process a complete intent from wound to musical parameters")
      .def("process_journey", &IntentPipeline::processJourney,
           py::arg("current"), py::arg("desired"),
           "Process Side A (current state) and Side B (desired state) to "
           "create a musical journey")
      .def(
          "thesaurus",
          [](IntentPipeline &self) -> const EmotionThesaurus & {
            return self.thesaurus();
          },
          py::return_value_policy::reference_internal,
          "Get direct access to the thesaurus");

  // =========================================================================
  // EmotionThesaurus - Query API
  // =========================================================================

  py::class_<EmotionThesaurus>(m, "EmotionThesaurus")
      .def(
          "find_by_id",
          [](EmotionThesaurus &self, int id) -> py::object {
            auto node = self.findById(id);
            if (node.has_value()) {
              return py::cast(node.value());
            }
            return py::none();
          },
          py::arg("id"), "Find emotion by ID")
      .def(
          "find_by_name",
          [](EmotionThesaurus &self, const std::string &name) -> py::object {
            auto node = self.findByName(name);
            if (node.has_value()) {
              return py::cast(node.value());
            }
            return py::none();
          },
          py::arg("name"), "Find emotion by name")
      .def(
          "find_nearest",
          [](EmotionThesaurus &self, float v, float a, float i) {
            return self.findNearest(v, a, i);
          },
          py::arg("valence"), py::arg("arousal"), py::arg("intensity"),
          "Find nearest emotion by valence/arousal/intensity")
      .def(
          "find_nearest_vad",
          [](EmotionThesaurus &self, float v, float a, float d) {
            return self.findNearestVAD(v, a, d);
          },
          py::arg("valence"), py::arg("arousal"), py::arg("dominance"),
          "Find nearest emotion by valence/arousal/dominance")
      .def(
          "find_related",
          [](EmotionThesaurus &self, int emotionId) {
            return self.findRelated(emotionId);
          },
          py::arg("emotion_id"), "Find emotions related to a given emotion")
      .def(
          "find_by_category",
          [](EmotionThesaurus &self, EmotionCategory cat) {
            return self.findByCategory(cat);
          },
          py::arg("category"), "Get all emotions in a category")
      .def("size", &EmotionThesaurus::size,
           "Get total number of emotions in thesaurus");

  // =========================================================================
  // Utility Functions
  // =========================================================================

  // Utility functions - inline implementations to avoid Kelly.h dependency
  m.def(
      "midi_note_to_name",
      [](int noteNumber) -> std::string {
        static const char *noteNames[] = {"C",  "C#", "D",  "D#", "E",  "F",
                                          "F#", "G",  "G#", "A",  "A#", "B"};
        int octave = (noteNumber / 12) - 1;
        int noteIndex = noteNumber % 12;
        return std::string(noteNames[noteIndex]) + std::to_string(octave);
      },
      py::arg("note_number"),
      "Convert MIDI note number to note name (e.g., 60 -> 'C4')");

  m.def(
      "note_name_to_midi",
      [](const std::string &name) -> int {
        static const std::map<std::string, int> noteOffsets = {
            {"C", 0},  {"C#", 1}, {"Db", 1},  {"D", 2},   {"D#", 3}, {"Eb", 3},
            {"E", 4},  {"F", 5},  {"F#", 6},  {"Gb", 6},  {"G", 7},  {"G#", 8},
            {"Ab", 8}, {"A", 9},  {"A#", 10}, {"Bb", 10}, {"B", 11}};
        std::string notePart;
        int octave = 4;
        for (size_t i = 0; i < name.size(); ++i) {
          if (std::isdigit(name[i]) || name[i] == '-') {
            notePart = name.substr(0, i);
            octave = std::stoi(name.substr(i));
            break;
          }
        }
        if (notePart.empty())
          notePart = name;
        auto it = noteOffsets.find(notePart);
        return (it != noteOffsets.end()) ? (octave + 1) * 12 + it->second : 60;
      },
      py::arg("name"), "Convert note name to MIDI number (e.g., 'C4' -> 60)");

  m.def(
      "category_to_string",
      [](EmotionCategory category) -> std::string {
        switch (category) {
        case EmotionCategory::Joy:
          return "Joy";
        case EmotionCategory::Sadness:
          return "Sadness";
        case EmotionCategory::Anger:
          return "Anger";
        case EmotionCategory::Fear:
          return "Fear";
        case EmotionCategory::Surprise:
          return "Surprise";
        case EmotionCategory::Disgust:
          return "Disgust";
        case EmotionCategory::Trust:
          return "Trust";
        case EmotionCategory::Anticipation:
          return "Anticipation";
        default:
          return "Unknown";
        }
      },
      py::arg("category"), "Get category name as string");

  // =========================================================================
  // Module Info
  // =========================================================================

  m.attr("__version__") = "2.0.0";
  m.attr("__author__") = "Kelly Project";
}
