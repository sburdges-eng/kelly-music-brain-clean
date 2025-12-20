#include "MusicTheoryBridge.h"

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <regex>

namespace kelly {

MusicTheoryBridge::MusicTheoryBridge()
    : available_(false)
    , brain_(nullptr)
    , explainConceptFunc_(nullptr)
    , generateExerciseFunc_(nullptr)
    , provideFeedbackFunc_(nullptr)
    , createLessonPlanFunc_(nullptr)
{
    available_ = initializePython();
}

MusicTheoryBridge::~MusicTheoryBridge() {
    shutdownPython();
}

bool MusicTheoryBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
    // Check if Python is already initialized
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "MusicTheoryBridge: Failed to initialize Python" << std::endl;
            return false;
        }
    }

    // Import the music theory bridge module
    PyObject* module = PyImport_ImportModule("music_brain.learning.music_theory_bridge");
    if (!module) {
        PyErr_Print();
        std::cerr << "MusicTheoryBridge: Failed to import music_theory_bridge module" << std::endl;
        // Fallback: bridge will use C++ MusicTheoryBrain directly
        return false;
    }

    // Get function pointers
    explainConceptFunc_ = PyObject_GetAttrString(module, "explain_concept");
    generateExerciseFunc_ = PyObject_GetAttrString(module, "generate_exercise");
    provideFeedbackFunc_ = PyObject_GetAttrString(module, "provide_feedback");
    createLessonPlanFunc_ = PyObject_GetAttrString(module, "create_lesson_plan");

    Py_DECREF(module);

    if (!explainConceptFunc_) {
        std::cerr << "MusicTheoryBridge: Failed to get function pointers" << std::endl;
        return false;
    }

    return true;
#else
    // Python not available - bridge will use C++ MusicTheoryBrain directly
    std::cerr << "MusicTheoryBridge: Python not available, using C++ MusicTheoryBrain only" << std::endl;
    return false;
#endif
}

void MusicTheoryBridge::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    if (Py_IsInitialized()) {
        if (explainConceptFunc_) {
            Py_DECREF(static_cast<PyObject*>(explainConceptFunc_));
            explainConceptFunc_ = nullptr;
        }
        if (generateExerciseFunc_) {
            Py_DECREF(static_cast<PyObject*>(generateExerciseFunc_));
            generateExerciseFunc_ = nullptr;
        }
        if (provideFeedbackFunc_) {
            Py_DECREF(static_cast<PyObject*>(provideFeedbackFunc_));
            provideFeedbackFunc_ = nullptr;
        }
        if (createLessonPlanFunc_) {
            Py_DECREF(static_cast<PyObject*>(createLessonPlanFunc_));
            createLessonPlanFunc_ = nullptr;
        }
    }
#endif
}

MusicTheoryBridge::Explanation MusicTheoryBridge::explainConcept(
    const std::string& conceptName,
    const std::string& style,
    int userLevel)
{
    // Check cache first
    std::string cacheKey = hashRequest(conceptName, style, userLevel);
    auto cached = getCachedExplanation(cacheKey);
    if (!cached.text.empty()) {
        return cached;
    }

    // Try Python bridge first
    if (available_ && explainConceptFunc_) {
#ifdef PYTHON_AVAILABLE
        PyObject* args = PyTuple_New(3);
        PyObject* conceptStr = PyUnicode_FromString(conceptName.c_str());
        PyObject* styleStr = PyUnicode_FromString(style.c_str());
        PyObject* levelInt = PyLong_FromLong(userLevel);

        PyTuple_SetItem(args, 0, conceptStr);
        PyTuple_SetItem(args, 1, styleStr);
        PyTuple_SetItem(args, 2, levelInt);

        PyObject* result = PyObject_CallObject(static_cast<PyObject*>(explainConceptFunc_), args);
        Py_DECREF(args);

        if (result && PyDict_Check(result)) {
            Explanation explanation;
            explanation.depth = midikompanion::theory::ExplanationDepth::Intermediate;
            PyObject* textObj = PyDict_GetItemString(result, "text");
            if (textObj && PyUnicode_Check(textObj)) {
                explanation.text = PyUnicode_AsUTF8(textObj);
            }
            PyObject* styleObj = PyDict_GetItemString(result, "style");
            if (styleObj && PyUnicode_Check(styleObj)) {
                std::string styleStr = PyUnicode_AsUTF8(styleObj);
                if (styleStr == "intuitive") {
                    explanation.style = midikompanion::theory::ExplanationType::Intuitive;
                } else if (styleStr == "mathematical") {
                    explanation.style = midikompanion::theory::ExplanationType::Mathematical;
                } else if (styleStr == "historical") {
                    explanation.style = midikompanion::theory::ExplanationType::Historical;
                } else if (styleStr == "acoustic") {
                    explanation.style = midikompanion::theory::ExplanationType::Acoustic;
                }
            }
            Py_DECREF(result);
            cacheExplanation(cacheKey, explanation);
            return explanation;
        }
        if (result) Py_DECREF(result);
#endif
    }

    // Fallback to C++ MusicTheoryBrain
    if (brain_) {
        const auto& knowledge = brain_->getKnowledge();
        auto conceptNode = knowledge.getConcept(conceptName);
        if (conceptNode.has_value()) {
            Explanation explanation;
            explanation.depth = midikompanion::theory::ExplanationDepth::Intermediate;
            explanation.style = midikompanion::theory::ExplanationType::Intuitive;

            // Convert style string to ExplanationType
            midikompanion::theory::ExplanationType targetStyle = midikompanion::theory::ExplanationType::Intuitive;
            if (style == "mathematical") targetStyle = midikompanion::theory::ExplanationType::Mathematical;
            else if (style == "historical") targetStyle = midikompanion::theory::ExplanationType::Historical;
            else if (style == "acoustic") targetStyle = midikompanion::theory::ExplanationType::Acoustic;

            // Access explanations as map
            auto it = conceptNode->explanations.find(targetStyle);
            if (it != conceptNode->explanations.end()) {
                explanation.text = it->second;
                explanation.style = it->first;
            } else if (!conceptNode->explanations.empty()) {
                // Use first available
                explanation.text = conceptNode->explanations.begin()->second;
                explanation.style = conceptNode->explanations.begin()->first;
            }
            cacheExplanation(cacheKey, explanation);
            return explanation;
        }
    }

    // Default fallback
    Explanation explanation;
    explanation.text = "Explanation not available for: " + conceptName;
    explanation.style = midikompanion::theory::ExplanationType::Intuitive;
    explanation.depth = midikompanion::theory::ExplanationDepth::Intermediate;
    return explanation;
}

midikompanion::theory::Exercise MusicTheoryBridge::generateExercise(
    const std::string& conceptName,
    const std::string& userProfileJson)
{
    midikompanion::theory::Exercise exercise;
    exercise.conceptName = conceptName;

    // Try Python bridge first
    if (available_ && generateExerciseFunc_) {
#ifdef PYTHON_AVAILABLE
        PyObject* args = PyTuple_New(2);
        PyObject* conceptStr = PyUnicode_FromString(conceptName.c_str());
        PyObject* profileStr = PyUnicode_FromString(userProfileJson.c_str());

        PyTuple_SetItem(args, 0, conceptStr);
        PyTuple_SetItem(args, 1, profileStr);

        PyObject* result = PyObject_CallObject(static_cast<PyObject*>(generateExerciseFunc_), args);
        Py_DECREF(args);

        if (result && PyDict_Check(result)) {
            PyObject* questionObj = PyDict_GetItemString(result, "question");
            if (questionObj && PyUnicode_Check(questionObj)) {
                exercise.instruction = PyUnicode_AsUTF8(questionObj);
            }
            PyObject* answerObj = PyDict_GetItemString(result, "answer");
            if (answerObj && PyUnicode_Check(answerObj)) {
                exercise.correctAnswer = PyUnicode_AsUTF8(answerObj);
            }
            Py_DECREF(result);
            return exercise;
        }
        if (result) Py_DECREF(result);
#endif
    }

    // Fallback to C++ MusicTheoryBrain
    if (brain_) {
        const auto& knowledge = brain_->getKnowledge();
        auto conceptNode = knowledge.getConcept(conceptName);
        if (conceptNode.has_value()) {
            exercise.instruction = "What is " + conceptName + "?";
            // Get first available explanation as answer
            if (!conceptNode->explanations.empty()) {
                exercise.correctAnswer = conceptNode->explanations.begin()->second;
            } else {
                exercise.correctAnswer = "Answer not available";
            }
        }
    }

    return exercise;
}

MusicTheoryBridge::Feedback MusicTheoryBridge::provideFeedback(
    const std::string& exerciseJson,
    const std::string& attemptJson)
{
    Feedback feedback;

    // Try Python bridge first
    if (available_ && provideFeedbackFunc_) {
#ifdef PYTHON_AVAILABLE
        PyObject* args = PyTuple_New(2);
        PyObject* exerciseStr = PyUnicode_FromString(exerciseJson.c_str());
        PyObject* attemptStr = PyUnicode_FromString(attemptJson.c_str());

        PyTuple_SetItem(args, 0, exerciseStr);
        PyTuple_SetItem(args, 1, attemptStr);

        PyObject* result = PyObject_CallObject(static_cast<PyObject*>(provideFeedbackFunc_), args);
        Py_DECREF(args);

        if (result && PyDict_Check(result)) {
            PyObject* correctObj = PyDict_GetItemString(result, "is_correct");
            if (correctObj && PyBool_Check(correctObj)) {
                feedback.isCorrect = (correctObj == Py_True);
            }
            PyObject* explanationObj = PyDict_GetItemString(result, "explanation");
            if (explanationObj && PyUnicode_Check(explanationObj)) {
                feedback.explanation = PyUnicode_AsUTF8(explanationObj);
            }
            Py_DECREF(result);
            return feedback;
        }
        if (result) Py_DECREF(result);
#endif
    }

    // Fallback
    feedback.isCorrect = false;
    feedback.explanation = "Feedback not available";
    return feedback;
}

std::string MusicTheoryBridge::createLessonPlan(
    const std::string& conceptName,
    const std::string& userProfileJson)
{
    // Try Python bridge first
    if (available_ && createLessonPlanFunc_) {
#ifdef PYTHON_AVAILABLE
        PyObject* args = PyTuple_New(2);
        PyObject* conceptStr = PyUnicode_FromString(conceptName.c_str());
        PyObject* profileStr = PyUnicode_FromString(userProfileJson.c_str());

        PyTuple_SetItem(args, 0, conceptStr);
        PyTuple_SetItem(args, 1, profileStr);

        PyObject* result = PyObject_CallObject(static_cast<PyObject*>(createLessonPlanFunc_), args);
        Py_DECREF(args);

        if (result && PyUnicode_Check(result)) {
            std::string json = PyUnicode_AsUTF8(result);
            Py_DECREF(result);
            return json;
        }
        if (result) Py_DECREF(result);
#endif
    }

    // Fallback to C++ MusicTheoryBrain
    // FIXME: Temporarily disabled due to struct member mismatches
    /*
    if (brain_) {
        midikompanion::theory::UserProfile profile;
        auto curriculum = brain_->getCustomLearningPath(conceptName, profile);

        // Convert to JSON
        std::ostringstream json;
        json << "{\"concept\":\"" << conceptName << "\",\"lessons\":[";
        for (size_t i = 0; i < curriculum.lessons.size(); ++i) {
            if (i > 0) json << ",";
            json << "{\"step\":" << (i + 1) << ",\"concept\":\""
                 << curriculum.lessons[i].conceptName << "\"}";
        }
        json << "]}";
        return json.str();
    }
    */

    return "{\"error\":\"Lesson plan not available\"}";
}

MusicTheoryBridge::Explanation MusicTheoryBridge::getCachedExplanation(const std::string& cacheKey) {
    auto it = explanationCache_.find(cacheKey);
    if (it != explanationCache_.end()) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - it->second.timestamp).count();
        if (elapsed < CACHE_TTL_MS) {
            return it->second.explanation;
        }
        explanationCache_.erase(it);
    }
    Explanation empty;
    empty.text = "";
    empty.style = midikompanion::theory::ExplanationType::Intuitive;
    empty.depth = midikompanion::theory::ExplanationDepth::Intermediate;
    return empty;
}

void MusicTheoryBridge::cacheExplanation(
    const std::string& cacheKey,
    const Explanation& explanation)
{
    CachedExplanation cached;
    cached.explanation = explanation;
    cached.cacheKey = cacheKey;
    cached.timestamp = std::chrono::steady_clock::now();
    explanationCache_[cacheKey] = cached;
}

std::string MusicTheoryBridge::hashRequest(
    const std::string& conceptName,
    const std::string& style,
    int userLevel)
{
    return conceptName + "|" + style + "|" + std::to_string(userLevel);
}

} // namespace kelly
