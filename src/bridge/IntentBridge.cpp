#include "IntentBridge.h"

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <functional>
#include <algorithm>
#include <map>

// JSON parsing (simple implementation or use a library)
// For now, we'll use string manipulation for basic parsing
#include <regex>

namespace kelly {

IntentBridge::IntentBridge()
    : available_(false)
    , processIntentFunc_(nullptr)
    , convertToCppFunc_(nullptr)
    , convertToPythonFunc_(nullptr)
    , validateResultFunc_(nullptr)
    , getRuleBreaksFunc_(nullptr)
{
    available_ = initializePython();
}

IntentBridge::~IntentBridge() {
    shutdownPython();
}

bool IntentBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
    // Check if Python is already initialized
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "IntentBridge: Failed to initialize Python" << std::endl;
            return false;
        }
    }

    // Import the intent bridge module
    PyObject* module = PyImport_ImportModule("music_brain.session.intent_bridge");
    if (!module) {
        PyErr_Print();
        std::cerr << "IntentBridge: Failed to import intent_bridge module" << std::endl;
        return false;
    }

    // Get function pointers
    processIntentFunc_ = PyObject_GetAttrString(module, "process_intent");
    convertToCppFunc_ = PyObject_GetAttrString(module, "convert_to_cpp_intent");
    convertToPythonFunc_ = PyObject_GetAttrString(module, "convert_to_python_intent");
    validateResultFunc_ = PyObject_GetAttrString(module, "validate_result");
    getRuleBreaksFunc_ = PyObject_GetAttrString(module, "get_suggested_rule_breaks");

    Py_DECREF(module);

    // Not all functions are required - processIntent is the main one
    if (!processIntentFunc_) {
        std::cerr << "IntentBridge: Failed to get process_intent function" << std::endl;
        return false;
    }

    return true;
#else
    // Python not available - bridge will use fallback
    std::cerr << "IntentBridge: Python not available, intent processing disabled" << std::endl;
    return false;
#endif
}

void IntentBridge::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    if (processIntentFunc_) {
        Py_DECREF(static_cast<PyObject*>(processIntentFunc_));
        processIntentFunc_ = nullptr;
    }
    if (convertToCppFunc_) {
        Py_DECREF(static_cast<PyObject*>(convertToCppFunc_));
        convertToCppFunc_ = nullptr;
    }
    if (convertToPythonFunc_) {
        Py_DECREF(static_cast<PyObject*>(convertToPythonFunc_));
        convertToPythonFunc_ = nullptr;
    }
    if (validateResultFunc_) {
        Py_DECREF(static_cast<PyObject*>(validateResultFunc_));
        validateResultFunc_ = nullptr;
    }
    if (getRuleBreaksFunc_) {
        Py_DECREF(static_cast<PyObject*>(getRuleBreaksFunc_));
        getRuleBreaksFunc_ = nullptr;
    }
#endif
    resultCache_.clear();
}

std::string IntentBridge::processIntent(const std::string& intentJson) {
    if (!available_) {
        return "{}";
    }

    // Check cache first
    std::string cacheKey = hashIntent(intentJson);
    std::string cached = getCachedResult(cacheKey);
    if (!cached.empty()) {
        return cached;
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(processIntentFunc_);
    if (!func) {
        return "{}";
    }

    PyObject* args = PyTuple_New(1);
    PyObject* intentStr = PyUnicode_FromString(intentJson.c_str());

    PyTuple_SetItem(args, 0, intentStr);

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        return "{}";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : "{}";
    } else {
        resultStr = "{}";
    }

    // Cache the result
    cacheResult(cacheKey, resultStr);

    Py_DECREF(result);
    return resultStr;
#else
    return "{}";
#endif
}

IntentResult IntentBridge::convertToCppIntent(const std::string& intentJson) {
    std::string resultJson = processIntent(intentJson);
    return parseIntentResult(resultJson);
}

std::string IntentBridge::convertToPythonIntent(const IntentResult& intent) {
    if (!available_ || !convertToPythonFunc_) {
        // Fallback: create basic Python intent format
        std::ostringstream oss;
        oss << R"({"phase_1": {"mood_primary": ")" << intent.sourceWound.primaryEmotion.name
            << R"(", "vulnerability_scale": 0.5}, "phase_2": {"technical_key": ")"
            << intent.key << R"(", "technical_mode": ")" << intent.mode << R"("}})";
        return oss.str();
    }

#ifdef PYTHON_AVAILABLE
    // Convert IntentResult to JSON first (simplified)
    std::ostringstream intentJson;
    intentJson << R"({"key": ")" << intent.key
               << R"(", "mode": ")" << intent.mode
               << R"(", "tempoBpm": )" << intent.tempoBpm
               << R"(, "emotion": ")" << intent.sourceWound.primaryEmotion.name << R"("})";

    PyObject* func = static_cast<PyObject*>(convertToPythonFunc_);
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(intentJson.str().c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return intentJson.str();
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : intentJson.str();
    } else {
        resultStr = intentJson.str();
    }

    Py_DECREF(result);
    return resultStr;
#else
    return "{}";
#endif
}

bool IntentBridge::validateResult(const std::string& resultJson) {
    if (!available_ || !validateResultFunc_) {
        // Basic validation: check if it's valid JSON structure
        return !resultJson.empty() && resultJson != "{}";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(validateResultFunc_);
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(resultJson.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return false;
    }

    bool valid = PyObject_IsTrue(result);
    Py_DECREF(result);
    return valid;
#else
    return false;
#endif
}

std::string IntentBridge::getSuggestedRuleBreaks(const std::string& emotion) {
    if (!available_ || !getRuleBreaksFunc_) {
        return R"({"rule_breaks": []})";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(getRuleBreaksFunc_);
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(emotion.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return R"({"rule_breaks": []})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"rule_breaks": []})";
    } else {
        resultStr = R"({"rule_breaks": []})";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"rule_breaks": []})";
#endif
}

std::string IntentBridge::getCachedResult(const std::string& cacheKey) {
    auto it = resultCache_.find(cacheKey);
    if (it == resultCache_.end()) {
        return "";
    }

    // Check if cache entry is still valid
    auto now = std::chrono::steady_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - it->second.timestamp
    ).count();

    if (age > CACHE_TTL_MS) {
        resultCache_.erase(it);
        return "";
    }

    return it->second.resultJson;
}

void IntentBridge::cacheResult(
    const std::string& cacheKey,
    const std::string& resultJson
) {
    CachedResult cached;
    cached.resultJson = resultJson;
    cached.intentHash = cacheKey;
    cached.timestamp = std::chrono::steady_clock::now();

    resultCache_[cacheKey] = cached;

    // Limit cache size (keep most recent 50 entries)
    if (resultCache_.size() > 50) {
        // Remove oldest entry
        auto oldest = resultCache_.begin();
        for (auto it = resultCache_.begin(); it != resultCache_.end(); ++it) {
            if (it->second.timestamp < oldest->second.timestamp) {
                oldest = it;
            }
        }
        resultCache_.erase(oldest);
    }
}

std::string IntentBridge::hashIntent(const std::string& intentJson) {
    // Simple hash function for cache key
    std::hash<std::string> hasher;
    return std::to_string(hasher(intentJson));
}

IntentResult IntentBridge::parseIntentResult(const std::string& resultJson) {
    IntentResult result;

    // Simple JSON parsing (in production, use a proper JSON library)
    // For now, extract key fields using regex
    // Using delimiter to handle quotes in raw string literals
    std::regex keyRegex(R"delim("key"\s*:\s*"([^"]+)")delim");
    std::regex modeRegex(R"delim("mode"\s*:\s*"([^"]+)")delim");
    std::regex tempoRegex(R"delim("tempoBpm"\s*:\s*(\d+))delim");
    std::regex emotionRegex(R"delim("emotion"\s*:\s*"([^"]+)")delim");

    std::smatch match;
    if (std::regex_search(resultJson, match, keyRegex)) {
        result.key = match[1].str();
    }
    if (std::regex_search(resultJson, match, modeRegex)) {
        result.mode = match[1].str();
    }
    if (std::regex_search(resultJson, match, tempoRegex)) {
        result.tempoBpm = std::stoi(match[1].str());
    }

    // Parse chord progression
    std::regex chordRegex(R"delim("chordProgression"\s*:\s*\[(.*?)\])delim");
    if (std::regex_search(resultJson, match, chordRegex)) {
        std::string chordsStr = match[1].str();
        std::regex chordNameRegex(R"delim("([^"]+)")delim");
        std::sregex_iterator iter(chordsStr.begin(), chordsStr.end(), chordNameRegex);
        std::sregex_iterator end;
        for (; iter != end; ++iter) {
            result.chordProgression.push_back((*iter)[1].str());
        }
    }

    // Default values if not found
    if (result.key.empty()) result.key = "C";
    if (result.mode.empty()) result.mode = "major";
    if (result.tempoBpm == 0) result.tempoBpm = 120;

    return result;
}

} // namespace kelly
