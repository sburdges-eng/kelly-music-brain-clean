#include "EngineIntelligenceBridge.h"

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

namespace kelly {

EngineIntelligenceBridge::EngineIntelligenceBridge()
    : available_(false)
    , getEngineSuggestionsFunc_(nullptr)
    , getBatchSuggestionsFunc_(nullptr)
    , recordAppliedFunc_(nullptr)
    , reportEngineStateFunc_(nullptr)
{
    available_ = initializePython();
}

EngineIntelligenceBridge::~EngineIntelligenceBridge() {
    shutdownPython();
}

bool EngineIntelligenceBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
    // Check if Python is already initialized
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "EngineIntelligenceBridge: Failed to initialize Python" << std::endl;
            return false;
        }
    }

    // Import the engine bridge module
    PyObject* module = PyImport_ImportModule("music_brain.intelligence.engine_bridge");
    if (!module) {
        PyErr_Print();
        std::cerr << "EngineIntelligenceBridge: Failed to import engine_bridge module" << std::endl;
        // Fallback: try to create the module if it doesn't exist yet
        // (This will be created in Phase 3)
        return false;
    }

    // Get function pointers
    getEngineSuggestionsFunc_ = PyObject_GetAttrString(module, "get_engine_suggestions");
    getBatchSuggestionsFunc_ = PyObject_GetAttrString(module, "get_batch_engine_suggestions");
    recordAppliedFunc_ = PyObject_GetAttrString(module, "record_suggestion_applied");
    reportEngineStateFunc_ = PyObject_GetAttrString(module, "report_engine_state");

    Py_DECREF(module);

    if (!getEngineSuggestionsFunc_) {
        std::cerr << "EngineIntelligenceBridge: Failed to get function pointers" << std::endl;
        return false;
    }

    return true;
#else
    // Python not available - bridge will use fallback (no-op)
    std::cerr << "EngineIntelligenceBridge: Python not available, engine intelligence disabled" << std::endl;
    return false;
#endif
}

void EngineIntelligenceBridge::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    if (getEngineSuggestionsFunc_) {
        Py_DECREF(static_cast<PyObject*>(getEngineSuggestionsFunc_));
        getEngineSuggestionsFunc_ = nullptr;
    }
    if (getBatchSuggestionsFunc_) {
        Py_DECREF(static_cast<PyObject*>(getBatchSuggestionsFunc_));
        getBatchSuggestionsFunc_ = nullptr;
    }
    if (recordAppliedFunc_) {
        Py_DECREF(static_cast<PyObject*>(recordAppliedFunc_));
        recordAppliedFunc_ = nullptr;
    }
    if (reportEngineStateFunc_) {
        Py_DECREF(static_cast<PyObject*>(reportEngineStateFunc_));
        reportEngineStateFunc_ = nullptr;
    }
#endif
    suggestionCache_.clear();
}

std::string EngineIntelligenceBridge::getEngineSuggestions(
    const std::string& engineType,
    const std::string& currentStateJson
) {
    if (!available_) {
        // Return empty suggestions if Python not available
        return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
    }

    // Check cache first
    std::string cacheKey = engineType + ":" + hashState(currentStateJson);
    auto cacheIt = suggestionCache_.find(cacheKey);
    if (cacheIt != suggestionCache_.end()) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - cacheIt->second.timestamp).count();
        if (elapsed < CACHE_TTL_MS) {
            return cacheIt->second.suggestionJson;
        }
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(getEngineSuggestionsFunc_);
    if (!func) {
        return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
    }

    PyObject* args = PyTuple_New(2);
    PyObject* engineTypeStr = PyUnicode_FromString(engineType.c_str());
    PyObject* stateStr = PyUnicode_FromString(currentStateJson.c_str());

    PyTuple_SetItem(args, 0, engineTypeStr);
    PyTuple_SetItem(args, 1, stateStr);

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
    } else {
        resultStr = R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
    }

    // Cache the result
    CachedSuggestion entry;
    entry.suggestionJson = resultStr;
    entry.stateHash = hashState(currentStateJson);
    entry.timestamp = std::chrono::steady_clock::now();

    pruneCache();
    suggestionCache_[cacheKey] = entry;

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
#endif
}

void EngineIntelligenceBridge::reportEngineState(
    const std::string& engineType,
    const std::string& stateJson,
    const std::string& generatedNotesJson
) {
    if (!available_) return;

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(reportEngineStateFunc_);
    if (!func) return;

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(engineType.c_str()));
    PyTuple_SetItem(args, 1, PyUnicode_FromString(stateJson.c_str()));
    PyTuple_SetItem(args, 2, PyUnicode_FromString(generatedNotesJson.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - state reporting is not critical
    }
#endif
}

void EngineIntelligenceBridge::clearCache() {
    suggestionCache_.clear();
}

std::string EngineIntelligenceBridge::hashState(const std::string& stateJson) {
    // Simple hash function (can be improved with proper SHA if needed)
    // For now, use a simple hash based on string length and first/last chars
    // In production, use a proper hash function
    std::hash<std::string> hasher;
    return std::to_string(hasher(stateJson));
}

void EngineIntelligenceBridge::pruneCache() {
    if (suggestionCache_.size() > MAX_CACHE_SIZE) {
        // Remove oldest entries (simple: remove first 20%)
        size_t toRemove = suggestionCache_.size() / 5;
        auto it = suggestionCache_.begin();
        for (size_t i = 0; i < toRemove && it != suggestionCache_.end(); ++i) {
            it = suggestionCache_.erase(it);
        }
    }
}

} // namespace kelly
