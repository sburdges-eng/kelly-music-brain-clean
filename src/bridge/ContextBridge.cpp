#include "ContextBridge.h"

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <iostream>
#include <sstream>
#include <chrono>
#include <functional>
#include <map>

namespace kelly {

ContextBridge::ContextBridge()
    : available_(false)
    , analyzeContextFunc_(nullptr)
    , getContextualParametersFunc_(nullptr)
    , updateContextFunc_(nullptr)
    , getSuggestionsFunc_(nullptr)
{
    available_ = initializePython();
}

ContextBridge::~ContextBridge() {
    shutdownPython();
}

bool ContextBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
    // Check if Python is already initialized
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "ContextBridge: Failed to initialize Python" << std::endl;
            return false;
        }
    }

    // Import the context bridge module
    PyObject* module = PyImport_ImportModule("music_brain.intelligence.context_bridge");
    if (!module) {
        PyErr_Print();
        std::cerr << "ContextBridge: Failed to import context_bridge module" << std::endl;
        return false;
    }

    // Get function pointers
    analyzeContextFunc_ = PyObject_GetAttrString(module, "analyze_context");
    getContextualParametersFunc_ = PyObject_GetAttrString(module, "get_contextual_parameters");
    updateContextFunc_ = PyObject_GetAttrString(module, "update_context");
    getSuggestionsFunc_ = PyObject_GetAttrString(module, "get_contextual_suggestions");

    Py_DECREF(module);

    if (!analyzeContextFunc_) {
        std::cerr << "ContextBridge: Failed to get analyze_context function" << std::endl;
        return false;
    }

    return true;
#else
    // Python not available - bridge will use fallback
    std::cerr << "ContextBridge: Python not available, context analysis disabled" << std::endl;
    return false;
#endif
}

void ContextBridge::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    if (analyzeContextFunc_) {
        Py_DECREF(static_cast<PyObject*>(analyzeContextFunc_));
        analyzeContextFunc_ = nullptr;
    }
    if (getContextualParametersFunc_) {
        Py_DECREF(static_cast<PyObject*>(getContextualParametersFunc_));
        getContextualParametersFunc_ = nullptr;
    }
    if (updateContextFunc_) {
        Py_DECREF(static_cast<PyObject*>(updateContextFunc_));
        updateContextFunc_ = nullptr;
    }
    if (getSuggestionsFunc_) {
        Py_DECREF(static_cast<PyObject*>(getSuggestionsFunc_));
        getSuggestionsFunc_ = nullptr;
    }
#endif
    contextCache_.clear();
}

std::string ContextBridge::analyzeContext(const std::string& stateJson) {
    if (!available_) {
        return R"({"emotion_category": "unknown", "complexity_level": "moderate"})";
    }

    // Check cache first
    std::string cacheKey = hashState(stateJson);
    std::string cached = getCachedContext(cacheKey);
    if (!cached.empty()) {
        return cached;
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(analyzeContextFunc_);
    if (!func) {
        return R"({"emotion_category": "unknown"})";
    }

    PyObject* args = PyTuple_New(1);
    PyObject* stateStr = PyUnicode_FromString(stateJson.c_str());

    PyTuple_SetItem(args, 0, stateStr);

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        return R"({"emotion_category": "unknown"})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"emotion_category": "unknown"})";
    } else {
        resultStr = R"({"emotion_category": "unknown"})";
    }

    // Cache the result
    cacheContext(cacheKey, resultStr);

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"emotion_category": "unknown"})";
#endif
}

std::string ContextBridge::getContextualParameters(const std::string& stateJson) {
    if (!available_ || !getContextualParametersFunc_) {
        return "{}";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(getContextualParametersFunc_);
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(stateJson.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return "{}";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : "{}";
    } else {
        resultStr = "{}";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return "{}";
#endif
}

void ContextBridge::updateContext(const std::string& stateJson) {
    if (!available_ || !updateContextFunc_) {
        return;
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(updateContextFunc_);
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(stateJson.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - context update is not critical
    }

    // Clear cache to force fresh analysis
    clearCache();
#endif
}

std::string ContextBridge::getContextualSuggestions(const std::string& stateJson) {
    if (!available_ || !getSuggestionsFunc_) {
        return R"({"suggestions": []})";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(getSuggestionsFunc_);
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(stateJson.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return R"({"suggestions": []})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"suggestions": []})";
    } else {
        resultStr = R"({"suggestions": []})";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"suggestions": []})";
#endif
}

void ContextBridge::clearCache() {
    contextCache_.clear();
}

std::string ContextBridge::getCachedContext(const std::string& cacheKey) {
    auto it = contextCache_.find(cacheKey);
    if (it == contextCache_.end()) {
        return "";
    }

    // Check if cache entry is still valid
    auto now = std::chrono::steady_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - it->second.timestamp
    ).count();

    if (age > CACHE_TTL_MS) {
        contextCache_.erase(it);
        return "";
    }

    return it->second.contextJson;
}

void ContextBridge::cacheContext(
    const std::string& cacheKey,
    const std::string& contextJson
) {
    CachedContext cached;
    cached.contextJson = contextJson;
    cached.stateHash = cacheKey;
    cached.timestamp = std::chrono::steady_clock::now();

    contextCache_[cacheKey] = cached;

    // Limit cache size (keep most recent 50 entries)
    if (contextCache_.size() > 50) {
        // Remove oldest entry
        auto oldest = contextCache_.begin();
        for (auto it = contextCache_.begin(); it != contextCache_.end(); ++it) {
            if (it->second.timestamp < oldest->second.timestamp) {
                oldest = it;
            }
        }
        contextCache_.erase(oldest);
    }
}

std::string ContextBridge::hashState(const std::string& stateJson) {
    // Simple hash function for cache key
    std::hash<std::string> hasher;
    return std::to_string(hasher(stateJson));
}

} // namespace kelly
