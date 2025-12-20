#include "SuggestionBridge.h"

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <iostream>
#include <sstream>

namespace kelly {

SuggestionBridge::SuggestionBridge()
    : available_(false)
    , getSuggestionsFunc_(nullptr)
    , recordShownFunc_(nullptr)
    , recordAcceptedFunc_(nullptr)
    , recordDismissedFunc_(nullptr)
{
    available_ = initializePython();
}

SuggestionBridge::~SuggestionBridge() {
    shutdownPython();
}

bool SuggestionBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
    // Check if Python is already initialized
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "SuggestionBridge: Failed to initialize Python" << std::endl;
            return false;
        }
    }

    // Import the suggestion bridge module
    PyObject* module = PyImport_ImportModule("music_brain.intelligence.suggestion_bridge");
    if (!module) {
        PyErr_Print();
        std::cerr << "SuggestionBridge: Failed to import suggestion_bridge module" << std::endl;
        return false;
    }

    // Get function pointers
    getSuggestionsFunc_ = PyObject_GetAttrString(module, "get_suggestions");
    recordShownFunc_ = PyObject_GetAttrString(module, "record_suggestion_shown");
    recordAcceptedFunc_ = PyObject_GetAttrString(module, "record_suggestion_accepted");
    recordDismissedFunc_ = PyObject_GetAttrString(module, "record_suggestion_dismissed");

    Py_DECREF(module);

    if (!getSuggestionsFunc_ || !recordShownFunc_ || !recordAcceptedFunc_ || !recordDismissedFunc_) {
        std::cerr << "SuggestionBridge: Failed to get function pointers" << std::endl;
        return false;
    }

    return true;
#else
    // Python not available - bridge will use fallback (file-based or no-op)
    std::cerr << "SuggestionBridge: Python not available, suggestions disabled" << std::endl;
    return false;
#endif
}

void SuggestionBridge::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    if (getSuggestionsFunc_) {
        Py_DECREF(static_cast<PyObject*>(getSuggestionsFunc_));
        getSuggestionsFunc_ = nullptr;
    }
    if (recordShownFunc_) {
        Py_DECREF(static_cast<PyObject*>(recordShownFunc_));
        recordShownFunc_ = nullptr;
    }
    if (recordAcceptedFunc_) {
        Py_DECREF(static_cast<PyObject*>(recordAcceptedFunc_));
        recordAcceptedFunc_ = nullptr;
    }
    if (recordDismissedFunc_) {
        Py_DECREF(static_cast<PyObject*>(recordDismissedFunc_));
        recordDismissedFunc_ = nullptr;
    }
#endif
}

std::string SuggestionBridge::getSuggestions(
    const std::string& currentStateJson,
    int maxSuggestions
) {
    if (!available_) {
        return "[]";  // Return empty suggestions if Python not available
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(getSuggestionsFunc_);
    if (!func) {
        return "[]";
    }

    PyObject* args = PyTuple_New(2);
    PyObject* stateStr = PyUnicode_FromString(currentStateJson.c_str());
    PyObject* maxSuggestionsInt = PyLong_FromLong(maxSuggestions);

    PyTuple_SetItem(args, 0, stateStr);
    PyTuple_SetItem(args, 1, maxSuggestionsInt);

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        return "[]";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : "[]";
    } else {
        resultStr = "[]";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return "[]";
#endif
}

void SuggestionBridge::recordSuggestionShown(
    const std::string& suggestionId,
    const std::string& suggestionType,
    const std::string& contextJson
) {
    if (!available_) return;

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(recordShownFunc_);
    if (!func) return;

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(suggestionId.c_str()));
    PyTuple_SetItem(args, 1, PyUnicode_FromString(suggestionType.c_str()));
    PyTuple_SetItem(args, 2, PyUnicode_FromString(contextJson.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - tracking is not critical
    }
#endif
}

void SuggestionBridge::recordSuggestionAccepted(const std::string& suggestionId) {
    if (!available_) return;

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(recordAcceptedFunc_);
    if (!func) return;

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(suggestionId.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - tracking is not critical
    }
#endif
}

void SuggestionBridge::recordSuggestionDismissed(const std::string& suggestionId) {
    if (!available_) return;

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(recordDismissedFunc_);
    if (!func) return;

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(suggestionId.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - tracking is not critical
    }
#endif
}

} // namespace kelly
