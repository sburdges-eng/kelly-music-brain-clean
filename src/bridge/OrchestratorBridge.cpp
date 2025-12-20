#include "OrchestratorBridge.h"

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <iostream>
#include <sstream>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <functional>

namespace kelly {

OrchestratorBridge::OrchestratorBridge()
    : available_(false)
    , executePipelineFunc_(nullptr)
    , executePipelineAsyncFunc_(nullptr)
    , getStatusFunc_(nullptr)
    , cancelExecutionFunc_(nullptr)
{
    available_ = initializePython();
}

OrchestratorBridge::~OrchestratorBridge() {
    shutdownPython();
}

bool OrchestratorBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
    // Check if Python is already initialized
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "OrchestratorBridge: Failed to initialize Python" << std::endl;
            return false;
        }
    }

    // Import the orchestrator bridge module
    PyObject* module = PyImport_ImportModule("music_brain.orchestrator.bridge_api");
    if (!module) {
        PyErr_Print();
        std::cerr << "OrchestratorBridge: Failed to import orchestrator bridge_api module" << std::endl;
        return false;
    }

    // Get function pointers
    executePipelineFunc_ = PyObject_GetAttrString(module, "execute_pipeline_sync");
    executePipelineAsyncFunc_ = PyObject_GetAttrString(module, "execute_pipeline_async");
    getStatusFunc_ = PyObject_GetAttrString(module, "get_execution_status");
    cancelExecutionFunc_ = PyObject_GetAttrString(module, "cancel_execution");

    Py_DECREF(module);

    // Note: async functions may not exist yet, that's OK
    if (!executePipelineFunc_) {
        std::cerr << "OrchestratorBridge: Failed to get execute_pipeline_sync function" << std::endl;
        return false;
    }

    return true;
#else
    // Python not available - bridge will use fallback
    std::cerr << "OrchestratorBridge: Python not available, orchestrator disabled" << std::endl;
    return false;
#endif
}

void OrchestratorBridge::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    if (executePipelineFunc_) {
        Py_DECREF(static_cast<PyObject*>(executePipelineFunc_));
        executePipelineFunc_ = nullptr;
    }
    if (executePipelineAsyncFunc_) {
        Py_DECREF(static_cast<PyObject*>(executePipelineAsyncFunc_));
        executePipelineAsyncFunc_ = nullptr;
    }
    if (getStatusFunc_) {
        Py_DECREF(static_cast<PyObject*>(getStatusFunc_));
        getStatusFunc_ = nullptr;
    }
    if (cancelExecutionFunc_) {
        Py_DECREF(static_cast<PyObject*>(cancelExecutionFunc_));
        cancelExecutionFunc_ = nullptr;
    }
#endif
    activeExecutions_.clear();
    engineCallbacks_.clear();
}

std::string OrchestratorBridge::executePipeline(
    const std::string& pipelineName,
    const std::string& inputDataJson
) {
    if (!available_) {
        return R"({"success": false, "error": "Python orchestrator not available"})";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(executePipelineFunc_);
    if (!func) {
        return R"({"success": false, "error": "Pipeline execution function not available"})";
    }

    PyObject* args = PyTuple_New(2);
    PyObject* pipelineNameStr = PyUnicode_FromString(pipelineName.c_str());
    PyObject* inputDataStr = PyUnicode_FromString(inputDataJson.c_str());

    PyTuple_SetItem(args, 0, pipelineNameStr);
    PyTuple_SetItem(args, 1, inputDataStr);

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        return R"({"success": false, "error": "Pipeline execution failed"})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"success": false, "error": "Invalid result format"})";
    } else {
        resultStr = R"({"success": false, "error": "Result is not a string"})";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"success": false, "error": "Python not available"})";
#endif
}

void OrchestratorBridge::executePipelineAsync(
    const std::string& pipelineName,
    const std::string& inputDataJson,
    std::function<void(const std::string&)> callback
) {
    if (!available_ || !callback) {
        return;
    }

    // Execute in background thread
    std::thread([this, pipelineName, inputDataJson, callback]() {
        std::string result = executePipeline(pipelineName, inputDataJson);
        callback(result);
    }).detach();
}

std::string OrchestratorBridge::getExecutionStatus(const std::string& executionId) {
    if (!available_ || !getStatusFunc_) {
        return R"({"status": "unknown", "error": "Status check not available"})";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(getStatusFunc_);
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(executionId.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return R"({"status": "unknown", "error": "Status check failed"})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"status": "unknown"})";
    } else {
        resultStr = R"({"status": "unknown"})";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"status": "unknown"})";
#endif
}

bool OrchestratorBridge::cancelExecution(const std::string& executionId) {
    if (!available_ || !cancelExecutionFunc_) {
        return false;
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(cancelExecutionFunc_);
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(executionId.c_str()));

    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return false;
    }

    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
#else
    return false;
#endif
}

void OrchestratorBridge::registerEngineCallback(
    const std::string& engineType,
    std::function<std::string(const std::string&, const std::string&)> callback
) {
    engineCallbacks_[engineType] = callback;
}

std::string OrchestratorBridge::generateExecutionId() {
    // Simple UUID-like generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8; ++i) {
        ss << dis(gen);
    }
    ss << "-";
    for (int i = 0; i < 4; ++i) {
        ss << dis(gen);
    }
    ss << "-";
    for (int i = 0; i < 4; ++i) {
        ss << dis(gen);
    }
    ss << "-";
    for (int i = 0; i < 4; ++i) {
        ss << dis(gen);
    }
    ss << "-";
    for (int i = 0; i < 12; ++i) {
        ss << dis(gen);
    }

    return ss.str();
}

} // namespace kelly
