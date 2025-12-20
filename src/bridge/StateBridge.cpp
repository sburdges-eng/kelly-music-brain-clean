#include "StateBridge.h"

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>
#include <atomic>
#include <memory>

namespace kelly {

// Worker thread for processing state updates
class StateBridge::StateWorkerThread {
public:
    StateWorkerThread(StateBridge* bridge)
        : bridge_(bridge)
        , running_(true)
        , thread_(&StateWorkerThread::run, this)
    {}

    ~StateWorkerThread() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    StateBridge* bridge_;
    std::atomic<bool> running_;
    std::thread thread_;

    void run() {
        while (running_ && !bridge_->shutdownRequested_.load()) {
            bridge_->processStateQueue();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

StateBridge::StateBridge()
    : available_(false)
    , shutdownRequested_(false)
    , emitStateFunc_(nullptr)
    , getCurrentStateFunc_(nullptr)
    , getEngineStateFunc_(nullptr)
    , workerThread_(nullptr)
{
}

StateBridge::~StateBridge() {
    shutdown();
}

bool StateBridge::initialize() {
    available_ = initializePython();
    if (available_) {
        workerThread_ = std::make_unique<StateWorkerThread>(this);
    }
    return available_;
}

void StateBridge::shutdown() {
    shutdownRequested_ = true;

    if (workerThread_) {
        workerThread_.reset();
    }

    // Flush remaining updates
    flush();

    shutdownPython();
    available_ = false;
}

bool StateBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
    // Check if Python is already initialized
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "StateBridge: Failed to initialize Python" << std::endl;
            return false;
        }
    }

    // Import the state bridge module
    PyObject* module = PyImport_ImportModule("music_brain.intelligence.state_bridge");
    if (!module) {
        PyErr_Print();
        std::cerr << "StateBridge: Failed to import state_bridge module" << std::endl;
        return false;
    }

    // Get function pointers
    emitStateFunc_ = PyObject_GetAttrString(module, "emit_state_update");
    getCurrentStateFunc_ = PyObject_GetAttrString(module, "get_current_state");
    getEngineStateFunc_ = PyObject_GetAttrString(module, "get_engine_state");

    Py_DECREF(module);

    // emitStateFunc is the most important one
    if (!emitStateFunc_) {
        std::cerr << "StateBridge: Failed to get emit_state_update function" << std::endl;
        return false;
    }

    return true;
#else
    // Python not available - bridge will use fallback
    std::cerr << "StateBridge: Python not available, state synchronization disabled" << std::endl;
    return false;
#endif
}

void StateBridge::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    if (emitStateFunc_) {
        Py_DECREF(static_cast<PyObject*>(emitStateFunc_));
        emitStateFunc_ = nullptr;
    }
    if (getCurrentStateFunc_) {
        Py_DECREF(static_cast<PyObject*>(getCurrentStateFunc_));
        getCurrentStateFunc_ = nullptr;
    }
    if (getEngineStateFunc_) {
        Py_DECREF(static_cast<PyObject*>(getEngineStateFunc_));
        getEngineStateFunc_ = nullptr;
    }
#endif
}

void StateBridge::emitStateUpdate(
    const std::string& engineType,
    const std::string& stateJson
) {
    if (!available_.load() || shutdownRequested_.load()) {
        return;
    }

    // Add to queue (lock-free for audio thread safety)
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (stateQueue_.size() >= MAX_QUEUE_SIZE) {
            // Queue full - drop oldest entry
            stateQueue_.pop();
        }

        StateUpdate update;
        update.engineType = engineType;
        update.stateJson = stateJson;
        update.timestamp = std::chrono::steady_clock::now();
        stateQueue_.push(update);
    }
}

std::string StateBridge::getCurrentState() {
    if (!available_.load() || !getCurrentStateFunc_) {
        return "{}";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(getCurrentStateFunc_);
    PyObject* result = PyObject_CallObject(func, nullptr);

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

std::string StateBridge::getEngineState(const std::string& engineType) {
    if (!available_.load() || !getEngineStateFunc_) {
        return "{}";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* func = static_cast<PyObject*>(getEngineStateFunc_);
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(engineType.c_str()));

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

void StateBridge::flush() {
    processStateQueue();
}

void StateBridge::processStateQueue() {
    if (!available_.load() || !emitStateFunc_) {
        return;
    }

    // Process up to 10 updates per call (batch processing)
    int processed = 0;
    const int MAX_BATCH = 10;

    while (processed < MAX_BATCH) {
        StateUpdate update;
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (stateQueue_.empty()) {
                break;
            }
            update = stateQueue_.front();
            stateQueue_.pop();
        }

#ifdef PYTHON_AVAILABLE
        PyObject* func = static_cast<PyObject*>(emitStateFunc_);
        PyObject* args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, PyUnicode_FromString(update.engineType.c_str()));
        PyTuple_SetItem(args, 1, PyUnicode_FromString(update.stateJson.c_str()));

        PyObject* result = PyObject_CallObject(func, args);
        Py_DECREF(args);

        if (result) {
            Py_DECREF(result);
        } else {
            PyErr_Clear();  // Clear error - state updates are not critical
        }
#endif

        processed++;
    }
}

} // namespace kelly
