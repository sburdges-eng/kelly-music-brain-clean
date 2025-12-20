#include "PreferenceBridge.h"
#include <juce_core/juce_core.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

namespace kelly {

// Worker thread for processing Python calls
class PreferenceBridge::WorkerThread : public juce::Thread {
public:
    explicit WorkerThread(PreferenceBridge& bridge)
        : juce::Thread("PreferenceBridgeWorker"), bridge_(bridge) {}

    void run() override {
        while (!threadShouldExit()) {
            bridge_.processPendingOperations();
            bridge_.updateCachedStatistics();
            wait(500);  // Check every 500ms
        }
    }

private:
    PreferenceBridge& bridge_;
};

PreferenceBridge::PreferenceBridge() {
    workerThread_ = std::make_unique<WorkerThread>(*this);
}

PreferenceBridge::~PreferenceBridge() {
    shutdown();
}

bool PreferenceBridge::initialize() {
    if (initialized_.load()) {
        return true;
    }

    // Initialize Python if available
    bool pythonOk = initializePython();

    if (pythonOk) {
        initialized_.store(true);
        workerThread_->startThread();
        return true;
    } else {
        // Fallback: use file-based queue (works without Python embedding)
        // This allows the system to work even if Python isn't embedded
        initialized_.store(true);
        workerThread_->startThread();
        return true;
    }
}

void PreferenceBridge::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    shutdownRequested_.store(true);

    if (workerThread_ && workerThread_->isThreadRunning()) {
        workerThread_->stopThread(1000);
    }

    flush();  // Flush any pending operations
    shutdownPython();
    initialized_.store(false);
}

bool PreferenceBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
    if (Py_IsInitialized()) {
        // Python already initialized
    } else {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            return false;
        }
    }

    // Import music_brain.learning.user_preferences
    PyObject* moduleName = PyUnicode_FromString("music_brain.learning.user_preferences");
    if (!moduleName) return false;

    PyObject* module = PyImport_Import(moduleName);
    Py_DECREF(moduleName);
    if (!module) {
        PyErr_Print();
        return false;
    }

    pythonModule_ = module;

    // Get UserPreferenceModel class
    PyObject* modelClass = PyObject_GetAttrString(module, "UserPreferenceModel");
    if (!modelClass) {
        Py_DECREF(module);
        pythonModule_ = nullptr;
        PyErr_Print();
        return false;
    }

    // Create instance: UserPreferenceModel()
    PyObject* args = PyTuple_New(0);
    PyObject* instance = PyObject_CallObject(modelClass, args);
    Py_DECREF(args);
    Py_DECREF(modelClass);

    if (!instance) {
        Py_DECREF(module);
        pythonModule_ = nullptr;
        PyErr_Print();
        return false;
    }

    preferenceModel_ = instance;
    return true;
#else
    // Python not available - use file-based fallback
    return false;
#endif
}

void PreferenceBridge::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    if (preferenceModel_) {
        Py_DECREF(static_cast<PyObject*>(preferenceModel_));
        preferenceModel_ = nullptr;
    }
    if (pythonModule_) {
        Py_DECREF(static_cast<PyObject*>(pythonModule_));
        pythonModule_ = nullptr;
    }
#endif
}

void PreferenceBridge::recordParameterAdjustment(
    const juce::String& parameterName,
    float oldValue,
    float newValue,
    const std::map<std::string, std::string>& context
) {
    if (!initialized_.load()) return;

    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
        // Drop oldest operation
        pendingOperations_.erase(pendingOperations_.begin());
    }

    PendingOperation op;
    op.type = PendingOperation::ParameterAdjustment;
    op.data["parameter_name"] = parameterName.toStdString();
    op.data["old_value"] = juce::String(oldValue).toStdString();
    op.data["new_value"] = juce::String(newValue).toStdString();
    for (const auto& [key, value] : context) {
        op.data["context_" + key] = value;
    }

    pendingOperations_.push_back(op);
}

void PreferenceBridge::recordEmotionSelection(
    const juce::String& emotionName,
    float valence,
    float arousal,
    const std::map<std::string, std::string>& context
) {
    if (!initialized_.load()) return;

    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
        pendingOperations_.erase(pendingOperations_.begin());
    }

    PendingOperation op;
    op.type = PendingOperation::EmotionSelection;
    op.data["emotion_name"] = emotionName.toStdString();
    op.data["valence"] = juce::String(valence).toStdString();
    op.data["arousal"] = juce::String(arousal).toStdString();
    for (const auto& [key, value] : context) {
        op.data["context_" + key] = value;
    }

    pendingOperations_.push_back(op);
}

juce::String PreferenceBridge::recordMidiGeneration(
    const juce::String& intentText,
    const std::map<juce::String, float>& parameters,
    const juce::String& emotion,
    const std::vector<juce::String>& ruleBreaks
) {
    if (!initialized_.load()) return {};

    juce::String generationId = generateId();

    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
        pendingOperations_.erase(pendingOperations_.begin());
    }

    PendingOperation op;
    op.type = PendingOperation::MidiGeneration;
    op.data["generation_id"] = generationId.toStdString();
    op.data["intent_text"] = intentText.toStdString();
    op.data["emotion"] = emotion.toStdString();

    // Serialize parameters
    std::ostringstream paramsStream;
    bool first = true;
    for (const auto& [key, value] : parameters) {
        if (!first) paramsStream << ",";
        paramsStream << key.toStdString() << "=" << value;
        first = false;
    }
    op.data["parameters"] = paramsStream.str();

    // Serialize rule breaks
    std::ostringstream rbStream;
    first = true;
    for (const auto& rb : ruleBreaks) {
        if (!first) rbStream << ",";
        rbStream << rb.toStdString();
        first = false;
    }
    op.data["rule_breaks"] = rbStream.str();

    pendingOperations_.push_back(op);
    return generationId;
}

void PreferenceBridge::recordMidiFeedback(const juce::String& generationId, bool accepted) {
    if (!initialized_.load()) return;

    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
        pendingOperations_.erase(pendingOperations_.begin());
    }

    PendingOperation op;
    op.type = PendingOperation::MidiFeedback;
    op.data["generation_id"] = generationId.toStdString();
    op.data["accepted"] = accepted ? "true" : "false";

    pendingOperations_.push_back(op);
}

void PreferenceBridge::recordMidiModification(
    const juce::String& generationId,
    const juce::String& parameterName,
    float oldValue,
    float newValue
) {
    if (!initialized_.load()) return;

    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
        pendingOperations_.erase(pendingOperations_.begin());
    }

    PendingOperation op;
    op.type = PendingOperation::MidiModification;
    op.data["generation_id"] = generationId.toStdString();
    op.data["parameter_name"] = parameterName.toStdString();
    op.data["old_value"] = juce::String(oldValue).toStdString();
    op.data["new_value"] = juce::String(newValue).toStdString();

    pendingOperations_.push_back(op);
}

void PreferenceBridge::recordRuleBreakModification(
    const juce::String& ruleBreak,
    const juce::String& action,
    const std::map<std::string, std::string>& context
) {
    if (!initialized_.load()) return;

    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
        pendingOperations_.erase(pendingOperations_.begin());
    }

    PendingOperation op;
    op.type = PendingOperation::RuleBreakModification;
    op.data["rule_break"] = ruleBreak.toStdString();
    op.data["action"] = action.toStdString();
    for (const auto& [key, value] : context) {
        op.data["context_" + key] = value;
    }

    pendingOperations_.push_back(op);
}

void PreferenceBridge::processPendingOperations() {
    std::vector<PendingOperation> ops;
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        if (pendingOperations_.empty()) {
            return;
        }
        ops.swap(pendingOperations_);
    }

#ifdef PYTHON_AVAILABLE
    if (preferenceModel_) {
        // Process with Python
        for (const auto& op : ops) {
            switch (op.type) {
                case PendingOperation::ParameterAdjustment: {
                    float oldVal = std::stof(op.data.at("old_value"));
                    float newVal = std::stof(op.data.at("new_value"));
                    // Call: preferenceModel.record_parameter_adjustment(...)
                    // Implementation would use PyObject_CallMethod
                    break;
                }
                // ... other cases
                default:
                    break;
            }
        }
    } else
#endif
    {
        // File-based fallback: write to queue file
        juce::File queueFile = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
            .getChildFile("Kelly")
            .getChildFile("preference_queue.json");

        queueFile.getParentDirectory().createDirectory();

        // Read existing queue
        juce::String existingContent;
        if (queueFile.existsAsFile()) {
            existingContent = queueFile.loadFileAsString();
        }

        // Append new operations (simplified JSON format)
        std::ostringstream jsonStream;
        jsonStream << "[\n";
        bool first = true;
        for (const auto& op : ops) {
            if (!first) jsonStream << ",\n";
            jsonStream << "  {\n";
            jsonStream << "    \"type\": " << static_cast<int>(op.type) << ",\n";
            jsonStream << "    \"data\": {\n";
            bool firstData = true;
            for (const auto& [key, value] : op.data) {
                if (!firstData) jsonStream << ",\n";
                jsonStream << "      \"" << key << "\": \"" << value << "\"";
                firstData = false;
            }
            jsonStream << "\n    }\n";
            jsonStream << "  }";
            first = false;
        }
        jsonStream << "\n]";

        // Write to file (append mode for simplicity)
        std::ofstream file(queueFile.getFullPathName().toStdString(), std::ios::app);
        if (file.is_open()) {
            file << jsonStream.str();
            file.close();
        }
    }
}

void PreferenceBridge::updateCachedStatistics() {
    auto now = juce::Time::currentTimeMillis();
    if (now - lastCacheUpdate_ < CACHE_UPDATE_INTERVAL_MS) {
        return;
    }

    // Update cache from Python (or file-based system)
    // For now, leave empty - will be populated when Python integration is complete
    lastCacheUpdate_ = now;
}

std::map<std::string, PreferenceBridge::ParameterStatistics> PreferenceBridge::getParameterStatistics() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return cachedStats_;
}

std::map<std::string, int> PreferenceBridge::getEmotionPreferences() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return cachedEmotionPrefs_;
}

std::map<std::string, std::pair<float, float>> PreferenceBridge::getPreferredParameterRanges() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return cachedRanges_;
}

float PreferenceBridge::getAcceptanceRate() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return cachedAcceptanceRate_;
}

void PreferenceBridge::flush() {
    processPendingOperations();
}

juce::String PreferenceBridge::generateId() {
    auto time = juce::Time::currentTimeMillis();
    auto random = juce::Random::getSystemRandom().nextInt64();
    return juce::String(time) + "_" + juce::String(random);
}

bool PreferenceBridge::callPythonMethod(
    const char* methodName,
    const std::vector<std::string>& args,
    const std::map<std::string, std::string>& kwargs
) {
#ifdef PYTHON_AVAILABLE
    if (!preferenceModel_) return false;

    PyObject* method = PyObject_GetAttrString(static_cast<PyObject*>(preferenceModel_), methodName);
    if (!method) {
        PyErr_Print();
        return false;
    }

    // Build arguments
    PyObject* pyArgs = PyTuple_New(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        PyObject* arg = PyUnicode_FromString(args[i].c_str());
        PyTuple_SetItem(pyArgs, i, arg);
    }

    PyObject* result = PyObject_CallObject(method, pyArgs);
    Py_DECREF(pyArgs);
    Py_DECREF(method);

    if (!result) {
        PyErr_Print();
        return false;
    }

    Py_DECREF(result);
    return true;
#else
    return false;
#endif
}

bool PreferenceBridge::callPythonMethodWithResult(
    const char* methodName,
    void* resultOut,
    const std::vector<std::string>& args,
    const std::map<std::string, std::string>& kwargs
) {
    // Implementation would extract result from PyObject
    // For now, return false
    return false;
}

} // namespace kelly
