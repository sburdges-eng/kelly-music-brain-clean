#pragma once

#include <juce_osc/juce_osc.h>
#include <memory>
#include <string>
#include <functional>
#include <optional>

namespace kelly {

/**
 * OSC Client for communicating with Python Brain Server
 *
 * Sends requests to Python brain_server.py and receives responses.
 * Used for the Python Brain ↔ C++ Body architecture.
 */
class OSCClient : public juce::OSCReceiver::Listener<juce::OSCReceiver::RealtimeCallback> {
public:
    OSCClient();
    ~OSCClient();

    /**
     * Connect to Python brain server
     * @param host Server host (default: "127.0.0.1")
     * @param serverPort Server port (default: 5005)
     * @param responsePort Local port to receive responses (default: 5006)
     */
    bool connect(const std::string& host = "127.0.0.1",
                 int serverPort = 5005,
                 int responsePort = 5006);

    /**
     * Disconnect from server
     */
    void disconnect();

    /**
     * Check if connected
     */
    bool isConnected() const { return connected_; }

    /**
     * Request music generation from Python brain
     * @param text User text input
     * @param motivation Motivation parameter (0.0-1.0)
     * @param chaos Chaos parameter (0.0-1.0)
     * @param vulnerability Vulnerability parameter (0.0-1.0)
     * @param callback Callback with response JSON string
     */
    void requestGenerate(
        const std::string& text,
        float motivation = 0.5f,
        float chaos = 0.5f,
        float vulnerability = 0.5f,
        std::function<void(const std::string&)> callback = nullptr
    );

    /**
     * Request chord progression analysis
     * @param progression Chord progression string (e.g., "C F G Am")
     * @param callback Callback with analysis JSON string
     */
    void requestAnalyzeChords(
        const std::string& progression,
        std::function<void(const std::string&)> callback = nullptr
    );

    /**
     * Request intent processing
     * @param intentFile Path to intent file
     * @param callback Callback with response JSON string
     */
    void requestIntentProcess(
        const std::string& intentFile,
        std::function<void(const std::string&)> callback = nullptr
    );

    /**
     * Request intent suggestions
     * @param emotion Emotion name
     * @param callback Callback with suggestions JSON string
     */
    void requestIntentSuggest(
        const std::string& emotion,
        std::function<void(const std::string&)> callback = nullptr
    );

    /**
     * Ping server (health check)
     * @param callback Callback with ping response JSON string
     */
    void ping(std::function<void(const std::string&)> callback = nullptr);

    /**
     * Process incoming OSC messages (call from message thread or timer)
     */
    void processMessages();

private:
    std::unique_ptr<juce::OSCSender> sender_;
    std::unique_ptr<juce::OSCReceiver> receiver_;

    bool connected_ = false;
    std::string serverHost_;
    int serverPort_ = 5005;
    int responsePort_ = 5006;

    // Response callbacks (keyed by request ID)
    struct PendingRequest {
        std::function<void(const std::string&)> callback;
        juce::Time timestamp;
    };

    std::map<int, PendingRequest> pendingRequests_;
    int nextRequestId_ = 1;
    static constexpr int REQUEST_TIMEOUT_MS = 5000;

    // OSC message handlers (from juce::OSCReceiver::Listener)
    void oscMessageReceived(const juce::OSCMessage& message) override;
    void oscBundleReceived(const juce::OSCBundle& bundle) override;

    void setupReceiver();
    void handleGenerateResponse(const juce::OSCMessage& message);
    void handleAnalyzeChordsResponse(const juce::OSCMessage& message);
    void handleIntentProcessResponse(const juce::OSCMessage& message);
    void handleIntentSuggestResponse(const juce::OSCMessage& message);
    void handlePingResponse(const juce::OSCMessage& message);
    void handleUnknownMessage(const juce::OSCMessage& message);

    // Clean up timed-out requests
    void cleanupTimedOutRequests();
};

} // namespace kelly
