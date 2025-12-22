#pragma once

/**
 * OSCBridge - OSC Client for C++ Plugin ↔ Python Brain Communication
 * ====================================================================
 *
 * Implements OSC client functionality to communicate with the Python brain server.
 * This enables the hybrid architecture:
 * - C++ Body (plugin): Sends requests via OSC
 * - Python Brain (server): Processes requests and responds
 * - OSC Bridge: Communication layer
 */

#include <juce_core/juce_core.h>
#include <juce_osc/juce_osc.h>
#include <memory>
#include <functional>
#include <string>
#include <vector>
#include <map>

namespace kelly {
namespace bridge {

/**
 * OSC Message structure for communication
 */
struct OSCMessage {
    std::string address;
    std::vector<juce::var> arguments;

    OSCMessage(const std::string& addr) : address(addr) {}

    template<typename T>
    OSCMessage& add(T value) {
        arguments.push_back(juce::var(value));
        return *this;
    }
};

/**
 * OSC Response handler callback
 */
using OSCResponseHandler = std::function<void(const juce::var& response)>;

/**
 * OSCBridge - Client for communicating with Python brain server
 */
class OSCBridge : public juce::OSCReceiver::Listener<juce::OSCReceiver::MessageLoopCallback> {
public:
    OSCBridge();
    ~OSCBridge();

    /**
     * Initialize OSC bridge
     * @param brainHost Python brain server host (default: 127.0.0.1)
     * @param brainPort Python brain server port (default: 5005)
     * @param listenPort Local port for receiving responses (default: 5006)
     * @return true if initialized successfully
     */
    bool initialize(
        const std::string& brainHost = "127.0.0.1",
        int brainPort = 5005,
        int listenPort = 5006
    );

    /**
     * Check if bridge is connected
     */
    bool isConnected() const { return connected_; }

    /**
     * Send generation request to brain
     * @param text User input text
     * @param motivation Motivation description
     * @param chaos Chaos level (0.0-1.0)
     * @param vulnerability Vulnerability level (0.0-1.0)
     * @param callback Response handler
     */
    void requestGenerate(
        const std::string& text,
        const std::string& motivation,
        float chaos,
        float vulnerability,
        OSCResponseHandler callback
    );

    /**
     * Request chord analysis
     * @param progression Chord progression string
     * @param callback Response handler
     */
    void requestAnalyzeChords(
        const std::string& progression,
        OSCResponseHandler callback
    );

    /**
     * Request intent processing
     * @param intentFile Path to intent file
     * @param callback Response handler
     */
    void requestIntentProcess(
        const std::string& intentFile,
        OSCResponseHandler callback
    );

    /**
     * Request intent suggestions
     * @param emotion Emotion name
     * @param callback Response handler
     */
    void requestIntentSuggest(
        const std::string& emotion,
        OSCResponseHandler callback
    );

    /**
     * Send ping to check brain server availability
     * @param callback Response handler
     */
    void ping(OSCResponseHandler callback);

    /**
     * Shutdown OSC bridge
     */
    void shutdown();

private:
    bool connected_ = false;
    std::string brainHost_;
    int brainPort_;
    int listenPort_;

    // Response handlers (message ID -> handler)
    std::map<int, OSCResponseHandler> responseHandlers_;
    int nextMessageId_ = 0;

    // OSC sender and receiver (would use JUCE OSC classes)
    std::unique_ptr<juce::OSCSender> sender_;
    std::unique_ptr<juce::OSCReceiver> receiver_;

    /**
     * Send OSC message
     */
    bool sendMessage(const OSCMessage& message);

    /**
     * Handle incoming OSC message
     */
    void handleOSCMessage(const juce::OSCMessage& message);

    /**
     * Generate unique message ID
     */
    int generateMessageId() { return ++nextMessageId_; }

    /**
     * OSCReceiver::ListenerWithOSCAddress callback
     */
    void oscMessageReceived(const juce::OSCMessage& message) override;
};

} // namespace bridge
} // namespace kelly
