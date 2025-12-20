#include "bridge/OSCClient.h"
#include <juce_osc/juce_osc.h>
#include <iostream>
#include <map>

namespace kelly {

OSCClient::OSCClient() {
    sender_ = std::make_unique<juce::OSCSender>();
    receiver_ = std::make_unique<juce::OSCReceiver>();
}

OSCClient::~OSCClient() {
    disconnect();
}

bool OSCClient::connect(const std::string& host, int serverPort, int responsePort) {
    if (connected_) {
        disconnect();
    }

    serverHost_ = host;
    serverPort_ = serverPort;
    responsePort_ = responsePort;

    // Connect sender to Python brain server
    if (!sender_->connect(host, serverPort)) {
        std::cerr << "OSCClient: Failed to connect sender to " << host << ":" << serverPort << std::endl;
        return false;
    }

    // Connect receiver for responses
    if (!receiver_->connect(responsePort)) {
        std::cerr << "OSCClient: Failed to bind receiver to port " << responsePort << std::endl;
        sender_->disconnect();
        return false;
    }

    setupReceiver();
    connected_ = true;

    std::cout << "OSCClient: Connected to " << host << ":" << serverPort
              << ", listening on port " << responsePort << std::endl;
    return true;
}

void OSCClient::disconnect() {
    if (sender_) {
        sender_->disconnect();
    }
    if (receiver_) {
        receiver_->disconnect();
    }
    connected_ = false;
    pendingRequests_.clear();
}

void OSCClient::setupReceiver() {
    receiver_->addListener(this);
}

void OSCClient::processMessages() {
    // Clean up timed-out requests
    cleanupTimedOutRequests();

    // OSC messages are handled via callbacks (addListener)
}

void OSCClient::requestGenerate(
    const std::string& text,
    float motivation,
    float chaos,
    float vulnerability,
    std::function<void(const std::string&)> callback)
{
    if (!connected_) {
        std::cerr << "OSCClient: Not connected, cannot send request" << std::endl;
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    // Create JSON request
    juce::String jsonRequest = juce::String::formatted(
        R"({"text":"%s","motivation":%.2f,"chaos":%.2f,"vulnerability":%.2f,"response_port":%d})",
        text.c_str(), motivation, chaos, vulnerability, responsePort_
    );

    // Store callback
    int requestId = nextRequestId_++;
    if (callback) {
        pendingRequests_[requestId] = {
            callback,
            juce::Time::getCurrentTime()
        };
    }

    // Send OSC message
    juce::OSCMessage message("/daiw/generate");
    message.addString(jsonRequest.toStdString());

    if (!sender_->send(message)) {
        std::cerr << "OSCClient: Failed to send generate request" << std::endl;
        if (callback) {
            pendingRequests_.erase(requestId);
            callback(R"({"status":"error","message":"Send failed"})");
        }
    }
}

void OSCClient::requestAnalyzeChords(
    const std::string& progression,
    std::function<void(const std::string&)> callback)
{
    if (!connected_) {
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    int requestId = nextRequestId_++;
    if (callback) {
        pendingRequests_[requestId] = {
            callback,
            juce::Time::getCurrentTime()
        };
    }

    juce::OSCMessage message("/daiw/analyze/chords");
    message.addString(progression);

    if (!sender_->send(message)) {
        if (callback) {
            pendingRequests_.erase(requestId);
            callback(R"({"status":"error","message":"Send failed"})");
        }
    }
}

void OSCClient::requestIntentProcess(
    const std::string& intentFile,
    std::function<void(const std::string&)> callback)
{
    if (!connected_) {
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    int requestId = nextRequestId_++;
    if (callback) {
        pendingRequests_[requestId] = {
            callback,
            juce::Time::getCurrentTime()
        };
    }

    juce::OSCMessage message("/daiw/intent/process");
    message.addString(intentFile);

    if (!sender_->send(message)) {
        if (callback) {
            pendingRequests_.erase(requestId);
            callback(R"({"status":"error","message":"Send failed"})");
        }
    }
}

void OSCClient::requestIntentSuggest(
    const std::string& emotion,
    std::function<void(const std::string&)> callback)
{
    if (!connected_) {
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    int requestId = nextRequestId_++;
    if (callback) {
        pendingRequests_[requestId] = {
            callback,
            juce::Time::getCurrentTime()
        };
    }

    juce::OSCMessage message("/daiw/intent/suggest");
    message.addString(emotion);

    if (!sender_->send(message)) {
        if (callback) {
            pendingRequests_.erase(requestId);
            callback(R"({"status":"error","message":"Send failed"})");
        }
    }
}

void OSCClient::ping(std::function<void(const std::string&)> callback) {
    if (!connected_) {
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    int requestId = nextRequestId_++;
    if (callback) {
        pendingRequests_[requestId] = {
            callback,
            juce::Time::getCurrentTime()
        };
    }

    juce::OSCMessage message("/daiw/ping");

    if (!sender_->send(message)) {
        if (callback) {
            pendingRequests_.erase(requestId);
            callback(R"({"status":"error","message":"Send failed"})");
        }
    }
}

void OSCClient::oscMessageReceived(const juce::OSCMessage& message) {
    juce::String address = message.getAddressPattern().toString();

    if (address == "/daiw/generate/response") {
        handleGenerateResponse(message);
    } else if (address == "/daiw/analyze/chords/response") {
        handleAnalyzeChordsResponse(message);
    } else if (address == "/daiw/intent/process/response") {
        handleIntentProcessResponse(message);
    } else if (address == "/daiw/intent/suggest/response") {
        handleIntentSuggestResponse(message);
    } else if (address == "/daiw/ping/response") {
        handlePingResponse(message);
    } else {
        handleUnknownMessage(message);
    }
}

void OSCClient::oscBundleReceived(const juce::OSCBundle& bundle) {
    // Handle bundles if needed
    for (const auto& element : bundle) {
        if (element.isMessage()) {
            oscMessageReceived(element.getMessage());
        }
    }
}

void OSCClient::handleGenerateResponse(const juce::OSCMessage& message) {
    if (message.size() >= 1 && message[0].isString()) {
        std::string responseJson = message[0].getString().toStdString();
        // Find and call callback (simplified - in real implementation, match request IDs)
        for (auto& [id, request] : pendingRequests_) {
            if (request.callback) {
                request.callback(responseJson);
                pendingRequests_.erase(id);
                break;
            }
        }
    }
}

void OSCClient::handleAnalyzeChordsResponse(const juce::OSCMessage& message) {
    if (message.size() >= 1 && message[0].isString()) {
        std::string responseJson = message[0].getString().toStdString();
        for (auto& [id, request] : pendingRequests_) {
            if (request.callback) {
                request.callback(responseJson);
                pendingRequests_.erase(id);
                break;
            }
        }
    }
}

void OSCClient::handleIntentProcessResponse(const juce::OSCMessage& message) {
    if (message.size() >= 1 && message[0].isString()) {
        std::string responseJson = message[0].getString().toStdString();
        for (auto& [id, request] : pendingRequests_) {
            if (request.callback) {
                request.callback(responseJson);
                pendingRequests_.erase(id);
                break;
            }
        }
    }
}

void OSCClient::handleIntentSuggestResponse(const juce::OSCMessage& message) {
    if (message.size() >= 1 && message[0].isString()) {
        std::string responseJson = message[0].getString().toStdString();
        for (auto& [id, request] : pendingRequests_) {
            if (request.callback) {
                request.callback(responseJson);
                pendingRequests_.erase(id);
                break;
            }
        }
    }
}

void OSCClient::handlePingResponse(const juce::OSCMessage& message) {
    if (message.size() >= 1 && message[0].isString()) {
        std::string responseJson = message[0].getString().toStdString();
        for (auto& [id, request] : pendingRequests_) {
            if (request.callback) {
                request.callback(responseJson);
                pendingRequests_.erase(id);
                break;
            }
        }
    }
}

void OSCClient::handleUnknownMessage(const juce::OSCMessage& message) {
    std::cout << "OSCClient: Received unknown message: "
              << message.getAddressPattern().toString() << std::endl;
}

void OSCClient::cleanupTimedOutRequests() {
    auto now = juce::Time::getCurrentTime();
    auto it = pendingRequests_.begin();
    while (it != pendingRequests_.end()) {
        auto elapsed = now - it->second.timestamp;
        if (elapsed.inMilliseconds() > REQUEST_TIMEOUT_MS) {
            // Call callback with timeout error
            if (it->second.callback) {
                it->second.callback(R"({"status":"error","message":"Request timeout"})");
            }
            it = pendingRequests_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace kelly
