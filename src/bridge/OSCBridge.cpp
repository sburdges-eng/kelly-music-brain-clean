#include "bridge/OSCBridge.h"
#include <juce_osc/juce_osc.h>
#include <juce_core/juce_core.h>

namespace kelly {
namespace bridge {

OSCBridge::OSCBridge() {
}

OSCBridge::~OSCBridge() {
    shutdown();
}

bool OSCBridge::initialize(
    const std::string& brainHost,
    int brainPort,
    int listenPort)
{
    brainHost_ = brainHost;
    brainPort_ = brainPort;
    listenPort_ = listenPort;

    try {
        // Create OSC sender
        sender_ = std::make_unique<juce::OSCSender>();
        if (!sender_->connect(brainHost, brainPort)) {
            juce::Logger::writeToLog(juce::String("OSCBridge: Failed to connect sender to ") +
                                     juce::String(brainHost.c_str()) + juce::String(":") + juce::String(brainPort));
            return false;
        }

        // Create OSC receiver
        receiver_ = std::make_unique<juce::OSCReceiver>();
        if (!receiver_->connect(listenPort)) {
            juce::Logger::writeToLog(juce::String("OSCBridge: Failed to bind receiver to port ") +
                                     juce::String(listenPort));
            return false;
        }

        // Register message handler (will receive all messages)
        receiver_->addListener(this);

        connected_ = true;
        juce::Logger::writeToLog(juce::String("OSCBridge: Connected to brain server at ") +
                                 juce::String(brainHost.c_str()) + juce::String(":") + juce::String(brainPort));
        return true;
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("OSCBridge: Exception during initialization: " +
                                 std::string(e.what()));
        return false;
    }
}

void OSCBridge::requestGenerate(
    const std::string& text,
    const std::string& motivation,
    float chaos,
    float vulnerability,
    OSCResponseHandler callback)
{
    if (!connected_) {
        juce::Logger::writeToLog("OSCBridge: Not connected, cannot send generate request");
        return;
    }

    // Create JSON parameters
    juce::DynamicObject::Ptr params = new juce::DynamicObject();
    params->setProperty("text", juce::String(text));
    params->setProperty("motivation", juce::String(motivation));
    params->setProperty("chaos", chaos);
    params->setProperty("vulnerability", vulnerability);

    juce::var paramsVar(params);
    juce::String jsonParams = juce::JSON::toString(paramsVar);

    // Store callback
    int msgId = generateMessageId();
    responseHandlers_[msgId] = callback;

    // Send OSC message
    juce::OSCMessage msg("/daiw/generate");
    msg.addString(jsonParams);
    msg.addInt32(msgId);  // Include message ID for response matching

    if (!sender_->send(msg)) {
        juce::Logger::writeToLog("OSCBridge: Failed to send generate request");
        responseHandlers_.erase(msgId);
    }
}

void OSCBridge::requestAnalyzeChords(
    const std::string& progression,
    OSCResponseHandler callback)
{
    if (!connected_) return;

    int msgId = generateMessageId();
    responseHandlers_[msgId] = callback;

    juce::OSCMessage msg("/daiw/analyze/chords");
    msg.addString(progression);
    msg.addInt32(msgId);

    sender_->send(msg);
}

void OSCBridge::requestIntentProcess(
    const std::string& intentFile,
    OSCResponseHandler callback)
{
    if (!connected_) return;

    int msgId = generateMessageId();
    responseHandlers_[msgId] = callback;

    juce::OSCMessage msg("/daiw/intent/process");
    msg.addString(intentFile);
    msg.addInt32(msgId);

    sender_->send(msg);
}

void OSCBridge::requestIntentSuggest(
    const std::string& emotion,
    OSCResponseHandler callback)
{
    if (!connected_) return;

    int msgId = generateMessageId();
    responseHandlers_[msgId] = callback;

    juce::OSCMessage msg("/daiw/intent/suggest");
    msg.addString(emotion);
    msg.addInt32(msgId);

    sender_->send(msg);
}

void OSCBridge::ping(OSCResponseHandler callback)
{
    if (!connected_) return;

    int msgId = generateMessageId();
    responseHandlers_[msgId] = callback;

    juce::OSCMessage msg("/daiw/ping");
    msg.addInt32(msgId);

    sender_->send(msg);
}

void OSCBridge::shutdown()
{
    if (receiver_) {
        receiver_->removeListener(this);
        receiver_->disconnect();
    }

    if (sender_) {
        sender_->disconnect();
    }

    connected_ = false;
    responseHandlers_.clear();
}

bool OSCBridge::sendMessage(const OSCMessage& message)
{
    if (!connected_ || !sender_) {
        return false;
    }

    juce::OSCMessage oscMsg(juce::OSCAddressPattern(message.address));
    for (const auto& arg : message.arguments) {
        // Convert juce::var to appropriate OSC argument type
        if (arg.isInt()) {
            oscMsg.addInt32(static_cast<int>(arg));
        } else if (arg.isDouble() || arg.isInt64()) {
            oscMsg.addFloat32(static_cast<float>(static_cast<double>(arg)));
        } else if (arg.isString()) {
            oscMsg.addString(arg.toString());
        } else if (arg.isBool()) {
            oscMsg.addInt32(arg ? 1 : 0);
        }
    }

    return sender_->send(oscMsg);
}

void OSCBridge::oscMessageReceived(const juce::OSCMessage& message)
{
    // Extract message ID if present
    int msgId = -1;
    if (message.size() > 0 && message[0].isInt32()) {
        msgId = message[0].getInt32();
    }

    // Find handler
    if (msgId > 0 && responseHandlers_.count(msgId)) {
        auto handler = responseHandlers_[msgId];
        responseHandlers_.erase(msgId);

        // Parse response (expect JSON string)
        if (message.size() > 1 && message[1].isString()) {
            juce::String jsonStr = message[1].getString();
            juce::var response = juce::JSON::parse(jsonStr);
            handler(response);
        } else {
            // Fallback: pass raw message
            juce::var response;
            handler(response);
        }
    } else {
        // No handler found or no message ID
        juce::Logger::writeToLog("OSCBridge: Received message without handler: " +
                                 message.getAddressPattern().toString());
    }
}

void OSCBridge::handleOSCMessage(const juce::OSCMessage& message)
{
    oscMessageReceived(message);
}

} // namespace bridge
} // namespace kelly
