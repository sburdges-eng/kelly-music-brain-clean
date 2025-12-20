#include "penta/osc/OSCHub.h"

namespace penta::osc {

OSCHub::OSCHub() : OSCHub(Config{}) {}

OSCHub::OSCHub(const Config& config)
    : config_(config)
    , server_(std::make_unique<OSCServer>(config.serverAddress, config.serverPort))
    , client_(std::make_unique<OSCClient>(config.clientAddress, config.clientPort))
    , messageQueue_(std::make_unique<RTMessageQueue>(config.queueSize))
{
}

OSCHub::~OSCHub() {
    stop();
}

bool OSCHub::start() {
    if (!server_->start()) {
        return false;
    }
    
    // Client doesn't need explicit connect in current implementation
    return true;
}

void OSCHub::stop() {
    if (server_) {
        server_->stop();
    }
    
    // Client doesn't need explicit disconnect
}

bool OSCHub::sendMessage(const OSCMessage& message) noexcept {
    if (!client_) {
        return false;
    }
    return client_->send(message);
}

bool OSCHub::receiveMessage(OSCMessage& outMessage) noexcept {
    if (!server_) {
        return false;
    }
    return server_->getMessageQueue().pop(outMessage);
}

void OSCHub::registerCallback(const std::string& pattern, MessageCallback callback) {
    (void)pattern;
    (void)callback;
    // TODO: Implement pattern-based routing in Week 10
}

void OSCHub::updateConfig(const Config& config) {
    bool wasRunning = false;
    
    if (server_ && server_->isRunning()) {
        wasRunning = true;
        stop();
    }
    
    config_ = config;
    
    // Recreate server and client with new config
    server_ = std::make_unique<OSCServer>(config.serverAddress, config.serverPort);
    client_ = std::make_unique<OSCClient>(config.clientAddress, config.clientPort);
    messageQueue_ = std::make_unique<RTMessageQueue>(config.queueSize);
    
    if (wasRunning) {
        start();
    }
}

} // namespace penta::osc
