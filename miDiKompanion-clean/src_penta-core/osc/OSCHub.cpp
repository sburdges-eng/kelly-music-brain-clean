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
    // Store pattern-callback mapping
    callbacks_[pattern] = callback;
    
    // Process messages from queue and dispatch to matching callbacks
    // This would typically be called from a non-RT thread periodically
}

void OSCHub::processCallbacks() {
    OSCMessage message;
    while (receiveMessage(message)) {
        // Find matching callbacks for this message
        const std::string& address = message.getAddress();
        
        // Exact match
        auto it = callbacks_.find(address);
        if (it != callbacks_.end()) {
            it->second(message);
        }
        
        // Pattern matching (simple wildcard support)
        for (const auto& [pattern, callback] : callbacks_) {
            if (matchPattern(address, pattern)) {
                callback(message);
            }
        }
    }
}

bool OSCHub::matchPattern(const std::string& address, const std::string& pattern) const {
    // Simple pattern matching with wildcards
    // * matches any sequence of characters
    // ? matches any single character
    
    size_t addrPos = 0;
    size_t patPos = 0;
    
    while (addrPos < address.length() && patPos < pattern.length()) {
        if (pattern[patPos] == '*') {
            // Wildcard - try to match rest of pattern
            if (patPos + 1 >= pattern.length()) {
                return true;  // * at end matches everything
            }
            
            // Try to match remaining pattern at each position
            for (size_t i = addrPos; i <= address.length(); ++i) {
                if (matchPattern(address.substr(i), pattern.substr(patPos + 1))) {
                    return true;
                }
            }
            return false;
        } else if (pattern[patPos] == '?' || pattern[patPos] == address[addrPos]) {
            // Single char match or exact match
            addrPos++;
            patPos++;
        } else {
            return false;
        }
    }
    
    // Check if we matched entire pattern and address
    return addrPos == address.length() && patPos == pattern.length();
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
