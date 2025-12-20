#pragma once

#include <cstdint>
#include <string>
#include <memory>

namespace penta::osc {

class OSCMessage;

/**
 * OSC client for sending messages
 * RT-safe message sending to DAWs and controllers
 */
class OSCClient {
public:
    explicit OSCClient(const std::string& address, uint16_t port);
    ~OSCClient();
    
    // Non-copyable, non-movable
    OSCClient(const OSCClient&) = delete;
    OSCClient& operator=(const OSCClient&) = delete;
    
    // RT-safe: Send OSC message
    bool send(const OSCMessage& message) noexcept;
    
    // RT-safe: Send simple messages
    bool sendFloat(const char* address, float value) noexcept;
    bool sendInt(const char* address, int32_t value) noexcept;
    bool sendString(const char* address, const char* value) noexcept;
    
    // Configuration
    void setDestination(const std::string& address, uint16_t port);
    
private:
    std::string address_;
    uint16_t port_;
    
    // Platform-specific socket handle
    struct SocketImpl;
    std::unique_ptr<SocketImpl> socket_;
};

} // namespace penta::osc
