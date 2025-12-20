#pragma once

#include "penta/osc/RTMessageQueue.h"
#include <cstdint>
#include <string>
#include <memory>
#include <atomic>
#include <thread>

namespace penta::osc {

/**
 * OSC server using lock-free message queue
 * Receives OSC messages without blocking RT threads
 */
class OSCServer {
public:
    explicit OSCServer(const std::string& address, uint16_t port);
    ~OSCServer();
    
    // Non-copyable, non-movable
    OSCServer(const OSCServer&) = delete;
    OSCServer& operator=(const OSCServer&) = delete;
    
    // Start/stop server
    bool start();
    void stop();
    
    bool isRunning() const { return running_.load(); }
    
    // RT-safe: Get message queue for polling
    RTMessageQueue& getMessageQueue() { return *messageQueue_; }
    
private:
    void receiveThread();
    
    std::string address_;
    uint16_t port_;
    std::atomic<bool> running_;
    std::unique_ptr<RTMessageQueue> messageQueue_;
    std::thread receiveThread_;
    
    // Platform-specific socket handle
    struct SocketImpl;
    std::unique_ptr<SocketImpl> socket_;
};

} // namespace penta::osc
