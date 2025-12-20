#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

#include "penta/osc/OSCMessage.h"

namespace penta::osc {

/**
 * Lock-free message queue for RT-safe OSC communication
 * Single-producer, single-consumer queue
 */
class RTMessageQueue {
public:
    explicit RTMessageQueue(size_t capacity = 4096);
    ~RTMessageQueue();
    
    // Non-copyable, non-movable
    RTMessageQueue(const RTMessageQueue&) = delete;
    RTMessageQueue& operator=(const RTMessageQueue&) = delete;
    
    // RT-safe: Push message (returns false if full)
    bool push(const OSCMessage& message) noexcept;
    
    // RT-safe: Pop message (returns false if empty)
    bool pop(OSCMessage& outMessage) noexcept;
    
    // RT-safe: Check if queue is empty
    bool isEmpty() const noexcept;
    
    // RT-safe: Get approximate size
    size_t size() const noexcept;
    
    size_t capacity() const noexcept { return capacity_; }
    
private:
    std::vector<OSCMessage> buffer_;
    size_t capacity_;
    std::atomic<size_t> writeIndex_;
    std::atomic<size_t> readIndex_;
};

} // namespace penta::osc
