#include "penta/osc/RTMessageQueue.h"

namespace penta::osc {

RTMessageQueue::RTMessageQueue(size_t capacity)
    : buffer_(capacity)
    , capacity_(capacity)
    , writeIndex_(0)
    , readIndex_(0)
{
    // Pre-allocate message buffer for lock-free operation
}

RTMessageQueue::~RTMessageQueue() = default;

bool RTMessageQueue::push(const OSCMessage& message) noexcept {
    size_t currentWrite = writeIndex_.load(std::memory_order_relaxed);
    size_t nextWrite = (currentWrite + 1) % capacity_;
    
    // Check if queue is full
    if (nextWrite == readIndex_.load(std::memory_order_acquire)) {
        return false;  // Queue full
    }
    
    // Copy message to buffer
    buffer_[currentWrite] = message;
    
    // Update write index (release semantics ensures message is written before index update)
    writeIndex_.store(nextWrite, std::memory_order_release);
    
    return true;
}

bool RTMessageQueue::pop(OSCMessage& outMessage) noexcept {
    size_t currentRead = readIndex_.load(std::memory_order_relaxed);
    
    // Check if queue is empty
    if (currentRead == writeIndex_.load(std::memory_order_acquire)) {
        return false;  // Queue empty
    }
    
    // Copy message from buffer
    outMessage = buffer_[currentRead];
    
    // Update read index (release semantics)
    size_t nextRead = (currentRead + 1) % capacity_;
    readIndex_.store(nextRead, std::memory_order_release);
    
    return true;
}

bool RTMessageQueue::isEmpty() const noexcept {
    return readIndex_.load(std::memory_order_acquire) == 
           writeIndex_.load(std::memory_order_acquire);
}

size_t RTMessageQueue::size() const noexcept {
    size_t write = writeIndex_.load(std::memory_order_acquire);
    size_t read = readIndex_.load(std::memory_order_acquire);
    
    if (write >= read) {
        return write - read;
    } else {
        return capacity_ - read + write;
    }
}

} // namespace penta::osc
