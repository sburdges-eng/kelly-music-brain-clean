#include "penta/osc/RTMessageQueue.h"

namespace penta::osc {

RTMessageQueue::RTMessageQueue(size_t capacity)
    : capacity_(capacity)
{
    // TODO: Week 6 implementation - lock-free queue using readerwriterqueue
}

RTMessageQueue::~RTMessageQueue() = default;

bool RTMessageQueue::push(const OSCMessage& message) noexcept { (void)message;
    // Stub implementation
    return true;
}

bool RTMessageQueue::pop(OSCMessage& message) noexcept { (void)message;
    // Stub implementation
    return false;
}

bool RTMessageQueue::isEmpty() const noexcept {
    return true;
}

size_t RTMessageQueue::size() const noexcept {
    return 0;
}

} // namespace penta::osc
