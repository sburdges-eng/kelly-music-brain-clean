#include "penta/osc/OSCMessage.h"

#include <stdexcept>
#include <utility>

namespace penta::osc {

OSCMessage::OSCMessage()
    : OSCMessage("") {}

OSCMessage::OSCMessage(std::string address)
    : address_(std::move(address))
    , arguments_()
    , timestamp_(0) {}

void OSCMessage::setAddress(const std::string& address) {
    address_ = address;
}

const std::string& OSCMessage::getAddress() const noexcept {
    return address_;
}

void OSCMessage::setTimestamp(uint64_t timestamp) noexcept {
    timestamp_ = timestamp;
}

uint64_t OSCMessage::getTimestamp() const noexcept {
    return timestamp_;
}

void OSCMessage::addInt(int32_t value) {
    arguments_.emplace_back(value);
}

void OSCMessage::addFloat(float value) {
    arguments_.emplace_back(value);
}

void OSCMessage::addString(const std::string& value) {
    arguments_.emplace_back(value);
}

void OSCMessage::addString(const char* value) {
    arguments_.emplace_back(std::string(value));
}

size_t OSCMessage::getArgumentCount() const noexcept {
    return arguments_.size();
}

const OSCValue& OSCMessage::getArgument(size_t index) const {
    if (index >= arguments_.size()) {
        throw std::out_of_range("OSCMessage argument index out of range");
    }
    return arguments_[index];
}

void OSCMessage::clear() noexcept {
    arguments_.clear();
    timestamp_ = 0;
}

} // namespace penta::osc
