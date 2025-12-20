#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace penta::osc {

using OSCValue = std::variant<int32_t, float, std::string, std::vector<uint8_t>>;

class OSCMessage {
public:
    OSCMessage();
    explicit OSCMessage(std::string address);

    void setAddress(const std::string& address);
    const std::string& getAddress() const noexcept;

    void setTimestamp(uint64_t timestamp) noexcept;
    uint64_t getTimestamp() const noexcept;

    void addInt(int32_t value);
    void addFloat(float value);
    void addString(const std::string& value);
    void addString(const char* value);

    size_t getArgumentCount() const noexcept;
    const OSCValue& getArgument(size_t index) const;

    void clear() noexcept;

private:
    std::string address_;
    std::vector<OSCValue> arguments_;
    uint64_t timestamp_;
};

} // namespace penta::osc
