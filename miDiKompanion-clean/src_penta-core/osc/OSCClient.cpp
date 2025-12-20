#include "penta/osc/OSCClient.h"
#include "penta/osc/RTMessageQueue.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

namespace penta::osc {

// Forward declare SocketImpl
struct OSCClient::SocketImpl {
    int fd = -1;
    sockaddr_in destAddr{};
};

OSCClient::OSCClient(const std::string& address, uint16_t port)
    : address_(address)
    , port_(port)
    , socket_(std::make_unique<SocketImpl>())
{
    // Create UDP socket
    socket_->fd = socket(AF_INET, SOCK_DGRAM, 0);
    
    if (socket_->fd >= 0) {
        // Setup destination address
        std::memset(&socket_->destAddr, 0, sizeof(socket_->destAddr));
        socket_->destAddr.sin_family = AF_INET;
        socket_->destAddr.sin_port = htons(port);
        inet_pton(AF_INET, address.c_str(), &socket_->destAddr.sin_addr);
    }
}

OSCClient::~OSCClient() {
    if (socket_->fd >= 0) {
        close(socket_->fd);
    }
}

bool OSCClient::send(const OSCMessage& message) noexcept {
    if (socket_->fd < 0) {
        return false;
    }
    
    // Simple OSC message encoding
    // Format: /address\0,types\0[args...]
    std::vector<uint8_t> buffer;
    
    // Add address with null terminator and padding to 4-byte boundary
    const std::string& addr = message.getAddress();
    buffer.insert(buffer.end(), addr.begin(), addr.end());
    buffer.push_back(0);
    
    // Pad to 4-byte boundary
    while (buffer.size() % 4 != 0) {
        buffer.push_back(0);
    }
    
    // Add type tag string
    std::string typeTags = ",";
    for (size_t i = 0; i < message.getArgumentCount(); ++i) {
        const auto& arg = message.getArgument(i);
        if (std::holds_alternative<int32_t>(arg)) {
            typeTags += "i";
        } else if (std::holds_alternative<float>(arg)) {
            typeTags += "f";
        } else if (std::holds_alternative<std::string>(arg)) {
            typeTags += "s";
        }
    }
    
    buffer.insert(buffer.end(), typeTags.begin(), typeTags.end());
    buffer.push_back(0);
    
    // Pad to 4-byte boundary
    while (buffer.size() % 4 != 0) {
        buffer.push_back(0);
    }
    
    // Add arguments
    for (size_t i = 0; i < message.getArgumentCount(); ++i) {
        const auto& arg = message.getArgument(i);
        
        if (std::holds_alternative<int32_t>(arg)) {
            int32_t val = std::get<int32_t>(arg);
            uint32_t netVal = htonl(static_cast<uint32_t>(val));
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&netVal);
            buffer.insert(buffer.end(), bytes, bytes + 4);
        } else if (std::holds_alternative<float>(arg)) {
            float val = std::get<float>(arg);
            uint32_t temp;
            std::memcpy(&temp, &val, sizeof(float));
            uint32_t netVal = htonl(temp);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&netVal);
            buffer.insert(buffer.end(), bytes, bytes + 4);
        } else if (std::holds_alternative<std::string>(arg)) {
            const std::string& str = std::get<std::string>(arg);
            buffer.insert(buffer.end(), str.begin(), str.end());
            buffer.push_back(0);
            while (buffer.size() % 4 != 0) {
                buffer.push_back(0);
            }
        }
    }
    
    // Send via UDP
    ssize_t sent = sendto(socket_->fd, buffer.data(), buffer.size(), 0,
                          reinterpret_cast<sockaddr*>(&socket_->destAddr),
                          sizeof(socket_->destAddr));
    
    return sent >= 0;
}

bool OSCClient::sendFloat(const char* address, float value) noexcept {
    OSCMessage msg(address);
    msg.addFloat(value);
    return send(msg);
}

bool OSCClient::sendInt(const char* address, int32_t value) noexcept {
    OSCMessage msg(address);
    msg.addInt(value);
    return send(msg);
}

bool OSCClient::sendString(const char* address, const char* value) noexcept {
    OSCMessage msg(address);
    msg.addString(value);
    return send(msg);
}

void OSCClient::setDestination(const std::string& address, uint16_t port) {
    address_ = address;
    port_ = port;
    
    // Update destination address
    socket_->destAddr.sin_port = htons(port);
    inet_pton(AF_INET, address.c_str(), &socket_->destAddr.sin_addr);
}

} // namespace penta::osc
