#include "penta/osc/OSCServer.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <thread>

namespace penta::osc {

// Forward declare SocketImpl
struct OSCServer::SocketImpl {
    int fd = -1;
    sockaddr_in addr{};
};

OSCServer::OSCServer(const std::string& address, uint16_t port)
    : address_(address)
    , port_(port)
    , running_(false)
    , messageQueue_(std::make_unique<RTMessageQueue>(4096))
    , socket_(std::make_unique<SocketImpl>())
{
    // Create UDP socket
    socket_->fd = socket(AF_INET, SOCK_DGRAM, 0);
    
    if (socket_->fd >= 0) {
        // Setup socket options for reuse
        int opt = 1;
        setsockopt(socket_->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        // Bind to address
        std::memset(&socket_->addr, 0, sizeof(socket_->addr));
        socket_->addr.sin_family = AF_INET;
        socket_->addr.sin_port = htons(port);
        
        if (address.empty() || address == "0.0.0.0") {
            socket_->addr.sin_addr.s_addr = INADDR_ANY;
        } else {
            inet_pton(AF_INET, address.c_str(), &socket_->addr.sin_addr);
        }
        
        bind(socket_->fd, reinterpret_cast<sockaddr*>(&socket_->addr), sizeof(socket_->addr));
    }
}

OSCServer::~OSCServer() {
    stop();
    
    if (socket_->fd >= 0) {
        close(socket_->fd);
    }
}

bool OSCServer::start() {
    if (running_.load(std::memory_order_acquire) || socket_->fd < 0) {
        return false;
    }
    
    running_.store(true, std::memory_order_release);
    
    // Start receiver thread
    receiveThread_ = std::thread(&OSCServer::receiveThread, this);
    
    return true;
}

void OSCServer::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }
    
    running_.store(false, std::memory_order_release);
    
    if (receiveThread_.joinable()) {
        receiveThread_.join();
    }
}

void OSCServer::receiveThread() {
    const size_t bufferSize = 8192;
    std::vector<uint8_t> buffer(bufferSize);
    
    sockaddr_in senderAddr;
    socklen_t senderLen = sizeof(senderAddr);
    
    while (running_.load(std::memory_order_acquire)) {
        // Receive OSC packet
        ssize_t received = recvfrom(socket_->fd, buffer.data(), bufferSize, 0,
                                    reinterpret_cast<sockaddr*>(&senderAddr), &senderLen);
        
        if (received < 0) {
            // Error or timeout - continue
            continue;
        }
        
        // Parse OSC message (simplified parser)
        if (received > 0) {
            OSCMessage message;
            
            // Extract address
            size_t pos = 0;
            std::string address;
            while (pos < static_cast<size_t>(received) && buffer[pos] != 0) {
                address += static_cast<char>(buffer[pos++]);
            }
            
            if (!address.empty()) {
                message.setAddress(address);
                
                // Align to 4-byte boundary
                pos = (pos + 4) & ~3;
                
                // Extract type tags
                if (pos < static_cast<size_t>(received) && buffer[pos] == ',') {
                    pos++;  // Skip comma
                    std::string typeTags;
                    while (pos < static_cast<size_t>(received) && buffer[pos] != 0) {
                        typeTags += static_cast<char>(buffer[pos++]);
                    }
                    
                    // Align to 4-byte boundary
                    pos = (pos + 4) & ~3;
                    
                    // Extract arguments based on type tags
                    for (char typeTag : typeTags) {
                        if (pos + 4 > static_cast<size_t>(received)) {
                            break;
                        }
                        
                        if (typeTag == 'i') {
                            // Integer
                            uint32_t netVal;
                            std::memcpy(&netVal, &buffer[pos], 4);
                            int32_t val = static_cast<int32_t>(ntohl(netVal));
                            message.addInt(val);
                            pos += 4;
                        } else if (typeTag == 'f') {
                            // Float
                            uint32_t netVal;
                            std::memcpy(&netVal, &buffer[pos], 4);
                            netVal = ntohl(netVal);
                            float val;
                            std::memcpy(&val, &netVal, sizeof(float));
                            message.addFloat(val);
                            pos += 4;
                        } else if (typeTag == 's') {
                            // String
                            std::string str;
                            while (pos < static_cast<size_t>(received) && buffer[pos] != 0) {
                                str += static_cast<char>(buffer[pos++]);
                            }
                            message.addString(str);
                            // Align to 4-byte boundary
                            pos = (pos + 4) & ~3;
                        }
                    }
                }
                
                // Push message to queue
                messageQueue_->push(message);
            }
        }
    }
}

} // namespace penta::osc
