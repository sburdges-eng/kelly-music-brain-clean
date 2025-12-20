#pragma once

#include <atomic>
#include <array>
#include <cstring>
#include <cstddef>

namespace kelly {

/**
 * LockFreeRingBuffer - Lock-free circular buffer for audio/ML thread communication.
 *
 * Thread-safe ring buffer using atomic operations for producer-consumer pattern.
 * Designed for real-time audio processing where blocking is not acceptable.
 *
 * @tparam T Element type
 * @tparam Capacity Buffer capacity (must be power of 2 for optimal performance)
 */
template<typename T, size_t Capacity>
class LockFreeRingBuffer {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

    LockFreeRingBuffer() : writePos_(0), readPos_(0) {
        buffer_.fill(T{});
    }

    /**
     * Push data into buffer (producer - typically audio thread).
     * @param data Pointer to data to push
     * @param count Number of elements to push
     * @return true if successful, false if buffer full
     */
    bool push(const T* data, size_t count) {
        const size_t currentWrite = writePos_.load(std::memory_order_relaxed);
        const size_t currentRead = readPos_.load(std::memory_order_acquire);

        const size_t available = Capacity - (currentWrite - currentRead);
        if (count > available) {
            return false;  // Buffer full
        }

        const size_t writeIndex = currentWrite & (Capacity - 1);  // Modulo (power of 2)
        const size_t firstPart = std::min(count, Capacity - writeIndex);

        // Copy first part
        std::memcpy(&buffer_[writeIndex], data, firstPart * sizeof(T));

        // Copy wrap-around part if needed
        if (count > firstPart) {
            std::memcpy(&buffer_[0], data + firstPart, (count - firstPart) * sizeof(T));
        }

        writePos_.store(currentWrite + count, std::memory_order_release);
        return true;
    }

    /**
     * Pop data from buffer (consumer - typically ML thread).
     * @param data Pointer to destination buffer
     * @param count Number of elements to pop
     * @return true if successful, false if buffer empty
     */
    bool pop(T* data, size_t count) {
        const size_t currentRead = readPos_.load(std::memory_order_relaxed);
        const size_t currentWrite = writePos_.load(std::memory_order_acquire);

        const size_t available = currentWrite - currentRead;
        if (count > available) {
            return false;  // Not enough data
        }

        const size_t readIndex = currentRead & (Capacity - 1);  // Modulo (power of 2)
        const size_t firstPart = std::min(count, Capacity - readIndex);

        // Copy first part
        std::memcpy(data, &buffer_[readIndex], firstPart * sizeof(T));

        // Copy wrap-around part if needed
        if (count > firstPart) {
            std::memcpy(data + firstPart, &buffer_[0], (count - firstPart) * sizeof(T));
        }

        readPos_.store(currentRead + count, std::memory_order_release);
        return true;
    }

    /**
     * Get number of elements available to read.
     */
    size_t availableToRead() const {
        return writePos_.load(std::memory_order_acquire) -
               readPos_.load(std::memory_order_relaxed);
    }

    /**
     * Get number of elements available to write.
     */
    size_t availableToWrite() const {
        return Capacity - availableToRead();
    }

    /**
     * Check if buffer is empty.
     */
    bool isEmpty() const {
        return availableToRead() == 0;
    }

    /**
     * Check if buffer is full.
     */
    bool isFull() const {
        return availableToWrite() == 0;
    }

    /**
     * Clear buffer (reset positions).
     */
    void clear() {
        writePos_.store(0, std::memory_order_release);
        readPos_.store(0, std::memory_order_release);
    }

private:
    std::array<T, Capacity> buffer_;
    std::atomic<size_t> writePos_;
    std::atomic<size_t> readPos_;
};

} // namespace kelly
