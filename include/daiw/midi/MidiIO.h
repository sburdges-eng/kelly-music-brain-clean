/**
 * @file MidiIO.h
 * @brief MIDI input/output device management
 *
 * Implementation using JUCE MIDI classes.
 */

#pragma once

#include "daiw/midi/MidiMessage.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace daiw {
namespace midi {

/**
 * @brief Callback function for incoming MIDI messages
 */
using MidiInputCallback = std::function<void(const MidiMessage&)>;

/**
 * @brief Information about a MIDI device
 */
struct MidiDeviceInfo {
    int deviceId = -1;
    std::string name;
    std::string identifier;
    bool isInput = false;
    bool isOutput = false;
};

/**
 * @brief MIDI input device interface
 */
class MidiInput {
public:
    MidiInput();
    ~MidiInput();

    /**
     * @brief Enumerate available MIDI input devices
     */
    static std::vector<MidiDeviceInfo> getAvailableDevices();

    /**
     * @brief Open a MIDI input device
     * @param deviceId Device ID from getAvailableDevices()
     * @return true if successful
     */
    bool open(int deviceId);

    /**
     * @brief Close the MIDI input device
     */
    void close();

    /**
     * @brief Check if device is open
     */
    [[nodiscard]] bool isOpen() const;

    /**
     * @brief Set callback for incoming messages
     */
    void setCallback(MidiInputCallback callback);

    /**
     * @brief Start receiving MIDI messages
     */
    void start();

    /**
     * @brief Stop receiving MIDI messages
     */
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief MIDI output device interface
 */
class MidiOutput {
public:
    MidiOutput();
    ~MidiOutput();

    /**
     * @brief Enumerate available MIDI output devices
     */
    static std::vector<MidiDeviceInfo> getAvailableDevices();

    /**
     * @brief Open a MIDI output device
     * @param deviceId Device ID from getAvailableDevices()
     * @return true if successful
     */
    bool open(int deviceId);

    /**
     * @brief Close the MIDI output device
     */
    void close();

    /**
     * @brief Check if device is open
     */
    [[nodiscard]] bool isOpen() const;

    /**
     * @brief Send a MIDI message
     * @param message The message to send
     * @return true if successful
     */
    bool sendMessage(const MidiMessage& message);

    /**
     * @brief Send all notes off on all channels
     */
    void allNotesOff();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace midi
} // namespace daiw
