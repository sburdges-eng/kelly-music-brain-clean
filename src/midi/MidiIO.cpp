#include "daiw/midi/MidiIO.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <algorithm>
#include <cmath>
#include <utility>

namespace daiw {
namespace midi {

namespace {

// Convert JUCE velocity to DAiW velocity without losing dynamics.
static uint8_t toDaiwVelocity(float juceVelocity) {
  // JUCE getVelocity() normally returns 0..1, but some code paths may store 0..127.
  if (juceVelocity > 1.5f) {
    return static_cast<uint8_t>(std::clamp(juceVelocity, 0.0f, 127.0f));
  }
  return static_cast<uint8_t>(std::clamp(std::round(juceVelocity * 127.0f), 0.0f, 127.0f));
}

MidiMessage juceToDaiwMessage(const juce::MidiMessage &juceMsg) {
  MidiMessage msg;
  const auto channel = static_cast<MidiChannel>(
      std::clamp(juceMsg.getChannel() - 1, 0, 15)); // JUCE is 1-16

  if (juceMsg.isNoteOn()) {
    const auto velocity = toDaiwVelocity(juceMsg.getVelocity());
    msg = MidiMessage::noteOn(
        channel, static_cast<uint8_t>(juceMsg.getNoteNumber()), velocity);
  } else if (juceMsg.isNoteOff()) {
    const auto velocity = toDaiwVelocity(juceMsg.getVelocity());
    msg = MidiMessage::noteOff(
        channel, static_cast<uint8_t>(juceMsg.getNoteNumber()), velocity);
  } else if (juceMsg.isProgramChange()) {
    msg = MidiMessage::programChange(
        channel, static_cast<uint8_t>(juceMsg.getProgramChangeNumber()));
  } else if (juceMsg.isController()) {
    msg = MidiMessage::controlChange(
        channel, static_cast<uint8_t>(juceMsg.getControllerNumber()),
        static_cast<uint8_t>(juceMsg.getControllerValue()));
  } else if (juceMsg.isPitchWheel()) {
    msg = MidiMessage::pitchBend(
        channel, static_cast<uint16_t>(juceMsg.getPitchWheelValue()));
  } else if (juceMsg.isChannelPressure()) {
    msg = MidiMessage::channelPressure(
        channel, static_cast<uint8_t>(juceMsg.getChannelPressureValue()));
  } else {
    // Fallback: map from raw bytes when possible
    const auto *raw = juceMsg.getRawData();
    const int size = juceMsg.getRawDataSize();
    if (raw != nullptr && size >= 1) {
      const uint8_t status = static_cast<uint8_t>(raw[0] & 0xF0);
      const uint8_t data1 = size > 1 ? static_cast<uint8_t>(raw[1]) : 0;
      const uint8_t data2 = size > 2 ? static_cast<uint8_t>(raw[2]) : 0;

      switch (status) {
      case static_cast<uint8_t>(MessageType::NoteOn):
        msg = MidiMessage::noteOn(channel, data1, data2);
        break;
      case static_cast<uint8_t>(MessageType::NoteOff):
        msg = MidiMessage::noteOff(channel, data1, data2);
        break;
      case static_cast<uint8_t>(MessageType::ControlChange):
        msg = MidiMessage::controlChange(channel, data1, data2);
        break;
      case static_cast<uint8_t>(MessageType::ProgramChange):
        msg = MidiMessage::programChange(channel, data1);
        break;
      case static_cast<uint8_t>(MessageType::ChannelPressure):
        msg = MidiMessage::channelPressure(channel, data1);
        break;
      case static_cast<uint8_t>(MessageType::PitchBend):
        msg = MidiMessage::pitchBend(
            channel,
            static_cast<uint16_t>(static_cast<uint16_t>(data2) << 7 | data1));
        break;
      default:
        break;
      }
    }
  }

  msg.setTimestamp(static_cast<TickCount>(juceMsg.getTimeStamp() * 1000.0));
  return msg;
}

juce::MidiMessage daiwToJuceMessage(const MidiMessage &msg) {
  const int channel = std::clamp<int>(msg.getChannel(), 0, 15) + 1; // JUCE 1-16

  switch (msg.getType()) {
  case MessageType::NoteOn:
    return juce::MidiMessage::noteOn(channel, msg.getNoteNumber(),
                                     static_cast<juce::uint8>(msg.getVelocity()));
  case MessageType::NoteOff:
    return juce::MidiMessage::noteOff(
        channel, msg.getNoteNumber(), static_cast<juce::uint8>(msg.getVelocity()));
  case MessageType::ControlChange:
    return juce::MidiMessage::controllerEvent(channel, msg.getControllerNumber(),
                                              msg.getControllerValue());
  case MessageType::ProgramChange:
    return juce::MidiMessage::programChange(channel, msg.getData1());
  case MessageType::ChannelPressure:
    return juce::MidiMessage::channelPressureChange(
        channel, static_cast<int>(msg.getData1()));
  case MessageType::PitchBend:
    return juce::MidiMessage::pitchWheel(channel, msg.getPitchBendValue());
  default: {
    const uint8_t raw[3] = {msg.getStatusByte(), msg.getData1(), msg.getData2()};
    return juce::MidiMessage(raw, 3);
  }
  }
}

class InputCallbackWrapper : public juce::MidiInputCallback {
public:
  explicit InputCallbackWrapper(std::function<void(const juce::MidiMessage &)> handler)
      : handler_(std::move(handler)) {}

  void setHandler(std::function<void(const juce::MidiMessage &)> handler) {
    handler_ = std::move(handler);
  }

  void handleIncomingMidiMessage(juce::MidiInput *, const juce::MidiMessage &message) override {
    if (handler_) {
      handler_(message);
    }
  }

private:
  std::function<void(const juce::MidiMessage &)> handler_;
};

} // namespace

// ============================================================================
// MidiInput Implementation
// ============================================================================

struct MidiInput::Impl {
  std::unique_ptr<juce::MidiInput> input;
  MidiInputCallback callback;
  std::unique_ptr<InputCallbackWrapper> callbackWrapper;
  int deviceId = -1;
  bool isOpen = false;
  bool isRunning = false;
};

MidiInput::MidiInput() : impl_(std::make_unique<Impl>()) {
  impl_->callbackWrapper = std::make_unique<InputCallbackWrapper>(
      [this](const juce::MidiMessage &msg) {
        if (impl_->callback) {
          impl_->callback(juceToDaiwMessage(msg));
        }
      });
}

MidiInput::~MidiInput() { close(); }

std::vector<MidiDeviceInfo> MidiInput::getAvailableDevices() {
  std::vector<MidiDeviceInfo> devices;
  auto juceDevices = juce::MidiInput::getAvailableDevices();

  devices.reserve(static_cast<size_t>(juceDevices.size()));
  for (int i = 0; i < juceDevices.size(); ++i) {
    MidiDeviceInfo info;
    info.deviceId = i;
    info.name = juceDevices[i].name.toStdString();
    info.identifier = juceDevices[i].identifier.toStdString();
    info.isInput = true;
    info.isOutput = false;
    devices.push_back(info);
  }

  return devices;
}

bool MidiInput::open(int deviceId) {
  if (impl_->isOpen) {
    close();
  }

  auto devices = juce::MidiInput::getAvailableDevices();
  if (deviceId < 0 || deviceId >= devices.size()) {
    juce::Logger::writeToLog("MidiInput::open - invalid device id: " +
                             juce::String(deviceId));
    return false;
  }

  auto device = devices[deviceId];
  impl_->input = juce::MidiInput::openDevice(device.identifier, impl_->callbackWrapper.get());

  if (!impl_->input) {
    juce::Logger::writeToLog("MidiInput::open - failed to open: " + device.name);
    return false;
  }

  impl_->deviceId = deviceId;
  impl_->isOpen = true;
  return true;
}

void MidiInput::close() {
  if (impl_->isRunning && impl_->input) {
    impl_->input->stop();
  }

  impl_->input.reset();
  impl_->isOpen = false;
  impl_->isRunning = false;
  impl_->deviceId = -1;
}

bool MidiInput::isOpen() const { return impl_->isOpen; }

void MidiInput::setCallback(MidiInputCallback callback) {
  impl_->callback = std::move(callback);
  if (impl_->callbackWrapper) {
    impl_->callbackWrapper->setHandler([this](const juce::MidiMessage &msg) {
      if (impl_->callback) {
        impl_->callback(juceToDaiwMessage(msg));
      }
    });
  }
}

void MidiInput::start() {
  if (!impl_->isOpen || !impl_->input) {
    return;
  }

  impl_->input->start();
  impl_->isRunning = true;
}

void MidiInput::stop() {
  if (impl_->input && impl_->isRunning) {
    impl_->input->stop();
    impl_->isRunning = false;
  }
}

// ============================================================================
// MidiOutput Implementation
// ============================================================================

struct MidiOutput::Impl {
  std::unique_ptr<juce::MidiOutput> output;
  int deviceId = -1;
  bool isOpen = false;
};

MidiOutput::MidiOutput() : impl_(std::make_unique<Impl>()) {}
MidiOutput::~MidiOutput() { close(); }

std::vector<MidiDeviceInfo> MidiOutput::getAvailableDevices() {
  std::vector<MidiDeviceInfo> devices;
  auto juceDevices = juce::MidiOutput::getAvailableDevices();

  devices.reserve(static_cast<size_t>(juceDevices.size()));
  for (int i = 0; i < juceDevices.size(); ++i) {
    MidiDeviceInfo info;
    info.deviceId = i;
    info.name = juceDevices[i].name.toStdString();
    info.identifier = juceDevices[i].identifier.toStdString();
    info.isInput = false;
    info.isOutput = true;
    devices.push_back(info);
  }

  return devices;
}

bool MidiOutput::open(int deviceId) {
  if (impl_->isOpen) {
    close();
  }

  auto devices = juce::MidiOutput::getAvailableDevices();
  if (deviceId < 0 || deviceId >= devices.size()) {
    return false;
  }

  auto device = devices[deviceId];
  impl_->output = juce::MidiOutput::openDevice(device.identifier);
  if (!impl_->output) {
    return false;
  }

  impl_->deviceId = deviceId;
  impl_->isOpen = true;
  return true;
}

void MidiOutput::close() {
  if (impl_->isOpen && impl_->output) {
    allNotesOff();
  }

  impl_->output.reset();
  impl_->deviceId = -1;
  impl_->isOpen = false;
}

bool MidiOutput::isOpen() const { return impl_->isOpen; }

bool MidiOutput::sendMessage(const MidiMessage &message) {
  if (!impl_->isOpen || !impl_->output) {
    return false;
  }

  auto juceMsg = daiwToJuceMessage(message);
  impl_->output->sendMessageNow(juceMsg);
  return true;
}

void MidiOutput::allNotesOff() {
  if (!impl_->isOpen || !impl_->output) {
    return;
  }

  for (int channel = 0; channel < 16; ++channel) {
    MidiMessage cc =
        MidiMessage::controlChange(static_cast<MidiChannel>(channel),
                                   static_cast<uint8_t>(CC::AllNotesOff), 0);
    impl_->output->sendMessageNow(daiwToJuceMessage(cc));
  }
}

} // namespace midi
} // namespace daiw
