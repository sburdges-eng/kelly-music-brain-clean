# Audio Software/Hardware Interface Guide

This guide covers audio interfaces, drivers, and the connection between software and hardware for the penta-core team.

## Overview

Audio interfaces bridge the gap between analog audio (microphones, instruments) and digital audio (computers, software). Understanding both hardware and software components is essential for audio development.

## Hardware Components

### Audio Interface Types

| Type | Connection | Use Case |
|------|------------|----------|
| USB | USB 2.0/3.0 | Home studios, portable recording |
| Thunderbolt | Thunderbolt 3/4 | Low latency, high channel count |
| PCIe | Internal card | Professional studios, lowest latency |
| FireWire | FireWire 400/800 | Legacy devices |
| Dante/AVB | Ethernet | Large installations, networked audio |

### Key Specifications

- **Sample Rate**: 44.1kHz, 48kHz, 96kHz, 192kHz
- **Bit Depth**: 16-bit, 24-bit, 32-bit float
- **Latency**: Round-trip delay (input to output)
- **I/O Count**: Number of inputs and outputs
- **Preamp Quality**: Gain range, noise floor, THD

### Signal Flow

```
Microphone/Instrument
        │
        ▼
┌───────────────────┐
│  Analog Preamp    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   A/D Converter   │  ← Analog to Digital
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Audio Driver    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   DAW / Software  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Audio Driver    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   D/A Converter   │  ← Digital to Analog
└───────────────────┘
        │
        ▼
  Speakers/Headphones
```

## Software Components

### Audio Drivers

| Driver Type | Platform | Latency | Description |
|------------|----------|---------|-------------|
| ASIO | Windows | Low | Industry standard for Windows audio |
| Core Audio | macOS | Low | Native macOS audio system |
| ALSA | Linux | Low | Advanced Linux Sound Architecture |
| JACK | Linux/macOS | Very Low | Pro audio server with routing |
| WASAPI | Windows | Medium | Windows Audio Session API |
| WDM/KS | Windows | Variable | Windows Driver Model |

### Core Audio (macOS)

Core Audio is Apple's low-level audio API:

```swift
import AudioToolbox
import AVFoundation

class AudioEngine {
    let engine = AVAudioEngine()
    let playerNode = AVAudioPlayerNode()
    
    func setup() throws {
        engine.attach(playerNode)
        engine.connect(playerNode, 
                      to: engine.mainMixerNode, 
                      format: nil)
        try engine.start()
    }
}
```

### ASIO (Windows)

ASIO provides low-latency audio on Windows:

```cpp
class AsioDriver {
public:
    virtual ASIOError init(void* sysRef) = 0;
    virtual ASIOError getChannels(long* numInputs, long* numOutputs) = 0;
    virtual ASIOError getSampleRate(ASIOSampleRate* rate) = 0;
    virtual ASIOError createBuffers(
        ASIOBufferInfo* bufferInfos,
        long numChannels,
        long bufferSize,
        ASIOCallbacks* callbacks
    ) = 0;
    virtual ASIOError start() = 0;
    virtual ASIOError stop() = 0;
};
```

### JACK Audio

JACK provides professional audio routing:

```cpp
#include <jack/jack.h>

class JackClient {
    jack_client_t* client;
    jack_port_t* outputPort;
    
public:
    bool connect() {
        client = jack_client_open("PentaCore", 
                                  JackNoStartServer, 
                                  nullptr);
        if (!client) return false;
        
        outputPort = jack_port_register(
            client,
            "output",
            JACK_DEFAULT_AUDIO_TYPE,
            JackPortIsOutput,
            0
        );
        
        jack_set_process_callback(client, processCallback, this);
        jack_activate(client);
        return true;
    }
    
    static int processCallback(jack_nframes_t nframes, void* arg) {
        auto* self = static_cast<JackClient*>(arg);
        float* buffer = static_cast<float*>(
            jack_port_get_buffer(self->outputPort, nframes)
        );
        // Process audio here
        return 0;
    }
};
```

## Buffer Size and Latency

### Understanding Buffer Sizes

Buffer size directly affects latency:

```
Latency (ms) = (Buffer Size / Sample Rate) × 1000

Examples at 48kHz:
- 64 samples   =  1.33 ms
- 128 samples  =  2.67 ms
- 256 samples  =  5.33 ms
- 512 samples  = 10.67 ms
- 1024 samples = 21.33 ms
```

### Round-Trip Latency

Total latency includes:
- Input buffer latency
- Driver/USB overhead
- Processing time
- Output buffer latency
- D/A converter delay

## MIDI Interfaces

### MIDI Protocol

- **MIDI 1.0**: Original protocol (1983)
  - 31.25 kbaud serial
  - 7-bit values (0-127)
  - 16 channels per port

- **MIDI 2.0**: Modern protocol (2020)
  - Higher resolution (32-bit values)
  - Per-note articulation
  - Property exchange
  - Backward compatible

### MIDI Implementation

```cpp
struct MidiMessage {
    uint8_t status;
    uint8_t data1;
    uint8_t data2;
    
    bool isNoteOn() const {
        return (status & 0xF0) == 0x90 && data2 > 0;
    }
    
    bool isNoteOff() const {
        return (status & 0xF0) == 0x80 || 
               ((status & 0xF0) == 0x90 && data2 == 0);
    }
    
    int channel() const {
        return status & 0x0F;
    }
    
    int noteNumber() const {
        return data1;
    }
    
    int velocity() const {
        return data2;
    }
};
```

## Best Practices

1. **Use appropriate buffer sizes**: Balance latency vs. CPU load
2. **Avoid USB hubs** for audio interfaces when possible
3. **Dedicate USB controllers** for audio devices
4. **Update drivers regularly** for stability and performance
5. **Monitor CPU usage** to prevent audio dropouts

## Resources

- [Core Audio Documentation](https://developer.apple.com/documentation/coreaudio)
- [ASIO SDK](https://www.steinberg.net/developers/)
- [JACK Audio](https://jackaudio.org/)
- [MIDI Association](https://www.midi.org/)
