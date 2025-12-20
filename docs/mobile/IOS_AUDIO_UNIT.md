# iOS Audio Unit Plugin Development

> Building iDAW Penta Core as an iOS Audio Unit (AUv3) plugin.

## Overview

iOS Audio Units (AUv3) allow Penta Core to run inside iOS DAW apps like GarageBand, AUM, Cubasis, and BeatMaker.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    iOS Host App (e.g., AUM)                  │
├─────────────────────────────────────────────────────────────┤
│                     Audio Unit Host API                      │
├─────────────────────────────────────────────────────────────┤
│               Penta Core AUv3 Extension                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Swift UI Wrapper    │    C++ Core (Penta-Core)        │ │
│  │  - Parameter Views   │    - HarmonyEngine              │ │
│  │  - Chord Display     │    - GrooveEngine               │ │
│  │  - Settings Panel    │    - DiagnosticsEngine          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
iDAW-iOS/
├── PentaCore-AU/
│   ├── PentaCoreAU.xcodeproj
│   ├── Shared/
│   │   ├── PentaCoreAudioUnit.swift       # Main AU class
│   │   ├── PentaCoreDSPKernel.h          # C++ DSP wrapper
│   │   ├── PentaCoreDSPKernel.mm         # Obj-C++ bridge
│   │   └── Parameters.swift              # AU parameters
│   ├── AUv3Extension/
│   │   ├── Info.plist                    # Extension metadata
│   │   ├── PentaCoreExtension.swift      # Extension entry
│   │   └── AudioUnitFactory.swift        # Factory class
│   ├── HostApp/
│   │   ├── ContentView.swift             # Demo host UI
│   │   └── AudioEngine.swift             # AVAudioEngine setup
│   └── Resources/
│       └── Assets.xcassets               # Icons and images
├── penta-core-ios/                       # C++ library for iOS
│   ├── include/
│   └── src/
└── scripts/
    └── build-ios.sh                      # Build script
```

## Audio Unit Implementation

### Main Audio Unit Class

```swift
// PentaCoreAudioUnit.swift
import AudioToolbox
import AVFoundation
import CoreAudioKit

public class PentaCoreAudioUnit: AUAudioUnit {

    // MARK: - Properties

    private var _inputBusArray: AUAudioUnitBusArray!
    private var _outputBusArray: AUAudioUnitBusArray!
    private var dspKernel: PentaCoreDSPKernelWrapper!

    // Parameters
    private var harmonyModeParam: AUParameter!
    private var grooveIntensityParam: AUParameter!
    private var swingAmountParam: AUParameter!

    // MARK: - Initialization

    public override init(
        componentDescription: AudioComponentDescription,
        options: AudioComponentInstantiationOptions = []
    ) throws {
        try super.init(componentDescription: componentDescription, options: options)

        // Initialize DSP kernel
        dspKernel = PentaCoreDSPKernelWrapper(sampleRate: 44100.0)

        // Create parameter tree
        setupParameters()

        // Setup buses
        setupBuses()
    }

    // MARK: - Parameter Setup

    private func setupParameters() {
        let harmonyMode = AUParameterTree.createParameter(
            withIdentifier: "harmonyMode",
            name: "Harmony Mode",
            address: 0,
            min: 0, max: 4, unit: .indexed,
            unitName: nil,
            flags: [.flag_IsReadable, .flag_IsWritable],
            valueStrings: ["Auto", "Major", "Minor", "Modal", "Custom"],
            dependentParameters: nil
        )

        let grooveIntensity = AUParameterTree.createParameter(
            withIdentifier: "grooveIntensity",
            name: "Groove Intensity",
            address: 1,
            min: 0, max: 1, unit: .generic,
            unitName: nil,
            flags: [.flag_IsReadable, .flag_IsWritable],
            valueStrings: nil,
            dependentParameters: nil
        )

        let swingAmount = AUParameterTree.createParameter(
            withIdentifier: "swingAmount",
            name: "Swing Amount",
            address: 2,
            min: 0, max: 1, unit: .percent,
            unitName: nil,
            flags: [.flag_IsReadable, .flag_IsWritable],
            valueStrings: nil,
            dependentParameters: nil
        )

        parameterTree = AUParameterTree.createTree(withChildren: [
            harmonyMode,
            grooveIntensity,
            swingAmount
        ])

        harmonyModeParam = harmonyMode
        grooveIntensityParam = grooveIntensity
        swingAmountParam = swingAmount

        // Parameter observer
        parameterTree?.implementorValueObserver = { [weak self] param, value in
            self?.dspKernel.setParameter(param.address, value: value)
        }

        parameterTree?.implementorValueProvider = { [weak self] param in
            return self?.dspKernel.getParameter(param.address) ?? 0
        }
    }

    // MARK: - Bus Setup

    private func setupBuses() {
        let format = AVAudioFormat(
            standardFormatWithSampleRate: 44100.0,
            channels: 2
        )!

        let inputBus = try! AUAudioUnitBus(format: format)
        let outputBus = try! AUAudioUnitBus(format: format)

        _inputBusArray = AUAudioUnitBusArray(
            audioUnit: self,
            busType: .input,
            busses: [inputBus]
        )

        _outputBusArray = AUAudioUnitBusArray(
            audioUnit: self,
            busType: .output,
            busses: [outputBus]
        )
    }

    // MARK: - AUAudioUnit Overrides

    public override var inputBusses: AUAudioUnitBusArray {
        return _inputBusArray
    }

    public override var outputBusses: AUAudioUnitBusArray {
        return _outputBusArray
    }

    public override func allocateRenderResources() throws {
        try super.allocateRenderResources()

        let sampleRate = outputBusses[0].format.sampleRate
        let maxFrames = maximumFramesToRender

        dspKernel.allocateResources(sampleRate: sampleRate, maxFrames: maxFrames)
    }

    public override func deallocateRenderResources() {
        dspKernel.deallocateResources()
        super.deallocateRenderResources()
    }

    public override var internalRenderBlock: AUInternalRenderBlock {
        let kernel = dspKernel!

        return { [weak self] actionFlags, timestamp, frameCount, outputBusNumber,
                  outputData, realtimeEventListHead, pullInputBlock in

            guard let self = self else { return kAudioUnitErr_InvalidParameter }

            // Pull input
            var pullFlags = AudioUnitRenderActionFlags(rawValue: 0)
            let inputBuffer = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
            defer { inputBuffer.deallocate() }

            inputBuffer.pointee.mNumberBuffers = outputData.pointee.mNumberBuffers

            let err = pullInputBlock?(&pullFlags, timestamp, frameCount, 0, inputBuffer)
            if let err = err, err != noErr {
                return err
            }

            // Process MIDI events
            var event = realtimeEventListHead?.pointee
            while event != nil {
                if event!.head.eventType == .MIDI {
                    kernel.handleMIDI(
                        event!.MIDI.data.0,
                        event!.MIDI.data.1,
                        event!.MIDI.data.2
                    )
                }
                event = event!.head.next?.pointee
            }

            // Process audio
            kernel.process(
                inputBuffer: inputBuffer,
                outputBuffer: outputData,
                frameCount: frameCount
            )

            return noErr
        }
    }
}
```

### C++ DSP Kernel Wrapper

```objc
// PentaCoreDSPKernel.h
#pragma once

#import <AudioToolbox/AudioToolbox.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PentaCoreDSPKernel* PentaCoreDSPKernelRef;

PentaCoreDSPKernelRef PentaCoreDSPKernel_Create(double sampleRate);
void PentaCoreDSPKernel_Destroy(PentaCoreDSPKernelRef kernel);

void PentaCoreDSPKernel_AllocateResources(
    PentaCoreDSPKernelRef kernel,
    double sampleRate,
    uint32_t maxFrames
);

void PentaCoreDSPKernel_DeallocateResources(PentaCoreDSPKernelRef kernel);

void PentaCoreDSPKernel_SetParameter(
    PentaCoreDSPKernelRef kernel,
    uint64_t address,
    float value
);

float PentaCoreDSPKernel_GetParameter(
    PentaCoreDSPKernelRef kernel,
    uint64_t address
);

void PentaCoreDSPKernel_HandleMIDI(
    PentaCoreDSPKernelRef kernel,
    uint8_t status,
    uint8_t data1,
    uint8_t data2
);

void PentaCoreDSPKernel_Process(
    PentaCoreDSPKernelRef kernel,
    const AudioBufferList* input,
    AudioBufferList* output,
    uint32_t frameCount
);

#ifdef __cplusplus
}
#endif
```

```objc
// PentaCoreDSPKernel.mm
#import "PentaCoreDSPKernel.h"
#include "penta/PentaCore.h"
#include <vector>

struct PentaCoreDSPKernel {
    std::unique_ptr<penta::PentaCore> core;
    double sampleRate;
    uint32_t maxFrames;
    std::vector<float> inputBuffer;
    std::vector<float> outputBuffer;
    std::vector<penta::MIDINote> activeNotes;

    PentaCoreDSPKernel(double sr) : sampleRate(sr), maxFrames(0) {
        core = std::make_unique<penta::PentaCore>(sr);
    }
};

extern "C" {

PentaCoreDSPKernelRef PentaCoreDSPKernel_Create(double sampleRate) {
    return new PentaCoreDSPKernel(sampleRate);
}

void PentaCoreDSPKernel_Destroy(PentaCoreDSPKernelRef kernel) {
    delete kernel;
}

void PentaCoreDSPKernel_AllocateResources(
    PentaCoreDSPKernelRef kernel,
    double sampleRate,
    uint32_t maxFrames
) {
    kernel->sampleRate = sampleRate;
    kernel->maxFrames = maxFrames;
    kernel->inputBuffer.resize(maxFrames * 2);
    kernel->outputBuffer.resize(maxFrames * 2);
    kernel->core->setSampleRate(sampleRate);
}

void PentaCoreDSPKernel_DeallocateResources(PentaCoreDSPKernelRef kernel) {
    kernel->inputBuffer.clear();
    kernel->outputBuffer.clear();
}

void PentaCoreDSPKernel_SetParameter(
    PentaCoreDSPKernelRef kernel,
    uint64_t address,
    float value
) {
    switch (address) {
        case 0: // Harmony Mode
            kernel->core->setHarmonyMode(static_cast<int>(value));
            break;
        case 1: // Groove Intensity
            kernel->core->setGrooveIntensity(value);
            break;
        case 2: // Swing Amount
            kernel->core->setSwingAmount(value);
            break;
    }
}

float PentaCoreDSPKernel_GetParameter(
    PentaCoreDSPKernelRef kernel,
    uint64_t address
) {
    switch (address) {
        case 0: return static_cast<float>(kernel->core->getHarmonyMode());
        case 1: return kernel->core->getGrooveIntensity();
        case 2: return kernel->core->getSwingAmount();
        default: return 0.0f;
    }
}

void PentaCoreDSPKernel_HandleMIDI(
    PentaCoreDSPKernelRef kernel,
    uint8_t status,
    uint8_t data1,
    uint8_t data2
) {
    uint8_t messageType = status & 0xF0;

    if (messageType == 0x90 && data2 > 0) {
        // Note On
        penta::MIDINote note{data1, data2};
        kernel->activeNotes.push_back(note);
        kernel->core->noteOn(data1, data2);
    } else if (messageType == 0x80 || (messageType == 0x90 && data2 == 0)) {
        // Note Off
        kernel->activeNotes.erase(
            std::remove_if(kernel->activeNotes.begin(), kernel->activeNotes.end(),
                [data1](const penta::MIDINote& n) { return n.pitch == data1; }),
            kernel->activeNotes.end()
        );
        kernel->core->noteOff(data1);
    }
}

void PentaCoreDSPKernel_Process(
    PentaCoreDSPKernelRef kernel,
    const AudioBufferList* input,
    AudioBufferList* output,
    uint32_t frameCount
) {
    // Convert interleaved to planar
    const float* inL = static_cast<const float*>(input->mBuffers[0].mData);
    const float* inR = input->mNumberBuffers > 1
        ? static_cast<const float*>(input->mBuffers[1].mData)
        : inL;

    float* outL = static_cast<float*>(output->mBuffers[0].mData);
    float* outR = output->mNumberBuffers > 1
        ? static_cast<float*>(output->mBuffers[1].mData)
        : outL;

    // Process through Penta Core
    kernel->core->process(inL, inR, outL, outR, frameCount);
}

} // extern "C"
```

### Swift Wrapper

```swift
// PentaCoreDSPKernelWrapper.swift
import Foundation

class PentaCoreDSPKernelWrapper {
    private var kernel: PentaCoreDSPKernelRef

    init(sampleRate: Double) {
        kernel = PentaCoreDSPKernel_Create(sampleRate)
    }

    deinit {
        PentaCoreDSPKernel_Destroy(kernel)
    }

    func allocateResources(sampleRate: Double, maxFrames: UInt32) {
        PentaCoreDSPKernel_AllocateResources(kernel, sampleRate, maxFrames)
    }

    func deallocateResources() {
        PentaCoreDSPKernel_DeallocateResources(kernel)
    }

    func setParameter(_ address: UInt64, value: Float) {
        PentaCoreDSPKernel_SetParameter(kernel, address, value)
    }

    func getParameter(_ address: UInt64) -> Float {
        return PentaCoreDSPKernel_GetParameter(kernel, address)
    }

    func handleMIDI(_ status: UInt8, _ data1: UInt8, _ data2: UInt8) {
        PentaCoreDSPKernel_HandleMIDI(kernel, status, data1, data2)
    }

    func process(
        inputBuffer: UnsafeMutablePointer<AudioBufferList>,
        outputBuffer: UnsafeMutablePointer<AudioBufferList>,
        frameCount: UInt32
    ) {
        PentaCoreDSPKernel_Process(kernel, inputBuffer, outputBuffer, frameCount)
    }
}
```

## Extension Info.plist

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Penta Core</string>
    <key>CFBundleExecutable</key>
    <string>$(EXECUTABLE_NAME)</string>
    <key>CFBundleIdentifier</key>
    <string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>
    <key>CFBundlePackageType</key>
    <string>XPC!</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>NSExtension</key>
    <dict>
        <key>NSExtensionAttributes</key>
        <dict>
            <key>AudioComponents</key>
            <array>
                <dict>
                    <key>description</key>
                    <string>Penta Core - AI-Powered Harmony Engine</string>
                    <key>manufacturer</key>
                    <string>Pent</string>
                    <key>name</key>
                    <string>Penta Core</string>
                    <key>subtype</key>
                    <string>Pcor</string>
                    <key>tags</key>
                    <array>
                        <string>Effects</string>
                        <string>MIDI</string>
                    </array>
                    <key>type</key>
                    <string>aumf</string>
                    <key>version</key>
                    <integer>1</integer>
                    <key>sandboxSafe</key>
                    <true/>
                </dict>
            </array>
        </dict>
        <key>NSExtensionPointIdentifier</key>
        <string>com.apple.AudioUnit-UI</string>
        <key>NSExtensionPrincipalClass</key>
        <string>$(PRODUCT_MODULE_NAME).AudioUnitFactory</string>
    </dict>
</dict>
</plist>
```

## Build Configuration

### CMakeLists.txt for iOS

```cmake
# iDAW-iOS/penta-core-ios/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(penta_core_ios)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_OSX_SYSROOT iphoneos)

# iOS-specific compile options
add_compile_options(
    -fembed-bitcode
    -DPENTA_IOS=1
)

# Source files
set(SOURCES
    ../../../src_penta-core/harmony/HarmonyEngine.cpp
    ../../../src_penta-core/harmony/ChordAnalyzer.cpp
    ../../../src_penta-core/harmony/ScaleDetector.cpp
    ../../../src_penta-core/groove/GrooveEngine.cpp
    ../../../src_penta-core/groove/OnsetDetector.cpp
    ../../../src_penta-core/groove/TempoEstimator.cpp
    ../../../src_penta-core/diagnostics/DiagnosticsEngine.cpp
    ../../../src_penta-core/PentaCore.cpp
)

# Headers
include_directories(
    ../../../include
    ../../../external
)

# Static library for iOS
add_library(penta_core_ios STATIC ${SOURCES})

# No SIMD on iOS ARM (use NEON instead if needed)
target_compile_definitions(penta_core_ios PRIVATE
    PENTA_NO_AVX=1
)

# Install
install(TARGETS penta_core_ios
    ARCHIVE DESTINATION lib
)

install(DIRECTORY ../../../include/penta
    DESTINATION include
)
```

### Build Script

```bash
#!/bin/bash
# scripts/build-ios.sh

set -e

# Configuration
BUILD_DIR="build-ios"
INSTALL_DIR="install-ios"

echo "Building Penta Core for iOS..."

# Build C++ library
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ../penta-core-ios \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="../$INSTALL_DIR" \
    -G "Ninja"

ninja
ninja install

cd ..

echo "Building iOS App and Extension..."

# Build Xcode project
xcodebuild \
    -project PentaCore-AU/PentaCoreAU.xcodeproj \
    -scheme "PentaCore Host" \
    -configuration Release \
    -destination "generic/platform=iOS" \
    -derivedDataPath build-xcode \
    CODE_SIGN_IDENTITY="" \
    CODE_SIGNING_REQUIRED=NO

echo "Build complete!"
echo "Output: build-xcode/Build/Products/Release-iphoneos/"
```

## Testing

### AU Validation

```bash
# Test AU validation on macOS (after building for Mac Catalyst)
auval -v aumf Pcor Pent
```

### Test Host App

```swift
// HostApp/ContentView.swift
import SwiftUI
import AVFoundation
import AudioToolbox

struct ContentView: View {
    @State private var isLoaded = false
    @State private var harmonyMode: Float = 0
    @State private var grooveIntensity: Float = 0.5

    var body: some View {
        VStack(spacing: 20) {
            Text("Penta Core Test Host")
                .font(.title)

            if isLoaded {
                VStack {
                    Text("Harmony Mode: \(Int(harmonyMode))")
                    Slider(value: $harmonyMode, in: 0...4, step: 1)

                    Text("Groove Intensity: \(grooveIntensity, specifier: "%.2f")")
                    Slider(value: $grooveIntensity, in: 0...1)
                }
                .padding()
            } else {
                Button("Load Audio Unit") {
                    loadAudioUnit()
                }
            }
        }
        .padding()
    }

    func loadAudioUnit() {
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_MusicEffect,
            componentSubType: FourCharCode("Pcor"),
            componentManufacturer: FourCharCode("Pent"),
            componentFlags: 0,
            componentFlagsMask: 0
        )

        AUAudioUnit.instantiate(with: desc, options: []) { audioUnit, error in
            if let error = error {
                print("Failed to load: \(error)")
                return
            }

            DispatchQueue.main.async {
                isLoaded = true
            }
        }
    }
}

extension FourCharCode {
    init(_ string: String) {
        let bytes = string.utf8
        precondition(bytes.count == 4)
        var value: UInt32 = 0
        for byte in bytes {
            value = value << 8 + UInt32(byte)
        }
        self.init(value)
    }
}
```

## App Store Submission

### Requirements

1. **App Groups** - For sharing data between host and extension
2. **Inter-App Audio** - For audio routing between apps
3. **Background Audio** - For continuous playback

### Entitlements

```xml
<!-- PentaCore.entitlements -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.application-groups</key>
    <array>
        <string>group.dev.idaw.pentacore</string>
    </array>
    <key>inter-app-audio</key>
    <true/>
    <key>com.apple.developer.coreaudio.allow-by-default</key>
    <true/>
</dict>
</plist>
```

## Compatible iOS DAWs

| App | Support Level | Notes |
|-----|---------------|-------|
| GarageBand | Full | Built-in AUv3 host |
| AUM | Full | Professional AU host |
| Cubasis 3 | Full | Steinberg DAW |
| BeatMaker 3 | Full | Sample-based DAW |
| Audiobus 3 | Full | Audio routing app |
| ApeMatrix | Full | Modular routing |
| Drambo | Full | Modular groovebox |

## Performance Considerations

### Memory Limits

- iOS extensions have ~120MB memory limit
- Pre-allocate buffers at initialization
- Avoid dynamic allocation in audio callback

### CPU Optimization

- Use NEON SIMD instead of AVX2 on ARM
- Keep processing under 50% CPU for stability
- Use Instruments to profile

### Latency

- Minimum buffer: 128 samples (~2.9ms at 44.1kHz)
- Target latency: <10ms for real-time play
- Use `kAudioUnitProperty_MaximumFramesPerSlice`

---

*"The audience doesn't hear 'Audio Unit.' They hear 'that pad sound from GarageBand.'"*
