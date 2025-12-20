# Mobile Frameworks Evaluation

> Technical evaluation of native mobile frameworks for iDAW mobile app.

## Executive Summary

| Framework | Platform | Language | Audio Support | Recommendation |
|-----------|----------|----------|---------------|----------------|
| React Native | iOS + Android | JavaScript | ⚠️ Limited | Secondary option |
| Flutter | iOS + Android | Dart | ⚠️ Limited | Primary choice |
| Swift/SwiftUI | iOS only | Swift | ✅ Excellent | iOS native |
| Kotlin | Android only | Kotlin | ✅ Excellent | Android native |

**Recommendation:** Flutter for cross-platform MVP, native (Swift/Kotlin) for audio-intensive features.

## Framework Analysis

### 1. Flutter

**Pros:**
- ✅ Single codebase for iOS and Android
- ✅ Excellent UI performance (Skia rendering)
- ✅ Hot reload for fast development
- ✅ Growing plugin ecosystem
- ✅ Dart language is easy to learn

**Cons:**
- ⚠️ Audio plugin ecosystem still maturing
- ⚠️ Platform-specific code needed for MIDI/Audio
- ⚠️ Larger app size (~10MB minimum)

**Audio Plugins:**
- `audioplayers` - Basic audio playback
- `flutter_midi` - MIDI playback
- `just_audio` - Advanced audio features
- `audio_service` - Background audio

**Suitability:** 7/10 for iDAW

### 2. React Native

**Pros:**
- ✅ Large developer community
- ✅ JavaScript/TypeScript familiar
- ✅ Code sharing with web app
- ✅ Many third-party libraries

**Cons:**
- ❌ Performance overhead (JS bridge)
- ⚠️ Audio latency issues on Android
- ⚠️ Native modules needed for MIDI
- ⚠️ Debugging can be complex

**Audio Libraries:**
- `react-native-audio-api` - Web Audio API
- `react-native-track-player` - Background audio
- `react-native-midi` - MIDI support

**Suitability:** 6/10 for iDAW

### 3. Native iOS (Swift/SwiftUI)

**Pros:**
- ✅ Best audio performance (Core Audio, AudioKit)
- ✅ Full Audio Unit support
- ✅ Perfect platform integration
- ✅ Inter-app audio support

**Cons:**
- ❌ iOS only
- ⚠️ Separate Android codebase needed
- ⚠️ Slower cross-platform development

**Audio Frameworks:**
- Core Audio
- AVFoundation
- AudioKit
- MIDI Services

**Suitability:** 9/10 for iDAW (iOS)

### 4. Native Android (Kotlin)

**Pros:**
- ✅ Good audio with Oboe library
- ✅ AAudio for low latency
- ✅ Perfect platform integration

**Cons:**
- ❌ Android only
- ⚠️ Audio latency varies by device
- ⚠️ Fragmentation issues

**Audio Frameworks:**
- Oboe (low-latency audio)
- AAudio API
- OpenSL ES
- Android MIDI API

**Suitability:** 8/10 for iDAW (Android)

## Recommended Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Flutter UI Layer                   │
│  (Shared UI for both platforms)                      │
├─────────────────────────────────────────────────────┤
│                   Platform Channels                  │
├──────────────────────┬──────────────────────────────┤
│      iOS Native      │        Android Native         │
│   (Swift/Obj-C)      │        (Kotlin/Java)          │
│                      │                               │
│   - Core Audio       │   - Oboe                      │
│   - Audio Unit       │   - AAudio                    │
│   - MIDI Services    │   - Android MIDI              │
└──────────────────────┴──────────────────────────────┘
```

## Feature Mapping

| Feature | Flutter | Native | Priority |
|---------|---------|--------|----------|
| Intent Editor | Flutter | - | P0 |
| Harmony View | Flutter | - | P0 |
| MIDI Preview | Flutter + Native | Native audio | P1 |
| Audio Recording | Native only | Core Audio/Oboe | P2 |
| Audio Unit Host | Native only | iOS only | P3 |
| Background Audio | Flutter + Native | Platform-specific | P1 |

## Development Phases

### Phase 1: Flutter MVP
- Basic intent editor
- Harmony visualization
- Cloud sync with web app
- No audio features

### Phase 2: Audio Integration
- Add platform channels for audio
- Implement native MIDI playback
- Basic audio preview

### Phase 3: Full Audio
- Real-time audio processing
- Audio Unit support (iOS)
- Low-latency MIDI (Oboe/Android)

### Phase 4: Platform Polish
- Platform-specific UI refinements
- App Store optimization
- Haptic feedback
- Widget support

## Development Setup

### Flutter Setup
```bash
# Install Flutter
git clone https://github.com/flutter/flutter.git
export PATH="$PATH:`pwd`/flutter/bin"

# Create iDAW mobile project
flutter create --org dev.idaw idaw_mobile
cd idaw_mobile

# Add dependencies
flutter pub add http provider audioplayers
```

### Platform Channel Example
```dart
// lib/audio_bridge.dart
import 'package:flutter/services.dart';

class AudioBridge {
  static const platform = MethodChannel('dev.idaw/audio');

  static Future<void> playMidi(List<int> midiData) async {
    await platform.invokeMethod('playMidi', {'data': midiData});
  }

  static Future<void> stopPlayback() async {
    await platform.invokeMethod('stopPlayback');
  }
}
```

```swift
// ios/Runner/AudioBridge.swift
import Flutter
import AudioKit

class AudioBridge: NSObject, FlutterPlugin {
    static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "dev.idaw/audio",
            binaryMessenger: registrar.messenger()
        )
        registrar.addMethodCallDelegate(AudioBridge(), channel: channel)
    }

    func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "playMidi":
            playMidi(call: call, result: result)
        case "stopPlayback":
            stopPlayback(result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
}
```

## Conclusion

**For iDAW Mobile:**

1. **Start with Flutter** for cross-platform UI
2. **Use platform channels** for audio features
3. **Consider native Audio Units** for iOS plugin hosting
4. **Evaluate performance** before committing to full native

**Timeline Estimate:**
- Flutter MVP: 2-3 months
- Audio integration: 1-2 months
- Full feature parity: 3-4 months

---

*"The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"*
