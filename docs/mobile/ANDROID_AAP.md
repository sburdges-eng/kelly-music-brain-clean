# Android AAP Plugin Development

> Building iDAW Penta Core as an Android Audio Plugin (AAP).

## Overview

Android Audio Plugin (AAP) is an open audio plugin format for Android, similar to VST/AU on desktop. It allows Penta Core to run inside Android DAW apps like Caustic, FL Studio Mobile, and AAP-compatible hosts.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Android Host App (e.g., AAP Host)           │
├─────────────────────────────────────────────────────────────┤
│                      AAP Binder IPC                         │
├─────────────────────────────────────────────────────────────┤
│                    Penta Core AAP Plugin                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Kotlin/Java Wrapper  │    C++ Core (Oboe + Penta)     ││
│  │  - Service binding    │    - HarmonyEngine             ││
│  │  - Parameter mapping  │    - GrooveEngine              ││
│  │  - UI Activity        │    - Low-latency audio (Oboe)  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
iDAW-Android/
├── penta-core-aap/
│   ├── build.gradle.kts
│   ├── src/
│   │   ├── main/
│   │   │   ├── AndroidManifest.xml
│   │   │   ├── kotlin/
│   │   │   │   └── dev/idaw/pentacore/
│   │   │   │       ├── PentaCorePlugin.kt
│   │   │   │       ├── PentaCoreService.kt
│   │   │   │       ├── PentaCoreUI.kt
│   │   │   │       └── Parameters.kt
│   │   │   ├── cpp/
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── penta_core_aap.cpp
│   │   │   │   ├── jni_bridge.cpp
│   │   │   │   └── oboe_callback.cpp
│   │   │   └── res/
│   │   │       ├── xml/
│   │   │       │   └── aap_metadata.xml
│   │   │       └── layout/
│   │   │           └── plugin_ui.xml
│   │   └── androidTest/
│   └── aap-metadata.xml
├── penta-core-native/           # C++ library for Android
│   ├── CMakeLists.txt
│   └── src/
└── scripts/
    └── build-android.sh
```

## AAP Plugin Implementation

### AndroidManifest.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="dev.idaw.pentacore">

    <uses-feature android:name="android.software.midi" android:required="false"/>
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>

    <application
        android:name=".PentaCoreApplication"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/Theme.PentaCore">

        <!-- AAP Plugin Service -->
        <service
            android:name=".PentaCoreService"
            android:exported="true"
            android:label="Penta Core">
            <intent-filter>
                <action android:name="org.androidaudioplugin.AudioPluginService" />
            </intent-filter>
            <meta-data
                android:name="org.androidaudioplugin.AudioPluginService#Plugins"
                android:resource="@xml/aap_metadata" />
        </service>

        <!-- Plugin UI Activity -->
        <activity
            android:name=".PentaCoreUI"
            android:exported="true"
            android:label="Penta Core Settings">
            <intent-filter>
                <action android:name="org.androidaudioplugin.AudioPluginView" />
            </intent-filter>
        </activity>

    </application>

</manifest>
```

### AAP Metadata (aap_metadata.xml)

```xml
<?xml version="1.0" encoding="utf-8"?>
<plugins xmlns="urn:org.androidaudioplugin.core"
    xmlns:pp="urn:org.androidaudioplugin.port">

    <plugin
        name="Penta Core"
        category="Effect"
        author="iDAW"
        manufacturer="dev.idaw"
        unique-id="dev.idaw.pentacore"
        version="1.0.0"
        backend="Native"
        product="https://idaw.dev">

        <!-- Ports Definition -->
        <ports>
            <!-- Audio Input -->
            <port direction="input" content="audio" name="Audio In L"/>
            <port direction="input" content="audio" name="Audio In R"/>

            <!-- Audio Output -->
            <port direction="output" content="audio" name="Audio Out L"/>
            <port direction="output" content="audio" name="Audio Out R"/>

            <!-- MIDI Input -->
            <port direction="input" content="midi2" name="MIDI In"/>

            <!-- Parameters -->
            <port direction="input" content="other"
                pp:name="Harmony Mode"
                pp:default="0" pp:minimum="0" pp:maximum="4"
                pp:propertyType="integer"/>

            <port direction="input" content="other"
                pp:name="Groove Intensity"
                pp:default="0.5" pp:minimum="0.0" pp:maximum="1.0"/>

            <port direction="input" content="other"
                pp:name="Swing Amount"
                pp:default="0.0" pp:minimum="0.0" pp:maximum="1.0"/>

            <port direction="input" content="other"
                pp:name="Scale Root"
                pp:default="0" pp:minimum="0" pp:maximum="11"
                pp:propertyType="integer"/>

            <port direction="input" content="other"
                pp:name="Scale Type"
                pp:default="0" pp:minimum="0" pp:maximum="7"
                pp:propertyType="integer"/>
        </ports>

    </plugin>

</plugins>
```

### AAP Service Implementation

```kotlin
// PentaCoreService.kt
package dev.idaw.pentacore

import android.content.Context
import org.androidaudioplugin.AudioPluginService

class PentaCoreService : AudioPluginService() {

    companion object {
        init {
            System.loadLibrary("penta_core_aap")
        }

        // Plugin factory registration
        @JvmStatic
        fun registerPluginFactory(context: Context) {
            // Native registration handled in JNI
            nativeRegisterFactory()
        }

        @JvmStatic
        private external fun nativeRegisterFactory()
    }

    override fun onCreate() {
        super.onCreate()
        registerPluginFactory(this)
    }
}
```

### Plugin Implementation (Kotlin)

```kotlin
// PentaCorePlugin.kt
package dev.idaw.pentacore

import org.androidaudioplugin.PluginInformation

class PentaCorePlugin {

    companion object {
        // Port indices matching aap_metadata.xml
        const val PORT_AUDIO_IN_L = 0
        const val PORT_AUDIO_IN_R = 1
        const val PORT_AUDIO_OUT_L = 2
        const val PORT_AUDIO_OUT_R = 3
        const val PORT_MIDI_IN = 4
        const val PORT_HARMONY_MODE = 5
        const val PORT_GROOVE_INTENSITY = 6
        const val PORT_SWING_AMOUNT = 7
        const val PORT_SCALE_ROOT = 8
        const val PORT_SCALE_TYPE = 9

        @JvmStatic
        val pluginInfo: PluginInformation
            get() = PluginInformation(
                packageName = "dev.idaw.pentacore",
                localName = "Penta Core",
                displayName = "Penta Core",
                version = "1.0.0",
                developerName = "iDAW",
                productUrl = "https://idaw.dev",
                category = "Effect",
                pluginId = "dev.idaw.pentacore",
                uiActivity = PentaCoreUI::class.java.name,
                isInstrument = false
            )
    }
}
```

### Native C++ Implementation

```cpp
// penta_core_aap.cpp
#include <jni.h>
#include <android/log.h>
#include <oboe/Oboe.h>
#include "aap/android-audio-plugin.h"
#include "penta/PentaCore.h"
#include <memory>
#include <vector>

#define LOG_TAG "PentaCoreAAP"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Port indices
constexpr int PORT_AUDIO_IN_L = 0;
constexpr int PORT_AUDIO_IN_R = 1;
constexpr int PORT_AUDIO_OUT_L = 2;
constexpr int PORT_AUDIO_OUT_R = 3;
constexpr int PORT_MIDI_IN = 4;
constexpr int PORT_HARMONY_MODE = 5;
constexpr int PORT_GROOVE_INTENSITY = 6;
constexpr int PORT_SWING_AMOUNT = 7;
constexpr int PORT_SCALE_ROOT = 8;
constexpr int PORT_SCALE_TYPE = 9;

struct PentaCoreInstance {
    std::unique_ptr<penta::PentaCore> core;
    double sampleRate{48000.0};
    int32_t framesPerBuffer{256};

    // Port buffers
    float* audioInL{nullptr};
    float* audioInR{nullptr};
    float* audioOutL{nullptr};
    float* audioOutR{nullptr};
    void* midiIn{nullptr};

    // Parameters
    float harmonyMode{0.0f};
    float grooveIntensity{0.5f};
    float swingAmount{0.0f};
    float scaleRoot{0.0f};
    float scaleType{0.0f};

    PentaCoreInstance(double sr) : sampleRate(sr) {
        core = std::make_unique<penta::PentaCore>(sr);
        LOGI("PentaCoreInstance created with sample rate: %f", sr);
    }
};

// AAP Plugin Implementation
extern "C" {

void* penta_core_create(int sampleRate) {
    return new PentaCoreInstance(static_cast<double>(sampleRate));
}

void penta_core_destroy(void* instance) {
    delete static_cast<PentaCoreInstance*>(instance);
}

void penta_core_prepare(void* instance, aap_buffer_t* buffer) {
    auto* inst = static_cast<PentaCoreInstance*>(instance);

    // Get buffer pointers
    inst->audioInL = static_cast<float*>(aap_buffer_get_audio_buffer(buffer, PORT_AUDIO_IN_L));
    inst->audioInR = static_cast<float*>(aap_buffer_get_audio_buffer(buffer, PORT_AUDIO_IN_R));
    inst->audioOutL = static_cast<float*>(aap_buffer_get_audio_buffer(buffer, PORT_AUDIO_OUT_L));
    inst->audioOutR = static_cast<float*>(aap_buffer_get_audio_buffer(buffer, PORT_AUDIO_OUT_R));
    inst->midiIn = aap_buffer_get_midi2_buffer(buffer, PORT_MIDI_IN);

    inst->framesPerBuffer = buffer->num_frames;

    LOGI("Prepared with %d frames per buffer", inst->framesPerBuffer);
}

void penta_core_activate(void* instance) {
    auto* inst = static_cast<PentaCoreInstance*>(instance);
    LOGI("Activated");
}

void penta_core_deactivate(void* instance) {
    auto* inst = static_cast<PentaCoreInstance*>(instance);
    LOGI("Deactivated");
}

void penta_core_process(void* instance, aap_buffer_t* buffer, int32_t frameCount) {
    auto* inst = static_cast<PentaCoreInstance*>(instance);

    // Read parameters
    float* paramHarmony = static_cast<float*>(
        aap_buffer_get_parameter_buffer(buffer, PORT_HARMONY_MODE));
    float* paramGroove = static_cast<float*>(
        aap_buffer_get_parameter_buffer(buffer, PORT_GROOVE_INTENSITY));
    float* paramSwing = static_cast<float*>(
        aap_buffer_get_parameter_buffer(buffer, PORT_SWING_AMOUNT));
    float* paramRoot = static_cast<float*>(
        aap_buffer_get_parameter_buffer(buffer, PORT_SCALE_ROOT));
    float* paramScale = static_cast<float*>(
        aap_buffer_get_parameter_buffer(buffer, PORT_SCALE_TYPE));

    // Apply parameters if changed
    if (paramHarmony && *paramHarmony != inst->harmonyMode) {
        inst->harmonyMode = *paramHarmony;
        inst->core->setHarmonyMode(static_cast<int>(inst->harmonyMode));
    }
    if (paramGroove && *paramGroove != inst->grooveIntensity) {
        inst->grooveIntensity = *paramGroove;
        inst->core->setGrooveIntensity(inst->grooveIntensity);
    }
    if (paramSwing && *paramSwing != inst->swingAmount) {
        inst->swingAmount = *paramSwing;
        inst->core->setSwingAmount(inst->swingAmount);
    }

    // Process MIDI
    if (inst->midiIn) {
        auto* midiBuffer = static_cast<aap_midi2_buffer_t*>(inst->midiIn);
        for (size_t i = 0; i < midiBuffer->count; ++i) {
            auto& event = midiBuffer->events[i];
            // Parse MIDI 2.0 messages
            uint8_t status = (event.data >> 16) & 0xFF;
            uint8_t data1 = (event.data >> 8) & 0xFF;
            uint8_t data2 = event.data & 0xFF;

            if ((status & 0xF0) == 0x90 && data2 > 0) {
                inst->core->noteOn(data1, data2);
            } else if ((status & 0xF0) == 0x80 || ((status & 0xF0) == 0x90 && data2 == 0)) {
                inst->core->noteOff(data1);
            }
        }
    }

    // Process audio
    if (inst->audioInL && inst->audioOutL) {
        inst->core->process(
            inst->audioInL,
            inst->audioInR ? inst->audioInR : inst->audioInL,
            inst->audioOutL,
            inst->audioOutR ? inst->audioOutR : inst->audioOutL,
            frameCount
        );
    }
}

aap_plugin_info_t penta_core_get_plugin_info() {
    static aap_plugin_info_t info = {
        .plugin_id = "dev.idaw.pentacore",
        .display_name = "Penta Core",
        .manufacturer_name = "iDAW",
        .version = "1.0.0",
        .category = "Effect",
        .num_ports = 10,
        .is_instrument = false
    };
    return info;
}

// Plugin Factory
aap_plugin_t* penta_core_factory_create(aap_plugin_info_t info, int sampleRate, const char* pluginId) {
    auto* plugin = new aap_plugin_t();
    plugin->instance = penta_core_create(sampleRate);
    plugin->prepare = penta_core_prepare;
    plugin->activate = penta_core_activate;
    plugin->process = penta_core_process;
    plugin->deactivate = penta_core_deactivate;
    plugin->get_state_size = [](void*) -> int32_t { return 0; };
    plugin->get_state = [](void*, aap_state_t*) {};
    plugin->set_state = [](void*, aap_state_t*) {};
    return plugin;
}

void penta_core_factory_destroy(aap_plugin_t* plugin) {
    penta_core_destroy(plugin->instance);
    delete plugin;
}

// JNI Bridge
JNIEXPORT void JNICALL
Java_dev_idaw_pentacore_PentaCoreService_nativeRegisterFactory(JNIEnv* env, jclass clazz) {
    LOGI("Registering Penta Core AAP factory");
    // Factory registration handled by AAP framework
}

} // extern "C"

} // anonymous namespace
```

### JNI Bridge for UI

```cpp
// jni_bridge.cpp
#include <jni.h>
#include "penta/PentaCore.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_dev_idaw_pentacore_PentaCoreNative_createInstance(JNIEnv* env, jclass clazz, jdouble sampleRate) {
    auto* core = new penta::PentaCore(sampleRate);
    return reinterpret_cast<jlong>(core);
}

JNIEXPORT void JNICALL
Java_dev_idaw_pentacore_PentaCoreNative_destroyInstance(JNIEnv* env, jclass clazz, jlong handle) {
    auto* core = reinterpret_cast<penta::PentaCore*>(handle);
    delete core;
}

JNIEXPORT jstring JNICALL
Java_dev_idaw_pentacore_PentaCoreNative_getDetectedChord(JNIEnv* env, jclass clazz, jlong handle) {
    auto* core = reinterpret_cast<penta::PentaCore*>(handle);
    std::string chord = core->getDetectedChord();
    return env->NewStringUTF(chord.c_str());
}

JNIEXPORT jstring JNICALL
Java_dev_idaw_pentacore_PentaCoreNative_getDetectedScale(JNIEnv* env, jclass clazz, jlong handle) {
    auto* core = reinterpret_cast<penta::PentaCore*>(handle);
    std::string scale = core->getDetectedScale();
    return env->NewStringUTF(scale.c_str());
}

JNIEXPORT jfloat JNICALL
Java_dev_idaw_pentacore_PentaCoreNative_getDetectedTempo(JNIEnv* env, jclass clazz, jlong handle) {
    auto* core = reinterpret_cast<penta::PentaCore*>(handle);
    return static_cast<jfloat>(core->getDetectedTempo());
}

JNIEXPORT void JNICALL
Java_dev_idaw_pentacore_PentaCoreNative_setHarmonyMode(JNIEnv* env, jclass clazz, jlong handle, jint mode) {
    auto* core = reinterpret_cast<penta::PentaCore*>(handle);
    core->setHarmonyMode(mode);
}

JNIEXPORT void JNICALL
Java_dev_idaw_pentacore_PentaCoreNative_setGrooveIntensity(JNIEnv* env, jclass clazz, jlong handle, jfloat intensity) {
    auto* core = reinterpret_cast<penta::PentaCore*>(handle);
    core->setGrooveIntensity(intensity);
}

} // extern "C"
```

### CMakeLists.txt for Android

```cmake
# src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(penta_core_aap)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(oboe REQUIRED CONFIG)

# Penta Core sources
set(PENTA_SOURCES
    ${CMAKE_SOURCE_DIR}/../../../../src_penta-core/harmony/HarmonyEngine.cpp
    ${CMAKE_SOURCE_DIR}/../../../../src_penta-core/harmony/ChordAnalyzer.cpp
    ${CMAKE_SOURCE_DIR}/../../../../src_penta-core/harmony/ScaleDetector.cpp
    ${CMAKE_SOURCE_DIR}/../../../../src_penta-core/groove/GrooveEngine.cpp
    ${CMAKE_SOURCE_DIR}/../../../../src_penta-core/groove/OnsetDetector.cpp
    ${CMAKE_SOURCE_DIR}/../../../../src_penta-core/groove/TempoEstimator.cpp
    ${CMAKE_SOURCE_DIR}/../../../../src_penta-core/diagnostics/DiagnosticsEngine.cpp
    ${CMAKE_SOURCE_DIR}/../../../../src_penta-core/PentaCore.cpp
)

# AAP plugin sources
set(AAP_SOURCES
    penta_core_aap.cpp
    jni_bridge.cpp
    oboe_callback.cpp
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../../../../include
    ${CMAKE_SOURCE_DIR}/../../../../external
)

# Shared library
add_library(penta_core_aap SHARED
    ${PENTA_SOURCES}
    ${AAP_SOURCES}
)

# Android-specific definitions
target_compile_definitions(penta_core_aap PRIVATE
    PENTA_ANDROID=1
    PENTA_NO_AVX=1
)

# Link libraries
target_link_libraries(penta_core_aap
    oboe::oboe
    log
    android
)
```

### Gradle Build Configuration

```kotlin
// build.gradle.kts
plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "dev.idaw.pentacore"
    compileSdk = 34

    defaultConfig {
        applicationId = "dev.idaw.pentacore"
        minSdk = 29  // Android 10+ for low-latency audio
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-O3")
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_TOOLCHAIN=clang"
                )
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    implementation("org.androidaudioplugin:androidaudioplugin:0.7.8")
    implementation("com.google.oboe:oboe:1.8.0")

    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
}
```

### Plugin UI Activity

```kotlin
// PentaCoreUI.kt
package dev.idaw.pentacore

import android.os.Bundle
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class PentaCoreUI : AppCompatActivity() {

    private var nativeHandle: Long = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.plugin_ui)

        // Initialize native core for display
        nativeHandle = PentaCoreNative.createInstance(48000.0)

        setupUI()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (nativeHandle != 0L) {
            PentaCoreNative.destroyInstance(nativeHandle)
        }
    }

    private fun setupUI() {
        val harmonySpinner = findViewById<Spinner>(R.id.harmony_mode_spinner)
        val grooveSlider = findViewById<SeekBar>(R.id.groove_intensity_slider)
        val swingSlider = findViewById<SeekBar>(R.id.swing_slider)
        val chordDisplay = findViewById<TextView>(R.id.chord_display)
        val scaleDisplay = findViewById<TextView>(R.id.scale_display)
        val tempoDisplay = findViewById<TextView>(R.id.tempo_display)

        // Harmony mode selection
        harmonySpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                PentaCoreNative.setHarmonyMode(nativeHandle, position)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Groove intensity slider
        grooveSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val intensity = progress / 100f
                PentaCoreNative.setGrooveIntensity(nativeHandle, intensity)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Update display periodically
        updateDisplayRunnable = object : Runnable {
            override fun run() {
                chordDisplay.text = PentaCoreNative.getDetectedChord(nativeHandle)
                scaleDisplay.text = PentaCoreNative.getDetectedScale(nativeHandle)
                tempoDisplay.text = "${PentaCoreNative.getDetectedTempo(nativeHandle).toInt()} BPM"
                handler.postDelayed(this, 100)
            }
        }
        handler.post(updateDisplayRunnable)
    }

    private val handler = Handler(Looper.getMainLooper())
    private var updateDisplayRunnable: Runnable? = null

    override fun onPause() {
        super.onPause()
        updateDisplayRunnable?.let { handler.removeCallbacks(it) }
    }

    override fun onResume() {
        super.onResume()
        updateDisplayRunnable?.let { handler.post(it) }
    }
}
```

### Native Kotlin Bridge

```kotlin
// PentaCoreNative.kt
package dev.idaw.pentacore

object PentaCoreNative {
    init {
        System.loadLibrary("penta_core_aap")
    }

    external fun createInstance(sampleRate: Double): Long
    external fun destroyInstance(handle: Long)
    external fun getDetectedChord(handle: Long): String
    external fun getDetectedScale(handle: Long): String
    external fun getDetectedTempo(handle: Long): Float
    external fun setHarmonyMode(handle: Long, mode: Int)
    external fun setGrooveIntensity(handle: Long, intensity: Float)
}
```

## Build Script

```bash
#!/bin/bash
# scripts/build-android.sh

set -e

echo "Building Penta Core AAP for Android..."

# Check for Android SDK
if [ -z "$ANDROID_HOME" ]; then
    echo "Error: ANDROID_HOME not set"
    exit 1
fi

cd iDAW-Android/penta-core-aap

# Build with Gradle
./gradlew assembleRelease

echo "Build complete!"
echo "APK: app/build/outputs/apk/release/penta-core-aap-release.apk"

# List AAP metadata for verification
echo ""
echo "AAP Plugin Info:"
aapt dump xmltree app/build/outputs/apk/release/*.apk res/xml/aap_metadata.xml
```

## Testing

### Unit Testing

```kotlin
// PentaCoreTest.kt
@RunWith(AndroidJUnit4::class)
class PentaCoreTest {

    private var nativeHandle: Long = 0

    @Before
    fun setup() {
        nativeHandle = PentaCoreNative.createInstance(48000.0)
    }

    @After
    fun teardown() {
        PentaCoreNative.destroyInstance(nativeHandle)
    }

    @Test
    fun testChordDetection() {
        // Simulate note input and check chord detection
        val chord = PentaCoreNative.getDetectedChord(nativeHandle)
        assertNotNull(chord)
    }

    @Test
    fun testParameterChange() {
        PentaCoreNative.setHarmonyMode(nativeHandle, 1)
        PentaCoreNative.setGrooveIntensity(nativeHandle, 0.75f)
        // Parameters should be applied without crash
    }
}
```

### AAP Host Testing

Use the official AAP Host app to test the plugin:

1. Install AAP Host from GitHub releases
2. Install Penta Core AAP
3. Open AAP Host and load Penta Core
4. Connect MIDI keyboard
5. Verify audio pass-through and analysis

## Compatible Android DAWs

| App | AAP Support | Notes |
|-----|-------------|-------|
| AAP Host | Full | Official test host |
| Caustic 3 | Partial | Via AAP wrapper |
| FL Studio Mobile | No | Uses proprietary format |
| n-Track Studio | Partial | VST-like support |
| Audio Evolution | No | Uses internal plugins |

## Performance Considerations

### Low-Latency Audio

```kotlin
// Request low-latency mode
val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
val sampleRate = audioManager.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE)?.toInt() ?: 48000
val framesPerBuffer = audioManager.getProperty(AudioManager.PROPERTY_OUTPUT_FRAMES_PER_BUFFER)?.toInt() ?: 256
```

### Memory Management

- Android has stricter memory limits
- Pre-allocate all buffers at initialization
- Use `android:largeHeap="true"` for UI-heavy plugins

### Battery Optimization

- Register as foreground service for background audio
- Use Oboe's performance mode hints
- Disable diagnostics in power-save mode

## Troubleshooting

### Common Issues

1. **Oboe not found**: Ensure Oboe is in the prefab dependencies
2. **Plugin not detected**: Check AndroidManifest service export
3. **Audio glitches**: Reduce buffer size, check CPU governor
4. **MIDI not working**: Verify android.software.midi feature

---

*"The audience doesn't hear 'AAP plugin.' They hear 'that effect I discovered on Android.'"*
