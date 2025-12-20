# Repository Research: DAW Plugin Architecture

## TARGET DOMAIN
DAW plugin architecture and best practices for VST3/AUv3/CLAP plugin development, transport sync, host integration, and cross-platform compatibility.

## WHAT I NEED TO LEARN
- How to structure VST3/AUv3/CLAP plugins for maximum host compatibility
- Best practices for transport sync and tempo following
- Cross-platform plugin architecture patterns
- Host-specific integration requirements
- Plugin validation and testing strategies

## MY CURRENT IMPLEMENTATION
- Plugin processor: `src/plugin/plugin_processor.cpp`
- Plugin editor: `src/plugin/plugin_editor.cpp`
- Host support matrix: `docs/daw_integration/HOST_SUPPORT_MATRIX.md`
- Transport sync requirements documented but implementation needs validation

## RESEARCH TASK
You are a senior developer researching GitHub repositories to improve my codebase.

1. **Identify Top Repos:**
   - Search GitHub for repos matching: VST3 plugin architecture, AUv3 development, CLAP plugin framework
   - Filter by: stars > 100, updated in last 2 years, active community
   - Focus on: production-ready code, good documentation, similar tech stack (JUCE, C++)

2. **Extract Patterns:**
   - Code architecture (how they structure modules)
   - API design (function signatures, class hierarchies)
   - Performance optimizations
   - Error handling approaches
   - Testing strategies
   - Transport sync implementation
   - Host compatibility patterns

3. **Compare to My Code:**
   - What am I doing right?
   - What can be improved?
   - What patterns should I adopt?
   - What anti-patterns should I avoid?

4. **Actionable Recommendations:**
   - Specific code changes (with examples)
   - Libraries/dependencies to add
   - Refactoring opportunities
   - Performance improvements
   - Testing improvements

## OUTPUT FORMAT
### Top 5 Repositories

1. **JUCE Framework** (https://github.com/juce-framework/JUCE)
   - Stars: 4.5k+
   - Why relevant: Official JUCE framework with VST3/AUv3/CLAP support, actively maintained, comprehensive documentation
   - Key features: Cross-platform plugin format support, transport sync APIs, parameter management

2. **iPlug2** (https://github.com/iPlug2/iPlug2)
   - Stars: 1.2k+
   - Why relevant: Modern C++ plugin framework supporting CLAP, VST3, AU, AAX; cross-platform GUI toolkit
   - Key features: Simplified API, web/mobile support, good architecture patterns

3. **DISTRHO Plugin Framework (DPF)** (https://github.com/DISTRHO/DPF)
   - Stars: 500+
   - Why relevant: Lightweight C++ framework with CLAP support, custom UI toolkit, minimal dependencies
   - Key features: Simple architecture, good for learning patterns, CLAP-first design

4. **CPLUG** (https://github.com/Tremus/CPLUG)
   - Stars: 200+
   - Why relevant: Minimal C99 wrapper for VST3/AUv2/CLAP, easy to understand, good for cross-format patterns
   - Key features: Simple API, minimal code, good documentation

5. **IEM Plugin Suite** (https://github.com/tu-studio/IEMPluginSuite)
   - Stars: 300+
   - Why relevant: Production JUCE-based plugin suite, demonstrates real-world patterns, transport sync examples
   - Key features: Professional code quality, host compatibility patterns, testing strategies

### Key Patterns Found

#### 1. Plugin Lifecycle Management
**Pattern**: Resource initialization in `initialize()`, cleanup in `terminate()`
```cpp
// Good pattern from JUCE/iPlug2
bool initialize(const juce::String&, double sampleRate, int maxBlockSize) override {
    // Initialize resources here, not in constructor
    return true;
}

void terminate() override {
    // Cleanup resources here, not in destructor
}
```
**Why better**: VST3/CLAP require specific lifecycle management; constructors/destructors may be called at wrong times.

**My code**: Need to verify `PluginProcessor` constructor vs `prepareToPlay()` usage.

#### 2. Thread Safety Patterns
**Pattern**: Separate audio thread from UI thread, use lock-free queues
```cpp
// Common pattern: lock-free queue for parameter changes
class ParameterQueue {
    std::atomic<ParameterChange> pendingChanges;
    // Process on audio thread, update on UI thread
};
```
**Why better**: VST3 requires main thread for certain operations; audio thread must be lock-free.

**My code**: Check parameter listener callbacks - ensure thread-safe updates.

#### 3. Transport Sync Implementation
**Pattern**: Query host for transport info, cache locally, update periodically
```cpp
// Pattern from IEM Plugin Suite
void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer& midi) override {
    auto* playhead = getPlayHead();
    if (playhead) {
        juce::AudioPlayHead::CurrentPositionInfo posInfo;
        if (playhead->getCurrentPosition(posInfo)) {
            // Use posInfo.bpm, posInfo.timeInSamples, etc.
        }
    }
}
```
**Why better**: Transport info should be queried per-block, not cached indefinitely.

**My code**: Verify transport sync implementation matches this pattern.

#### 4. Latency Reporting
**Pattern**: Report latency in `getLatencySamples()`, update when processing mode changes
```cpp
int getLatencySamples() const override {
    return currentLatency; // Update based on processing mode
}
```
**Why better**: Host needs accurate latency for PDC (Plugin Delay Compensation).

**My code**: Ensure latency reporting matches HOST_SUPPORT_MATRIX.md requirements (Kelly path vs Dee path).

#### 5. Parameter State Management
**Pattern**: Use AudioProcessorValueTreeState for parameter automation, save/load state
```cpp
// JUCE pattern
juce::AudioProcessorValueTreeState parameters;
// Automatically handles automation, state saving, host sync
```
**Why better**: Handles host automation, state persistence, thread safety automatically.

**My code**: Already using AudioProcessorValueTreeState - good! Verify all parameters are properly exposed.

#### 6. Host-Specific Handling
**Pattern**: Detect host type, adjust behavior accordingly
```cpp
// Pattern from iPlug2
if (getWrapperType() == wrapperType_VST3) {
    // VST3-specific code
} else if (getWrapperType() == wrapperType_AudioUnit) {
    // AU-specific code
}
```
**Why better**: Different hosts have different requirements (e.g., Logic offline bounce, Ableton Link).

**My code**: Add host detection for Logic Pro offline bounce handling per HOST_SUPPORT_MATRIX.md.

### Recommendations for My Code

#### 1. Fix Arousal Range Inconsistency
**Issue**: Code uses 0.0-1.0, QUICK_START says -1.0 to +1.0
**Action**: Update QUICK_START.md to match code (0.0-1.0) OR update code to match QUICK_START
**Code location**: `src/plugin/plugin_processor.cpp:84-90`
```cpp
// Current (matches USER_GUIDE):
juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f)

// If changing to match QUICK_START:
juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f)
```

#### 2. Add Host Detection for Offline Bounce
**Pattern from research**: Detect offline rendering, disable cloud calls
**Action**: Add offline detection per HOST_SUPPORT_MATRIX.md
```cpp
// Add to PluginProcessor
bool isOfflineRender() const {
    auto* playhead = getPlayHead();
    if (playhead) {
        juce::AudioPlayHead::CurrentPositionInfo posInfo;
        if (playhead->getCurrentPosition(posInfo)) {
            return posInfo.isPlaying && posInfo.isRecording;
        }
    }
    return false;
}

// In processBlock or generation logic:
if (isOfflineRender() && parameters_.getRawParameterValue(ParameterID::enableCloud)->load() > 0.5f) {
    // Disable cloud calls, use local only
}
```

#### 3. Improve Transport Sync Implementation
**Pattern from research**: Query transport per-block, cache for efficiency
**Action**: Verify current implementation matches best practices
**Code location**: Check `processBlock()` implementation
```cpp
// Recommended pattern:
void processBlock(...) override {
    auto* playhead = getPlayHead();
    juce::AudioPlayHead::CurrentPositionInfo posInfo;
    if (playhead && playhead->getCurrentPosition(posInfo)) {
        // Use posInfo.bpm, posInfo.timeInSamples, posInfo.timeSigNumerator, etc.
        // Update internal tempo/position state
    }
}
```

#### 4. Add Latency Reporting Differentiation
**Pattern from research**: Report different latency for different processing modes
**Action**: Implement Kelly vs Dee path latency reporting
```cpp
int getLatencySamples() const override {
    // Kelly path: report actual latency
    if (isKellyPath()) {
        return kellyLatencySamples;
    }
    // Dee path: report zero (per HOST_SUPPORT_MATRIX.md)
    return 0;
}
```

#### 5. Add CLAP Support Investigation
**Pattern from research**: CLAP is gaining traction, especially on Linux
**Action**: Evaluate adding CLAP support using DPF or CLAP wrapper
**Consideration**: HOST_SUPPORT_MATRIX.md mentions CLAP for Bitwig/Reaper/Linux

#### 6. Improve Error Handling
**Pattern from research**: Graceful degradation, host crash prevention
**Action**: Add try-catch blocks in processBlock, validate all host callbacks
```cpp
void processBlock(...) override {
    try {
        // Processing code
    } catch (...) {
        // Log error, output silence, prevent host crash
    }
}
```

#### 7. Add Plugin Validation Testing
**Pattern from research**: Use auval (macOS), VST3 validator, CLAP lint
**Action**: Add CI validation per HOST_SUPPORT_MATRIX.md section "Validation Targets"
**Tools**:
- macOS: `auval -v aufx`
- VST3: VST3 SDK validator
- CLAP: clap-validator

#### 8. Parameter Automation Verification
**Pattern from research**: All parameters should support automation
**Action**: Verify all parameters in `createParameterLayout()` are automatable
**Current**: Using AudioProcessorValueTreeState - good! Just verify all params exposed.

#### 9. Thread Safety Audit
**Pattern from research**: Separate audio/UI threads, lock-free audio processing
**Action**: Audit parameter listeners, ensure thread-safe updates
**Code location**: `plugin_processor.cpp:42-49` - verify listener callbacks are thread-safe

#### 10. Add Host-Specific Workarounds
**Pattern from research**: Different hosts need different handling
**Action**: Add Logic Pro offline bounce detection, Ableton Link integration
**Code location**: Add host detection utility, use in generation logic

### Summary

**What I'm doing right:**
- ✅ Using AudioProcessorValueTreeState for parameters
- ✅ Using JUCE framework (industry standard)
- ✅ Proper plugin structure (processor/editor separation)

**What needs improvement:**
- ⚠️ Fix Arousal range documentation inconsistency
- ⚠️ Add offline bounce detection
- ⚠️ Verify transport sync implementation
- ⚠️ Add latency reporting differentiation
- ⚠️ Add host-specific handling
- ⚠️ Improve error handling
- ⚠️ Add plugin validation in CI

**Priority actions:**
1. Fix Arousal range (documentation or code)
2. Add offline bounce detection
3. Verify transport sync per-block querying
4. Add latency reporting for Kelly vs Dee paths
5. Add CI validation (auval, VST3 validator)
