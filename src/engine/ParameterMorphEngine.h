#pragma once
/*
 * ParameterMorphEngine.h - Real-Time Parameter Morphing Engine
 * ============================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - UI Layer: Used by EmotionWorkstation (smooth parameter transitions)
 * - Plugin Layer: Used by PluginProcessor (real-time parameter morphing)
 * - Engine Layer: Provides smooth interpolation for parameter changes
 *
 * Purpose: Smooth real-time parameter interpolation for transitions between values.
 *          Supports multi-parameter morphing and gesture-based control.
 *
 * Features:
 * - Smooth transitions between parameter values
 * - Multi-parameter simultaneous morphing
 * - Configurable morph duration
 * - Gesture-based control support
 */

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <map>
#include <atomic>
#include <functional>

namespace kelly {

/**
 * ParameterMorphEngine - Smooth real-time parameter interpolation
 *
 * Provides smooth transitions between parameter values for real-time adjustment.
 * Supports multi-parameter morphing and gesture-based control.
 */
class ParameterMorphEngine : public juce::Timer {
public:
    ParameterMorphEngine();
    ~ParameterMorphEngine() override;

    /**
     * Start morphing a parameter from current value to target value
     * @param parameterName Parameter to morph
     * @param targetValue Target value
     * @param durationMs Duration of morph in milliseconds (default: 200ms)
     */
    void morphParameter(const juce::String& parameterName, float targetValue, int durationMs = 200);

    /**
     * Morph multiple parameters simultaneously
     * @param targets Map of parameter names to target values
     * @param durationMs Duration of morph in milliseconds
     */
    void morphParameters(const std::map<juce::String, float>& targets, int durationMs = 200);

    /**
     * Get current interpolated value for a parameter
     * @param parameterName Parameter name
     * @return Current interpolated value (or 0.0 if not morphing)
     */
    float getCurrentValue(const juce::String& parameterName) const;

    /**
     * Check if a parameter is currently morphing
     */
    bool isMorphing(const juce::String& parameterName) const;

    /**
     * Stop morphing for a parameter (snap to current value)
     */
    void stopMorph(const juce::String& parameterName);

    /**
     * Stop all morphing
     */
    void stopAllMorphs();

    /**
     * Set interpolation curve type
     */
    enum class CurveType {
        Linear,
        EaseIn,
        EaseOut,
        EaseInOut,
        Exponential
    };
    void setCurveType(CurveType type) { curveType_ = type; }
    CurveType getCurveType() const { return curveType_; }

    /**
     * Callback for parameter value updates
     * Called during morphing with updated parameter values
     */
    std::function<void(const juce::String&, float)> onParameterUpdated;

    /**
     * Callback for morph completion
     */
    std::function<void(const juce::String&)> onMorphComplete;

    // Timer callback (called at 60fps for smooth interpolation)
    void timerCallback() override;

private:
    struct MorphState {
        float startValue = 0.0f;
        float targetValue = 0.0f;
        juce::int64 startTime = 0;
        int durationMs = 200;
        bool active = false;
    };

    std::map<juce::String, MorphState> morphStates_;
    mutable std::mutex morphMutex_;
    CurveType curveType_ = CurveType::EaseInOut;

    // Interpolation helpers
    float interpolate(float start, float end, float t, CurveType curve) const;
    float applyCurve(float t, CurveType curve) const;
};

} // namespace kelly
