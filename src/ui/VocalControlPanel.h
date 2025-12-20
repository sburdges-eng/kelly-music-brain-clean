#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "../voice/LyricTypes.h"
#include "../voice/VoiceSynthesizer.h"

namespace kelly {

/**
 * VocalControlPanel - Controls for vocal synthesis parameters.
 *
 * Features:
 * - Voice type selector (Male/Female/Child)
 * - Vibrato depth/rate sliders
 * - Breathiness control
 * - Brightness control
 * - Expression curve visualization
 */
class VocalControlPanel : public juce::Component {
public:
    VocalControlPanel();
    ~VocalControlPanel() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    /**
     * Get current voice type.
     */
    VoiceType getVoiceType() const;

    /**
     * Get current expression parameters.
     */
    VocalExpression getExpression() const;

    /**
     * Set voice type.
     */
    void setVoiceType(VoiceType voiceType);

    /**
     * Set expression parameters.
     */
    void setExpression(const VocalExpression& expression);

    // Callbacks
    std::function<void(VoiceType)> onVoiceTypeChanged;
    std::function<void(const VocalExpression&)> onExpressionChanged;

private:
    // Voice type selector
    juce::ComboBox voiceTypeComboBox_;
    juce::Label voiceTypeLabel_{"", "Voice Type"};

    // Expression controls
    juce::Slider vibratoDepthSlider_;
    juce::Slider vibratoRateSlider_;
    juce::Slider breathinessSlider_;
    juce::Slider brightnessSlider_;
    juce::Slider dynamicsSlider_;
    juce::Slider articulationSlider_;

    // Labels
    juce::Label vibratoDepthLabel_{"", "Vibrato Depth"};
    juce::Label vibratoRateLabel_{"", "Vibrato Rate (Hz)"};
    juce::Label breathinessLabel_{"", "Breathiness"};
    juce::Label brightnessLabel_{"", "Brightness"};
    juce::Label dynamicsLabel_{"", "Dynamics"};
    juce::Label articulationLabel_{"", "Articulation"};

    // Current values
    VoiceType currentVoiceType_ = VoiceType::Neutral;
    VocalExpression currentExpression_;

    /**
     * Setup sliders with appropriate ranges.
     */
    void setupSliders();

    /**
     * Update expression from sliders.
     */
    void updateExpressionFromSliders();

    /**
     * Voice type selection changed.
     */
    void voiceTypeChanged();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VocalControlPanel)
};

} // namespace kelly
