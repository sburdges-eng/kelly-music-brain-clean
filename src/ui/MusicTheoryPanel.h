#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "../midi/InstrumentSelector.h"

namespace kelly {

/**
 * Music Theory Panel - A-side of cassette
 * 
 * Strictly music theory based interface with:
 * - Sheet music notation options
 * - Traditional music theory controls
 * - Instrument selection with techniques
 * - Effects options
 */
class MusicTheoryPanel : public juce::Component {
public:
    MusicTheoryPanel();
    ~MusicTheoryPanel() override = default;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // Getters for current settings
    struct TheorySettings {
        // Key and Scale
        juce::String key = "C";
        juce::String scale = "Major";
        juce::String mode = "Ionian";
        
        // Time Signature
        int timeSigNumerator = 4;
        int timeSigDenominator = 4;
        
        // Tempo
        int tempoBpm = 120;
        
        // Instruments (GM program numbers)
        int leadInstrument = GM::ACOUSTIC_GRAND_PIANO;
        int harmonyInstrument = GM::STRING_ENSEMBLE_1;
        int bassInstrument = GM::ACOUSTIC_BASS;
        int textureInstrument = GM::PAD_WARM;
        
        // Techniques per instrument
        juce::String leadTechnique = "Legato";
        juce::String harmonyTechnique = "Sustained";
        juce::String bassTechnique = "Root Notes";
        juce::String textureTechnique = "Pad";
        
        // Effects
        bool useReverb = false;
        float reverbAmount = 0.3f;
        bool useDelay = false;
        float delayAmount = 0.2f;
        bool useChorus = false;
        float chorusAmount = 0.15f;
        
        // EQ
        bool useEQ = false;
        juce::String eqPreset = "neutral";  // Preset name or "custom"
        bool eqAutoApply = true;  // Auto-apply based on emotion/genre
        
        // Sheet Music Options
        bool showNotation = true;
        juce::String notationStyle = "Traditional";
        bool showChordSymbols = true;
        bool showRomanNumerals = false;
        
        // Custom Progression (optional override)
        bool useCustomProgression = false;
        juce::String customProgression = "";  // e.g., "1,4,5,1" or "I,IV,V,I"
        bool strictCustomProgression = false;  // If true, use EXACT progression with NO modifications
    };
    
    TheorySettings getSettings() const;
    
    /** Set custom progression (comma-separated degrees, e.g., "1,4,5,1") */
    void setCustomProgression(const juce::String& progression, bool strict = false);
    
    // Setter methods for linked parameter system
    void setKey(const juce::String& key);
    void setMode(const juce::String& mode);
    void setTempo(int tempo);
    void setProgression(const std::vector<juce::String>& progression);
    
    // EQ methods
    void setEQPreset(const juce::String& presetName);
    void setEQAutoApply(bool autoApply);
    juce::String getCurrentEQPreset() const;
    
    // Callbacks
    std::function<void()> onSettingsChanged;
    
private:
    // Key and Scale
    juce::ComboBox keySelector_;
    juce::Label keyLabel_;
    juce::ComboBox scaleSelector_;
    juce::Label scaleLabel_;
    juce::ComboBox modeSelector_;
    juce::Label modeLabel_;
    
    // Time Signature
    juce::ComboBox timeSigNumSelector_;
    juce::ComboBox timeSigDenSelector_;
    juce::Label timeSigLabel_;
    
    // Tempo
    juce::Slider tempoSlider_;
    juce::Label tempoLabel_;
    
    // Instruments
    juce::ComboBox leadInstrumentSelector_;
    juce::Label leadInstrumentLabel_;
    juce::ComboBox leadTechniqueSelector_;
    juce::Label leadTechniqueLabel_;
    
    juce::ComboBox harmonyInstrumentSelector_;
    juce::Label harmonyInstrumentLabel_;
    juce::ComboBox harmonyTechniqueSelector_;
    juce::Label harmonyTechniqueLabel_;
    
    juce::ComboBox bassInstrumentSelector_;
    juce::Label bassInstrumentLabel_;
    juce::ComboBox bassTechniqueSelector_;
    juce::Label bassTechniqueLabel_;
    
    juce::ComboBox textureInstrumentSelector_;
    juce::Label textureInstrumentLabel_;
    juce::ComboBox textureTechniqueSelector_;
    juce::Label textureTechniqueLabel_;
    
    // Effects
    juce::ToggleButton reverbToggle_;
    juce::Slider reverbSlider_;
    juce::Label reverbLabel_;
    
    juce::ToggleButton delayToggle_;
    juce::Slider delaySlider_;
    juce::Label delayLabel_;
    
    juce::ToggleButton chorusToggle_;
    juce::Slider chorusSlider_;
    juce::Label chorusLabel_;
    
    // Sheet Music Options
    juce::ToggleButton notationToggle_;
    juce::ComboBox notationStyleSelector_;
    juce::Label notationStyleLabel_;
    juce::ToggleButton chordSymbolsToggle_;
    juce::ToggleButton romanNumeralsToggle_;
    
    // EQ Controls
    juce::ToggleButton eqToggle_;
    juce::Label eqLabel_;
    juce::ComboBox eqPresetSelector_;
    juce::Label eqPresetLabel_;
    juce::ToggleButton eqAutoApplyToggle_;
    juce::Label eqAutoApplyLabel_;
    
    void initializeInstruments();
    void initializeTechniques();
    void notifySettingsChanged();
    
    // Helper to parse progression string
    std::vector<int> parseProgressionString(const juce::String& progression) const;
    
    // Stored custom progression
    juce::String storedCustomProgression_;
    bool storedStrictMode_ = false;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MusicTheoryPanel)
};

} // namespace kelly
