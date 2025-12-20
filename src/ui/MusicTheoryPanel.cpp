#include "MusicTheoryPanel.h"
#include "KellyLookAndFeel.h"
#include "../common/EQPresetManager.h"
#include <algorithm>
#include <map>

namespace kelly {

MusicTheoryPanel::MusicTheoryPanel() {
    setOpaque(true);
    
    // Key selector
    keyLabel_.setText("Key", juce::dontSendNotification);
    keyLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    keyLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(keyLabel_);
    
    keySelector_.addItem("C", 1);
    keySelector_.addItem("C#", 2);
    keySelector_.addItem("D", 3);
    keySelector_.addItem("D#", 4);
    keySelector_.addItem("E", 5);
    keySelector_.addItem("F", 6);
    keySelector_.addItem("F#", 7);
    keySelector_.addItem("G", 8);
    keySelector_.addItem("G#", 9);
    keySelector_.addItem("A", 10);
    keySelector_.addItem("A#", 11);
    keySelector_.addItem("B", 12);
    keySelector_.setSelectedId(1);
    keySelector_.setTooltip("Musical key: Select the root note (C, D, E, F, G, A, B)");
    keySelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(keySelector_);
    
    // Scale selector
    scaleLabel_.setText("Scale", juce::dontSendNotification);
    scaleLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    scaleLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(scaleLabel_);
    
    scaleSelector_.addItem("Major", 1);
    scaleSelector_.addItem("Minor", 2);
    scaleSelector_.addItem("Dorian", 3);
    scaleSelector_.addItem("Phrygian", 4);
    scaleSelector_.addItem("Lydian", 5);
    scaleSelector_.addItem("Mixolydian", 6);
    scaleSelector_.addItem("Locrian", 7);
    scaleSelector_.addItem("Harmonic Minor", 8);
    scaleSelector_.addItem("Melodic Minor", 9);
    scaleSelector_.addItem("Pentatonic Major", 10);
    scaleSelector_.addItem("Pentatonic Minor", 11);
    scaleSelector_.addItem("Blues", 12);
    scaleSelector_.setSelectedId(1);
    scaleSelector_.setTooltip("Scale type: Major, Minor, or other scales (Dorian, Phrygian, etc.)");
    scaleSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(scaleSelector_);
    
    // Mode selector
    modeLabel_.setText("Mode", juce::dontSendNotification);
    modeLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    modeLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(modeLabel_);
    
    modeSelector_.addItem("Ionian", 1);
    modeSelector_.addItem("Dorian", 2);
    modeSelector_.addItem("Phrygian", 3);
    modeSelector_.addItem("Lydian", 4);
    modeSelector_.addItem("Mixolydian", 5);
    modeSelector_.addItem("Aeolian", 6);
    modeSelector_.addItem("Locrian", 7);
    modeSelector_.setSelectedId(1);
    modeSelector_.setTooltip("Musical mode: Ionian (major), Aeolian (minor), Dorian, Phrygian, Lydian, Mixolydian, or Locrian");
    modeSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(modeSelector_);
    
    // Time Signature
    timeSigLabel_.setText("Time Signature", juce::dontSendNotification);
    timeSigLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    timeSigLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    timeSigLabel_.setTooltip("Time signature: numerator/denominator (e.g., 4/4 = four beats per measure)");
    addAndMakeVisible(timeSigLabel_);
    
    timeSigNumSelector_.addItem("2", 1);
    timeSigNumSelector_.addItem("3", 2);
    timeSigNumSelector_.addItem("4", 3);
    timeSigNumSelector_.addItem("5", 4);
    timeSigNumSelector_.addItem("6", 5);
    timeSigNumSelector_.addItem("7", 6);
    timeSigNumSelector_.addItem("9", 7);
    timeSigNumSelector_.addItem("12", 8);
    timeSigNumSelector_.setSelectedId(3);
    timeSigNumSelector_.setTooltip("Beats per measure (numerator)");
    timeSigNumSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(timeSigNumSelector_);
    
    timeSigDenSelector_.addItem("2", 1);
    timeSigDenSelector_.addItem("4", 2);
    timeSigDenSelector_.addItem("8", 3);
    timeSigDenSelector_.addItem("16", 4);
    timeSigDenSelector_.setSelectedId(2);
    timeSigDenSelector_.setTooltip("Note value for beat (denominator)");
    timeSigDenSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(timeSigDenSelector_);
    
    // Tempo
    tempoLabel_.setText("Tempo (BPM)", juce::dontSendNotification);
    tempoLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    tempoLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(tempoLabel_);
    
    tempoSlider_.setRange(40.0, 200.0, 1.0);
    tempoSlider_.setValue(120.0);
    tempoSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    tempoSlider_.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 80, 24);  // Larger, above slider
    tempoSlider_.setNumDecimalPlacesToDisplay(0);  // Integer value
    tempoSlider_.setColour(juce::Slider::textBoxTextColourId, KellyLookAndFeel::textPrimary);
    tempoSlider_.setColour(juce::Slider::textBoxBackgroundColourId, KellyLookAndFeel::surfaceColor);
    tempoSlider_.setColour(juce::Slider::textBoxOutlineColourId, KellyLookAndFeel::borderColor);
    tempoSlider_.setTooltip("Tempo in beats per minute (BPM): 40-200. Lower = slower, higher = faster");
    tempoSlider_.onValueChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(tempoSlider_);
    
    // Initialize instruments and techniques
    initializeInstruments();
    initializeTechniques();
    
    // Effects
    reverbLabel_.setText("Reverb", juce::dontSendNotification);
    reverbLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    reverbLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(reverbLabel_);
    
    reverbToggle_.setButtonText("On");
    reverbToggle_.setToggleState(false, juce::dontSendNotification);
    reverbToggle_.setTooltip("Enable reverb effect: Adds spatial depth and room ambience");
    reverbToggle_.onClick = [this] { notifySettingsChanged(); };
    addAndMakeVisible(reverbToggle_);
    
    reverbSlider_.setRange(0.0, 1.0, 0.01);
    reverbSlider_.setValue(0.3);
    reverbSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    reverbSlider_.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 70, 22);  // Larger, above slider
    reverbSlider_.setNumDecimalPlacesToDisplay(2);
    reverbSlider_.setColour(juce::Slider::textBoxTextColourId, KellyLookAndFeel::textPrimary);
    reverbSlider_.setColour(juce::Slider::textBoxBackgroundColourId, KellyLookAndFeel::surfaceColor);
    reverbSlider_.setColour(juce::Slider::textBoxOutlineColourId, KellyLookAndFeel::borderColor);
    reverbSlider_.onValueChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(reverbSlider_);
    
    delayLabel_.setText("Delay", juce::dontSendNotification);
    delayLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    delayLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(delayLabel_);
    
    delayToggle_.setButtonText("On");
    delayToggle_.setToggleState(false, juce::dontSendNotification);
    delayToggle_.setTooltip("Enable delay effect: Creates echo and rhythmic repeats");
    delayToggle_.onClick = [this] { notifySettingsChanged(); };
    addAndMakeVisible(delayToggle_);
    
    delaySlider_.setRange(0.0, 1.0, 0.01);
    delaySlider_.setValue(0.2);
    delaySlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    delaySlider_.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 70, 22);  // Larger, above slider
    delaySlider_.setNumDecimalPlacesToDisplay(2);
    delaySlider_.setColour(juce::Slider::textBoxTextColourId, KellyLookAndFeel::textPrimary);
    delaySlider_.setColour(juce::Slider::textBoxBackgroundColourId, KellyLookAndFeel::surfaceColor);
    delaySlider_.setColour(juce::Slider::textBoxOutlineColourId, KellyLookAndFeel::borderColor);
    delaySlider_.onValueChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(delaySlider_);
    
    chorusLabel_.setText("Chorus", juce::dontSendNotification);
    chorusLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    chorusLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(chorusLabel_);
    
    chorusToggle_.setButtonText("On");
    chorusToggle_.setToggleState(false, juce::dontSendNotification);
    chorusToggle_.setTooltip("Enable chorus effect: Adds width and shimmer to the sound");
    chorusToggle_.onClick = [this] { notifySettingsChanged(); };
    addAndMakeVisible(chorusToggle_);
    
    chorusSlider_.setRange(0.0, 1.0, 0.01);
    chorusSlider_.setValue(0.15);
    chorusSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    chorusSlider_.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 70, 22);  // Larger, above slider
    chorusSlider_.setNumDecimalPlacesToDisplay(2);
    chorusSlider_.setColour(juce::Slider::textBoxTextColourId, KellyLookAndFeel::textPrimary);
    chorusSlider_.setColour(juce::Slider::textBoxBackgroundColourId, KellyLookAndFeel::surfaceColor);
    chorusSlider_.setColour(juce::Slider::textBoxOutlineColourId, KellyLookAndFeel::borderColor);
    chorusSlider_.onValueChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(chorusSlider_);
    
    // Sheet Music Options
    notationToggle_.setButtonText("Show Notation");
    notationToggle_.setToggleState(true, juce::dontSendNotification);
    notationToggle_.setTooltip("Show sheet music notation: Display musical notation for generated MIDI");
    notationToggle_.onClick = [this] { notifySettingsChanged(); };
    addAndMakeVisible(notationToggle_);
    
    notationStyleLabel_.setText("Notation Style", juce::dontSendNotification);
    notationStyleLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    notationStyleLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(notationStyleLabel_);
    
    notationStyleSelector_.addItem("Traditional", 1);
    notationStyleSelector_.addItem("Modern", 2);
    notationStyleSelector_.addItem("Jazz", 3);
    notationStyleSelector_.setSelectedId(1);
    notationStyleSelector_.setTooltip("Notation style: Traditional (classical), Modern (contemporary), or Jazz notation");
    notationStyleSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(notationStyleSelector_);
    
    chordSymbolsToggle_.setButtonText("Chord Symbols");
    chordSymbolsToggle_.setToggleState(true, juce::dontSendNotification);
    chordSymbolsToggle_.setTooltip("Show chord symbols: Display chord names (C, Am, F, etc.) on notation");
    chordSymbolsToggle_.onClick = [this] { notifySettingsChanged(); };
    addAndMakeVisible(chordSymbolsToggle_);
    
    romanNumeralsToggle_.setButtonText("Roman Numerals");
    romanNumeralsToggle_.setToggleState(false, juce::dontSendNotification);
    romanNumeralsToggle_.setTooltip("Show Roman numerals: Display chord functions (I, IV, V, vi, etc.)");
    romanNumeralsToggle_.onClick = [this] { notifySettingsChanged(); };
    addAndMakeVisible(romanNumeralsToggle_);
    
    // =========================================================================
    // EQ CONTROLS
    // =========================================================================
    
    eqLabel_.setText("EQ", juce::dontSendNotification);
    eqLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));
    eqLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);
    eqLabel_.setTooltip("Equalizer: Adjust frequency response based on emotion or genre");
    addAndMakeVisible(eqLabel_);
    
    eqToggle_.setButtonText("Enable EQ");
    eqToggle_.setToggleState(false, juce::dontSendNotification);
    eqToggle_.setTooltip("Enable EQ processing: Apply frequency shaping based on selected preset");
    eqToggle_.onClick = [this] { notifySettingsChanged(); };
    addAndMakeVisible(eqToggle_);
    
    eqPresetLabel_.setText("EQ Preset", juce::dontSendNotification);
    eqPresetLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));
    eqPresetLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);
    eqPresetLabel_.setTooltip("EQ Preset: Choose emotion or genre-based EQ curve");
    addAndMakeVisible(eqPresetLabel_);
    
    // Populate EQ preset selector with emotion and genre presets
    eqPresetSelector_.addItem("Auto (from Emotion)", 1);
    eqPresetSelector_.addSeparator();
    
    // Emotion presets
    eqPresetSelector_.addItem("Joy", 2);
    eqPresetSelector_.addItem("Sadness", 3);
    eqPresetSelector_.addItem("Anger", 4);
    eqPresetSelector_.addItem("Fear", 5);
    eqPresetSelector_.addItem("Peace", 6);
    eqPresetSelector_.addItem("Grief", 7);
    eqPresetSelector_.addItem("Triumph", 8);
    eqPresetSelector_.addItem("Anxiety", 9);
    eqPresetSelector_.addItem("Hope", 10);
    eqPresetSelector_.addItem("Neutral", 11);
    
    eqPresetSelector_.addSeparator();
    
    // Genre presets
    eqPresetSelector_.addItem("Pop", 12);
    eqPresetSelector_.addItem("Rock", 13);
    eqPresetSelector_.addItem("Jazz", 14);
    eqPresetSelector_.addItem("Classical", 15);
    eqPresetSelector_.addItem("Electronic", 16);
    eqPresetSelector_.addItem("Ambient", 17);
    eqPresetSelector_.addItem("Blues", 18);
    eqPresetSelector_.addItem("Folk", 19);
    eqPresetSelector_.addItem("Cinematic", 20);
    eqPresetSelector_.addItem("R&B", 21);
    
    eqPresetSelector_.setSelectedId(1);  // Default to Auto
    eqPresetSelector_.setTooltip("Select EQ preset: Auto applies based on current emotion, or choose manually");
    eqPresetSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(eqPresetSelector_);
    
    eqAutoApplyLabel_.setText("Auto-Apply", juce::dontSendNotification);
    eqAutoApplyLabel_.setFont(juce::FontOptions(12.0f));
    eqAutoApplyLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textSecondary);
    eqAutoApplyLabel_.setTooltip("Auto-apply EQ: Automatically update EQ preset when emotion or genre changes");
    addAndMakeVisible(eqAutoApplyLabel_);
    
    eqAutoApplyToggle_.setButtonText("");
    eqAutoApplyToggle_.setToggleState(true, juce::dontSendNotification);  // Default: enabled
    eqAutoApplyToggle_.setTooltip("Auto-apply EQ preset based on emotion/genre");
    eqAutoApplyToggle_.onClick = [this] { notifySettingsChanged(); };
    addAndMakeVisible(eqAutoApplyToggle_);
}

void MusicTheoryPanel::initializeInstruments() {
    // Lead Instruments
    leadInstrumentLabel_.setText("Lead Instrument", juce::dontSendNotification);
    leadInstrumentLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    leadInstrumentLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(leadInstrumentLabel_);
    
    leadInstrumentSelector_.addItem("Acoustic Grand Piano", GM::ACOUSTIC_GRAND_PIANO + 1);
    leadInstrumentSelector_.addItem("Electric Piano", GM::ELECTRIC_PIANO_1 + 1);
    leadInstrumentSelector_.addItem("Violin", GM::VIOLIN + 1);
    leadInstrumentSelector_.addItem("Flute", GM::FLUTE + 1);
    leadInstrumentSelector_.addItem("Trumpet", GM::TRUMPET + 1);
    leadInstrumentSelector_.addItem("Soprano Sax", GM::SOPRANO_SAX + 1);
    leadInstrumentSelector_.addItem("Acoustic Guitar (Nylon)", GM::ACOUSTIC_GUITAR_NYLON + 1);
    leadInstrumentSelector_.addItem("Electric Guitar (Clean)", GM::ELECTRIC_GUITAR_CLEAN + 1);
    leadInstrumentSelector_.setSelectedId(GM::ACOUSTIC_GRAND_PIANO + 1);
    leadInstrumentSelector_.setTooltip("Lead instrument: The main melodic instrument in your composition");
    leadInstrumentSelector_.onChange = [this] { 
        initializeTechniques(); 
        notifySettingsChanged(); 
    };
    addAndMakeVisible(leadInstrumentSelector_);
    
    leadTechniqueLabel_.setText("Lead Technique", juce::dontSendNotification);
    leadTechniqueLabel_.setFont(juce::FontOptions(13.0f).withStyle("Medium"));  // Larger, readable
    leadTechniqueLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(leadTechniqueLabel_);
    
    leadTechniqueSelector_.setTooltip("Playing technique for lead instrument: Legato, Staccato, Pizzicato, etc.");
    leadTechniqueSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(leadTechniqueSelector_);
    
    // Harmony Instruments
    harmonyInstrumentLabel_.setText("Harmony Instrument", juce::dontSendNotification);
    harmonyInstrumentLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    harmonyInstrumentLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(harmonyInstrumentLabel_);
    
    harmonyInstrumentSelector_.addItem("String Ensemble 1", GM::STRING_ENSEMBLE_1 + 1);
    harmonyInstrumentSelector_.addItem("String Ensemble 2", GM::STRING_ENSEMBLE_2 + 1);
    harmonyInstrumentSelector_.addItem("Choir Aahs", GM::CHOIR_AAHS + 1);
    harmonyInstrumentSelector_.addItem("Brass Section", GM::BRASS_SECTION + 1);
    harmonyInstrumentSelector_.addItem("Synth Strings 1", GM::SYNTH_STRINGS_1 + 1);
    harmonyInstrumentSelector_.setSelectedId(GM::STRING_ENSEMBLE_1 + 1);
    harmonyInstrumentSelector_.setTooltip("Harmony instrument: Provides chordal accompaniment");
    harmonyInstrumentSelector_.onChange = [this] { 
        initializeTechniques(); 
        notifySettingsChanged(); 
    };
    addAndMakeVisible(harmonyInstrumentSelector_);
    
    harmonyTechniqueLabel_.setText("Harmony Technique", juce::dontSendNotification);
    harmonyTechniqueLabel_.setFont(juce::FontOptions(13.0f).withStyle("Medium"));  // Larger, readable
    harmonyTechniqueLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(harmonyTechniqueLabel_);
    
    harmonyTechniqueSelector_.setTooltip("Playing technique for harmony instrument");
    harmonyTechniqueSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(harmonyTechniqueSelector_);
    
    // Bass Instruments
    bassInstrumentLabel_.setText("Bass Instrument", juce::dontSendNotification);
    bassInstrumentLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    bassInstrumentLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(bassInstrumentLabel_);
    
    bassInstrumentSelector_.addItem("Acoustic Bass", GM::ACOUSTIC_BASS + 1);
    bassInstrumentSelector_.addItem("Electric Bass (Finger)", GM::ELECTRIC_BASS_FINGER + 1);
    bassInstrumentSelector_.addItem("Electric Bass (Pick)", GM::ELECTRIC_BASS_PICK + 1);
    bassInstrumentSelector_.addItem("Fretless Bass", GM::FRETLESS_BASS + 1);
    bassInstrumentSelector_.addItem("Slap Bass 1", GM::SLAP_BASS_1 + 1);
    bassInstrumentSelector_.addItem("Synth Bass 1", GM::SYNTH_BASS_1 + 1);
    bassInstrumentSelector_.setSelectedId(GM::ACOUSTIC_BASS + 1);
    bassInstrumentSelector_.setTooltip("Bass instrument: Provides the low-frequency foundation");
    bassInstrumentSelector_.onChange = [this] { 
        initializeTechniques(); 
        notifySettingsChanged(); 
    };
    addAndMakeVisible(bassInstrumentSelector_);
    
    bassTechniqueLabel_.setText("Bass Technique", juce::dontSendNotification);
    bassTechniqueLabel_.setFont(juce::FontOptions(13.0f).withStyle("Medium"));  // Larger, readable
    bassTechniqueLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(bassTechniqueLabel_);
    
    bassTechniqueSelector_.setTooltip("Playing technique for bass instrument");
    bassTechniqueSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(bassTechniqueSelector_);
    
    // Texture Instruments
    textureInstrumentLabel_.setText("Texture Instrument", juce::dontSendNotification);
    textureInstrumentLabel_.setFont(juce::FontOptions(14.0f).withStyle("SemiBold"));  // Larger, bolder
    textureInstrumentLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(textureInstrumentLabel_);
    
    textureInstrumentSelector_.addItem("Pad Warm", GM::PAD_WARM + 1);
    textureInstrumentSelector_.addItem("Pad New Age", GM::PAD_NEW_AGE + 1);
    textureInstrumentSelector_.addItem("Pad Choir", GM::PAD_CHOIR + 1);
    textureInstrumentSelector_.addItem("Pad Bowed", GM::PAD_BOWED + 1);
    textureInstrumentSelector_.addItem("Pad Halo", GM::PAD_HALO + 1);
    textureInstrumentSelector_.setSelectedId(GM::PAD_WARM + 1);
    textureInstrumentSelector_.setTooltip("Texture instrument: Adds atmospheric pads and background layers");
    textureInstrumentSelector_.onChange = [this] { 
        initializeTechniques(); 
        notifySettingsChanged(); 
    };
    addAndMakeVisible(textureInstrumentSelector_);
    
    textureTechniqueLabel_.setText("Texture Technique", juce::dontSendNotification);
    textureTechniqueLabel_.setFont(juce::FontOptions(13.0f).withStyle("Medium"));  // Larger, readable
    textureTechniqueLabel_.setColour(juce::Label::textColourId, KellyLookAndFeel::textPrimary);  // Brighter text
    addAndMakeVisible(textureTechniqueLabel_);
    
    textureTechniqueSelector_.setTooltip("Playing technique for texture instrument");
    textureTechniqueSelector_.onChange = [this] { notifySettingsChanged(); };
    addAndMakeVisible(textureTechniqueSelector_);
}

void MusicTheoryPanel::initializeTechniques() {
    // Clear existing techniques
    leadTechniqueSelector_.clear();
    harmonyTechniqueSelector_.clear();
    bassTechniqueSelector_.clear();
    textureTechniqueSelector_.clear();
    
    // Lead techniques based on selected instrument
    int leadInst = leadInstrumentSelector_.getSelectedId() - 1;
    if (leadInst >= GM::ACOUSTIC_GRAND_PIANO && leadInst <= GM::CLAVINET) {
        // Piano techniques
        leadTechniqueSelector_.addItem("Legato", 1);
        leadTechniqueSelector_.addItem("Staccato", 2);
        leadTechniqueSelector_.addItem("Arpeggiated", 3);
        leadTechniqueSelector_.addItem("Block Chords", 4);
        leadTechniqueSelector_.addItem("Broken Chords", 5);
    } else if (leadInst >= GM::VIOLIN && leadInst <= GM::TIMPANI) {
        // String techniques
        leadTechniqueSelector_.addItem("Legato", 1);
        leadTechniqueSelector_.addItem("Staccato", 2);
        leadTechniqueSelector_.addItem("Pizzicato", 3);
        leadTechniqueSelector_.addItem("Tremolo", 4);
        leadTechniqueSelector_.addItem("Vibrato", 5);
        leadTechniqueSelector_.addItem("Harmonics", 6);
    } else if (leadInst >= GM::TRUMPET && leadInst <= GM::SYNTH_BRASS_2) {
        // Brass techniques
        leadTechniqueSelector_.addItem("Sustained", 1);
        leadTechniqueSelector_.addItem("Staccato", 2);
        leadTechniqueSelector_.addItem("Muted", 3);
        leadTechniqueSelector_.addItem("Falls", 4);
        leadTechniqueSelector_.addItem("Glissando", 5);
    } else if (leadInst >= GM::SOPRANO_SAX && leadInst <= GM::CLARINET) {
        // Woodwind techniques
        leadTechniqueSelector_.addItem("Legato", 1);
        leadTechniqueSelector_.addItem("Staccato", 2);
        leadTechniqueSelector_.addItem("Trills", 3);
        leadTechniqueSelector_.addItem("Flutter Tongue", 4);
        leadTechniqueSelector_.addItem("Slap Tongue", 5);
    } else if (leadInst >= GM::ACOUSTIC_GUITAR_NYLON && leadInst <= GM::GUITAR_HARMONICS) {
        // Guitar techniques
        leadTechniqueSelector_.addItem("Fingerpicking", 1);
        leadTechniqueSelector_.addItem("Strumming", 2);
        leadTechniqueSelector_.addItem("Arpeggiated", 3);
        leadTechniqueSelector_.addItem("Harmonics", 4);
        leadTechniqueSelector_.addItem("Bends", 5);
        leadTechniqueSelector_.addItem("Slides", 6);
        leadTechniqueSelector_.addItem("Muted", 7);
    } else {
        // Default
        leadTechniqueSelector_.addItem("Legato", 1);
        leadTechniqueSelector_.addItem("Staccato", 2);
        leadTechniqueSelector_.addItem("Sustained", 3);
    }
    leadTechniqueSelector_.setSelectedId(1);
    
    // Harmony techniques
    int harmonyInst = harmonyInstrumentSelector_.getSelectedId() - 1;
    if (harmonyInst >= GM::STRING_ENSEMBLE_1 && harmonyInst <= GM::SYNTH_STRINGS_2) {
        harmonyTechniqueSelector_.addItem("Sustained", 1);
        harmonyTechniqueSelector_.addItem("Legato", 2);
        harmonyTechniqueSelector_.addItem("Staccato", 3);
        harmonyTechniqueSelector_.addItem("Tremolo", 4);
        harmonyTechniqueSelector_.addItem("Pizzicato", 5);
    } else if (harmonyInst >= GM::CHOIR_AAHS && harmonyInst <= GM::SYNTH_VOICE) {
        harmonyTechniqueSelector_.addItem("Sustained", 1);
        harmonyTechniqueSelector_.addItem("Staccato", 2);
        harmonyTechniqueSelector_.addItem("Legato", 3);
    } else {
        harmonyTechniqueSelector_.addItem("Sustained", 1);
        harmonyTechniqueSelector_.addItem("Block Chords", 2);
        harmonyTechniqueSelector_.addItem("Arpeggiated", 3);
    }
    harmonyTechniqueSelector_.setSelectedId(1);
    
    // Bass techniques
    int bassInst = bassInstrumentSelector_.getSelectedId() - 1;
    if (bassInst >= GM::ACOUSTIC_BASS && bassInst <= GM::FRETLESS_BASS) {
        bassTechniqueSelector_.addItem("Root Notes", 1);
        bassTechniqueSelector_.addItem("Walking Bass", 2);
        bassTechniqueSelector_.addItem("Slap", 3);
        bassTechniqueSelector_.addItem("Pop", 4);
        bassTechniqueSelector_.addItem("Fingerstyle", 5);
        bassTechniqueSelector_.addItem("Picked", 6);
    } else if (bassInst >= GM::SLAP_BASS_1 && bassInst <= GM::SLAP_BASS_2) {
        bassTechniqueSelector_.addItem("Slap", 1);
        bassTechniqueSelector_.addItem("Pop", 2);
        bassTechniqueSelector_.addItem("Thumb", 3);
    } else {
        bassTechniqueSelector_.addItem("Root Notes", 1);
        bassTechniqueSelector_.addItem("Arpeggiated", 2);
        bassTechniqueSelector_.addItem("Sustained", 3);
    }
    bassTechniqueSelector_.setSelectedId(1);
    
    // Texture techniques
    textureTechniqueSelector_.addItem("Pad", 1);
    textureTechniqueSelector_.addItem("Sustained", 2);
    textureTechniqueSelector_.addItem("Swelling", 3);
    textureTechniqueSelector_.addItem("Pulsing", 4);
    textureTechniqueSelector_.addItem("Evolving", 5);
    textureTechniqueSelector_.setSelectedId(1);
}

void MusicTheoryPanel::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();
    
    // Modern gradient background matching main UI
    juce::ColourGradient gradient(
        KellyLookAndFeel::backgroundDark, bounds.getTopLeft(),
        KellyLookAndFeel::backgroundLight, bounds.getBottomLeft(),
        false
    );
    g.setGradientFill(gradient);
    g.fillAll();
    
    // Section header with exact documented colors (Focus Blue & Creative Purple)
    auto headerBounds = bounds.removeFromTop(35.0f);
    juce::ColourGradient headerGradient(
        juce::Colour(0xFF3B82F6).withAlpha(0.15f), headerBounds.getTopLeft(),  // Focus Blue #3B82F6
        juce::Colour(0xFFA855F7).withAlpha(0.1f), headerBounds.getBottomLeft(), // Creative Purple #A855F7
        false
    );
    g.setGradientFill(headerGradient);
    g.fillRoundedRectangle(headerBounds, 0.0f);
    
    // Title with modern typography
    g.setColour(KellyLookAndFeel::textPrimary);
    g.setFont(juce::FontOptions(20.0f).withStyle("Bold"));
    
    // Text shadow for depth
    g.setColour(juce::Colours::black.withAlpha(0.3f));
    g.drawText("MUSIC THEORY", headerBounds.translated(1.0f, 1.0f), juce::Justification::centred);
    
    // Main text
    g.setColour(KellyLookAndFeel::textPrimary);
    g.drawText("MUSIC THEORY", headerBounds, juce::Justification::centred);
    
    // Accent line - Focus Blue
    g.setColour(juce::Colour(0xFF3B82F6).withAlpha(0.6f));  // Focus Blue #3B82F6
    auto accentLine = juce::Rectangle<float>(0.0f, headerBounds.getBottom() - 2.0f, 
                                             headerBounds.getWidth(), 2.0f);
    g.fillRect(accentLine);
}

void MusicTheoryPanel::resized() {
    auto bounds = getLocalBounds().reduced(15);  // More padding
    bounds.removeFromTop(40);  // Title space (slightly more)
    
    // Key, Scale, Mode row - larger spacing
    auto theoryRow = bounds.removeFromTop(60);  // More height
    auto third = theoryRow.getWidth() / 3;
    
    keyLabel_.setBounds(theoryRow.removeFromLeft(third).removeFromTop(20));  // Larger label
    keySelector_.setBounds(theoryRow.removeFromLeft(third).removeFromTop(28).translated(0, 22));  // Larger selector
    
    scaleLabel_.setBounds(theoryRow.removeFromLeft(third).removeFromTop(20));
    scaleSelector_.setBounds(theoryRow.removeFromLeft(third).removeFromTop(28).translated(0, 22));
    
    modeLabel_.setBounds(theoryRow.removeFromTop(20));
    modeSelector_.setBounds(theoryRow.removeFromTop(28));
    
    bounds.removeFromTop(15);  // More spacing
    
    // Time Signature and Tempo row - larger spacing
    auto timeRow = bounds.removeFromTop(60);  // More height for tempo slider with value box
    auto half = timeRow.getWidth() / 2;
    
    timeSigLabel_.setBounds(timeRow.removeFromLeft(half).removeFromTop(20));
    auto timeSigArea = timeRow.removeFromLeft(half).removeFromTop(28).translated(0, 22);
    timeSigNumSelector_.setBounds(timeSigArea.removeFromLeft(70));  // Larger
    timeSigDenSelector_.setBounds(timeSigArea.removeFromLeft(70).translated(10, 0));  // Larger
    
    tempoLabel_.setBounds(timeRow.removeFromTop(20));
    tempoSlider_.setBounds(timeRow.removeFromTop(50));  // More height for value box above
    
    bounds.removeFromTop(20);  // More spacing
    
    // Instruments section - larger spacing
    auto instHeight = 70;  // More height per instrument
    
    // Lead
    auto leadArea = bounds.removeFromTop(instHeight);
    leadInstrumentLabel_.setBounds(leadArea.removeFromTop(20));  // Larger label
    auto leadControls = leadArea.removeFromTop(28);  // Larger controls
    leadInstrumentSelector_.setBounds(leadControls.removeFromLeft(static_cast<int>(leadControls.getWidth() * 0.6f)));
    leadTechniqueLabel_.setBounds(leadControls.removeFromTop(18).translated(8, 0));  // Larger, more spacing
    leadTechniqueSelector_.setBounds(leadControls.translated(8, 0));
    
    bounds.removeFromTop(8);  // More spacing between instruments
    
    // Harmony
    auto harmonyArea = bounds.removeFromTop(instHeight);
    harmonyInstrumentLabel_.setBounds(harmonyArea.removeFromTop(20));
    auto harmonyControls = harmonyArea.removeFromTop(28);
    harmonyInstrumentSelector_.setBounds(harmonyControls.removeFromLeft(static_cast<int>(harmonyControls.getWidth() * 0.6f)));
    harmonyTechniqueLabel_.setBounds(harmonyControls.removeFromTop(18).translated(8, 0));
    harmonyTechniqueSelector_.setBounds(harmonyControls.translated(8, 0));
    
    bounds.removeFromTop(8);
    
    // Bass
    auto bassArea = bounds.removeFromTop(instHeight);
    bassInstrumentLabel_.setBounds(bassArea.removeFromTop(20));
    auto bassControls = bassArea.removeFromTop(28);
    bassInstrumentSelector_.setBounds(bassControls.removeFromLeft(static_cast<int>(bassControls.getWidth() * 0.6f)));
    bassTechniqueLabel_.setBounds(bassControls.removeFromTop(18).translated(8, 0));
    bassTechniqueSelector_.setBounds(bassControls.translated(8, 0));
    
    bounds.removeFromTop(8);
    
    // Texture
    auto textureArea = bounds.removeFromTop(instHeight);
    textureInstrumentLabel_.setBounds(textureArea.removeFromTop(20));
    auto textureControls = textureArea.removeFromTop(28);
    textureInstrumentSelector_.setBounds(textureControls.removeFromLeft(static_cast<int>(textureControls.getWidth() * 0.6f)));
    textureTechniqueLabel_.setBounds(textureControls.removeFromTop(18).translated(8, 0));
    textureTechniqueSelector_.setBounds(textureControls.translated(8, 0));
    
    bounds.removeFromTop(20);  // More spacing
    
    // Effects section - larger spacing
    auto effectsHeight = 60;  // More height for slider with value box
    
    // Reverb
    auto reverbArea = bounds.removeFromTop(effectsHeight);
    reverbLabel_.setBounds(reverbArea.removeFromTop(20));  // Larger label
    reverbToggle_.setBounds(reverbArea.removeFromLeft(90).removeFromTop(22));  // Larger toggle
    reverbSlider_.setBounds(reverbArea.removeFromTop(50));  // More height for value box above
    
    bounds.removeFromTop(8);  // More spacing
    
    // Delay
    auto delayArea = bounds.removeFromTop(effectsHeight);
    delayLabel_.setBounds(delayArea.removeFromTop(20));
    delayToggle_.setBounds(delayArea.removeFromLeft(90).removeFromTop(22));
    delaySlider_.setBounds(delayArea.removeFromTop(50));
    
    bounds.removeFromTop(8);
    
    // Chorus
    auto chorusArea = bounds.removeFromTop(effectsHeight);
    chorusLabel_.setBounds(chorusArea.removeFromTop(20));
    chorusToggle_.setBounds(chorusArea.removeFromLeft(90).removeFromTop(22));
    chorusSlider_.setBounds(chorusArea.removeFromTop(50));
    
    bounds.removeFromTop(15);
    
    // EQ Section
    auto eqArea = bounds.removeFromTop(80);
    eqLabel_.setBounds(eqArea.removeFromTop(20));
    
    auto eqControls = eqArea.removeFromTop(28);
    eqToggle_.setBounds(eqControls.removeFromLeft(100).removeFromTop(22));
    eqAutoApplyLabel_.setBounds(eqControls.removeFromLeft(80).removeFromTop(20).translated(10, 0));
    eqAutoApplyToggle_.setBounds(eqControls.removeFromLeft(24).removeFromTop(22).translated(10, 0));
    
    eqPresetLabel_.setBounds(eqArea.removeFromTop(20));
    eqPresetSelector_.setBounds(eqArea.removeFromTop(28));
    
    bounds.removeFromTop(15);
    
    // Sheet Music Options
    notationToggle_.setBounds(bounds.removeFromTop(25));
    bounds.removeFromTop(5);
    
    notationStyleLabel_.setBounds(bounds.removeFromTop(18));
    notationStyleSelector_.setBounds(bounds.removeFromTop(25));
    bounds.removeFromTop(5);
    
    chordSymbolsToggle_.setBounds(bounds.removeFromTop(25));
    romanNumeralsToggle_.setBounds(bounds.removeFromTop(25));
}

MusicTheoryPanel::TheorySettings MusicTheoryPanel::getSettings() const {
    TheorySettings settings;
    
    settings.key = keySelector_.getText();
    settings.scale = scaleSelector_.getText();
    settings.mode = modeSelector_.getText();
    
    settings.timeSigNumerator = timeSigNumSelector_.getSelectedId();
    settings.timeSigDenominator = timeSigDenSelector_.getSelectedId() == 1 ? 2 : 
                                  (timeSigDenSelector_.getSelectedId() == 2 ? 4 : 
                                  (timeSigDenSelector_.getSelectedId() == 3 ? 8 : 16));
    
    settings.tempoBpm = static_cast<int>(tempoSlider_.getValue());
    
    settings.leadInstrument = leadInstrumentSelector_.getSelectedId() - 1;
    settings.harmonyInstrument = harmonyInstrumentSelector_.getSelectedId() - 1;
    settings.bassInstrument = bassInstrumentSelector_.getSelectedId() - 1;
    settings.textureInstrument = textureInstrumentSelector_.getSelectedId() - 1;
    
    settings.leadTechnique = leadTechniqueSelector_.getText();
    settings.harmonyTechnique = harmonyTechniqueSelector_.getText();
    settings.bassTechnique = bassTechniqueSelector_.getText();
    settings.textureTechnique = textureTechniqueSelector_.getText();
    
    settings.useReverb = reverbToggle_.getToggleState();
    settings.reverbAmount = static_cast<float>(reverbSlider_.getValue());
    settings.useDelay = delayToggle_.getToggleState();
    settings.delayAmount = static_cast<float>(delaySlider_.getValue());
    settings.useChorus = chorusToggle_.getToggleState();
    settings.chorusAmount = static_cast<float>(chorusSlider_.getValue());
    
    settings.showNotation = notationToggle_.getToggleState();
    settings.notationStyle = notationStyleSelector_.getText();
    settings.showChordSymbols = chordSymbolsToggle_.getToggleState();
    settings.showRomanNumerals = romanNumeralsToggle_.getToggleState();
    
    // Custom progression
    settings.useCustomProgression = !storedCustomProgression_.isEmpty();
    settings.customProgression = storedCustomProgression_;
    settings.strictCustomProgression = storedStrictMode_;
    
    // EQ settings
    settings.useEQ = eqToggle_.getToggleState();
    settings.eqPreset = getCurrentEQPreset();
    settings.eqAutoApply = eqAutoApplyToggle_.getToggleState();
    
    return settings;
}

void MusicTheoryPanel::setCustomProgression(const juce::String& progression, bool strict) {
    storedCustomProgression_ = progression.trim();
    storedStrictMode_ = strict;
    notifySettingsChanged();
}

std::vector<int> MusicTheoryPanel::parseProgressionString(const juce::String& progression) const {
    std::vector<int> degrees;
    if (progression.isEmpty()) return degrees;
    
    // Map of Roman numerals to scale degrees
    static const std::map<juce::String, int> romanToDegree = {
        {"I", 1}, {"i", 1}, {"1", 1},
        {"II", 2}, {"ii", 2}, {"2", 2},
        {"III", 3}, {"iii", 3}, {"3", 3},
        {"IV", 4}, {"iv", 4}, {"4", 4},
        {"V", 5}, {"v", 5}, {"5", 5},
        {"VI", 6}, {"vi", 6}, {"6", 6},
        {"VII", 7}, {"vii", 7}, {"7", 7},
        // Flats
        {"bII", 2}, {"bii", 2}, {"b2", 2},
        {"bIII", 3}, {"biii", 3}, {"b3", 3},
        {"bVI", 6}, {"bvi", 6}, {"b6", 6},
        {"bVII", 7}, {"bvii", 7}, {"b7", 7},
        // Sharps (less common but possible)
        {"#IV", 4}, {"#iv", 4}, {"#4", 4},
        {"#V", 5}, {"#v", 5}, {"#5", 5}
    };
    
    // Split by comma
    juce::StringArray tokens;
    tokens.addTokens(progression, ",", "");
    
    for (const auto& token : tokens) {
        juce::String trimmed = token.trim();
        if (trimmed.isEmpty()) continue;
        
        // Try to find in map
        auto it = romanToDegree.find(trimmed);
        if (it != romanToDegree.end()) {
            degrees.push_back(it->second);
        } else {
            // Try to parse as integer
            int degree = trimmed.getIntValue();
            if (degree >= 1 && degree <= 7) {
                degrees.push_back(degree);
            }
        }
    }
    
    return degrees;
}

void MusicTheoryPanel::notifySettingsChanged() {
    if (onSettingsChanged) {
        onSettingsChanged();
    }
}

void MusicTheoryPanel::setKey(const juce::String& key) {
    // Find key in selector (don't trigger callback)
    for (int i = 1; i <= keySelector_.getNumItems(); ++i) {
        if (keySelector_.getItemText(i).equalsIgnoreCase(key)) {
            keySelector_.setSelectedId(i, juce::dontSendNotification);
            return;
        }
    }
    // If not found, try to set by text directly
    keySelector_.setText(key, juce::dontSendNotification);
    // Note: don't call notifySettingsChanged() - this is programmatic update
}

void MusicTheoryPanel::setMode(const juce::String& mode) {
    // Map mode names to selector items (don't trigger callback)
    juce::String modeLower = mode.toLowerCase();
    
    // Try mode selector first
    for (int i = 1; i <= modeSelector_.getNumItems(); ++i) {
        if (modeSelector_.getItemText(i).equalsIgnoreCase(mode)) {
            modeSelector_.setSelectedId(i, juce::dontSendNotification);
            return;
        }
    }
    
    // Also try scale selector (mode might be in scale)
    for (int i = 1; i <= scaleSelector_.getNumItems(); ++i) {
        if (scaleSelector_.getItemText(i).equalsIgnoreCase(mode)) {
            scaleSelector_.setSelectedId(i, juce::dontSendNotification);
            return;
        }
    }
    
    // Map common mode names
    if (modeLower == "major" || modeLower == "ionian") {
        modeSelector_.setSelectedId(1, juce::dontSendNotification);  // Ionian
        scaleSelector_.setSelectedId(1, juce::dontSendNotification);  // Major
    } else if (modeLower == "minor" || modeLower == "aeolian") {
        modeSelector_.setSelectedId(6, juce::dontSendNotification);  // Aeolian
        scaleSelector_.setSelectedId(2, juce::dontSendNotification);  // Minor
    } else if (modeLower == "dorian") {
        modeSelector_.setSelectedId(2, juce::dontSendNotification);
    } else if (modeLower == "phrygian") {
        modeSelector_.setSelectedId(3, juce::dontSendNotification);
    } else if (modeLower == "lydian") {
        modeSelector_.setSelectedId(4, juce::dontSendNotification);
    } else if (modeLower == "mixolydian") {
        modeSelector_.setSelectedId(5, juce::dontSendNotification);
    } else if (modeLower == "locrian") {
        modeSelector_.setSelectedId(7, juce::dontSendNotification);
    }
    // Note: don't call notifySettingsChanged() - this is programmatic update
}

void MusicTheoryPanel::setTempo(int tempo) {
    tempoSlider_.setValue(juce::jlimit(40.0, 200.0, static_cast<double>(tempo)), juce::dontSendNotification);
    // Note: don't call notifySettingsChanged() - this is programmatic update
}

void MusicTheoryPanel::setProgression(const std::vector<juce::String>& progression) {
    // Convert vector to comma-separated string
    juce::String progString;
    for (size_t i = 0; i < progression.size(); ++i) {
        if (i > 0) progString += ",";
        progString += progression[i];
    }
    // Store without triggering callback (programmatic update)
    storedCustomProgression_ = progString.trim();
    storedStrictMode_ = false;
    // Note: don't call notifySettingsChanged() - this is programmatic update
}

void MusicTheoryPanel::setEQPreset(const juce::String& presetName) {
    // Find preset in selector
    for (int i = 1; i <= eqPresetSelector_.getNumItems(); ++i) {
        if (eqPresetSelector_.getItemText(i).equalsIgnoreCase(presetName)) {
            eqPresetSelector_.setSelectedId(i, juce::dontSendNotification);
            return;
        }
    }
    // If not found, try "Auto" as fallback
    eqPresetSelector_.setSelectedId(1, juce::dontSendNotification);
}

void MusicTheoryPanel::setEQAutoApply(bool autoApply) {
    eqAutoApplyToggle_.setToggleState(autoApply, juce::dontSendNotification);
}

juce::String MusicTheoryPanel::getCurrentEQPreset() const {
    int selectedId = eqPresetSelector_.getSelectedId();
    if (selectedId > 0 && selectedId <= eqPresetSelector_.getNumItems()) {
        return eqPresetSelector_.getItemText(selectedId);
    }
    return "Auto (from Emotion)";
}

} // namespace kelly
