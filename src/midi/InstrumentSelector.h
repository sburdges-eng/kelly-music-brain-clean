#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace kelly {

//=============================================================================
// GM MIDI INSTRUMENT NUMBERS
//=============================================================================

namespace GM {
    // Piano (0-7)
    constexpr int ACOUSTIC_GRAND_PIANO = 0;
    constexpr int BRIGHT_ACOUSTIC_PIANO = 1;
    constexpr int ELECTRIC_GRAND_PIANO = 2;
    constexpr int HONKY_TONK_PIANO = 3;
    constexpr int ELECTRIC_PIANO_1 = 4;
    constexpr int ELECTRIC_PIANO_2 = 5;
    constexpr int HARPSICHORD = 6;
    constexpr int CLAVINET = 7;
    
    // Chromatic Percussion (8-15)
    constexpr int CELESTA = 8;
    constexpr int GLOCKENSPIEL = 9;
    constexpr int MUSIC_BOX = 10;
    constexpr int VIBRAPHONE = 11;
    constexpr int MARIMBA = 12;
    constexpr int XYLOPHONE = 13;
    constexpr int TUBULAR_BELLS = 14;
    constexpr int DULCIMER = 15;
    
    // Organ (16-23)
    constexpr int DRAWBAR_ORGAN = 16;
    constexpr int PERCUSSIVE_ORGAN = 17;
    constexpr int ROCK_ORGAN = 18;
    constexpr int CHURCH_ORGAN = 19;
    constexpr int REED_ORGAN = 20;
    constexpr int ACCORDION = 21;
    constexpr int HARMONICA = 22;
    constexpr int TANGO_ACCORDION = 23;
    
    // Guitar (24-31)
    constexpr int ACOUSTIC_GUITAR_NYLON = 24;
    constexpr int ACOUSTIC_GUITAR_STEEL = 25;
    constexpr int ELECTRIC_GUITAR_JAZZ = 26;
    constexpr int ELECTRIC_GUITAR_CLEAN = 27;
    constexpr int ELECTRIC_GUITAR_MUTED = 28;
    constexpr int OVERDRIVEN_GUITAR = 29;
    constexpr int DISTORTION_GUITAR = 30;
    constexpr int GUITAR_HARMONICS = 31;
    
    // Bass (32-39)
    constexpr int ACOUSTIC_BASS = 32;
    constexpr int ELECTRIC_BASS_FINGER = 33;
    constexpr int ELECTRIC_BASS_PICK = 34;
    constexpr int FRETLESS_BASS = 35;
    constexpr int SLAP_BASS_1 = 36;
    constexpr int SLAP_BASS_2 = 37;
    constexpr int SYNTH_BASS_1 = 38;
    constexpr int SYNTH_BASS_2 = 39;
    
    // Strings (40-47)
    constexpr int VIOLIN = 40;
    constexpr int VIOLA = 41;
    constexpr int CELLO = 42;
    constexpr int CONTRABASS = 43;
    constexpr int TREMOLO_STRINGS = 44;
    constexpr int PIZZICATO_STRINGS = 45;
    constexpr int ORCHESTRAL_HARP = 46;
    constexpr int TIMPANI = 47;
    
    // Ensemble (48-55)
    constexpr int STRING_ENSEMBLE_1 = 48;
    constexpr int STRING_ENSEMBLE_2 = 49;
    constexpr int SYNTH_STRINGS_1 = 50;
    constexpr int SYNTH_STRINGS_2 = 51;
    constexpr int CHOIR_AAHS = 52;
    constexpr int VOICE_OOHS = 53;
    constexpr int SYNTH_VOICE = 54;
    constexpr int ORCHESTRA_HIT = 55;
    
    // Brass (56-63)
    constexpr int TRUMPET = 56;
    constexpr int TROMBONE = 57;
    constexpr int TUBA = 58;
    constexpr int MUTED_TRUMPET = 59;
    constexpr int FRENCH_HORN = 60;
    constexpr int BRASS_SECTION = 61;
    constexpr int SYNTH_BRASS_1 = 62;
    constexpr int SYNTH_BRASS_2 = 63;
    
    // Reed (64-71)
    constexpr int SOPRANO_SAX = 64;
    constexpr int ALTO_SAX = 65;
    constexpr int TENOR_SAX = 66;
    constexpr int BARITONE_SAX = 67;
    constexpr int OBOE = 68;
    constexpr int ENGLISH_HORN = 69;
    constexpr int BASSOON = 70;
    constexpr int CLARINET = 71;
    
    // Pipe (72-79)
    constexpr int PICCOLO = 72;
    constexpr int FLUTE = 73;
    constexpr int RECORDER = 74;
    constexpr int PAN_FLUTE = 75;
    constexpr int BLOWN_BOTTLE = 76;
    constexpr int SHAKUHACHI = 77;
    constexpr int WHISTLE = 78;
    constexpr int OCARINA = 79;
    
    // Synth Lead (80-87)
    constexpr int LEAD_SQUARE = 80;
    constexpr int LEAD_SAWTOOTH = 81;
    constexpr int LEAD_CALLIOPE = 82;
    constexpr int LEAD_CHIFF = 83;
    constexpr int LEAD_CHARANG = 84;
    constexpr int LEAD_VOICE = 85;
    constexpr int LEAD_FIFTHS = 86;
    constexpr int LEAD_BASS_LEAD = 87;
    
    // Synth Pad (88-95)
    constexpr int PAD_NEW_AGE = 88;
    constexpr int PAD_WARM = 89;
    constexpr int PAD_POLYSYNTH = 90;
    constexpr int PAD_CHOIR = 91;
    constexpr int PAD_BOWED = 92;
    constexpr int PAD_METALLIC = 93;
    constexpr int PAD_HALO = 94;
    constexpr int PAD_SWEEP = 95;
    
    // Synth Effects (96-103)
    constexpr int FX_RAIN = 96;
    constexpr int FX_SOUNDTRACK = 97;
    constexpr int FX_CRYSTAL = 98;
    constexpr int FX_ATMOSPHERE = 99;
    constexpr int FX_BRIGHTNESS = 100;
    constexpr int FX_GOBLINS = 101;
    constexpr int FX_ECHOES = 102;
    constexpr int FX_SCI_FI = 103;
    
    // Ethnic (104-111)
    constexpr int SITAR = 104;
    constexpr int BANJO = 105;
    constexpr int SHAMISEN = 106;
    constexpr int KOTO = 107;
    constexpr int KALIMBA = 108;
    constexpr int BAGPIPE = 109;
    constexpr int FIDDLE = 110;
    constexpr int SHANAI = 111;
    
    // Percussive (112-119)
    constexpr int TINKLE_BELL = 112;
    constexpr int AGOGO = 113;
    constexpr int STEEL_DRUMS = 114;
    constexpr int WOODBLOCK = 115;
    constexpr int TAIKO_DRUM = 116;
    constexpr int MELODIC_TOM = 117;
    constexpr int SYNTH_DRUM = 118;
    constexpr int REVERSE_CYMBAL = 119;
    
    // Sound Effects (120-127)
    constexpr int GUITAR_FRET_NOISE = 120;
    constexpr int BREATH_NOISE = 121;
    constexpr int SEASHORE = 122;
    constexpr int BIRD_TWEET = 123;
    constexpr int TELEPHONE_RING = 124;
    constexpr int HELICOPTER = 125;
    constexpr int APPLAUSE = 126;
    constexpr int GUNSHOT = 127;
    
    // GM Drum Map (Channel 10) - MIDI note numbers
    namespace Drum {
        constexpr int KICK = 36;
        constexpr int SNARE = 38;
        constexpr int SIDESTICK = 37;
        constexpr int CLAP = 39;
        constexpr int CLOSED_HAT = 42;
        constexpr int OPEN_HAT = 46;
        constexpr int PEDAL_HAT = 44;
        constexpr int CRASH_1 = 49;
        constexpr int CRASH_2 = 57;
        constexpr int RIDE = 51;
        constexpr int RIDE_BELL = 53;
        constexpr int LOW_TOM = 45;
        constexpr int MID_TOM = 47;
        constexpr int HIGH_TOM = 50;
        constexpr int TAMBOURINE = 54;
        constexpr int COWBELL = 56;
        constexpr int SHAKER = 70;
        constexpr int LOW_CONGA = 64;
        constexpr int HIGH_CONGA = 63;
        constexpr int LOW_BONGO = 61;
        constexpr int HIGH_BONGO = 60;
    }
}

//=============================================================================
// INSTRUMENT PROFILE
//=============================================================================

struct InstrumentProfile {
    int gmProgram;
    std::string name;
    std::string family;
    
    // Emotional characteristics (0.0 to 1.0)
    float vulnerability;    // Exposed, intimate sound
    float intimacy;         // Close, personal feel
    float emotionalWeight;  // Gravitas, heaviness
    float brightness;       // Tonal brightness
    float aggression;       // Attack, edge
    float warmth;           // Soft, rounded tone
    
    // Musical role suitability (0.0 to 1.0)
    float leadSuitability;
    float harmonySuitability;
    float bassSuitability;
    float textureSuitability;
    float accentSuitability;
    
    // Range info
    int lowNote;
    int highNote;
};

//=============================================================================
// INSTRUMENT PALETTE
//=============================================================================

struct InstrumentPalette {
    int lead;
    int harmony;
    int bass;
    int texture;
    int accent;
    
    std::string leadName;
    std::string harmonyName;
    std::string bassName;
    std::string textureName;
    std::string accentName;
};

//=============================================================================
// INSTRUMENT RECOMMENDATION
//=============================================================================

struct InstrumentRecommendation {
    int gmProgram;
    std::string name;
    float score;
    std::string reason;
};

//=============================================================================
// INSTRUMENT SELECTOR
//=============================================================================

class InstrumentSelector {
public:
    InstrumentSelector();
    
    /**
     * Get full palette of instruments for an emotion.
     */
    InstrumentPalette getPaletteForEmotion(const std::string& emotion) const;
    
    /**
     * Get palette for emotion node.
     */
    InstrumentPalette getPaletteForEmotion(const EmotionNode& emotion) const;
    
    /**
     * Recommend instruments with scoring.
     * @param emotion Emotion name or node
     * @param vulnerabilityLevel 0.0-1.0, how vulnerable the expression should be
     * @param intimacyPreference 0.0-1.0, how intimate/close
     * @param genre Optional genre context
     * @param count Number of recommendations to return
     */
    std::vector<InstrumentRecommendation> recommend(
        const std::string& emotion,
        float vulnerabilityLevel = 0.5f,
        float intimacyPreference = 0.5f,
        const std::string& genre = "",
        int count = 5
    ) const;
    
    /**
     * Get lead instrument for emotion.
     */
    int getLeadInstrument(const std::string& emotion) const;
    
    /**
     * Get bass instrument for emotion.
     */
    int getBassInstrument(const std::string& emotion) const;
    
    /**
     * Get texture/pad instrument for emotion.
     */
    int getTextureInstrument(const std::string& emotion) const;
    
    /**
     * Get instrument profile by GM program number.
     */
    const InstrumentProfile& getProfile(int gmProgram) const;
    
    /**
     * Get instrument name by GM program number.
     */
    std::string getInstrumentName(int gmProgram) const;
    
    /**
     * Score an instrument for an emotion.
     */
    float scoreInstrumentForEmotion(
        int gmProgram,
        const std::string& emotion,
        float vulnerabilityLevel = 0.5f,
        float intimacyPreference = 0.5f
    ) const;
    
    /**
     * Get all instruments in a family.
     */
    std::vector<int> getInstrumentsByFamily(const std::string& family) const;

private:
    std::unordered_map<int, InstrumentProfile> profiles_;
    std::unordered_map<std::string, InstrumentPalette> emotionPalettes_;
    
    void initializeProfiles();
    void initializeEmotionPalettes();
    
    // Emotion characteristic extraction
    struct EmotionCharacteristics {
        float valence;
        float arousal;
        float intensity;
        bool needsVulnerability;
        bool needsAggression;
        bool needsWarmth;
    };
    
    EmotionCharacteristics getEmotionCharacteristics(const std::string& emotion) const;
};

//=============================================================================
// CONVENIENCE FUNCTIONS
//=============================================================================

/** Get recommended lead instrument for emotion */
int recommendLeadInstrument(const std::string& emotion);

/** Get recommended bass instrument for emotion */
int recommendBassInstrument(const std::string& emotion);

/** Get complete palette for emotion */
InstrumentPalette getEmotionPalette(const std::string& emotion);

} // namespace kelly
