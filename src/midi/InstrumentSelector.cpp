#include "InstrumentSelector.h"
#include <algorithm>
#include <cmath>

namespace kelly {

InstrumentSelector::InstrumentSelector() {
    initializeProfiles();
    initializeEmotionPalettes();
}

void InstrumentSelector::initializeProfiles() {
    // Format: gmProgram, name, family, vulnerability, intimacy, weight, brightness, aggression, warmth,
    //         leadSuit, harmonySuit, bassSuit, textureSuit, accentSuit, lowNote, highNote
    
    //=========================================================================
    // PIANO (0-7)
    //=========================================================================
    profiles_[GM::ACOUSTIC_GRAND_PIANO] = {
        GM::ACOUSTIC_GRAND_PIANO, "Acoustic Grand Piano", "piano",
        1.0f, 0.9f, 0.8f, 0.6f, 0.2f, 0.7f,
        0.9f, 0.9f, 0.3f, 0.7f, 0.6f, 21, 108
    };
    profiles_[GM::BRIGHT_ACOUSTIC_PIANO] = {
        GM::BRIGHT_ACOUSTIC_PIANO, "Bright Acoustic Piano", "piano",
        0.8f, 0.8f, 0.7f, 0.8f, 0.3f, 0.5f,
        0.9f, 0.8f, 0.3f, 0.6f, 0.7f, 21, 108
    };
    profiles_[GM::ELECTRIC_GRAND_PIANO] = {
        GM::ELECTRIC_GRAND_PIANO, "Electric Grand Piano", "piano",
        0.7f, 0.7f, 0.6f, 0.7f, 0.2f, 0.6f,
        0.8f, 0.8f, 0.2f, 0.7f, 0.5f, 21, 108
    };
    profiles_[GM::HONKY_TONK_PIANO] = {
        GM::HONKY_TONK_PIANO, "Honky-tonk Piano", "piano",
        0.4f, 0.5f, 0.5f, 0.7f, 0.4f, 0.4f,
        0.7f, 0.6f, 0.2f, 0.4f, 0.6f, 21, 108
    };
    profiles_[GM::ELECTRIC_PIANO_1] = {
        GM::ELECTRIC_PIANO_1, "Electric Piano 1 (Rhodes)", "piano",
        0.8f, 0.85f, 0.5f, 0.5f, 0.1f, 0.9f,
        0.85f, 0.9f, 0.1f, 0.8f, 0.4f, 28, 103
    };
    profiles_[GM::ELECTRIC_PIANO_2] = {
        GM::ELECTRIC_PIANO_2, "Electric Piano 2 (DX)", "piano",
        0.6f, 0.7f, 0.4f, 0.7f, 0.2f, 0.6f,
        0.8f, 0.85f, 0.1f, 0.75f, 0.5f, 28, 103
    };
    profiles_[GM::HARPSICHORD] = {
        GM::HARPSICHORD, "Harpsichord", "piano",
        0.3f, 0.4f, 0.4f, 0.9f, 0.5f, 0.2f,
        0.7f, 0.7f, 0.1f, 0.5f, 0.7f, 29, 89
    };
    profiles_[GM::CLAVINET] = {
        GM::CLAVINET, "Clavinet", "piano",
        0.3f, 0.5f, 0.4f, 0.8f, 0.6f, 0.3f,
        0.75f, 0.6f, 0.2f, 0.4f, 0.8f, 36, 96
    };
    
    //=========================================================================
    // CHROMATIC PERCUSSION (8-15)
    //=========================================================================
    profiles_[GM::CELESTA] = {
        GM::CELESTA, "Celesta", "chromatic_perc",
        0.6f, 0.5f, 0.2f, 0.95f, 0.1f, 0.4f,
        0.7f, 0.5f, 0.0f, 0.8f, 0.9f, 60, 108
    };
    profiles_[GM::GLOCKENSPIEL] = {
        GM::GLOCKENSPIEL, "Glockenspiel", "chromatic_perc",
        0.5f, 0.4f, 0.2f, 1.0f, 0.2f, 0.2f,
        0.6f, 0.4f, 0.0f, 0.7f, 0.95f, 72, 108
    };
    profiles_[GM::MUSIC_BOX] = {
        GM::MUSIC_BOX, "Music Box", "chromatic_perc",
        0.7f, 0.6f, 0.3f, 0.9f, 0.0f, 0.5f,
        0.8f, 0.5f, 0.0f, 0.85f, 0.7f, 60, 96
    };
    profiles_[GM::VIBRAPHONE] = {
        GM::VIBRAPHONE, "Vibraphone", "chromatic_perc",
        0.7f, 0.7f, 0.4f, 0.7f, 0.1f, 0.8f,
        0.85f, 0.7f, 0.0f, 0.8f, 0.6f, 53, 89
    };
    profiles_[GM::MARIMBA] = {
        GM::MARIMBA, "Marimba", "chromatic_perc",
        0.5f, 0.6f, 0.5f, 0.6f, 0.2f, 0.7f,
        0.8f, 0.6f, 0.3f, 0.6f, 0.7f, 45, 96
    };
    profiles_[GM::XYLOPHONE] = {
        GM::XYLOPHONE, "Xylophone", "chromatic_perc",
        0.3f, 0.4f, 0.3f, 0.95f, 0.4f, 0.2f,
        0.7f, 0.4f, 0.0f, 0.5f, 0.85f, 65, 108
    };
    profiles_[GM::TUBULAR_BELLS] = {
        GM::TUBULAR_BELLS, "Tubular Bells", "chromatic_perc",
        0.6f, 0.4f, 0.7f, 0.8f, 0.3f, 0.4f,
        0.5f, 0.5f, 0.0f, 0.7f, 0.9f, 60, 77
    };
    profiles_[GM::DULCIMER] = {
        GM::DULCIMER, "Dulcimer", "chromatic_perc",
        0.6f, 0.7f, 0.3f, 0.7f, 0.1f, 0.6f,
        0.75f, 0.7f, 0.0f, 0.7f, 0.5f, 36, 96
    };
    
    //=========================================================================
    // ORGAN (16-23)
    //=========================================================================
    profiles_[GM::DRAWBAR_ORGAN] = {
        GM::DRAWBAR_ORGAN, "Drawbar Organ", "organ",
        0.4f, 0.5f, 0.6f, 0.6f, 0.3f, 0.7f,
        0.7f, 0.85f, 0.4f, 0.8f, 0.4f, 36, 96
    };
    profiles_[GM::PERCUSSIVE_ORGAN] = {
        GM::PERCUSSIVE_ORGAN, "Percussive Organ", "organ",
        0.3f, 0.4f, 0.5f, 0.7f, 0.5f, 0.5f,
        0.75f, 0.8f, 0.3f, 0.6f, 0.6f, 36, 96
    };
    profiles_[GM::ROCK_ORGAN] = {
        GM::ROCK_ORGAN, "Rock Organ", "organ",
        0.2f, 0.3f, 0.7f, 0.7f, 0.7f, 0.4f,
        0.7f, 0.8f, 0.4f, 0.7f, 0.5f, 36, 96
    };
    profiles_[GM::CHURCH_ORGAN] = {
        GM::CHURCH_ORGAN, "Church Organ", "organ",
        0.5f, 0.3f, 0.95f, 0.5f, 0.2f, 0.6f,
        0.6f, 0.9f, 0.6f, 0.9f, 0.3f, 21, 108
    };
    profiles_[GM::REED_ORGAN] = {
        GM::REED_ORGAN, "Reed Organ", "organ",
        0.6f, 0.6f, 0.5f, 0.4f, 0.1f, 0.7f,
        0.6f, 0.8f, 0.3f, 0.8f, 0.3f, 36, 96
    };
    profiles_[GM::ACCORDION] = {
        GM::ACCORDION, "Accordion", "organ",
        0.6f, 0.7f, 0.4f, 0.5f, 0.2f, 0.6f,
        0.8f, 0.7f, 0.2f, 0.6f, 0.5f, 53, 89
    };
    profiles_[GM::HARMONICA] = {
        GM::HARMONICA, "Harmonica", "organ",
        0.85f, 0.9f, 0.3f, 0.6f, 0.2f, 0.7f,
        0.9f, 0.5f, 0.0f, 0.5f, 0.7f, 60, 84
    };
    profiles_[GM::TANGO_ACCORDION] = {
        GM::TANGO_ACCORDION, "Tango Accordion", "organ",
        0.7f, 0.75f, 0.5f, 0.5f, 0.3f, 0.6f,
        0.85f, 0.7f, 0.2f, 0.6f, 0.6f, 53, 89
    };
    
    //=========================================================================
    // GUITAR (24-31)
    //=========================================================================
    profiles_[GM::ACOUSTIC_GUITAR_NYLON] = {
        GM::ACOUSTIC_GUITAR_NYLON, "Acoustic Guitar (Nylon)", "guitar",
        0.9f, 1.0f, 0.4f, 0.5f, 0.1f, 0.9f,
        0.9f, 0.85f, 0.2f, 0.7f, 0.6f, 40, 84
    };
    profiles_[GM::ACOUSTIC_GUITAR_STEEL] = {
        GM::ACOUSTIC_GUITAR_STEEL, "Acoustic Guitar (Steel)", "guitar",
        0.75f, 0.85f, 0.45f, 0.7f, 0.25f, 0.7f,
        0.85f, 0.85f, 0.2f, 0.65f, 0.7f, 40, 84
    };
    profiles_[GM::ELECTRIC_GUITAR_JAZZ] = {
        GM::ELECTRIC_GUITAR_JAZZ, "Electric Guitar (Jazz)", "guitar",
        0.6f, 0.7f, 0.4f, 0.4f, 0.15f, 0.85f,
        0.85f, 0.8f, 0.1f, 0.7f, 0.5f, 40, 86
    };
    profiles_[GM::ELECTRIC_GUITAR_CLEAN] = {
        GM::ELECTRIC_GUITAR_CLEAN, "Electric Guitar (Clean)", "guitar",
        0.5f, 0.6f, 0.35f, 0.7f, 0.2f, 0.6f,
        0.8f, 0.75f, 0.1f, 0.65f, 0.65f, 40, 86
    };
    profiles_[GM::ELECTRIC_GUITAR_MUTED] = {
        GM::ELECTRIC_GUITAR_MUTED, "Electric Guitar (Muted)", "guitar",
        0.3f, 0.4f, 0.3f, 0.5f, 0.4f, 0.4f,
        0.6f, 0.7f, 0.1f, 0.5f, 0.8f, 40, 86
    };
    profiles_[GM::OVERDRIVEN_GUITAR] = {
        GM::OVERDRIVEN_GUITAR, "Overdriven Guitar", "guitar",
        0.2f, 0.3f, 0.7f, 0.7f, 0.8f, 0.3f,
        0.8f, 0.7f, 0.2f, 0.6f, 0.7f, 40, 86
    };
    profiles_[GM::DISTORTION_GUITAR] = {
        GM::DISTORTION_GUITAR, "Distortion Guitar", "guitar",
        0.1f, 0.2f, 0.85f, 0.75f, 0.95f, 0.2f,
        0.85f, 0.75f, 0.3f, 0.7f, 0.8f, 40, 86
    };
    profiles_[GM::GUITAR_HARMONICS] = {
        GM::GUITAR_HARMONICS, "Guitar Harmonics", "guitar",
        0.7f, 0.6f, 0.2f, 0.95f, 0.1f, 0.4f,
        0.6f, 0.5f, 0.0f, 0.85f, 0.7f, 52, 96
    };
    
    //=========================================================================
    // BASS (32-39)
    //=========================================================================
    profiles_[GM::ACOUSTIC_BASS] = {
        GM::ACOUSTIC_BASS, "Acoustic Bass", "bass",
        0.6f, 0.7f, 0.8f, 0.3f, 0.2f, 0.8f,
        0.2f, 0.3f, 1.0f, 0.4f, 0.5f, 28, 55
    };
    profiles_[GM::ELECTRIC_BASS_FINGER] = {
        GM::ELECTRIC_BASS_FINGER, "Electric Bass (Finger)", "bass",
        0.5f, 0.6f, 0.75f, 0.4f, 0.25f, 0.7f,
        0.15f, 0.25f, 0.95f, 0.35f, 0.5f, 28, 60
    };
    profiles_[GM::ELECTRIC_BASS_PICK] = {
        GM::ELECTRIC_BASS_PICK, "Electric Bass (Pick)", "bass",
        0.35f, 0.45f, 0.7f, 0.55f, 0.45f, 0.5f,
        0.2f, 0.3f, 0.95f, 0.3f, 0.6f, 28, 60
    };
    profiles_[GM::FRETLESS_BASS] = {
        GM::FRETLESS_BASS, "Fretless Bass", "bass",
        0.7f, 0.75f, 0.7f, 0.35f, 0.15f, 0.85f,
        0.3f, 0.35f, 0.95f, 0.5f, 0.4f, 28, 60
    };
    profiles_[GM::SLAP_BASS_1] = {
        GM::SLAP_BASS_1, "Slap Bass 1", "bass",
        0.2f, 0.3f, 0.65f, 0.7f, 0.7f, 0.35f,
        0.25f, 0.3f, 0.9f, 0.25f, 0.8f, 28, 60
    };
    profiles_[GM::SLAP_BASS_2] = {
        GM::SLAP_BASS_2, "Slap Bass 2", "bass",
        0.2f, 0.3f, 0.65f, 0.65f, 0.65f, 0.4f,
        0.25f, 0.3f, 0.9f, 0.25f, 0.75f, 28, 60
    };
    profiles_[GM::SYNTH_BASS_1] = {
        GM::SYNTH_BASS_1, "Synth Bass 1", "bass",
        0.15f, 0.2f, 0.8f, 0.6f, 0.6f, 0.3f,
        0.2f, 0.25f, 0.95f, 0.4f, 0.65f, 24, 60
    };
    profiles_[GM::SYNTH_BASS_2] = {
        GM::SYNTH_BASS_2, "Synth Bass 2", "bass",
        0.1f, 0.15f, 0.85f, 0.55f, 0.7f, 0.25f,
        0.15f, 0.2f, 0.95f, 0.35f, 0.7f, 24, 60
    };
    
    //=========================================================================
    // STRINGS (40-47)
    //=========================================================================
    profiles_[GM::VIOLIN] = {
        GM::VIOLIN, "Violin", "strings",
        0.9f, 0.85f, 0.6f, 0.75f, 0.3f, 0.6f,
        0.95f, 0.7f, 0.0f, 0.7f, 0.8f, 55, 103
    };
    profiles_[GM::VIOLA] = {
        GM::VIOLA, "Viola", "strings",
        0.85f, 0.8f, 0.65f, 0.55f, 0.25f, 0.7f,
        0.85f, 0.75f, 0.0f, 0.75f, 0.7f, 48, 91
    };
    profiles_[GM::CELLO] = {
        GM::CELLO, "Cello", "strings",
        1.0f, 0.9f, 0.85f, 0.45f, 0.2f, 0.85f,
        0.9f, 0.8f, 0.6f, 0.85f, 0.7f, 36, 76
    };
    profiles_[GM::CONTRABASS] = {
        GM::CONTRABASS, "Contrabass", "strings",
        0.7f, 0.65f, 0.95f, 0.25f, 0.15f, 0.8f,
        0.4f, 0.5f, 0.9f, 0.7f, 0.5f, 28, 55
    };
    profiles_[GM::TREMOLO_STRINGS] = {
        GM::TREMOLO_STRINGS, "Tremolo Strings", "strings",
        0.7f, 0.5f, 0.7f, 0.6f, 0.5f, 0.5f,
        0.6f, 0.7f, 0.3f, 0.85f, 0.75f, 36, 96
    };
    profiles_[GM::PIZZICATO_STRINGS] = {
        GM::PIZZICATO_STRINGS, "Pizzicato Strings", "strings",
        0.5f, 0.55f, 0.4f, 0.65f, 0.3f, 0.5f,
        0.6f, 0.6f, 0.4f, 0.6f, 0.85f, 36, 96
    };
    profiles_[GM::ORCHESTRAL_HARP] = {
        GM::ORCHESTRAL_HARP, "Orchestral Harp", "strings",
        0.8f, 0.75f, 0.5f, 0.8f, 0.1f, 0.7f,
        0.8f, 0.85f, 0.2f, 0.9f, 0.7f, 24, 103
    };
    profiles_[GM::TIMPANI] = {
        GM::TIMPANI, "Timpani", "strings",
        0.3f, 0.25f, 0.95f, 0.35f, 0.7f, 0.4f,
        0.2f, 0.3f, 0.5f, 0.5f, 0.95f, 36, 57
    };
    
    //=========================================================================
    // ENSEMBLE (48-55)
    //=========================================================================
    profiles_[GM::STRING_ENSEMBLE_1] = {
        GM::STRING_ENSEMBLE_1, "String Ensemble 1", "ensemble",
        0.85f, 0.7f, 0.8f, 0.55f, 0.2f, 0.75f,
        0.7f, 0.95f, 0.4f, 0.95f, 0.5f, 28, 103
    };
    profiles_[GM::STRING_ENSEMBLE_2] = {
        GM::STRING_ENSEMBLE_2, "String Ensemble 2", "ensemble",
        0.8f, 0.65f, 0.75f, 0.5f, 0.15f, 0.8f,
        0.65f, 0.9f, 0.35f, 0.9f, 0.45f, 28, 103
    };
    profiles_[GM::SYNTH_STRINGS_1] = {
        GM::SYNTH_STRINGS_1, "Synth Strings 1", "ensemble",
        0.5f, 0.45f, 0.6f, 0.55f, 0.15f, 0.65f,
        0.5f, 0.8f, 0.3f, 0.9f, 0.35f, 36, 96
    };
    profiles_[GM::SYNTH_STRINGS_2] = {
        GM::SYNTH_STRINGS_2, "Synth Strings 2", "ensemble",
        0.45f, 0.4f, 0.55f, 0.6f, 0.2f, 0.6f,
        0.45f, 0.75f, 0.25f, 0.85f, 0.4f, 36, 96
    };
    profiles_[GM::CHOIR_AAHS] = {
        GM::CHOIR_AAHS, "Choir Aahs", "ensemble",
        0.75f, 0.6f, 0.7f, 0.5f, 0.15f, 0.8f,
        0.6f, 0.85f, 0.2f, 0.95f, 0.4f, 48, 79
    };
    profiles_[GM::VOICE_OOHS] = {
        GM::VOICE_OOHS, "Voice Oohs", "ensemble",
        0.8f, 0.65f, 0.6f, 0.45f, 0.1f, 0.85f,
        0.55f, 0.8f, 0.15f, 0.9f, 0.35f, 48, 79
    };
    profiles_[GM::SYNTH_VOICE] = {
        GM::SYNTH_VOICE, "Synth Voice", "ensemble",
        0.5f, 0.4f, 0.5f, 0.6f, 0.2f, 0.6f,
        0.5f, 0.7f, 0.15f, 0.85f, 0.4f, 48, 84
    };
    profiles_[GM::ORCHESTRA_HIT] = {
        GM::ORCHESTRA_HIT, "Orchestra Hit", "ensemble",
        0.1f, 0.15f, 0.9f, 0.7f, 0.9f, 0.3f,
        0.3f, 0.4f, 0.2f, 0.3f, 1.0f, 48, 72
    };
    
    //=========================================================================
    // BRASS (56-63)
    //=========================================================================
    profiles_[GM::TRUMPET] = {
        GM::TRUMPET, "Trumpet", "brass",
        0.4f, 0.45f, 0.7f, 0.85f, 0.6f, 0.4f,
        0.9f, 0.6f, 0.0f, 0.5f, 0.85f, 52, 82
    };
    profiles_[GM::TROMBONE] = {
        GM::TROMBONE, "Trombone", "brass",
        0.45f, 0.4f, 0.8f, 0.6f, 0.55f, 0.5f,
        0.75f, 0.65f, 0.4f, 0.55f, 0.8f, 40, 72
    };
    profiles_[GM::TUBA] = {
        GM::TUBA, "Tuba", "brass",
        0.35f, 0.3f, 0.95f, 0.35f, 0.5f, 0.55f,
        0.4f, 0.5f, 0.85f, 0.5f, 0.7f, 29, 55
    };
    profiles_[GM::MUTED_TRUMPET] = {
        GM::MUTED_TRUMPET, "Muted Trumpet", "brass",
        0.6f, 0.65f, 0.5f, 0.55f, 0.35f, 0.5f,
        0.85f, 0.55f, 0.0f, 0.6f, 0.7f, 52, 82
    };
    profiles_[GM::FRENCH_HORN] = {
        GM::FRENCH_HORN, "French Horn", "brass",
        0.6f, 0.5f, 0.75f, 0.5f, 0.4f, 0.7f,
        0.7f, 0.8f, 0.3f, 0.75f, 0.65f, 34, 77
    };
    profiles_[GM::BRASS_SECTION] = {
        GM::BRASS_SECTION, "Brass Section", "brass",
        0.25f, 0.3f, 0.9f, 0.7f, 0.75f, 0.4f,
        0.65f, 0.85f, 0.3f, 0.7f, 0.9f, 36, 84
    };
    profiles_[GM::SYNTH_BRASS_1] = {
        GM::SYNTH_BRASS_1, "Synth Brass 1", "brass",
        0.2f, 0.25f, 0.75f, 0.75f, 0.7f, 0.35f,
        0.6f, 0.75f, 0.25f, 0.65f, 0.8f, 36, 84
    };
    profiles_[GM::SYNTH_BRASS_2] = {
        GM::SYNTH_BRASS_2, "Synth Brass 2", "brass",
        0.15f, 0.2f, 0.7f, 0.7f, 0.65f, 0.4f,
        0.55f, 0.7f, 0.2f, 0.6f, 0.75f, 36, 84
    };
    
    //=========================================================================
    // REED (64-71)
    //=========================================================================
    profiles_[GM::SOPRANO_SAX] = {
        GM::SOPRANO_SAX, "Soprano Sax", "reed",
        0.7f, 0.75f, 0.5f, 0.8f, 0.4f, 0.5f,
        0.9f, 0.55f, 0.0f, 0.55f, 0.75f, 56, 88
    };
    profiles_[GM::ALTO_SAX] = {
        GM::ALTO_SAX, "Alto Sax", "reed",
        0.75f, 0.8f, 0.55f, 0.65f, 0.35f, 0.6f,
        0.9f, 0.6f, 0.0f, 0.6f, 0.7f, 49, 81
    };
    profiles_[GM::TENOR_SAX] = {
        GM::TENOR_SAX, "Tenor Sax", "reed",
        0.8f, 0.85f, 0.65f, 0.55f, 0.4f, 0.7f,
        0.9f, 0.65f, 0.15f, 0.65f, 0.7f, 42, 75
    };
    profiles_[GM::BARITONE_SAX] = {
        GM::BARITONE_SAX, "Baritone Sax", "reed",
        0.65f, 0.7f, 0.8f, 0.4f, 0.45f, 0.65f,
        0.7f, 0.6f, 0.5f, 0.6f, 0.65f, 36, 68
    };
    profiles_[GM::OBOE] = {
        GM::OBOE, "Oboe", "reed",
        0.85f, 0.8f, 0.45f, 0.7f, 0.25f, 0.5f,
        0.95f, 0.6f, 0.0f, 0.65f, 0.7f, 58, 91
    };
    profiles_[GM::ENGLISH_HORN] = {
        GM::ENGLISH_HORN, "English Horn", "reed",
        0.9f, 0.85f, 0.55f, 0.5f, 0.2f, 0.7f,
        0.9f, 0.65f, 0.0f, 0.7f, 0.6f, 52, 81
    };
    profiles_[GM::BASSOON] = {
        GM::BASSOON, "Bassoon", "reed",
        0.75f, 0.7f, 0.7f, 0.4f, 0.25f, 0.65f,
        0.75f, 0.7f, 0.6f, 0.7f, 0.55f, 34, 75
    };
    profiles_[GM::CLARINET] = {
        GM::CLARINET, "Clarinet", "reed",
        0.8f, 0.8f, 0.5f, 0.6f, 0.2f, 0.7f,
        0.9f, 0.65f, 0.1f, 0.65f, 0.65f, 50, 94
    };
    
    //=========================================================================
    // PIPE (72-79)
    //=========================================================================
    profiles_[GM::PICCOLO] = {
        GM::PICCOLO, "Piccolo", "pipe",
        0.5f, 0.5f, 0.25f, 1.0f, 0.35f, 0.3f,
        0.85f, 0.4f, 0.0f, 0.5f, 0.85f, 74, 108
    };
    profiles_[GM::FLUTE] = {
        GM::FLUTE, "Flute", "pipe",
        0.7f, 0.7f, 0.35f, 0.85f, 0.15f, 0.55f,
        0.95f, 0.55f, 0.0f, 0.7f, 0.7f, 60, 96
    };
    profiles_[GM::RECORDER] = {
        GM::RECORDER, "Recorder", "pipe",
        0.6f, 0.65f, 0.3f, 0.8f, 0.1f, 0.5f,
        0.8f, 0.5f, 0.0f, 0.6f, 0.6f, 60, 96
    };
    profiles_[GM::PAN_FLUTE] = {
        GM::PAN_FLUTE, "Pan Flute", "pipe",
        0.75f, 0.75f, 0.35f, 0.7f, 0.1f, 0.65f,
        0.85f, 0.55f, 0.0f, 0.75f, 0.6f, 60, 96
    };
    profiles_[GM::BLOWN_BOTTLE] = {
        GM::BLOWN_BOTTLE, "Blown Bottle", "pipe",
        0.7f, 0.65f, 0.25f, 0.55f, 0.05f, 0.6f,
        0.7f, 0.45f, 0.0f, 0.8f, 0.5f, 60, 84
    };
    profiles_[GM::SHAKUHACHI] = {
        GM::SHAKUHACHI, "Shakuhachi", "pipe",
        0.9f, 0.85f, 0.45f, 0.55f, 0.2f, 0.7f,
        0.9f, 0.5f, 0.0f, 0.8f, 0.6f, 55, 84
    };
    profiles_[GM::WHISTLE] = {
        GM::WHISTLE, "Whistle", "pipe",
        0.55f, 0.6f, 0.2f, 0.9f, 0.15f, 0.4f,
        0.8f, 0.4f, 0.0f, 0.6f, 0.7f, 60, 96
    };
    profiles_[GM::OCARINA] = {
        GM::OCARINA, "Ocarina", "pipe",
        0.7f, 0.7f, 0.3f, 0.65f, 0.1f, 0.6f,
        0.85f, 0.5f, 0.0f, 0.7f, 0.55f, 60, 84
    };
    
    //=========================================================================
    // SYNTH LEAD (80-87)
    //=========================================================================
    profiles_[GM::LEAD_SQUARE] = {
        GM::LEAD_SQUARE, "Lead Square", "synth_lead",
        0.2f, 0.25f, 0.5f, 0.75f, 0.6f, 0.25f,
        0.9f, 0.5f, 0.1f, 0.5f, 0.75f, 36, 96
    };
    profiles_[GM::LEAD_SAWTOOTH] = {
        GM::LEAD_SAWTOOTH, "Lead Sawtooth", "synth_lead",
        0.15f, 0.2f, 0.55f, 0.8f, 0.7f, 0.2f,
        0.9f, 0.55f, 0.15f, 0.5f, 0.8f, 36, 96
    };
    profiles_[GM::LEAD_CALLIOPE] = {
        GM::LEAD_CALLIOPE, "Lead Calliope", "synth_lead",
        0.4f, 0.35f, 0.35f, 0.85f, 0.3f, 0.35f,
        0.8f, 0.45f, 0.0f, 0.55f, 0.65f, 36, 96
    };
    profiles_[GM::LEAD_CHIFF] = {
        GM::LEAD_CHIFF, "Lead Chiff", "synth_lead",
        0.35f, 0.3f, 0.3f, 0.8f, 0.35f, 0.3f,
        0.75f, 0.4f, 0.0f, 0.5f, 0.7f, 36, 96
    };
    profiles_[GM::LEAD_CHARANG] = {
        GM::LEAD_CHARANG, "Lead Charang", "synth_lead",
        0.2f, 0.25f, 0.5f, 0.75f, 0.65f, 0.25f,
        0.85f, 0.5f, 0.1f, 0.45f, 0.75f, 36, 96
    };
    profiles_[GM::LEAD_VOICE] = {
        GM::LEAD_VOICE, "Lead Voice", "synth_lead",
        0.55f, 0.5f, 0.45f, 0.6f, 0.2f, 0.6f,
        0.8f, 0.6f, 0.0f, 0.7f, 0.5f, 48, 84
    };
    profiles_[GM::LEAD_FIFTHS] = {
        GM::LEAD_FIFTHS, "Lead Fifths", "synth_lead",
        0.25f, 0.3f, 0.6f, 0.7f, 0.55f, 0.35f,
        0.85f, 0.65f, 0.15f, 0.55f, 0.7f, 36, 96
    };
    profiles_[GM::LEAD_BASS_LEAD] = {
        GM::LEAD_BASS_LEAD, "Lead Bass+Lead", "synth_lead",
        0.2f, 0.25f, 0.7f, 0.65f, 0.6f, 0.3f,
        0.8f, 0.55f, 0.4f, 0.5f, 0.7f, 24, 96
    };
    
    //=========================================================================
    // SYNTH PAD (88-95)
    //=========================================================================
    profiles_[GM::PAD_NEW_AGE] = {
        GM::PAD_NEW_AGE, "Pad New Age", "synth_pad",
        0.6f, 0.55f, 0.5f, 0.65f, 0.1f, 0.7f,
        0.5f, 0.75f, 0.2f, 0.95f, 0.3f, 36, 96
    };
    profiles_[GM::PAD_WARM] = {
        GM::PAD_WARM, "Pad Warm", "synth_pad",
        0.7f, 0.65f, 0.55f, 0.45f, 0.05f, 0.9f,
        0.45f, 0.8f, 0.25f, 1.0f, 0.25f, 36, 96
    };
    profiles_[GM::PAD_POLYSYNTH] = {
        GM::PAD_POLYSYNTH, "Pad Polysynth", "synth_pad",
        0.4f, 0.4f, 0.5f, 0.6f, 0.25f, 0.5f,
        0.5f, 0.75f, 0.2f, 0.9f, 0.35f, 36, 96
    };
    profiles_[GM::PAD_CHOIR] = {
        GM::PAD_CHOIR, "Pad Choir", "synth_pad",
        0.65f, 0.55f, 0.6f, 0.5f, 0.1f, 0.75f,
        0.45f, 0.8f, 0.15f, 0.95f, 0.3f, 36, 96
    };
    profiles_[GM::PAD_BOWED] = {
        GM::PAD_BOWED, "Pad Bowed", "synth_pad",
        0.6f, 0.5f, 0.55f, 0.55f, 0.15f, 0.65f,
        0.5f, 0.75f, 0.2f, 0.9f, 0.35f, 36, 96
    };
    profiles_[GM::PAD_METALLIC] = {
        GM::PAD_METALLIC, "Pad Metallic", "synth_pad",
        0.35f, 0.35f, 0.5f, 0.75f, 0.35f, 0.35f,
        0.45f, 0.65f, 0.15f, 0.85f, 0.45f, 36, 96
    };
    profiles_[GM::PAD_HALO] = {
        GM::PAD_HALO, "Pad Halo", "synth_pad",
        0.7f, 0.6f, 0.45f, 0.7f, 0.05f, 0.65f,
        0.4f, 0.7f, 0.1f, 0.95f, 0.3f, 36, 96
    };
    profiles_[GM::PAD_SWEEP] = {
        GM::PAD_SWEEP, "Pad Sweep", "synth_pad",
        0.5f, 0.45f, 0.5f, 0.6f, 0.2f, 0.55f,
        0.4f, 0.65f, 0.15f, 0.9f, 0.4f, 36, 96
    };
    
    //=========================================================================
    // SYNTH EFFECTS (96-103)
    //=========================================================================
    profiles_[GM::FX_RAIN] = {
        GM::FX_RAIN, "FX Rain", "synth_fx",
        0.6f, 0.5f, 0.35f, 0.6f, 0.1f, 0.6f,
        0.3f, 0.4f, 0.0f, 0.95f, 0.3f, 36, 96
    };
    profiles_[GM::FX_SOUNDTRACK] = {
        GM::FX_SOUNDTRACK, "FX Soundtrack", "synth_fx",
        0.55f, 0.45f, 0.6f, 0.55f, 0.2f, 0.55f,
        0.4f, 0.6f, 0.1f, 0.9f, 0.35f, 36, 96
    };
    profiles_[GM::FX_CRYSTAL] = {
        GM::FX_CRYSTAL, "FX Crystal", "synth_fx",
        0.6f, 0.55f, 0.3f, 0.9f, 0.1f, 0.45f,
        0.55f, 0.5f, 0.0f, 0.85f, 0.5f, 48, 96
    };
    profiles_[GM::FX_ATMOSPHERE] = {
        GM::FX_ATMOSPHERE, "FX Atmosphere", "synth_fx",
        0.65f, 0.5f, 0.5f, 0.5f, 0.15f, 0.6f,
        0.35f, 0.55f, 0.1f, 0.95f, 0.3f, 36, 96
    };
    profiles_[GM::FX_BRIGHTNESS] = {
        GM::FX_BRIGHTNESS, "FX Brightness", "synth_fx",
        0.5f, 0.45f, 0.35f, 0.95f, 0.15f, 0.4f,
        0.45f, 0.5f, 0.0f, 0.85f, 0.45f, 48, 96
    };
    profiles_[GM::FX_GOBLINS] = {
        GM::FX_GOBLINS, "FX Goblins", "synth_fx",
        0.3f, 0.25f, 0.5f, 0.6f, 0.5f, 0.3f,
        0.35f, 0.4f, 0.0f, 0.85f, 0.55f, 36, 96
    };
    profiles_[GM::FX_ECHOES] = {
        GM::FX_ECHOES, "FX Echoes", "synth_fx",
        0.6f, 0.5f, 0.4f, 0.6f, 0.1f, 0.55f,
        0.4f, 0.5f, 0.0f, 0.9f, 0.4f, 36, 96
    };
    profiles_[GM::FX_SCI_FI] = {
        GM::FX_SCI_FI, "FX Sci-Fi", "synth_fx",
        0.25f, 0.2f, 0.55f, 0.7f, 0.45f, 0.25f,
        0.4f, 0.45f, 0.1f, 0.85f, 0.5f, 36, 96
    };
    
    //=========================================================================
    // ETHNIC (104-111)
    //=========================================================================
    profiles_[GM::SITAR] = {
        GM::SITAR, "Sitar", "ethnic",
        0.7f, 0.65f, 0.5f, 0.7f, 0.3f, 0.5f,
        0.85f, 0.6f, 0.0f, 0.7f, 0.6f, 48, 96
    };
    profiles_[GM::BANJO] = {
        GM::BANJO, "Banjo", "ethnic",
        0.4f, 0.5f, 0.35f, 0.85f, 0.35f, 0.4f,
        0.8f, 0.6f, 0.0f, 0.5f, 0.75f, 48, 84
    };
    profiles_[GM::SHAMISEN] = {
        GM::SHAMISEN, "Shamisen", "ethnic",
        0.6f, 0.6f, 0.4f, 0.75f, 0.35f, 0.4f,
        0.8f, 0.55f, 0.0f, 0.6f, 0.7f, 48, 84
    };
    profiles_[GM::KOTO] = {
        GM::KOTO, "Koto", "ethnic",
        0.7f, 0.7f, 0.45f, 0.7f, 0.2f, 0.55f,
        0.85f, 0.65f, 0.0f, 0.75f, 0.65f, 48, 96
    };
    profiles_[GM::KALIMBA] = {
        GM::KALIMBA, "Kalimba", "ethnic",
        0.65f, 0.7f, 0.3f, 0.75f, 0.1f, 0.6f,
        0.8f, 0.55f, 0.0f, 0.75f, 0.6f, 60, 96
    };
    profiles_[GM::BAGPIPE] = {
        GM::BAGPIPE, "Bagpipe", "ethnic",
        0.4f, 0.35f, 0.7f, 0.65f, 0.5f, 0.4f,
        0.7f, 0.6f, 0.3f, 0.7f, 0.5f, 36, 77
    };
    profiles_[GM::FIDDLE] = {
        GM::FIDDLE, "Fiddle", "ethnic",
        0.7f, 0.75f, 0.5f, 0.7f, 0.35f, 0.55f,
        0.9f, 0.6f, 0.0f, 0.6f, 0.75f, 55, 96
    };
    profiles_[GM::SHANAI] = {
        GM::SHANAI, "Shanai", "ethnic",
        0.65f, 0.6f, 0.55f, 0.7f, 0.4f, 0.45f,
        0.85f, 0.5f, 0.0f, 0.65f, 0.7f, 48, 84
    };
    
    //=========================================================================
    // PERCUSSIVE (112-119) - Minimal profiles
    //=========================================================================
    profiles_[GM::TINKLE_BELL] = {
        GM::TINKLE_BELL, "Tinkle Bell", "percussive",
        0.5f, 0.45f, 0.2f, 0.95f, 0.15f, 0.4f,
        0.4f, 0.3f, 0.0f, 0.7f, 0.85f, 72, 108
    };
    profiles_[GM::AGOGO] = {
        GM::AGOGO, "Agogo", "percussive",
        0.3f, 0.35f, 0.3f, 0.8f, 0.4f, 0.3f,
        0.35f, 0.25f, 0.0f, 0.5f, 0.9f, 60, 84
    };
    profiles_[GM::STEEL_DRUMS] = {
        GM::STEEL_DRUMS, "Steel Drums", "percussive",
        0.45f, 0.5f, 0.4f, 0.8f, 0.3f, 0.5f,
        0.75f, 0.55f, 0.0f, 0.6f, 0.7f, 52, 84
    };
    profiles_[GM::WOODBLOCK] = {
        GM::WOODBLOCK, "Woodblock", "percussive",
        0.2f, 0.25f, 0.25f, 0.7f, 0.5f, 0.25f,
        0.2f, 0.15f, 0.0f, 0.3f, 0.95f, 60, 77
    };
    profiles_[GM::TAIKO_DRUM] = {
        GM::TAIKO_DRUM, "Taiko Drum", "percussive",
        0.25f, 0.2f, 0.95f, 0.4f, 0.85f, 0.35f,
        0.15f, 0.2f, 0.4f, 0.4f, 1.0f, 36, 60
    };
    profiles_[GM::MELODIC_TOM] = {
        GM::MELODIC_TOM, "Melodic Tom", "percussive",
        0.25f, 0.25f, 0.7f, 0.5f, 0.65f, 0.35f,
        0.2f, 0.25f, 0.3f, 0.4f, 0.9f, 36, 72
    };
    profiles_[GM::SYNTH_DRUM] = {
        GM::SYNTH_DRUM, "Synth Drum", "percussive",
        0.15f, 0.15f, 0.65f, 0.6f, 0.7f, 0.2f,
        0.15f, 0.2f, 0.35f, 0.35f, 0.85f, 36, 72
    };
    profiles_[GM::REVERSE_CYMBAL] = {
        GM::REVERSE_CYMBAL, "Reverse Cymbal", "percussive",
        0.35f, 0.3f, 0.5f, 0.7f, 0.4f, 0.3f,
        0.1f, 0.15f, 0.0f, 0.6f, 0.8f, 48, 84
    };
    
    //=========================================================================
    // SOUND EFFECTS (120-127) - Minimal profiles
    //=========================================================================
    for (int i = GM::GUITAR_FRET_NOISE; i <= GM::GUNSHOT; ++i) {
        if (profiles_.find(i) == profiles_.end()) {
            profiles_[i] = {
                i, "SFX " + std::to_string(i), "sfx",
                0.2f, 0.2f, 0.3f, 0.5f, 0.3f, 0.2f,
                0.1f, 0.1f, 0.0f, 0.3f, 0.5f, 36, 96
            };
        }
    }
}

void InstrumentSelector::initializeEmotionPalettes() {
    // Grief palette
    emotionPalettes_["grief"] = {
        GM::ACOUSTIC_GRAND_PIANO, GM::STRING_ENSEMBLE_1, GM::ELECTRIC_BASS_FINGER,
        GM::PAD_WARM, GM::CELLO,
        "Acoustic Grand Piano", "String Ensemble", "Electric Bass Finger",
        "Warm Pad", "Cello"
    };
    
    // Sadness palette
    emotionPalettes_["sadness"] = {
        GM::ELECTRIC_PIANO_1, GM::STRING_ENSEMBLE_2, GM::FRETLESS_BASS,
        GM::PAD_CHOIR, GM::VIOLA,
        "Rhodes", "String Ensemble 2", "Fretless Bass",
        "Choir Pad", "Viola"
    };
    
    // Hope palette
    emotionPalettes_["hope"] = {
        GM::FLUTE, GM::ACOUSTIC_GUITAR_STEEL, GM::ACOUSTIC_BASS,
        GM::PAD_NEW_AGE, GM::ORCHESTRAL_HARP,
        "Flute", "Acoustic Guitar Steel", "Acoustic Bass",
        "New Age Pad", "Harp"
    };
    
    // Joy palette
    emotionPalettes_["joy"] = {
        GM::BRIGHT_ACOUSTIC_PIANO, GM::ACOUSTIC_GUITAR_STEEL, GM::SLAP_BASS_1,
        GM::PAD_POLYSYNTH, GM::VIBRAPHONE,
        "Bright Piano", "Acoustic Guitar Steel", "Slap Bass",
        "Polysynth Pad", "Vibraphone"
    };
    
    // Anger/Rage palette
    emotionPalettes_["anger"] = {
        GM::DISTORTION_GUITAR, GM::BRASS_SECTION, GM::SYNTH_BASS_2,
        GM::FX_GOBLINS, GM::ORCHESTRA_HIT,
        "Distortion Guitar", "Brass Section", "Synth Bass 2",
        "FX Goblins", "Orchestra Hit"
    };
    emotionPalettes_["rage"] = emotionPalettes_["anger"];
    
    // Fear palette
    emotionPalettes_["fear"] = {
        GM::TREMOLO_STRINGS, GM::STRING_ENSEMBLE_2, GM::CONTRABASS,
        GM::FX_ATMOSPHERE, GM::TIMPANI,
        "Tremolo Strings", "String Ensemble 2", "Contrabass",
        "FX Atmosphere", "Timpani"
    };
    
    // Anxiety palette
    emotionPalettes_["anxiety"] = {
        GM::ELECTRIC_PIANO_2, GM::SYNTH_STRINGS_1, GM::ELECTRIC_BASS_PICK,
        GM::PAD_METALLIC, GM::PIZZICATO_STRINGS,
        "DX Piano", "Synth Strings", "Electric Bass Pick",
        "Metallic Pad", "Pizzicato Strings"
    };
    
    // Peace/Serenity palette
    emotionPalettes_["peace"] = {
        GM::ACOUSTIC_GUITAR_NYLON, GM::PAD_WARM, GM::ACOUSTIC_BASS,
        GM::PAD_HALO, GM::FLUTE,
        "Nylon Guitar", "Warm Pad", "Acoustic Bass",
        "Halo Pad", "Flute"
    };
    emotionPalettes_["serenity"] = emotionPalettes_["peace"];
    
    // Love palette
    emotionPalettes_["love"] = {
        GM::ACOUSTIC_GUITAR_NYLON, GM::STRING_ENSEMBLE_1, GM::FRETLESS_BASS,
        GM::PAD_WARM, GM::VIOLIN,
        "Nylon Guitar", "String Ensemble", "Fretless Bass",
        "Warm Pad", "Violin"
    };
    
    // Nostalgia palette
    emotionPalettes_["nostalgia"] = {
        GM::MUSIC_BOX, GM::STRING_ENSEMBLE_1, GM::ACOUSTIC_BASS,
        GM::PAD_CHOIR, GM::CELESTA,
        "Music Box", "String Ensemble", "Acoustic Bass",
        "Choir Pad", "Celesta"
    };
    
    // Euphoria palette
    emotionPalettes_["euphoria"] = {
        GM::LEAD_SAWTOOTH, GM::SYNTH_BRASS_1, GM::SYNTH_BASS_1,
        GM::PAD_POLYSYNTH, GM::ORCHESTRA_HIT,
        "Lead Sawtooth", "Synth Brass", "Synth Bass",
        "Polysynth Pad", "Orchestra Hit"
    };
    
    // Loneliness palette
    emotionPalettes_["loneliness"] = {
        GM::ACOUSTIC_GRAND_PIANO, GM::PAD_WARM, GM::ACOUSTIC_BASS,
        GM::FX_ECHOES, GM::OBOE,
        "Acoustic Piano", "Warm Pad", "Acoustic Bass",
        "FX Echoes", "Oboe"
    };
    
    // Neutral palette
    emotionPalettes_["neutral"] = {
        GM::ACOUSTIC_GRAND_PIANO, GM::STRING_ENSEMBLE_1, GM::ELECTRIC_BASS_FINGER,
        GM::PAD_NEW_AGE, GM::VIBRAPHONE,
        "Acoustic Piano", "String Ensemble", "Electric Bass Finger",
        "New Age Pad", "Vibraphone"
    };
}

InstrumentSelector::EmotionCharacteristics 
InstrumentSelector::getEmotionCharacteristics(const std::string& emotion) const {
    EmotionCharacteristics chars{0.0f, 0.5f, 0.5f, false, false, false};
    
    std::string e = emotion;
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    
    // Map emotions to characteristics
    if (e == "grief" || e == "devastated" || e == "heartbroken") {
        chars = {-0.8f, 0.3f, 0.9f, true, false, true};
    } else if (e == "sadness" || e == "sad" || e == "sorrowful") {
        chars = {-0.5f, 0.25f, 0.6f, true, false, true};
    } else if (e == "hope" || e == "hopeful" || e == "optimistic") {
        chars = {0.4f, 0.5f, 0.5f, false, false, true};
    } else if (e == "joy" || e == "happy" || e == "joyful") {
        chars = {0.8f, 0.7f, 0.7f, false, false, false};
    } else if (e == "anger" || e == "rage" || e == "furious") {
        chars = {-0.7f, 0.9f, 0.9f, false, true, false};
    } else if (e == "fear" || e == "terrified" || e == "scared") {
        chars = {-0.6f, 0.7f, 0.8f, true, false, false};
    } else if (e == "anxiety" || e == "anxious" || e == "nervous") {
        chars = {-0.4f, 0.6f, 0.7f, true, false, false};
    } else if (e == "peace" || e == "serene" || e == "calm") {
        chars = {0.5f, 0.15f, 0.4f, false, false, true};
    } else if (e == "love" || e == "loving" || e == "affectionate") {
        chars = {0.7f, 0.4f, 0.7f, true, false, true};
    } else if (e == "nostalgia" || e == "nostalgic" || e == "wistful") {
        chars = {-0.2f, 0.3f, 0.6f, true, false, true};
    } else if (e == "euphoria" || e == "ecstatic" || e == "elated") {
        chars = {0.95f, 0.9f, 1.0f, false, false, false};
    } else if (e == "loneliness" || e == "lonely" || e == "isolated") {
        chars = {-0.5f, 0.2f, 0.7f, true, false, true};
    }
    
    return chars;
}

InstrumentPalette InstrumentSelector::getPaletteForEmotion(const std::string& emotion) const {
    std::string e = emotion;
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    
    auto it = emotionPalettes_.find(e);
    if (it != emotionPalettes_.end()) {
        return it->second;
    }
    
    // Return neutral if not found
    return emotionPalettes_.at("neutral");
}

InstrumentPalette InstrumentSelector::getPaletteForEmotion(const EmotionNode& emotion) const {
    return getPaletteForEmotion(emotion.name);
}

float InstrumentSelector::scoreInstrumentForEmotion(
    int gmProgram,
    const std::string& emotion,
    float vulnerabilityLevel,
    float intimacyPreference
) const {
    auto profIt = profiles_.find(gmProgram);
    if (profIt == profiles_.end()) return 0.0f;
    
    const auto& profile = profIt->second;
    auto chars = getEmotionCharacteristics(emotion);
    
    float score = 0.0f;
    
    // Vulnerability match (higher weight if emotion needs vulnerability)
    if (chars.needsVulnerability) {
        score += profile.vulnerability * vulnerabilityLevel * 5.0f;
    }
    
    // Intimacy match
    score += (1.0f - std::abs(profile.intimacy - intimacyPreference)) * 3.0f;
    
    // Aggression match
    if (chars.needsAggression) {
        score += profile.aggression * 4.0f;
    } else {
        score += (1.0f - profile.aggression) * 2.0f;
    }
    
    // Warmth match
    if (chars.needsWarmth) {
        score += profile.warmth * 3.0f;
    }
    
    // Brightness based on valence
    if (chars.valence > 0) {
        score += profile.brightness * chars.valence * 2.0f;
    } else {
        score += (1.0f - profile.brightness) * std::abs(chars.valence) * 2.0f;
    }
    
    // Emotional weight based on intensity
    score += profile.emotionalWeight * chars.intensity * 2.0f;
    
    return score;
}

std::vector<InstrumentRecommendation> InstrumentSelector::recommend(
    const std::string& emotion,
    float vulnerabilityLevel,
    float intimacyPreference,
    const std::string& genre,
    int count
) const {
    std::vector<InstrumentRecommendation> recommendations;
    
    // Score all instruments
    for (const auto& [program, profile] : profiles_) {
        float score = scoreInstrumentForEmotion(program, emotion, vulnerabilityLevel, intimacyPreference);
        
        // Genre modifiers
        if (!genre.empty()) {
            std::string g = genre;
            std::transform(g.begin(), g.end(), g.begin(), ::tolower);
            
            if (g == "singer-songwriter" || g == "acoustic") {
                if (profile.family == "guitar" || profile.family == "piano") {
                    score *= 1.3f;
                }
            } else if (g == "electronic" || g == "edm") {
                if (profile.family == "synth_lead" || profile.family == "synth_pad") {
                    score *= 1.3f;
                }
            } else if (g == "orchestral" || g == "cinematic") {
                if (profile.family == "strings" || profile.family == "ensemble" || profile.family == "brass") {
                    score *= 1.3f;
                }
            } else if (g == "jazz") {
                if (profile.family == "reed" || profile.family == "brass" || 
                    program == GM::ELECTRIC_PIANO_1 || program == GM::ACOUSTIC_BASS) {
                    score *= 1.3f;
                }
            }
        }
        
        std::string reason;
        auto chars = getEmotionCharacteristics(emotion);
        
        if (chars.needsVulnerability && profile.vulnerability > 0.7f) {
            reason = "High vulnerability support";
        } else if (chars.needsAggression && profile.aggression > 0.7f) {
            reason = "Aggressive character";
        } else if (chars.needsWarmth && profile.warmth > 0.7f) {
            reason = "Warm, comforting tone";
        } else if (profile.intimacy > 0.7f && intimacyPreference > 0.6f) {
            reason = "Intimate, personal sound";
        } else {
            reason = "Emotional fit";
        }
        
        recommendations.push_back({program, profile.name, score, reason});
    }
    
    // Sort by score descending
    std::sort(recommendations.begin(), recommendations.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    
    // Return top N
    if (static_cast<int>(recommendations.size()) > count) {
        recommendations.resize(static_cast<size_t>(count));
    }
    
    return recommendations;
}

int InstrumentSelector::getLeadInstrument(const std::string& emotion) const {
    return getPaletteForEmotion(emotion).lead;
}

int InstrumentSelector::getBassInstrument(const std::string& emotion) const {
    return getPaletteForEmotion(emotion).bass;
}

int InstrumentSelector::getTextureInstrument(const std::string& emotion) const {
    return getPaletteForEmotion(emotion).texture;
}

const InstrumentProfile& InstrumentSelector::getProfile(int gmProgram) const {
    static InstrumentProfile defaultProfile{0, "Unknown", "unknown", 
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 21, 108};
    
    auto it = profiles_.find(gmProgram);
    if (it != profiles_.end()) {
        return it->second;
    }
    return defaultProfile;
}

std::string InstrumentSelector::getInstrumentName(int gmProgram) const {
    return getProfile(gmProgram).name;
}

std::vector<int> InstrumentSelector::getInstrumentsByFamily(const std::string& family) const {
    std::vector<int> result;
    for (const auto& [program, profile] : profiles_) {
        if (profile.family == family) {
            result.push_back(program);
        }
    }
    return result;
}

//=============================================================================
// CONVENIENCE FUNCTIONS
//=============================================================================

int recommendLeadInstrument(const std::string& emotion) {
    static InstrumentSelector selector;
    return selector.getLeadInstrument(emotion);
}

int recommendBassInstrument(const std::string& emotion) {
    static InstrumentSelector selector;
    return selector.getBassInstrument(emotion);
}

InstrumentPalette getEmotionPalette(const std::string& emotion) {
    static InstrumentSelector selector;
    return selector.getPaletteForEmotion(emotion);
}

} // namespace kelly
