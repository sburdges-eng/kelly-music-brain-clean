#pragma once

/**
 * MusicConstants.h
 *
 * Named constants for music theory, MIDI, and emotion thresholds.
 * Replaces magic numbers throughout the codebase.
 */

namespace kelly {
namespace MusicConstants {

// =============================================================================
// MIDI NOTE NUMBERS
// =============================================================================

// Octave 0
constexpr int MIDI_C0 = 12;
constexpr int MIDI_A0 = 21;

// Octave 1
constexpr int MIDI_C1 = 24;
constexpr int MIDI_A1 = 33;

// Octave 2
constexpr int MIDI_C2 = 36;
constexpr int MIDI_E2 = 40;
constexpr int MIDI_F2 = 41;
constexpr int MIDI_Fs2 = 42;  // F#2
constexpr int MIDI_A2 = 45;

// Octave 3 (common range)
constexpr int MIDI_C3 = 48;
constexpr int MIDI_E3 = 52;
constexpr int MIDI_G3 = 55;
constexpr int MIDI_A3 = 57;

// Octave 4 (middle C and common range)
constexpr int MIDI_C4 = 60;  // Middle C
constexpr int MIDI_E4 = 64;
constexpr int MIDI_G4 = 67;
constexpr int MIDI_A4 = 69;

// Octave 5
constexpr int MIDI_C5 = 72;
constexpr int MIDI_C6 = 84;

// MIDI Channel Constants
constexpr int MIDI_CHANNEL_DRUMS = 9;  // Standard drum channel (10 in 1-indexed)
constexpr int MIDI_CHANNEL_MIN = 0;
constexpr int MIDI_CHANNEL_MAX = 15;

// MIDI Channel Assignments (0-indexed, matching PluginProcessor)
constexpr int MIDI_CHANNEL_CHORDS = 0;         // Channel 1: Chords
constexpr int MIDI_CHANNEL_MELODY = 1;         // Channel 2: Melody
constexpr int MIDI_CHANNEL_BASS = 2;           // Channel 3: Bass
constexpr int MIDI_CHANNEL_COUNTER_MELODY = 3; // Channel 4: Counter-melody
constexpr int MIDI_CHANNEL_PAD = 4;            // Channel 5: Pad
constexpr int MIDI_CHANNEL_STRINGS = 5;        // Channel 6: Strings
constexpr int MIDI_CHANNEL_FILLS = 6;          // Channel 7: Fills
constexpr int MIDI_CHANNEL_RHYTHM = 9;         // Channel 10: Rhythm/Drums

// MIDI Velocity Range
constexpr int MIDI_VELOCITY_MIN = 0;
constexpr int MIDI_VELOCITY_MAX = 127;
constexpr int MIDI_VELOCITY_DEFAULT = 60;
constexpr int MIDI_VELOCITY_SOFT = 45;
constexpr int MIDI_VELOCITY_MEDIUM = 75;
constexpr int MIDI_VELOCITY_LOUD = 100;

// MIDI Pitch Range
constexpr int MIDI_PITCH_MIN = 0;
constexpr int MIDI_PITCH_MAX = 127;
constexpr int MIDI_PITCH_REASONABLE_MAX = 108;  // Common upper limit for musical notes (C8)

// MIDI Timing
constexpr int MIDI_PPQ = 480;  // Pulses per quarter note (standard)

// MIDI Tempo Calculation
constexpr int MIDI_MICROSECONDS_PER_MINUTE = 60000000;  // For tempo meta events

// =============================================================================
// EMOTION THRESHOLDS (Valence: -1.0 to 1.0)
// =============================================================================

// Valence thresholds
constexpr float VALENCE_VERY_NEGATIVE = -0.7f;
constexpr float VALENCE_NEGATIVE = -0.5f;
constexpr float VALENCE_SLIGHTLY_NEGATIVE = -0.3f;
constexpr float VALENCE_NEUTRAL = 0.0f;
constexpr float VALENCE_SLIGHTLY_POSITIVE = 0.3f;
constexpr float VALENCE_POSITIVE = 0.5f;
constexpr float VALENCE_VERY_POSITIVE = 0.7f;

// Arousal thresholds (0.0 to 1.0)
constexpr float AROUSAL_VERY_LOW = 0.2f;
constexpr float AROUSAL_LOW = 0.3f;
constexpr float AROUSAL_MODERATE = 0.5f;
constexpr float AROUSAL_HIGH = 0.7f;
constexpr float AROUSAL_VERY_HIGH = 0.9f;

// Intensity thresholds (0.0 to 1.0)
constexpr float INTENSITY_LOW = 0.3f;
constexpr float INTENSITY_MODERATE = 0.5f;
constexpr float INTENSITY_HIGH = 0.6f;
constexpr float INTENSITY_VERY_HIGH = 0.7f;
constexpr float INTENSITY_EXTREME = 0.9f;

// =============================================================================
// TIMING CONSTANTS (in beats)
// =============================================================================

constexpr double BEATS_PER_BAR = 4.0;
constexpr double BEATS_PER_HALF_NOTE = 2.0;
constexpr double BEATS_PER_QUARTER_NOTE = 1.0;
constexpr double BEATS_PER_EIGHTH_NOTE = 0.5;
constexpr double BEATS_PER_SIXTEENTH_NOTE = 0.25;
constexpr double BEATS_PER_THIRTY_SECOND_NOTE = 0.125;

// Minimum note length
constexpr double MINIMUM_NOTE_LENGTH = 0.25;  // 16th note

// Timing variation (for humanization)
constexpr float TIMING_VARIATION_BASE = 0.02f;  // ±2% of beat
constexpr float TIMING_BIAS_SAD = -0.01f;      // Drag behind for sad emotions
constexpr float TIMING_BIAS_ANGRY = 0.01f;      // Rush ahead for angry emotions

// =============================================================================
// TEMPO CONSTANTS (BPM)
// =============================================================================

constexpr int TEMPO_MIN = 60;
constexpr int TEMPO_DEFAULT = 100;
constexpr int TEMPO_MAX = 180;
constexpr int TEMPO_VERY_SLOW = 60;
constexpr int TEMPO_SLOW = 80;
constexpr int TEMPO_MODERATE = 100;
constexpr int TEMPO_FAST = 120;
constexpr int TEMPO_VERY_FAST = 160;

// =============================================================================
// MUSICAL INTERVALS (in semitones)
// =============================================================================

constexpr int INTERVAL_UNISON = 0;
constexpr int INTERVAL_MINOR_SECOND = 1;
constexpr int INTERVAL_MAJOR_SECOND = 2;
constexpr int INTERVAL_MINOR_THIRD = 3;
constexpr int INTERVAL_MAJOR_THIRD = 4;
constexpr int INTERVAL_PERFECT_FOURTH = 5;
constexpr int INTERVAL_TRITONE = 6;
constexpr int INTERVAL_PERFECT_FIFTH = 7;
constexpr int INTERVAL_MINOR_SIXTH = 8;
constexpr int INTERVAL_MAJOR_SIXTH = 9;
constexpr int INTERVAL_MINOR_SEVENTH = 10;
constexpr int INTERVAL_MAJOR_SEVENTH = 11;
constexpr int INTERVAL_OCTAVE = 12;

// =============================================================================
// RULE BREAK SEVERITY THRESHOLDS
// =============================================================================

constexpr float RULE_BREAK_LOW = 0.3f;
constexpr float RULE_BREAK_MODERATE = 0.5f;
constexpr float RULE_BREAK_HIGH = 0.7f;
constexpr float RULE_BREAK_EXTREME = 0.9f;

// =============================================================================
// CHORD GENERATION CONSTANTS
// =============================================================================

// Default root notes by emotion
constexpr int ROOT_NOTE_DEFAULT = MIDI_C3;      // 48
constexpr int ROOT_NOTE_DARK = MIDI_A2;         // 45 (for negative valence)
constexpr int ROOT_NOTE_BRIGHT = MIDI_E3;       // 52 (for positive valence)

// Chord extension thresholds
constexpr float CHORD_EXTENSION_INTENSITY_THRESHOLD = 0.6f;

// =============================================================================
// HUMANIZATION CONSTANTS
// =============================================================================

// Timing humanization
constexpr float HUMANIZATION_TIMING_VARIANCE_BASE = 0.02f;  // ±2% of beat
constexpr float HUMANIZATION_TIMING_VARIANCE_MAX = 0.05f;   // ±5% max

// Velocity humanization
constexpr float HUMANIZATION_VELOCITY_VARIANCE = 0.1f;  // ±10% of velocity

// =============================================================================
// UI CONSTANTS
// =============================================================================

constexpr int UI_SLIDER_TEXT_BOX_WIDTH = 60;
constexpr int UI_SLIDER_TEXT_BOX_HEIGHT = 20;
constexpr int UI_TRACK_HEIGHT = 60;
constexpr int UI_HEADER_HEIGHT = 45;
constexpr int UI_ROW_HEIGHT = 60;

// Animation
constexpr int ANIMATION_FPS = 60;
constexpr int ANIMATION_TIMER_INTERVAL_MS = 16;  // ~60 FPS

// =============================================================================
// MIDI GENERATION CONSTANTS
// =============================================================================

// Velocity multipliers
constexpr float BASS_VELOCITY_MULTIPLIER = 1.1f;  // Bass notes slightly louder

// Syncopation and timing
constexpr float SYNCOPATION_MAX_SHIFT = 0.1f;  // Max 10% timing shift for syncopation

// Probability factors for rule breaks
constexpr float CHROMATICISM_PROBABILITY_FACTOR = 0.3f;  // Probability multiplier for chromatic notes
constexpr float REST_PROBABILITY_FACTOR = 0.3f;  // Probability multiplier for rests

// Rule break velocity adjustments
constexpr int DYNAMICS_RULE_BREAK_VELOCITY_ADJUSTMENT = 5;  // Velocity adjustment for dynamics rule breaks

// Variation thresholds and multipliers
constexpr float MAX_VARIATION_BLEND = 0.3f;  // Max 30% variation blend
constexpr float VARIATION_BLEND_THRESHOLD = 0.01f;  // Minimum blend threshold
constexpr float VARIATION_REPLACE_THRESHOLD = 0.7f;  // Threshold to replace with variation
constexpr float BASS_VARIATION_THRESHOLD = 0.6f;  // Threshold for bass variations
constexpr float BASS_VARIATION_INTENSITY_MULTIPLIER = 0.7f;  // Bass variation intensity multiplier
constexpr float BASS_VARIATION_REPLACE_THRESHOLD = 0.8f;  // Threshold to replace bass with variation
constexpr float COUNTER_MELODY_VARIATION_THRESHOLD = 0.75f;  // Threshold for counter melody variations
constexpr float COUNTER_MELODY_VARIATION_INTENSITY_MULTIPLIER = 0.5f;  // Counter melody variation intensity multiplier
constexpr float COUNTER_MELODY_VARIATION_REPLACE_THRESHOLD = 0.85f;  // Threshold to replace counter melody with variation

// Humanization multipliers
constexpr float BASS_HUMANIZE_MULTIPLIER = 0.7f;  // Bass gets less humanization
constexpr float COUNTER_MELODY_HUMANIZE_MULTIPLIER = 0.8f;  // Counter melody gets moderate humanization
constexpr float MIN_HUMANIZE_THRESHOLD = 0.01f;  // Minimum threshold for humanization

// Complexity thresholds
constexpr float PADS_COMPLEXITY_THRESHOLD = 0.4f;  // Threshold for pad generation

// =============================================================================
// DRUM NOTE NUMBERS (General MIDI Standard)
// =============================================================================

constexpr int DRUM_KICK = 36;      // C2 - Bass drum
constexpr int DRUM_SNARE = 38;     // D2 - Snare drum
constexpr int DRUM_HIHAT = 42;     // F#2 - Hi-hat
constexpr int DRUM_RIDE = 51;      // Eb3 - Ride cymbal
constexpr int DRUM_CRASH = 49;     // C#3 - Crash cymbal
constexpr int DRUM_TOM = 45;       // A2 - Tom-tom
constexpr int DRUM_PERCUSSION = 56; // G#3 - General percussion

// =============================================================================
// TIMING OFFSETS AND DURATIONS
// =============================================================================

// Timing humanization offsets (in ticks)
constexpr int TIMING_OFFSET_MIN = -20;  // Minimum timing offset for humanization
constexpr int TIMING_OFFSET_MAX = 20;   // Maximum timing offset for humanization

// Minimum note durations (in ticks)
constexpr int MIN_STACCATO_DURATION = 60;  // Minimum duration for staccato notes
constexpr int MIN_NOTE_DURATION_OFFSET = 10; // Minimum offset to subtract from base duration

// Duration multipliers
constexpr float DURATION_MULTIPLIER_75_PERCENT = 0.75f;  // 75% duration multiplier
constexpr float DURATION_MULTIPLIER_25_PERCENT = 0.25f;  // 25% duration multiplier
constexpr float DURATION_MULTIPLIER_60_PERCENT = 0.6f;   // 60% duration multiplier
constexpr float DURATION_MULTIPLIER_20_PERCENT = 0.2f;  // 20% duration multiplier

// Velocity adjustments
constexpr int VELOCITY_ACCENT_BOOST = 15;  // Velocity boost for accented notes
constexpr int VELOCITY_GHOST_REDUCTION = 10; // Velocity reduction for ghost notes
constexpr float VELOCITY_OFFBEAT_MULTIPLIER = 0.7f;  // Multiplier for offbeat notes

// Fill Engine velocity boost values
constexpr int FILL_VELOCITY_BOOST_SUBTLE = -10;      // Velocity boost for subtle fills
constexpr int FILL_VELOCITY_BOOST_MODERATE = 0;     // Velocity boost for moderate fills
constexpr int FILL_VELOCITY_BOOST_INTENSE = 15;     // Velocity boost for intense fills
constexpr int FILL_VELOCITY_BOOST_EXPLOSIVE = 30;   // Velocity boost for explosive fills

// Fill Engine timing constants
constexpr int FLAM_GRACE_NOTE_GAP_TICKS = 15;       // Ticks between flam grace note and main note
constexpr int FLAM_GRACE_NOTE_VELOCITY_REDUCTION = 25; // Velocity reduction for flam grace notes

// Note duration offsets (for staccato/legato effects)
constexpr int NOTE_DURATION_OFFSET_STANDARD = -10;  // Standard note duration offset
constexpr int NOTE_DURATION_OFFSET_SMALL = -5;      // Small note duration offset
constexpr int NOTE_DURATION_OFFSET_TINY = -2;       // Tiny note duration offset

// Velocity variation ranges (for humanization)
constexpr int VELOCITY_VARIATION_WIDE = 10;          // Wide velocity variation range (±10)
constexpr int VELOCITY_VARIATION_MEDIUM = 8;         // Medium velocity variation range (±8)
constexpr int VELOCITY_VARIATION_NARROW = 5;         // Narrow velocity variation range (±5)

// Probability thresholds
constexpr float DISSONANCE_APPLICATION_FACTOR = 0.5f;  // 50% of severity for dissonance
constexpr float CHROMATIC_PASSING_PROBABILITY = 0.3f;  // 30% probability for chromatic passing chords
constexpr float MODAL_INTERCHANGE_PROBABILITY = 0.3f;  // 30% probability for modal interchange

// UI constants
constexpr float EMOTION_WHEEL_AUTO_SELECT_THRESHOLD = 0.15f;  // Threshold for auto-selecting emotion on wheel

} // namespace MusicConstants
} // namespace kelly

