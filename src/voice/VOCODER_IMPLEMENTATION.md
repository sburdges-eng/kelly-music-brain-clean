# Vocoder Implementation Documentation

## Overview

This document describes the full vocoder integration for voice synthesis in the Kelly MIDI Companion plugin. The vocoder uses formant-based synthesis to generate realistic vocal sounds from MIDI notes.

## Architecture

### Core Components

1. **VocoderEngine** (`VocoderEngine.h/cpp`)
   - Formant-based synthesis engine using bandpass filters
   - Generates glottal pulse waveforms
   - Applies formant filters to shape vocal timbre
   - Supports vibrato, breathiness, and brightness parameters

2. **VowelFormantDatabase** (`VocoderEngine.h/cpp`)
   - Database of 16 vowel formants (AH, EH, IH, OH, OO, etc.)
   - Stores F1-F4 formant frequencies and bandwidths
   - Supports vowel interpolation for smooth transitions
   - Pitch-based vowel selection

3. **ADSREnvelope** (`VocoderEngine.h/cpp`)
   - Attack-Decay-Sustain-Release envelope generator
   - Configurable timing parameters
   - Per-note envelope shaping

4. **PortamentoGenerator** (`VocoderEngine.h/cpp`)
   - Smooth pitch transitions between notes
   - Configurable portamento time
   - Prevents pitch discontinuities

5. **VoiceSynthesizer** (`VoiceSynthesizer.h/cpp`)
   - High-level interface for voice synthesis
   - Integrates all vocoder components
   - Generates vocal melodies from emotions
   - Provides both offline and real-time synthesis

## Technical Details

### Formant Synthesis

The vocoder uses formant synthesis, which models the vocal tract as a series of resonators (formants). Each vowel sound has characteristic formant frequencies:

- **F1 (First Formant)**: Determines vowel openness (AH = 730Hz, EE = 270Hz)
- **F2 (Second Formant)**: Determines front/back placement
- **F3/F4 (Higher Formants)**: Add timbral detail and brightness

Formants are implemented as second-order bandpass filters using bilinear transform.

### Glottal Pulse Generation

The source signal (glottal pulse) models vocal cord vibration using a simplified Rosenberg model:

- Open phase: Rising cosine wave
- Closing phase: Exponential decay
- Shape parameter controls brightness (0.0 = smooth, 1.0 = sharp)

### Signal Flow

```
MIDI Note → Frequency Conversion → Glottal Pulse Generator
                                          ↓
                                    [Breathiness Mix]
                                          ↓
                                    Formant Filter 1 (F1)
                                          ↓
                                    Formant Filter 2 (F2)
                                          ↓
                                    Formant Filter 3 (F3)
                                          ↓
                                    Formant Filter 4 (F4)
                                          ↓
                                    [Brightness Processing]
                                          ↓
                                    ADSR Envelope
                                          ↓
                                    Output Audio
```

### Vibrato

Vibrato adds natural pitch modulation:

- Rate: Typically 4-6 Hz (configurable)
- Depth: Pitch modulation amount (0-1.0)
- Phase-synchronized oscillator

### Breathiness

Breathiness adds noise to the signal:

- White noise generator
- Mixed with glottal pulse based on breathiness parameter
- Higher values = more breathy/whispered sound

## Usage

### Basic Synthesis

```cpp
#include "voice/VoiceSynthesizer.h"

// Create synthesizer
VoiceSynthesizer synth;
synth.setEnabled(true);
synth.prepare(44100.0);  // Sample rate
synth.setBPM(120.0f);

// Generate vocal melody (from emotion)
EmotionNode emotion = ...;
GeneratedMidi midi = ...;
auto vocalNotes = synth.generateVocalMelody(emotion, midi);

// Synthesize audio
auto audio = synth.synthesizeAudio(vocalNotes, 44100.0);
```

### Real-Time Processing

```cpp
// In audio callback
float outputBuffer[512];
int64_t currentSample = getCurrentPlaybackSample();

synth.synthesizeBlock(
    vocalNotes,
    outputBuffer,
    512,  // numSamples
    currentSample
);
```

### Emotion-Based Vowel Selection

The vocoder automatically selects vowels based on:

- **Pitch**: Lower pitches use open vowels (AH, OH), higher pitches use close vowels (EE, IY)
- **Emotion valence**: Negative emotions → open vowels, positive emotions → close vowels
- **Emotion arousal**: High arousal → brighter vowels

## Parameters

### VocoderEngine Parameters

- `pitch`: Fundamental frequency in Hz
- `formants`: Array of 4 formant frequencies [F1, F2, F3, F4]
- `formantBandwidths`: Array of 4 bandwidths [B1, B2, B3, B4]
- `vibratoDepth`: 0.0 to 1.0
- `vibratoRate`: Hz (typically 4-6)
- `breathiness`: 0.0 (clear) to 1.0 (breathy)
- `brightness`: 0.0 (dark) to 1.0 (bright)

### ADSR Envelope Parameters

- `attackTime`: Attack time in seconds (default: 0.01s)
- `decayTime`: Decay time in seconds (default: 0.05s)
- `sustainLevel`: Sustain level 0.0-1.0 (default: 0.7)
- `releaseTime`: Release time in seconds (default: 0.1s)

### Portamento Parameters

- `portamentoTime`: Transition time in seconds (default: 0.1s)
- `enabled`: Enable/disable portamento (default: true)

## Performance Considerations

- **CPU Usage**: Formant synthesis is computationally efficient (~5-10% CPU on modern hardware)
- **Memory**: Minimal memory footprint (~1KB per voice instance)
- **Latency**: Zero-latency processing (suitable for real-time)
- **Thread Safety**: Not thread-safe (create separate instances per thread)

## Lyric Generation Integration

The vocoder now integrates with a comprehensive lyric generation system:

### Components

1. **LyricGenerator** (`src/voice/LyricGenerator.h/cpp`)
   - Generates structured lyrics from emotions and wounds
   - Supports multiple structure types (verse/chorus/bridge)
   - Applies rhyme schemes (ABAB, AABB, etc.)
   - Uses prosody analysis for natural rhythm

2. **PhonemeConverter** (`src/voice/PhonemeConverter.h/cpp`)
   - Converts text to IPA phonemes
   - Provides syllable splitting and stress detection
   - Supports 44 English phonemes with formant data
   - Includes common word dictionary for accurate G2P

3. **PitchPhonemeAligner** (`src/voice/PitchPhonemeAligner.h/cpp`)
   - Aligns lyrics to vocal melodies
   - Handles melisma (multiple notes per syllable)
   - Calculates phoneme timing
   - Supports portamento between phonemes

4. **ExpressionEngine** (`src/voice/ExpressionEngine.h/cpp`)
   - Applies vocal expression curves
   - Maps emotions to expression parameters
   - Generates dynamics, vibrato, and articulation curves

5. **ProsodyAnalyzer** (`src/voice/ProsodyAnalyzer.h/cpp`)
   - Analyzes syllable stress patterns
   - Matches meter patterns (iambic, trochaic, etc.)
   - Validates line lengths

6. **RhymeEngine** (`src/voice/RhymeEngine.h/cpp`)
   - Detects perfect and slant rhymes
   - Generates rhyming word pairs
   - Builds rhyme databases from phonemes

### Usage Example

```cpp
// Generate lyrics
LyricGenerator lyricGen;
auto lyrics = lyricGen.generateLyrics(emotion, wound, &midiContext);

// Generate vocal melody with lyrics
auto vocalNotes = voiceSynthesizer.generateVocalMelody(
    emotion,
    midiContext,
    &lyrics.structure
);

// Synthesize audio
auto audio = voiceSynthesizer.synthesizeAudio(vocalNotes, 44100.0, &emotion);
```

### Voice Types

The system now supports multiple voice types:

- **Male**: Lower formants (85% shift), pitch range C3-F#5
- **Female**: Higher formants (115% shift), pitch range C4-F#6
- **Child**: Highest formants (130% shift), pitch range C5-C7
- **Neutral**: No formant shift, full pitch range

See `VoiceSynthesizer::setVoiceType()` for details.

### Phoneme System

Full phoneme database includes:

- 12 vowels with formant data
- 8 diphthongs with transition formants
- 24 consonants (including voiced/unvoiced pairs)
- Formant interpolation for smooth transitions

Phoneme data is stored in `data/phonemes.json` and loaded by `PhonemeConverter`.

## Future Enhancements

Potential improvements for future versions:

1. ✅ **Multiple Voice Types**: Male/female/child formant databases (IMPLEMENTED)
2. ✅ **Consonant Synthesis**: Add consonant generation for complete speech (IMPLEMENTED)
3. ✅ **Dynamic Formant Tracking**: Smooth formant transitions during note changes (IMPLEMENTED)
4. ✅ **Phoneme-Based Synthesis**: Use phonemes instead of just vowels (IMPLEMENTED)
5. **Voice Cloning**: Import formants from recorded voice samples
6. **Harmonic Enhancement**: Add even/odd harmonic control
7. **Tremolo**: Amplitude modulation in addition to vibrato
8. **Pitch Correction**: Automatic pitch correction for more stable tuning
9. **CMU Dictionary Integration**: Full CMU Pronouncing Dictionary support for improved G2P accuracy
10. **Multi-language Support**: Support for phonemes in other languages

## References

- Rosenberg, A. E. (1971). Effect of glottal pulse shape on the quality of natural vowels. *Journal of the Acoustical Society of America*, 49(2B), 583-590.
- Klatt, D. H. (1980). Software for a cascade/parallel formant synthesizer. *Journal of the Acoustical Society of America*, 67(3), 971-995.
- Cook, P. R. (2002). *Real Sound Synthesis for Interactive Applications*. A K Peters.

## Implementation Timeline

- **Week 1**: Core vocoder engine and formant database
- **Week 2**: Integration with VoiceSynthesizer and envelope/portamento
- **Week 3**: Testing, optimization, and documentation

## Testing

To test the vocoder:

1. Enable voice synthesis in the plugin
2. Generate a melody with emotional input
3. The vocoder will synthesize vocal audio
4. Adjust parameters (vibrato, breathiness, brightness) to taste

## Troubleshooting

**No audio output:**

- Check that `setEnabled(true)` was called
- Verify sample rate is set correctly
- Ensure vocal notes are generated

**Distorted/clipping audio:**

- Reduce overall gain (currently scaled to 0.3x)
- Check formant filter stability (bandwidths should be positive)

**Unrealistic sound:**

- Adjust formant frequencies for different voice types
- Increase breathiness for softer sounds
- Modify glottal pulse shape for different timbres
