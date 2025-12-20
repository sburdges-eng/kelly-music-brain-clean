# miDiKompanion User Guide

This comprehensive guide covers all features of miDiKompanion, the emotion-driven MIDI generation plugin.

## Table of Contents

- [Overview](#overview)
- [Emotion System](#emotion-system)
- [VAD Parameters](#vad-parameters)
- [Generation Settings](#generation-settings)
- [Project Management](#project-management)
- [MIDI Export](#midi-export)
- [Cloud Generation](#cloud-generation)
- [Advanced Usage](#advanced-usage)

## Overview

miDiKompanion uses a sophisticated emotion-to-music mapping system to generate MIDI patterns that reflect specific emotional states. The system is based on the 216-node Emotion Thesaurus, which maps emotions to musical attributes.

### Key Concepts

1. **Emotion Nodes**: 216 unique emotional states organized hierarchically
2. **VAD Model**: Valence-Arousal-Dominance emotional dimensions
3. **Musical Mapping**: Automatic translation of emotions to musical parameters

## Emotion System

### Base Emotions

The plugin provides 6 base emotions as starting points:

| Emotion | Description | Musical Characteristics |
|---------|-------------|------------------------|
| Happy | Joyful, bright | Medium-fast tempo, major key, upward contours |
| Sad | Melancholic, reflective | Slow tempo, minor key, descending phrases |
| Angry | Intense, forceful | Fast tempo, dissonance, accented rhythms |
| Fear | Anxious, tense | Uneven rhythms, sparse textures, low register |
| Surprise | Sudden, dramatic | Dynamic shifts, tempo feints, jumps in register |
| Disgust | Rejecting, unsettled | Chromatic tension, darker timbres, percussive emphasis |

### 216-Node Thesaurus

Behind the base emotions is a 216-node emotion thesaurus with:

- **6 base emotions** × **6 sub-emotions** × **6 intensity levels**
- Each node has specific VAD coordinates and musical attributes
- Nodes are interconnected for smooth emotional transitions

## VAD Parameters

### Valence (Negative ↔ Positive)

Range: -1.0 to 1.0

Controls the positive/negative emotional tone:

| Value | Effect |
|-------|--------|
| -1.0 | Extremely negative: minor keys, darker tones |
| -0.5 | Moderately negative: minor key tendencies |
| 0.0 | Neutral: ambiguous tonality |
| 0.5 | Moderately positive: major key tendencies |
| 1.0 | Extremely positive: major keys, bright tones |

**Musical Impact**:

- Key selection (major vs minor)
- Chord quality
- Melodic contour direction

### Arousal (Calm ↔ Excited)

Range: 0.0 to 1.0

Controls the energy level:

| Value | Effect |
|-------|--------|
| 0.0 | Very calm: slow, sustained, sparse |
| 0.25 | Relaxed: moderate pace, legato |
| 0.5 | Balanced: medium tempo, varied articulation |
| 0.75 | Energetic: faster, more active |
| 1.0 | Very excited: fast, dense, staccato |

**Musical Impact**:

- Tempo adjustment
- Note density
- Articulation style
- Rhythmic complexity

### Dominance (Submissive ↔ Dominant)

Range: 0.0 to 1.0

Controls the sense of control or power:

| Value | Effect |
|-------|--------|
| 0.0 | Submissive: supportive role, gentle dynamics |
| 0.5 | Balanced: shared lead/support presence |
| 1.0 | Dominant: commanding lead, forward mix placement |

**Musical Impact**:

- Role emphasis (lead vs accompaniment)
- Dynamic weight and articulation strength
- Register choice and density
- Percussive/transient prominence

### Intensity (Subtle ↔ Extreme)

Range: 0.0 to 1.0

Controls the expressive intensity:

| Value | Effect |
|-------|--------|
| 0.0 | Very subtle: soft dynamics, simple patterns |
| 0.5 | Moderate: balanced dynamics, medium complexity |
| 1.0 | Extreme: strong dynamics, complex patterns |

**Musical Impact**:

- Velocity range
- Dynamic contrast
- Pattern complexity
- Harmonic density

## Generation Settings

### Bars

How many bars of MIDI to generate per generation cycle.

- **Range**: 1 to 8 bars
- **Recommendation**: Start with 4 bars for balanced phrases

### Generation Rate

How frequently the plugin generates new MIDI (measured in audio blocks).

- **Range**: 1 to 32 blocks
- **Lower values**: More frequent generation
- **Higher values**: Less frequent, more stable output

## Project Management

### Creating Projects

1. **New Project**: File > New Project
   - Clears all settings and generated MIDI
   - Resets to default parameters

### Saving Projects

1. **Save**: File > Save Project (Cmd/Ctrl + S)
   - Saves to current file if one exists
   - Prompts for location if new project

2. **Save As**: File > Save Project As
   - Always prompts for new location
   - Good for creating variations

### Project File Contents

The `.mkp` file stores:

- All parameter settings (including cassette/tape state)
- Emotion state (216-node selection, VAD sliders)
- Generated MIDI data (melody, bass, chords, drums, vocals)
- Vocal notes and lyrics
- Tempo and time signature
- ML toggle and model path (if enabled)
- Custom metadata

### Loading Projects

1. **Open**: File > Open Project
2. Select a `.mkp` file
3. All settings and MIDI are restored

### Project File Format

Projects use JSON format with the following structure:

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "My Project",
    "author": "Your Name",
    "created": "2025-01-01T00:00:00",
    "modified": "2025-01-01T12:00:00"
  },
  "tempo": 120.0,
  "timeSignature": {"numerator": 4, "denominator": 4},
  "pluginState": {
    "emotionState": {...},
    "parameters": {...}
  },
  "generatedMidi": [...]
}
```

## MIDI Export

### Basic Export

1. Click **Export MIDI** button
2. Choose location and filename
3. File is saved as standard MIDI (.mid)

### Export Options

When using Export > Export MIDI with Options:

| Option | Description |
|--------|-------------|
| Format | SMF Type 0 (single track) or Type 1 (multi-track) |
| PPQ | Pulses per quarter note resolution (e.g., 480 default) |
| Include Tempo | Embed tempo meta events |
| Include Time Signature | Embed time signature |
| Include Lyrics | Export lyric meta events (if present) |
| Include CC/Expression | Export CC data for dynamics/automation |
| Include Vocals | Enable/disable vocal track export |
| Quantize | Snap notes/lyrics to grid |
| Velocity Scaling | Apply per-track velocity/volume scaling |

### Export Statistics

After export, the status bar shows:

- Total notes exported
- File size
- Duration in beats
- Total lyrics/events exported (if applicable)

### MIDI File Compatibility

Exported MIDI files are compatible with:

- All major DAWs (Ableton, Logic, Cubase, FL Studio, etc.)
- Notation software (Sibelius, Finale, MuseScore)
- Other MIDI applications

## Cloud Generation

### Enabling Cloud

Toggle "Enable Cloud Generation" to use cloud-based AI inference.

**When to use**:

- For higher quality generation
- When local ONNX model isn't available
- For more complex emotional mappings

**Requirements**:

- Internet connection
- Cloud API endpoint configured (via environment variable)

### Environment Variables

```bash
# Set ONNX model path (for local inference)
export KELLY_ONNX_MODEL=/path/to/model.onnx

# Set cloud endpoint (for cloud inference)
export KELLY_AI_ENDPOINT=https://api.example.com/generate
```

## Advanced Usage

### Layering Multiple Instances

For richer arrangements:

1. Create multiple MIDI tracks
2. Add miDiKompanion to each
3. Set different emotions for each:
   - Track 1: Melody (Happy, high arousal)
   - Track 2: Bass (Calm, low arousal)
   - Track 3: Chords (Neutral, medium intensity)

### Automation

Most parameters can be automated in your DAW:

- Automate Valence for emotional arc
- Automate Arousal for energy builds
- Automate Intensity for dramatic moments

### Recording Generated MIDI

To capture generated MIDI for editing:

1. Arm the track for recording
2. Press record in your DAW
3. Let the plugin generate
4. Stop recording
5. Edit the recorded MIDI freely

### Parameter Combinations

Effective emotional presets:

| Mood | Valence | Arousal | Intensity |
|------|---------|---------|-----------|
| Epic | 0.7 | 0.9 | 0.8 |
| Tender | 0.3 | 0.2 | 0.3 |
| Mysterious | -0.2 | 0.4 | 0.5 |
| Triumphant | 0.9 | 0.8 | 0.9 |
| Melancholic | -0.5 | 0.3 | 0.4 |
| Playful | 0.6 | 0.7 | 0.5 |

## Keyboard Shortcuts

| Action | Windows | macOS |
|--------|---------|-------|
| New Project | Ctrl+N | Cmd+N |
| Open Project | Ctrl+O | Cmd+O |
| Save Project | Ctrl+S | Cmd+S |
| Save As | Ctrl+Shift+S | Cmd+Shift+S |
| Export MIDI | Ctrl+E | Cmd+E |

## Tips and Best Practices

1. **Start with presets**: Use base emotions before fine-tuning
2. **Small adjustments**: Change one parameter at a time
3. **Save often**: Save before major experiments
4. **Use variations**: Create multiple projects for different takes
5. **Export regularly**: Export MIDI to preserve good generations
6. **Layer intelligently**: Use different emotions for different parts

---

For troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
