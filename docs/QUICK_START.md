# miDiKompanion Quick Start Guide

Welcome to miDiKompanion! This guide will help you get started with emotion-driven music generation.

## Table of Contents

1. [Installation](#installation)
2. [First Steps](#first-steps)
3. [Emotion Wheel Usage](#emotion-wheel-usage)
4. [Creating Your First Project](#creating-your-first-project)
5. [Exporting MIDI](#exporting-midi)
6. [Troubleshooting](#troubleshooting)

## Installation

### Requirements

- macOS 13.0+ (Apple Silicon native; Intel via Rosetta), Windows 11 22H2+, or iOS/iPadOS 17+ for AUv3 on mobile. Linux is experimental (CLAP/VST3 where supported).
- DAW supporting VST3 or AUv3 (CLAP supported on Bitwig/Reaper); see `docs/daw_integration/HOST_SUPPORT_MATRIX.md` for host versions.
- 4GB RAM minimum (8GB recommended); 2GB free disk space for models/cache.

### Plugin Installation

1. Download the latest release from the releases page
2. Copy the plugin to your DAW's plugin folder:
   - **macOS AU**: `/Library/Audio/Plug-Ins/Components/`
   - **macOS VST3**: `/Library/Audio/Plug-Ins/VST3/`
   - **Windows VST3**: `C:\Program Files\Common Files\VST3\`
3. Rescan plugins in your DAW
4. Insert miDiKompanion on a MIDI track

### Standalone Installation

1. Download the standalone application
2. Run the installer
3. Launch miDiKompanion from your Applications folder

## First Steps

### Opening miDiKompanion

1. In your DAW, create a new MIDI track
2. Insert miDiKompanion as a MIDI effect
3. The Emotion Workstation will open

### Interface Overview

```
┌─────────────────────────────────────────────────┐
│  miDiKompanion Emotion Workstation              │
├─────────────────────────────────────────────────┤
│                                                 │
│        [Emotion Wheel - 216 nodes]              │
│                                                 │
├─────────────────────────────────────────────────┤
│  Valence: [───●───]  Arousal: [───●───]        │
│  Intensity: [───●───]  Dominance: [───●───]    │
├─────────────────────────────────────────────────┤
│  [Generate]  [Export MIDI]  [Save Project]     │
└─────────────────────────────────────────────────┘
```

## Emotion Wheel Usage

### Understanding the 216-Node Structure

miDiKompanion uses a hierarchical emotion structure:

- **6 Base Emotions**: Happy, Sad, Angry, Fear, Surprise, Disgust
- **36 Sub-Emotions**: 6 variations per base emotion
- **216 Nodes**: 6 intensity levels per sub-emotion

### Selecting an Emotion

1. **Click** on the emotion wheel to select a base emotion
2. **Drag** to refine the selection to a specific node
3. The sliders will update to show the VAD coordinates:
   - **Valence**: Negative (-1) to Positive (+1)
   - **Arousal**: Calm (-1) to Excited (+1)
   - **Dominance**: Submissive (-1) to Dominant (+1)
   - **Intensity**: Subtle (0) to Extreme (1)

### Using the Sliders

You can also set emotions directly using the sliders:

1. Adjust **Valence** for mood (sad → happy)
2. Adjust **Arousal** for energy (calm → excited)
3. Adjust **Intensity** for expression strength
4. The wheel will highlight matching emotion nodes

### ML Enhancement (Optional)

Enable ML enhancement for more nuanced generation:

1. Click the **ML Toggle** button
2. Adjust the **ML Blend** slider (0 = rule-based, 1 = ML-only)
3. The hybrid mode combines rule-based structure with ML predictions

## Creating Your First Project

### Step 1: Set the Mood

1. Select an emotion that matches your musical vision
2. For example: "Hopeful (Strong)" for an uplifting melody

### Step 2: Configure Parameters

1. Set **Tempo** using your DAW's transport or the plugin's tempo field
2. Set **Time Signature** if needed (default: 4/4)
3. Choose **Number of Bars** to generate (1-8)

### Step 3: Generate Music

1. Click **Generate** to create MIDI
2. The plugin will generate:
   - Melody track
   - Bass track
   - Chord track
   - Drum pattern
3. Listen to the result using your DAW's playback

### Step 4: Refine

1. Adjust emotion parameters and regenerate
2. Use the **Suggestion Panel** for related emotions
3. Try the **Intensity Ladder** to explore variations

### Step 5: Save Your Project

1. Click **Save Project** (or File → Save)
2. Choose a location and filename
3. The `.mkp` file saves:
   - All emotion settings (216-node selection and VAD sliders)
   - Generated MIDI data (melody, bass, chords, drums, vocals)
   - Plugin parameters and cassette state
   - Lyrics and vocal notes
   - ML toggle and model path (if enabled)

## Exporting MIDI

### Basic Export

1. Click **Export MIDI** button
2. Choose a filename and location
3. Select format options:
   - **Type 0**: Single track (all instruments merged)
   - **Type 1**: Multi-track (separate tracks per instrument)

### Export Options

- **Include Tempo**: Embeds tempo changes in the MIDI file
- **Include Time Signature**: Adds time signature events
- **Include Lyrics**: Adds lyric events (if vocals generated)
- **Include Expression**: Adds CC data for dynamics
- **Include Vocals**: Toggle vocal track export on/off
- **Quantize**: Optional grid snapping for note/lyric timing
- **Velocity Scaling**: Honors per-track velocity/volume scaling

### Track Selection

Choose which tracks to export:
- ✓ Melody
- ✓ Bass
- ✓ Chords
- ✓ Drums
- ✓ Vocals (if available)

### Using Exported MIDI

1. Import the MIDI file into your DAW
2. Assign instruments to each track
3. Edit and arrange as needed

## Troubleshooting

### Plugin Doesn't Load

1. Verify the plugin is in the correct folder
2. Check your DAW's plugin scan log
3. Try rescanning plugins
4. Ensure your system meets minimum requirements

### No Audio Output

miDiKompanion generates MIDI, not audio. To hear output:
1. Route the MIDI to a virtual instrument
2. Or use the built-in preview sounds (standalone only)

### Generation is Slow

1. Reduce the number of bars
2. Disable ML enhancement if not needed
3. Check CPU usage in your DAW

### Project Won't Load

1. Check if the file is a valid `.mkp` file
2. Try opening a backup if available
3. Check the console for error messages
4. Verify the file extension is `.mkp`

### MIDI Export Issues

1. Ensure there's generated MIDI to export
2. Check write permissions for the destination folder
3. If vocals are missing, enable **Include Vocals** in export options
4. Try exporting to a different location

## Next Steps

- Read the [User Guide](USER_GUIDE.md) for advanced features
- Explore the [Emotion Wheel](EMOTION_WHEEL.md) for emotion mapping details
- Learn about [ML Training](ML_TRAINING_GUIDE.md) to customize models
- Join our [Community Forum](https://community.midikompanion.com) for tips and support

## Keyboard Shortcuts

| Action | macOS | Windows |
|--------|-------|---------|
| Save Project | ⌘S | Ctrl+S |
| Open Project | ⌘O | Ctrl+O |
| Export MIDI | ⌘E | Ctrl+E |
| Generate | Space | Space |
| Undo | ⌘Z | Ctrl+Z |
| Redo | ⌘⇧Z | Ctrl+Shift+Z |

## Getting Help

- **Documentation**: [docs.midikompanion.com](https://docs.midikompanion.com)
- **Issues**: [GitHub Issues](https://github.com/midikompanion/issues)
- **Email**: support@midikompanion.com

Happy music making! 🎵
