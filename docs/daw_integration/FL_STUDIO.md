# FL Studio Integration Guide

> Complete guide for using iDAW Penta Core plugin with FL Studio.

## Overview

iDAW Penta Core is fully compatible with FL Studio 20.8+ via VST3 format. The plugin provides real-time harmony analysis, groove extraction, and intent-driven music generation.

## Installation

### Windows (Primary Platform)

1. **Build the plugin:**
   ```bash
   cd iDAW
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target PentaCorePlugin_VST3
   ```

2. **Plugin location after build:**
   ```
   C:\Program Files\Common Files\VST3\Penta Core.vst3
   ```

3. **Verify in FL Studio:**
   - Open FL Studio
   - Go to `Options > Manage Plugins`
   - Click `Start scan` or `Find more plugins...`
   - Search for "Penta Core"
   - Enable the plugin

### macOS / Linux

Same build process. Plugin installed to:
- **macOS:** `~/Library/Audio/Plug-Ins/VST3/Penta Core.vst3`
- **Linux:** `~/.vst3/Penta Core.vst3`

## Usage in FL Studio

### Adding the Plugin

1. Open the Channel Rack or Mixer
2. Click `+` to add a new plugin
3. Search for "Penta Core" under Effects > Tools
4. Insert on a Mixer track or as a Generator

### MIDI Routing

Penta Core processes MIDI for harmony analysis and generation:

1. **As MIDI Effect:**
   - Add to a MIDI track
   - Route MIDI through the plugin
   - Enable "MIDI through" in plugin settings

2. **As Generator:**
   - Use the plugin's built-in intent input
   - Generated MIDI appears on the plugin's output
   - Route to other instruments via MIDI Out

### Parameter Automation

All Penta Core parameters can be automated in FL Studio:

1. Right-click any plugin knob
2. Select "Link to controller..."
3. Choose automation clip or MIDI CC

**Key Parameters:**
| Parameter | CC | Description |
|-----------|-----|-------------|
| Harmony Mode | CC 1 | Major/Minor/Modal selection |
| Groove Intensity | CC 11 | Humanization amount |
| Swing Amount | CC 74 | Swing timing adjustment |
| Intent Temperature | CC 75 | AI generation creativity |

## FL Studio Project Template

A pre-configured FL Studio template is available:

**Location:** `iDAW/templates/fl_studio/iDAW_Starter.flp`

**Template includes:**
- Penta Core on Track 1 (MIDI processor)
- Pre-routed MIDI channels
- Automation clips for common parameters
- Mixer routing for generated content

## Workflow Examples

### 1. Chord Progression Analysis

1. Record or draw MIDI chords in Piano Roll
2. Route to Penta Core
3. View real-time chord analysis in plugin UI
4. Export analysis as JSON for further processing

### 2. Intent-Driven Generation

1. Open Penta Core UI
2. Enter emotional intent (e.g., "melancholic yearning")
3. Set key and scale
4. Click "Generate"
5. MIDI output routed to your instruments

### 3. Groove Humanization

1. Quantized MIDI → Penta Core
2. Select genre template (funk, jazz, etc.)
3. Adjust swing and pocket settings
4. Humanized MIDI output

## Troubleshooting

### Plugin Not Found
- Verify VST3 is in FL Studio's plugin paths
- Run plugin scan in `Options > Manage Plugins`
- Check that VST3 format is enabled

### MIDI Not Passing Through
- Enable "MIDI through" in plugin
- Check MIDI routing in Mixer
- Verify MIDI channel settings

### High CPU Usage
- Reduce buffer size in plugin settings
- Disable unused analysis features
- Use "Low latency" mode

### Automation Issues
- Ensure CC mapping matches plugin parameters
- Check for parameter smoothing settings
- Update to latest plugin version

## Performance Tips

1. **Buffer Settings:**
   - FL Studio: 512-1024 samples recommended
   - Plugin internal: Match FL Studio buffer

2. **Multi-core:**
   - Enable multi-threaded processing in FL Studio
   - Penta Core uses SIMD optimizations automatically

3. **Real-time Analysis:**
   - Disable unused engines (harmony/groove/diagnostics)
   - Use "Essential" mode for live performance

## Version Compatibility

| FL Studio Version | Penta Core Support |
|-------------------|-------------------|
| FL Studio 21+ | Full VST3 support |
| FL Studio 20.8+ | Full VST3 support |
| FL Studio 20.0-20.7 | VST3 (limited features) |
| FL Studio 12-19 | Not supported (VST2 only) |

## Support

- **Issues:** https://github.com/iDAW/issues
- **Documentation:** https://idaw.dev/docs/fl-studio
- **Community:** https://discord.gg/idaw

---

*"The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"*
