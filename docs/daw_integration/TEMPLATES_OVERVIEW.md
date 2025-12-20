# DAW Template Projects Overview

> Pre-configured project templates for all supported DAWs.

## Available Templates

| DAW | Format | Location | Plugin Format |
|-----|--------|----------|---------------|
| Logic Pro X | .logicx / JSON | `templates/logic_pro/` | AU |
| Ableton Live | .als / JSON | `templates/ableton_live/` | VST3 |
| FL Studio | .flp / JSON | `templates/fl_studio/` | VST3 |
| Pro Tools | .ptxt / JSON | `templates/pro_tools/` | AAX |
| Reaper | .RPP / JSON | `templates/reaper/` | VST3/OSC |

## Template Contents

Each template includes:

### Track Configuration
- **Penta Core Track:** Pre-loaded with iDAW plugin
- **Input Track:** MIDI input routing
- **Instrument Tracks:** Piano, strings, bass, drums
- **Effects Returns:** Reverb, delay

### Plugin Settings
- Harmony analysis enabled
- Groove extraction active
- OSC communication configured
- Default parameters optimized

### Routing
- MIDI flow from input to Penta Core
- Generated MIDI to instruments
- Effects sends configured
- Master bus processing

## Installation

### Logic Pro X

```bash
# Copy template to Logic templates folder
cp templates/logic_pro/iDAW_Starter.logicx \
   ~/Music/Audio\ Music\ Apps/Project\ Templates/

# Or import JSON via Logic Pro settings
```

### Ableton Live

```bash
# Copy to User Library
cp templates/ableton_live/iDAW_Starter.als \
   ~/Music/Ableton/User\ Library/Templates/

# Drag .als into Live to open
```

### FL Studio

```bash
# Copy to FL Studio templates
cp templates/fl_studio/iDAW_Starter.flp \
   ~/Documents/Image-Line/FL\ Studio/Projects/Templates/
```

### Pro Tools

```bash
# Copy to Pro Tools templates
cp templates/pro_tools/iDAW_Starter.ptx \
   ~/Documents/Pro\ Tools/Session\ Templates/
```

### Reaper

```bash
# Copy to Reaper templates
cp templates/reaper/iDAW_Starter.RPP \
   ~/.config/REAPER/ProjectTemplates/
```

## Template Structure

```
templates/
├── logic_pro/
│   ├── iDAW_Starter_Template.json    # Template configuration
│   └── README.md                      # Setup instructions
├── ableton_live/
│   ├── iDAW_Starter_Template.json
│   ├── devices/                       # Max for Live devices
│   │   ├── iDAW_Bridge.amxd
│   │   └── iDAW_Chord_Display.amxd
│   └── README.md
├── fl_studio/
│   ├── iDAW_Starter_Template.json
│   └── README.md
├── pro_tools/
│   ├── iDAW_Starter_Template.json
│   └── README.md
└── reaper/
    ├── iDAW_OSC.ReaperOSC            # OSC control surface
    ├── scripts/                       # ReaScripts
    │   └── iDAW_Analyze_Selection.lua
    ├── effects/                       # JSFX plugins
    │   └── penta_core.jsfx
    └── README.md
```

## Customization

### Creating Custom Templates

1. **Open base template** in your DAW
2. **Modify** tracks, routing, plugins as needed
3. **Save as template** in DAW's template format
4. **Export JSON** (optional) for version control

### JSON Configuration Format

All templates share a common JSON structure:

```json
{
  "template_name": "My Custom Template",
  "daw_version": "11.0+",
  "description": "Custom template for specific workflow",
  "tracks": [...],
  "plugins": [...],
  "routing": {...},
  "setup_instructions": [...]
}
```

### Track Schema

```json
{
  "name": "Track Name",
  "type": "midi|audio|aux|master",
  "color": "blue|#hex",
  "plugins": [
    {
      "slot": "instrument|insert_1|etc",
      "name": "Plugin Name",
      "preset": "Preset Name"
    }
  ],
  "input": "source",
  "output": "destination"
}
```

## DAW-Specific Notes

### Logic Pro X
- Uses Audio Units (AU) format
- Smart Controls pre-mapped
- Environment routing configured
- Screensets for different workflows

### Ableton Live
- Uses VST3 format
- Push controller mappings included
- Groove Pool templates
- Max for Live devices available

### FL Studio
- Uses VST3 format
- Pattern-based template
- Automation clips pre-configured
- Mixer routing set up

### Pro Tools
- Uses AAX format
- Requires AAX SDK for development
- I/O setup matches common interfaces
- Session import/export supported

### Reaper
- Uses VST3 + OSC
- JSFX plugin for analysis
- ReaScript actions included
- Custom control surface

## Workflow Examples

### Composition Workflow

1. Open iDAW template
2. Play MIDI on input track
3. View chord analysis in Penta Core
4. Generate harmony based on intent
5. Route to instruments
6. Arrange and produce

### Analysis Workflow

1. Import MIDI file
2. Route to Penta Core track
3. Play and analyze
4. Export analysis results
5. Use insights for arrangement

### Humanization Workflow

1. Record quantized MIDI
2. Apply Penta Core groove
3. Adjust swing and pocket
4. Render humanized MIDI
5. Continue production

## Support

- **Issues:** https://github.com/iDAW/issues
- **Documentation:** https://idaw.dev/docs/templates
- **Community:** https://discord.gg/idaw

---

*"The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"*
