# Export Workflow Guide

This guide covers all aspects of exporting your generated music from MidiKompanion.

## MIDI File Export

### Quick Export

The fastest way to export your generated MIDI:

1. Generate some music using the emotion controls
2. Click **Export MIDI** button or **File → Export MIDI**
3. Choose a location and filename
4. Click **Save**

### Export with Options

For more control over the export:

1. Click **File → Export MIDI (Options)**
2. Configure the export settings:
   - **Format**: Type 0 or Type 1
   - **PPQ**: Pulses per quarter note (resolution)
3. Click **Export**
4. Choose file location

## MIDI File Formats

### Type 0 (Single Track)
- All MIDI data in one track
- Simpler structure
- Good for: Simple melodies, single-instrument pieces

### Type 1 (Multi-Track)
- Separate tracks for each instrument/part
- Includes tempo track (track 0)
- Good for: Complex arrangements, DAW import

**Recommendation**: Use Type 1 for maximum compatibility with DAWs.

## Export Options

### PPQ (Pulses Per Quarter Note)
Controls the timing resolution of the MIDI file:

| PPQ | Resolution | Use Case |
|-----|------------|----------|
| 96 | Basic | Simple rhythms |
| 120 | Standard | General use |
| 240 | High | Detailed rhythms |
| 480 | Very High | Complex timing (default) |
| 960 | Maximum | Extreme precision |

**Recommendation**: 480 PPQ provides good balance of precision and compatibility.

### Track Mapping

When exporting, tracks are mapped to MIDI channels:

| Track Type | MIDI Channel |
|------------|--------------|
| Melody | 1 |
| Bass | 2 |
| Chords | 3 |
| Vocals | 4 |
| Drums | 10 (GM standard) |

### Content Toggles

- **Include Tempo / Time Signature**: Embed timing meta-events.
- **Include Lyrics**: Export lyric events for vocal lines.
- **Include CC/Expression**: Export dynamics/automation data.
- **Include Vocals**: Enable/disable vocal track export.
- **Quantize**: Optional grid snap for notes/lyrics.
- **Velocity Scaling**: Honors per-track velocity/volume scaling.

### Included Data

The export includes:
- ✅ Note events (pitch, velocity, timing)
- ✅ Tempo information
- ✅ Time signature
- ✅ Track names
- ✅ Lyric events (if available)
- ⚠️ Expression data (if enabled)

## Project Save/Load

### Saving Projects

Projects save your complete session:

1. **File → Save Project** or **Cmd/Ctrl + S**
2. Choose location and filename (`.mkp` extension)
3. Click **Save**

**What's saved:**
- All emotion parameters
- Generated MIDI data
- Tempo and time signature
- Vocal notes and lyrics
- Plugin state

### Loading Projects

1. **File → Open Project** or **Cmd/Ctrl + O**
2. Select a `.mkp` file
3. Click **Open**

The project fully restores your previous session.

### Recent Projects

Quick access to recent work:
- **File → Recent Projects**
- Shows last 10 opened projects
- Click to open directly

## Best Practices

### Before Export

1. **Generate enough content**: Make sure you have the music you want
2. **Check tempo**: Verify the tempo matches your intent
3. **Preview if possible**: Listen to the generated content

### Export Workflow

1. Save your project first (backup)
2. Export MIDI with appropriate settings
3. Import into your DAW
4. Verify the import is correct
5. Make any DAW-specific adjustments

### After Export

1. Import the MIDI file into your DAW
2. Assign appropriate instruments/sounds
3. Adjust velocities and timing as needed
4. Mix and master your track

## DAW-Specific Tips

### Ableton Live
- Import MIDI file to arrangement view
- Drag tracks to MIDI tracks with instruments
- Use the Drum Rack for channel 10 drums

### Logic Pro
- File → Import → MIDI File
- Tracks are created automatically
- Assign software instruments

### FL Studio
- Drag MIDI file to playlist
- Choose "Split by channel" for separate patterns
- Assign instruments to each channel

### Pro Tools
- File → Import → MIDI
- Choose track creation options
- Map to instrument tracks

### Cubase
- File → Import → MIDI File
- Select import options
- Assign VST instruments

## Troubleshooting

### Export Button is Disabled
- No MIDI data has been generated
- Generate music first using the emotion controls

### MIDI File Won't Open in DAW
- Check file isn't corrupted (re-export)
- Try different export format (Type 0 vs Type 1)
- Verify file extension is `.mid` or `.midi`

### Wrong Tempo in DAW
- DAW may override with project tempo
- Check DAW's MIDI import tempo settings
- Manually set tempo after import

### Missing Notes
- Check if tracks were muted during export
- Verify all parameters were set correctly
- Try exporting with higher PPQ

### Timing Issues
- Increase PPQ for better resolution
- Check quantization settings in DAW
- Verify tempo is consistent

## Advanced Export

### Batch Export
Currently not supported. Export each project individually.

### Stem Export (Future)
Audio stem export will be available in a future version.

### MIDI 2.0 (Future)
Support for MIDI 2.0 features is planned.
