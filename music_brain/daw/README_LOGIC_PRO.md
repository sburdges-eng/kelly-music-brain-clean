# Logic Pro Integration - Emotion to Mixer Automation

## Overview

This module provides a complete pipeline for translating emotional intent into Logic Pro mixer automation parameters. Following the "Interrogate Before Generate" philosophy, every mixer setting is justified by emotional intent, not arbitrary technical choices.

## Quick Start

### Simple Text-Based Generation

```python
from music_brain import MusicBrain

brain = MusicBrain()
music = brain.generate_from_text("grief and loss")

# Export to Logic Pro format
brain.export_to_logic(music, "my_song")
# Creates: my_song_automation.json
```

### Full Intent-Based Generation

```python
from music_brain import MusicBrain
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    SongRoot,
    SongIntent,
    TechnicalConstraints,
)

# Define your intent
intent = CompleteSongIntent(
    title="My Song",
    song_root=SongRoot(
        core_event="What happened",
        core_longing="What you want to feel",
    ),
    song_intent=SongIntent(
        mood_primary="grief",
        vulnerability_scale="High",
    ),
    technical_constraints=TechnicalConstraints(
        technical_key="F",
        technical_mode="major",
        technical_tempo_range=(78, 86),
    ),
)

# Generate
brain = MusicBrain()
music = brain.generate_from_intent(intent)

# Export
brain.export_to_logic(music, "output")
```

### Fluent API (Advanced Control)

```python
from music_brain import MusicBrain

brain = MusicBrain()
result = (brain.process("anxiety and tension")
               .map_to_emotion()
               .map_to_music()
               .with_tempo(110)
               .map_to_mixer()
               .export_logic("anxiety_track_automation.json"))
```

## Emotion Presets

Available mixer presets mapped to emotions:

| Emotion | Description | Key Characteristics |
|---------|-------------|---------------------|
| `grief` | Deep, spacious, lo-fi | Rolled-off highs, long reverb, tape saturation |
| `anxiety` | Tight, compressed, hyper-present | Fast compression, narrow stereo, boosted presence |
| `anger` | Aggressive, saturated, forward | Heavy saturation, boosted bass, minimal reverb |
| `nostalgia` | Warm, dreamy, vintage | Tape saturation, plate reverb, modulated delay |
| `hope` | Bright, open, lifting | Enhanced air frequencies, wide stereo |
| `calm` | Smooth, warm, peaceful | Soft compression, gentle EQ, chamber reverb |
| `tension` | Building, unsettled | Boosted sub bass, enhanced presence |
| `catharsis` | Full, releasing, overwhelming | Maximum stereo width, long reverb |
| `dissociation` | Distant, hazy, detached | Heavy reverb/delay, scooped mids |
| `intimacy` | Close, dry, personal | Minimal reverb, narrow stereo |

## Mixer Parameters Reference

### EQ (7 Bands)

| Band | Frequency Range | Emotional Association |
|------|-----------------|----------------------|
| Sub Bass | 20-60 Hz | Physical weight, power |
| Bass | 60-250 Hz | Warmth, foundation |
| Low Mid | 250-500 Hz | Body, mud |
| Mid | 500-2000 Hz | Presence, clarity |
| High Mid | 2-6 kHz | Edge, aggression |
| Presence | 6-12 kHz | Air, brightness |
| Air | 12-20 kHz | Shimmer, openness |

### Compression

| Parameter | Range | Emotional Effect |
|-----------|-------|------------------|
| Ratio | 1:1 - 20:1 | Higher = more control, intensity |
| Threshold | -60 to 0 dB | Lower = more compression |
| Attack | 0.1 - 100 ms | Fast = punchy; Slow = natural |
| Release | 10 - 500 ms | Fast = tight; Slow = breathing |
| Knee | 0 - 10 dB | Hard = aggressive; Soft = gentle |

### Reverb

| Type | Character | Use Case |
|------|-----------|----------|
| room | Small, intimate | Close, personal |
| hall | Large, classical | Epic, spacious |
| plate | Bright, vintage | Warm, nostalgic |
| chamber | Warm, diffuse | Enveloping |
| spring | Lo-fi, vintage | Character |
| shimmer | Ethereal, modulated | Otherworldly |

### Saturation

| Type | Character | Emotional Effect |
|------|-----------|------------------|
| tape | Warm, soft compression | Nostalgia, warmth |
| tube | Harmonic richness | Warmth, depth |
| transistor | Aggressive, odd harmonics | Anger, edge |
| digital | Hard, harsh | Damage, breakdown |

## Importing to Logic Pro

### Manual Import

1. **Open Logic Pro** and create a new project

2. **Set project tempo** from the exported settings

3. **Apply EQ** (Channel EQ plugin):
   - Read values from `channel_eq` section
   - Apply to each frequency band

4. **Apply Compression** (Compressor plugin):
   - Set ratio, threshold, attack, release from `compressor` section

5. **Apply Reverb** (Space Designer or ChromaVerb):
   - Use the `reverb.type` to select appropriate preset
   - Adjust mix, decay, predelay, size

6. **Apply Saturation** (Tape plugins or Phat FX):
   - Use the saturation type and amount

### Automation Export Format

The exported JSON has this structure:

```json
{
  "daw": "Logic Pro",
  "format_version": "1.0",
  "parameters": {
    "channel_eq": {
      "sub_bass_gain": -2.0,
      "bass_gain": 1.0,
      ...
    },
    "compressor": {
      "ratio": 2.5,
      "threshold": -18.0,
      ...
    },
    "reverb": {
      "type": "hall",
      "mix": 45,
      ...
    },
    ...
  },
  "metadata": {
    "description": "...",
    "tags": ["grief", "lo-fi"],
    "emotional_justification": "..."
  }
}
```

## Example: Kelly Song

```bash
python examples_music-brain/kelly_song_logic_export.py
```

This generates `kelly_song_automation.json` with:
- EQ settings (rolled-off highs for lo-fi aesthetic)
- Compression (gentle, for intimacy)
- Reverb (spacious hall for grief processing)
- Saturation (tape for warmth and imperfection)
- All parameters justified by emotional intent

## Philosophy

> "Interrogate Before Generate"

Each mixer setting is justified by emotional intent:

- **Grief**: Rolled-off highs create distance from reality
- **Anxiety**: Fast compression mimics racing heart
- **Anger**: Heavy saturation represents burning intensity
- **Nostalgia**: Lo-fi degradation represents imperfect memory

The audience doesn't hear "borrowed from parallel minor" - they hear "that part made me cry."

## API Reference

### MusicBrain

Main API class for emotion-to-music generation.

```python
class MusicBrain:
    def generate_from_intent(intent: CompleteSongIntent) -> GeneratedMusic
    def generate_from_text(emotional_text: str) -> GeneratedMusic
    def export_to_logic(music: GeneratedMusic, output_base: str) -> Dict[str, str]
    def process(emotional_text: str) -> FluentChain
    def create_intent(**kwargs) -> CompleteSongIntent
    def suggest_rules(emotion: str) -> List[Dict]
    def list_mixer_presets() -> List[str]
    def get_mixer_preset(emotion: str) -> MixerParameters
```

### GeneratedMusic

Result of music generation.

```python
@dataclass
class GeneratedMusic:
    emotional_state: EmotionalState
    musical_params: MusicalParameters
    mixer_params: MixerParameters
    intent: Optional[CompleteSongIntent]
    midi_path: Optional[str]
    automation_path: Optional[str]

    def to_dict() -> Dict[str, Any]
    def summary() -> str
```

### FluentChain

Fluent API for step-by-step control.

```python
class FluentChain:
    def map_to_emotion() -> FluentChain
    def map_to_music() -> FluentChain
    def map_to_mixer() -> FluentChain
    def with_tempo(tempo: int) -> FluentChain
    def with_dissonance(dissonance: float) -> FluentChain
    def with_timing(feel: str) -> FluentChain
    def export_logic(output_path: str) -> Dict[str, str]
    def export_json(output_path: str) -> str
    def get() -> Dict[str, Any]
    def describe() -> str
```

## Future Enhancements

- [ ] AppleScript automation for direct Logic Pro import
- [ ] Real-time OSC control of Logic Pro parameters
- [ ] Time-based automation curves (not just static settings)
- [ ] Plugin preset generation (AUPreset files)
- [ ] Integration with Logic Pro's Smart Tempo
