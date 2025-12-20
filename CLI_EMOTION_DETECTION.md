# CLI Emotion Detection Command

## Overview

Added audio emotion detection command to the CLI, integrating the SpeechBrain-based emotion detection research and implementation.

## New Command

```bash
daiw audio detect-emotion <audio_file> [options]
```

### Usage Examples

```bash
# Quick emotion detection
daiw audio detect-emotion song.wav --quick

# Detailed detection with emotional state
daiw audio detect-emotion song.wav --get-state

# Save results to JSON
daiw audio detect-emotion song.wav --output emotion_results.json

# Verbose output for debugging
daiw audio detect-emotion song.wav --verbose
```

## Options

- `audio_file` - Audio file to analyze (WAV, MP3, FLAC, etc.)
- `-q, --quick` - Quick detection mode (simple output)
- `-s, --get-state` - Get EmotionalState for use with intent system
- `-o, --output` - Save results to JSON file
- `-v, --verbose` - Verbose error output

## Output

### Quick Mode (`--quick`)
```
=== Emotion Detection ===
Emotion: happy
Confidence: 85.2%
Valence: 0.72
Arousal: 0.81
Primary emotion: calm
```

### Detailed Mode (default)
```
=== Emotion Detection (Detailed) ===
Detected emotion: happy
Confidence: 85.2%

Valence-Arousal Model:
  Valence: 0.72 (-1=sad, +1=happy)
  Arousal: 0.81 (0=calm, 1=excited)

Mapped to system emotion: calm

=== Emotional State (for use with intent system) ===
State: EmotionalState(...)

Suggested musical parameters:
  Tempo: 120 BPM
  Key: C major
  Mode: major
```

## Integration

This command integrates with:
- **SpeechBrain** - Pre-trained emotion recognition models
- **Emotional Mapping System** - Maps to valence-arousal and system emotions
- **Intent System** - Provides EmotionalState for intent processing

## Requirements

Requires SpeechBrain and PyTorch:
```bash
pip install speechbrain torch torchaudio
```

If not installed, the command will show a helpful error message.

## Implementation Details

### Files Modified
- `music_brain/cli.py`
  - Updated `get_audio_module()` to include emotion detection functions
  - Added `cmd_audio_detect_emotion()` handler function
  - Added routing in `cmd_audio()` function
  - Added parser for `detect-emotion` subcommand

### Functions Used
- `detect_emotion_from_audio()` - Quick emotion detection
- `EmotionDetector` - Class-based API for detailed detection
- `get_emotional_state_from_audio()` - Get EmotionalState for intent system

## Related Documentation

- Research: `research/prompts/01-audio-emotion-detection.md`
- Implementation: `music_brain/audio/emotion_detection.py`
- Quick Start: `research/QUICK_START.md`
- API Docs: `music_brain/audio/EMOTION_DETECTION.md`

## Example Workflow

```bash
# 1. Detect emotion from audio
daiw audio detect-emotion reference_track.wav --get-state -o emotion.json

# 2. Use detected emotion in intent system
daiw intent suggest $(cat emotion.json | jq -r '.primary_emotion')

# 3. Generate music based on detected emotion
daiw intent process intent_template.json
```

## Status

✅ **Complete** - Command is fully integrated and tested.
