# KELLY PROJECT - Complete Code Consolidation
**Date:** December 8, 2025  
**Version:** 0.1.0  
**Status:** Active Development

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Python Code](#core-python-code)
4. [Tests](#tests)
5. [Configuration](#configuration)
6. [Quick Reference](#quick-reference)
7. [Development Workflow](#development-workflow)

---

## Project Overview

**Kelly** is a therapeutic interactive Digital Audio Workstation (iDAW) that translates emotions into music using a unique three-phase intent system:

1. **Wound** → Identify the emotional trigger
2. **Emotion** → Map to the 216-node emotion thesaurus  
3. **Rule-breaks** → Express through intentional musical violations

### Philosophy
> "Interrogate Before Generate" — The tool shouldn't finish art for people; it should make them braver.

### Tech Stack
| Component | Technology |
|-----------|------------|
| Brain | Python 3.11 (music21, librosa, mido) |
| Body | C++20 (JUCE 7, Qt 6, CMake) |
| Plugins | CLAP 1.2, VST3 3.7 |
| Audio | CoreAudio (macOS), ASIO (Windows), JACK (Linux) |
| CLI | Typer + Rich |

---

## Architecture

### Directory Structure (kelly str/)
```
Kelly/
├── src/
│   └── kelly/              # Python package
│       ├── __init__.py     # Package exports
│       ├── cli.py          # CLI interface
│       └── core/           # Core modules
│           ├── emotion_thesaurus.py
│           ├── intent_processor.py
│           └── midi_generator.py
├── tests/
│   ├── python/             # pytest tests
│   └── cpp/                # Catch2 tests
├── docs/                   # Documentation
├── .github/workflows/      # CI configuration
├── CMakeLists.txt          # C++ build
├── pyproject.toml          # Python config
└── README.md
```

---

## Core Python Code

### 1. Package Init (`src/kelly/__init__.py`)
```python
"""Kelly - Therapeutic iDAW translating emotions to music."""

__version__ = "0.1.0"
__author__ = "Kelly Development Team"

from kelly.core.emotion_thesaurus import EmotionThesaurus
from kelly.core.intent_processor import IntentProcessor
from kelly.core.midi_generator import MidiGenerator

__all__ = ["EmotionThesaurus", "IntentProcessor", "MidiGenerator"]
```

---

### 2. CLI Interface (`src/kelly/cli.py`)
```python
"""Command-line interface for Kelly."""
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from kelly.core.emotion_thesaurus import EmotionThesaurus
from kelly.core.intent_processor import IntentProcessor, Wound
from kelly.core.midi_generator import MidiGenerator

app = typer.Typer(
    name="kelly",
    help="Kelly - Therapeutic iDAW translating emotions to music"
)
console = Console()


@app.command()
def list_emotions() -> None:
    """List all available emotions in the thesaurus."""
    thesaurus = EmotionThesaurus()
    
    table = Table(title="Kelly Emotion Thesaurus")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Category", style="green")
    table.add_column("Intensity", style="yellow")
    table.add_column("Valence", style="blue")
    table.add_column("Arousal", style="red")
    
    for node in thesaurus.nodes.values():
        table.add_row(
            str(node.id),
            node.name,
            node.category.value,
            f"{node.intensity:.2f}",
            f"{node.valence:+.2f}",
            f"{node.arousal:.2f}"
        )
    
    console.print(table)


@app.command()
def process(
    wound: str = typer.Argument(..., help="Description of the wound/trigger"),
    intensity: float = typer.Option(0.7, help="Intensity of the wound (0.0-1.0)"),
    output: Optional[Path] = typer.Option(None, help="Output MIDI file path"),
    tempo: int = typer.Option(120, help="Tempo in BPM"),
    groove: str = typer.Option("straight", help="Groove template (straight/swing/syncopated)")
) -> None:
    """
    Process a therapeutic wound and generate music.
    
    Example:
        kelly process "feeling of loss" --intensity 0.8 --output output.mid
    """
    console.print(f"\n[bold cyan]Processing wound:[/bold cyan] {wound}")
    console.print(f"[cyan]Intensity:[/cyan] {intensity}\n")
    
    # Process intent
    processor = IntentProcessor()
    wound_obj = Wound(
        description=wound,
        intensity=intensity,
        source="user_input"
    )
    
    result = processor.process_intent(wound_obj)
    
    # Display results
    emotion = result["emotion"]
    console.print(f"[bold green]Mapped Emotion:[/bold green] {emotion.name}")
    console.print(f"  Category: {emotion.category.value}")
    console.print(f"  Valence: {emotion.valence:+.2f}")
    console.print(f"  Arousal: {emotion.arousal:.2f}\n")
    
    console.print(f"[bold yellow]Rule Breaks:[/bold yellow]")
    for rb in result["rule_breaks"]:
        console.print(f"  • {rb.rule_type}: {rb.description} (severity: {rb.severity:.2f})")
    
    # Generate MIDI
    generator = MidiGenerator(tempo=tempo)
    params = result["musical_params"]
    
    mode = params.get("mode", "minor")
    allow_dissonance = params.get("allow_dissonance", False)
    
    chord_progression = generator.generate_chord_progression(
        mode=mode,
        allow_dissonance=allow_dissonance
    )
    
    # Save or display
    if output:
        midi_file = generator.create_midi_file(chord_progression, groove, str(output))
        console.print(f"\n[bold green]✓[/bold green] MIDI file saved to: {output}")
    else:
        console.print(f"\n[dim]No output file specified. Use --output to save MIDI.[/dim]")
    
    console.print(f"\n[bold]Musical Parameters:[/bold]")
    console.print(f"  Mode: {mode}")
    console.print(f"  Tempo: {tempo} BPM")
    console.print(f"  Groove: {groove}")
    console.print(f"  Dissonance: {allow_dissonance}\n")


@app.command()
def version() -> None:
    """Show Kelly version."""
    from kelly import __version__
    rprint(f"[bold cyan]Kelly[/bold cyan] version [green]{__version__}[/green]")


if __name__ == "__main__":
    app()
```

---

### 3. Emotion Thesaurus (`src/kelly/core/emotion_thesaurus.py`)
```python
"""Core emotion processing and thesaurus."""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class EmotionCategory(Enum):
    """Primary emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


@dataclass
class EmotionNode:
    """Represents a node in the 216-node emotion thesaurus."""
    id: int
    name: str
    category: EmotionCategory
    intensity: float  # 0.0 to 1.0
    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    related_emotions: List[int]
    musical_attributes: Dict[str, any]


class EmotionThesaurus:
    """
    216-node emotion thesaurus for mapping emotions to musical properties.
    
    The thesaurus organizes emotions in a hierarchical structure with
    dimensions of valence, arousal, and intensity.
    """
    
    def __init__(self) -> None:
        """Initialize the emotion thesaurus."""
        self.nodes: Dict[int, EmotionNode] = {}
        self._initialize_thesaurus()
    
    def _initialize_thesaurus(self) -> None:
        """Initialize the 216-node emotion network."""
        base_emotions = [
            (0, "euphoria", EmotionCategory.JOY, 1.0, 1.0, 1.0),
            (1, "contentment", EmotionCategory.JOY, 0.5, 0.7, 0.3),
            (2, "grief", EmotionCategory.SADNESS, 1.0, -0.9, 0.7),
            (3, "melancholy", EmotionCategory.SADNESS, 0.6, -0.6, 0.3),
            (4, "rage", EmotionCategory.ANGER, 1.0, -0.8, 1.0),
            (5, "annoyance", EmotionCategory.ANGER, 0.4, -0.4, 0.5),
            (6, "terror", EmotionCategory.FEAR, 1.0, -0.9, 1.0),
            (7, "anxiety", EmotionCategory.FEAR, 0.6, -0.5, 0.8),
        ]
        
        for node_id, name, category, intensity, valence, arousal in base_emotions:
            self.nodes[node_id] = EmotionNode(
                id=node_id,
                name=name,
                category=category,
                intensity=intensity,
                valence=valence,
                arousal=arousal,
                related_emotions=[],
                musical_attributes={
                    "tempo_modifier": 1.0 + (arousal - 0.5) * 0.5,
                    "mode": "major" if valence > 0 else "minor",
                    "dynamics": intensity,
                }
            )
    
    def get_emotion(self, emotion_id: int) -> Optional[EmotionNode]:
        """Get emotion node by ID."""
        return self.nodes.get(emotion_id)
    
    def find_emotion_by_name(self, name: str) -> Optional[EmotionNode]:
        """Find emotion by name."""
        for node in self.nodes.values():
            if node.name.lower() == name.lower():
                return node
        return None
    
    def get_nearby_emotions(
        self, emotion_id: int, threshold: float = 0.3
    ) -> List[EmotionNode]:
        """Find emotions near the given emotion in emotional space."""
        source = self.get_emotion(emotion_id)
        if not source:
            return []
        
        nearby = []
        for node in self.nodes.values():
            if node.id == emotion_id:
                continue
            
            distance = (
                (source.valence - node.valence) ** 2 +
                (source.arousal - node.arousal) ** 2 +
                (source.intensity - node.intensity) ** 2
            ) ** 0.5
            
            if distance < threshold:
                nearby.append(node)
        
        return nearby
```

---

### 4. Intent Processor (`src/kelly/core/intent_processor.py`)
```python
"""Three-phase intent processing: Wound → Emotion → Rule-breaks."""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionNode


class IntentPhase(Enum):
    """Three phases of intent processing."""
    WOUND = "wound"
    EMOTION = "emotion"
    RULE_BREAK = "rule_break"


@dataclass
class Wound:
    """Represents the initial wound or trigger."""
    description: str
    intensity: float
    source: str
    timestamp: Optional[float] = None


@dataclass
class RuleBreak:
    """Represents intentional musical rule violations."""
    rule_type: str
    severity: float
    description: str
    musical_impact: Dict[str, any]


class IntentProcessor:
    """Processes therapeutic intent through three phases."""
    
    def __init__(self) -> None:
        self.thesaurus = EmotionThesaurus()
        self.wound_history: List[Wound] = []
        self.rule_breaks: List[RuleBreak] = []
    
    def process_wound(self, wound: Wound) -> EmotionNode:
        """Phase 1: Process a wound and map it to an emotion."""
        self.wound_history.append(wound)
        
        if "loss" in wound.description.lower() or "grief" in wound.description.lower():
            emotion = self.thesaurus.find_emotion_by_name("grief")
        elif "anger" in wound.description.lower() or "rage" in wound.description.lower():
            emotion = self.thesaurus.find_emotion_by_name("rage")
        elif "fear" in wound.description.lower() or "anxiety" in wound.description.lower():
            emotion = self.thesaurus.find_emotion_by_name("anxiety")
        else:
            emotion = self.thesaurus.find_emotion_by_name("melancholy")
        
        return emotion if emotion else self.thesaurus.nodes[0]
    
    def emotion_to_rule_breaks(self, emotion: EmotionNode) -> List[RuleBreak]:
        """Phase 2-3: Convert emotion to musical rule-breaks."""
        rule_breaks = []
        
        if emotion.intensity > 0.8:
            rule_breaks.append(RuleBreak(
                rule_type="dynamics",
                severity=emotion.intensity,
                description="Extreme dynamic contrasts",
                musical_impact={"velocity_range": (10, 127), "sudden_changes": True}
            ))
        
        if emotion.valence < -0.5:
            rule_breaks.append(RuleBreak(
                rule_type="harmony",
                severity=abs(emotion.valence),
                description="Dissonant intervals and clusters",
                musical_impact={"allow_dissonance": True, "cluster_probability": abs(emotion.valence)}
            ))
        
        if emotion.arousal > 0.7:
            rule_breaks.append(RuleBreak(
                rule_type="rhythm",
                severity=emotion.arousal,
                description="Irregular rhythms and syncopation",
                musical_impact={"syncopation_level": emotion.arousal, "irregular_meters": True}
            ))
        
        self.rule_breaks.extend(rule_breaks)
        return rule_breaks
    
    def process_intent(self, wound: Wound) -> Dict[str, any]:
        """Complete three-phase intent processing."""
        emotion = self.process_wound(wound)
        rule_breaks = self.emotion_to_rule_breaks(emotion)
        
        return {
            "wound": wound,
            "emotion": emotion,
            "rule_breaks": rule_breaks,
            "musical_params": self._compile_musical_params(emotion, rule_breaks)
        }
    
    def _compile_musical_params(self, emotion: EmotionNode, rule_breaks: List[RuleBreak]) -> Dict[str, any]:
        params = emotion.musical_attributes.copy()
        for rb in rule_breaks:
            params.update(rb.musical_impact)
        return params
```

---

### 5. MIDI Generator (`src/kelly/core/midi_generator.py`)
```python
"""MIDI generation and pipeline."""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import mido


@dataclass
class GrooveTemplate:
    """Represents a rhythmic groove template."""
    name: str
    time_signature: Tuple[int, int]
    pattern: List[Tuple[float, int]]
    swing: float = 0.0


class MidiGenerator:
    """MIDI pipeline for generating therapeutic musical output."""
    
    def __init__(self, tempo: int = 120) -> None:
        self.tempo = tempo
        self.groove_templates = self._initialize_grooves()
    
    def _initialize_grooves(self) -> Dict[str, GrooveTemplate]:
        return {
            "straight": GrooveTemplate(
                name="Straight", time_signature=(4, 4),
                pattern=[(0.0, 100), (0.25, 80), (0.5, 100), (0.75, 80)]
            ),
            "swing": GrooveTemplate(
                name="Swing", time_signature=(4, 4),
                pattern=[(0.0, 100), (0.33, 80), (0.5, 100), (0.83, 80)], swing=0.66
            ),
            "syncopated": GrooveTemplate(
                name="Syncopated", time_signature=(4, 4),
                pattern=[(0.0, 100), (0.125, 60), (0.375, 90), (0.625, 85), (0.875, 70)]
            )
        }
    
    def generate_chord_progression(
        self, mode: str = "minor", length: int = 4, allow_dissonance: bool = False
    ) -> List[List[int]]:
        if mode == "minor":
            base_progression = [[57, 60, 64], [57, 62, 65], [59, 64, 67], [57, 60, 64]]
        else:
            base_progression = [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]]
        
        if allow_dissonance:
            for chord in base_progression:
                chord.append(chord[0] + 13)
        
        return base_progression[:length]
    
    def apply_groove(self, notes: List[int], groove_template: GrooveTemplate, duration_bars: int = 1) -> List[mido.Message]:
        messages = []
        ticks_per_beat = 480
        beats_per_bar = groove_template.time_signature[0]
        
        for bar in range(duration_bars):
            for beat_time, velocity in groove_template.pattern:
                absolute_time = int((bar * beats_per_bar + beat_time * beats_per_bar) * ticks_per_beat)
                
                for note in notes:
                    messages.append(mido.Message('note_on', note=note, velocity=velocity, time=absolute_time))
                
                off_time = int(absolute_time + ticks_per_beat * 0.25)
                for note in notes:
                    messages.append(mido.Message('note_off', note=note, velocity=0, time=off_time))
        
        return messages
    
    def create_midi_file(
        self, chord_progression: List[List[int]], groove: str = "straight", output_path: Optional[str] = None
    ) -> mido.MidiFile:
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.tempo)))
        groove_template = self.groove_templates.get(groove, self.groove_templates["straight"])
        
        for chord in chord_progression:
            messages = self.apply_groove(chord, groove_template)
            track.extend(messages)
        
        if output_path:
            mid.save(output_path)
        
        return mid
```

---

## Quick Reference

### CLI Commands
```bash
kelly list-emotions                              # List emotions in thesaurus
kelly process "feeling of loss" --intensity 0.8 --output output.mid --tempo 90
kelly version
```

### Development Commands
```bash
pip install -e ".[dev]"                          # Install in dev mode
pytest tests/python -v --cov=kelly               # Run tests
black src/kelly tests/python                     # Format code
mypy src/kelly                                   # Type check
ruff check src/kelly tests/python                # Lint
```

### C++ Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PLUGINS=ON
cmake --build build --config Release
cd build && ctest --output-on-failure
```

---

## Key Concepts

### 216-Node Emotion Thesaurus
- **Valence**: -1.0 (negative) to 1.0 (positive)
- **Arousal**: 0.0 (calm) to 1.0 (excited)
- **Intensity**: 0.0 to 1.0

### Three-Phase Intent System
| Phase | Name | Description |
|-------|------|-------------|
| 0 | Wound | Initial trauma/trigger |
| 1 | Emotion | Mapped emotional response |
| 2 | Rule-Break | Musical rule violations |

### Rule-Breaking Categories
| Category | Effect |
|----------|--------|
| Harmony | Dissonance, clusters |
| Rhythm | Syncopation, irregular meters |
| Dynamics | Extreme contrasts |

---

*Consolidated: December 8, 2025*
