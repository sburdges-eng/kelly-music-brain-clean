"""
Arrangement Generator - Generate complete song arrangements from intent.

Combines:
- Section templates
- Energy arcs
- Instrumentation planning
- Multi-track MIDI generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

from music_brain.arrangement.templates import (
    SectionTemplate,
    ArrangementTemplate,
    get_genre_template,
    SectionType,
)
from music_brain.arrangement.energy_arc import (
    EnergyArc,
    NarrativeArc,
    calculate_energy_curve,
    map_emotion_to_arc,
)
from music_brain.arrangement.bass_generator import (
    BassPattern,
    generate_bass_line,
    bass_line_to_midi,
    suggest_bass_pattern,
)

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


@dataclass
class InstrumentTrack:
    """Single instrument track in arrangement."""
    name: str
    midi_channel: int = 0
    entry_section: int = 0  # Section index where instrument enters
    exit_section: Optional[int] = None  # Section where instrument exits (None = plays to end)
    velocity_curve: List[int] = field(default_factory=list)  # Velocity per section
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "midi_channel": self.midi_channel,
            "entry_section": self.entry_section,
            "exit_section": self.exit_section,
            "velocity_curve": self.velocity_curve,
        }


@dataclass
class GeneratedArrangement:
    """Complete generated arrangement."""
    template: ArrangementTemplate
    energy_arc: EnergyArc
    instruments: List[InstrumentTrack]
    chord_progression: List[str] = field(default_factory=list)
    ppq: int = 480
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "template": self.template.to_dict(),
            "energy_arc": self.energy_arc.to_dict(),
            "instruments": [i.to_dict() for i in self.instruments],
            "chord_progression": self.chord_progression,
            "ppq": self.ppq,
        }
    
    def get_production_notes(self) -> List[str]:
        """Generate production/mixing notes."""
        notes = []
        
        # Overall structure notes
        notes.append(f"Genre: {self.template.genre}")
        notes.append(f"Tempo: {self.template.tempo_bpm} BPM")
        notes.append(f"Total sections: {len(self.template.sections)}")
        notes.append(f"Total bars: {self.template.total_bars}")
        
        # Energy arc notes
        arc_name = self.energy_arc.narrative_arc.value.replace('-', ' ').title()
        notes.append(f"Narrative arc: {arc_name}")
        notes.append(
            f"Peak energy at {self.energy_arc.peak_position*100:.0f}% through song"
        )
        
        # Section breakdown
        notes.append("\n## Section Breakdown:")
        cumulative_bars = 0
        for i, section in enumerate(self.template.sections):
            energy = self.energy_arc.get_energy_at_position(
                cumulative_bars / self.template.total_bars
            )
            notes.append(
                f"- {section.section_type.value.title()}: "
                f"Bars {cumulative_bars}-{cumulative_bars + section.length_bars}, "
                f"Energy {energy:.1%}"
            )
            cumulative_bars += section.length_bars
        
        # Instrumentation notes
        notes.append("\n## Instrumentation:")
        for inst in self.instruments:
            entry_name = self.template.sections[inst.entry_section].section_type.value
            exit_info = (
                f"exits at {self.template.sections[inst.exit_section].section_type.value}"
                if inst.exit_section is not None
                else "plays to end"
            )
            notes.append(f"- {inst.name}: enters at {entry_name}, {exit_info}")
        
        return notes


class ArrangementGenerator:
    """Generator for complete song arrangements."""
    
    def __init__(self, ppq: int = 480):
        """
        Initialize generator.
        
        Args:
            ppq: Pulses per quarter note for MIDI generation
        """
        self.ppq = ppq
    
    def generate(
        self,
        genre: str = "pop",
        primary_emotion: str = "neutral",
        chord_progression: Optional[List[str]] = None,
        narrative_arc: Optional[NarrativeArc] = None,
        base_intensity: float = 0.6,
    ) -> GeneratedArrangement:
        """
        Generate complete arrangement from parameters.
        
        Args:
            genre: Musical genre
            primary_emotion: Primary emotion for energy mapping
            chord_progression: Optional chord progression (None = use default)
            narrative_arc: Optional narrative arc (None = map from emotion)
            base_intensity: Base energy level (0.0-1.0)
        
        Returns:
            GeneratedArrangement with all components
        """
        # Get genre template
        template = get_genre_template(genre)
        
        # Determine narrative arc
        if narrative_arc is None:
            narrative_arc = map_emotion_to_arc(primary_emotion)
        
        # Calculate energy arc
        num_sections = len(template.sections)
        energy_arc = calculate_energy_curve(
            narrative_arc,
            num_sections,
            base_intensity=base_intensity,
        )
        
        # Apply energy levels to sections
        for i, section in enumerate(template.sections):
            section.energy_level = energy_arc.get_energy_at_position(i / num_sections)
        
        # Plan instrumentation
        instruments = self._plan_instrumentation(template, energy_arc)
        
        # Use provided or default chord progression
        if chord_progression is None:
            # Simple default progression
            chord_progression = self._generate_default_chords(template)
        
        return GeneratedArrangement(
            template=template,
            energy_arc=energy_arc,
            instruments=instruments,
            chord_progression=chord_progression,
            ppq=self.ppq,
        )
    
    def _plan_instrumentation(
        self,
        template: ArrangementTemplate,
        energy_arc: EnergyArc,
    ) -> List[InstrumentTrack]:
        """Plan instrument entry/exit points based on energy."""
        instruments = []
        
        # Core instruments (always present or nearly always)
        instruments.append(InstrumentTrack(
            name="drums",
            midi_channel=9,  # Standard drum channel
            entry_section=0 if template.sections[0].section_type != SectionType.INTRO else 1,
        ))
        
        instruments.append(InstrumentTrack(
            name="bass",
            midi_channel=1,
            entry_section=0 if template.sections[0].section_type != SectionType.INTRO else 1,
        ))
        
        # Harmonic instruments - enter based on energy
        # Find first medium-energy section
        first_chorus_idx = next(
            (i for i, s in enumerate(template.sections) 
             if s.section_type == SectionType.CHORUS),
            1  # Default to second section
        )
        
        instruments.append(InstrumentTrack(
            name="guitar",
            midi_channel=2,
            entry_section=1,  # Usually enters after intro
        ))
        
        instruments.append(InstrumentTrack(
            name="keys",
            midi_channel=3,
            entry_section=0,  # Often in from start
        ))
        
        # Lead instruments for high-energy sections
        instruments.append(InstrumentTrack(
            name="lead_synth",
            midi_channel=4,
            entry_section=first_chorus_idx,
        ))
        
        # Calculate velocity curves based on energy arc
        for inst in instruments:
            velocity_curve = []
            for i, _ in enumerate(template.sections):
                if i < inst.entry_section or (
                    inst.exit_section is not None and i >= inst.exit_section
                ):
                    velocity_curve.append(0)
                else:
                    energy = energy_arc.get_energy_at_position(
                        i / len(template.sections)
                    )
                    # Map energy to velocity (40-110 range)
                    velocity = int(40 + energy * 70)
                    velocity_curve.append(velocity)
            inst.velocity_curve = velocity_curve
        
        return instruments
    
    def _generate_default_chords(self, template: ArrangementTemplate) -> List[str]:
        """Generate simple default chord progression."""
        # Simple progressions by genre
        genre_progressions = {
            "pop": ["C", "G", "Am", "F"],
            "rock": ["C", "F", "G", "C"],
            "edm": ["Am", "F", "C", "G"],
            "lofi": ["Dm7", "G7", "Cmaj7", "Am7"],
            "indie": ["C", "Em", "F", "G"],
        }
        
        base_prog = genre_progressions.get(
            template.genre.lower(),
            ["C", "G", "Am", "F"]
        )
        
        # Repeat progression to match number of sections
        num_sections = len(template.sections)
        chords = (base_prog * ((num_sections // len(base_prog)) + 1))[:num_sections]
        
        return chords
    
    def export_arrangement_markers(
        self,
        arrangement: GeneratedArrangement,
        output_path: str,
    ) -> None:
        """
        Export arrangement with DAW-compatible markers.
        
        Creates MIDI file with marker events for each section.
        """
        if not MIDO_AVAILABLE:
            raise ImportError("mido required. Install with: pip install mido")
        
        mid = mido.MidiFile(ticks_per_beat=self.ppq)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Add tempo
        tempo_microseconds = int(60_000_000 / arrangement.template.tempo_bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))
        
        # Add time signature
        num, denom = arrangement.template.time_signature
        track.append(mido.MetaMessage(
            'time_signature',
            numerator=num,
            denominator=denom,
            time=0,
        ))
        
        # Add section markers
        ticks_per_bar = self.ppq * num
        current_tick = 0
        current_bar = 0
        
        for i, section in enumerate(arrangement.template.sections):
            # Add marker at section start
            marker_name = f"{section.section_type.value.title()} {i+1}"
            delta = current_bar * ticks_per_bar - current_tick
            track.append(mido.MetaMessage('marker', text=marker_name, time=delta))
            current_tick = current_bar * ticks_per_bar
            
            current_bar += section.length_bars
        
        # Add end marker
        delta = current_bar * ticks_per_bar - current_tick
        track.append(mido.MetaMessage('marker', text='End', time=delta))
        
        mid.save(output_path)
    
    def export_bass_track(
        self,
        arrangement: GeneratedArrangement,
        output_path: str,
        pattern: Optional[BassPattern] = None,
    ) -> None:
        """
        Export bass track as separate MIDI file.
        
        Args:
            arrangement: Arrangement to export from
            output_path: Output MIDI file path
            pattern: Bass pattern (None = auto-select based on genre)
        """
        if pattern is None:
            # Suggest pattern based on genre and avg energy
            avg_energy = sum(arrangement.energy_arc.energy_curve) / len(
                arrangement.energy_arc.energy_curve
            )
            pattern = suggest_bass_pattern(arrangement.template.genre, avg_energy)
        
        # Generate bass line
        bass_line = generate_bass_line(
            chord_progression=arrangement.chord_progression,
            pattern=pattern,
            ppq=self.ppq,
            time_signature=arrangement.template.time_signature,
        )
        
        # Export to MIDI
        bass_line_to_midi(
            bass_line,
            output_path,
            ppq=self.ppq,
            tempo_bpm=arrangement.template.tempo_bpm,
        )


# =================================================================
# CONVENIENCE FUNCTION
# =================================================================

def generate_arrangement(
    genre: str = "pop",
    emotion: str = "neutral",
    chords: Optional[List[str]] = None,
    intensity: float = 0.6,
    ppq: int = 480,
) -> GeneratedArrangement:
    """
    Convenience function to generate arrangement.
    
    Args:
        genre: Musical genre
        emotion: Primary emotion
        chords: Optional chord progression
        intensity: Base intensity level
        ppq: Pulses per quarter note
    
    Returns:
        GeneratedArrangement
    """
    generator = ArrangementGenerator(ppq=ppq)
    return generator.generate(
        genre=genre,
        primary_emotion=emotion,
        chord_progression=chords,
        base_intensity=intensity,
    )
