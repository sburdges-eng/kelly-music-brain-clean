"""MIDI generation and pipeline."""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import mido


@dataclass
class GrooveTemplate:
    """Represents a rhythmic groove template."""
    name: str
    time_signature: Tuple[int, int]
    pattern: List[Tuple[float, int]]  # (time, velocity)
    swing: float = 0.0


class MidiGenerator:
    """
    MIDI pipeline for generating therapeutic musical output.
    
    Handles chord progressions, groove templates, and MIDI message generation.
    """
    
    def __init__(self, tempo: int = 120) -> None:
        """
        Initialize MIDI generator.
        
        Args:
            tempo: Initial tempo in BPM
        """
        self.tempo = tempo
        self.groove_templates = self._initialize_grooves()
    
    def _initialize_grooves(self) -> Dict[str, GrooveTemplate]:
        """Initialize standard groove templates."""
        return {
            "straight": GrooveTemplate(
                name="Straight",
                time_signature=(4, 4),
                pattern=[(0.0, 100), (0.25, 80), (0.5, 100), (0.75, 80)]
            ),
            "swing": GrooveTemplate(
                name="Swing",
                time_signature=(4, 4),
                pattern=[(0.0, 100), (0.33, 80), (0.5, 100), (0.83, 80)],
                swing=0.66
            ),
            "syncopated": GrooveTemplate(
                name="Syncopated",
                time_signature=(4, 4),
                pattern=[(0.0, 100), (0.125, 60), (0.375, 90), (0.625, 85), (0.875, 70)]
            )
        }
    
    def generate_chord_progression(
        self, 
        mode: str = "minor",
        length: int = 4,
        allow_dissonance: bool = False
    ) -> List[List[int]]:
        """
        Generate a chord progression based on emotional parameters.
        
        Args:
            mode: "major" or "minor"
            length: Number of chords
            allow_dissonance: Whether to allow dissonant intervals
            
        Returns:
            List of chords (each chord is a list of MIDI note numbers)
        """
        # Simple chord progressions
        if mode == "minor":
            # i - iv - v - i in A minor
            base_progression = [
                [57, 60, 64],  # Am
                [57, 62, 65],  # Dm
                [59, 64, 67],  # Em
                [57, 60, 64],  # Am
            ]
        else:
            # I - IV - V - I in C major
            base_progression = [
                [60, 64, 67],  # C
                [65, 69, 72],  # F
                [67, 71, 74],  # G
                [60, 64, 67],  # C
            ]
        
        # Add dissonance if requested
        if allow_dissonance:
            for chord in base_progression:
                # Add a minor 9th above root
                chord.append(chord[0] + 13)
        
        return base_progression[:length]
    
    def apply_groove(
        self,
        notes: List[int],
        groove_template: GrooveTemplate,
        duration_bars: int = 1
    ) -> List[mido.Message]:
        """
        Apply groove template to notes.
        
        Args:
            notes: MIDI note numbers
            groove_template: The groove to apply
            duration_bars: Duration in bars
            
        Returns:
            List of MIDI messages
        """
        messages = []
        ticks_per_beat = 480
        beats_per_bar = groove_template.time_signature[0]
        
        for bar in range(duration_bars):
            for beat_time, velocity in groove_template.pattern:
                absolute_time = int(
                    (bar * beats_per_bar + beat_time * beats_per_bar) * ticks_per_beat
                )
                
                # Note on for each note in chord
                for note in notes:
                    messages.append(
                        mido.Message(
                            'note_on',
                            note=note,
                            velocity=velocity,
                            time=absolute_time
                        )
                    )
                
                # Note off after quarter note
                off_time = int(absolute_time + ticks_per_beat * 0.25)
                for note in notes:
                    messages.append(
                        mido.Message(
                            'note_off',
                            note=note,
                            velocity=0,
                            time=off_time
                        )
                    )
        
        return messages
    
    def create_midi_file(
        self,
        chord_progression: List[List[int]],
        groove: str = "straight",
        output_path: Optional[str] = None
    ) -> mido.MidiFile:
        """
        Create a MIDI file from chord progression.
        
        Args:
            chord_progression: List of chords
            groove: Name of groove template to use
            output_path: Optional path to save MIDI file
            
        Returns:
            MIDI file object
        """
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.tempo)))
        
        # Get groove template
        groove_template = self.groove_templates.get(groove, self.groove_templates["straight"])
        
        # Generate messages for each chord
        for chord in chord_progression:
            messages = self.apply_groove(chord, groove_template)
            track.extend(messages)
        
        if output_path:
            mid.save(output_path)
        
        return mid
