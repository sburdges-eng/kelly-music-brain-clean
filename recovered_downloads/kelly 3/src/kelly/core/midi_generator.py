"""MIDI Generator - Creates MIDI files from emotional parameters.

Handles the final stage of the pipeline: converting processed emotional
intent into playable MIDI sequences.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import random
import math

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False


# Constants
TICKS_PER_BEAT = 480
CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Scale intervals
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
}

# Common chord progressions
PROGRESSIONS = {
    "major": [
        ["I", "V", "vi", "IV"],
        ["I", "IV", "V", "I"],
        ["I", "vi", "IV", "V"],
        ["I", "IV", "vi", "V"],
    ],
    "minor": [
        ["i", "iv", "VII", "III"],
        ["i", "VI", "III", "VII"],
        ["i", "iv", "v", "i"],
        ["i", "VII", "VI", "VII"],
    ],
}

# Roman numeral to scale degree
ROMAN_TO_DEGREE = {
    "I": 0, "i": 0,
    "II": 2, "ii": 2,
    "III": 4, "iii": 4,
    "IV": 5, "iv": 5,
    "V": 7, "v": 7,
    "VI": 9, "vi": 9,
    "VII": 11, "vii": 11,
}


@dataclass
class NoteEvent:
    """A single MIDI note event."""
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int
    channel: int = 0
    
    def to_messages(self) -> List[Tuple[int, Any]]:
        """Convert to MIDI messages with absolute tick times."""
        if not HAS_MIDO:
            return []
        return [
            (self.start_tick, Message('note_on', note=self.pitch, velocity=self.velocity, channel=self.channel)),
            (self.start_tick + self.duration_ticks, Message('note_off', note=self.pitch, velocity=0, channel=self.channel)),
        ]


@dataclass
class ChordVoicing:
    """A chord voicing with timing."""
    pitches: List[int]
    start_tick: int
    duration_ticks: int
    velocity: int
    name: str = ""


def note_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    note = note.upper().replace('♯', '#').replace('♭', 'b')
    # Handle flats
    if 'B' in note and note != 'B':
        flat_map = {'DB': 'C#', 'EB': 'D#', 'FB': 'E', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#'}
        note = flat_map.get(note, note)
    
    base = note.replace('#', '').replace('B', '')
    idx = CHROMATIC.index(base if base in CHROMATIC else base[0])
    if '#' in note:
        idx += 1
    return 12 + (octave * 12) + idx


def get_scale_pitches(root: str, mode: str, octave: int = 4) -> List[int]:
    """Get MIDI pitches for a scale."""
    root_midi = note_to_midi(root, octave)
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["minor"])
    return [root_midi + i for i in intervals]


def build_chord(root_midi: int, quality: str) -> List[int]:
    """Build chord pitches from root and quality."""
    if quality in ["M", "maj", "major"]:
        return [root_midi, root_midi + 4, root_midi + 7]
    elif quality in ["m", "min", "minor"]:
        return [root_midi, root_midi + 3, root_midi + 7]
    elif quality in ["dim"]:
        return [root_midi, root_midi + 3, root_midi + 6]
    elif quality in ["aug"]:
        return [root_midi, root_midi + 4, root_midi + 8]
    elif quality in ["7", "dom7"]:
        return [root_midi, root_midi + 4, root_midi + 7, root_midi + 10]
    elif quality in ["maj7"]:
        return [root_midi, root_midi + 4, root_midi + 7, root_midi + 11]
    elif quality in ["m7", "min7"]:
        return [root_midi, root_midi + 3, root_midi + 7, root_midi + 10]
    else:
        return [root_midi, root_midi + 4, root_midi + 7]


class MidiGenerator:
    """Generates MIDI files from emotional parameters.
    
    Usage:
        generator = MidiGenerator(tempo=82)
        progression = generator.generate_chord_progression(mode="minor")
        generator.create_midi_file(progression, "straight", "output.mid")
    """
    
    def __init__(
        self,
        tempo: int = 120,
        key: str = "C",
        mode: str = "minor",
        time_signature: Tuple[int, int] = (4, 4)
    ):
        self.tempo = tempo
        self.key = key
        self.mode = mode
        self.time_signature = time_signature
        self.ticks_per_beat = TICKS_PER_BEAT
    
    def generate_chord_progression(
        self,
        mode: Optional[str] = None,
        bars: int = 4,
        allow_dissonance: bool = False
    ) -> List[ChordVoicing]:
        """Generate a chord progression based on mode."""
        mode = mode or self.mode
        mode_type = "major" if mode in ["major", "lydian", "mixolydian"] else "minor"
        
        # Select progression template
        templates = PROGRESSIONS[mode_type]
        template = random.choice(templates)
        
        # Build voicings
        voicings = []
        root_midi = note_to_midi(self.key, 3)  # Bass octave
        ticks_per_bar = self.ticks_per_beat * self.time_signature[0]
        
        for i, roman in enumerate(template[:bars]):
            # Get scale degree
            degree = ROMAN_TO_DEGREE.get(roman.upper().replace('♭', '').replace('B', ''), 0)
            chord_root = root_midi + degree
            
            # Determine quality from Roman case
            quality = "minor" if roman.islower() else "major"
            
            # Build chord
            pitches = build_chord(chord_root, quality)
            
            # Add extensions for dissonance
            if allow_dissonance and random.random() > 0.6:
                pitches.append(chord_root + 10)  # 7th
            
            voicing = ChordVoicing(
                pitches=pitches,
                start_tick=i * ticks_per_bar,
                duration_ticks=ticks_per_bar - 10,
                velocity=70,
                name=f"{self.key}{roman}",
            )
            voicings.append(voicing)
        
        return voicings
    
    def apply_groove(
        self,
        notes: List[NoteEvent],
        groove_type: str = "straight"
    ) -> List[NoteEvent]:
        """Apply groove/humanization to notes."""
        grooved = []
        
        for note in notes:
            new_note = NoteEvent(
                pitch=note.pitch,
                start_tick=note.start_tick,
                duration_ticks=note.duration_ticks,
                velocity=note.velocity,
                channel=note.channel,
            )
            
            if groove_type == "swing":
                # Delay off-beat 8ths
                beat_pos = note.start_tick % self.ticks_per_beat
                if beat_pos == self.ticks_per_beat // 2:
                    new_note.start_tick += int(self.ticks_per_beat * 0.1)
            
            elif groove_type == "behind":
                # Slight delay on everything
                new_note.start_tick += random.randint(5, 25)
            
            elif groove_type == "syncopated":
                # Random small timing shifts
                new_note.start_tick += random.randint(-15, 15)
            
            # Velocity humanization
            new_note.velocity = max(1, min(127, note.velocity + random.randint(-8, 8)))
            
            grooved.append(new_note)
        
        return grooved
    
    def chords_to_notes(
        self,
        voicings: List[ChordVoicing],
        channel: int = 0
    ) -> List[NoteEvent]:
        """Convert chord voicings to note events."""
        notes = []
        for voicing in voicings:
            for pitch in voicing.pitches:
                notes.append(NoteEvent(
                    pitch=pitch,
                    start_tick=voicing.start_tick,
                    duration_ticks=voicing.duration_ticks,
                    velocity=voicing.velocity,
                    channel=channel,
                ))
        return notes
    
    def create_midi_file(
        self,
        progression: List[ChordVoicing],
        groove: str = "straight",
        output_path: str = "output.mid"
    ) -> Optional[str]:
        """Create MIDI file from chord progression."""
        if not HAS_MIDO:
            print("Warning: mido not installed. Cannot create MIDI file.")
            return None
        
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        
        # Tempo track
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.tempo)))
        tempo_track.append(MetaMessage('time_signature', 
                                       numerator=self.time_signature[0],
                                       denominator=self.time_signature[1]))
        tempo_track.append(MetaMessage('track_name', name='Kelly Output'))
        
        # Chord track
        chord_track = MidiTrack()
        mid.tracks.append(chord_track)
        chord_track.append(MetaMessage('track_name', name='Chords'))
        
        # Convert to notes and apply groove
        notes = self.chords_to_notes(progression)
        notes = self.apply_groove(notes, groove)
        
        # Sort by start time
        events = []
        for note in notes:
            events.extend(note.to_messages())
        events.sort(key=lambda x: x[0])
        
        # Convert to delta time
        last_tick = 0
        for tick, msg in events:
            delta = tick - last_tick
            msg.time = delta
            chord_track.append(msg)
            last_tick = tick
        
        # End of track
        chord_track.append(MetaMessage('end_of_track'))
        
        # Save
        mid.save(output_path)
        return output_path
    
    def generate_melody(
        self,
        bars: int = 4,
        density: float = 0.5,
        register: Tuple[int, int] = (60, 84)
    ) -> List[NoteEvent]:
        """Generate a simple melody."""
        scale_pitches = get_scale_pitches(self.key, self.mode, 5)
        notes = []
        
        ticks_per_bar = self.ticks_per_beat * self.time_signature[0]
        total_ticks = bars * ticks_per_bar
        
        # Notes per bar based on density
        notes_per_bar = int(2 + density * 8)
        
        current_tick = 0
        last_pitch = random.choice(scale_pitches)
        
        while current_tick < total_ticks:
            # Skip for rest
            if random.random() > density:
                current_tick += self.ticks_per_beat
                continue
            
            # Choose pitch (stepwise motion preferred)
            idx = scale_pitches.index(last_pitch) if last_pitch in scale_pitches else 0
            step = random.choice([-2, -1, 0, 1, 1, 2])
            new_idx = max(0, min(len(scale_pitches) - 1, idx + step))
            pitch = scale_pitches[new_idx]
            
            # Octave adjustment
            while pitch < register[0]:
                pitch += 12
            while pitch > register[1]:
                pitch -= 12
            
            duration = random.choice([
                self.ticks_per_beat // 2,
                self.ticks_per_beat,
                self.ticks_per_beat * 2,
            ])
            
            notes.append(NoteEvent(
                pitch=pitch,
                start_tick=current_tick,
                duration_ticks=duration,
                velocity=random.randint(60, 90),
                channel=1,
            ))
            
            last_pitch = pitch
            current_tick += duration
        
        return notes
    
    def generate_bass(
        self,
        progression: List[ChordVoicing],
        pattern: str = "root"
    ) -> List[NoteEvent]:
        """Generate bass line from chord progression."""
        notes = []
        
        for voicing in progression:
            root = min(voicing.pitches)  # Lowest note is root
            bass_pitch = root - 12  # One octave down
            
            if pattern == "root":
                notes.append(NoteEvent(
                    pitch=bass_pitch,
                    start_tick=voicing.start_tick,
                    duration_ticks=voicing.duration_ticks,
                    velocity=80,
                    channel=2,
                ))
            
            elif pattern == "root_fifth":
                half_dur = voicing.duration_ticks // 2
                notes.append(NoteEvent(
                    pitch=bass_pitch,
                    start_tick=voicing.start_tick,
                    duration_ticks=half_dur,
                    velocity=80,
                    channel=2,
                ))
                notes.append(NoteEvent(
                    pitch=bass_pitch + 7,  # Fifth
                    start_tick=voicing.start_tick + half_dur,
                    duration_ticks=half_dur,
                    velocity=70,
                    channel=2,
                ))
            
            elif pattern == "walking":
                eighth = self.ticks_per_beat // 2
                num_notes = voicing.duration_ticks // eighth
                for i in range(num_notes):
                    # Chromatic approach
                    offset = [0, 2, 4, 7, 9, 11, 12][i % 7]
                    notes.append(NoteEvent(
                        pitch=bass_pitch + offset,
                        start_tick=voicing.start_tick + i * eighth,
                        duration_ticks=eighth - 10,
                        velocity=70 + (i % 4 == 0) * 15,
                        channel=2,
                    ))
        
        return notes
    
    def create_full_arrangement(
        self,
        bars: int = 4,
        include_melody: bool = True,
        include_bass: bool = True,
        groove: str = "straight",
        output_path: str = "output.mid"
    ) -> Optional[str]:
        """Create a full MIDI arrangement."""
        if not HAS_MIDO:
            return None
        
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        
        # Tempo track
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.tempo)))
        tempo_track.append(MetaMessage('time_signature',
                                       numerator=self.time_signature[0],
                                       denominator=self.time_signature[1]))
        
        # Generate content
        progression = self.generate_chord_progression(bars=bars)
        
        def add_track(name: str, notes: List[NoteEvent]):
            track = MidiTrack()
            mid.tracks.append(track)
            track.append(MetaMessage('track_name', name=name))
            
            notes = self.apply_groove(notes, groove)
            events = []
            for note in notes:
                events.extend(note.to_messages())
            events.sort(key=lambda x: x[0])
            
            last_tick = 0
            for tick, msg in events:
                msg.time = tick - last_tick
                track.append(msg)
                last_tick = tick
            
            track.append(MetaMessage('end_of_track'))
        
        # Chords
        add_track("Chords", self.chords_to_notes(progression))
        
        # Melody
        if include_melody:
            melody = self.generate_melody(bars=bars)
            add_track("Melody", melody)
        
        # Bass
        if include_bass:
            bass = self.generate_bass(progression, pattern="root_fifth")
            add_track("Bass", bass)
        
        mid.save(output_path)
        return output_path


def quick_generate(
    emotion: str,
    output_path: str = "kelly_output.mid",
    bars: int = 4
) -> Optional[str]:
    """Quick generation from emotion name."""
    from kelly.core.emotion_thesaurus import EmotionThesaurus
    
    thesaurus = EmotionThesaurus()
    node = thesaurus.get_emotion(emotion)
    
    if not node:
        print(f"Unknown emotion: {emotion}")
        return None
    
    mapping = node.musical_mapping
    
    # Calculate tempo from modifier
    tempo = int(100 * mapping.tempo_modifier)
    
    generator = MidiGenerator(
        tempo=tempo,
        key="C",
        mode=mapping.mode,
    )
    
    return generator.create_full_arrangement(
        bars=bars,
        groove="swing" if mapping.articulation == "legato" else "straight",
        output_path=output_path,
    )
