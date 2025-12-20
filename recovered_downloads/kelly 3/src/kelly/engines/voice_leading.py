"""Voice Leading Engine - Handles smooth voice transitions."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import random


@dataclass
class Voice:
    pitch: int
    voice_number: int
    active: bool = True


@dataclass
class VoicingResult:
    pitches: List[int]
    voice_movements: List[Tuple[int, int]]  # (from, to) for each voice


class VoiceLeadingEngine:
    """Handles smooth voice leading between chords."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed: random.seed(seed)
        self.num_voices = 4
    
    def voice_chord(
        self,
        chord_pitches: List[int],
        previous_voicing: Optional[List[int]] = None,
        target_register: Tuple[int, int] = (48, 84),
    ) -> VoicingResult:
        """Voice a chord with smooth voice leading from previous."""
        if not previous_voicing:
            # Initial voicing - spread in register
            voiced = self._initial_voicing(chord_pitches, target_register)
            return VoicingResult(voiced, [(p, p) for p in voiced])
        
        movements = []
        voiced = []
        available = chord_pitches.copy()
        
        # For each previous voice, find closest available pitch
        for prev_pitch in previous_voicing:
            if not available:
                break
            
            # Find closest pitch in chord
            closest = min(available, key=lambda p: self._voice_distance(prev_pitch, p))
            voiced.append(closest)
            movements.append((prev_pitch, closest))
            available.remove(closest)
        
        # Add any remaining chord tones
        for pitch in available:
            voiced.append(pitch)
            movements.append((pitch, pitch))
        
        # Ensure in register
        voiced = [self._constrain_to_register(p, target_register) for p in voiced]
        
        return VoicingResult(sorted(voiced), movements)
    
    def _voice_distance(self, from_pitch: int, to_pitch: int) -> int:
        """Calculate voice leading distance (prefer small intervals)."""
        raw_dist = abs(to_pitch - from_pitch)
        # Penalize large jumps
        if raw_dist > 7:  # More than a fifth
            return raw_dist * 2
        return raw_dist
    
    def _initial_voicing(self, pitches: List[int], register: Tuple[int, int]) -> List[int]:
        """Create initial voicing spread across register."""
        result = []
        center = (register[0] + register[1]) // 2
        
        for i, pitch in enumerate(pitches):
            # Spread voices
            target = center + (i - len(pitches) // 2) * 5
            voiced = pitch
            while voiced < target - 12:
                voiced += 12
            while voiced > target + 12:
                voiced -= 12
            result.append(self._constrain_to_register(voiced, register))
        
        return sorted(result)
    
    def _constrain_to_register(self, pitch: int, register: Tuple[int, int]) -> int:
        """Ensure pitch is within register."""
        while pitch < register[0]:
            pitch += 12
        while pitch > register[1]:
            pitch -= 12
        return pitch
    
    def analyze_voice_leading(
        self,
        from_voicing: List[int],
        to_voicing: List[int]
    ) -> Dict:
        """Analyze the quality of voice leading between voicings."""
        if len(from_voicing) != len(to_voicing):
            return {"error": "Voicings must have same number of voices"}
        
        total_distance = sum(
            abs(f - t) for f, t in zip(sorted(from_voicing), sorted(to_voicing))
        )
        
        avg_distance = total_distance / len(from_voicing) if from_voicing else 0
        
        parallel_fifths = self._check_parallel_fifths(from_voicing, to_voicing)
        parallel_octaves = self._check_parallel_octaves(from_voicing, to_voicing)
        
        return {
            "total_distance": total_distance,
            "average_distance": avg_distance,
            "parallel_fifths": parallel_fifths,
            "parallel_octaves": parallel_octaves,
            "quality": "good" if avg_distance < 4 and not parallel_fifths else "acceptable",
        }
    
    def _check_parallel_fifths(self, from_v: List[int], to_v: List[int]) -> bool:
        """Check for parallel fifths."""
        from_sorted = sorted(from_v)
        to_sorted = sorted(to_v)
        
        for i in range(len(from_sorted) - 1):
            for j in range(i + 1, len(from_sorted)):
                from_interval = abs(from_sorted[j] - from_sorted[i]) % 12
                to_interval = abs(to_sorted[j] - to_sorted[i]) % 12
                if from_interval == 7 and to_interval == 7:
                    return True
        return False
    
    def _check_parallel_octaves(self, from_v: List[int], to_v: List[int]) -> bool:
        """Check for parallel octaves."""
        from_sorted = sorted(from_v)
        to_sorted = sorted(to_v)
        
        for i in range(len(from_sorted) - 1):
            for j in range(i + 1, len(from_sorted)):
                from_interval = abs(from_sorted[j] - from_sorted[i]) % 12
                to_interval = abs(to_sorted[j] - to_sorted[i]) % 12
                if from_interval == 0 and to_interval == 0:
                    return True
        return False
