"""Arrangement Engine - Generates full song arrangements."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class SectionType(Enum):
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    BREAKDOWN = "breakdown"
    BUILD = "build"
    DROP = "drop"
    OUTRO = "outro"


class StructureTemplate(Enum):
    POP = "pop"  # Intro-Verse-Chorus-Verse-Chorus-Bridge-Chorus-Outro
    VERSE_CHORUS = "verse_chorus"
    AABA = "aaba"
    THROUGH_COMPOSED = "through_composed"
    EDM = "edm"


class EnergyLevel(Enum):
    LOWEST = 1
    VERY_LOW = 2
    LOW = 3
    MEDIUM_LOW = 4
    MEDIUM = 5
    MEDIUM_HIGH = 6
    HIGH = 7
    HIGHEST = 8


@dataclass
class SectionProfile:
    section_type: SectionType
    bars: int
    energy: EnergyLevel
    density: float = 0.5
    emotion_intensity: float = 0.5
    drums_active: bool = True
    bass_active: bool = True
    chords_active: bool = True
    melody_active: bool = True
    pads_active: bool = False
    strings_active: bool = False


@dataclass
class ArrangementPlan:
    sections: List[SectionProfile]
    total_bars: int
    structure_template: StructureTemplate
    base_tempo: int = 120
    key: str = "C"
    mode: str = "minor"


STRUCTURE_TEMPLATES = {
    StructureTemplate.POP: [
        (SectionType.INTRO, 4),
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.BRIDGE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.OUTRO, 4),
    ],
    StructureTemplate.VERSE_CHORUS: [
        (SectionType.INTRO, 4),
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.OUTRO, 4),
    ],
    StructureTemplate.EDM: [
        (SectionType.INTRO, 8),
        (SectionType.BUILD, 8),
        (SectionType.DROP, 16),
        (SectionType.BREAKDOWN, 8),
        (SectionType.BUILD, 8),
        (SectionType.DROP, 16),
        (SectionType.OUTRO, 8),
    ],
}

EMOTION_ENERGY = {
    "grief": EnergyLevel.LOW,
    "sadness": EnergyLevel.MEDIUM_LOW,
    "anger": EnergyLevel.HIGH,
    "anxiety": EnergyLevel.MEDIUM_HIGH,
    "joy": EnergyLevel.HIGH,
    "hope": EnergyLevel.MEDIUM,
    "serenity": EnergyLevel.LOW,
    "emptiness": EnergyLevel.LOWEST,
}


class ArrangementEngine:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
    
    def generate_arrangement(
        self,
        emotion: str,
        duration_minutes: float = 3.5,
        tempo_bpm: int = 120,
        key: str = "C",
        mode: str = "minor",
        structure: Optional[StructureTemplate] = None,
    ) -> ArrangementPlan:
        structure = structure or StructureTemplate.POP
        template = STRUCTURE_TEMPLATES.get(structure, STRUCTURE_TEMPLATES[StructureTemplate.POP])
        
        base_energy = EMOTION_ENERGY.get(emotion.lower(), EnergyLevel.MEDIUM)
        
        sections = []
        total_bars = 0
        
        for section_type, bars in template:
            energy = self._calculate_section_energy(section_type, base_energy)
            
            section = SectionProfile(
                section_type=section_type,
                bars=bars,
                energy=energy,
                density=energy.value / 8,
                emotion_intensity=base_energy.value / 8,
                drums_active=section_type not in [SectionType.INTRO, SectionType.BREAKDOWN],
                bass_active=section_type not in [SectionType.INTRO],
                chords_active=True,
                melody_active=section_type in [SectionType.VERSE, SectionType.CHORUS],
                pads_active=section_type in [SectionType.INTRO, SectionType.BRIDGE, SectionType.OUTRO],
            )
            sections.append(section)
            total_bars += bars
        
        return ArrangementPlan(
            sections=sections,
            total_bars=total_bars,
            structure_template=structure,
            base_tempo=tempo_bpm,
            key=key,
            mode=mode,
        )
    
    def _calculate_section_energy(self, section_type: SectionType, base: EnergyLevel) -> EnergyLevel:
        offsets = {
            SectionType.INTRO: -2,
            SectionType.VERSE: 0,
            SectionType.PRE_CHORUS: 1,
            SectionType.CHORUS: 2,
            SectionType.BRIDGE: -1,
            SectionType.BREAKDOWN: -3,
            SectionType.BUILD: 1,
            SectionType.DROP: 3,
            SectionType.OUTRO: -2,
        }
        new_value = base.value + offsets.get(section_type, 0)
        new_value = max(1, min(8, new_value))
        return EnergyLevel(new_value)
