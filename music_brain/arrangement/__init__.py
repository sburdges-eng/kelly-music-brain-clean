"""
Arrangement Generator - Create song arrangements from emotional intent.

Generates complete song structures with:
- Section templates (verse, chorus, bridge, etc.)
- Energy arcs and progression
- Instrumentation planning
- Genre-specific structures
"""

from music_brain.arrangement.templates import (
    SectionTemplate,
    ArrangementTemplate,
    get_genre_template,
)
from music_brain.arrangement.energy_arc import (
    EnergyArc,
    NarrativeArc,
    calculate_energy_curve,
)
from music_brain.arrangement.generator import (
    ArrangementGenerator,
    GeneratedArrangement,
    generate_arrangement,
)

__all__ = [
    # Templates
    "SectionTemplate",
    "ArrangementTemplate",
    "get_genre_template",
    # Energy arcs
    "EnergyArc",
    "NarrativeArc",
    "calculate_energy_curve",
    # Generator
    "ArrangementGenerator",
    "GeneratedArrangement",
    "generate_arrangement",
]
