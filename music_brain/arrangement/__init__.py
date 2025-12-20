"""
Arrangement Generator - Create song arrangements from emotional intent.

Generates complete song structures with:
- Section templates (verse, chorus, bridge, etc.)
- Energy arcs and progression
- Instrumentation planning
- Genre-specific structures
"""

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
from music_brain.arrangement.templates import (
    ArrangementTemplate,
    SectionTemplate,
    get_genre_template,
)

__all__ = [
    # Generator
    "ArrangementGenerator",
    "ArrangementTemplate",
    # Energy arcs
    "EnergyArc",
    "GeneratedArrangement",
    "NarrativeArc",
    # Templates
    "SectionTemplate",
    "calculate_energy_curve",
    "generate_arrangement",
    "get_genre_template",
]
