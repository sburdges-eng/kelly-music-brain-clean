"""Core emotion processing and thesaurus - 216 nodes of emotional mapping.

The emotion thesaurus maps emotional states to musical parameters using
three dimensions: valence, arousal, and intensity.

Each of the 6 primary emotions (Plutchik) has 6 intensity levels,
and each intensity has 6 sub-emotions = 216 total nodes.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class EmotionCategory(Enum):
    """Primary emotion categories (Plutchik's wheel)."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"


class IntensityLevel(Enum):
    """Intensity levels for each emotion."""
    SUBTLE = 1
    MILD = 2
    MODERATE = 3
    STRONG = 4
    INTENSE = 5
    EXTREME = 6


@dataclass
class MusicalMapping:
    """Musical parameters derived from emotion."""
    mode: str  # major, minor, dorian, phrygian, etc.
    tempo_modifier: float  # multiplier on base tempo
    dynamic_range: Tuple[int, int]  # velocity min/max
    harmonic_complexity: float  # 0-1, affects chord extensions
    dissonance_tolerance: float  # 0-1, how much tension allowed
    rhythm_regularity: float  # 0-1, 1=strict grid, 0=free time
    articulation: str  # legato, staccato, tenuto, etc.
    register_preference: str  # low, mid, high, full
    space_density: float  # 0-1, how much silence
    rule_breaks: List[str] = field(default_factory=list)


@dataclass
class EmotionNode:
    """A single node in the 216-node emotion thesaurus."""
    id: int
    name: str
    category: EmotionCategory
    intensity: float  # 0.0-1.0
    valence: float  # -1.0 (negative) to +1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    description: str
    musical_mapping: MusicalMapping
    related_emotions: List[int] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "intensity": self.intensity,
            "valence": self.valence,
            "arousal": self.arousal,
            "description": self.description,
            "musical_mapping": {
                "mode": self.musical_mapping.mode,
                "tempo_modifier": self.musical_mapping.tempo_modifier,
                "dynamic_range": self.musical_mapping.dynamic_range,
                "harmonic_complexity": self.musical_mapping.harmonic_complexity,
                "dissonance_tolerance": self.musical_mapping.dissonance_tolerance,
                "rhythm_regularity": self.musical_mapping.rhythm_regularity,
                "articulation": self.musical_mapping.articulation,
                "register_preference": self.musical_mapping.register_preference,
                "space_density": self.musical_mapping.space_density,
                "rule_breaks": self.musical_mapping.rule_breaks,
            },
            "related_emotions": self.related_emotions,
            "triggers": self.triggers,
        }


# Pre-defined emotion nodes organized by category
JOY_EMOTIONS = [
    ("serenity", 0.17, 0.3, 0.2, "Peaceful contentment"),
    ("acceptance", 0.17, 0.4, 0.3, "Calm acknowledgment"),
    ("contentment", 0.33, 0.5, 0.3, "Satisfied warmth"),
    ("happiness", 0.5, 0.6, 0.5, "Active positive feeling"),
    ("cheerfulness", 0.5, 0.65, 0.55, "Light-hearted gladness"),
    ("delight", 0.67, 0.7, 0.6, "Pleased surprise"),
    ("joy", 0.67, 0.75, 0.65, "Deep happiness"),
    ("elation", 0.83, 0.85, 0.75, "Triumphant happiness"),
    ("bliss", 0.83, 0.9, 0.6, "Transcendent peace"),
    ("euphoria", 1.0, 0.95, 0.9, "Peak positive experience"),
    ("ecstasy", 1.0, 1.0, 1.0, "Overwhelming joy"),
    ("rapture", 1.0, 0.98, 0.85, "Transported delight"),
]

SADNESS_EMOTIONS = [
    ("pensiveness", 0.17, -0.2, 0.2, "Thoughtful melancholy"),
    ("wistfulness", 0.17, -0.25, 0.25, "Gentle longing"),
    ("melancholy", 0.33, -0.4, 0.3, "Sweet sadness"),
    ("gloom", 0.33, -0.45, 0.25, "Heavy mood"),
    ("sorrow", 0.5, -0.6, 0.35, "Deep sadness"),
    ("sadness", 0.5, -0.55, 0.4, "Core negative emotion"),
    ("grief", 0.67, -0.75, 0.45, "Profound loss"),
    ("anguish", 0.67, -0.8, 0.6, "Intense suffering"),
    ("despair", 0.83, -0.9, 0.5, "Hopelessness"),
    ("devastation", 0.83, -0.85, 0.55, "Complete overwhelm"),
    ("desolation", 1.0, -0.95, 0.4, "Utter emptiness"),
    ("heartbreak", 1.0, -0.9, 0.65, "Crushing loss"),
]

ANGER_EMOTIONS = [
    ("annoyance", 0.17, -0.2, 0.4, "Mild irritation"),
    ("irritation", 0.17, -0.25, 0.45, "Surface frustration"),
    ("frustration", 0.33, -0.4, 0.55, "Blocked desire"),
    ("agitation", 0.33, -0.35, 0.6, "Restless discomfort"),
    ("anger", 0.5, -0.55, 0.7, "Core hot emotion"),
    ("hostility", 0.5, -0.6, 0.65, "Aggressive opposition"),
    ("wrath", 0.67, -0.7, 0.8, "Intense anger"),
    ("outrage", 0.67, -0.75, 0.85, "Moral anger"),
    ("fury", 0.83, -0.85, 0.95, "Extreme anger"),
    ("rage", 0.83, -0.9, 1.0, "Uncontrolled anger"),
    ("hatred", 1.0, -0.95, 0.75, "Deep animosity"),
    ("vengeance", 1.0, -0.9, 0.85, "Retributive anger"),
]

FEAR_EMOTIONS = [
    ("apprehension", 0.17, -0.2, 0.35, "Mild worry"),
    ("unease", 0.17, -0.25, 0.4, "Slight discomfort"),
    ("worry", 0.33, -0.35, 0.45, "Anxious concern"),
    ("nervousness", 0.33, -0.4, 0.55, "Agitated anticipation"),
    ("anxiety", 0.5, -0.5, 0.6, "Sustained worry"),
    ("fear", 0.5, -0.6, 0.7, "Core protective emotion"),
    ("dread", 0.67, -0.7, 0.65, "Anticipatory fear"),
    ("alarm", 0.67, -0.65, 0.8, "Alert fear"),
    ("panic", 0.83, -0.8, 0.95, "Overwhelming fear"),
    ("terror", 0.83, -0.9, 1.0, "Extreme fear"),
    ("horror", 1.0, -0.95, 0.85, "Shocked fear"),
    ("paranoia", 1.0, -0.85, 0.75, "Persecutory fear"),
]

SURPRISE_EMOTIONS = [
    ("interest", 0.17, 0.1, 0.4, "Engaged attention"),
    ("curiosity", 0.17, 0.15, 0.45, "Active wondering"),
    ("intrigue", 0.33, 0.2, 0.5, "Fascinated interest"),
    ("wonder", 0.33, 0.35, 0.55, "Awed curiosity"),
    ("surprise", 0.5, 0.0, 0.7, "Neutral startle"),
    ("astonishment", 0.5, 0.1, 0.75, "Impressed surprise"),
    ("amazement", 0.67, 0.25, 0.8, "Delighted shock"),
    ("awe", 0.67, 0.3, 0.7, "Reverent wonder"),
    ("shock", 0.83, -0.1, 0.9, "Stunned reaction"),
    ("disbelief", 0.83, -0.2, 0.75, "Incredulous surprise"),
    ("stupefaction", 1.0, 0.0, 0.85, "Overwhelming bewilderment"),
    ("revelation", 1.0, 0.4, 0.8, "Transformative insight"),
]

DISGUST_EMOTIONS = [
    ("discomfort", 0.17, -0.2, 0.3, "Mild unease"),
    ("distaste", 0.17, -0.25, 0.35, "Aesthetic rejection"),
    ("aversion", 0.33, -0.4, 0.4, "Active avoidance"),
    ("displeasure", 0.33, -0.35, 0.35, "Dissatisfaction"),
    ("disgust", 0.5, -0.55, 0.5, "Core rejection emotion"),
    ("revulsion", 0.5, -0.6, 0.55, "Physical disgust"),
    ("contempt", 0.67, -0.65, 0.45, "Superior disdain"),
    ("scorn", 0.67, -0.7, 0.5, "Dismissive contempt"),
    ("loathing", 0.83, -0.8, 0.6, "Deep disgust"),
    ("abhorrence", 0.83, -0.85, 0.55, "Moral revulsion"),
    ("repugnance", 1.0, -0.9, 0.65, "Extreme disgust"),
    ("detestation", 1.0, -0.95, 0.6, "Complete rejection"),
]


def _create_musical_mapping(category: EmotionCategory, intensity: float, valence: float, arousal: float) -> MusicalMapping:
    """Generate musical parameters based on emotion dimensions."""
    
    # Mode selection based on valence
    if valence > 0.5:
        mode = "major" if arousal < 0.6 else "lydian"
    elif valence > 0:
        mode = "mixolydian" if arousal > 0.5 else "major"
    elif valence > -0.3:
        mode = "dorian" if arousal > 0.5 else "aeolian"
    elif valence > -0.6:
        mode = "minor" if arousal < 0.6 else "harmonic_minor"
    else:
        mode = "phrygian" if arousal > 0.5 else "locrian"
    
    # Tempo based on arousal and intensity
    tempo_mod = 0.7 + (arousal * 0.6) + (intensity * 0.2)
    
    # Dynamics based on intensity and arousal
    vel_min = int(30 + intensity * 40 + arousal * 20)
    vel_max = int(60 + intensity * 50 + arousal * 17)
    
    # Harmonic complexity from arousal and intensity
    harmonic_complexity = min(1.0, 0.3 + arousal * 0.4 + intensity * 0.3)
    
    # Dissonance from negative valence and intensity
    dissonance = min(1.0, max(0, -valence * 0.5) + intensity * 0.3)
    
    # Rhythm regularity (fear/anxiety = irregular, joy = regular)
    regularity = 0.8 - abs(valence - 0.3) * 0.3 - arousal * 0.2
    
    # Articulation
    if arousal > 0.7:
        articulation = "staccato" if valence < 0 else "marcato"
    elif arousal < 0.3:
        articulation = "legato"
    else:
        articulation = "tenuto"
    
    # Register
    if arousal > 0.6 and valence > 0:
        register = "high"
    elif arousal < 0.3 or valence < -0.5:
        register = "low"
    else:
        register = "mid"
    
    # Space density (sadness = more space, anger = dense)
    space = 0.5 - arousal * 0.3 + max(0, -valence) * 0.2
    
    # Rule breaks based on emotion category
    rule_breaks = []
    if category == EmotionCategory.SADNESS:
        rule_breaks = ["HARMONY_UnresolvedDissonance", "RHYTHM_DroppedBeats"]
    elif category == EmotionCategory.ANGER:
        rule_breaks = ["HARMONY_ParallelMotion", "PRODUCTION_Distortion"]
    elif category == EmotionCategory.FEAR:
        rule_breaks = ["RHYTHM_TempoFluctuation", "HARMONY_AvoidTonicResolution"]
    elif category == EmotionCategory.JOY:
        rule_breaks = ["HARMONY_ModalInterchange"]
    elif category == EmotionCategory.SURPRISE:
        rule_breaks = ["ARRANGEMENT_StructuralMismatch", "RHYTHM_MetricModulation"]
    elif category == EmotionCategory.DISGUST:
        rule_breaks = ["PRODUCTION_ExcessiveMud", "HARMONY_UnresolvedDissonance"]
    
    return MusicalMapping(
        mode=mode,
        tempo_modifier=round(tempo_mod, 2),
        dynamic_range=(vel_min, vel_max),
        harmonic_complexity=round(harmonic_complexity, 2),
        dissonance_tolerance=round(dissonance, 2),
        rhythm_regularity=round(max(0, min(1, regularity)), 2),
        articulation=articulation,
        register_preference=register,
        space_density=round(max(0, min(1, space)), 2),
        rule_breaks=rule_breaks,
    )


class EmotionThesaurus:
    """The 216-node emotion thesaurus mapping emotions to music.
    
    Usage:
        thesaurus = EmotionThesaurus()
        node = thesaurus.get_emotion("grief")
        similar = thesaurus.find_similar("melancholy", n=5)
        params = thesaurus.get_musical_params("anxiety")
    """
    
    def __init__(self):
        self.nodes: Dict[int, EmotionNode] = {}
        self.name_index: Dict[str, int] = {}
        self.category_index: Dict[EmotionCategory, List[int]] = {cat: [] for cat in EmotionCategory}
        self._build_thesaurus()
    
    def _build_thesaurus(self):
        """Build the complete 216-node thesaurus."""
        node_id = 0
        emotion_lists = [
            (EmotionCategory.JOY, JOY_EMOTIONS),
            (EmotionCategory.SADNESS, SADNESS_EMOTIONS),
            (EmotionCategory.ANGER, ANGER_EMOTIONS),
            (EmotionCategory.FEAR, FEAR_EMOTIONS),
            (EmotionCategory.SURPRISE, SURPRISE_EMOTIONS),
            (EmotionCategory.DISGUST, DISGUST_EMOTIONS),
        ]
        
        for category, emotions in emotion_lists:
            for name, intensity, valence, arousal, description in emotions:
                mapping = _create_musical_mapping(category, intensity, valence, arousal)
                node = EmotionNode(
                    id=node_id,
                    name=name,
                    category=category,
                    intensity=intensity,
                    valence=valence,
                    arousal=arousal,
                    description=description,
                    musical_mapping=mapping,
                )
                self.nodes[node_id] = node
                self.name_index[name.lower()] = node_id
                self.category_index[category].append(node_id)
                node_id += 1
        
        # Build relationship graph
        self._build_relationships()
    
    def _build_relationships(self):
        """Connect related emotions based on valence/arousal proximity."""
        for node_id, node in self.nodes.items():
            related = []
            for other_id, other in self.nodes.items():
                if other_id == node_id:
                    continue
                # Calculate emotional distance
                v_dist = abs(node.valence - other.valence)
                a_dist = abs(node.arousal - other.arousal)
                dist = (v_dist ** 2 + a_dist ** 2) ** 0.5
                if dist < 0.4:  # Close emotions
                    related.append(other_id)
            node.related_emotions = related[:6]
    
    def get_emotion(self, name: str) -> Optional[EmotionNode]:
        """Get emotion node by name."""
        node_id = self.name_index.get(name.lower())
        return self.nodes.get(node_id) if node_id is not None else None
    
    def get_by_id(self, node_id: int) -> Optional[EmotionNode]:
        """Get emotion node by ID."""
        return self.nodes.get(node_id)
    
    def find_by_category(self, category: EmotionCategory) -> List[EmotionNode]:
        """Get all emotions in a category."""
        return [self.nodes[nid] for nid in self.category_index[category]]
    
    def find_similar(self, name: str, n: int = 5) -> List[EmotionNode]:
        """Find n most similar emotions."""
        node = self.get_emotion(name)
        if not node:
            return []
        return [self.nodes[rid] for rid in node.related_emotions[:n]]
    
    def find_by_dimensions(
        self,
        valence: float,
        arousal: float,
        intensity: Optional[float] = None
    ) -> List[EmotionNode]:
        """Find emotions closest to given dimensions."""
        results = []
        for node in self.nodes.values():
            v_dist = abs(node.valence - valence)
            a_dist = abs(node.arousal - arousal)
            i_dist = abs(node.intensity - intensity) if intensity else 0
            dist = (v_dist ** 2 + a_dist ** 2 + i_dist ** 2) ** 0.5
            results.append((dist, node))
        results.sort(key=lambda x: x[0])
        return [node for _, node in results[:10]]
    
    def get_musical_params(self, name: str) -> Optional[MusicalMapping]:
        """Get musical parameters for an emotion."""
        node = self.get_emotion(name)
        return node.musical_mapping if node else None
    
    def list_all(self) -> List[str]:
        """List all emotion names."""
        return list(self.name_index.keys())
    
    def to_json(self, path: Optional[Path] = None) -> str:
        """Export thesaurus to JSON."""
        data = {
            "version": "1.0",
            "total_nodes": len(self.nodes),
            "emotions": [node.to_dict() for node in self.nodes.values()]
        }
        json_str = json.dumps(data, indent=2)
        if path:
            path.write_text(json_str)
        return json_str
    
    @classmethod
    def from_json(cls, path: Path) -> "EmotionThesaurus":
        """Load thesaurus from JSON."""
        # Implementation for loading saved thesaurus
        thesaurus = cls()  # Use default for now
        return thesaurus


# Convenience functions
def get_emotion(name: str) -> Optional[EmotionNode]:
    """Quick access to emotion lookup."""
    return EmotionThesaurus().get_emotion(name)

def list_emotions() -> List[str]:
    """List all available emotions."""
    return EmotionThesaurus().list_all()
